"""
Behavior Cloning Trainer.

Trains a policy to imitate expert demonstrations from BC dataset.
Features:
- Early stopping with patience
- EMA (Exponential Moving Average) model
- Modality dropout for robustness
- Gradient accumulation
- Data diversity validation
"""

from __future__ import annotations
import os
import sys
import time
from typing import Dict, Any, Optional
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.core.trainer_base import BaseTrainer
from src.core.registry import register, get
from tqdm import tqdm
from data.norms import SPATIAL_MAX, V_MAX

# Add project root to path for data module imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../"))
from data.gpu_augmentations import GPUBatchAugmentation


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.0001,
        mode: str = "min",
    ):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: "min" for loss, "max" for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, score: float, epoch: int) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current metric value
            epoch: Current epoch number
            
        Returns:
            True if training should stop
        """
        if self.mode == "min":
            improved = self.best_score is None or score < self.best_score - self.min_delta
        else:
            improved = self.best_score is None or score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
        return self.early_stop
    
    def status(self) -> str:
        """Return status string for logging."""
        return f"patience={self.counter}/{self.patience}, best={self.best_score:.6f}@epoch{self.best_epoch}"


@register("trainer", "bc")
class BCTrainer(BaseTrainer):
    """
    Behavior Cloning Trainer.
    
    Trains a policy model to predict future waypoints and speeds
    from expert demonstrations.
    """
    
    required_components = ["model"]
    
    def __init__(self, cfg, **components):
        super().__init__(cfg, **components)
        
        self.model: nn.Module = self.components["model"].to(self.device)
        
        # Channels-last for throughput on CUDA
        if self.device.type == "cuda" and bool(getattr(getattr(cfg, "speed", {}), "channels_last", True)):
            self.model.to(memory_format=torch.channels_last)
        
        # Get trainer config
        try:
            from omegaconf import OmegaConf
            
            def _collect(c):
                nodes = []
                base = getattr(c, "trainer", None)
                if base is not None and getattr(base, "name", None) is not None:
                    nodes.append(base)
                inner = getattr(getattr(c, "trainer", None), "trainer", None)
                if inner is not None and getattr(inner, "name", None) is not None:
                    nodes.append(inner)
                exp = getattr(c, "exp", None)
                if exp is not None:
                    node = getattr(exp, "trainer", None)
                    if node is not None and getattr(node, "name", None) is not None:
                        nodes.append(node)
                    node2 = getattr(node, "trainer", None) if node is not None else None
                    if node2 is not None and getattr(node2, "name", None) is not None:
                        nodes.append(node2)
                    exp2 = getattr(exp, "exp", None)
                    if exp2 is not None:
                        node3 = getattr(exp2, "trainer", None)
                        if node3 is not None and getattr(node3, "name", None) is not None:
                            nodes.append(node3)
                        node4 = getattr(node3, "trainer", None) if node3 is not None else None
                        if node4 is not None and getattr(node4, "name", None) is not None:
                            nodes.append(node4)
                return nodes
            
            nodes = _collect(cfg)
            tr_cfg = nodes[0] if nodes else getattr(cfg, "trainer", {})
            for n in nodes[1:]:
                tr_cfg = OmegaConf.merge(tr_cfg, n)
            self.tr_cfg = tr_cfg
        except Exception:
            self.tr_cfg = getattr(cfg, "trainer", {})
        
        # Loss weights
        self.loss_waypoint_weight = float(getattr(self.tr_cfg, "loss_waypoint_weight", 1.0))
        self.loss_speed_weight = float(getattr(self.tr_cfg, "loss_speed_weight", 0.5))
        self.horizon_decay = float(getattr(self.tr_cfg, "horizon_decay", 0.0))
        # Modality dropout for robustness
        self.drop_bev_p = float(getattr(self.tr_cfg, "drop_bev_p", 0.0))
        self.drop_objects_p = float(getattr(self.tr_cfg, "drop_objects_p", 0.0))
        
        # Build optimizer from registry
        opt_cfg = getattr(self.tr_cfg, "optimizer", None)
        
        if opt_cfg is None or getattr(opt_cfg, "name", None) is None:
            raise KeyError("Missing trainer.optimizer.name in config")
        
        opt_factory = get("optimizer", opt_cfg.name)
        
        if hasattr(opt_factory, "build"):
            self.optimizer = opt_factory().build(opt_cfg, {"model": self.model})
        else:
            self.optimizer = opt_factory(opt_cfg, {"model": self.model})
        
        # EMA (Exponential Moving Average) for stability
        self.use_ema = bool(getattr(self.tr_cfg, "use_ema", True))
        self.ema_decay = float(getattr(self.tr_cfg, "ema_decay", 0.9999))
        if self.use_ema:
            from copy import deepcopy
            self.ema_model = deepcopy(self.model)
            for param in self.ema_model.parameters():
                param.requires_grad = False
            print(f"[BCTrainer] Using EMA with decay={self.ema_decay}")
        else:
            self.ema_model = None
        
        # Gradient accumulation
        self.gradient_accumulation_steps = int(getattr(self.tr_cfg, "gradient_accumulation_steps", 1))
        if self.gradient_accumulation_steps > 1:
            print(f"[BCTrainer] Using gradient accumulation with {self.gradient_accumulation_steps} steps")
        
        # Early stopping
        es_cfg = getattr(self.tr_cfg, "early_stopping", None)
        if es_cfg is not None and bool(getattr(es_cfg, "enabled", False)):
            self.early_stopping = EarlyStopping(
                patience=int(getattr(es_cfg, "patience", 7)),
                min_delta=float(getattr(es_cfg, "min_delta", 0.0001)),
                mode="min",  # Lower loss is better
            )
            print(f"[BCTrainer] Early stopping enabled: patience={self.early_stopping.patience}, min_delta={self.early_stopping.min_delta}")
        else:
            self.early_stopping = None

        # GPU augmentation (for heavy ops like rotation, cutout)
        aug_cfg = getattr(cfg, "augmentation", None)
        if aug_cfg is None:
            exp = getattr(cfg, "exp", None)
            if exp is not None:
                aug_cfg = getattr(exp, "augmentation", None)

        self.gpu_augmentation = None
        if aug_cfg is not None and bool(getattr(aug_cfg, "enabled", False)):
            enable_rotation = bool(getattr(aug_cfg, "enable_rotation", False))
            self.gpu_augmentation = GPUBatchAugmentation(
                p_rot=float(getattr(aug_cfg, "p_rot", 0.0)),
                enable_rotation=enable_rotation,
                max_rot_deg=float(getattr(aug_cfg, "max_rot_deg", 20.0)),
                p_bev_cutout=float(getattr(aug_cfg, "p_bev_cutout", 0.2)),
                p_bev_channel_drop=float(getattr(aug_cfg, "p_bev_channel_drop", 0.1)),
            ).to(self.device)
            if enable_rotation and self.gpu_augmentation.p_rot > 0.0:
                print(f"[BCTrainer] GPU augmentation enabled: rotation ON, cutout, channel drop")
            else:
                print(f"[BCTrainer] GPU augmentation enabled: rotation OFF, cutout, channel drop")
    
    def _update_ema(self):
        """Update EMA model parameters."""
        if self.ema_model is None:
            return
        with torch.no_grad():
            for ema_param, model_param in zip(
                self.ema_model.parameters(), 
                self.model.parameters()
            ):
                ema_param.data.mul_(self.ema_decay).add_(
                    model_param.data, alpha=1.0 - self.ema_decay
                )
    
    def _make_loaders(self) -> Dict[str, DataLoader]:
        """Create BC dataloaders from dataset config."""
        # Import here to avoid circular imports
        from data.torch_dataset import create_bc_dataloader
        
        # Get dataset config
        ds_cfg = getattr(self.cfg, "dataset", None)
        if ds_cfg is None:
            exp = getattr(self.cfg, "exp", None)
            if exp is not None:
                ds_cfg = getattr(exp, "dataset", None)
        
        if ds_cfg is None:
            raise ValueError("Missing dataset config in cfg.dataset or cfg.exp.dataset")
        
        # Extract params
        run_dir = str(getattr(ds_cfg, "run_dir", "data/BC_v1/run-20251116-140827"))
        future_horizon = int(getattr(ds_cfg, "future_horizon", 12))
        route_points = int(getattr(ds_cfg, "route_points", 32))
        max_objects = int(getattr(ds_cfg, "max_objects", 64))
        
        # Loader config
        loader_cfg = getattr(self.tr_cfg, "loader", None)
        batch_size = int(getattr(self.tr_cfg, "batch_size", 32))
        num_workers = int(getattr(loader_cfg, "num_workers", 8)) if loader_cfg else 8
        prefetch_factor = int(getattr(loader_cfg, "prefetch_factor", 2)) if loader_cfg else 2
        persistent_workers = bool(getattr(loader_cfg, "persistent_workers", True)) if loader_cfg else True
        pin_memory = (self.device.type == "cuda") and (bool(getattr(loader_cfg, "pin_memory", True)) if loader_cfg else True)
        shuffle_buffer_size = int(getattr(loader_cfg, "shuffle_buffer_size", 5000)) if loader_cfg else 5000
        
        # Ensure run_dir is absolute or relative to project root
        if not os.path.isabs(run_dir):
            try:
                from hydra.utils import get_original_cwd
                project_root = get_original_cwd()
                run_dir = os.path.join(project_root, run_dir)
            except Exception:
                pass  # Use as-is if hydra not available
        
        # Get augmentation config from exp config
        aug_cfg = getattr(self.cfg, "augmentation", None)
        if aug_cfg is None:
            exp = getattr(self.cfg, "exp", None)
            if exp is not None:
                aug_cfg = getattr(exp, "augmentation", None)
        
        # Convert OmegaConf to dict if needed
        augment_config = None
        if aug_cfg is not None and bool(getattr(aug_cfg, "enabled", False)):
            try:
                from omegaconf import OmegaConf
                augment_config = OmegaConf.to_container(aug_cfg, resolve=True)
            except Exception:
                augment_config = dict(aug_cfg) if hasattr(aug_cfg, "__iter__") else None
        
        # Get diversity validation config
        div_cfg = getattr(ds_cfg, "diversity", None)
        validate_diversity = div_cfg is not None and bool(getattr(div_cfg, "enabled", False))
        diversity_config = None
        if validate_diversity:
            try:
                from omegaconf import OmegaConf
                diversity_config = OmegaConf.to_container(div_cfg, resolve=True)
            except Exception:
                diversity_config = dict(div_cfg) if hasattr(div_cfg, "__iter__") else None
        
        # Create dataloaders using WebLoader pattern for high throughput
        train_loader = create_bc_dataloader(
            run_dir=run_dir,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory,
            drop_last=False,
            future_horizon=future_horizon,
            route_points=route_points,
            max_objects=max_objects,
            augment=True,
            split="train",
            val_ratio=0.2,
            shuffle_buffer_size=shuffle_buffer_size,
            validate_diversity=validate_diversity,
            diversity_config=diversity_config,
            augment_config=augment_config,
        )
        
        valid_loader = create_bc_dataloader(
            run_dir=run_dir,
            batch_size=batch_size,
            shuffle=False,
            num_workers=6,  # Fewer workers for validation
            prefetch_factor=3,
            persistent_workers=True,  # Keep workers alive to avoid restart overhead
            pin_memory=pin_memory,
            drop_last=False,
            future_horizon=future_horizon,
            route_points=route_points,
            max_objects=max_objects,
            augment=False,
            split="val",
            val_ratio=0.2,
            shuffle_buffer_size=1000,  # Smaller buffer for validation
        )
        
        return {"train": train_loader, "valid": valid_loader}
    
    def compute_loss(self, predictions: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute BC losses with masking.
        
        Args:
            predictions: Dict with 'future_xy' (B, N, 2) and 'future_v' (B, N)
            batch: Dict with ground truth futures and masks
            
        Returns:
            Dict with 'loss', 'loss_waypoint', 'loss_speed'
        """
        # Extract predictions
        pred_xy = predictions["future_xy"]  # (B, N, 2)
        pred_v = predictions["future_v"]    # (B, N)
        
        # Extract ground truth
        gt_xy = batch["future_xy"]          # (B, N, 2)
        gt_v = batch["future_v"]            # (B, N)
        future_mask = batch["future_mask"]  # (B, N)
        
        # Optional horizon weighting (emphasize near-term steps)
        if self.horizon_decay > 0.0:
            weights = torch.exp(
                -self.horizon_decay * torch.arange(1, future_mask.shape[1] + 1, device=future_mask.device)
            )  # (N,)
            weights = weights.unsqueeze(0)  # (1, N)
        else:
            weights = torch.ones_like(future_mask)
        
        valid_weights = future_mask * weights
        if torch.sum(valid_weights) <= 0:
            raise ValueError("future_mask has no valid entries; cannot compute loss.")
        
        # Compute loss in physical units (meters, m/s) for healthy gradients
        waypoint_diff_m = (pred_xy - gt_xy) * SPATIAL_MAX  # (B, N, 2)
        waypoint_loss = waypoint_diff_m.pow(2).sum(dim=-1)  # (B, N)
        waypoint_loss = (waypoint_loss * valid_weights).sum() / (valid_weights.sum() + 1e-8)
        
        # Huber loss on speeds in m/s
        speed_loss_raw = F.smooth_l1_loss(pred_v * V_MAX, gt_v * V_MAX, reduction="none")  # (B, N)
        speed_loss = (speed_loss_raw * valid_weights).sum() / (valid_weights.sum() + 1e-8)
        
        # Combined loss
        total_loss = (
            self.loss_waypoint_weight * waypoint_loss +
            self.loss_speed_weight * speed_loss
        )
        
        return {
            "loss": total_loss,
            "loss_waypoint": waypoint_loss,
            "loss_speed": speed_loss,
        }
    
    def fit(self):
        """Main training loop."""
        loaders = self._make_loaders()
        epochs = int(getattr(self.tr_cfg, "epochs", 100))
        use_amp, amp_dtype, scaler = self.get_amp_settings()
        
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = bool(getattr(getattr(self.cfg, "speed", {}), "cudnn_benchmark", True))
        
        # Calculate batches per epoch correctly
        # For WebLoader, we need to count actual batches in first epoch
        # due to potential remainder batches not accounted for in .with_epoch()
        train_loader = loaders["train"]

        # Try to get initial estimate
        # Priority order: .epoch attribute (from .with_epoch), dataset sample count, __len__, config fallback
        estimated_steps_per_epoch = None

        # 1. WebLoader with .with_epoch(num_batches) sets this attribute
        if hasattr(train_loader, 'epoch') and train_loader.epoch is not None:
            estimated_steps_per_epoch = train_loader.epoch
            print(f"[BCTrainer] Estimated steps from loader.epoch: {estimated_steps_per_epoch}")

        # 2. BCWebDataset with known sample count
        if estimated_steps_per_epoch is None and hasattr(train_loader, 'dataset') and hasattr(train_loader.dataset, 'split_sample_count'):
            batch_size = self.tr_cfg.batch_size
            estimated_steps_per_epoch = max(1, train_loader.dataset.split_sample_count // batch_size)
            print(f"[BCTrainer] Estimated steps from dataset: {train_loader.dataset.split_sample_count} samples // {batch_size} batch_size = {estimated_steps_per_epoch}")

        # 3. Loader with __len__
        if estimated_steps_per_epoch is None and hasattr(train_loader, '__len__'):
            estimated_steps_per_epoch = len(train_loader)
            print(f"[BCTrainer] Estimated steps from len(loader): {estimated_steps_per_epoch}")

        # 4. Fallback: config
        if estimated_steps_per_epoch is None:
            batch_size = self.tr_cfg.batch_size
            ds_cfg = getattr(self.cfg, "dataset", None)
            total_samples = getattr(ds_cfg, "total_samples", 10000) if ds_cfg else 10000
            estimated_steps_per_epoch = max(1, total_samples // batch_size)
            print(f"[BCTrainer] Estimated steps from config (fallback): {total_samples} // {batch_size} = {estimated_steps_per_epoch}")

        # We'll determine actual steps_per_epoch from first epoch
        actual_steps_per_epoch = None
        steps_per_epoch = estimated_steps_per_epoch  # Initial estimate

        # IMPORTANT: Don't build scheduler yet - wait for actual step count
        # Estimated steps can be wildly wrong (e.g., 437 estimated vs 55 actual)
        scheduler = None
        
        log_interval = int(getattr(getattr(self.cfg, "logger", {}), "log_interval", 100))
        
        try:
            pbar_outer = tqdm(total=epochs, desc="Epochs", leave=True)
        except Exception:
            pbar_outer = None
        
        self.model.train()
        global_step = 0
        
        try:
            for epoch in range(1, epochs + 1):
                t0 = time.perf_counter()
                running_loss = 0.0
                running_waypoint_loss = 0.0
                running_speed_loss = 0.0
                running_samples = 0

                # Use actual count from first epoch, otherwise estimated
                epoch_steps = actual_steps_per_epoch if actual_steps_per_epoch is not None else steps_per_epoch

                try:
                    pbar_inner = tqdm(total=epoch_steps, desc=f"Train {epoch}/{epochs}", leave=False)
                except Exception:
                    pbar_inner = None

                epoch_batch_count = 0
                for batch_idx, batch in enumerate(loaders["train"], start=1):
                    # Skip empty batches (shouldn't happen with proper WebDataset, but safety check)
                    if not batch or "ego_vec" not in batch:
                        continue

                    # Only count valid batches that actually train
                    epoch_batch_count += 1

                    batch = self.to_device(batch)

                    # Fail fast on obviously degenerate data (first batch only)
                    if epoch == 1 and batch_idx == 1:
                        fut_v_mps = batch["future_v"] * V_MAX
                        fut_xy_m = batch["future_xy"] * SPATIAL_MAX
                        speed_std = float(fut_v_mps.std().item())
                        xy_std = float(fut_xy_m.std().item())
                        if speed_std < 0.1:
                            raise ValueError(f"[BCTrainer] future_v std too low ({speed_std:.3f} m/s). Check data normalization/collection.")
                        if xy_std < 0.5:
                            raise ValueError(f"[BCTrainer] future_xy std too low ({xy_std:.3f} m). Check data normalization/collection.")
                        print(f"[BCTrainer] Data sanity (first batch): future_v std={speed_std:.3f} m/s, future_xy std={xy_std:.3f} m")

                    # Apply GPU augmentations (heavy ops: rotation, cutout)
                    if self.gpu_augmentation is not None:
                        with torch.no_grad():
                            batch = self.gpu_augmentation(batch)

                    # Modality dropout for robustness (trainer-level, separate from data augmentation)
                    if self.drop_bev_p > 0.0 and torch.rand(1).item() < self.drop_bev_p:
                        batch["bev"] = torch.zeros_like(batch["bev"])
                    if self.drop_objects_p > 0.0 and torch.rand(1).item() < self.drop_objects_p:
                        batch["objects"] = torch.zeros_like(batch["objects"])
                        batch["object_mask"] = torch.zeros_like(batch["object_mask"])

                    # Forward pass
                    if use_amp:
                        with torch.autocast(device_type=self.device.type, dtype=amp_dtype):
                            predictions = self.model(batch)
                            losses = self.compute_loss(predictions, batch)
                            loss = losses["loss"]
                    else:
                        predictions = self.model(batch)
                        losses = self.compute_loss(predictions, batch)
                        loss = losses["loss"]
                    
                    # Store unscaled loss for logging
                    unscaled_loss = loss.item()
                    unscaled_waypoint_loss = losses["loss_waypoint"].item()
                    unscaled_speed_loss = losses["loss_speed"].item()
                    
                    # Scale loss for gradient accumulation
                    loss = loss / self.gradient_accumulation_steps
                    
                    # Backward pass
                    if scaler.is_enabled():
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    
                    # Only step optimizer every N accumulation steps
                    if batch_idx % self.gradient_accumulation_steps == 0:
                        if scaler.is_enabled():
                            self.clip_grad(self.model.parameters(), float(getattr(self.tr_cfg, "max_grad_norm", 0.0)))
                            scaler.step(self.optimizer)
                            scaler.update()
                        else:
                            self.clip_grad(self.model.parameters(), float(getattr(self.tr_cfg, "max_grad_norm", 0.0)))
                            self.optimizer.step()
                        
                        self.optimizer.zero_grad(set_to_none=True)
                        
                        # Update EMA after optimizer step
                        if self.use_ema:
                            self._update_ema()
                        
                        # Update learning rate (only if scheduler is built)
                        if scheduler is not None:
                            scheduler.step()
                        elif global_step == 0:
                            # First step, no scheduler yet - that's expected
                            pass
                    
                    # Track metrics (use unscaled loss)
                    batch_size = batch["ego_vec"].size(0)
                    running_loss += unscaled_loss * batch_size
                    running_waypoint_loss += unscaled_waypoint_loss * batch_size
                    running_speed_loss += unscaled_speed_loss * batch_size
                    running_samples += batch_size
                    
                    global_step += 1
                    
                    if pbar_inner is not None:
                        try:
                            pbar_inner.update(1)
                            pbar_inner.set_postfix({
                                "loss": f"{unscaled_loss:.4f}",
                                "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                            })
                        except Exception:
                            pass
                    
                    if global_step % log_interval == 0:
                        self.logger.log({
                            "step": global_step,
                            "epoch": epoch,
                            "train/step_loss": float(unscaled_loss),
                            "train/step_waypoint_loss": float(unscaled_waypoint_loss),
                            "train/step_speed_loss": float(unscaled_speed_loss),
                            "lr": float(self.optimizer.param_groups[0]["lr"]),
                        })
                
                # Capture actual batch count from first epoch and build scheduler
                if actual_steps_per_epoch is None:
                    actual_steps_per_epoch = epoch_batch_count
                    if actual_steps_per_epoch != estimated_steps_per_epoch:
                        print(f"[BCTrainer] Note: actual batches/epoch = {actual_steps_per_epoch} (estimated {estimated_steps_per_epoch})")

                    # NOW build scheduler with correct total_steps
                    total_steps = actual_steps_per_epoch * epochs
                    scheduler = self.build_lr_scheduler(self.optimizer, total_steps)
                    print(f"[BCTrainer] LR scheduler: {total_steps} total steps ({actual_steps_per_epoch} steps/epoch × {epochs} epochs)")

                    # Fast-forward scheduler to current step and update LR
                    if scheduler is not None and global_step > 0:
                        # Set the scheduler to be at the current step
                        # After epoch 1, global_step = actual_steps_per_epoch
                        # We want scheduler to resume from there
                        scheduler.last_epoch = global_step
                        # Manually update LR without calling step() to avoid warning
                        # The scheduler will compute the correct LR based on last_epoch
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else scheduler.get_lr()[0]
                        print(f"[BCTrainer] Scheduler initialized at step {global_step}, LR={self.optimizer.param_groups[0]['lr']:.2e}")

                if pbar_inner is not None:
                    try:
                        pbar_inner.close()
                    except Exception:
                        pass
                
                epoch_time = time.perf_counter() - t0
                train_loss = running_loss / max(1, running_samples)
                train_waypoint_loss = running_waypoint_loss / max(1, running_samples)
                train_speed_loss = running_speed_loss / max(1, running_samples)
                
                # Validation (use EMA model if available)
                eval_model = self.ema_model if self.use_ema else self.model
                eval_model.eval()
                val_loss = 0.0
                val_waypoint_loss = 0.0
                val_speed_loss = 0.0
                val_samples = 0

                # Add progress bar for validation
                # Estimate number of validation batches
                val_batches = len(loaders["valid"]) if hasattr(loaders["valid"], '__len__') else None
                if val_batches is None:
                    # Fallback: estimate from dataset
                    try:
                        val_samples = loaders["valid"].dataset.split_sample_count
                        val_batches = val_samples // self.tr_cfg.batch_size
                    except:
                        val_batches = None

                try:
                    pbar_val = tqdm(total=val_batches, desc=f"Valid {epoch}/{epochs}", leave=False)
                except Exception:
                    pbar_val = None

                with torch.no_grad():
                    for batch in loaders["valid"]:
                        batch = self.to_device(batch)
                        predictions = eval_model(batch)
                        losses = self.compute_loss(predictions, batch)
                        batch_size = batch["ego_vec"].size(0)
                        val_loss += losses["loss"].item() * batch_size
                        val_waypoint_loss += losses["loss_waypoint"].item() * batch_size
                        val_speed_loss += losses["loss_speed"].item() * batch_size
                        val_samples += batch_size

                        if pbar_val is not None:
                            try:
                                pbar_val.update(1)
                                pbar_val.set_postfix({"loss": f"{losses['loss'].item():.4f}"})
                            except Exception:
                                pass

                if pbar_val is not None:
                    try:
                        pbar_val.close()
                    except Exception:
                        pass

                val_loss = val_loss / max(1, val_samples)
                val_waypoint_loss = val_waypoint_loss / max(1, val_samples)
                val_speed_loss = val_speed_loss / max(1, val_samples)
                self.model.train()
                
                # Log epoch metrics
                self.logger.log({
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/waypoint_loss": train_waypoint_loss,
                    "train/speed_loss": train_speed_loss,
                    "valid/loss": val_loss,
                    "valid/waypoint_loss": val_waypoint_loss,
                    "valid/speed_loss": val_speed_loss,
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "time/epoch_sec": epoch_time,
                })
                
                # Save checkpoint (lower loss is better)
                self.save_checkpoint(step=epoch, objective=val_loss, extra={"val_loss": val_loss})
                
                # Early stopping check
                if self.early_stopping is not None:
                    should_stop = self.early_stopping(val_loss, epoch)
                    es_status = self.early_stopping.status()
                    
                    if pbar_outer is not None:
                        try:
                            pbar_outer.update(1)
                            pbar_outer.set_postfix({
                                "train_loss": f"{train_loss:.4f}",
                                "val_loss": f"{val_loss:.4f}",
                                "es": es_status,
                            })
                        except Exception:
                            pass
                    
                    if should_stop:
                        print(f"\n[BCTrainer] Early stopping triggered at epoch {epoch}!")
                        print(f"[BCTrainer] Best validation loss: {self.early_stopping.best_score:.6f} at epoch {self.early_stopping.best_epoch}")
                        break
                else:
                    if pbar_outer is not None:
                        try:
                            pbar_outer.update(1)
                            pbar_outer.set_postfix({
                                "train_loss": f"{train_loss:.4f}",
                                "val_loss": f"{val_loss:.4f}",
                            })
                        except Exception:
                            pass
        
        finally:
            if pbar_outer is not None:
                try:
                    pbar_outer.close()
                except Exception:
                    pass
            try:
                self.logger.finish()
            except Exception:
                pass

