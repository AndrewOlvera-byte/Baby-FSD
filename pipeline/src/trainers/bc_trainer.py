"""
Behavior Cloning Trainer.

Trains a policy to imitate expert demonstrations from BC dataset.
"""

from __future__ import annotations
import os
import time
from typing import Dict, Any
import torch
from torch import nn
from torch.utils.data import DataLoader
from src.core.trainer_base import BaseTrainer
from src.core.registry import register, get
from tqdm import tqdm


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
        self.loss_speed_weight = float(getattr(self.tr_cfg, "loss_speed_weight", 0.1))
        
        # Build optimizer from registry
        opt_cfg = getattr(self.tr_cfg, "optimizer", None)
        
        if opt_cfg is None or getattr(opt_cfg, "name", None) is None:
            raise KeyError("Missing trainer.optimizer.name in config")
        
        opt_factory = get("optimizer", opt_cfg.name)
        
        if hasattr(opt_factory, "build"):
            self.optimizer = opt_factory().build(opt_cfg, {"model": self.model})
        else:
            self.optimizer = opt_factory(opt_cfg, {"model": self.model})
    
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
        prefetch_factor = int(getattr(loader_cfg, "prefetch_factor", 4)) if loader_cfg else 4
        persistent_workers = bool(getattr(loader_cfg, "persistent_workers", True)) if loader_cfg else True
        pin_memory = (self.device.type == "cuda") and (bool(getattr(loader_cfg, "pin_memory", True)) if loader_cfg else True)
        
        # Ensure run_dir is absolute or relative to project root
        if not os.path.isabs(run_dir):
            try:
                from hydra.utils import get_original_cwd
                project_root = get_original_cwd()
                run_dir = os.path.join(project_root, run_dir)
            except Exception:
                pass  # Use as-is if hydra not available
        
        # Create dataloaders
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
        )
        
        # For now, use same data for validation (TODO: split dataset)
        valid_loader = create_bc_dataloader(
            run_dir=run_dir,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory,
            drop_last=False,
            future_horizon=future_horizon,
            route_points=route_points,
            max_objects=max_objects,
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
        
        # Compute MSE losses
        # Waypoint loss (x, y)
        waypoint_loss = (pred_xy - gt_xy).pow(2).sum(dim=-1)  # (B, N)
        waypoint_loss = (waypoint_loss * future_mask).sum() / (future_mask.sum() + 1e-8)
        
        # Speed loss
        speed_loss = (pred_v - gt_v).pow(2)  # (B, N)
        speed_loss = (speed_loss * future_mask).sum() / (future_mask.sum() + 1e-8)
        
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
        
        steps_per_epoch = max(1, len(loaders["train"]))
        total_steps = steps_per_epoch * epochs
        scheduler = self.build_lr_scheduler(self.optimizer, total_steps)
        
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
                
                try:
                    pbar_inner = tqdm(total=len(loaders["train"]), desc=f"Train {epoch}/{epochs}", leave=False)
                except Exception:
                    pbar_inner = None
                
                for batch_idx, batch in enumerate(loaders["train"], start=1):
                    batch = self.to_device(batch)
                    
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
                    
                    # Backward pass
                    self.optimizer.zero_grad(set_to_none=True)
                    if scaler.is_enabled():
                        scaler.scale(loss).backward()
                        self.clip_grad(self.model.parameters(), float(getattr(self.tr_cfg, "max_grad_norm", 0.0)))
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        self.clip_grad(self.model.parameters(), float(getattr(self.tr_cfg, "max_grad_norm", 0.0)))
                        self.optimizer.step()
                    
                    # Update learning rate
                    if scheduler is not None:
                        scheduler.step()
                    
                    # Track metrics
                    batch_size = batch["ego_vec"].size(0)
                    running_loss += loss.item() * batch_size
                    running_waypoint_loss += losses["loss_waypoint"].item() * batch_size
                    running_speed_loss += losses["loss_speed"].item() * batch_size
                    running_samples += batch_size
                    
                    global_step += 1
                    
                    if pbar_inner is not None:
                        try:
                            pbar_inner.update(1)
                            pbar_inner.set_postfix({
                                "loss": f"{loss.item():.4f}",
                                "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                            })
                        except Exception:
                            pass
                    
                    if global_step % log_interval == 0:
                        self.logger.log({
                            "step": global_step,
                            "epoch": epoch,
                            "train/step_loss": float(loss.item()),
                            "train/step_waypoint_loss": float(losses["loss_waypoint"].item()),
                            "train/step_speed_loss": float(losses["loss_speed"].item()),
                            "lr": float(self.optimizer.param_groups[0]["lr"]),
                        })
                
                if pbar_inner is not None:
                    try:
                        pbar_inner.close()
                    except Exception:
                        pass
                
                epoch_time = time.perf_counter() - t0
                train_loss = running_loss / max(1, running_samples)
                train_waypoint_loss = running_waypoint_loss / max(1, running_samples)
                train_speed_loss = running_speed_loss / max(1, running_samples)
                
                # Validation (simple pass for now)
                self.model.eval()
                val_loss = 0.0
                val_samples = 0
                with torch.no_grad():
                    for batch in loaders["valid"]:
                        batch = self.to_device(batch)
                        predictions = self.model(batch)
                        losses = self.compute_loss(predictions, batch)
                        batch_size = batch["ego_vec"].size(0)
                        val_loss += losses["loss"].item() * batch_size
                        val_samples += batch_size
                val_loss = val_loss / max(1, val_samples)
                self.model.train()
                
                # Log epoch metrics
                self.logger.log({
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/waypoint_loss": train_waypoint_loss,
                    "train/speed_loss": train_speed_loss,
                    "valid/loss": val_loss,
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "time/epoch_sec": epoch_time,
                })
                
                # Save checkpoint (lower loss is better)
                self.save_checkpoint(step=epoch, objective=val_loss, extra={"val_loss": val_loss})
                
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

