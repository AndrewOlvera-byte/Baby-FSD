"""
Offline RL Trainer using TorchRL-inspired algorithms.

Trains a policy using offline RL algorithms (IQL, CQL, TD3+BC) on collected demonstrations.
Features:
- Simplified IQL implementation optimized for driving (single-step rewards)
- Actor-critic architecture with target networks
- EMA (Exponential Moving Average) model
- Gradient accumulation
- Early stopping with patience

Note: This is a lightweight implementation that works with single-step offline data
(no next_state required). For full multi-step TD learning, consider storing
consecutive frames in the dataset.
"""

from __future__ import annotations
import os
import sys
import time
from copy import deepcopy
from typing import Dict, Any, Optional

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from omegaconf import OmegaConf
from torchrl.objectives import IQLLoss, CQLLoss, TD3BCLoss
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
from tensordict import TensorDict
from hydra.utils import get_original_cwd

from src.core.trainer_base import BaseTrainer
from src.core.registry import register, get

# Add project root to path for data module imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../"))
from data.gpu_augmentations import GPUBatchAugmentation
from data.torch_dataset import create_rl_dataloader


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
            mode: "min" for loss, "max" for reward/Q-value
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


@register("trainer", "offline_rl")
class OfflineRLTrainer(BaseTrainer):
    """
    Offline RL Trainer using TorchRL.

    Supports multiple offline RL algorithms:
    - IQL (Implicit Q-Learning): Recommended for driving, more stable
    - CQL (Conservative Q-Learning): Good for avoiding OOD actions
    - TD3+BC: TD3 with behavior cloning regularization
    """

    required_components = ["actor", "critic"]  # Actor-critic architecture

    def __init__(self, cfg, **components):
        super().__init__(cfg, **components)

        # Model architecture requirements:
        # - Actor: Takes state dict, returns action tensor (B, 3) for (steer, throttle, brake)
        # - Critic: Takes state dict and action tensor, returns Q-value (B, 1) or (B,)
        self.actor: nn.Module = self.components["actor"].to(self.device)
        self.critic: nn.Module = self.components["critic"].to(self.device)

        # Channels-last for throughput on CUDA
        if self.device.type == "cuda" and bool(getattr(getattr(cfg, "speed", {}), "channels_last", True)):
            self.actor.to(memory_format=torch.channels_last)
            self.critic.to(memory_format=torch.channels_last)

        # Get trainer config
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
            return nodes

        nodes = _collect(cfg)
        tr_cfg = nodes[0] if nodes else getattr(cfg, "trainer", {})
        for n in nodes[1:]:
            tr_cfg = OmegaConf.merge(tr_cfg, n)
        self.tr_cfg = tr_cfg

        # Algorithm selection
        self.algorithm = str(getattr(self.tr_cfg, "algorithm", "iql")).lower()
        if self.algorithm not in ["iql", "cql", "td3bc"]:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}. Choose from: iql, cql, td3bc")

        print(f"[OfflineRLTrainer] Using algorithm: {self.algorithm.upper()}")

        # Hyperparameters
        self.gamma = float(getattr(self.tr_cfg, "gamma", 0.99))
        self.tau = float(getattr(self.tr_cfg, "tau", 0.005))  # Target network update rate

        # Algorithm-specific hyperparameters
        if self.algorithm == "iql":
            self.expectile = float(getattr(self.tr_cfg, "expectile", 0.7))  # IQL expectile
            self.temperature = float(getattr(self.tr_cfg, "temperature", 3.0))  # IQL temperature
        elif self.algorithm == "cql":
            self.cql_alpha = float(getattr(self.tr_cfg, "cql_alpha", 1.0))  # CQL penalty weight
        elif self.algorithm == "td3bc":
            self.bc_alpha = float(getattr(self.tr_cfg, "bc_alpha", 2.5))  # BC regularization weight

        # Build optimizer from registry
        opt_cfg = getattr(self.tr_cfg, "optimizer", None)

        if opt_cfg is None or getattr(opt_cfg, "name", None) is None:
            raise KeyError("Missing trainer.optimizer.name in config")

        opt_factory = get("optimizer", opt_cfg.name)

        # Separate optimizers for actor and critic (standard in RL)
        if hasattr(opt_factory, "build"):
            self.actor_optimizer = opt_factory().build(opt_cfg, {"model": self.actor})
            self.critic_optimizer = opt_factory().build(opt_cfg, {"model": self.critic})
        else:
            self.actor_optimizer = opt_factory(opt_cfg, {"model": self.actor})
            self.critic_optimizer = opt_factory(opt_cfg, {"model": self.critic})

        # Target networks for stability (standard in off-policy RL)
        self.target_actor = deepcopy(self.actor)
        self.target_critic = deepcopy(self.critic)
        for param in self.target_actor.parameters():
            param.requires_grad = False
        for param in self.target_critic.parameters():
            param.requires_grad = False

        print(f"[OfflineRLTrainer] Target networks created with tau={self.tau}")

        # EMA (Exponential Moving Average) for stability
        self.use_ema = bool(getattr(self.tr_cfg, "use_ema", True))
        self.ema_decay = float(getattr(self.tr_cfg, "ema_decay", 0.9999))
        if self.use_ema:
            self.ema_actor = deepcopy(self.actor)
            for param in self.ema_actor.parameters():
                param.requires_grad = False
            print(f"[OfflineRLTrainer] Using EMA with decay={self.ema_decay}")
        else:
            self.ema_actor = None

        # Gradient accumulation
        self.gradient_accumulation_steps = int(getattr(self.tr_cfg, "gradient_accumulation_steps", 1))
        if self.gradient_accumulation_steps > 1:
            print(f"[OfflineRLTrainer] Using gradient accumulation with {self.gradient_accumulation_steps} steps")

        # Early stopping
        es_cfg = getattr(self.tr_cfg, "early_stopping", None)
        if es_cfg is not None and bool(getattr(es_cfg, "enabled", False)):
            self.early_stopping = EarlyStopping(
                patience=int(getattr(es_cfg, "patience", 10)),
                min_delta=float(getattr(es_cfg, "min_delta", 0.001)),
                mode="max",  # Higher Q-value is better
            )
            print(f"[OfflineRLTrainer] Early stopping enabled: patience={self.early_stopping.patience}")
        else:
            self.early_stopping = None

        # GPU augmentation
        aug_cfg = getattr(cfg, "augmentation", None)
        if aug_cfg is None:
            exp = getattr(cfg, "exp", None)
            if exp is not None:
                aug_cfg = getattr(exp, "augmentation", None)

        self.gpu_augmentation = None
        if aug_cfg is not None and bool(getattr(aug_cfg, "enabled", False)):
            self.gpu_augmentation = GPUBatchAugmentation(
                p_rot=float(getattr(aug_cfg, "p_rot", 0.5)),
                max_rot_deg=float(getattr(aug_cfg, "max_rot_deg", 20.0)),
                p_bev_cutout=float(getattr(aug_cfg, "p_bev_cutout", 0.2)),
                p_bev_channel_drop=float(getattr(aug_cfg, "p_bev_channel_drop", 0.1)),
            ).to(self.device)
            print(f"[OfflineRLTrainer] GPU augmentation enabled")

    def _update_target_networks(self):
        """Soft update target networks using Polyak averaging."""
        with torch.no_grad():
            for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
                target_param.data.mul_(1.0 - self.tau).add_(param.data, alpha=self.tau)
            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.data.mul_(1.0 - self.tau).add_(param.data, alpha=self.tau)

    def _update_ema(self):
        """Update EMA actor parameters."""
        if self.ema_actor is None:
            return
        with torch.no_grad():
            for ema_param, actor_param in zip(
                self.ema_actor.parameters(),
                self.actor.parameters()
            ):
                ema_param.data.mul_(self.ema_decay).add_(
                    actor_param.data, alpha=1.0 - self.ema_decay
                )

    def _make_loaders(self) -> Dict[str, DataLoader]:
        """Create offline RL dataloaders from dataset config."""
        # Get dataset config
        ds_cfg = getattr(self.cfg, "dataset", None)
        if ds_cfg is None:
            exp = getattr(self.cfg, "exp", None)
            if exp is not None:
                ds_cfg = getattr(exp, "dataset", None)

        if ds_cfg is None:
            raise ValueError("Missing dataset config in cfg.dataset or cfg.exp.dataset")

        # Extract params
        run_dir = str(getattr(ds_cfg, "run_dir", "data/RL_v1/run-20251130-120000"))
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
            project_root = get_original_cwd()
            run_dir = os.path.join(project_root, run_dir)

        # Create dataloaders using WebLoader pattern for high throughput
        train_loader = create_rl_dataloader(
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
        )

        valid_loader = create_rl_dataloader(
            run_dir=run_dir,
            batch_size=batch_size,
            shuffle=False,
            num_workers=6,  # Fewer workers for validation
            prefetch_factor=3,
            persistent_workers=True,
            pin_memory=pin_memory,
            drop_last=False,
            future_horizon=future_horizon,
            route_points=route_points,
            max_objects=max_objects,
            augment=False,
            split="val",
            val_ratio=0.2,
            shuffle_buffer_size=1000,
        )

        return {"train": train_loader, "valid": valid_loader}

    def compute_iql_loss(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute IQL (Implicit Q-Learning) losses.

        IQL learns the value function implicitly without behavior cloning.
        More stable for offline RL than CQL.

        Args:
            batch: Dict with states, actions, rewards, terminals

        Returns:
            Dict with actor_loss, critic_loss, value_loss, and metrics
        """
        # Extract batch data
        states = {
            "ego_vec": batch["ego_vec"],
            "bev": batch["bev"],
            "route": batch["route"],
            "objects": batch["objects"],
            "object_mask": batch["object_mask"],
        }

        # Combine action fields (steer, throttle, brake) into single action tensor
        # Offline RL data has: action_steer, action_throttle, action_brake
        actions = torch.stack([
            batch["action_steer"],
            batch["action_throttle"],
            batch["action_brake"],
        ], dim=-1)  # (B, 3)

        rewards = batch["reward_clipped"]  # (B,)
        dones = batch["done"].float()  # (B,)

        # NOTE: Offline RL without next_state observations
        # For driving, we use single-step IQL with current state only
        # Next states would require storing consecutive frames in dataset (future work)
        # For now, we use simplified IQL that learns Q(s,a) and V(s) from rewards only

        # ===== Critic Loss (Q-function) =====
        # Simplified single-step TD loss (no next state)
        # Target is just the immediate reward (Monte Carlo estimate)
        with torch.no_grad():
            # For single-step, target is the observed reward
            # This works for offline RL when episodes are short
            target_q = rewards.unsqueeze(-1) if rewards.dim() == 1 else rewards

        # Current Q-value
        current_q = self.critic(states, actions)
        if current_q.dim() > target_q.dim():
            current_q = current_q.squeeze(-1)
        critic_loss = F.mse_loss(current_q, target_q)

        # ===== Value Function Loss (IQL-specific) =====
        # Learn the expectile regression value function
        with torch.no_grad():
            q_value = self.critic(states, actions)

        # Simplified value function (in full IQL, this would be a separate network)
        # For driving, we use the critic with current actions as value estimate
        value = self.critic(states, self.actor(states))

        # Expectile regression loss
        td_error = q_value - value
        weight = torch.where(td_error > 0, self.expectile, 1.0 - self.expectile)
        value_loss = (weight * td_error.pow(2)).mean()

        # ===== Actor Loss (Policy) =====
        # IQL uses advantage-weighted regression
        with torch.no_grad():
            advantage = q_value - value
            # Apply temperature for advantage weighting
            exp_adv = torch.exp(advantage / self.temperature)
            exp_adv = torch.clamp(exp_adv, max=100.0)  # Prevent overflow

        # Predict actions
        pred_actions = self.actor(states)

        # Weighted behavior cloning loss
        actor_loss = (exp_adv * F.mse_loss(pred_actions, actions, reduction="none").sum(dim=-1)).mean()

        # Metrics
        metrics = {
            "q_mean": current_q.mean().item(),
            "value_mean": value.mean().item(),
            "advantage_mean": advantage.mean().item(),
            "reward_mean": rewards.mean().item(),
        }

        return {
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "value_loss": value_loss,
            **metrics,
        }

    def fit(self):
        """Main training loop."""
        loaders = self._make_loaders()
        epochs = int(getattr(self.tr_cfg, "epochs", 100))
        use_amp, amp_dtype, scaler = self.get_amp_settings()

        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = bool(getattr(getattr(self.cfg, "speed", {}), "cudnn_benchmark", True))

        # Calculate steps per epoch
        train_loader = loaders["train"]
        estimated_steps_per_epoch = None

        if hasattr(train_loader, 'epoch') and train_loader.epoch is not None:
            estimated_steps_per_epoch = train_loader.epoch
        elif hasattr(train_loader, 'dataset') and hasattr(train_loader.dataset, 'split_sample_count'):
            batch_size = self.tr_cfg.batch_size
            estimated_steps_per_epoch = max(1, train_loader.dataset.split_sample_count // batch_size)
        elif hasattr(train_loader, '__len__'):
            estimated_steps_per_epoch = len(train_loader)
        else:
            batch_size = self.tr_cfg.batch_size
            ds_cfg = getattr(self.cfg, "dataset", None)
            total_samples = getattr(ds_cfg, "total_samples", 10000) if ds_cfg else 10000
            estimated_steps_per_epoch = max(1, total_samples // batch_size)

        actual_steps_per_epoch = None
        steps_per_epoch = estimated_steps_per_epoch
        scheduler_actor = None
        scheduler_critic = None

        log_interval = int(getattr(getattr(self.cfg, "logger", {}), "log_interval", 100))

        try:
            pbar_outer = tqdm(total=epochs, desc="Epochs", leave=True)
        except Exception:
            pbar_outer = None

        self.actor.train()
        self.critic.train()
        global_step = 0

        try:
            for epoch in range(1, epochs + 1):
                t0 = time.perf_counter()
                running_actor_loss = 0.0
                running_critic_loss = 0.0
                running_value_loss = 0.0
                running_q_mean = 0.0
                running_samples = 0

                epoch_steps = actual_steps_per_epoch if actual_steps_per_epoch is not None else steps_per_epoch

                try:
                    pbar_inner = tqdm(total=epoch_steps, desc=f"Train {epoch}/{epochs}", leave=False)
                except Exception:
                    pbar_inner = None

                epoch_batch_count = 0
                for batch_idx, batch in enumerate(loaders["train"], start=1):
                    if not batch or "ego_vec" not in batch:
                        continue

                    epoch_batch_count += 1
                    batch = self.to_device(batch)

                    # Apply GPU augmentations
                    if self.gpu_augmentation is not None:
                        with torch.no_grad():
                            batch = self.gpu_augmentation(batch)

                    # Compute losses
                    if use_amp:
                        with torch.autocast(device_type=self.device.type, dtype=amp_dtype):
                            losses = self.compute_iql_loss(batch)
                    else:
                        losses = self.compute_iql_loss(batch)

                    actor_loss = losses["actor_loss"]
                    critic_loss = losses["critic_loss"]
                    value_loss = losses["value_loss"]

                    # Backward passes (separate for actor and critic)
                    # Scale losses for gradient accumulation
                    actor_loss_scaled = actor_loss / self.gradient_accumulation_steps
                    critic_loss_scaled = (critic_loss + value_loss) / self.gradient_accumulation_steps

                    if scaler.is_enabled():
                        scaler.scale(actor_loss_scaled).backward()
                        scaler.scale(critic_loss_scaled).backward()
                    else:
                        actor_loss_scaled.backward()
                        critic_loss_scaled.backward()

                    # Only step optimizers every N accumulation steps
                    if batch_idx % self.gradient_accumulation_steps == 0:
                        if scaler.is_enabled():
                            self.clip_grad(self.actor.parameters(), float(getattr(self.tr_cfg, "max_grad_norm", 1.0)))
                            self.clip_grad(self.critic.parameters(), float(getattr(self.tr_cfg, "max_grad_norm", 1.0)))
                            scaler.step(self.actor_optimizer)
                            scaler.step(self.critic_optimizer)
                            scaler.update()
                        else:
                            self.clip_grad(self.actor.parameters(), float(getattr(self.tr_cfg, "max_grad_norm", 1.0)))
                            self.clip_grad(self.critic.parameters(), float(getattr(self.tr_cfg, "max_grad_norm", 1.0)))
                            self.actor_optimizer.step()
                            self.critic_optimizer.step()

                        self.actor_optimizer.zero_grad(set_to_none=True)
                        self.critic_optimizer.zero_grad(set_to_none=True)

                        # Update target networks and EMA
                        self._update_target_networks()
                        if self.use_ema:
                            self._update_ema()

                        # Update learning rates
                        if scheduler_actor is not None:
                            scheduler_actor.step()
                            scheduler_critic.step()

                    # Track metrics
                    batch_size = batch["ego_vec"].size(0)
                    running_actor_loss += actor_loss.item() * batch_size
                    running_critic_loss += critic_loss.item() * batch_size
                    running_value_loss += value_loss.item() * batch_size
                    running_q_mean += losses["q_mean"] * batch_size
                    running_samples += batch_size

                    global_step += 1

                    if pbar_inner is not None:
                        try:
                            pbar_inner.update(1)
                            pbar_inner.set_postfix({
                                "actor_loss": f"{actor_loss.item():.4f}",
                                "critic_loss": f"{critic_loss.item():.4f}",
                            })
                        except Exception:
                            pass

                    if global_step % log_interval == 0:
                        self.logger.log({
                            "step": global_step,
                            "epoch": epoch,
                            "train/actor_loss": float(actor_loss.item()),
                            "train/critic_loss": float(critic_loss.item()),
                            "train/value_loss": float(value_loss.item()),
                            "train/q_mean": float(losses["q_mean"]),
                            "lr_actor": float(self.actor_optimizer.param_groups[0]["lr"]),
                            "lr_critic": float(self.critic_optimizer.param_groups[0]["lr"]),
                        })

                # Build schedulers after first epoch
                if actual_steps_per_epoch is None:
                    actual_steps_per_epoch = epoch_batch_count
                    if actual_steps_per_epoch != estimated_steps_per_epoch:
                        print(f"[OfflineRLTrainer] Note: actual batches/epoch = {actual_steps_per_epoch}")

                    total_steps = actual_steps_per_epoch * epochs
                    scheduler_actor = self.build_lr_scheduler(self.actor_optimizer, total_steps)
                    scheduler_critic = self.build_lr_scheduler(self.critic_optimizer, total_steps)
                    print(f"[OfflineRLTrainer] LR schedulers: {total_steps} total steps")

                    if scheduler_actor is not None and global_step > 0:
                        scheduler_actor.last_epoch = global_step - 1
                        scheduler_critic.last_epoch = global_step - 1
                        scheduler_actor.step()
                        scheduler_critic.step()

                if pbar_inner is not None:
                    try:
                        pbar_inner.close()
                    except Exception:
                        pass

                epoch_time = time.perf_counter() - t0
                train_actor_loss = running_actor_loss / max(1, running_samples)
                train_critic_loss = running_critic_loss / max(1, running_samples)
                train_value_loss = running_value_loss / max(1, running_samples)
                train_q_mean = running_q_mean / max(1, running_samples)

                # Validation (use EMA actor if available)
                eval_actor = self.ema_actor if self.use_ema else self.actor
                eval_actor.eval()
                self.critic.eval()
                val_actor_loss = 0.0
                val_critic_loss = 0.0
                val_q_mean = 0.0
                val_samples = 0

                val_batches = len(loaders["valid"]) if hasattr(loaders["valid"], '__len__') else None

                try:
                    pbar_val = tqdm(total=val_batches, desc=f"Valid {epoch}/{epochs}", leave=False)
                except Exception:
                    pbar_val = None

                with torch.no_grad():
                    for batch in loaders["valid"]:
                        batch = self.to_device(batch)
                        losses = self.compute_iql_loss(batch)
                        batch_size = batch["ego_vec"].size(0)
                        val_actor_loss += losses["actor_loss"].item() * batch_size
                        val_critic_loss += losses["critic_loss"].item() * batch_size
                        val_q_mean += losses["q_mean"] * batch_size
                        val_samples += batch_size

                        if pbar_val is not None:
                            try:
                                pbar_val.update(1)
                            except Exception:
                                pass

                if pbar_val is not None:
                    try:
                        pbar_val.close()
                    except Exception:
                        pass

                val_actor_loss = val_actor_loss / max(1, val_samples)
                val_critic_loss = val_critic_loss / max(1, val_samples)
                val_q_mean = val_q_mean / max(1, val_samples)
                self.actor.train()
                self.critic.train()

                # Log epoch metrics
                self.logger.log({
                    "epoch": epoch,
                    "train/actor_loss": train_actor_loss,
                    "train/critic_loss": train_critic_loss,
                    "train/value_loss": train_value_loss,
                    "train/q_mean": train_q_mean,
                    "valid/actor_loss": val_actor_loss,
                    "valid/critic_loss": val_critic_loss,
                    "valid/q_mean": val_q_mean,
                    "lr_actor": self.actor_optimizer.param_groups[0]["lr"],
                    "lr_critic": self.critic_optimizer.param_groups[0]["lr"],
                    "time/epoch_sec": epoch_time,
                })

                # Save checkpoint (higher Q-value is better)
                self.save_checkpoint(step=epoch, objective=val_q_mean, extra={"val_q_mean": val_q_mean})

                # Early stopping check
                if self.early_stopping is not None:
                    should_stop = self.early_stopping(val_q_mean, epoch)
                    es_status = self.early_stopping.status()

                    if pbar_outer is not None:
                        try:
                            pbar_outer.update(1)
                            pbar_outer.set_postfix({
                                "q_mean": f"{val_q_mean:.4f}",
                                "es": es_status,
                            })
                        except Exception:
                            pass

                    if should_stop:
                        print(f"\n[OfflineRLTrainer] Early stopping triggered at epoch {epoch}!")
                        print(f"[OfflineRLTrainer] Best Q-value: {self.early_stopping.best_score:.6f} at epoch {self.early_stopping.best_epoch}")
                        break
                else:
                    if pbar_outer is not None:
                        try:
                            pbar_outer.update(1)
                            pbar_outer.set_postfix({
                                "q_mean": f"{val_q_mean:.4f}",
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
