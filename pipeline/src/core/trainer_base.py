from __future__ import annotations
import time
from pathlib import Path
from typing import Dict, Any, Iterable, Optional, Callable
import torch
from .speed import apply_speed_flags
from .seed import set_seed_everywhere
from .logger_router import make_logger
from .checkpoint import CheckpointManager

class BaseTrainer:
    required_components: list[str] = []

    def __init__(self, cfg, **components):
        self.cfg = cfg

        # resolve accelerator preference
        accel_pref = str(getattr(cfg.speed, "accelerator", "auto")).lower() if hasattr(cfg, "speed") else "auto"
        if accel_pref == "cpu":
            use_cuda = False
        elif accel_pref == "gpu":
            use_cuda = torch.cuda.is_available()
        else:  # auto
            use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.components: Dict[str, Any] = components
        apply_speed_flags(cfg.speed)

        # resolve seed from flat or nested config
        def _maybe_get(obj, name):
            try:
                return getattr(obj, name)
            except Exception:
                return None
        seed_node = _maybe_get(cfg, "seed") or _maybe_get(_maybe_get(cfg, "exp"), "seed")
        seed_value = getattr(seed_node, "seed", 42) if seed_node is not None else 42
        try:
            seed_int = int(seed_value)
        except Exception:
            seed_int = 42
        set_seed_everywhere(seed_int)
        self.logger = make_logger(cfg)

        # write under current Hydra run's reports directory
        from src.core.run_io import get_run_dir
        self.reports_dir = get_run_dir() / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # checkpoint manager writes to reports/checkpoints
        direction = str(getattr(getattr(cfg, "hpo", {}), "direction", "minimize")).lower()
        direction = "max" if direction.startswith("max") else "min"
        self.checkpoints = CheckpointManager(self.reports_dir, top_k=5, direction=("max" if direction=="max" else "min"))

    def to_device(self, batch: Any) -> Any:
        """Recursively move tensors in a nested structure to device."""
        if torch.is_tensor(batch):
            return batch.to(self.device, non_blocking=True)
        if isinstance(batch, dict):
            return {k: self.to_device(v) for k, v in batch.items()}
        if isinstance(batch, (list, tuple)):
            return type(batch)(self.to_device(v) for v in batch)
        return batch

    def clip_grad(self, params, max_norm: float | None):
        if max_norm and max_norm > 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm)

    def save_checkpoint(self, step: int, objective: float | None = None, extra: Dict[str, Any] | None = None):
        state = {"step": step}
        for name, obj in self.components.items():
            if hasattr(obj, "state_dict"):
                state[name] = obj.state_dict()
        if extra:
            state.update(extra)
        return self.checkpoints.save_step(step, state, objective=objective)

    # ===== Modular hooks and utilities =====
    def get_amp_settings(self) -> tuple[bool, Optional[torch.dtype], torch.amp.GradScaler]:
        prec = str(getattr(getattr(self.cfg, "speed", {}), "precision", "bf16")).lower()
        use_amp = torch.cuda.is_available() and prec in ("bf16", "fp16")
        amp_dtype = torch.bfloat16 if prec == "bf16" else (torch.float16 if prec == "fp16" else None)
        scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and prec == "fp16"))
        return use_amp, amp_dtype, scaler

    def build_lr_scheduler(self, optimizer: torch.optim.Optimizer, total_steps: int) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        import math
        
        tr_cfg = getattr(self.cfg, "trainer", None) or getattr(getattr(self.cfg, "exp", {}), "trainer", None)
        lr_cfg = getattr(tr_cfg, "lr_scheduler", None)

        if lr_cfg is None:
            return None

        name = str(getattr(lr_cfg, "name", "cosine")).lower()
        t_max = int(getattr(lr_cfg, "T_max", total_steps))
        warmup = int(getattr(lr_cfg, "warmup", 0))
        from torch.optim.lr_scheduler import LambdaLR

        def lr_lambda(step_idx: int):
            if warmup > 0 and step_idx < warmup:
                return float(step_idx + 1) / float(max(1, warmup))
            if name == "cosine":
                progress = (step_idx - warmup) / max(1, t_max - warmup)
                progress = max(0.0, min(1.0, progress))
                return 0.5 * (1.0 + math.cos(math.pi * progress))
            return 1.0

        return LambdaLR(optimizer, lr_lambda=lr_lambda)

    def apply_augmentations(self, batch: Any) -> Any:
        augmenter: Optional[Callable[[Any], Any]] = self.components.get("augmenter", None)
        if callable(augmenter):
            try:
                return augmenter(batch)
            except Exception:
                return batch
        return batch

    def apply_normalization(self, batch: Any) -> Any:
        normalizer: Optional[Callable[[Any], Any]] = self.components.get("normalizer", None)
        if callable(normalizer):
            try:
                return normalizer(batch)
            except Exception:
                return batch
        return batch

    def compute_regularization(self) -> torch.Tensor:
        regularizer: Optional[Callable[[], Any]] = self.components.get("regularizer", None)
        if callable(regularizer):
            try:
                val = regularizer()
                if torch.is_tensor(val):
                    return val
                try:
                    return torch.as_tensor(float(val), device=self.device)
                except Exception:
                    return torch.tensor(0.0, device=self.device)
            except Exception:
                return torch.tensor(0.0, device=self.device)
        return torch.tensor(0.0, device=self.device)

    def fit(self):
        """
        Subclasses:
         - build loaders/iterators from components
         - for step in range(max_steps): compute loss -> backward -> step
         - use logger.log() and save_checkpoint() as needed
        """
        raise NotImplementedError
