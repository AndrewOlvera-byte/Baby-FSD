from __future__ import annotations
from torch.utils.data import DataLoader
from src.core.registry import register

@register("datacoll", "dataset_iter")
class DatasetIteratorFactory:
    def build(self, cfg_node, context) -> dict:
        """
        Wraps a torch Dataset into a DataLoader iterator.
        Expects context["dataset"] == {"train": Dataset, "valid": Dataset}
        cfg_node is optional for BC; we take batch_size from cfg.dataset.
        """
        cfg = context["cfg"]
        train_ds = context["dataset"]["train"]
        valid_ds = context["dataset"]["valid"]

        num_workers = int(getattr(cfg.dataset, "num_workers", 0))
        prefetch_factor = int(getattr(cfg.dataset, "prefetch_factor", 4)) if num_workers > 0 else None
        persistent_workers = bool(getattr(cfg.dataset, "persistent_workers", True)) if num_workers > 0 else False
        pin_memory = bool(getattr(cfg.dataset, "pin_memory", True))

        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.dataset.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )
        valid_loader = DataLoader(
            valid_ds,
            batch_size=cfg.dataset.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )
        return {"train_loader": train_loader, "valid_loader": valid_loader}
