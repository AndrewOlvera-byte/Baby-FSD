from __future__ import annotations
from typing import Any, Dict, Tuple
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from src.core.registry import register


def _build_transforms(cfg_node) -> Tuple[transforms.Compose, transforms.Compose]:
    # CIFAR-10 statistics
    mean = getattr(cfg_node, "mean", [0.4914, 0.4822, 0.4465])
    std = getattr(cfg_node, "std", [0.2470, 0.2435, 0.2616])

    aug_list = []
    if getattr(cfg_node, "random_crop", True):
        aug_list.append(transforms.RandomCrop(32, padding=4))
    if getattr(cfg_node, "hflip", True):
        aug_list.append(transforms.RandomHorizontalFlip())
    if getattr(cfg_node, "autoaugment", False):
        aug_list.append(transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10))

    train_tf = transforms.Compose([
        *aug_list,
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    valid_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return train_tf, valid_tf


class _WrapCIFAR(Dataset):
    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        x, y = self.base[idx]
        return {"images": x, "targets": torch.tensor(y, dtype=torch.long)}


@register("dataset", "cifar10")
class CIFAR10Factory:
    def build(self, cfg_node, context: Dict[str, Any]):
        root = getattr(cfg_node, "root", "./data")
        # Resolve dataset root relative to the original working directory (not Hydra run dir)
        try:
            from hydra.utils import to_absolute_path
            if not Path(root).is_absolute():
                root = to_absolute_path(str(root))
        except Exception:
            # Fallback: make absolute relative to current CWD
            if not Path(root).is_absolute():
                root = str((Path.cwd() / str(root)).resolve())
        download = bool(getattr(cfg_node, "download", True))
        train_tf, valid_tf = _build_transforms(cfg_node)

        train_base = datasets.CIFAR10(root=root, train=True, transform=train_tf, download=download)
        test_base = datasets.CIFAR10(root=root, train=False, transform=valid_tf, download=download)

        # Optional split of train into train/valid
        val_ratio = float(getattr(cfg_node, "val_ratio", 0.05))
        if val_ratio > 0.0:
            n_total = len(train_base)
            n_val = int(n_total * val_ratio)
            n_train = n_total - n_val
            train_base, valid_base = torch.utils.data.random_split(
                train_base, [n_train, n_val], generator=torch.Generator().manual_seed(42)
            )
            # random_split returns Subset; wrap to apply the dict format
            train_ds = _WrapCIFAR(train_base)
            valid_ds = _WrapCIFAR(valid_base)
        else:
            train_ds = _WrapCIFAR(train_base)
            valid_ds = _WrapCIFAR(test_base)

        test_ds = _WrapCIFAR(test_base)
        return {"train": train_ds, "valid": valid_ds, "test": test_ds}


