"""
PyTorch Dataset and DataLoader for BC trajectories with HDF5+LZ4 backend.

Optimized for high-throughput training with lazy loading and efficient random access.
"""

import os
import glob
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False
    h5py = None

from data.norms import (
    normalize_ego_vector,
    normalize_route_points,
    normalize_futures,
    normalize_object_tokens,
    normalize_bev,
)
from data.transforms import BCTransform


class BCTrajectoryDataset(Dataset):
    """
    PyTorch Dataset for BC trajectories from HDF5 episode-set files.
    
    Each sample returns normalized observations and labels:
        - ego_vec: (d_ego,) ego state vector
        - bev: (C, H, W) BEV tensor
        - route: (K, 2) route waypoints in ego frame
        - objects: (M, d_obj) object tokens
        - object_mask: (M,) attention mask for objects
        - future_xy: (N, 2) future waypoints
        - future_v: (N,) future speeds
        - future_mask: (N,) loss mask for futures
    """
    
    def __init__(
        self,
        run_dir: str,
        future_horizon: int = 12,
        route_points: int = 32,
        max_objects: int = 64,
        augment: bool = False,
        split: str = "all",     # "train", "val", "all"
        val_ratio: float = 0.2,
        use_bev_mmap: bool = False,  # Ignored, kept for API compatibility
    ):
        """
        Args:
            run_dir: Path to BC run directory (contains .h5 episode-set files)
            future_horizon: Number of future waypoints (N)
            route_points: Number of route points (K)
            max_objects: Maximum number of object tokens (M)
            augment: Whether to apply data augmentation
            split: Split type ("train", "val", "all")
            val_ratio: Fraction of data to use for validation (if split is not "all")
            use_bev_mmap: Ignored (kept for backward compatibility)
        """
        if not HDF5_AVAILABLE:
            raise ImportError("h5py is required for HDF5 dataset")
        
        self.run_dir = os.path.abspath(run_dir)
        self.future_horizon = future_horizon
        self.route_points = route_points
        self.max_objects = max_objects
        self.augment = augment
        self.transform = BCTransform(augment=augment)
        
        print(f"[BCTrajectoryDataset] Initializing HDF5-backed dataset from {run_dir}")
        
        # Find all HDF5 episode-set files
        h5_files = sorted(glob.glob(os.path.join(self.run_dir, "*.h5")))
        
        if not h5_files:
            raise FileNotFoundError(f"No HDF5 files found in {run_dir}")
        
        print(f"[BCTrajectoryDataset] Found {len(h5_files)} HDF5 episode-set files")
        
        # Build global index: map global_idx -> (file_idx, local_idx)
        self._file_paths = h5_files
        self._file_lengths = []
        self._file_cumsum = [0]
        
        for fpath in h5_files:
            with h5py.File(fpath, "r") as f:
                n_samples = len(f["frame_id"])
                self._file_lengths.append(n_samples)
                self._file_cumsum.append(self._file_cumsum[-1] + n_samples)
        
        self._total_samples = self._file_cumsum[-1]
        
        print(f"[BCTrajectoryDataset] Total samples across all files: {self._total_samples:,}")
        
        # Apply train/val split
        if split != "all":
            split_idx = int(self._total_samples * (1 - val_ratio))
            
            if split == "train":
                self._start_idx = 0
                self._end_idx = split_idx
                print(f"[BCTrajectoryDataset] Split 'train': Using samples [0, {split_idx})")
            elif split == "val":
                self._start_idx = split_idx
                self._end_idx = self._total_samples
                print(f"[BCTrajectoryDataset] Split 'val': Using samples [{split_idx}, {self._total_samples})")
        else:
            self._start_idx = 0
            self._end_idx = self._total_samples
        
        # File handles (opened lazily per worker)
        self._file_handles: Optional[List[h5py.File]] = None
        
        print(f"[BCTrajectoryDataset] Dataset ready: {len(self)} samples")
    
    def _ensure_files_open(self):
        """Lazily open HDF5 files (once per worker process)."""
        if self._file_handles is None:
            self._file_handles = []
            for fpath in self._file_paths:
                f = h5py.File(fpath, "r")
                self._file_handles.append(f)
    
    def _global_to_local(self, global_idx: int) -> Tuple[int, int]:
        """Convert global index to (file_idx, local_idx)."""
        # Binary search in cumsum
        file_idx = np.searchsorted(self._file_cumsum, global_idx, side="right") - 1
        local_idx = global_idx - self._file_cumsum[file_idx]
        return file_idx, local_idx
    
    def __len__(self) -> int:
        return self._end_idx - self._start_idx
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single normalized sample (lazy loaded from HDF5)."""
        # Adjust for split offset
        global_idx = self._start_idx + idx
        
        # Open files if not already open
        self._ensure_files_open()
        
        # Map to file and local index
        file_idx, local_idx = self._global_to_local(global_idx)
        f = self._file_handles[file_idx]
        
        # Read all data for this sample from HDF5
        # Note: HDF5 stores raw unnormalized data, so we normalize here
        frame_id = int(f["frame_id"][local_idx])
        ego_vec = torch.from_numpy(f["ego_vec"][local_idx].astype(np.float32))
        bev = torch.from_numpy(f["bev"][local_idx].astype(np.float32))
        route = torch.from_numpy(f["route"][local_idx].astype(np.float32))
        objects = torch.from_numpy(f["objects"][local_idx].astype(np.float32))
        object_mask = torch.from_numpy(f["object_mask"][local_idx].astype(np.float32))
        future_xy = torch.from_numpy(f["future_xy"][local_idx].astype(np.float32))
        future_v = torch.from_numpy(f["future_v"][local_idx].astype(np.float32))
        future_mask = torch.from_numpy(f["future_mask"][local_idx].astype(np.float32))
        
        # Normalize all features
        # Note: HDF5 stores raw data, apply normalization
        ego_vec = ego_vec  # Already normalized during collection
        bev = normalize_bev(bev)
        route = normalize_route_points(route)
        objects = normalize_object_tokens(objects)
        future_xy, future_v = normalize_futures(future_xy, future_v)
        
        sample = {
            "frame_id": frame_id,
            # Observations (normalized)
            "ego_vec": ego_vec,          # (d_ego,)
            "bev": bev,                  # (C, H, W)
            "route": route,              # (K, 2)
            "objects": objects,          # (M, d_obj)
            "object_mask": object_mask,  # (M,)
            # Labels (normalized)
            "future_xy": future_xy,      # (N, 2)
            "future_v": future_v,        # (N,)
            "future_mask": future_mask,  # (N,)
        }

        # Apply augmentation BEFORE returning
        sample = self.transform(sample)

        return sample
    
    def __del__(self):
        """Close file handles on cleanup."""
        if self._file_handles is not None:
            for f in self._file_handles:
                try:
                    f.close()
                except:
                    pass


def fast_collate_fn(batch):
    """
    Fast collate function using pre-allocated tensors.
    
    Avoids default_collate overhead by pre-allocating output tensors
    and directly copying data. This is faster than PyTorch's default
    collate which does multiple intermediate allocations.
    
    Args:
        batch: List of sample dicts from BCTrajectoryDataset
        
    Returns:
        Batched dict with all tensors stacked along batch dimension
    """
    if not batch:
        return {}
    
    B = len(batch)
    
    # Get dimensions from first sample
    first = batch[0]
    d_ego = first["ego_vec"].shape[0]
    C, H, W = first["bev"].shape
    K = first["route"].shape[0]
    M = first["objects"].shape[0]
    N = first["future_xy"].shape[0]
    
    # Pre-allocate output tensors
    ego_vec = torch.zeros((B, d_ego), dtype=torch.float32)
    bev = torch.zeros((B, C, H, W), dtype=torch.float32)
    route = torch.zeros((B, K, 2), dtype=torch.float32)
    objects = torch.zeros((B, M, first["objects"].shape[1]), dtype=torch.float32)
    object_mask = torch.zeros((B, M), dtype=torch.float32)
    future_xy = torch.zeros((B, N, 2), dtype=torch.float32)
    future_v = torch.zeros((B, N), dtype=torch.float32)
    future_mask = torch.zeros((B, N), dtype=torch.float32)
    
    # Fast copy loop
    for i, sample in enumerate(batch):
        ego_vec[i] = sample["ego_vec"]
        bev[i] = sample["bev"]
        route[i] = sample["route"]
        objects[i] = sample["objects"]
        object_mask[i] = sample["object_mask"]
        future_xy[i] = sample["future_xy"]
        future_v[i] = sample["future_v"]
        future_mask[i] = sample["future_mask"]
    
    return {
        "ego_vec": ego_vec,
        "bev": bev,
        "route": route,
        "objects": objects,
        "object_mask": object_mask,
        "future_xy": future_xy,
        "future_v": future_v,
        "future_mask": future_mask,
    }


def _worker_init_fn(worker_id):
    """
    Initialize worker for HDF5 file access.
    
    Each worker process opens its own HDF5 file handles to avoid
    thread-safety issues with h5py.
    
    Args:
        worker_id: Worker process ID (0 to num_workers-1)
    """
    import torch.utils.data
    
    # Get worker info and dataset
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset
        
        # Force lazy file opening in this worker
        # HDF5 files will be opened on first __getitem__ call
        dataset._file_handles = None
        
        print(f"[Worker {worker_id}] Initialized for HDF5 access")


def create_bc_dataloader(
    run_dir: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    prefetch_factor: int = 4,
    persistent_workers: bool = True,
    pin_memory: bool = True,
    drop_last: bool = False,
    future_horizon: int = 12,
    route_points: int = 32,
    max_objects: int = 64,
    use_bev_mmap: bool = False,  # Ignored
    augment: bool = False,
    split: str = "all",
    val_ratio: float = 0.2,
    **kwargs
) -> torch.utils.data.DataLoader:
    """
    Create an optimized PyTorch DataLoader for BC trajectories (HDF5 backend).
    
    Args:
        run_dir: Path to BC run directory
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes for data loading (default: 4)
        prefetch_factor: Batches to prefetch per worker (default: 4)
        persistent_workers: Keep workers alive between epochs (default: True)
        pin_memory: Pin memory for faster GPU transfer (default: True)
        drop_last: Drop last incomplete batch (default: False)
        future_horizon: Number of future waypoints (N)
        route_points: Number of route points (K)
        max_objects: Maximum number of object tokens (M)
        use_bev_mmap: Ignored (legacy parameter)
        augment: If True, apply data augmentation to the dataset.
        split: Dataset split ("train", "val", "all")
        val_ratio: Fraction of data to use for validation
        **kwargs: Additional arguments for DataLoader
    
    Returns:
        DataLoader instance with optimized settings
    """
    import time
    start_time = time.time()
    
    dataset = BCTrajectoryDataset(
        run_dir=run_dir,
        future_horizon=future_horizon,
        route_points=route_points,
        max_objects=max_objects,
        use_bev_mmap=use_bev_mmap,
        augment=augment,
        split=split,
        val_ratio=val_ratio,
    )
    
    init_time = time.time() - start_time
    print(f"[BCDataLoader] Dataset initialized in {init_time:.2f}s (HDF5 lazy loading)")
    print(f"[BCDataLoader] Creating DataLoader: {len(dataset)} samples, batch_size={batch_size}, num_workers={num_workers}")
    
    # Persistent workers requires num_workers > 0
    if num_workers == 0:
        persistent_workers = False
    
    dataloader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
        "persistent_workers": persistent_workers,
        "collate_fn": fast_collate_fn,
    }
    
    # prefetch_factor only valid when num_workers > 0
    if num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = prefetch_factor
        # Add worker init function for HDF5 file handle management
        dataloader_kwargs["worker_init_fn"] = _worker_init_fn
    
    # Merge any additional kwargs
    dataloader_kwargs.update(kwargs)
    
    return torch.utils.data.DataLoader(dataset, **dataloader_kwargs)
