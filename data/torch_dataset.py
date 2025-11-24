"""
PyTorch Dataset and DataLoader for BC trajectories with WebDataset backend.

Optimized for high-throughput training with batch-level data loading and efficient shard iteration.
"""

import os
import glob
import json
import io
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset

try:
    import webdataset as wds
    WEBDATASET_AVAILABLE = True
except ImportError:
    WEBDATASET_AVAILABLE = False
    wds = None

from data.norms import (
    normalize_route_points,
    normalize_futures,
    normalize_object_tokens,
    normalize_bev,
)
from data.transforms import BCTransform

# Required fields per sample
REQUIRED_KEYS = [
    "frame_id",
    "episode_id",
    "ego_vec",
    "bev",
    "route",
    "objects",
    "object_mask",
    "future_xy",
    "future_v",
    "future_mask",
]


def _load_metadata(run_dir: str) -> Dict:
    """Load metadata.json from WebDataset directory."""
    metadata_path = os.path.join(run_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"metadata.json not found in {run_dir}")
    
    with open(metadata_path, "r") as f:
        return json.load(f)


def _has_all_required_keys(sample_dict: Dict[str, object]) -> bool:
    """Return True if the sample dict contains all required .npy fields."""
    if not hasattr(sample_dict, "keys"):
        return False
    keys = sample_dict.keys()
    for field in REQUIRED_KEYS:
        if not any(k.endswith(f".{field}.npy") or k == f"{field}.npy" for k in keys):
            return False
    return True


def _parse_sample(sample_dict: Dict) -> Dict[str, np.ndarray]:
    """
    Parse a WebDataset sample dict into numpy arrays.
    
    WebDataset groups files by base name (before first dot).
    Files like "00000000.frame_id.npy" and "00000000.bev.npy" are grouped together.
    The dict keys are the full filenames like "00000000.frame_id.npy".
    
    Args:
        sample_dict: Dict from WebDataset with keys like "00000000.frame_id.npy", etc.
        
    Returns:
        Dict with parsed numpy arrays for a single sample
    """
    sample_data = {}
    
    for key, value in sample_dict.items():
        if not key.endswith(".npy"):
            continue
        
        # Extract field name from filename
        # Format: "{idx:08d}.{field}.npy" or just "{field}.npy"
        parts = key.rsplit(".", 2)  # Split from right: [idx, field, "npy"] or [field, "npy"]
        if len(parts) == 3:
            # Format: "{idx}.{field}.npy"
            field_name = parts[1]
        elif len(parts) == 2:
            # Format: "{field}.npy"
            field_name = parts[0]
        else:
            continue
        
        # Parse numpy array from bytes
        if isinstance(value, bytes):
            arr = np.load(io.BytesIO(value), allow_pickle=False)
        elif isinstance(value, io.BytesIO):
            arr = np.load(value, allow_pickle=False)
        else:
            # Already a numpy array
            arr = value
        
        sample_data[field_name] = arr
    
    # Ensure all required keys are present
    for key in REQUIRED_KEYS:
        if key not in sample_data:
            raise ValueError(f"Missing required key '{key}' in sample. Available keys: {list(sample_data.keys())}")
    
    return sample_data


def _normalize_sample(sample: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    """
    Normalize a sample and convert to torch tensors.
    
    Args:
        sample: Dict with numpy arrays
        
    Returns:
        Dict with normalized torch tensors
    """
    # Convert to torch tensors
    frame_id = int(sample["frame_id"])
    ego_vec = torch.from_numpy(sample["ego_vec"].astype(np.float32))
    bev = torch.from_numpy(sample["bev"].astype(np.float32))
    route = torch.from_numpy(sample["route"].astype(np.float32))
    objects = torch.from_numpy(sample["objects"].astype(np.float32))
    object_mask = torch.from_numpy(sample["object_mask"].astype(np.float32))
    future_xy = torch.from_numpy(sample["future_xy"].astype(np.float32))
    future_v = torch.from_numpy(sample["future_v"].astype(np.float32))
    future_mask = torch.from_numpy(sample["future_mask"].astype(np.float32))
    
    # Normalize (ego_vec is already normalized during collection)
    bev = normalize_bev(bev)
    route = normalize_route_points(route)
    objects = normalize_object_tokens(objects)
    future_xy, future_v = normalize_futures(future_xy, future_v)
    
    return {
        "frame_id": frame_id,
        "ego_vec": ego_vec,
        "bev": bev,
        "route": route,
        "objects": objects,
        "object_mask": object_mask,
        "future_xy": future_xy,
        "future_v": future_v,
        "future_mask": future_mask,
    }


class BCWebDataset(IterableDataset):
    """
    WebDataset-based PyTorch Dataset for BC trajectories.
    
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
        shuffle_buffer_size: int = 1000,
        batch_size: Optional[int] = None,  # If None, don't batch in WebDataset
    ):
        """
        Args:
            run_dir: Path to WebDataset directory (contains shard-*.tar and metadata.json)
            future_horizon: Number of future waypoints (N)
            route_points: Number of route points (K)
            max_objects: Maximum number of object tokens (M)
            augment: Whether to apply data augmentation
            split: Split type ("train", "val", "all")
            val_ratio: Fraction of data to use for validation (if split is not "all")
            use_bev_mmap: Ignored (kept for backward compatibility)
            shuffle_buffer_size: Buffer size for WebDataset shuffling
            batch_size: Batch size for WebDataset batching (None = no batching in WebDataset)
        """
        if not WEBDATASET_AVAILABLE:
            raise ImportError("webdataset is required. Install with: pip install webdataset")
        
        self.run_dir = os.path.abspath(run_dir)
        self.future_horizon = future_horizon
        self.route_points = route_points
        self.max_objects = max_objects
        self.augment = augment
        self.transform = BCTransform(augment=augment)
        self.split = split
        self.val_ratio = val_ratio
        self.shuffle_buffer_size = shuffle_buffer_size
        self.batch_size = batch_size
        
        print(f"[BCWebDataset] Initializing WebDataset from {run_dir}")
        
        # Load metadata
        metadata = _load_metadata(self.run_dir)
        self.metadata = metadata
        self.total_samples = metadata.get("total_samples", 0)
        
        # Validate metadata dimensions match config
        K = metadata.get("K", route_points)
        N = metadata.get("N_future", future_horizon)
        M = metadata.get("M", max_objects)
        
        if K != route_points:
            print(f"[BCWebDataset] WARNING: metadata K={K} != config route_points={route_points}, using metadata value")
            self.route_points = K
        if N != future_horizon:
            print(f"[BCWebDataset] WARNING: metadata N_future={N} != config future_horizon={future_horizon}, using metadata value")
            self.future_horizon = N
        if M != max_objects:
            print(f"[BCWebDataset] WARNING: metadata M={M} != config max_objects={max_objects}, using metadata value")
            self.max_objects = M
        
        print(f"[BCWebDataset] Total samples: {self.total_samples:,}")
        print(f"[BCWebDataset] Dimensions: K={K}, N={N}, M={M}, C={metadata.get('C', 18)}, H={metadata.get('H', 150)}, W={metadata.get('W', 200)}")
        
        # Find all shard files
        shard_pattern = os.path.join(self.run_dir, "shard-*.tar")
        shard_files = sorted(glob.glob(shard_pattern))
        
        if not shard_files:
            raise FileNotFoundError(f"No shard files found matching {shard_pattern}")
        
        print(f"[BCWebDataset] Found {len(shard_files)} shard files")
        
        # Apply train/val split at shard level
        if split != "all":
            split_idx = int(len(shard_files) * (1 - val_ratio))
            
            if split == "train":
                self.shard_files = shard_files[:split_idx]
                print(f"[BCWebDataset] Split 'train': Using {len(self.shard_files)} shards")
            elif split == "val":
                self.shard_files = shard_files[split_idx:]
                print(f"[BCWebDataset] Split 'val': Using {len(self.shard_files)} shards")
        else:
            self.shard_files = shard_files
        
        # Build WebDataset URLs (file:// protocol for local files)
        self.urls = [f"file://{os.path.abspath(f)}" for f in self.shard_files]
        
        print(f"[BCWebDataset] Dataset ready: {len(self.urls)} shards")
    
    def __iter__(self):
        """Create WebDataset iterator with proper batching and epoch handling.
        
        Follows WebDataset best practices for multi-worker DataLoader:
        1. nodesplitter ensures each worker processes unique shards (no overlap)
        2. with_epoch() ensures proper epoch boundaries across workers
        3. Batching on WebDataset side for efficiency
        """
        # Create WebDataset pipeline
        # Set shardshuffle explicitly: positive integer for train, 0 for val
        if self.split == "train" or self.split == "all":
            # Use number of shards as shuffle buffer for train
            shardshuffle = len(self.urls)
        else:
            # No shard shuffling for validation
            shardshuffle = 0
        
        # Prefer built-in nodesplitter to avoid missing attribute errors on older WebDataset
        nodesplitter = getattr(wds, "split_by_worker", None)
        if nodesplitter is None and hasattr(wds, "utils"):
            # Older versions expose split_by_worker under webdataset.utils
            nodesplitter = getattr(wds.utils, "split_by_worker", None)
        if nodesplitter is None:
            print("[BCWebDataset] WARNING: webdataset.split_by_worker unavailable; workers may duplicate shards")
        
        dataset = wds.WebDataset(
            self.urls,
            shardshuffle=shardshuffle,
            nodesplitter=nodesplitter,  # ensures unique shards per worker when available
            empty_check=False,  # allow fewer shards than workers without raising
        )
        
        # Shuffle samples within shards (if training)
        if self.split == "train" or self.split == "all":
            dataset = dataset.shuffle(self.shuffle_buffer_size)
        
        # Decode files (WebDataset will handle basic decoding)
        # We'll parse numpy arrays from bytes in _parse_sample
        dataset = dataset.decode()

        # Skip non-sample entries like __metadata__.json that don't carry npy fields
        dataset = dataset.select(_has_all_required_keys)
        
        # Map samples to our format
        dataset = dataset.map(_parse_sample)
        dataset = dataset.map(_normalize_sample)
        
        # Apply transforms
        if self.augment:
            dataset = dataset.map(self.transform)
        
        # Batch on WebDataset side (best practice for efficiency)
        # This allows each worker to process batches independently
        if self.batch_size is not None:
            dataset = dataset.batched(self.batch_size)
        
        # CRITICAL: with_epoch ensures proper epoch boundaries with multi-worker DataLoader.
        # Older WebDataset versions expect an int, so use the approximate split length.
        epoch_length = max(1, len(self))
        dataset = dataset.with_epoch(epoch_length)
        
        return iter(dataset)
    
    def __len__(self):
        """Approximate length (based on metadata and shard split)."""
        if self.split == "all":
            return self.total_samples
        elif self.split == "train":
            return int(self.total_samples * (1 - self.val_ratio))
        else:  # val
            return int(self.total_samples * self.val_ratio)


def fast_collate_fn(batch):
    """
    Fast collate function using pre-allocated tensors.
    
    Handles batches from WebDataset in multiple formats:
    1. Dict with lists (WebDataset .batched() format): {"bev": [tensor1, tensor2, ...], ...}
    2. List of sample dicts: [{"bev": tensor1, ...}, {"bev": tensor2, ...}, ...]
    3. Single dict with already-batched tensors (4D BEV)
    
    Args:
        batch: Batch from WebDataset (format depends on WebDataset version and batching method)
        
    Returns:
        Batched dict with all tensors stacked along batch dimension
    """
    # Handle empty batch
    if not batch:
        return {}
    
    # Check if already batched (4D tensor in dict)
    if isinstance(batch, dict):
        bev = batch.get("bev", None)
        if torch.is_tensor(bev) and bev.ndim == 4:
            # Already properly batched, return as-is
            return batch
        
        # Check if WebDataset batched format: dict with lists
        # WebDataset .batched() can return {"bev": [t1, t2, ...], "route": [r1, r2, ...], ...}
        if isinstance(bev, list) and len(bev) > 0:
            # WebDataset batched format: dict with lists
            B = len(bev)
            ego_vec_list = batch.get("ego_vec", [])
            route_list = batch.get("route", [])
            objects_list = batch.get("objects", [])
            object_mask_list = batch.get("object_mask", [])
            future_xy_list = batch.get("future_xy", [])
            future_v_list = batch.get("future_v", [])
            future_mask_list = batch.get("future_mask", [])
            
            if not all([ego_vec_list, route_list, objects_list, object_mask_list, future_xy_list, future_v_list, future_mask_list]):
                raise ValueError("Missing required keys in batched dict")

            def _to_tensor_list(items):
                out = []
                for t in items:
                    if not torch.is_tensor(t):
                        t = torch.as_tensor(t, dtype=torch.float32)
                    out.append(t)
                return out

            bev_list = []
            for t in bev:
                if not torch.is_tensor(t):
                    t = torch.as_tensor(t, dtype=torch.float32)
                if t.ndim == 4 and t.shape[0] == 1:
                    t = t[0]
                bev_list.append(t)
            bev_out = torch.stack(bev_list, dim=0)

            ego_vec = torch.stack(_to_tensor_list(ego_vec_list), dim=0)
            route = torch.stack(_to_tensor_list(route_list), dim=0)
            objects = torch.stack(_to_tensor_list(objects_list), dim=0)
            object_mask = torch.stack(_to_tensor_list(object_mask_list), dim=0)
            future_xy = torch.stack(_to_tensor_list(future_xy_list), dim=0)
            future_v = torch.stack(_to_tensor_list(future_v_list), dim=0)
            future_mask = torch.stack(_to_tensor_list(future_mask_list), dim=0)
            
            return {
                "ego_vec": ego_vec,
                "bev": bev_out,
                "route": route,
                "objects": objects,
                "object_mask": object_mask,
                "future_xy": future_xy,
                "future_v": future_v,
                "future_mask": future_mask,
            }
        
        # Single dict wrapped in list (fallback)
        batch = [batch]
    
    # Handle list of sample dicts (standard format)
    # If we received a single already-batched dict inside a list, return it.
    if len(batch) == 1 and isinstance(batch[0], dict):
        bev = batch[0].get("bev", None)
        if torch.is_tensor(bev) and bev.ndim == 4:
            return batch[0]
    
    # Batch is a list of sample dicts from WebDataset batching
    B = len(batch)
    
    # Get dimensions from first sample
    first = batch[0]
    d_ego = first["ego_vec"].shape[0]
    bev_shape = first["bev"].shape
    if len(bev_shape) == 3:
        C, H, W = bev_shape
    elif len(bev_shape) == 4:
        # Fallback: sample already has a leading batch dim; use spatial dims
        _, C, H, W = bev_shape
    else:
        raise ValueError(f"Unexpected BEV shape: {bev_shape}")
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
        bev_tensor = sample["bev"]
        if bev_tensor.ndim == 4 and bev_tensor.shape[0] == 1:
            bev[i] = bev_tensor[0]
        else:
            bev[i] = bev_tensor
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
    Create an optimized PyTorch DataLoader for BC trajectories (WebDataset backend).
    
    Args:
        run_dir: Path to WebDataset directory (contains shard-*.tar and metadata.json)
        batch_size: Batch size
        shuffle: Whether to shuffle data (handled by WebDataset)
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
    
    # Create dataset with batching on WebDataset side (best practice)
    dataset = BCWebDataset(
        run_dir=run_dir,
        future_horizon=future_horizon,
        route_points=route_points,
        max_objects=max_objects,
        use_bev_mmap=use_bev_mmap,
        augment=augment,
        split=split,
        val_ratio=val_ratio,
        batch_size=batch_size,  # WebDataset handles batching
    )
    
    # Clamp workers to shard count to avoid empty-check issues when shards < workers
    if num_workers > 0:
        max_workers = max(1, len(getattr(dataset, "shard_files", [])))
        if num_workers > max_workers:
            print(f"[BCDataLoader] Reducing num_workers from {num_workers} to {max_workers} (shards={max_workers})")
            num_workers = max_workers
    
    init_time = time.time() - start_time
    print(f"[BCDataLoader] Dataset initialized in {init_time:.2f}s (WebDataset)")
    print(f"[BCDataLoader] Creating DataLoader: batch_size={batch_size} (handled by WebDataset), num_workers={num_workers}")
    
    # Persistent workers requires num_workers > 0
    if num_workers == 0:
        persistent_workers = False
        prefetch_factor = None  # prefetch_factor only valid when num_workers > 0
    
    # Optimize prefetch_factor for GPU feeding (like CIFAR implementation)
    # Higher prefetch_factor = more batches ready, but more memory
    # For GPU training, we want data ready when GPU needs it
    # Default 4 is good balance: keeps GPU fed without excessive memory usage
    # Can reduce to 2 if OOM occurs, or increase to 6-8 if memory allows and GPU is starved
    
    # When WebDataset batches, set DataLoader batch_size=None (best practice)
    # The collate_fn still needed to convert list of dicts to batched tensors
    # Since we always pass batch_size to BCWebDataset, WebDataset always batches
    dataloader_kwargs = {
        "batch_size": None,  # WebDataset handles batching, so DataLoader doesn't batch again
        "shuffle": False,  # WebDataset handles shuffling internally
        "num_workers": num_workers,
        "pin_memory": pin_memory,  # Critical for GPU: enables async CPU->GPU transfer
        "drop_last": drop_last,
        "persistent_workers": persistent_workers,  # Keep workers alive between epochs (faster)
        "collate_fn": fast_collate_fn,  # Still needed to stack tensors from WebDataset batches
    }
    
    # prefetch_factor only valid when num_workers > 0
    if num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = prefetch_factor
    
    # Merge any additional kwargs
    dataloader_kwargs.update(kwargs)
    
    return torch.utils.data.DataLoader(dataset, **dataloader_kwargs)
