"""
PyTorch Dataset and DataLoader for BC trajectories.

Loads multi-table Parquet data and provides batched samples for training.
Includes normalization, object tokens, and future masking.

OPTIMIZED: Uses lazy loading to avoid loading entire dataset into memory at init.
"""

import io
import os
import zlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

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
    PyTorch Dataset for BC trajectories from Parquet files.
    
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
        use_bev_mmap: bool = True,
        augment: bool = False,
        split: str = "all",     # "train", "val", "all"
        val_ratio: float = 0.2,
    ):
        """
        Args:
            run_dir: Path to BC run directory (contains frames/, bev_frames/, etc.)
            future_horizon: Number of future waypoints (N)
            route_points: Number of route points (K)
            max_objects: Maximum number of object tokens (M)
            use_bev_mmap: If True, use pre-computed memory-mapped BEV file for fast loading
            augment: Whether to apply data augmentation
            split: Split type ("train", "val", "all")
            val_ratio: Fraction of data to use for validation (if split is not "all")
        """
        self.run_dir = os.path.abspath(run_dir)
        self.future_horizon = future_horizon
        self.route_points = route_points
        self.max_objects = max_objects
        self.augment = augment
        self.transform = BCTransform(augment=augment)
        
        # Paths to data tables
        frames_path = os.path.join(self.run_dir, "frames")
        futures_path = os.path.join(self.run_dir, "futures")
        route_path = os.path.join(self.run_dir, "route_points")
        bev_path = os.path.join(self.run_dir, "bev_frames")
        objects_path = os.path.join(self.run_dir, "object_tokens")
        
        # Check required tables exist
        required_paths = [frames_path, futures_path, route_path, bev_path]
        if not all(os.path.isdir(p) for p in required_paths):
            raise FileNotFoundError(f"Missing required tables in {run_dir}")
        
        # LAZY LOADING: Keep datasets as PyArrow datasets (don't convert to pandas)
        print(f"[BCTrajectoryDataset] Initializing lazy-loading dataset from {run_dir}")
        
        # Check for cached index file
        index_cache_path = os.path.join(self.run_dir, ".frame_index_cache.npy")
        
        if os.path.exists(index_cache_path):
            # Load pre-computed frame IDs (instant)
            print("[BCTrajectoryDataset] Loading cached frame index...")
            self._frame_ids = np.load(index_cache_path).tolist()
            print(f"[BCTrajectoryDataset] Loaded {len(self._frame_ids)} frames from cache (instant)")
        else:
            # Build frame index with parallel scanning
            print("[BCTrajectoryDataset] Building frame index (first time only)...")
            import multiprocessing as mp
            n_cores = min(8, mp.cpu_count())
            
            # Use PyArrow dataset with parallel fragment scanning
            self._frame_ids = self._build_frame_index_parallel(frames_path, n_cores)
            
            # Cache the frame IDs for next time
            print(f"[BCTrajectoryDataset] Caching frame index to {index_cache_path}")
            np.save(index_cache_path, np.array(self._frame_ids, dtype=np.int64))
            print(f"[BCTrajectoryDataset] Found {len(self._frame_ids)} frames (cached for future runs)")
        
        # Apply train/val split
        if split != "all":
            n_frames = len(self._frame_ids)
            split_idx = int(n_frames * (1 - val_ratio))
            
            if split == "train":
                self._frame_ids = self._frame_ids[:split_idx]
                print(f"[BCTrajectoryDataset] Split 'train': Using first {len(self._frame_ids)}/{n_frames} frames")
            elif split == "val":
                self._frame_ids = self._frame_ids[split_idx:]
                print(f"[BCTrajectoryDataset] Split 'val': Using last {len(self._frame_ids)}/{n_frames} frames")
        
        # Check for pre-computed BEV memory-map (Phase 2 optimization)
        mmap_path = os.path.join(self.run_dir, "bevs.mmap")
        bev_index_path = os.path.join(self.run_dir, "frame_id_index.npy")
        
        if use_bev_mmap and os.path.exists(mmap_path) and os.path.exists(bev_index_path):
            print("[BCTrajectoryDataset] ✓ Using pre-computed BEV memory-map (100-1000× faster loading)")
            self._use_bev_mmap = True
            
            # Load BEV frame ID index
            bev_frame_ids = np.load(bev_index_path)
            self._bev_frame_id_to_idx = {int(fid): idx for idx, fid in enumerate(bev_frame_ids)}
            
            # Open memory-mapped file in read-only mode
            # Shape is inferred from file size (N, 18, 150, 200)
            total_elements = os.path.getsize(mmap_path) // 4  # 4 bytes per float32
            N = total_elements // (18 * 150 * 200)
            self._bev_mmap = np.memmap(mmap_path, dtype='float32', mode='r', shape=(N, 18, 150, 200))
            print(f"[BCTrajectoryDataset]   BEV mmap shape: {self._bev_mmap.shape}, size: {os.path.getsize(mmap_path) / 1e9:.2f} GB")
            
            # Don't load BEV dataset from parquet
            self._bev_dataset = None
        else:
            self._use_bev_mmap = False
            self._bev_mmap = None
            self._bev_frame_id_to_idx = {}
            
            if use_bev_mmap:
                print("[BCTrajectoryDataset] ⚠ BEV memory-map not found, using parquet (slower)")
                print(f"[BCTrajectoryDataset]   Run 'python tools/preprocess_bev_mmap.py {self.run_dir}' to generate mmap")
            
            # Load BEV dataset from parquet
            self._bev_dataset = ds.dataset(bev_path, format="parquet")
        
        # Now load other datasets (this is fast, just creates references)
        self._frames_dataset = ds.dataset(frames_path, format="parquet")
        self._futures_dataset = ds.dataset(futures_path, format="parquet")
        self._route_dataset = ds.dataset(route_path, format="parquet")
        
        # Check if objects exist
        self._has_objects = os.path.isdir(objects_path)
        if self._has_objects:
            self._objects_dataset = ds.dataset(objects_path, format="parquet")
        
        # Cache for frequently accessed data (optional, can help with repeated access)
        self._cache_size = 1000
        self._frames_cache = {}
        self._bev_cache = {}
        self._route_cache = {}
        self._futures_cache = {}
        self._objects_cache = {}
        
        # Phase 2.2 optimization: Pre-build frame ID set for faster existence checks
        self._frame_id_set = set(self._frame_ids)
        print(f"[BCTrajectoryDataset] Frame ID index ready: {len(self._frame_id_set):,} unique frames")
    
    def _build_frame_index_parallel(self, frames_path: str, n_cores: int) -> List[int]:
        """
        Build frame index using parallel processing.
        Scans Parquet files in parallel to extract frame_ids quickly.
        Uses ThreadPoolExecutor for I/O-bound Parquet reading (faster than ProcessPool).
        """
        import glob
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # Get all parquet files
        parquet_files = sorted(glob.glob(os.path.join(frames_path, "*.parquet")))
        print(f"[BCTrajectoryDataset] Scanning {len(parquet_files)} Parquet files using {n_cores} threads...")
        
        def read_frame_ids(file_path):
            """Read frame_ids from a single parquet file."""
            try:
                table = pq.read_table(file_path, columns=["frame_id"])
                return table["frame_id"].to_pylist()
            except Exception as e:
                print(f"Warning: Failed to read {file_path}: {e}")
                return []
        
        # Parallel processing with threads (better for I/O)
        all_frame_ids = []
        with ThreadPoolExecutor(max_workers=n_cores) as executor:
            futures = {executor.submit(read_frame_ids, f): f for f in parquet_files}
            
            completed = 0
            for future in as_completed(futures):
                frame_ids = future.result()
                all_frame_ids.extend(frame_ids)
                completed += 1
                if completed % 1000 == 0:
                    print(f"  Processed {completed}/{len(parquet_files)} files...")
        
        # Sort and return
        print("[BCTrajectoryDataset] Sorting frame IDs...")
        return sorted(all_frame_ids)
    
    def __len__(self) -> int:
        return len(self._frame_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single normalized sample (lazy loaded)."""
        frame_id = self._frame_ids[idx]
        
        # Get frame data (ego state) - lazy load with caching
        if frame_id not in self._frames_cache:
            frame_filter = pc.field("frame_id") == frame_id
            frame_table = self._frames_dataset.to_table(filter=frame_filter)
            if len(frame_table) == 0:
                raise ValueError(f"Frame {frame_id} not found in frames table")
            # Store as dict for normalize_ego_vector compatibility
            row_dict = {col: frame_table[col][0].as_py() for col in frame_table.column_names}
            self._frames_cache[frame_id] = row_dict
            # Simple LRU: clear cache if too large
            if len(self._frames_cache) > self._cache_size:
                # Remove oldest 20% of entries
                to_remove = list(self._frames_cache.keys())[: self._cache_size // 5]
                for k in to_remove:
                    del self._frames_cache[k]
        
        row = self._frames_cache[frame_id]
        
        # Ego vector (normalized)
        ego_vec = self._normalize_ego_from_dict(row)
        
        # BEV tensor (normalized) - lazy load
        bev = self._get_bev_lazy(frame_id)
        bev = normalize_bev(bev)
        
        # Route points (normalized) - lazy load
        route = self._get_route_lazy(frame_id)
        route = normalize_route_points(route)
        
        # Object tokens (normalized) + mask - lazy load
        objects, object_mask = self._get_objects_lazy(frame_id)
        objects = normalize_object_tokens(objects)
        
        # Future waypoints and speeds (normalized) + mask - lazy load
        futures_xy, futures_speed, future_mask = self._get_futures_lazy(frame_id)
        futures_xy, futures_speed = normalize_futures(futures_xy, futures_speed)
        
        sample = {
            "frame_id": frame_id,
            # Observations (normalized)
            "ego_vec": ego_vec,          # (d_ego,)
            "bev": bev,                  # (C, H, W)
            "route": route,              # (K, 2)
            "objects": objects,          # (M, d_obj)
            "object_mask": object_mask,  # (M,)
            # Labels (normalized)
            "future_xy": futures_xy,     # (N, 2)
            "future_v": futures_speed,   # (N,)
            "future_mask": future_mask,  # (N,)
        }

        # Apply augmentation BEFORE returning
        sample = self.transform(sample)

        return sample
    
    def _normalize_ego_from_dict(self, row_dict: Dict) -> torch.Tensor:
        """
        Normalize ego vector from dict (compatible with normalize_ego_vector).
        Creates a simple namespace object to match pandas row interface.
        """
        from types import SimpleNamespace
        row = SimpleNamespace(**row_dict)
        return normalize_ego_vector(row)
    
    def _get_bev_lazy(self, frame_id: int) -> torch.Tensor:
        """Lazy load BEV tensor for a single frame."""
        if frame_id in self._bev_cache:
            return self._bev_cache[frame_id]
        
        if self._use_bev_mmap:
            # Fast path: Instant lookup from memory-mapped file (100-1000× faster)
            idx = self._bev_frame_id_to_idx.get(frame_id)
            if idx is not None:
                # Direct array access from mmap (no decompression needed)
                bev = torch.from_numpy(self._bev_mmap[idx].copy())
            else:
                # Frame not found in mmap, return zeros
                bev = torch.zeros((18, 150, 200), dtype=torch.float32)
        else:
            # Slow path: Load from parquet and decompress (legacy support)
            bev_filter = pc.field("frame_id") == frame_id
            bev_table = self._bev_dataset.to_table(filter=bev_filter, columns=["frame_id", "data"])
            
            if len(bev_table) == 0:
                # Return empty BEV if missing
                bev = torch.zeros((18, 150, 200), dtype=torch.float32)
            else:
                blob = bev_table["data"][0].as_py()
                raw = zlib.decompress(blob)
                arr = np.load(io.BytesIO(raw), allow_pickle=False)
                bev = torch.from_numpy(arr.astype(np.float32))
        
        # Cache it
        self._bev_cache[frame_id] = bev
        if len(self._bev_cache) > self._cache_size:
            to_remove = list(self._bev_cache.keys())[: self._cache_size // 5]
            for k in to_remove:
                del self._bev_cache[k]
        
        return bev
    
    def _get_route_lazy(self, frame_id: int) -> torch.Tensor:
        """Lazy load route points for a single frame."""
        if frame_id in self._route_cache:
            return self._route_cache[frame_id]
        
        # Filter and load only this frame's route
        route_filter = pc.field("frame_id") == frame_id
        route_table = self._route_dataset.to_table(filter=route_filter)
        
        # Convert to list and sort by idx
        route_list = []
        if len(route_table) > 0:
            df = route_table.to_pandas().sort_values("idx")
            route_list = [(float(row.x_ego), float(row.y_ego)) for row in df.itertuples(index=False)]
        
        # Pad or truncate to K points
        route = np.zeros((self.route_points, 2), dtype=np.float32)
        for i, (x, y) in enumerate(route_list[:self.route_points]):
            route[i] = [x, y]
        
        route_tensor = torch.from_numpy(route)
        
        # Cache it
        self._route_cache[frame_id] = route_tensor
        if len(self._route_cache) > self._cache_size:
            to_remove = list(self._route_cache.keys())[: self._cache_size // 5]
            for k in to_remove:
                del self._route_cache[k]
        
        return route_tensor
    
    def _get_futures_lazy(self, frame_id: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Lazy load future waypoints and speeds for a single frame."""
        if frame_id in self._futures_cache:
            return self._futures_cache[frame_id]
        
        # Filter and load only this frame's futures
        futures_filter = pc.field("frame_id") == frame_id
        futures_table = self._futures_dataset.to_table(filter=futures_filter)
        
        # Convert to list and sort by i
        futures_list = []
        if len(futures_table) > 0:
            df = futures_table.to_pandas().sort_values("i")
            futures_list = [
                (float(row.x_ego), float(row.y_ego), float(row.v_mps))
                for row in df.itertuples(index=False)
            ]
        
        # Pad or truncate to N steps
        futures_xy = np.zeros((self.future_horizon, 2), dtype=np.float32)
        futures_speed = np.zeros((self.future_horizon,), dtype=np.float32)
        future_mask = np.zeros((self.future_horizon,), dtype=np.float32)
        
        # Fill valid futures
        n_futures = min(len(futures_list), self.future_horizon)
        for i in range(n_futures):
            x, y, v = futures_list[i]
            futures_xy[i] = [x, y]
            futures_speed[i] = v
            future_mask[i] = 1.0  # Valid future
        
        result = (
            torch.from_numpy(futures_xy),
            torch.from_numpy(futures_speed),
            torch.from_numpy(future_mask),
        )
        
        # Cache it
        self._futures_cache[frame_id] = result
        if len(self._futures_cache) > self._cache_size:
            to_remove = list(self._futures_cache.keys())[: self._cache_size // 5]
            for k in to_remove:
                del self._futures_cache[k]
        
        return result
    
    def _get_objects_lazy(self, frame_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Lazy load object tokens for a single frame."""
        if frame_id in self._objects_cache:
            return self._objects_cache[frame_id]
        
        objects_list = []
        if self._has_objects:
            # Filter and load only this frame's objects
            objects_filter = pc.field("frame_id") == frame_id
            objects_table = self._objects_dataset.to_table(filter=objects_filter)
            
            # Convert to list and sort by idx
            if len(objects_table) > 0:
                df = objects_table.to_pandas().sort_values("idx")
                for row in df.itertuples(index=False):
                    obj = {
                        "type_id": int(row.type_id),
                        "x_ego": float(row.x_ego),
                        "y_ego": float(row.y_ego),
                        "sin_yaw": float(row.sin_yaw),
                        "cos_yaw": float(row.cos_yaw),
                        "length": float(row.length),
                        "width": float(row.width),
                        "vx": float(row.vx),
                        "vy": float(row.vy),
                        "oncoming_flag": int(row.oncoming_flag),
                        "priority_flag": int(row.priority_flag),
                    }
                    objects_list.append(obj)
        
        # Initialize padded arrays
        tokens = np.zeros((self.max_objects, 11), dtype=np.float32)
        mask = np.zeros((self.max_objects,), dtype=np.float32)
        
        # Fill valid objects
        n_objects = min(len(objects_list), self.max_objects)
        for i in range(n_objects):
            obj = objects_list[i]
            tokens[i] = [
                obj["type_id"],
                obj["x_ego"],
                obj["y_ego"],
                obj["sin_yaw"],
                obj["cos_yaw"],
                obj["length"],
                obj["width"],
                obj["vx"],
                obj["vy"],
                obj["oncoming_flag"],
                obj["priority_flag"],
            ]
            mask[i] = 1.0  # Valid object
        
        result = (torch.from_numpy(tokens), torch.from_numpy(mask))
        
        # Cache it
        self._objects_cache[frame_id] = result
        if len(self._objects_cache) > self._cache_size:
            to_remove = list(self._objects_cache.keys())[: self._cache_size // 5]
            for k in to_remove:
                del self._objects_cache[k]
        
        return result
    


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
    
    # Pre-allocate output tensors (Phase 3 optimization)
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
    Initialize worker to reduce memory footprint.
    
    This function is called in each DataLoader worker process to:
    - Reduce cache sizes to avoid OOM (split cache across workers)
    - Set memory limits for PyArrow to prevent leaks
    - Clear existing caches to free memory
    
    Args:
        worker_id: Worker process ID (0 to num_workers-1)
    """
    import pyarrow as pa
    
    # Get worker info and dataset
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset
        
        # Reduce cache size per worker to avoid OOM
        # With 2 workers and 200 cache each = 400 total vs 8 workers * 1000 = 8000 before
        dataset._cache_size = 200  # Down from 1000
        
        # Clear existing caches to free memory
        dataset._frames_cache.clear()
        dataset._bev_cache.clear()
        dataset._route_cache.clear()
        dataset._futures_cache.clear()
        dataset._objects_cache.clear()
        
        print(f"[Worker {worker_id}] Initialized with cache_size={dataset._cache_size}")
    
    # Limit PyArrow memory pool to prevent unbounded growth
    # This prevents memory leaks in long-running worker processes
    try:
        pa.set_memory_pool(pa.proxy_memory_pool(pa.default_memory_pool(), 512 * 1024 * 1024))  # 512 MB limit per worker
    except Exception as e:
        print(f"[Worker {worker_id}] Warning: Could not set PyArrow memory limit: {e}")


def create_bc_dataloader(
    run_dir: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 8,
    prefetch_factor: int = 4,
    persistent_workers: bool = True,
    pin_memory: bool = True,
    drop_last: bool = False,
    future_horizon: int = 12,
    route_points: int = 32,
    max_objects: int = 64,
    use_bev_mmap: bool = True,
    augment: bool = False,
    split: str = "all",
    val_ratio: float = 0.2,
    **kwargs
) -> torch.utils.data.DataLoader:
    """
    Create an optimized PyTorch DataLoader for BC trajectories.
    
    Args:
        run_dir: Path to BC run directory
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes for data loading (default: 8)
        prefetch_factor: Batches to prefetch per worker (default: 4)
        persistent_workers: Keep workers alive between epochs (default: True)
        pin_memory: Pin memory for faster GPU transfer (default: True)
        drop_last: Drop last incomplete batch (default: False)
        future_horizon: Number of future waypoints (N)
        route_points: Number of route points (K)
        max_objects: Maximum number of object tokens (M)
        use_bev_mmap: If True, use pre-computed memory-mapped BEV file (default: True)
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
    print(f"[BCDataLoader] Dataset initialized in {init_time:.2f}s (lazy loading enabled)")
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
        "collate_fn": fast_collate_fn,  # Phase 3 optimization: faster batching
    }
    
    # prefetch_factor only valid when num_workers > 0
    if num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = prefetch_factor
        # Add worker init function to reduce memory per worker
        dataloader_kwargs["worker_init_fn"] = _worker_init_fn
    
    # Merge any additional kwargs
    dataloader_kwargs.update(kwargs)
    
    return torch.utils.data.DataLoader(dataset, **dataloader_kwargs)
