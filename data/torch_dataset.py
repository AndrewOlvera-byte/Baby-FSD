"""
Optimized for high-throughput training using WebLoader pattern:
- Workers yield individual samples
- WebLoader collects from all workers and rebatches in main process
"""

import os
import time
import glob
import json
import io
import tarfile
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import IterableDataset
import webdataset as wds
from data.norms import (
    normalize_route_points,
    normalize_futures,
    normalize_object_tokens,
    normalize_bev,
)
from data.transforms import BCTransform

# Required fields
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
    metadata_path = os.path.join(run_dir, "metadata.json")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"metadata.json not found in {run_dir}")
    
    with open(metadata_path, "r") as f:
        return json.load(f)


def _read_shard_sample_count(shard_path: str) -> Optional[int]:
    try:
        with tarfile.open(shard_path, "r:*") as tar:
            try:
                meta_member = tar.getmember("__metadata__.json")
                meta_file = tar.extractfile(meta_member)

                if meta_file is not None:
                    shard_meta = json.load(meta_file)
                    sample_count = int(shard_meta.get("sample_count", 0))

                    if sample_count > 0:
                        return sample_count

            except KeyError:
                pass

            sample_count = 0

            for member in tar:
                if member.isfile() and member.name.endswith(".frame_id.npy"):
                    sample_count += 1

            if sample_count > 0:
                return sample_count

    except Exception as e:
        print(f"[BCWebDataset] WARNING: failed to read shard metadata from {shard_path}: {e}")

    return None


def _compute_shard_sample_counts(shard_files: List[str], dataset_metadata: Dict[str, object],) -> Tuple[List[int], int]:
    default_per_shard = int(dataset_metadata.get("samples_per_shard", 0) or 0)
    metadata_total_samples = int(dataset_metadata.get("total_samples", 0) or 0)

    shard_counts: List[int] = []
    missing_counts: List[str] = []

    for idx, shard_path in enumerate(shard_files):
        count = _read_shard_sample_count(shard_path)

        if count is None and default_per_shard > 0:
            remaining = max(0, metadata_total_samples - default_per_shard * idx)

            if remaining > 0:
                count = min(default_per_shard, remaining)

            else:
                count = default_per_shard

        if count is None:
            missing_counts.append(os.path.basename(shard_path))
            count = 0

        shard_counts.append(int(count))

    if missing_counts:
        preview = ", ".join(missing_counts[:3])

        if len(missing_counts) > 3:
            preview += f", ... ({len(missing_counts)} shards)"

        print(f"[BCWebDataset] WARNING: missing shard sample metadata for {preview}")

    total_from_shards = sum(shard_counts)

    if metadata_total_samples > 0 and total_from_shards > 0 and metadata_total_samples != total_from_shards:
        print(
            f"[BCWebDataset] WARNING: shard counts total {total_from_shards:,} "
            f"differs from metadata total {metadata_total_samples:,}"
        )

    return shard_counts, total_from_shards


def _has_all_required_keys(sample_dict: Dict[str, object]) -> bool:
    if not hasattr(sample_dict, "keys"):
        return False
    keys = sample_dict.keys()
    for field in REQUIRED_KEYS:
        if not any(k.endswith(f".{field}.npy") or k == f"{field}.npy" for k in keys):
            return False
    return True


def _parse_and_normalize_sample(sample_dict: Dict) -> Dict[str, torch.Tensor]:
    """
    Args:
        sample_dict: Dict from WebDataset with keys like "00000000.frame_id.npy", etc.
        
    Returns:
        Dict with normalized torch tensors for a single sample
    """
    # Parse numpy arrays from WebDataset dict
    sample_data = {}
    
    for key, value in sample_dict.items():
        if not key.endswith(".npy"):
            continue
        
        # Extract field name from filename
        parts = key.rsplit(".", 2)  # Split from right: [idx, field, "npy"] or [field, "npy"]
        if len(parts) == 3:
            field_name = parts[1]

        elif len(parts) == 2:
            field_name = parts[0]
            
        else:
            continue
        
        # Parse numpy array from bytes
        if isinstance(value, bytes):
            arr = np.load(io.BytesIO(value), allow_pickle=False)

        elif isinstance(value, io.BytesIO):
            arr = np.load(value, allow_pickle=False)

        else:
            arr = value
        
        sample_data[field_name] = arr
    
    # Ensure all required keys are present
    for key in REQUIRED_KEYS:
        if key not in sample_data:
            raise ValueError(f"Missing required key '{key}' in sample. Available keys: {list(sample_data.keys())}")
    
    frame_id = int(sample_data["frame_id"])
    
    ego_vec_arr = sample_data["ego_vec"]

    if ego_vec_arr.dtype != np.float32:
        ego_vec_arr = ego_vec_arr.astype(np.float32)

    ego_vec = torch.from_numpy(ego_vec_arr)
    
    bev_arr = sample_data["bev"]

    if bev_arr.dtype != np.float32:
        bev_arr = bev_arr.astype(np.float32)

    bev = torch.from_numpy(bev_arr)
    
    route_arr = sample_data["route"]

    if route_arr.dtype != np.float32:
        route_arr = route_arr.astype(np.float32)

    route = torch.from_numpy(route_arr)
    
    objects_arr = sample_data["objects"]

    if objects_arr.dtype != np.float32:
        objects_arr = objects_arr.astype(np.float32)

    objects = torch.from_numpy(objects_arr)
    
    object_mask_arr = sample_data["object_mask"]

    if object_mask_arr.dtype != np.float32:
        object_mask_arr = object_mask_arr.astype(np.float32)

    object_mask = torch.from_numpy(object_mask_arr)
    
    future_xy_arr = sample_data["future_xy"]

    if future_xy_arr.dtype != np.float32:
        future_xy_arr = future_xy_arr.astype(np.float32)

    future_xy = torch.from_numpy(future_xy_arr)
    
    future_v_arr = sample_data["future_v"]

    if future_v_arr.dtype != np.float32:
        future_v_arr = future_v_arr.astype(np.float32)

    future_v = torch.from_numpy(future_v_arr)
    
    future_mask_arr = sample_data["future_mask"]

    if future_mask_arr.dtype != np.float32:
        future_mask_arr = future_mask_arr.astype(np.float32)

    future_mask = torch.from_numpy(future_mask_arr)
    
    # Normalize
    bev = normalize_bev(bev)
    route = normalize_route_points(route)
    objects = normalize_object_tokens(objects)
    future_xy, future_v = normalize_futures(future_xy, future_v)
    route_mask = (route.abs().sum(dim=-1) > 1e-3)

    return {
        "frame_id": frame_id,
        "ego_vec": ego_vec,
        "bev": bev,
        "route": route,
        "route_mask": route_mask,
        "objects": objects,
        "object_mask": object_mask,
        "future_xy": future_xy,
        "future_v": future_v,
        "future_mask": future_mask,
    }


def _make_webdataset_pipeline(urls: List[str], split: str, shuffle_buffer_size: int, augment: bool, transform: Optional[BCTransform]) -> wds.WebDataset:
    """
    Args:
        urls: List of shard URLs (file:// protocol)
        split: "train", "val", or "all"
        shuffle_buffer_size: Buffer size for sample shuffling
        augment: Whether to apply augmentation
        transform: BCTransform instance (if augment=True)
    
    Returns:
        WebDataset pipeline yielding individual samples
    """
    # Shard shuffling
    if split == "train" or split == "all":
        shardshuffle = len(urls)

    else:
        shardshuffle = 0
    
    # Create base dataset
    dataset = wds.WebDataset(
        urls,
        shardshuffle=shardshuffle,
        nodesplitter=wds.shardlists.split_by_worker,
        empty_check=False,
    )
    
    dataset = dataset.decode()
    
    dataset = dataset.select(_has_all_required_keys)

    # Parse and normalize samples
    dataset = dataset.map(_parse_and_normalize_sample)
    
    # Apply augmentation transforms
    if augment and transform is not None:
        dataset = dataset.map(transform)
    
    return dataset


class BCWebDataset(IterableDataset):
    """
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
        split: str = "all",
        val_ratio: float = 0.2,
        use_bev_mmap: bool = False,
        shuffle_buffer_size: int = 5000,
        augment_config: dict = None,
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
            shuffle_buffer_size: Buffer size for WebDataset shuffling (per worker)
            augment_config: Optional dict with custom augmentation parameters
        """
        
        self.run_dir = os.path.abspath(run_dir)
        self.future_horizon = future_horizon
        self.route_points = route_points
        self.max_objects = max_objects
        self.augment = augment
        
        # Build transform
        if augment:
            aug_cfg = augment_config or {}

            self.transform = BCTransform(
                augment=True,
                p_rot=aug_cfg.get("p_rot", 0.5),
                max_rot_deg=aug_cfg.get("max_rot_deg", 20.0),
                p_bev_cutout=aug_cfg.get("p_bev_cutout", 0.2),
                p_bev_channel_drop=aug_cfg.get("p_bev_channel_drop", 0.1),
                p_drop_bev=aug_cfg.get("p_drop_bev", 0.05),
                p_drop_objects=aug_cfg.get("p_drop_objects", 0.05),
                p_ego_noise=aug_cfg.get("p_ego_noise", 0.1),
                ego_noise_scale=aug_cfg.get("ego_noise_scale", 0.02),
                p_speed_scale=aug_cfg.get("p_speed_scale", 0.2),
                speed_scale_range=tuple(aug_cfg.get("speed_scale_range", [0.8, 1.2])),
                p_trajectory_noise=aug_cfg.get("p_trajectory_noise", 0.3),
                trajectory_noise_std=aug_cfg.get("trajectory_noise_std", 0.02),
                p_lateral_shift=aug_cfg.get("p_lateral_shift", 0.2),
                max_lateral_shift=aug_cfg.get("max_lateral_shift", 0.05),
            )

        else:
            self.transform = None

        self.split = split
        self.val_ratio = val_ratio
        self.shuffle_buffer_size = shuffle_buffer_size
        
        print(f"[BCWebDataset] Initializing WebDataset from {run_dir}")
        
        metadata = _load_metadata(self.run_dir)

        self.metadata = metadata
        self.metadata_total_samples = int(metadata.get("total_samples", 0) or 0)
        self.total_samples = self.metadata_total_samples
        
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
        
        print(f"[BCWebDataset] Total samples (metadata): {self.total_samples:,}")
        print(f"[BCWebDataset] Dimensions: K={K}, N={N}, M={M}, C={metadata.get('C', 18)}, H={metadata.get('H', 150)}, W={metadata.get('W', 200)}")
        
        shard_pattern = os.path.join(self.run_dir, "shard-*.tar")
        shard_files = sorted(glob.glob(shard_pattern))
        
        if not shard_files:
            raise FileNotFoundError(f"No shard files found matching {shard_pattern}")
        
        print(f"[BCWebDataset] Found {len(shard_files)} shard files")

        # Compute shard sample counts
        shard_sample_counts, total_from_shards = _compute_shard_sample_counts(shard_files, metadata)

        if total_from_shards > 0:
            self.total_samples = total_from_shards
            print(f"[BCWebDataset] Total samples (from shards): {self.total_samples:,}")
        
        # Apply train/val split
        if split != "all":
            split_idx = int(len(shard_files) * (1 - val_ratio))
            # Edge case: ensure train gets at least 1 shard if shards exist
            if split_idx == 0 and len(shard_files) > 0:
                split_idx = 1
            
            if split == "train":
                self.shard_files = shard_files[:split_idx]
                self.shard_sample_counts = shard_sample_counts[:split_idx]
                print(f"[BCWebDataset] Split 'train': Using {len(self.shard_files)} shards")

            elif split == "val":
                self.shard_files = shard_files[split_idx:]
                self.shard_sample_counts = shard_sample_counts[split_idx:]
                print(f"[BCWebDataset] Split 'val': Using {len(self.shard_files)} shards")

            else:
                raise ValueError(f"Unsupported split '{split}'. Use 'train', 'val', or 'all'.")

        else:
            self.shard_files = shard_files
            self.shard_sample_counts = shard_sample_counts

        self.split_sample_count = sum(self.shard_sample_counts)

        if self.split_sample_count <= 0:
            raise ValueError(f"[BCWebDataset] Split '{self.split}' contains no samples")
        
        # Build WebDataset URLs
        self.urls = [f"file://{os.path.abspath(f)}" for f in self.shard_files]
        
        print(f"[BCWebDataset] Dataset ready: {len(self.urls)} shards, {self.split_sample_count:,} samples for split '{self.split}'")
    
    def __iter__(self):
        """Create WebDataset iterator yielding individual samples."""
        dataset = _make_webdataset_pipeline(
            urls=self.urls,
            split=self.split,
            shuffle_buffer_size=self.shuffle_buffer_size,
            augment=self.augment,
            transform=self.transform,
        )
        
        return iter(dataset)
    
    def __len__(self):
        """Return the number of samples available for this split."""
        return self.split_sample_count


def default_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Args:
        batch: List of sample dicts from WebDataset
        
    Returns:
        Dict with batched tensors
    """
    if not batch:
        raise ValueError("default_collate_fn received an empty batch")
    
    B = len(batch)
    first = batch[0]
    
    # Get dimensions
    d_ego = first["ego_vec"].shape[0]
    C, H, W = first["bev"].shape
    K = first["route"].shape[0]
    M = first["objects"].shape[0]
    d_obj = first["objects"].shape[1]
    N = first["future_xy"].shape[0]
    
    # Stack tensors
    return {
        "ego_vec": torch.stack([s["ego_vec"] for s in batch], dim=0),
        "bev": torch.stack([s["bev"] for s in batch], dim=0),
        "route": torch.stack([s["route"] for s in batch], dim=0),
        "route_mask": torch.stack([s["route_mask"] for s in batch], dim=0),
        "objects": torch.stack([s["objects"] for s in batch], dim=0),
        "object_mask": torch.stack([s["object_mask"] for s in batch], dim=0),
        "future_xy": torch.stack([s["future_xy"] for s in batch], dim=0),
        "future_v": torch.stack([s["future_v"] for s in batch], dim=0),
        "future_mask": torch.stack([s["future_mask"] for s in batch], dim=0),
    }


def validate_data_diversity(run_dir: str, sample_size: int = 1000, min_speed_std: float = 2.0, min_steer_std: float = 0.1, min_moving_ratio: float = 0.7, max_stationary_ratio: float = 0.25, warn_on_fail: bool = True) -> dict:
    """
    Args:
        run_dir: Path to WebDataset directory
        sample_size: Number of samples to check
        min_speed_std: Minimum speed standard deviation (m/s, after denorm)
        min_steer_std: Minimum steering standard deviation
        min_moving_ratio: Minimum ratio of moving frames (speed > 0.5 m/s)
        max_stationary_ratio: Maximum ratio of stationary frames
        warn_on_fail: If True, print warnings; if False, raise exceptions
        
    Returns:
        Dict with diversity statistics
    """
    
    # Quick check
    metadata_path = os.path.join(run_dir, "metadata.json")

    if not os.path.exists(metadata_path):
        if warn_on_fail:
            print(f"[DataDiversity] WARNING: metadata.json not found in {run_dir}")
        return {"status": "skipped", "reason": "no metadata"}
    
    # Sample shards
    shard_files = sorted(glob.glob(os.path.join(run_dir, "shard-*.tar")))

    if not shard_files:
        if warn_on_fail:
            print(f"[DataDiversity] WARNING: No shard files found in {run_dir}")
        return {"status": "skipped", "reason": "no shards"}
    
    # Load samples
    speeds = []
    steers = []
    future_speeds = []
    
    samples_collected = 0
    max_shards = min(3, len(shard_files))
    
    for shard_path in shard_files[:max_shards]:
        try:
            with tarfile.open(shard_path, "r:*") as tar:
                for member in tar:
                    if member.name.endswith(".ego_vec.npy"):
                        f = tar.extractfile(member)
                        if f is not None:
                            ego_vec = np.load(io.BytesIO(f.read()))
                            speed = float(ego_vec[0]) * 50.0
                            steer = float(ego_vec[6]) if len(ego_vec) > 6 else 0.0
                            speeds.append(speed)
                            steers.append(steer)
                            samples_collected += 1
                    
                    if member.name.endswith(".future_v.npy"):
                        f = tar.extractfile(member)
                        if f is not None:
                            future_v = np.load(io.BytesIO(f.read()))
                            # Denormalize
                            future_speeds.extend((future_v * 50.0).tolist())
                    
                    if samples_collected >= sample_size:
                        break
                        
        except Exception as e:
            if warn_on_fail:
                print(f"[DataDiversity] WARNING: Error reading shard {shard_path}: {e}")
            continue
        
        if samples_collected >= sample_size:
            break
    
    if samples_collected < 100:
        if warn_on_fail:
            print(f"[DataDiversity] WARNING: Only collected {samples_collected} samples, need at least 100")
        return {"status": "insufficient_samples", "samples": samples_collected}
    
    # Compute statistics
    speeds = np.array(speeds)
    steers = np.array(steers)
    
    speed_mean = float(np.mean(speeds))
    speed_std = float(np.std(speeds))
    steer_std = float(np.std(steers))
    
    moving_mask = speeds > 0.5  # Moving if speed > 0.5 m/s
    moving_ratio = float(np.mean(moving_mask))
    stationary_ratio = 1.0 - moving_ratio
    
    stats = {
        "status": "ok",
        "samples_checked": samples_collected,
        "speed_mean": speed_mean,
        "speed_std": speed_std,
        "steer_std": steer_std,
        "moving_ratio": moving_ratio,
        "stationary_ratio": stationary_ratio,
    }
    
    # Check thresholds
    issues = []
    if speed_std < min_speed_std:
        issues.append(f"speed_std={speed_std:.2f} < {min_speed_std} (low speed diversity)")
    if steer_std < min_steer_std:
        issues.append(f"steer_std={steer_std:.3f} < {min_steer_std} (low steering diversity)")
    if moving_ratio < min_moving_ratio:
        issues.append(f"moving_ratio={moving_ratio:.2f} < {min_moving_ratio} (too many stationary frames)")
    if stationary_ratio > max_stationary_ratio:
        issues.append(f"stationary_ratio={stationary_ratio:.2f} > {max_stationary_ratio} (too many stationary frames)")
    
    if issues:
        stats["status"] = "warning"
        stats["issues"] = issues
        if warn_on_fail:
            print(f"[DataDiversity] WARNING: Data quality issues detected:")
            for issue in issues:
                print(f"  - {issue}")
            print(f"[DataDiversity] Stats: speed_mean={speed_mean:.2f} m/s, speed_std={speed_std:.2f}, "
                  f"steer_std={steer_std:.3f}, moving={moving_ratio:.1%}")
        else:
            raise ValueError(f"Data diversity check failed: {issues}")
    else:
        print(f"[DataDiversity] Data quality OK: speed_mean={speed_mean:.2f} m/s, speed_std={speed_std:.2f}, "
              f"steer_std={steer_std:.3f}, moving={moving_ratio:.1%}")
    
    return stats


def create_bc_dataloader(
    run_dir: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    pin_memory: bool = True,
    drop_last: bool = False,
    future_horizon: int = 12,
    route_points: int = 32,
    max_objects: int = 64,
    use_bev_mmap: bool = False,
    augment: bool = False,
    split: str = "all",
    val_ratio: float = 0.2,
    shuffle_buffer_size: int = 5000,
    validate_diversity: bool = False,
    diversity_config: dict = None,
    augment_config: dict = None,
    **kwargs
) -> wds.WebLoader:
    """
    Args:
        run_dir: Path to WebDataset directory
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        prefetch_factor: Batches to prefetch per worker
        persistent_workers: Keep workers alive between epochs
        pin_memory: Pin memory
        drop_last: Drop last incomplete batch
        future_horizon: Number of future waypoints (N)
        route_points: Number of route points (K)
        max_objects: Maximum number of object tokens (M)
        use_bev_mmap: Ignored (legacy parameter)
        augment: If True, apply data augmentation
        split: Dataset split ("train", "val", "all")
        val_ratio: Fraction of data for validation
        shuffle_buffer_size: Per-worker shuffle buffer size
        validate_diversity: If True, run diversity validation before training
        diversity_config: Config dict for diversity validation thresholds
        **kwargs: Additional arguments (ignored for compatibility)
        
    Returns:
        wds.WebLoader instance
    """
    start_time = time.time()
    
    # Validate data diversity
    if validate_diversity and split == "train":
        div_cfg = diversity_config or {}
        diversity_stats = validate_data_diversity(
            run_dir=run_dir,
            sample_size=div_cfg.get("sample_size", 1000),
            min_speed_std=div_cfg.get("min_speed_std", 2.0),
            min_steer_std=div_cfg.get("min_steer_std", 0.1),
            min_moving_ratio=div_cfg.get("min_moving_ratio", 0.7),
            max_stationary_ratio=div_cfg.get("max_stationary_ratio", 0.25),
            warn_on_fail=div_cfg.get("warn_on_fail", True),
        )
    
    # Create dataset that yields INDIVIDUAL samples (no batching)
    dataset = BCWebDataset(
        run_dir=run_dir,
        future_horizon=future_horizon,
        route_points=route_points,
        max_objects=max_objects,
        use_bev_mmap=use_bev_mmap,
        augment=augment,
        split=split,
        val_ratio=val_ratio,
        shuffle_buffer_size=shuffle_buffer_size,
        augment_config=augment_config,
    )
    
    init_time = time.time() - start_time
    num_shards = len(dataset.shard_files)

    print(f"[BCDataLoader] Dataset initialized in {init_time:.2f}s")
    print(f"[BCDataLoader] Using WebLoader pattern: {num_shards} shards, {num_workers} workers")
    
    # Adjust workers
    if num_workers == 0:
        persistent_workers = False
        prefetch_factor = 2
    
    # Create WebLoader
    loader = wds.WebLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    
    # Shuffle samples, then batch
    if split == "train" or split == "all":
        loader = loader.shuffle(shuffle_buffer_size).batched(batch_size, collation_fn=default_collate_fn)
    else:
        loader = loader.batched(batch_size, collation_fn=default_collate_fn)

    # Set epoch length explicitly for all splits
    # Without this, WebLoader stops when workers exhaust their shards
    num_batches = dataset.split_sample_count // batch_size
    loader = loader.with_epoch(num_batches)

    if split == "train":
        print(f"[BCDataLoader] WebLoader ready: batch_size={batch_size}, {num_batches} batches/epoch (with_epoch set)")
    else:
        print(f"[BCDataLoader] WebLoader ready: batch_size={batch_size}, ~{num_batches} batches/epoch")

    return loader


def create_rl_dataloader(
    run_dir: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    pin_memory: bool = True,
    drop_last: bool = False,
    future_horizon: int = 12,
    route_points: int = 32,
    max_objects: int = 64,
    use_bev_mmap: bool = False,
    augment: bool = False,
    split: str = "all",
    val_ratio: float = 0.2,
    shuffle_buffer_size: int = 5000,
    **kwargs
) -> wds.WebLoader:
    """
    Create dataloader for offline RL training.

    Args:
        run_dir: Path to WebDataset directory (RL dataset with rewards/actions/terminals)
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        prefetch_factor: Batches to prefetch per worker
        persistent_workers: Keep workers alive between epochs
        pin_memory: Pin memory
        drop_last: Drop last incomplete batch
        future_horizon: Number of future waypoints (N)
        route_points: Number of route points (K)
        max_objects: Maximum number of object tokens (M)
        use_bev_mmap: Ignored (legacy parameter)
        augment: If True, apply data augmentation
        split: Dataset split ("train", "val", "all")
        val_ratio: Fraction of data for validation
        shuffle_buffer_size: Per-worker shuffle buffer size
        **kwargs: Additional arguments (ignored for compatibility)

    Returns:
        wds.WebLoader instance
    """
    start_time = time.time()

    # Create dataset that yields INDIVIDUAL samples (no batching)
    # Note: Uses BCWebDataset which should work with RL data (same base fields + optional RL fields)
    dataset = BCWebDataset(
        run_dir=run_dir,
        future_horizon=future_horizon,
        route_points=route_points,
        max_objects=max_objects,
        use_bev_mmap=use_bev_mmap,
        augment=augment,
        split=split,
        val_ratio=val_ratio,
        shuffle_buffer_size=shuffle_buffer_size,
        augment_config=None,  # RL typically doesn't use CPU augmentation
    )

    init_time = time.time() - start_time
    num_shards = len(dataset.shard_files)

    print(f"[RLDataLoader] Dataset initialized in {init_time:.2f}s")
    print(f"[RLDataLoader] Using WebLoader pattern: {num_shards} shards, {num_workers} workers")
    print(f"[RLDataLoader] RL mode: expects reward/action/done fields in dataset")

    # Adjust workers
    if num_workers == 0:
        persistent_workers = False
        prefetch_factor = 2

    # Create WebLoader
    loader = wds.WebLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )

    # Shuffle samples, then batch
    if split == "train" or split == "all":
        loader = loader.shuffle(shuffle_buffer_size).batched(batch_size, collation_fn=default_collate_fn)
    else:
        loader = loader.batched(batch_size, collation_fn=default_collate_fn)

    # Set epoch length explicitly for all splits
    num_batches = dataset.split_sample_count // batch_size
    loader = loader.with_epoch(num_batches)

    if split == "train":
        print(f"[RLDataLoader] WebLoader ready: batch_size={batch_size}, {num_batches} batches/epoch (with_epoch set)")
    else:
        print(f"[RLDataLoader] WebLoader ready: batch_size={batch_size}, ~{num_batches} batches/epoch")

    return loader


def fast_collate_fn(batch):
    if isinstance(batch, dict):
        return batch

    if isinstance(batch, list):
        if len(batch) == 1 and isinstance(batch[0], dict):
            if "bev" in batch[0] and torch.is_tensor(batch[0]["bev"]) and batch[0]["bev"].ndim == 4:
                return batch[0]

        return default_collate_fn(batch)
    return batch
