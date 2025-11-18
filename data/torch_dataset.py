"""
PyTorch Dataset and DataLoader for BC trajectories.

Loads multi-table Parquet data and provides batched samples for training.
Includes normalization, object tokens, and future masking.
"""

import io
import os
import zlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyarrow.dataset as ds
import torch
from torch.utils.data import Dataset

from data.norms import (
    normalize_ego_vector,
    normalize_route_points,
    normalize_futures,
    normalize_object_tokens,
    normalize_bev,
)


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
    ):
        """
        Args:
            run_dir: Path to BC run directory (contains frames/, bev_frames/, etc.)
            future_horizon: Number of future waypoints (N)
            route_points: Number of route points (K)
            max_objects: Maximum number of object tokens (M)
        """
        self.run_dir = os.path.abspath(run_dir)
        self.future_horizon = future_horizon
        self.route_points = route_points
        self.max_objects = max_objects
        
        # Load tables
        frames_path = os.path.join(self.run_dir, "frames")
        futures_path = os.path.join(self.run_dir, "futures")
        route_path = os.path.join(self.run_dir, "route_points")
        bev_path = os.path.join(self.run_dir, "bev_frames")
        objects_path = os.path.join(self.run_dir, "object_tokens")
        
        # Check required tables exist
        required_paths = [frames_path, futures_path, route_path, bev_path]
        if not all(os.path.isdir(p) for p in required_paths):
            raise FileNotFoundError(f"Missing required tables in {run_dir}")
        
        # Load frames table (main index)
        frames_dataset = ds.dataset(frames_path, format="parquet")
        frames_table = frames_dataset.to_table()
        self._frames = frames_table.to_pandas().sort_values("frame_id").reset_index(drop=True)
        self._frame_ids = self._frames["frame_id"].astype(int).tolist()
        
        # Build indices for other tables
        self._futures = self._build_futures_index(ds.dataset(futures_path, format="parquet").to_table())
        self._route = self._build_route_index(ds.dataset(route_path, format="parquet").to_table())
        self._bev = self._build_bev_index(ds.dataset(bev_path, format="parquet").to_table())
        
        # Build object tokens index (may not exist in all datasets)
        if os.path.isdir(objects_path):
            self._objects = self._build_objects_index(ds.dataset(objects_path, format="parquet").to_table())
        else:
            self._objects = {}
    
    def __len__(self) -> int:
        return len(self._frames)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single normalized sample."""
        row = self._frames.iloc[idx]
        frame_id = int(row.frame_id)
        
        # Ego vector (normalized)
        ego_vec = normalize_ego_vector(row)
        
        # BEV tensor (normalized)
        bev = self._decode_bev(frame_id)
        bev = normalize_bev(bev)
        
        # Route points (normalized)
        route = self._get_route(frame_id)
        route = normalize_route_points(route)
        
        # Object tokens (normalized) + mask
        objects, object_mask = self._get_objects(frame_id)
        objects = normalize_object_tokens(objects)
        
        # Future waypoints and speeds (normalized) + mask
        futures_xy, futures_speed, future_mask = self._get_futures(frame_id)
        futures_xy, futures_speed = normalize_futures(futures_xy, futures_speed)
        
        return {
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
    
    def _build_futures_index(self, table) -> Dict[int, List[Tuple[float, float, float]]]:
        """Build frame_id -> [(x, y, v), ...] index."""
        df = table.to_pandas()
        index: Dict[int, List[Tuple[float, float, float]]] = {}
        
        for frame_id, group in df.groupby("frame_id"):
            # Sort by future index (i)
            group = group.sort_values("i")
            futures = [
                (float(row.x_ego), float(row.y_ego), float(row.v_mps))
                for row in group.itertuples(index=False)
            ]
            index[int(frame_id)] = futures
        
        return index
    
    def _build_route_index(self, table) -> Dict[int, List[Tuple[float, float]]]:
        """Build frame_id -> [(x, y), ...] index."""
        df = table.to_pandas()
        index: Dict[int, List[Tuple[float, float]]] = {}
        
        for frame_id, group in df.groupby("frame_id"):
            # Sort by route index
            group = group.sort_values("idx")
            route = [
                (float(row.x_ego), float(row.y_ego))
                for row in group.itertuples(index=False)
            ]
            index[int(frame_id)] = route
        
        return index
    
    def _build_bev_index(self, table):
        """Build frame_id -> compressed_blob index."""
        df = table.select(["frame_id", "data"]).to_pandas()
        return {int(row.frame_id): row.data for row in df.itertuples(index=False)}
    
    def _build_objects_index(self, table) -> Dict[int, List[Dict]]:
        """Build frame_id -> [object_dict, ...] index."""
        df = table.to_pandas()
        index: Dict[int, List[Dict]] = {}
        
        for frame_id, group in df.groupby("frame_id"):
            # Sort by object index
            group = group.sort_values("idx")
            objects = []
            for row in group.itertuples(index=False):
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
                objects.append(obj)
            index[int(frame_id)] = objects
        
        return index
    
    def _decode_bev(self, frame_id: int) -> torch.Tensor:
        """Decode BEV from compressed bytes."""
        blob = self._bev.get(frame_id)
        if blob is None:
            # Return empty BEV if missing
            return torch.zeros((18, 150, 200), dtype=torch.float32)
        
        raw = zlib.decompress(blob)
        arr = np.load(io.BytesIO(raw), allow_pickle=False)
        return torch.from_numpy(arr.astype(np.float32))
    
    def _get_route(self, frame_id: int) -> torch.Tensor:
        """Get route points, padded to K."""
        route_list = self._route.get(frame_id, [])
        
        # Pad or truncate to K points
        route = np.zeros((self.route_points, 2), dtype=np.float32)
        for i, (x, y) in enumerate(route_list[:self.route_points]):
            route[i] = [x, y]
        
        return torch.from_numpy(route)
    
    def _get_objects(self, frame_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get object tokens and mask, padded to M."""
        objects_list = self._objects.get(frame_id, [])
        
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
        
        return torch.from_numpy(tokens), torch.from_numpy(mask)
    
    def _get_futures(self, frame_id: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get future waypoints, speeds, and mask. Padded to N.
        
        Mask strategy: 1.0 for valid futures, 0.0 for padded/missing.
        This allows model to learn terminal behavior while not penalizing
        missing predictions at episode end.
        """
        futures_list = self._futures.get(frame_id, [])
        
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
        
        return (
            torch.from_numpy(futures_xy),
            torch.from_numpy(futures_speed),
            torch.from_numpy(future_mask),
        )


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
        **kwargs: Additional arguments for DataLoader
    
    Returns:
        DataLoader instance with optimized settings
    """
    dataset = BCTrajectoryDataset(
        run_dir=run_dir,
        future_horizon=future_horizon,
        route_points=route_points,
        max_objects=max_objects,
    )
    
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
    }
    
    # prefetch_factor only valid when num_workers > 0
    if num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = prefetch_factor
    
    # Merge any additional kwargs
    dataloader_kwargs.update(kwargs)
    
    return torch.utils.data.DataLoader(dataset, **dataloader_kwargs)
