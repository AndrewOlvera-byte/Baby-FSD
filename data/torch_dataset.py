"""
PyTorch Dataset and DataLoader for BC trajectories.

Loads multi-table Parquet data and provides batched samples for training.
"""

import io
import os
import zlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyarrow.dataset as ds
import torch
from torch.utils.data import Dataset


class BCTrajectoryDataset(Dataset):
    """
    PyTorch Dataset for BC trajectories from Parquet files.
    
    Each sample returns:
        - bev: (C, H, W) BEV tensor
        - route: (K, 2) route waypoints in ego frame
        - futures: (N, 2) future waypoints in ego frame
        - futures_speed: (N,) future speeds
        - control: (3,) [steer, throttle, brake]
        - state: (7,) [speed, yaw_rate, accel_long, accel_lat, curvature, speed_limit, command]
    """
    
    def __init__(self, run_dir: str, future_horizon: int = 12, route_points: int = 32):
        """
        Args:
            run_dir: Path to BC run directory (contains frames/, bev_frames/, etc.)
            future_horizon: Number of future waypoints (N)
            route_points: Number of route points (K)
        """
        self.run_dir = os.path.abspath(run_dir)
        self.future_horizon = future_horizon
        self.route_points = route_points
        
        # Load tables
        frames_path = os.path.join(self.run_dir, "frames")
        futures_path = os.path.join(self.run_dir, "futures")
        route_path = os.path.join(self.run_dir, "route_points")
        bev_path = os.path.join(self.run_dir, "bev_frames")
        
        if not all(os.path.isdir(p) for p in [frames_path, futures_path, route_path, bev_path]):
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
    
    def __len__(self) -> int:
        return len(self._frames)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        row = self._frames.iloc[idx]
        frame_id = int(row.frame_id)
        
        # BEV tensor
        bev = self._decode_bev(frame_id)
        
        # Route points
        route = self._get_route(frame_id)
        
        # Future waypoints
        futures_xy, futures_speed = self._get_futures(frame_id)
        
        # Control labels
        control = torch.tensor(
            [row.steer_norm, row.throttle, row.brake],
            dtype=torch.float32
        )
        
        # State features
        state = torch.tensor(
            [
                row.speed_mps,
                row.yaw_rate,
                row.accel_long,
                row.accel_lat,
                row.curvature,
                row.speed_limit_mps,
                float(row.command),
            ],
            dtype=torch.float32
        )
        
        return {
            "frame_id": frame_id,
            "bev": bev,
            "route": route,
            "futures": futures_xy,
            "futures_speed": futures_speed,
            "control": control,
            "state": state,
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
    
    def _get_futures(self, frame_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get future waypoints and speeds, padded to N."""
        futures_list = self._futures.get(frame_id, [])
        
        # Pad or truncate to N steps
        futures_xy = np.zeros((self.future_horizon, 2), dtype=np.float32)
        futures_speed = np.zeros((self.future_horizon,), dtype=np.float32)
        
        for i, (x, y, v) in enumerate(futures_list[:self.future_horizon]):
            futures_xy[i] = [x, y]
            futures_speed[i] = v
        
        return torch.from_numpy(futures_xy), torch.from_numpy(futures_speed)


def create_bc_dataloader(
    run_dir: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    future_horizon: int = 12,
    route_points: int = 32,
    **kwargs
) -> torch.utils.data.DataLoader:
    """
    Create a PyTorch DataLoader for BC trajectories.
    
    Args:
        run_dir: Path to BC run directory
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes for data loading
        future_horizon: Number of future waypoints (N)
        route_points: Number of route points (K)
        **kwargs: Additional arguments for DataLoader
    
    Returns:
        DataLoader instance
    """
    dataset = BCTrajectoryDataset(
        run_dir=run_dir,
        future_horizon=future_horizon,
        route_points=route_points,
    )
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        **kwargs
    )

