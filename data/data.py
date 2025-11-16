"""
Utility dataset helpers for loading BC collection runs.

This module focuses on two responsibilities:
1. Reading the per-table Parquet outputs produced by `collect/collect_bc.py`
2. Packaging each frame as a training-ready sample with masked future labels
"""

from __future__ import annotations

import io
import os
import zlib
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pyarrow.dataset as ds


def _load_parquet_table(table_dir: str):
    """Load every parquet file under `table_dir` into a pyarrow.Table."""
    if not os.path.isdir(table_dir):
        raise FileNotFoundError(f"Missing table directory: {table_dir}")
    dataset = ds.dataset(table_dir, format="parquet")
    return dataset.to_table()


def _table_to_records(table, key: str):
    """Convert a pyarrow.Table into a dict keyed by `key` (usually frame_id)."""
    df = table.to_pandas()
    grouped: Dict[int, list] = {}
    for row in df.itertuples(index=False):
        grouped.setdefault(getattr(row, key), []).append(row)
    return grouped


@dataclass
class FutureBatch:
    xy: np.ndarray
    speed: np.ndarray
    mask: np.ndarray


class BCRunDataset:
    """
    Simple iterable dataset over a single BC run directory.

    Each sample returns:
        - metadata/features from frames table
        - decoded BEV tensor
        - padded + masked future waypoints
    """

    def __init__(self, run_dir: str, future_horizon: int):
        self.run_dir = os.path.abspath(run_dir)
        self.future_horizon = int(future_horizon)

        frames_table = _load_parquet_table(os.path.join(self.run_dir, "frames"))
        futures_table = _load_parquet_table(os.path.join(self.run_dir, "futures"))
        bev_table = _load_parquet_table(os.path.join(self.run_dir, "bev_frames"))

        frames_df = frames_table.to_pandas().sort_values("frame_id").reset_index(drop=True)
        self._frames = frames_df
        self._frame_ids = frames_df["frame_id"].astype(int).to_list()

        self._futures = self._build_future_index(futures_table)
        self._bev = self._build_bev_index(bev_table)

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, idx: int) -> Dict:
        row = self._frames.iloc[idx]
        frame_id = int(row.frame_id)
        futures = self._pad_futures(frame_id)
        bev = self._decode_bev(frame_id)

        sample = {
            "frame_id": frame_id,
            "ego_position": np.array(
                [row.ego_x_w, row.ego_y_w, row.ego_yaw_w], dtype=np.float32
            ),
            "speed_mps": float(row.speed_mps),
            "yaw_rate": float(row.yaw_rate),
            "control": np.array(
                [row.steer_norm, row.throttle, row.brake], dtype=np.float32
            ),
            "bev": bev,
            "futures_xy": futures.xy,
            "futures_speed": futures.speed,
            "futures_mask": futures.mask,
        }
        return sample

    def _build_future_index(self, table) -> Dict[int, Dict[int, Tuple[float, float, float]]]:
        records = _table_to_records(table, "frame_id")
        index: Dict[int, Dict[int, Tuple[float, float, float]]] = {}
        for frame_id, rows in records.items():
            ordered: Dict[int, Tuple[float, float, float]] = {}
            for row in rows:
                ordered[int(row.i)] = (
                    float(row.x_ego),
                    float(row.y_ego),
                    float(row.v_mps),
                )
            index[int(frame_id)] = ordered
        return index

    def _build_bev_index(self, table):
        df = table.select(["frame_id", "data"]).to_pandas()
        return {int(row.frame_id): row.data for row in df.itertuples(index=False)}

    def _pad_futures(self, frame_id: int) -> FutureBatch:
        xy = np.zeros((self.future_horizon, 2), dtype=np.float32)
        speed = np.zeros((self.future_horizon,), dtype=np.float32)
        mask = np.zeros((self.future_horizon,), dtype=np.float32)

        entries = self._futures.get(frame_id, {})
        for i, (x, y, v) in entries.items():
            if 0 <= i < self.future_horizon:
                xy[i] = (x, y)
                speed[i] = v
                mask[i] = 1.0

        return FutureBatch(xy=xy, speed=speed, mask=mask)

    def _decode_bev(self, frame_id: int) -> Optional[np.ndarray]:
        blob = self._bev.get(frame_id)
        if blob is None:
            return None
        raw = zlib.decompress(blob)
        arr = np.load(io.BytesIO(raw), allow_pickle=False)
        return arr.astype(np.float32)


