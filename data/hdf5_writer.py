"""
HDF5 writer for BC episode sets with LZ4 compression.

Stores 5-episode sets in a single HDF5 file with all frames flattened along
a sample dimension, optimized for high-throughput sequential and random access.
"""

import os
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime

try:
    import h5py
    import hdf5plugin  # For LZ4 compression
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False
    h5py = None
    hdf5plugin = None


class HDF5EpisodeSetWriter:
    """
    Write BC data to HDF5 files in episode-set format.
    
    Each file contains data from multiple episodes (default 5), with all frames
    flattened along a single sample dimension. This allows efficient random
    access during training while maintaining small file counts.
    
    File structure:
        /frame_id: [N] int64
        /episode_id: [N] int16
        /ego_vec: [N, 14] float32 - normalized ego state
        /bev: [N, C, H, W] float32 - BEV tensors
        /route: [N, K, 2] float32 - route waypoints
        /objects: [N, M, 11] float32 - object tokens
        /object_mask: [N, M] float32 - object validity mask
        /future_xy: [N, N_fut, 2] float32 - future waypoints
        /future_v: [N, N_fut] float32 - future speeds
        /future_mask: [N, N_fut] float32 - future validity mask
        /reward_*: [N] float32 - raw/clipped/normalized reward and components
        /noise_injected: [N] bool - whether control noise was applied
    
    Attributes (metadata):
        K, N_future, C, H, W, version, norms_version, etc.
    """
    
    def __init__(
        self,
        output_dir: str,
        run_id: str,
        episodes_per_set: int = 5,
        K: int = 32,
        N_future: int = 12,
        M: int = 64,
        C: int = 18,
        H: int = 150,
        W: int = 200,
        compression: str = "lz4",
        chunk_size: int = 100,
        include_rewards: bool = False,
    ):
        """
        Args:
            output_dir: Base directory for HDF5 files
            run_id: Unique run identifier
            episodes_per_set: Number of episodes per HDF5 file
            K: Number of route points
            N_future: Number of future waypoints
            M: Maximum number of objects
            C, H, W: BEV dimensions
            compression: Compression filter ("lz4", "lzf", "gzip", or None)
            chunk_size: Chunk size along sample dimension
            include_rewards: Whether to store reward/noise datasets (offline RL)
        """
        if not HDF5_AVAILABLE:
            raise ImportError("h5py and hdf5plugin are required for HDF5 writer")
        
        self.output_dir = output_dir
        self.run_id = run_id
        self.episodes_per_set = episodes_per_set
        self.K = K
        self.N_future = N_future
        self.M = M
        self.C = C
        self.H = H
        self.W = W
        self.chunk_size = chunk_size
        
        # Setup compression
        self.compression = self._get_compression_filter(compression)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # State
        self._current_file: Optional[h5py.File] = None
        self._current_set_idx = 0
        self._episodes_in_current_set = 0
        self._current_size = 0  # Frames written into the current set
        self._episode_boundaries: List[int] = []  # cumulative frame offsets per episode
        self.include_rewards = bool(include_rewards)
    
    def _get_compression_filter(self, compression: str):
        """Get HDF5 compression filter."""
        if compression is None or compression.lower() == "none":
            return None
        elif compression.lower() == "lz4":
            # LZ4 with compression level 9 (fast + good ratio)
            return hdf5plugin.LZ4(nbytes=0)
        elif compression.lower() == "lzf":
            return "lzf"  # Built-in HDF5 compression
        elif compression.lower() == "gzip":
            return "gzip"
        else:
            raise ValueError(f"Unknown compression: {compression}")
    
    def _open_new_set_file(self):
        """Open a new HDF5 file for the next episode set."""
        if self._current_file is not None:
            self._finalize_current_file()
        
        self._current_set_idx += 1
        filename = f"{self.run_id}_set{self._current_set_idx:03d}.h5"
        filepath = os.path.join(self.output_dir, filename)
        
        self._current_file = h5py.File(filepath, "w")
        self._episodes_in_current_set = 0
        self._current_size = 0
        self._episode_boundaries = [0]

        # Write metadata as file attributes
        self._current_file.attrs["run_id"] = self.run_id
        self._current_file.attrs["set_idx"] = self._current_set_idx
        self._current_file.attrs["K"] = self.K
        self._current_file.attrs["N_future"] = self.N_future
        self._current_file.attrs["M"] = self.M
        self._current_file.attrs["C"] = self.C
        self._current_file.attrs["H"] = self.H
        self._current_file.attrs["W"] = self.W
        self._current_file.attrs["version"] = "2.1" if self.include_rewards else "2.0"
        self._current_file.attrs["norms_version"] = 1
        if self.include_rewards:
            self._current_file.attrs["reward_stats_version"] = 1
        self._current_file.attrs["created_at"] = datetime.utcnow().isoformat()

        # Pre-create extendable datasets (unlimited along sample dimension)
        chunk_n = max(1, int(self.chunk_size) if self.chunk_size is not None else 1)
        self._current_file.create_dataset(
            "frame_id",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int64,
            chunks=(chunk_n,),
            compression=self.compression,
        )
        self._current_file.create_dataset(
            "episode_id",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int16,
            chunks=(chunk_n,),
            compression=self.compression,
        )
        self._current_file.create_dataset(
            "ego_vec",
            shape=(0, 14),
            maxshape=(None, 14),
            dtype=np.float32,
            chunks=(chunk_n, 14),
            compression=self.compression,
        )
        self._current_file.create_dataset(
            "bev",
            shape=(0, self.C, self.H, self.W),
            maxshape=(None, self.C, self.H, self.W),
            dtype=np.float32,
            chunks=(chunk_n, self.C, self.H, self.W),
            compression=self.compression,
        )
        self._current_file.create_dataset(
            "route",
            shape=(0, self.K, 2),
            maxshape=(None, self.K, 2),
            dtype=np.float32,
            chunks=(chunk_n, self.K, 2),
            compression=self.compression,
        )
        self._current_file.create_dataset(
            "objects",
            shape=(0, self.M, 11),
            maxshape=(None, self.M, 11),
            dtype=np.float32,
            chunks=(chunk_n, self.M, 11),
            compression=self.compression,
        )
        self._current_file.create_dataset(
            "object_mask",
            shape=(0, self.M),
            maxshape=(None, self.M),
            dtype=np.float32,
            chunks=(chunk_n, self.M),
            compression=self.compression,
        )
        self._current_file.create_dataset(
            "future_xy",
            shape=(0, self.N_future, 2),
            maxshape=(None, self.N_future, 2),
            dtype=np.float32,
            chunks=(chunk_n, self.N_future, 2),
            compression=self.compression,
        )
        self._current_file.create_dataset(
            "future_v",
            shape=(0, self.N_future),
            maxshape=(None, self.N_future),
            dtype=np.float32,
            chunks=(chunk_n, self.N_future),
            compression=self.compression,
        )
        self._current_file.create_dataset(
            "future_mask",
            shape=(0, self.N_future),
            maxshape=(None, self.N_future),
            dtype=np.float32,
            chunks=(chunk_n, self.N_future),
            compression=self.compression,
        )
        if self.include_rewards:
            for name in [
                "reward_raw",
                "reward_clipped",
                "reward_normalized",
                "reward_progress",
                "reward_collision",
                "reward_offroad",
                "reward_violation",
                "reward_comfort",
                "reward_completion",
                "reward_mean",
                "reward_std",
            ]:
                self._current_file.create_dataset(
                    name,
                    shape=(0,),
                    maxshape=(None,),
                    dtype=np.float32,
                    chunks=(chunk_n,),
                    compression=self.compression,
                )
            self._current_file.create_dataset(
                "noise_injected",
                shape=(0,),
                maxshape=(None,),
                dtype=np.bool_,
                chunks=(chunk_n,),
                compression=self.compression,
            )

        return filepath
    
    def append_episode(self, frames: List[Dict]):
        """
        Append an episode's frames to the current set.
        
        Args:
            frames: List of frame dicts, each containing:
                - frame_id: int
                - ego_vec: [14] array
                - bev: [C, H, W] array
                - route: [K, 2] array
                - objects: [M, 11] array
                - object_mask: [M] array
                - future_xy: [N_fut, 2] array
                - future_v: [N_fut] array
                - future_mask: [N_fut] array
        """
        if not frames:
            # Empty episode - don't increment counter, skip it
            print(f"Warning: Skipping empty episode (no valid frames)")
            return
        # Ensure a set file is open
        if self._current_file is None or self._episodes_in_current_set >= self.episodes_per_set:
            self._open_new_set_file()

        episode_id = self._episodes_in_current_set
        batch_size = max(1, min(int(self.chunk_size) if self.chunk_size is not None else 128, len(frames)))
        idx = 0
        while idx < len(frames):
            batch = frames[idx: idx + batch_size]
            b = len(batch)
            start = self._current_size
            end = start + b

            f = self._current_file
            # Extend datasets
            f["frame_id"].resize((end,))
            f["episode_id"].resize((end,))
            f["ego_vec"].resize((end, 14))
            f["bev"].resize((end, self.C, self.H, self.W))
            f["route"].resize((end, self.K, 2))
            f["objects"].resize((end, self.M, 11))
            f["object_mask"].resize((end, self.M))
            f["future_xy"].resize((end, self.N_future, 2))
            f["future_v"].resize((end, self.N_future))
            f["future_mask"].resize((end, self.N_future))
            if self.include_rewards:
                f["reward_raw"].resize((end,))
                f["reward_clipped"].resize((end,))
                f["reward_normalized"].resize((end,))
                f["reward_progress"].resize((end,))
                f["reward_collision"].resize((end,))
                f["reward_offroad"].resize((end,))
                f["reward_violation"].resize((end,))
                f["reward_comfort"].resize((end,))
                f["reward_completion"].resize((end,))
                f["reward_mean"].resize((end,))
                f["reward_std"].resize((end,))
                f["noise_injected"].resize((end,))

            # Materialize numpy batches and write
            f["frame_id"][start:end] = np.array([fr["frame_id"] for fr in batch], dtype=np.int64)
            f["episode_id"][start:end] = episode_id
            f["ego_vec"][start:end] = np.stack([fr["ego_vec"] for fr in batch]).astype(np.float32)
            f["bev"][start:end] = np.stack([fr["bev"] for fr in batch]).astype(np.float32)
            f["route"][start:end] = np.stack([fr["route"] for fr in batch]).astype(np.float32)
            f["objects"][start:end] = np.stack([fr["objects"] for fr in batch]).astype(np.float32)
            f["object_mask"][start:end] = np.stack([fr["object_mask"] for fr in batch]).astype(np.float32)
            f["future_xy"][start:end] = np.stack([fr["future_xy"] for fr in batch]).astype(np.float32)
            f["future_v"][start:end] = np.stack([fr["future_v"] for fr in batch]).astype(np.float32)
            f["future_mask"][start:end] = np.stack([fr["future_mask"] for fr in batch]).astype(np.float32)

            if self.include_rewards:
                # Rewards and noise (default zeros if missing)
                def _get(fr: Dict, key: str, default: float = 0.0) -> float:
                    return float(fr.get(key, default))

                f["reward_raw"][start:end] = np.array([_get(fr, "reward_raw") for fr in batch], dtype=np.float32)
                f["reward_clipped"][start:end] = np.array([_get(fr, "reward_clipped") for fr in batch], dtype=np.float32)
                f["reward_normalized"][start:end] = np.array(
                    [_get(fr, "reward_normalized") for fr in batch], dtype=np.float32
                )
                f["reward_progress"][start:end] = np.array(
                    [_get(fr, "reward_progress") for fr in batch], dtype=np.float32
                )
                f["reward_collision"][start:end] = np.array(
                    [_get(fr, "reward_collision") for fr in batch], dtype=np.float32
                )
                f["reward_offroad"][start:end] = np.array(
                    [_get(fr, "reward_offroad") for fr in batch], dtype=np.float32
                )
                f["reward_violation"][start:end] = np.array(
                    [_get(fr, "reward_violation") for fr in batch], dtype=np.float32
                )
                f["reward_comfort"][start:end] = np.array(
                    [_get(fr, "reward_comfort") for fr in batch], dtype=np.float32
                )
                f["reward_completion"][start:end] = np.array(
                    [_get(fr, "reward_completion") for fr in batch], dtype=np.float32
                )
                f["reward_mean"][start:end] = np.array([_get(fr, "reward_mean") for fr in batch], dtype=np.float32)
                f["reward_std"][start:end] = np.array([_get(fr, "reward_std") for fr in batch], dtype=np.float32)
                f["noise_injected"][start:end] = np.array(
                    [bool(fr.get("noise_injected", False)) for fr in batch], dtype=np.bool_
                )

            self._current_size = end
            idx += b

        # Record episode boundary and count
        self._episode_boundaries.append(self._current_size)
        self._episodes_in_current_set += 1

        # Close the set once the configured episode count is reached
        if self._episodes_in_current_set >= self.episodes_per_set:
            self._finalize_current_file()

    def _finalize_current_file(self):
        """Write boundary metadata and close the current set file."""
        if self._current_file is None:
            return
        try:
            self._current_file.attrs["episode_boundaries"] = np.array(self._episode_boundaries, dtype=np.int32)
            self._current_file.attrs["n_episodes"] = self._episodes_in_current_set
        finally:
            try:
                self._current_file.close()
            finally:
                self._current_file = None
                self._episodes_in_current_set = 0
                self._current_size = 0
                self._episode_boundaries = []
    
    def close(self):
        """Flush any remaining data and close."""
        # Finalize partial set if open
        if self._current_file is not None:
            self._finalize_current_file()

