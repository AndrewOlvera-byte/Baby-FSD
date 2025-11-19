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
        self._buffer: List[Dict] = []  # Buffer for current episode-set
        
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
            self._current_file.close()
        
        self._current_set_idx += 1
        filename = f"{self.run_id}_set{self._current_set_idx:03d}.h5"
        filepath = os.path.join(self.output_dir, filename)
        
        self._current_file = h5py.File(filepath, "w")
        self._episodes_in_current_set = 0
        self._buffer.clear()
        
        # Write metadata as file attributes
        self._current_file.attrs["run_id"] = self.run_id
        self._current_file.attrs["set_idx"] = self._current_set_idx
        self._current_file.attrs["K"] = self.K
        self._current_file.attrs["N_future"] = self.N_future
        self._current_file.attrs["M"] = self.M
        self._current_file.attrs["C"] = self.C
        self._current_file.attrs["H"] = self.H
        self._current_file.attrs["W"] = self.W
        self._current_file.attrs["version"] = "2.0"
        self._current_file.attrs["norms_version"] = 1
        self._current_file.attrs["created_at"] = datetime.utcnow().isoformat()
        
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
        
        # Buffer frames with episode marker
        episode_id = self._episodes_in_current_set
        for frame in frames:
            frame_copy = frame.copy()
            frame_copy["episode_id"] = episode_id
            self._buffer.append(frame_copy)
        
        self._episodes_in_current_set += 1
        
        # Flush if we've accumulated enough episodes
        if self._episodes_in_current_set >= self.episodes_per_set:
            self._flush_set()
    
    def _flush_set(self):
        """Flush buffered frames to HDF5 file."""
        # Early return if buffer is empty or invalid
        if not self._buffer or len(self._buffer) == 0:
            return
        
        N = len(self._buffer)
        
        # Double-check N is positive (defensive programming)
        if N <= 0:
            print(f"Warning: Skipping flush of empty buffer (N={N})")
            self._buffer.clear()
            self._episodes_in_current_set = 0
            return
        
        # Close any existing file before creating a new one
        # This ensures we don't try to write to the same file twice
        if self._current_file is not None:
            try:
                self._current_file.close()
            except:
                pass
            self._current_file = None
        
        # Now open new file for this set
        self._open_new_set_file()
        
        # Stack all frames into arrays
        frame_ids = np.array([f["frame_id"] for f in self._buffer], dtype=np.int64)
        episode_ids = np.array([f["episode_id"] for f in self._buffer], dtype=np.int16)
        
        # Validate that arrays were created successfully
        if len(frame_ids) == 0 or len(episode_ids) == 0:
            print(f"Warning: Failed to create frame arrays from buffer, skipping flush")
            self._buffer.clear()
            self._episodes_in_current_set = 0
            if self._current_file is not None:
                self._current_file.close()
                self._current_file = None
            return
        
        # Pre-allocate arrays
        ego_vecs = np.zeros((N, 14), dtype=np.float32)
        bevs = np.zeros((N, self.C, self.H, self.W), dtype=np.float32)
        routes = np.zeros((N, self.K, 2), dtype=np.float32)
        objects = np.zeros((N, self.M, 11), dtype=np.float32)
        object_masks = np.zeros((N, self.M), dtype=np.float32)
        future_xys = np.zeros((N, self.N_future, 2), dtype=np.float32)
        future_vs = np.zeros((N, self.N_future), dtype=np.float32)
        future_masks = np.zeros((N, self.N_future), dtype=np.float32)
        
        # Fill arrays
        for i, frame in enumerate(self._buffer):
            ego_vecs[i] = frame["ego_vec"]
            bevs[i] = frame["bev"]
            routes[i] = frame["route"]
            objects[i] = frame["objects"]
            object_masks[i] = frame["object_mask"]
            future_xys[i] = frame["future_xy"]
            future_vs[i] = frame["future_v"]
            future_masks[i] = frame["future_mask"]
        
        # Define chunking for optimal access patterns
        # Chunk along sample dimension with full feature dimensions
        # Guard against invalid (zero/negative) chunk sizes
        if self.chunk_size is None or self.chunk_size <= 0:
            chunk_n = N
        else:
            chunk_n = min(self.chunk_size, N)
        # HDF5 requires all chunk dimensions to be strictly positive
        chunk_n = max(1, int(chunk_n))
        
        # Write datasets
        self._current_file.create_dataset(
            "frame_id", data=frame_ids, dtype=np.int64,
            chunks=(chunk_n,), compression=self.compression
        )
        self._current_file.create_dataset(
            "episode_id", data=episode_ids, dtype=np.int16,
            chunks=(chunk_n,), compression=self.compression
        )
        self._current_file.create_dataset(
            "ego_vec", data=ego_vecs, dtype=np.float32,
            chunks=(chunk_n, 14), compression=self.compression
        )
        self._current_file.create_dataset(
            "bev", data=bevs, dtype=np.float32,
            chunks=(chunk_n, self.C, self.H, self.W), compression=self.compression
        )
        self._current_file.create_dataset(
            "route", data=routes, dtype=np.float32,
            chunks=(chunk_n, self.K, 2), compression=self.compression
        )
        self._current_file.create_dataset(
            "objects", data=objects, dtype=np.float32,
            chunks=(chunk_n, self.M, 11), compression=self.compression
        )
        self._current_file.create_dataset(
            "object_mask", data=object_masks, dtype=np.float32,
            chunks=(chunk_n, self.M), compression=self.compression
        )
        self._current_file.create_dataset(
            "future_xy", data=future_xys, dtype=np.float32,
            chunks=(chunk_n, self.N_future, 2), compression=self.compression
        )
        self._current_file.create_dataset(
            "future_v", data=future_vs, dtype=np.float32,
            chunks=(chunk_n, self.N_future), compression=self.compression
        )
        self._current_file.create_dataset(
            "future_mask", data=future_masks, dtype=np.float32,
            chunks=(chunk_n, self.N_future), compression=self.compression
        )
        
        # Store episode boundaries as attribute
        episode_boundaries = []
        current_ep = -1
        for i, ep_id in enumerate(episode_ids):
            if ep_id != current_ep:
                episode_boundaries.append(i)
                current_ep = ep_id
        episode_boundaries.append(N)  # End boundary
        self._current_file.attrs["episode_boundaries"] = np.array(episode_boundaries, dtype=np.int32)
        self._current_file.attrs["n_episodes"] = self._episodes_in_current_set
        
        # Close file and reset for next set
        self._current_file.close()
        self._current_file = None
        self._buffer.clear()
        self._episodes_in_current_set = 0
    
    def close(self):
        """Flush any remaining data and close."""
        # Only flush if we have buffered frames AND haven't reached the threshold
        # If we've reached episodes_per_set, flush was already called
        if self._buffer and len(self._buffer) > 0 and self._episodes_in_current_set < self.episodes_per_set:
            print(f"Flushing partial set ({self._episodes_in_current_set} episodes, {len(self._buffer)} frames)")
            self._flush_set()
        
        # Clean up any open file handle
        if self._current_file is not None:
            try:
                self._current_file.close()
            except:
                pass
            self._current_file = None
        
        # Final cleanup
        self._buffer.clear()
        self._episodes_in_current_set = 0

