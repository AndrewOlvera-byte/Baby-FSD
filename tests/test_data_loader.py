"""
Tests for HDF5-backed BC dataset basic functionality.
"""

import io
import os
import tempfile
import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
hdf5plugin = pytest.importorskip("hdf5plugin")

from data.hdf5_writer import HDF5EpisodeSetWriter


def test_hdf5_episode_set_write_read(tmp_path):
    """Test that we can write and read HDF5 episode sets."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    
    # Create writer
    writer = HDF5EpisodeSetWriter(
        output_dir=str(run_dir),
        run_id="test_run",
        episodes_per_set=2,
        K=16,
        N_future=6,
        M=32,
        C=18,
        H=100,
        W=150,
        compression="lz4",
        chunk_size=50,
    )
    
    # Create 2 episodes with 3 frames each
    episode1 = []
    for i in range(3):
        frame = {
            "frame_id": i,
            "ego_vec": np.random.randn(14).astype(np.float32),
            "bev": np.random.randn(18, 100, 150).astype(np.float32),
            "route": np.random.randn(16, 2).astype(np.float32),
            "objects": np.random.randn(32, 11).astype(np.float32),
            "object_mask": np.random.randint(0, 2, (32,)).astype(np.float32),
            "future_xy": np.random.randn(6, 2).astype(np.float32),
            "future_v": np.random.rand(6).astype(np.float32),
            "future_mask": np.ones((6,), dtype=np.float32),
        }
        episode1.append(frame)
    
    episode2 = []
    for i in range(3, 6):
        frame = {
            "frame_id": i,
            "ego_vec": np.random.randn(14).astype(np.float32),
            "bev": np.random.randn(18, 100, 150).astype(np.float32),
            "route": np.random.randn(16, 2).astype(np.float32),
            "objects": np.random.randn(32, 11).astype(np.float32),
            "object_mask": np.random.randint(0, 2, (32,)).astype(np.float32),
            "future_xy": np.random.randn(6, 2).astype(np.float32),
            "future_v": np.random.rand(6).astype(np.float32),
            "future_mask": np.ones((6,), dtype=np.float32),
        }
        episode2.append(frame)
    
    # Write episodes
    writer.append_episode(episode1)
    writer.append_episode(episode2)
    writer.close()
    
    # Check file was created
    h5_files = list(run_dir.glob("*.h5"))
    assert len(h5_files) == 1
    
    # Read back and verify
    with h5py.File(h5_files[0], "r") as f:
        assert len(f["frame_id"]) == 6
        assert f["frame_id"][0] == 0
        assert f["frame_id"][5] == 5
        
        # Check shapes
        assert f["ego_vec"].shape == (6, 14)
        assert f["bev"].shape == (6, 18, 100, 150)
        assert f["route"].shape == (6, 16, 2)
        assert f["objects"].shape == (6, 32, 11)
        assert f["future_xy"].shape == (6, 6, 2)
        
        # Check episode IDs
        assert f["episode_id"][0] == 0
        assert f["episode_id"][2] == 0
        assert f["episode_id"][3] == 1
        assert f["episode_id"][5] == 1
        
        # Check metadata
        assert f.attrs["K"] == 16
        assert f.attrs["N_future"] == 6
        assert f.attrs["M"] == 32
        assert f.attrs["C"] == 18
        assert f.attrs["H"] == 100
        assert f.attrs["W"] == 150


def test_hdf5_multi_set_files(tmp_path):
    """Test that multiple episode-set files are created when threshold is reached."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    
    writer = HDF5EpisodeSetWriter(
        output_dir=str(run_dir),
        run_id="test_run",
        episodes_per_set=2,  # 2 episodes per file
        K=16,
        N_future=6,
        M=32,
        C=18,
        H=100,
        W=150,
        compression="lz4",
    )
    
    # Write 5 episodes (should create 2 files + 1 partial)
    for ep in range(5):
        episode = []
        for i in range(2):  # 2 frames per episode
            frame = {
                "frame_id": ep * 2 + i,
                "ego_vec": np.random.randn(14).astype(np.float32),
                "bev": np.random.randn(18, 100, 150).astype(np.float32),
                "route": np.random.randn(16, 2).astype(np.float32),
                "objects": np.random.randn(32, 11).astype(np.float32),
                "object_mask": np.ones((32,), dtype=np.float32),
                "future_xy": np.random.randn(6, 2).astype(np.float32),
                "future_v": np.random.rand(6).astype(np.float32),
                "future_mask": np.ones((6,), dtype=np.float32),
            }
            episode.append(frame)
        writer.append_episode(episode)
    
    writer.close()
    
    # Check files were created
    h5_files = sorted(list(run_dir.glob("*.h5")))
    assert len(h5_files) == 3  # 2 full sets + 1 partial
    
    # Verify set 1 has 4 frames (2 episodes × 2 frames)
    with h5py.File(h5_files[0], "r") as f:
        assert len(f["frame_id"]) == 4
        assert f.attrs["n_episodes"] == 2
    
    # Verify set 2 has 4 frames
    with h5py.File(h5_files[1], "r") as f:
        assert len(f["frame_id"]) == 4
        assert f.attrs["n_episodes"] == 2
    
    # Verify set 3 has 2 frames (1 episode × 2 frames)
    with h5py.File(h5_files[2], "r") as f:
        assert len(f["frame_id"]) == 2
        assert f.attrs["n_episodes"] == 1
