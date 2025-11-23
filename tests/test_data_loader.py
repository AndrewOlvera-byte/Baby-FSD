"""
Tests for BC data IO: HDF5 writer and WebDataset loader.
"""

import io
import os
import tempfile
import time
import logging
from pathlib import Path
import numpy as np
import pytest
import torch
from typing import Optional

h5py = pytest.importorskip("h5py")
hdf5plugin = pytest.importorskip("hdf5plugin")
webdataset = pytest.importorskip("webdataset")

from data.hdf5_writer import HDF5EpisodeSetWriter
from data.torch_dataset import BCWebDataset, create_bc_dataloader

LOG = logging.getLogger(__name__)
DEFAULT_WDS_DIR = Path(__file__).resolve().parents[1] / "data" / "BC_v2" / "run-20251121-053341-5EP-DEBUG-wds"
WDS_ENV_VAR = "BC_WDS_RUN_DIR"


def _locate_wds_run_dir() -> Optional[Path]:
    """Find a real WebDataset run dir; prefer env override, fall back to repo default."""
    env_path = os.environ.get(WDS_ENV_VAR)
    candidates = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(DEFAULT_WDS_DIR)
    for path in candidates:
        if path and path.exists():
            return path
    return None


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


class TestRealWebDataset:
    """Smoke tests against a real WebDataset shard directory (optional)."""

    def _get_run_dir(self) -> Path:
        run_dir = _locate_wds_run_dir()
        if run_dir is None:
            pytest.skip(f"No WebDataset run dir found; set {WDS_ENV_VAR} or place shards at {DEFAULT_WDS_DIR}")
        return run_dir

    def test_real_wds_loader_shapes(self):
        """Ensure real shards load and shapes match metadata."""
        run_dir = self._get_run_dir()

        dataset = BCWebDataset(str(run_dir), split="train", batch_size=8)
        loader = create_bc_dataloader(
            run_dir=str(run_dir),
            batch_size=8,
            shuffle=True,
            num_workers=0,  # deterministic, avoids worker scatter in CI
            split="train",
            val_ratio=dataset.val_ratio,
        )

        batch = next(iter(loader))
        meta = dataset.metadata

        assert batch["bev"].ndim == 4
        assert batch["bev"].shape[1:] == (meta["C"], meta["H"], meta["W"])
        assert batch["route"].shape[1:] == (dataset.route_points, 2)
        assert batch["future_xy"].shape[1:] == (dataset.future_horizon, 2)
        assert batch["objects"].shape[1] == dataset.max_objects
        assert batch["objects"].shape[2] == 11

    def test_real_wds_throughput_smoke(self):
        """Iterate a few batches and log throughput to catch regressions."""
        run_dir = self._get_run_dir()

        batch_size = 32
        max_batches = 3
        loader = create_bc_dataloader(
            run_dir=str(run_dir),
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            prefetch_factor=2,
            persistent_workers=False,
            split="train",
            val_ratio=0.05,
        )

        t0 = time.perf_counter()
        n_batches = 0
        n_samples = 0
        for n_batches, batch in enumerate(loader, start=1):
            n_samples += int(batch["bev"].shape[0])
            if n_batches >= max_batches:
                break
        elapsed = time.perf_counter() - t0

        assert n_samples > 0
        samples_per_sec = n_samples / max(elapsed, 1e-6)
        LOG.info(
            "WebDataset throughput: %d samples across %d batches in %.3fs (%.1f samples/s)",
            n_samples,
            n_batches,
            elapsed,
            samples_per_sec,
        )


def _run_webdataset_smoke():
    """
    Allow running this file directly to smoke-test a real WebDataset with prints.
    
    Uses a hardcoded path inside the container so docker compose runs can just call:
        python /app/tests/test_data_loader.py
    """
    run_dir = Path("/app/data/BC_v2/run-20251121-053341-5EP-DEBUG-wds")
    if not run_dir.exists():
        print(f"[SMOKE] Missing WebDataset dir at {run_dir}")
        return

    print(f"[SMOKE] Using WebDataset run dir: {run_dir}")
    batch_size = 128
    num_workers = 8
    prefetch_factor = 4
    loader = create_bc_dataloader(
        run_dir=str(run_dir),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=True,
        split="train",
        val_ratio=0.05,
    )

    dataset = loader.dataset  # type: ignore[attr-defined]
    meta = dataset.metadata if hasattr(dataset, "metadata") else {}
    print(f"[SMOKE] Metadata: {meta}")

    max_batches = 5
    t0 = time.perf_counter()
    n_batches = 0
    n_samples = 0
    for n_batches, batch in enumerate(loader, start=1):
        bev_shape = tuple(batch["bev"].shape)
        print(f"[SMOKE] Batch {n_batches}: bev {bev_shape}, route {tuple(batch['route'].shape)}")
        n_samples += int(batch["bev"].shape[0])
        if n_batches >= max_batches:
            break
    elapsed = time.perf_counter() - t0
    if n_batches == 0:
        print("[SMOKE] No batches yielded.")
        return

    samples_per_sec = n_samples / max(elapsed, 1e-6)
    print(
        f"[SMOKE] Processed {n_samples} samples across {n_batches} batches in {elapsed:.3f}s "
        f"({samples_per_sec:.1f} samples/s)"
    )


if __name__ == "__main__":
    _run_webdataset_smoke()
