"""
Tests for PyTorch BC trajectory dataset and dataloader (HDF5 backend).

Validates shapes, dtypes, value ranges, and batch consistency.
"""

import os
import pytest
import torch
import numpy as np
import logging
import tempfile

from data.torch_dataset import BCTrajectoryDataset, create_bc_dataloader
from data.hdf5_writer import HDF5EpisodeSetWriter
from data.norms import (
    V_MAX, ACCEL_MAX, SPATIAL_MAX, BEV_VEL_MAX, BEV_SPEED_LIMIT_MAX
)

h5py = pytest.importorskip("h5py")
hdf5plugin = pytest.importorskip("hdf5plugin")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def create_test_hdf5_dataset(tmp_path, n_episodes=2, frames_per_ep=10):
    """Helper to create a test HDF5 dataset."""
    run_dir = tmp_path / "test_run"
    run_dir.mkdir()
    
    K, N, M = 32, 12, 64
    C, H, W = 18, 150, 200
    
    writer = HDF5EpisodeSetWriter(
        output_dir=str(run_dir),
        run_id="test",
        episodes_per_set=5,
        K=K,
        N_future=N,
        M=M,
        C=C,
        H=H,
        W=W,
        compression="lz4",
    )
    
    all_frames = []
    for ep in range(n_episodes):
        episode = []
        for i in range(frames_per_ep):
            frame_id = ep * frames_per_ep + i
            frame = {
                "frame_id": frame_id,
                "ego_vec": np.random.randn(14).astype(np.float32) * 0.5,  # Small values
                "bev": np.random.rand(C, H, W).astype(np.float32),  # [0, 1]
                "route": np.random.randn(K, 2).astype(np.float32) * 10.0,  # meters
                "objects": np.random.randn(M, 11).astype(np.float32) * 5.0,
                "object_mask": (np.random.rand(M) > 0.5).astype(np.float32),
                "future_xy": np.random.randn(N, 2).astype(np.float32) * 10.0,
                "future_v": np.random.rand(N).astype(np.float32) * 20.0,
                "future_mask": np.ones((N,), dtype=np.float32),
            }
            episode.append(frame)
            all_frames.append(frame)
        writer.append_episode(episode)
    
    writer.close()
    return str(run_dir), len(all_frames)


@pytest.fixture(scope="module")
def test_dataset(tmp_path_factory):
    """Create a test dataset once for all tests."""
    tmp_path = tmp_path_factory.mktemp("data")
    run_dir, n_frames = create_test_hdf5_dataset(tmp_path, n_episodes=3, frames_per_ep=20)
    logger.info(f"Created test dataset with {n_frames} frames in {run_dir}")
    return run_dir, n_frames


class TestBCTrajectoryDataset:
    """Test BCTrajectoryDataset."""
    
    def test_dataset_loads(self, test_dataset):
        """Test that dataset loads without errors."""
        run_dir, n_frames = test_dataset
        dataset = BCTrajectoryDataset(run_dir, future_horizon=12, route_points=32, max_objects=64)
        assert len(dataset) == n_frames
        logger.info(f"Dataset loaded: {len(dataset)} samples")
    
    def test_sample_structure(self, test_dataset):
        """Test that sample has correct structure."""
        run_dir, _ = test_dataset
        dataset = BCTrajectoryDataset(run_dir)
        sample = dataset[0]
        
        # Check keys
        expected_keys = {
            "frame_id", "ego_vec", "bev", "route", 
            "objects", "object_mask",
            "future_xy", "future_v", "future_mask"
        }
        assert set(sample.keys()) == expected_keys
        
        # Check types
        assert isinstance(sample["frame_id"], int)
        assert isinstance(sample["ego_vec"], torch.Tensor)
        assert isinstance(sample["bev"], torch.Tensor)
        assert isinstance(sample["route"], torch.Tensor)
        assert isinstance(sample["objects"], torch.Tensor)
        assert isinstance(sample["object_mask"], torch.Tensor)
        assert isinstance(sample["future_xy"], torch.Tensor)
        assert isinstance(sample["future_v"], torch.Tensor)
        assert isinstance(sample["future_mask"], torch.Tensor)
    
    def test_bev_shape(self, test_dataset):
        """Test BEV tensor shape."""
        run_dir, _ = test_dataset
        dataset = BCTrajectoryDataset(run_dir)
        sample = dataset[0]
        bev = sample["bev"]
        
        assert bev.ndim == 3
        assert bev.shape[0] == 18  # channels
        assert bev.shape[1] > 0    # height
        assert bev.shape[2] > 0    # width
        assert bev.dtype == torch.float32
    
    def test_bev_values(self, test_dataset):
        """Test BEV tensor value ranges."""
        run_dir, _ = test_dataset
        dataset = BCTrajectoryDataset(run_dir)
        sample = dataset[0]
        bev = sample["bev"]
        
        # Should be finite
        assert torch.isfinite(bev).all()
        
        # Binary channels (0-14) should be in [0, 1]
        assert bev[0:15].min() >= -0.1  # Allow small tolerance
        assert bev[0:15].max() <= 1.1
    
    def test_route_shape(self, test_dataset):
        """Test route tensor shape."""
        run_dir, _ = test_dataset
        dataset = BCTrajectoryDataset(run_dir)
        sample = dataset[0]
        route = sample["route"]
        
        assert route.shape == (32, 2)
        assert route.dtype == torch.float32
        assert torch.isfinite(route).all()
    
    def test_multiple_samples(self, test_dataset):
        """Test that we can load multiple samples."""
        run_dir, _ = test_dataset
        dataset = BCTrajectoryDataset(run_dir)
        n_samples = min(10, len(dataset))
        
        for i in range(n_samples):
            sample = dataset[i]
            assert sample is not None
            assert "bev" in sample
    
    def test_frame_id_consistency(self, test_dataset):
        """Test that frame IDs are consistent."""
        run_dir, _ = test_dataset
        dataset = BCTrajectoryDataset(run_dir)
        
        sample1 = dataset[0]
        sample2 = dataset[1]
        
        assert sample1["frame_id"] != sample2["frame_id"]
        assert isinstance(sample1["frame_id"], int)
        assert isinstance(sample2["frame_id"], int)
    
    def test_split_train_val(self, test_dataset):
        """Test train/val splitting."""
        run_dir, n_frames = test_dataset
        
        train_ds = BCTrajectoryDataset(run_dir, split="train", val_ratio=0.2)
        val_ds = BCTrajectoryDataset(run_dir, split="val", val_ratio=0.2)
        
        # Check split sizes
        expected_train = int(n_frames * 0.8)
        expected_val = n_frames - expected_train
        
        assert len(train_ds) == expected_train
        assert len(val_ds) == expected_val
        
        # Check no overlap
        train_sample = train_ds[0]
        val_sample = val_ds[0]
        assert train_sample["frame_id"] != val_sample["frame_id"]


class TestBCDataLoader:
    """Test create_bc_dataloader function."""
    
    def test_dataloader_creation(self, test_dataset):
        """Test that dataloader can be created."""
        run_dir, _ = test_dataset
        loader = create_bc_dataloader(
            run_dir,
            batch_size=4,
            shuffle=False,
            num_workers=0,
        )
        
        assert loader is not None
        assert isinstance(loader, torch.utils.data.DataLoader)
    
    def test_dataloader_iteration(self, test_dataset):
        """Test that we can iterate through dataloader."""
        run_dir, _ = test_dataset
        loader = create_bc_dataloader(
            run_dir,
            batch_size=4,
            shuffle=False,
            num_workers=0,
        )
        
        batch = next(iter(loader))
        assert batch is not None
        assert "bev" in batch
        assert "ego_vec" in batch
    
    def test_batch_shapes(self, test_dataset):
        """Test that batch has correct shapes."""
        run_dir, _ = test_dataset
        batch_size = 4
        loader = create_bc_dataloader(
            run_dir,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        
        batch = next(iter(loader))
        
        # BEV should be (B, C, H, W)
        assert batch["bev"].ndim == 4
        assert batch["bev"].shape[0] <= batch_size
        assert batch["bev"].shape[1] == 18
        
        # Route should be (B, K, 2)
        assert batch["route"].shape[1:] == (32, 2)
        
        # Futures should be (B, N, 2)
        assert batch["future_xy"].shape[1:] == (12, 2)
        
        # Ego vec should be (B, d_ego)
        assert batch["ego_vec"].shape[1:] == (14,)
        
        # Objects should be (B, M, d_obj)
        assert batch["objects"].shape[1:] == (64, 11)
    
    def test_batch_consistency(self, test_dataset):
        """Test that all tensors in batch have consistent batch dimension."""
        run_dir, _ = test_dataset
        loader = create_bc_dataloader(
            run_dir,
            batch_size=4,
            shuffle=False,
            num_workers=0,
        )
        
        batch = next(iter(loader))
        batch_size = batch["bev"].shape[0]
        
        # All tensors should have same batch size
        assert batch["route"].shape[0] == batch_size
        assert batch["future_xy"].shape[0] == batch_size
        assert batch["future_v"].shape[0] == batch_size
        assert batch["ego_vec"].shape[0] == batch_size
        assert batch["objects"].shape[0] == batch_size
        assert batch["object_mask"].shape[0] == batch_size
    
    def test_multiple_batches(self, test_dataset):
        """Test that we can iterate through multiple batches."""
        run_dir, _ = test_dataset
        loader = create_bc_dataloader(
            run_dir,
            batch_size=4,
            shuffle=False,
            num_workers=0,
        )
        
        n_batches = min(3, len(loader))
        batches = []
        
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            batches.append(batch)
        
        assert len(batches) == n_batches
        
        # All batches should have same structure
        for batch in batches:
            assert "bev" in batch


class TestObjectTokens:
    """Test object tokens and masking."""
    
    def test_objects_shape(self, test_dataset):
        """Test objects tensor shape."""
        run_dir, _ = test_dataset
        dataset = BCTrajectoryDataset(run_dir)
        sample = dataset[0]
        objects = sample["objects"]
        object_mask = sample["object_mask"]
        
        assert objects.shape == (64, 11)
        assert object_mask.shape == (64,)
        assert objects.dtype == torch.float32
        assert object_mask.dtype == torch.float32
    
    def test_object_mask_validity(self, test_dataset):
        """Test object mask values."""
        run_dir, _ = test_dataset
        dataset = BCTrajectoryDataset(run_dir)
        sample = dataset[0]
        object_mask = sample["object_mask"]
        
        # Mask should be binary (0 or 1)
        assert ((object_mask == 0.0) | (object_mask == 1.0)).all()
        
        # Should have at least some valid objects or be empty
        assert object_mask.sum() >= 0


class TestNormalization:
    """Test data normalization."""
    
    def test_ego_vec_normalization(self, test_dataset):
        """Test ego vector is normalized."""
        run_dir, _ = test_dataset
        dataset = BCTrajectoryDataset(run_dir)
        sample = dataset[0]
        ego_vec = sample["ego_vec"]
        
        assert ego_vec.ndim == 1
        assert ego_vec.shape[0] == 14
        
        # Values should be in reasonable normalized range
        assert ego_vec.abs().max() <= 2.0  # Allow some margin
    
    def test_route_normalization(self, test_dataset):
        """Test route points are normalized."""
        run_dir, _ = test_dataset
        dataset = BCTrajectoryDataset(run_dir)
        sample = dataset[0]
        route = sample["route"]
        
        assert route.shape == (32, 2)
        # After normalization should be in [-1, 1]
        assert route.abs().max() <= 1.0 + 1e-4
    
    def test_futures_normalization(self, test_dataset):
        """Test future waypoints and speeds are normalized."""
        run_dir, _ = test_dataset
        dataset = BCTrajectoryDataset(run_dir)
        sample = dataset[0]
        future_xy = sample["future_xy"]
        future_v = sample["future_v"]
        
        # Waypoints should be in [-1, 1]
        assert future_xy.abs().max() <= 1.0 + 1e-4
        
        # Speeds should be in [0, 1]
        assert future_v.min() >= -1e-4
        assert future_v.max() <= 1.0 + 1e-4


class TestFutureMasking:
    """Test future masking for incomplete episodes."""
    
    def test_future_mask_shape(self, test_dataset):
        """Test future mask has correct shape."""
        run_dir, _ = test_dataset
        dataset = BCTrajectoryDataset(run_dir)
        sample = dataset[0]
        future_mask = sample["future_mask"]
        
        assert future_mask.shape == (12,)
        assert future_mask.dtype == torch.float32
    
    def test_future_mask_values(self, test_dataset):
        """Test future mask has valid values."""
        run_dir, _ = test_dataset
        dataset = BCTrajectoryDataset(run_dir)
        sample = dataset[0]
        future_mask = sample["future_mask"]
        
        # Mask should be binary (0 or 1)
        assert ((future_mask == 0.0) | (future_mask == 1.0)).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
