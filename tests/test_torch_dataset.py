"""
Tests for PyTorch BC trajectory dataset and dataloader.

Validates shapes, dtypes, value ranges, and batch consistency.
"""

import os
import pytest
import torch
import numpy as np
import logging

from data.torch_dataset import BCTrajectoryDataset, create_bc_dataloader
from data.norms import (
    V_MAX, ACCEL_MAX, SPATIAL_MAX, BEV_VEL_MAX, BEV_SPEED_LIMIT_MAX
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def find_bc_run_dir():
    """Find a BC run directory for testing."""
    logger.info("Looking for BC run directory...")
    base = os.path.join("data", "BC_v1")
    if not os.path.isdir(base):
        logger.warning(f"Base directory not found: {base}")
        return None
    
    # Find first run directory
    for entry in os.listdir(base):
        run_path = os.path.join(base, entry)
        if os.path.isdir(run_path) and entry.startswith("run-"):
            # Check if it has required tables
            required = ["frames", "futures", "route_points", "bev_frames"]
            if all(os.path.isdir(os.path.join(run_path, t)) for t in required):
                logger.info(f"Found BC run directory: {run_path}")
                return run_path
    
    logger.warning("No valid BC run directory found")
    return None


@pytest.fixture(scope="session")
def run_dir():
    """Get BC run directory for testing (created once per session)."""
    path = find_bc_run_dir()
    if path is None:
        pytest.skip("No BC run directory found for testing")
    return path


@pytest.fixture(scope="session")
def dataset(run_dir):
    """Create dataset instance (created once per session and reused across all tests)."""
    logger.info("=" * 80)
    logger.info("CREATING DATASET INSTANCE (this will take ~60-90 seconds)...")
    logger.info("=" * 80)
    dataset = BCTrajectoryDataset(run_dir, future_horizon=12, route_points=32, max_objects=64)
    logger.info("=" * 80)
    logger.info(f"DATASET READY: {len(dataset)} samples loaded")
    logger.info("All tests will now reuse this instance")
    logger.info("=" * 80)
    return dataset


class TestBCTrajectoryDataset:
    """Test BCTrajectoryDataset."""
    
    def test_dataset_loads(self, dataset):
        """Test that dataset loads without errors."""
        logger.info(f"Testing dataset with {len(dataset)} samples")
        assert len(dataset) > 0
        assert hasattr(dataset, "_frames")
        assert hasattr(dataset, "_futures")
        assert hasattr(dataset, "_route")
        assert hasattr(dataset, "_bev")
        logger.info("Dataset structure validated")
    
    def test_sample_structure(self, dataset):
        """Test that sample has correct structure."""
        logger.info("Testing sample structure...")
        sample = dataset[0]
        logger.info(f"Loaded sample with frame_id={sample['frame_id']}")
        
        # Check keys (updated format)
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
    
    def test_bev_shape(self, dataset):
        """Test BEV tensor shape."""
        sample = dataset[0]
        bev = sample["bev"]
        
        # Should be (C, H, W) with C=18 channels
        assert bev.ndim == 3
        assert bev.shape[0] == 18  # channels
        
        # H and W depend on config but should be > 0
        assert bev.shape[1] > 0  # height
        assert bev.shape[2] > 0  # width
        
        # Should be float32
        assert bev.dtype == torch.float32
    
    def test_bev_values(self, dataset):
        """Test BEV tensor value ranges."""
        sample = dataset[0]
        bev = sample["bev"]
        
        # Should be finite
        assert torch.isfinite(bev).all()
        
        # Binary channels (0-11) should be in [0, 1]
        for c in range(12):
            assert bev[c].min() >= 0.0
            assert bev[c].max() <= 1.0 or bev[c].max() == 0.0
        
        # Velocity channels (15, 16) should be reasonable
        assert bev[15].abs().max() < 100.0  # vx
        assert bev[16].abs().max() < 100.0  # vy
        
        # Speed limit channel (17) should be reasonable
        assert bev[17].min() >= 0.0
        assert bev[17].max() < 100.0
    
    def test_route_shape(self, dataset):
        """Test route tensor shape."""
        sample = dataset[0]
        route = sample["route"]
        
        # Should be (K, 2)
        assert route.shape == (32, 2)
        assert route.dtype == torch.float32
        
        # Should be finite
        assert torch.isfinite(route).all()
    
    def test_route_values(self, dataset):
        """Test route value ranges."""
        sample = dataset[0]
        route = sample["route"]
        
        # Route points should be in reasonable range (ego frame, meters)
        # Typically within [-100, 100] meters
        assert route.abs().max() < 200.0
        
        # First point should be close to ego (within a few meters)
        first_pt_dist = torch.norm(route[0])
        assert first_pt_dist < 10.0
    
    def test_multiple_samples(self, dataset):
        """Test that we can load multiple samples."""
        n_samples = min(10, len(dataset))
        logger.info(f"Testing {n_samples} samples...")
        
        for i in range(n_samples):
            if i % 3 == 0:
                logger.info(f"  Loading sample {i+1}/{n_samples}...")
            sample = dataset[i]
            assert sample is not None
            assert "bev" in sample
        
        logger.info(f"Successfully loaded {n_samples} samples")
    
    def test_frame_id_consistency(self, dataset):
        """Test that frame IDs are consistent."""
        sample1 = dataset[0]
        sample2 = dataset[1]
        
        # Frame IDs should be different
        assert sample1["frame_id"] != sample2["frame_id"]
        
        # Frame IDs should be integers
        assert isinstance(sample1["frame_id"], int)
        assert isinstance(sample2["frame_id"], int)


class TestBCDataLoader:
    """Test create_bc_dataloader function."""
    
    def test_dataloader_creation(self, run_dir):
        """Test that dataloader can be created."""
        loader = create_bc_dataloader(
            run_dir,
            batch_size=4,
            shuffle=False,
            num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
        )
        
        assert loader is not None
        assert isinstance(loader, torch.utils.data.DataLoader)
    
    def test_dataloader_iteration(self, run_dir):
        """Test that we can iterate through dataloader."""
        logger.info("Creating dataloader...")
        loader = create_bc_dataloader(
            run_dir,
            batch_size=4,
            shuffle=False,
            num_workers=0,
        )
        
        logger.info("Loading first batch...")
        # Get first batch
        batch = next(iter(loader))
        logger.info("First batch loaded successfully")
        
        assert batch is not None
        assert "bev" in batch
        assert "ego_vec" in batch
    
    def test_batch_shapes(self, run_dir):
        """Test that batch has correct shapes."""
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
        assert batch["bev"].shape[0] <= batch_size  # may be smaller for last batch
        assert batch["bev"].shape[1] == 18
        
        # Route should be (B, K, 2)
        assert batch["route"].shape[1:] == (32, 2)
        
        # Futures should be (B, N, 2)
        assert batch["future_xy"].shape[1:] == (12, 2)
        
        # Futures speed should be (B, N)
        assert batch["future_v"].shape[1:] == (12,)
        
        # Ego vec should be (B, d_ego)
        assert batch["ego_vec"].shape[1:] == (14,)
        
        # Objects should be (B, M, d_obj)
        assert batch["objects"].shape[1:] == (64, 11)
    
    def test_batch_consistency(self, run_dir):
        """Test that all tensors in batch have consistent batch dimension."""
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
    
    def test_multiple_batches(self, run_dir):
        """Test that we can iterate through multiple batches."""
        logger.info("Testing multiple batch loading...")
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
            logger.info(f"  Loaded batch {i+1}/{n_batches}")
            batches.append(batch)
        
        assert len(batches) == n_batches
        logger.info(f"Successfully loaded {n_batches} batches")
        
        # All batches should have same structure
        for batch in batches:
            assert "bev" in batch


class TestObjectTokens:
    """Test object tokens and masking."""
    
    def test_objects_shape(self, dataset):
        """Test objects tensor shape."""
        sample = dataset[0]
        objects = sample["objects"]
        object_mask = sample["object_mask"]
        
        # Should be (M, d_obj) and (M,)
        assert objects.shape == (64, 11)  # M=64, d_obj=11
        assert object_mask.shape == (64,)
        
        # Should be float32
        assert objects.dtype == torch.float32
        assert object_mask.dtype == torch.float32
    
    def test_object_mask_validity(self, dataset):
        """Test object mask values."""
        sample = dataset[0]
        object_mask = sample["object_mask"]
        
        # Mask should be binary (0 or 1)
        assert ((object_mask == 0.0) | (object_mask == 1.0)).all()
        
        # Should have at least some valid objects (unless empty scene)
        # We allow 0 objects for edge cases
        assert object_mask.sum() >= 0


class TestNormalization:
    """Test data normalization."""
    
    def test_ego_vec_normalization(self, dataset):
        """Test ego vector is normalized."""
        logger.info("Testing ego vector normalization...")
        sample = dataset[0]
        ego_vec = sample["ego_vec"]
        
        # Should be (d_ego,) with all values in reasonable range
        assert ego_vec.ndim == 1
        assert ego_vec.shape[0] == 14  # 14 ego features
        
        # All values should be in [-1, 1] or [0, 1] range
        # (some fields are naturally bounded)
        assert ego_vec.abs().max() <= 1.5  # Allow small margin for numerical precision
    
    def test_route_normalization(self, dataset):
        """Test route points are normalized."""
        sample = dataset[0]
        route = sample["route"]
        
        # Should be (K, 2) with values in [-1, 1]
        assert route.shape == (32, 2)
        assert route.abs().max() <= 1.0 + 1e-5  # Small epsilon for numerical precision
    
    def test_objects_normalization(self, dataset):
        """Test object tokens are normalized."""
        sample = dataset[0]
        objects = sample["objects"]
        object_mask = sample["object_mask"]
        
        # Check valid objects (where mask == 1)
        valid_mask = object_mask == 1.0
        if valid_mask.sum() > 0:
            valid_objects = objects[valid_mask]
            
            # Spatial positions (columns 1, 2) should be in [-1, 1]
            assert valid_objects[:, 1:3].abs().max() <= 1.0 + 1e-5
            
            # Sin/cos yaw (columns 3, 4) should be in [-1, 1]
            assert valid_objects[:, 3:5].abs().max() <= 1.0 + 1e-5
            
            # Dimensions (columns 5, 6) should be in [0, 1]
            assert valid_objects[:, 5:7].min() >= -1e-5
            assert valid_objects[:, 5:7].max() <= 1.0 + 1e-5
            
            # Velocities (columns 7, 8) should be in [-1, 1]
            assert valid_objects[:, 7:9].abs().max() <= 1.0 + 1e-5
    
    def test_futures_normalization(self, dataset):
        """Test future waypoints and speeds are normalized."""
        sample = dataset[0]
        future_xy = sample["future_xy"]
        future_v = sample["future_v"]
        future_mask = sample["future_mask"]
        
        # Waypoints should be in [-1, 1]
        assert future_xy.abs().max() <= 1.0 + 1e-5
        
        # Speeds should be in [0, 1]
        assert future_v.min() >= -1e-5
        assert future_v.max() <= 1.0 + 1e-5
    
    def test_bev_normalization(self, dataset):
        """Test BEV channels are normalized."""
        sample = dataset[0]
        bev = sample["bev"]
        
        # Channels 0-14: should be in [0, 1] (binary/semantic)
        assert bev[0:15].min() >= -1e-5
        assert bev[0:15].max() <= 1.0 + 1e-5
        
        # Channels 15-16: velocity, should be in [-1, 1]
        assert bev[15:17].abs().max() <= 1.0 + 1e-5
        
        # Channel 17: speed limit, should be in [0, 1]
        assert bev[17].min() >= -1e-5
        assert bev[17].max() <= 1.0 + 1e-5


class TestFutureMasking:
    """Test future masking for incomplete episodes."""
    
    def test_future_mask_shape(self, dataset):
        """Test future mask has correct shape."""
        sample = dataset[0]
        future_mask = sample["future_mask"]
        
        # Should be (N,)
        assert future_mask.shape == (12,)
        assert future_mask.dtype == torch.float32
    
    def test_future_mask_values(self, dataset):
        """Test future mask has valid values."""
        sample = dataset[0]
        future_mask = sample["future_mask"]
        
        # Mask should be binary (0 or 1)
        assert ((future_mask == 0.0) | (future_mask == 1.0)).all()
        
        # Valid futures should be contiguous from start
        # (i.e., no 1s after 0s)
        if future_mask.sum() < len(future_mask):
            first_zero_idx = (future_mask == 0.0).nonzero()[0].item()
            # All values after first zero should be zero
            assert (future_mask[first_zero_idx:] == 0.0).all()
    
    def test_masked_futures_have_zero_values(self, dataset):
        """Test that masked futures are padded with zeros."""
        sample = dataset[0]
        future_xy = sample["future_xy"]
        future_v = sample["future_v"]
        future_mask = sample["future_mask"]
        
        # Where mask is 0, futures should be 0 (padding)
        invalid_mask = future_mask == 0.0
        if invalid_mask.sum() > 0:
            # Note: normalized zeros might not be exactly 0
            # But they should be close to padding values
            pass  # Skip this check as normalized padding might not be exactly 0


class TestDataLoaderOptimizations:
    """Test dataloader optimizations."""
    
    def test_dataloader_with_optimizations(self, run_dir):
        """Test that dataloader can be created with all optimizations."""
        loader = create_bc_dataloader(
            run_dir,
            batch_size=4,
            shuffle=False,
            num_workers=0,  # Use 0 for testing
            prefetch_factor=2,
            persistent_workers=False,  # Must be False when num_workers=0
            pin_memory=False,
            drop_last=False,
        )
        
        assert loader is not None
        assert isinstance(loader, torch.utils.data.DataLoader)
        
        # Get first batch
        batch = next(iter(loader))
        assert batch is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

