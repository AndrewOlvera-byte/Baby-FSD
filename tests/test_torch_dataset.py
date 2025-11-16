"""
Tests for PyTorch BC trajectory dataset and dataloader.

Validates shapes, dtypes, value ranges, and batch consistency.
"""

import os
import pytest
import torch
import numpy as np

from data.torch_dataset import BCTrajectoryDataset, create_bc_dataloader


def find_bc_run_dir():
    """Find a BC run directory for testing."""
    base = os.path.join("data", "BC_v1")
    if not os.path.isdir(base):
        return None
    
    # Find first run directory
    for entry in os.listdir(base):
        run_path = os.path.join(base, entry)
        if os.path.isdir(run_path) and entry.startswith("run-"):
            # Check if it has required tables
            required = ["frames", "futures", "route_points", "bev_frames"]
            if all(os.path.isdir(os.path.join(run_path, t)) for t in required):
                return run_path
    
    return None


@pytest.fixture
def run_dir():
    """Get BC run directory for testing."""
    path = find_bc_run_dir()
    if path is None:
        pytest.skip("No BC run directory found for testing")
    return path


@pytest.fixture
def dataset(run_dir):
    """Create dataset instance."""
    return BCTrajectoryDataset(run_dir, future_horizon=12, route_points=32)


class TestBCTrajectoryDataset:
    """Test BCTrajectoryDataset."""
    
    def test_dataset_loads(self, dataset):
        """Test that dataset loads without errors."""
        assert len(dataset) > 0
        assert hasattr(dataset, "_frames")
        assert hasattr(dataset, "_futures")
        assert hasattr(dataset, "_route")
        assert hasattr(dataset, "_bev")
    
    def test_sample_structure(self, dataset):
        """Test that sample has correct structure."""
        sample = dataset[0]
        
        # Check keys
        expected_keys = {"frame_id", "bev", "route", "futures", "futures_speed", "control", "state"}
        assert set(sample.keys()) == expected_keys
        
        # Check types
        assert isinstance(sample["frame_id"], int)
        assert isinstance(sample["bev"], torch.Tensor)
        assert isinstance(sample["route"], torch.Tensor)
        assert isinstance(sample["futures"], torch.Tensor)
        assert isinstance(sample["futures_speed"], torch.Tensor)
        assert isinstance(sample["control"], torch.Tensor)
        assert isinstance(sample["state"], torch.Tensor)
    
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
    
    def test_futures_shape(self, dataset):
        """Test futures tensor shapes."""
        sample = dataset[0]
        futures = sample["futures"]
        futures_speed = sample["futures_speed"]
        
        # Should be (N, 2) and (N,)
        assert futures.shape == (12, 2)
        assert futures_speed.shape == (12,)
        
        assert futures.dtype == torch.float32
        assert futures_speed.dtype == torch.float32
        
        # Should be finite
        assert torch.isfinite(futures).all()
        assert torch.isfinite(futures_speed).all()
    
    def test_futures_values(self, dataset):
        """Test futures value ranges."""
        sample = dataset[0]
        futures = sample["futures"]
        futures_speed = sample["futures_speed"]
        
        # Futures should be in reasonable range
        assert futures.abs().max() < 200.0
        
        # Speeds should be non-negative and reasonable
        assert futures_speed.min() >= 0.0
        assert futures_speed.max() < 100.0  # m/s (~360 km/h max)
    
    def test_control_shape(self, dataset):
        """Test control tensor shape."""
        sample = dataset[0]
        control = sample["control"]
        
        # Should be (3,) [steer, throttle, brake]
        assert control.shape == (3,)
        assert control.dtype == torch.float32
        
        # Should be finite
        assert torch.isfinite(control).all()
    
    def test_control_values(self, dataset):
        """Test control value ranges."""
        sample = dataset[0]
        control = sample["control"]
        
        steer, throttle, brake = control
        
        # Steer in [-1, 1]
        assert -1.0 <= steer <= 1.0
        
        # Throttle in [0, 1]
        assert 0.0 <= throttle <= 1.0
        
        # Brake in [0, 1]
        assert 0.0 <= brake <= 1.0
    
    def test_state_shape(self, dataset):
        """Test state tensor shape."""
        sample = dataset[0]
        state = sample["state"]
        
        # Should be (7,) [speed, yaw_rate, accel_long, accel_lat, curvature, speed_limit, command]
        assert state.shape == (7,)
        assert state.dtype == torch.float32
        
        # Should be finite
        assert torch.isfinite(state).all()
    
    def test_state_values(self, dataset):
        """Test state value ranges."""
        sample = dataset[0]
        state = sample["state"]
        
        speed, yaw_rate, accel_long, accel_lat, curvature, speed_limit, command = state
        
        # Speed should be non-negative and reasonable
        assert 0.0 <= speed < 100.0
        
        # Yaw rate should be reasonable (rad/s)
        assert abs(yaw_rate) < 10.0
        
        # Accelerations should be reasonable (m/s^2)
        assert abs(accel_long) < 20.0
        assert abs(accel_lat) < 20.0
        
        # Curvature should be reasonable
        assert abs(curvature) < 10.0
        
        # Speed limit should be reasonable
        assert 0.0 <= speed_limit < 100.0
        
        # Command should be integer 0-5
        assert 0 <= command <= 5
    
    def test_multiple_samples(self, dataset):
        """Test that we can load multiple samples."""
        n_samples = min(10, len(dataset))
        
        for i in range(n_samples):
            sample = dataset[i]
            assert sample is not None
            assert "bev" in sample
            assert "control" in sample
    
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
        loader = create_bc_dataloader(
            run_dir,
            batch_size=4,
            shuffle=False,
            num_workers=0,
        )
        
        # Get first batch
        batch = next(iter(loader))
        
        assert batch is not None
        assert "bev" in batch
        assert "control" in batch
    
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
        assert batch["futures"].shape[1:] == (12, 2)
        
        # Futures speed should be (B, N)
        assert batch["futures_speed"].shape[1:] == (12,)
        
        # Control should be (B, 3)
        assert batch["control"].shape[1:] == (3,)
        
        # State should be (B, 7)
        assert batch["state"].shape[1:] == (7,)
    
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
        assert batch["futures"].shape[0] == batch_size
        assert batch["futures_speed"].shape[0] == batch_size
        assert batch["control"].shape[0] == batch_size
        assert batch["state"].shape[0] == batch_size
    
    def test_multiple_batches(self, run_dir):
        """Test that we can iterate through multiple batches."""
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
            assert "control" in batch


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

