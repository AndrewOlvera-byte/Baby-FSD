"""
Tests for PyTorch BC trajectory dataset and dataloader (WebDataset backend).

Validates shapes, dtypes, value ranges, and batch consistency.
"""

import os
import pytest
import torch
import numpy as np
import logging
import json
import tarfile
import io

from data.torch_dataset import BCWebDataset, create_bc_dataloader, fast_collate_fn
from data.norms import (
    V_MAX, ACCEL_MAX, SPATIAL_MAX, BEV_VEL_MAX, BEV_SPEED_LIMIT_MAX
)

webdataset = pytest.importorskip("webdataset")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def create_test_webdataset(tmp_path, n_samples=60, samples_per_shard=20, shard_sizes=None):
    """Helper to create a test WebDataset (optionally with uneven shards)."""
    run_dir = tmp_path / "test_run_wds"
    run_dir.mkdir()

    if shard_sizes is None:
        shard_sizes = []
        remaining = n_samples
        while remaining > 0:
            shard_sizes.append(min(samples_per_shard, remaining))
            remaining -= shard_sizes[-1]
    else:
        n_samples = sum(shard_sizes)

    K, N, M = 32, 12, 64
    C, H, W = 18, 150, 200
    
    # Create metadata.json
    metadata = {
        "K": K,
        "N_future": N,
        "M": M,
        "C": C,
        "H": H,
        "W": W,
        "version": "2.0",
        "norms_version": 1,
        "run_id": "test",
        "samples_per_shard": samples_per_shard,
        "total_samples": n_samples,
    }
    
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Create shard files
    n_shards = len(shard_sizes)
    sample_offset = 0
    
    for shard_idx, shard_count in enumerate(shard_sizes):
        shard_path = run_dir / f"shard-{shard_idx:06d}.tar"
        
        with tarfile.open(shard_path, "w") as tar:
            start_idx = sample_offset
            end_idx = start_idx + shard_count
            
            for sample_idx in range(start_idx, end_idx):
                base_name = f"{sample_idx:08d}"
                
                # Create sample data
                frame_id = np.array(sample_idx, dtype=np.int64)
                episode_id = np.array(sample_idx // 20, dtype=np.int16)
                ego_vec = np.random.randn(14).astype(np.float32) * 0.5
                bev = np.random.rand(C, H, W).astype(np.float32)
                route = np.random.randn(K, 2).astype(np.float32) * 10.0
                objects = np.random.randn(M, 11).astype(np.float32) * 5.0
                object_mask = (np.random.rand(M) > 0.5).astype(np.float32)
                future_xy = np.random.randn(N, 2).astype(np.float32) * 10.0
                future_v = np.random.rand(N).astype(np.float32) * 20.0
                future_mask = np.ones((N,), dtype=np.float32)
                
                # Write each array as a separate file
                sample_data = {
                    "frame_id": frame_id,
                    "episode_id": episode_id,
                    "ego_vec": ego_vec,
                    "bev": bev,
                    "route": route,
                    "objects": objects,
                    "object_mask": object_mask,
                    "future_xy": future_xy,
                    "future_v": future_v,
                    "future_mask": future_mask,
                }
                
                for key, value in sample_data.items():
                    buffer = io.BytesIO()
                    np.save(buffer, value, allow_pickle=False)
                    buffer.seek(0)
                    
                    info = tarfile.TarInfo(name=f"{base_name}.{key}.npy")
                    info.size = len(buffer.getvalue())
                    tar.addfile(info, buffer)

            shard_meta = {
                "shard_idx": shard_idx,
                "first_sample_idx": start_idx,
                "last_sample_idx": end_idx - 1,
                "sample_count": shard_count,
            }
            meta_buffer = io.BytesIO()
            meta_buffer.write(json.dumps(shard_meta, indent=2).encode("utf-8"))
            meta_buffer.seek(0)
            meta_info = tarfile.TarInfo(name="__metadata__.json")
            meta_info.size = len(meta_buffer.getvalue())
            tar.addfile(meta_info, meta_buffer)

            sample_offset = end_idx
    
    return str(run_dir), n_samples


@pytest.fixture(scope="module")
def test_dataset(tmp_path_factory):
    """Create a test dataset once for all tests."""
    tmp_path = tmp_path_factory.mktemp("data")
    run_dir, n_frames = create_test_webdataset(tmp_path, n_samples=60, samples_per_shard=20)
    logger.info(f"Created test WebDataset with {n_frames} frames in {run_dir}")
    return run_dir, n_frames


class TestBCWebDataset:
    """Test BCWebDataset."""
    
    def test_dataset_loads(self, test_dataset):
        """Test that dataset loads without errors."""
        run_dir, n_frames = test_dataset
        dataset = BCWebDataset(run_dir, future_horizon=12, route_points=32, max_objects=64)
        # Note: __len__ is approximate for IterableDataset
        assert dataset.total_samples == n_frames
        logger.info(f"Dataset loaded: {dataset.total_samples} samples")
    
    def test_sample_structure(self, test_dataset):
        """Test that sample has correct structure."""
        run_dir, _ = test_dataset
        dataset = BCWebDataset(run_dir)
        
        # Get first sample from iterator
        sample = next(iter(dataset))
        
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
        dataset = BCWebDataset(run_dir)
        sample = next(iter(dataset))
        bev = sample["bev"]
        
        assert bev.ndim == 3
        assert bev.shape[0] == 18  # channels
        assert bev.shape[1] > 0    # height
        assert bev.shape[2] > 0    # width
        assert bev.dtype == torch.float32
    
    def test_bev_values(self, test_dataset):
        """Test BEV tensor value ranges."""
        run_dir, _ = test_dataset
        dataset = BCWebDataset(run_dir)
        sample = next(iter(dataset))
        bev = sample["bev"]
        
        # Should be finite
        assert torch.isfinite(bev).all()
        
        # Binary channels (0-14) should be in [0, 1] after normalization
        assert bev[0:15].min() >= -0.1  # Allow small tolerance
        assert bev[0:15].max() <= 1.1
    
    def test_route_shape(self, test_dataset):
        """Test route tensor shape."""
        run_dir, _ = test_dataset
        dataset = BCWebDataset(run_dir)
        sample = next(iter(dataset))
        route = sample["route"]
        
        assert route.shape == (32, 2)
        assert route.dtype == torch.float32
        assert torch.isfinite(route).all()
    
    def test_multiple_samples(self, test_dataset):
        """Test that we can load multiple samples."""
        run_dir, _ = test_dataset
        dataset = BCWebDataset(run_dir)
        n_samples = min(10, dataset.total_samples)
        
        samples = []
        for i, sample in enumerate(dataset):
            if i >= n_samples:
                break
            assert sample is not None
            assert "bev" in sample
            samples.append(sample)
        
        assert len(samples) == n_samples
    
    def test_frame_id_consistency(self, test_dataset):
        """Test that frame IDs are consistent."""
        run_dir, _ = test_dataset
        dataset = BCWebDataset(run_dir)
        
        iterator = iter(dataset)
        sample1 = next(iterator)
        sample2 = next(iterator)
        
        assert sample1["frame_id"] != sample2["frame_id"]
        assert isinstance(sample1["frame_id"], int)
        assert isinstance(sample2["frame_id"], int)
    
    def test_split_train_val(self, test_dataset):
        """Test train/val splitting."""
        run_dir, n_frames = test_dataset
        
        train_ds = BCWebDataset(run_dir, split="train", val_ratio=0.2)
        val_ds = BCWebDataset(run_dir, split="val", val_ratio=0.2)
        
        # Check split sizes (approximate for shard-level split)
        expected_train = int(n_frames * 0.8)
        expected_val = n_frames - expected_train
        
        # Length is approximate for IterableDataset, but should be close
        assert abs(len(train_ds) - expected_train) < 10  # Allow some tolerance
        assert abs(len(val_ds) - expected_val) < 10
        
        # Check no overlap (get first samples)
        train_sample = next(iter(train_ds))
        val_sample = next(iter(val_ds))
        assert train_sample["frame_id"] != val_sample["frame_id"]

    def test_len_tracks_shard_samples(self, tmp_path_factory):
        """Length should follow shard sample counts, not ratios."""
        tmp_path = tmp_path_factory.mktemp("uneven_shards")
        shard_sizes = [3, 7, 2]
        val_ratio = 0.25  # train takes first two shards
        run_dir, total = create_test_webdataset(tmp_path, shard_sizes=shard_sizes, samples_per_shard=10)

        all_ds = BCWebDataset(run_dir, split="all")
        train_ds = BCWebDataset(run_dir, split="train", val_ratio=val_ratio)
        val_ds = BCWebDataset(run_dir, split="val", val_ratio=val_ratio)

        assert len(all_ds) == total
        assert len(train_ds) == sum(shard_sizes[:2])
        assert len(val_ds) == shard_sizes[2]


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
        
        n_batches = min(3, len(loader) if hasattr(loader, "__len__") else 3)
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
        dataset = BCWebDataset(run_dir)
        sample = next(iter(dataset))
        objects = sample["objects"]
        object_mask = sample["object_mask"]
        
        assert objects.shape == (64, 11)
        assert object_mask.shape == (64,)
        assert objects.dtype == torch.float32
        assert object_mask.dtype == torch.float32
    
    def test_object_mask_validity(self, test_dataset):
        """Test object mask values."""
        run_dir, _ = test_dataset
        dataset = BCWebDataset(run_dir)
        sample = next(iter(dataset))
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
        dataset = BCWebDataset(run_dir)
        sample = next(iter(dataset))
        ego_vec = sample["ego_vec"]
        
        assert ego_vec.ndim == 1
        assert ego_vec.shape[0] == 14
        
        # Values should be in reasonable normalized range
        assert ego_vec.abs().max() <= 2.0  # Allow some margin
    
    def test_route_normalization(self, test_dataset):
        """Test route points are normalized."""
        run_dir, _ = test_dataset
        dataset = BCWebDataset(run_dir)
        sample = next(iter(dataset))
        route = sample["route"]
        
        assert route.shape == (32, 2)
        # After normalization should be in [-1, 1]
        assert route.abs().max() <= 1.0 + 1e-4
    
    def test_futures_normalization(self, test_dataset):
        """Test future waypoints and speeds are normalized."""
        run_dir, _ = test_dataset
        dataset = BCWebDataset(run_dir)
        sample = next(iter(dataset))
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
        dataset = BCWebDataset(run_dir)
        sample = next(iter(dataset))
        future_mask = sample["future_mask"]
        
        assert future_mask.shape == (12,)
        assert future_mask.dtype == torch.float32
    
    def test_future_mask_values(self, test_dataset):
        """Test future mask has valid values."""
        run_dir, _ = test_dataset
        dataset = BCWebDataset(run_dir)
        sample = next(iter(dataset))
        future_mask = sample["future_mask"]
        
        # Mask should be binary (0 or 1)
        assert ((future_mask == 0.0) | (future_mask == 1.0)).all()


def test_fast_collate_empty_batch_raises():
    """fast_collate_fn should fail fast on empty batches."""
    with pytest.raises(ValueError):
        fast_collate_fn([])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
