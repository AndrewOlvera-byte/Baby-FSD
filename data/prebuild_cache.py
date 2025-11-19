#!/usr/bin/env python3
"""
Helper script to pre-build the frame index cache for BCTrajectoryDataset.

This is useful if you want to build the cache ahead of time (e.g., after
collecting new data) rather than waiting for it to build on first training run.

Usage:
    python data/prebuild_cache.py data/BC_v1/run-20251116-140827
    
Or from Docker:
    docker compose run --rm trainer python data/prebuild_cache.py data/BC_v1/run-20251116-140827
"""

import sys
import time
from pathlib import Path

def prebuild_cache(run_dir: str):
    """Pre-build the frame index cache for a dataset."""
    print(f"Pre-building cache for: {run_dir}")
    
    # Import here to avoid circular imports
    from data.torch_dataset import BCTrajectoryDataset
    
    start_time = time.time()
    
    # Initialize dataset (this will build and cache the index)
    dataset = BCTrajectoryDataset(
        run_dir=run_dir,
        future_horizon=12,
        route_points=32,
        max_objects=64,
    )
    
    elapsed = time.time() - start_time
    
    print(f"\n✅ Cache built successfully!")
    print(f"   Dataset size: {len(dataset)} frames")
    print(f"   Time taken: {elapsed:.2f}s")
    print(f"   Cache location: {run_dir}/.frame_index_cache.npy")
    print(f"\n   Next training run will initialize in ~1-2 seconds! 🚀")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python data/prebuild_cache.py <run_dir>")
        print("Example: python data/prebuild_cache.py data/BC_v1/run-20251116-140827")
        sys.exit(1)
    
    run_dir = sys.argv[1]
    
    if not Path(run_dir).exists():
        print(f"Error: Directory not found: {run_dir}")
        sys.exit(1)
    
    prebuild_cache(run_dir)

