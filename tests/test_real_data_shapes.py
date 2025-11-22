"""
E2E sanity checks against a real BC_v2 HDF5 run (if present).

Skips gracefully when no BC_v2 run is available locally.
"""

import glob
import os
import pytest

from data.torch_dataset import BCTrajectoryDataset

h5py = pytest.importorskip("h5py")


def _find_first_h5_with_data():
    for path in sorted(glob.glob(os.path.join("data", "BC_v2", "run-*", "*.h5"))):
        with h5py.File(path, "r") as f:
            # Require core datasets to exist and have data
            required = {"bev", "future_xy", "future_v", "route", "objects", "object_mask"}
            if not required.issubset(set(f.keys())):
                continue
            if any(len(f[k]) == 0 for k in required):
                continue
            return path
    return None


@pytest.mark.skipif(_find_first_h5_with_data() is None, reason="No BC_v2 run with data found")
def test_real_hdf5_shapes_and_ranges():
    """Validate shapes and normalized ranges on a real collected run."""
    h5_path = _find_first_h5_with_data()
    assert h5_path is not None  # appease type checkers

    run_dir = os.path.dirname(h5_path)

    # Read expected dimensions from file attributes
    with h5py.File(h5_path, "r") as f:
        H = int(f.attrs["H"])
        W = int(f.attrs["W"])
        K = int(f.attrs["K"])
        N = int(f.attrs["N_future"])
        M = int(f.attrs["M"])

    ds = BCTrajectoryDataset(run_dir, future_horizon=N, route_points=K, max_objects=M)
    sample = ds[0]

    # Shape checks against file metadata
    assert sample["bev"].shape == (18, H, W)
    assert sample["route"].shape == (K, 2)
    assert sample["future_xy"].shape == (N, 2)
    assert sample["future_v"].shape == (N,)
    assert sample["object_mask"].shape == (M,)

    # Normalized range checks (allow tiny numerical noise)
    assert sample["bev"][15:17].abs().max() <= 1.01  # velocity channels
    assert sample["bev"][17].min() >= -1e-3 and sample["bev"][17].max() <= 1.01  # speed limit
    assert sample["route"].abs().max() <= 1.01
    assert sample["future_xy"].abs().max() <= 1.01
    assert sample["future_v"].min() >= -1e-3 and sample["future_v"].max() <= 1.01
    assert sample["objects"][:, 1:3].abs().max() <= 1.01
    assert sample["objects"][:, 7:9].abs().max() <= 1.01
