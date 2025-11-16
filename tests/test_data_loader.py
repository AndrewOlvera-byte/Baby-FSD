import io
import os
import zlib

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from data.data import BCRunDataset


def _write_parquet(directory, name, schema, data):
    os.makedirs(os.path.join(directory, name), exist_ok=True)
    table = pa.Table.from_pydict(data, schema=schema)
    pq.write_table(table, os.path.join(directory, name, "part-000.parquet"))


def test_future_padding(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    frame_schema = pa.schema(
        [
            ("frame_id", pa.int64()),
            ("ego_x_w", pa.float32()),
            ("ego_y_w", pa.float32()),
            ("ego_yaw_w", pa.float32()),
            ("speed_mps", pa.float32()),
            ("yaw_rate", pa.float32()),
            ("steer_norm", pa.float32()),
            ("throttle", pa.float32()),
            ("brake", pa.float32()),
        ]
    )
    frames = {
        "frame_id": [0, 1],
        "ego_x_w": [0.0, 1.0],
        "ego_y_w": [0.0, 2.0],
        "ego_yaw_w": [0.0, 3.0],
        "speed_mps": [5.0, 3.0],
        "yaw_rate": [0.1, 0.2],
        "steer_norm": [0.0, 0.05],
        "throttle": [0.5, 0.2],
        "brake": [0.0, 0.1],
    }
    _write_parquet(run_dir, "frames", frame_schema, frames)

    futures_schema = pa.schema(
        [
            ("frame_id", pa.int64()),
            ("i", pa.int8()),
            ("x_ego", pa.float32()),
            ("y_ego", pa.float32()),
            ("v_mps", pa.float32()),
        ]
    )
    futures = {
        "frame_id": [0, 0, 0, 1, 1],
        "i": [0, 1, 2, 0, 1],
        "x_ego": [1.0, 2.0, 3.0, 0.5, 1.0],
        "y_ego": [0.0, 0.5, 1.0, -0.5, -1.0],
        "v_mps": [6.0, 6.5, 7.0, 4.0, 4.5],
    }
    _write_parquet(run_dir, "futures", futures_schema, futures)

    bev_schema = pa.schema(
        [
            ("frame_id", pa.int64()),
            ("data", pa.binary()),
        ]
    )
    bev_arrays = []
    for _ in range(2):
        arr = np.zeros((2, 2, 2), dtype=np.float32)
        buf = io.BytesIO()
        np.save(buf, arr, allow_pickle=False)
        bev_arrays.append(zlib.compress(buf.getvalue()))
    bev = {"frame_id": [0, 1], "data": bev_arrays}
    _write_parquet(run_dir, "bev_frames", bev_schema, bev)

    dataset = BCRunDataset(str(run_dir), future_horizon=3)
    assert len(dataset) == 2

    sample_full = dataset[0]
    assert sample_full["futures_mask"].sum() == 3
    np.testing.assert_allclose(sample_full["futures_xy"][2], [3.0, 1.0])

    sample_truncated = dataset[1]
    assert sample_truncated["futures_mask"].tolist() == [1.0, 1.0, 0.0]
    np.testing.assert_allclose(sample_truncated["futures_xy"][2], [0.0, 0.0])


