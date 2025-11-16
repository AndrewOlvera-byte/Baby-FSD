import os
import glob
import json
from typing import Dict, Tuple

import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import zlib


def _read_concat(run_dir: str, prefix: str, columns=None) -> pa.Table:
    files = sorted(glob.glob(os.path.join(run_dir, f'{prefix}-*.parquet')))
    tabs = []
    for p in files:
        tabs.append(pq.read_table(p, columns=columns))
    return pa.concat_tables(tabs, promote=True) if tabs else pa.table({})


def _decode_bev_row(row) -> np.ndarray:
    C = int(row['C'].as_py())
    H = int(row['H'].as_py())
    W = int(row['W'].as_py())
    blob = row['data'].as_buffer().to_pybytes()
    raw = zlib.decompress(blob)
    arr = np.load(__import__('io').BytesIO(raw), allow_pickle=False)
    assert arr.shape == (C, H, W)
    return arr


def load_step(run_dir: str, frame_id: int, max_objects: int = 64) -> Tuple[Dict, Dict]:
    frames = _read_concat(run_dir, 'frames')
    bev = _read_concat(run_dir, 'bev_frames')
    route = _read_concat(run_dir, 'route_points')
    obj = _read_concat(run_dir, 'object_tokens')
    fut = _read_concat(run_dir, 'futures')

    # Filter by frame_id
    def eq_col(tab, name):
        return np.array(tab.column(name).to_pylist())

    # Ego vector
    fid_frames = eq_col(frames, 'frame_id')
    idx = np.where(fid_frames == frame_id)[0]
    if idx.size == 0:
        raise KeyError(f'frame_id {frame_id} not found')
    i = int(idx[0])
    row = {name: frames.column(name)[i] for name in frames.schema.names}
    ego_vec = np.array([
        float(row.get('speed_mps', 0).as_py()),
        float(row.get('accel_long', 0).as_py()),
        float(row.get('accel_lat', 0).as_py()),
        float(row.get('yaw_rate', 0).as_py()),
        float(row.get('heading_rad', 0).as_py()),
        float(row.get('steer_angle_rad', 0).as_py()),
        float(row.get('throttle', 0).as_py()),
        float(row.get('brake', 0).as_py()),
        float(row.get('curvature', 0).as_py()),
        float(row.get('gear', 0).as_py()),
        float(row.get('speed_limit_mps', 0).as_py()),
        float(row.get('command', 0).as_py()),
    ], dtype=np.float32)

    # BEV tensor
    fid_bev = eq_col(bev, 'frame_id')
    j = int(np.where(fid_bev == frame_id)[0][0])
    bev_row = {name: bev.column(name)[j] for name in bev.schema.names}
    bev_tensor = _decode_bev_row(bev_row)

    # Route polyline
    rp = route.filter(pa.compute.equal(route['frame_id'], pa.scalar(frame_id)))
    if rp.num_rows > 0:
        x = np.array(rp['x_ego'].to_pylist(), dtype=np.float32)
        y = np.array(rp['y_ego'].to_pylist(), dtype=np.float32)
        dx = np.array(rp['dx'].to_pylist(), dtype=np.float32) if 'dx' in rp.schema.names else np.zeros_like(x)
        dy = np.array(rp['dy'].to_pylist(), dtype=np.float32) if 'dy' in rp.schema.names else np.zeros_like(y)
        curv = np.array(rp['curvature'].to_pylist(), dtype=np.float32) if 'curvature' in rp.schema.names else np.zeros_like(x)
        s_frac = np.array(rp['s_frac'].to_pylist(), dtype=np.float32) if 's_frac' in rp.schema.names else np.zeros_like(x)
        route_poly = np.stack([x, y, dx, dy, curv, s_frac], axis=-1)
    else:
        route_poly = np.zeros((0, 6), dtype=np.float32)

    # Object tokens (+mask)
    ob = obj.filter(pa.compute.equal(obj['frame_id'], pa.scalar(frame_id)))
    tokens = np.zeros((max_objects, 11), dtype=np.float32)
    mask = np.zeros((max_objects,), dtype=np.float32)
    if ob.num_rows > 0:
        ox = np.array(ob['x_ego'].to_pylist(), dtype=np.float32)
        oy = np.array(ob['y_ego'].to_pylist(), dtype=np.float32)
        dist = ox * ox + oy * oy
        order = np.argsort(dist)[:max_objects]
        fields = {
            'x': ox[order],
            'y': oy[order],
            'sin': np.array(ob['sin_yaw'].to_pylist(), dtype=np.float32)[order] if 'sin_yaw' in ob.schema.names else np.zeros_like(order, dtype=np.float32),
            'cos': np.array(ob['cos_yaw'].to_pylist(), dtype=np.float32)[order] if 'cos_yaw' in ob.schema.names else np.ones_like(order, dtype=np.float32),
            'len': np.array(ob['length'].to_pylist(), dtype=np.float32)[order],
            'wid': np.array(ob['width'].to_pylist(), dtype=np.float32)[order],
            'vx': np.array(ob['vx'].to_pylist(), dtype=np.float32)[order],
            'vy': np.array(ob['vy'].to_pylist(), dtype=np.float32)[order],
            'type': np.array(ob['type_id'].to_pylist(), dtype=np.float32)[order],
            'oncoming': np.array(ob['oncoming_flag'].to_pylist(), dtype=np.float32)[order] if 'oncoming_flag' in ob.schema.names else np.zeros_like(order, dtype=np.float32),
            'priority': np.array(ob['priority_flag'].to_pylist(), dtype=np.float32)[order] if 'priority_flag' in ob.schema.names else np.zeros_like(order, dtype=np.float32),
        }
        K = len(order)
        tokens[:K, :] = np.stack([
            fields['x'], fields['y'], fields['sin'], fields['cos'], fields['len'], fields['wid'],
            fields['vx'], fields['vy'], fields['type'], fields['oncoming'], fields['priority']
        ], axis=-1)
        mask[:K] = 1.0

    # Futures (actions)
    fu = fut.filter(pa.compute.equal(fut['frame_id'], pa.scalar(frame_id)))
    if fu.num_rows > 0:
        # ensure sorted by i
        ii = np.array(fu['i'].to_pylist(), dtype=np.int32)
        ord_i = np.argsort(ii)
        x = np.array(fu['x_ego'].to_pylist(), dtype=np.float32)[ord_i]
        y = np.array(fu['y_ego'].to_pylist(), dtype=np.float32)[ord_i]
        v = np.array(fu['v_mps'].to_pylist(), dtype=np.float32)[ord_i]
        future_waypoints = np.stack([x, y], axis=-1)
        future_speeds = v
    else:
        future_waypoints = np.zeros((0, 2), dtype=np.float32)
        future_speeds = np.zeros((0,), dtype=np.float32)

    observation = {
        "ego_vec": ego_vec,                  # (d_ego,)
        "bev_tensor": bev_tensor,            # (C, H, W)
        "route_poly": route_poly,            # (R, 6) -> x,y,dx,dy,curv,s
        "object_tokens": tokens,             # (M, 11)
        "object_mask": mask,                 # (M,)
        "meta": {},                          # optional
    }
    action = {
        "future_waypoints": future_waypoints,   # (N, 2)
        "future_speeds": future_speeds,         # (N,)
    }
    return observation, action


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--frame_id", type=int, default=0)
    args = ap.parse_args()
    obs, act = load_step(args.run_dir, args.frame_id)
    print("obs:", {k: (np.array(v).shape if hasattr(v, 'shape') else type(v)) for k, v in obs.items()})
    print("act:", {k: (np.array(v).shape if hasattr(v, 'shape') else type(v)) for k, v in act.items()})


