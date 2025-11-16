import os
import glob
import json
import argparse
import math
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import zlib


def last_run_dir(root='data/BC_v1'):
    runs = sorted(glob.glob(os.path.join(root, 'run-*')), key=os.path.getmtime)
    if not runs:
        raise SystemExit('No runs found in ' + root)
    return runs[-1]


def parquet_sum_rows(run_dir, prefix):
    files = sorted(glob.glob(os.path.join(run_dir, f'{prefix}-*.parquet')))
    total = 0
    for p in files:
        try:
            total += pq.ParquetFile(p).metadata.num_rows
        except Exception:
            pass
    return total


def read_tables_indexed(run_dir, prefix, columns=None):
    files = sorted(glob.glob(os.path.join(run_dir, f'{prefix}-*.parquet')))
    rows = []
    for p in files:
        try:
            tbl = pq.read_table(p, columns=columns)
            rows.append(tbl)
        except Exception:
            continue
    if not rows:
        return None
    return pa.concat_tables(rows, promote=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', default=None)
    ap.add_argument('--root', default='data/BC_v1')
    ap.add_argument('--K', type=int, default=16)
    ap.add_argument('--N', type=int, default=6)
    ap.add_argument('--max_sample_frames', type=int, default=50)
    args = ap.parse_args()

    run_dir = args.run_dir or last_run_dir(args.root)
    print('Run dir:', run_dir)
    meta = json.load(open(os.path.join(run_dir, 'meta.json')))
    print('Meta:', meta)
    K = int(meta.get('K', args.K))
    N = int(meta.get('N', args.N))

    frames_total = parquet_sum_rows(run_dir, 'frames')
    route_total = parquet_sum_rows(run_dir, 'route_points')
    actors_total = parquet_sum_rows(run_dir, 'actors')
    tls_total = parquet_sum_rows(run_dir, 'traffic_lights')
    maps_total = parquet_sum_rows(run_dir, 'map_segments')
    futures_total = parquet_sum_rows(run_dir, 'futures')
    bev_total = parquet_sum_rows(run_dir, 'bev_frames')
    obj_total = parquet_sum_rows(run_dir, 'object_tokens')

    issues = []
    if frames_total <= 0:
        issues.append('no_frames')
    if futures_total % max(1, N) != 0:
        issues.append('futures_not_divisible_by_N')
    matured_frames = futures_total // max(1, N)
    route_cov = route_total / float(max(1, frames_total * K))
    actors_per_f = actors_total / float(max(1, frames_total))
    tls_per_f = tls_total / float(max(1, frames_total))
    maps_per_f = maps_total / float(max(1, frames_total))
    if bev_total != frames_total:
        issues.append('bev_count_mismatch')
    if obj_total < actors_total:
        issues.append('object_tokens_missing')

    print(f'Counts: frames={frames_total}, futures_rows={futures_total} (~matured_frames={matured_frames})')
    print(f'Coverage: route_cov≈{route_cov:.2f}, actors/f={actors_per_f:.1f}, tls/f={tls_per_f:.2f}, mapSegs/f={maps_per_f:.2f}')

    # Load select columns for per-step validation
    frames_tbl = read_tables_indexed(run_dir, 'frames', columns=[
        'frame_id', 'speed_mps', 'yaw_rate', 'accel_long', 'accel_lat',
        'throttle', 'brake', 'steer_norm', 'curvature', 'command',
        'time_of_day_sin', 'time_of_day_cos'
    ])
    bev_tbl = read_tables_indexed(run_dir, 'bev_frames', columns=['frame_id', 'C', 'H', 'W', 'dtype', 'data'])
    fut_tbl = read_tables_indexed(run_dir, 'futures', columns=['frame_id', 'i', 'x_ego', 'y_ego', 'v_mps'])
    route_tbl = read_tables_indexed(run_dir, 'route_points', columns=['frame_id', 'idx', 'x_ego', 'y_ego', 'dx', 'dy', 'curvature', 's_frac'])
    obj_tbl = read_tables_indexed(run_dir, 'object_tokens', columns=['frame_id', 'idx', 'type_id', 'x_ego', 'y_ego', 'vx', 'vy', 'sin_yaw', 'cos_yaw'])

    # Build simple indexes by frame
    sample_frame_ids = set()
    if frames_tbl is not None and frames_tbl.num_rows > 0:
        col = frames_tbl.column('frame_id').to_pylist()
        for fid in col[:args.max_sample_frames]:
            sample_frame_ids.add(int(fid))

    # Value/range checks (sample)
    max_speed = float(meta.get('validation', {}).get('max_speed_mps', 50.0)) if 'validation' in meta else 50.0
    max_accel = float(meta.get('validation', {}).get('max_abs_accel_mps2', 12.0)) if 'validation' in meta else 12.0

    bad_speed = bad_acc = 0
    if frames_tbl is not None:
        spd = np.array(frames_tbl.column('speed_mps').to_pylist(), dtype=np.float32)
        acc_l = np.array(frames_tbl.column('accel_long').to_pylist(), dtype=np.float32)
        acc_t = np.array(frames_tbl.column('accel_lat').to_pylist(), dtype=np.float32)
        bad_speed = int(np.sum((spd < 0) | (spd > max_speed) | ~np.isfinite(spd)))
        bad_acc = int(np.sum((np.abs(acc_l) > max_accel) | (np.abs(acc_t) > max_accel) | ~np.isfinite(acc_l) | ~np.isfinite(acc_t)))
        if bad_speed > 0:
            issues.append('bad_speed_values')
        if bad_acc > 0:
            issues.append('bad_accel_values')

    # Futures shape check per frame
    if fut_tbl is not None:
        # group by frame_id
        by_frame = defaultdict(int)
        for fid in fut_tbl.column('frame_id').to_pylist():
            by_frame[int(fid)] += 1
        bad_futures = sum(1 for k, v in by_frame.items() if v != N)
        if bad_futures > 0:
            issues.append('futures_rowcount_mismatch')

    # BEV decode check on first row
    try:
        if bev_tbl is not None and bev_tbl.num_rows > 0:
            C = int(bev_tbl.column('C')[0].as_py())
            H = int(bev_tbl.column('H')[0].as_py())
            W = int(bev_tbl.column('W')[0].as_py())
            blob = bev_tbl.column('data')[0].as_buffer().to_pybytes()
            raw = zlib.decompress(blob)
            arr = np.load(__import__('io').BytesIO(raw), allow_pickle=False)
            if arr.shape != (C, H, W):
                issues.append('bev_shape_mismatch')
    except Exception:
        issues.append('bev_decode_failed')

    print('Checks:', 'OK' if not issues else 'ISSUES -> ' + ','.join(sorted(set(issues)))))


if __name__ == '__main__':
    main()

