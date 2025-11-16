# Baby-FSD

conda activate carla-env-3.7

## Usage

### Setup
1) Install Python deps:

```bash
pip install -r requirements.txt
```

2) Start CARLA server (e.g., Town10HD_Opt) on the same host/ports as in `configs/collect_bc.yaml`.

### Quick visual sim check (BehaviorAgent + traffic + walkers)
1) Start CARLA server on Town10HD_Opt (port 2000).
2) Run:

```bash
python model3_visual_confirm_test.py --host 127.0.0.1 --port 2000 --tm-port 8000 --vehicles 50 --walkers 50 --behavior normal --timeout 300
```

### Collect Behavior Cloning dataset (normalized Parquet, full obs/action schema)
1) Ensure CARLA is running (Town10HD_Opt).
2) Run collector:

```bash
python collect/collect_bc.py --config configs/collect_bc.yaml
```

Output directory: `data/BC_v1/run-YYYYmmdd-HHMMSS/`
- Parquet tables: `frames`, `route_points`, `actors`, `traffic_lights`, `map_segments`, `futures`, `object_tokens`, `bev_frames`
- Configurable via `configs/collect_bc.yaml` (fixed_dt, future_dt, K route points, N futures, windows, shards, etc.)

Obs/action mapping for BC:
- Obs
  - Ego vector scalars in `frames` (speed/accel/yaw_rate/controls/curvature/speed_limit/command, etc., `frame_ref="ego_t"`).
  - Route polyline in `route_points` (ego frame) and route corridor/centerline in `bev_frames` channels.
  - Object tokens in `object_tokens` (x,y,sin/cos yaw, size, vx/vy, type/masks).
  - BEV raster in `bev_frames` (18 channels; see `carla_utils/bev.py`), with `meters_per_px`, extents, ego origin, and spec/norms versions.
  - Traffic lights / map segments feed BEV and audits.
- Action (labels)
  - N future waypoints and speeds in `futures` as `(i, x_ego, y_ego, v_mps)` (no yaw).

### Small debug run (sanity before large collection)
Edit `configs/collect_bc.yaml` for a quick pass:
```yaml
episodes: 1
episode_max_frames: 400
npc_count: 20
no_rendering: true          # set false if you want to watch
shard_rows: 1000
max_frames: 1000
K: 32                       # route points ahead
N: 12                       # 5 Hz x 2.4 s horizon if future_dt = 0.2
future_dt: 0.2
bev:
  meters_x: 100.0
  meters_y: 80.0
  resolution_m: 0.25
```
Run:
```bash
python collect/collect_bc.py --config configs/collect_bc.yaml
```
The collector prints end-of-run checks. Ensure:
- `bev_count_mismatch` is absent
- `futures_not_divisible_by_N` is absent
- `low_route_coverage` is absent or acceptable for your scene
- No decode errors (`bev_decode_failed`, `bev_shape_mismatch`)

Offline checks (reads Parquet and re-validates):

```bash
python tools/validate_bc_run.py --root data/BC_v1
```

### Inspect one step (assemble obs/action for training)
```bash
python tools/loader_example.py --run_dir data/BC_v1/run-YYYYmmdd-HHMMSS --frame_id 0
```
Prints shapes for:
- `ego_vec` (d_ego)
- `bev_tensor` (C,H,W) with metadata in `bev_frames`
- `route_poly` (R, d_route) containing x,y,dx,dy,curvature,s_frac
- `object_tokens` (M, d_obj) + `object_mask` (M,)
- `future_waypoints` (N,2) and `future_speeds` (N,)

### Notes on routes and test-time evaluation
- During collection, the expert agent is given a start and a goal; the route plan is sampled into `route_points` (ego-frame polyline) and rendered into BEV route channels.
- For BC training, the model receives the same route context via `route_poly` and BEV route channels, and learns to output N future waypoints + speeds.
- At test time, provide a start→goal route (same sampling policy K, spacing) to construct `route_poly` and BEV route channels; the planner should follow the route using the learned policy and generalize further with RL if applied later.
