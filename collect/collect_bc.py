import argparse
import glob
import io
import json
import logging
import math
import os
import random
import time
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional

import yaml
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import zlib
import sys

import carla

# Ensure repo root on sys.path so sibling packages (data, carla_utils) resolve
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data.schema import Schemas, K_ROUTE_POINTS, N_FUTURE_STEPS, FIXED_DELTA_SECONDS, FUTURE_DELTA_SECONDS, ACTOR_RADIUS_METERS, WINDOW_METERS
from data.writer import ParquetShardWriter, ParquetDatasetWriter, DuckDBWriter, table_from_pydict
from data.hdf5_writer import HDF5EpisodeSetWriter
from data.norms import normalize_ego_vector
from carla_utils.route import (
    map_cmd_and_route,
    distance_to_goal,
    command_int_to_label,
    command_int_to_text,
    ROAD_OPTION_TO_INT,
)
from carla_utils.actors import collect_nearby_actors, get_ego_state
from carla_utils.map_vectors import extract_lane_centerline_segments, extract_traffic_light_stoplines
from carla_utils.futures import FuturesBuffer, future_waypoints_ego
from carla_utils.bev import rasterize_bev, encode_bev_to_bytes


LOG = logging.getLogger("collect_bc")

ROUTE_PRIME_MIN_TICKS = 15  # Reduced for TM autopilot (was 30)
ROUTE_PRIME_MIN_SPEED_MPS = 1.0  # Reduced for TM autopilot (was 2.0)


def configure_logging() -> None:
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
        )


def ticks_for_seconds(seconds: float, fixed_dt: float) -> int:
    return max(1, int(math.ceil(seconds / max(1e-6, fixed_dt))))


def describe_route_geometry(route_xy: List[tuple]) -> str:
    if not route_xy:
        return "no local route available"
    first = route_xy[0]
    lookahead_idx = min(len(route_xy) - 1, 7)
    far = route_xy[lookahead_idx]
    first_heading = math.degrees(math.atan2(first[1], first[0]))
    far_heading = math.degrees(math.atan2(far[1], far[0]))
    delta = ((far_heading - first_heading + 180.0) % 360.0) - 180.0
    far_dist = math.hypot(far[0], far[1])
    if abs(delta) < 15.0:
        curvature_desc = "continue straight"
    elif delta > 0:
        curvature_desc = "curving left"
    else:
        curvature_desc = "curving right"
    return (
        f"{curvature_desc} (Δheading={delta:+.1f}°, first={first_heading:+.1f}°, "
        f"lookahead={far_heading:+.1f}°, lookahead_dist={far_dist:.1f}m)"
    )


def check_route_sanity(
    route_xy: List[tuple],
    frame_id: int,
    cmd_int: int,
    plan_source: str,
) -> None:
    """Light sanity check: just warn if route is completely missing (non-fatal)."""
    if not route_xy:
        LOG.warning(
            "Frame %d: no route points available (cmd=%s, source=%s)",
            frame_id,
            command_int_to_label(cmd_int),
            plan_source,
        )


def infer_command_from_futures(wp_future: List[tuple], lane_width_m: float = 3.5) -> Optional[int]:
    """Infer a coarse command from the executed future path (ego frame)."""
    if len(wp_future) < 2:
        return None

    far_x, far_y = wp_future[-1]
    heading_deg = math.degrees(math.atan2(far_y, far_x))
    lateral = far_y

    # Lane change: big lateral shift with small heading change
    if abs(lateral) > lane_width_m * 0.6 and abs(heading_deg) < 15.0:
        return ROAD_OPTION_TO_INT["ChangeLaneLeft"] if lateral > 0 else ROAD_OPTION_TO_INT["ChangeLaneRight"]

    if heading_deg > 20.0:
        return ROAD_OPTION_TO_INT["Left"]
    if heading_deg < -20.0:
        return ROAD_OPTION_TO_INT["Right"]
    if abs(heading_deg) < 10.0:
        return ROAD_OPTION_TO_INT["LaneFollow"]
    return ROAD_OPTION_TO_INT["Straight"]


def set_sync(world: carla.World, tm: carla.TrafficManager, enable: bool, fixed_delta: float, no_rendering: bool) -> None:
    s = world.get_settings()
    s.synchronous_mode = enable
    s.fixed_delta_seconds = fixed_delta if enable else None
    s.no_rendering_mode = no_rendering
    world.apply_settings(s)
    tm.set_synchronous_mode(enable)


def pick_far_spawn_pair(world: carla.World):
    sps = list(world.get_map().get_spawn_points())
    a = random.choice(sps)
    b = max(sps, key=lambda sp: (sp.location.x - a.location.x) ** 2 + (sp.location.y - a.location.y) ** 2)
    if a == b:
        b = random.choice([sp for sp in sps if sp != a])
    return a, b


def spawn_model3(world: carla.World, transform: carla.Transform, max_retries: int = 10) -> carla.Vehicle:
    bp = world.get_blueprint_library().find("vehicle.tesla.model3")
    bp.set_attribute("role_name", "hero")
    
    # Try the requested transform first
    v = world.try_spawn_actor(bp, transform)
    if v:
        return v
    
    # If that fails, try alternative spawn points
    spawn_points = list(world.get_map().get_spawn_points())
    random.shuffle(spawn_points)
    
    for i, sp in enumerate(spawn_points[:max_retries]):
        v = world.try_spawn_actor(bp, sp)
        if v:
            LOG.info("Spawned ego at alternative spawn point (attempt %d)", i + 2)
            return v
        # Give the world a tick to settle between attempts
        world.tick()
    
    raise RuntimeError(f"Failed to spawn ego vehicle after {max_retries + 1} attempts")


def make_run_dir(base_out: str) -> str:
    run_id = datetime.utcnow().strftime("run-%Y%m%d-%H%M%S")
    out = os.path.join(base_out, run_id)
    os.makedirs(out, exist_ok=True)
    return out


def write_meta(out_dir: str, meta: Dict) -> None:
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def build_hdf5_frame(rec: Dict, wp_future: List, vel_future: List, K: int, N: int, M: int) -> Dict:
    """
    Build a single frame dict for HDF5 output.
    
    Args:
        rec: Raw frame record with ego state, BEV, route, etc.
        wp_future: Future waypoints (N, 2)
        vel_future: Future speeds (N,)
        K: Number of route points
        N: Number of future steps
        M: Maximum number of objects
        
    Returns:
        Dict with normalized arrays ready for HDF5 writing
    """
    from types import SimpleNamespace
    
    # Build ego vector (14-D normalized)
    ego_row = SimpleNamespace(
        speed_mps=rec["speed_mps"],
        yaw_rate=rec["yaw_rate"],
        accel_long=rec["accel_long"],
        accel_lat=rec["accel_lat"],
        curvature=rec["curvature"],
        steer_angle_rad=rec["steer_angle_rad"],
        steer_norm=rec["steer_norm"],
        throttle=rec["throttle"],
        brake=rec["brake"],
        speed_limit_mps=rec["speed_limit_mps"],
        command=rec["command"],
        gear=rec["gear"],
        time_of_day_sin=rec["time_of_day_sin"],
        time_of_day_cos=rec["time_of_day_cos"],
    )
    ego_vec = normalize_ego_vector(ego_row).numpy()
    
    # BEV tensor - decode from bytes
    bev_bytes = rec["bev_bytes"]
    raw = zlib.decompress(bev_bytes)
    bev_arr = np.load(io.BytesIO(raw), allow_pickle=False).astype(np.float32)
    
    # Route points - pad/truncate to K
    route_xy = rec["route_xy"]
    route_arr = np.zeros((K, 2), dtype=np.float32)
    for i, (x, y) in enumerate(route_xy[:K]):
        route_arr[i] = [x, y]
    
    # Object tokens - pad/truncate to M
    actors_list = rec["actors"]
    objects_arr = np.zeros((M, 11), dtype=np.float32)
    object_mask = np.zeros((M,), dtype=np.float32)
    for i, actor in enumerate(actors_list[:M]):
        # Build object token vector
        yaw_rad = np.radians(actor["yaw"])
        objects_arr[i] = [
            actor["type_id"],
            actor["x_ego"],
            actor["y_ego"],
            np.sin(yaw_rad),
            np.cos(yaw_rad),
            actor["length"],
            actor["width"],
            actor["vx"],
            actor["vy"],
            actor.get("oncoming_flag", 0),
            actor.get("priority_flag", 0),
        ]
        object_mask[i] = 1.0
    
    # Future waypoints and speeds - pad/truncate to N
    future_xy = np.zeros((N, 2), dtype=np.float32)
    future_v = np.zeros((N,), dtype=np.float32)
    future_mask = np.zeros((N,), dtype=np.float32)
    
    n_futures = min(len(wp_future), N)
    for i in range(n_futures):
        future_xy[i] = wp_future[i]
        future_v[i] = vel_future[i]
        future_mask[i] = 1.0
    
    return {
        "frame_id": rec["frame_id"],
        "ego_vec": ego_vec,
        "bev": bev_arr,
        "route": route_arr,
        "objects": objects_arr,
        "object_mask": object_mask,
        "future_xy": future_xy,
        "future_v": future_v,
        "future_mask": future_mask,
    }


def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join("configs", "collect_bc.yaml"))
    args = ap.parse_args()

    configure_logging()
    LOG.info("Starting BC collection with config file %s", args.config)

    cfg = load_config(args.config)
    host = cfg.get("host", "10.0.0.121")
    port = int(cfg.get("port", 2000))
    tm_port = int(cfg.get("tm_port", 8000))
    out_root = cfg.get("out_dir", os.path.join("data", "BC_v1"))

    fixed_dt = float(cfg.get("fixed_dt", FIXED_DELTA_SECONDS))
    future_dt = float(cfg.get("future_dt", FUTURE_DELTA_SECONDS))
    K = int(cfg.get("K", K_ROUTE_POINTS))
    N = int(cfg.get("N", N_FUTURE_STEPS))
    radius = float(cfg.get("actor_radius", ACTOR_RADIUS_METERS))
    window_m = float(cfg.get("window_m", WINDOW_METERS))
    no_rendering = bool(cfg.get("no_rendering", False))
    spectator_follow = bool(cfg.get("spectator_follow", False))
    spectator_mode = cfg.get("spectator_mode", "chase")  # "chase" | "fpv"

    # Storage and frame limits
    shard_rows = int(cfg.get("shard_rows", 50_000))
    max_frames = int(cfg.get("max_frames", 100_000))

    # Episodes and NPCs
    episodes = int(cfg.get("episodes", 1))
    episode_max_frames = int(cfg.get("episode_max_frames", max_frames))
    npc_count = int(cfg.get("npc_count", 0))
    behavior = cfg.get("behavior", "normal")
    wheelbase_m = float(cfg.get("wheelbase_m", 2.875))

    # BEV config
    bev_cfg = cfg.get("bev", {}) or {}
    bev_mx = float(bev_cfg.get("meters_x", 100.0))
    bev_my = float(bev_cfg.get("meters_y", 80.0))
    bev_res = float(bev_cfg.get("resolution_m", 0.25))

    # Default speed limit (m/s) if the map reports 0
    speed_limit_default_mps = float(cfg.get("speed_limit_default_mps", 13.9))  # ~50 km/h

    # Futures buffer must hold the whole episode so early frames keep their horizon
    horizon_ticks = int(math.ceil(N * future_dt / max(fixed_dt, 1e-6)))
    future_buffer_capacity = max(
        128,
        episode_max_frames + horizon_ticks + 16,
        horizon_ticks * 5,
    )

    client = carla.Client(host, port)
    client.set_timeout(20.0)
    world = client.get_world()
    tm = client.get_trafficmanager(tm_port)
    # Align Traffic Manager settings with visual confirm test
    try:
        tm.set_global_distance_to_leading_vehicle(1.0)
        tm.set_hybrid_physics_mode(True)
        tm.global_percentage_speed_difference(10.0)  # NPCs slightly slower
    except Exception:
        pass

    original_settings = world.get_settings()
    original_sync = original_settings.synchronous_mode

    run_dir = make_run_dir(out_root)

    # Parquet writer options (compression configurable via YAML)
    pq_cfg = cfg.get("parquet", {}) or {}
    pq_compression = pq_cfg.get("compression", "snappy")
    # Fallback if codec is unavailable on this platform
    try:
        if hasattr(pa, "Codec") and not pa.Codec.is_available(pq_compression):
            print(f"Warning: parquet codec '{pq_compression}' unavailable; falling back to 'gzip'")
            pq_compression = "gzip"
    except Exception:
        pass

    # Storage backend
    storage_cfg = cfg.get("storage", {}) or {}
    backend = (storage_cfg.get("backend", "hdf5") or "hdf5").lower()

    # Writers per table
    if backend == "hdf5":
        # HDF5 writer (v2, optimized)
        episodes_per_set = int(storage_cfg.get("episodes_per_set", 5))
        compression = storage_cfg.get("compression", "lz4")
        chunk_size = int(storage_cfg.get("chunk_size", 100))
        
        run_id = os.path.basename(run_dir)
        hdf5_writer = HDF5EpisodeSetWriter(
            output_dir=run_dir,
            run_id=run_id,
            episodes_per_set=episodes_per_set,
            K=K,
            N_future=N,
            M=64,  # max_objects
            C=18,
            # Match rasterize_bev output: W=meters_x/res, H=meters_y/res
            H=int(round(bev_my / bev_res)),  # BEV height (rows)
            W=int(round(bev_mx / bev_res)),  # BEV width (cols)
            compression=compression,
            chunk_size=chunk_size,
        )
        LOG.info("Using HDF5 backend (v2) with %d episodes per set, %s compression", episodes_per_set, compression)
        frames_w = route_w = actors_w = tls_w = maps_w = fut_w = obj_w = bev_w = None
    elif backend == "duckdb":
        db_path = os.path.join(run_dir, "bc.duckdb")
        frames_w = DuckDBWriter(run_dir, "frames", Schemas.FRAMES, db_path)
        route_w = DuckDBWriter(run_dir, "route_points", Schemas.ROUTE_POINTS, db_path)
        actors_w = DuckDBWriter(run_dir, "actors", Schemas.ACTORS, db_path)
        tls_w = DuckDBWriter(run_dir, "traffic_lights", Schemas.TRAFFIC_LIGHTS, db_path)
        maps_w = DuckDBWriter(run_dir, "map_segments", Schemas.MAP_SEGMENTS, db_path)
        fut_w = DuckDBWriter(run_dir, "futures", Schemas.FUTURES, db_path)
        obj_w = DuckDBWriter(run_dir, "object_tokens", Schemas.OBJECT_TOKENS, db_path)
        bev_w = DuckDBWriter(run_dir, "bev_frames", Schemas.BEV_FRAMES, db_path)
    else:
        frames_w = ParquetDatasetWriter(os.path.join(run_dir), "frames", Schemas.FRAMES, compression=pq_compression)
        route_w = ParquetDatasetWriter(os.path.join(run_dir), "route_points", Schemas.ROUTE_POINTS, compression=pq_compression)
        actors_w = ParquetDatasetWriter(os.path.join(run_dir), "actors", Schemas.ACTORS, compression=pq_compression)
        tls_w = ParquetDatasetWriter(os.path.join(run_dir), "traffic_lights", Schemas.TRAFFIC_LIGHTS, compression=pq_compression)
        maps_w = ParquetDatasetWriter(os.path.join(run_dir), "map_segments", Schemas.MAP_SEGMENTS, compression=pq_compression)
        fut_w = ParquetDatasetWriter(os.path.join(run_dir), "futures", Schemas.FUTURES, compression=pq_compression)
        obj_w = ParquetDatasetWriter(os.path.join(run_dir), "object_tokens", Schemas.OBJECT_TOKENS, compression=pq_compression)
        bev_w = ParquetDatasetWriter(os.path.join(run_dir), "bev_frames", Schemas.BEV_FRAMES, compression=pq_compression)

    ego = None
    goal_tf = None
    goal_start_dist_m = 0.0
    futures = FuturesBuffer(capacity=future_buffer_capacity)

    # Metrics / integrity counters
    frames_rows = 0
    route_rows = 0
    actors_rows = 0
    tls_rows = 0
    map_rows = 0
    futures_rows = 0
    episodes_reached = 0

    npc_vehicles = []
    try:
        set_sync(world, tm, True, fixed_dt, no_rendering)
        # Optional NPC traffic
        def spawn_npc_traffic(world: carla.World, tm_port: int, count: int):
            bps = world.get_blueprint_library().filter('vehicle.*')
            sps = world.get_map().get_spawn_points()
            random.shuffle(sps)
            spawned = []
            for sp in sps[:min(count, len(sps))]:
                vbp = random.choice(bps)
                v = world.try_spawn_actor(vbp, sp)
                if v:
                    v.set_autopilot(True, tm_port)
                    spawned.append(v)
            world.tick()
            return spawned

        # Defer NPC spawning until after ego is spawned (to avoid blocking ego spawn)
        npc_vehicles = []

        frame_id = 0
        start_wall = time.time()

        # Write run meta once
        meta = {
            "carla_version": getattr(carla, "__version__", "unknown"),
            "map": world.get_map().name,
            "K": K,
            "N": N,
            "fixed_dt": fixed_dt,
            "future_dt": future_dt,
            "radius": radius,
            "window_m": window_m,
            "no_rendering": no_rendering,
            "behavior": behavior,
            "bev": {"meters_x": bev_mx, "meters_y": bev_my, "resolution_m": bev_res},
        }
        write_meta(run_dir, meta)

        spectator = world.get_spectator() if not no_rendering and spectator_follow else None

        for ep in range(episodes):
            # Reset ego and route
            if ego is not None:
                try:
                    ego.destroy()
                    # Give the world time to process the destruction
                    for _ in range(3):
                        world.tick()
                except Exception:
                    pass
            start_tf, end_tf = pick_far_spawn_pair(world)
            ego = spawn_model3(world, start_tf)

            # Hand ego to Traffic Manager autopilot (canonical CARLA way)
            ego.set_autopilot(True, tm_port)

            # Warm-up: give TM time to compute route and start accelerating
            for _ in range(20):
                world.tick()

            # Spawn background NPC traffic once per run, after ego is in the world
            if not npc_vehicles and npc_count > 0:
                npc_vehicles = spawn_npc_traffic(world, tm_port, npc_count)

            goal_tf = end_tf
            # Record starting goal distance for route progress estimation
            try:
                goal_start_dist_m = distance_to_goal(ego.get_transform(), goal_tf)
            except Exception:
                goal_start_dist_m = 0.0
            
            LOG.info("=== Episode %d/%d: Starting (goal distance: %.1fm) ===", ep + 1, episodes, goal_start_dist_m)

            # Reset futures buffer per episode
            futures = FuturesBuffer(capacity=future_buffer_capacity)

            ep_frames = 0
            # pending frame queue for lagged future labels
            # each item holds the extracted features for that frame time
            pending = deque()
            recording_ready = False
            prime_ticks = 0
            # Simple per-episode quality counters
            ep_stats = {
                "written": 0,
                "skipped_future_missing": 0,
                "skipped_stationary": 0,
                "kept_stationary": 0,  # Track how many stationary frames we kept
                "skipped_behind": 0,
                "skipped_jump": 0,
                "command_disagree": 0,
                "skipped_action_mismatch": 0,
                "speed_sum": 0.0,
                "speed_sq_sum": 0.0,
                "speed_count": 0,
                "cmd_hist": {i: 0 for i in range(6)},
                "speed_limit_fallback": 0,
            }
            
            # Stopping frame configuration from config
            stopping_cfg = cfg.get("stopping", {})
            keep_stationary_ratio = float(stopping_cfg.get("keep_stationary_ratio", 0.15))
            max_consecutive_stationary = int(stopping_cfg.get("max_consecutive_stationary", 5))
            keep_near_traffic_light = bool(stopping_cfg.get("keep_near_traffic_light", True))
            traffic_light_radius_m = float(stopping_cfg.get("traffic_light_radius_m", 15.0))
            consecutive_stationary_count = 0

            horizon_ticks = int(math.ceil(N * future_dt / max(fixed_dt, 1e-6)))

            while frame_id < max_frames and ep_frames < episode_max_frames:
                world.tick()
                snapshot = world.get_snapshot()
                sim_time = float(getattr(getattr(snapshot, "timestamp", None), "elapsed_seconds", ep_frames * fixed_dt))

                # Extract current state FIRST (for this frame)
                speed_mps, yaw_rate = get_ego_state(ego)
                ego_tf = ego.get_transform()

                # Route / command from map only (no BehaviorAgent internals)
                cmd_int, route_xy, plan_meta = map_cmd_and_route(
                    world,
                    ego_tf,
                    K,
                    spacing_m=2.0,
                    return_plan_stats=True,
                )
                plan_len = plan_meta.get("plan_len", 0)
                buffer_len = plan_len  # no separate buffer concept with pure map-based route
                plan_source = plan_meta.get("plan_source", "map")

                goal_dist_m = 0.0
                if goal_tf is not None:
                    try:
                        goal_dist_m = distance_to_goal(ego_tf, goal_tf)
                    except Exception:
                        goal_dist_m = 0.0

                # Simple recording gate: wait until ego is moving and we have a route
                if not recording_ready:
                    if speed_mps >= ROUTE_PRIME_MIN_SPEED_MPS and route_xy:
                        prime_ticks += 1
                        if prime_ticks >= ROUTE_PRIME_MIN_TICKS:
                            recording_ready = True
                            LOG.info("  Recording started at frame %d (speed=%.1f m/s)", frame_id, speed_mps)
                    else:
                        prime_ticks = 0

                # Light sanity check (non-fatal warning only)
                if recording_ready:
                    check_route_sanity(route_xy, frame_id, cmd_int, plan_source)

                # Traffic Manager autopilot already computed and applied control.
                # We only READ what it did for BC labels.
                ctrl = ego.get_control()
                steer_norm = float(getattr(ctrl, "steer", 0.0))
                throttle = float(getattr(ctrl, "throttle", 0.0))
                brake = float(getattr(ctrl, "brake", 0.0))
                gear = int(getattr(ctrl, "gear", 0))

                # Optional: treat reaching very close to the goal as episode completion.
                # Only check after recording starts to avoid premature episode endings.
                if recording_ready and goal_tf is not None and goal_dist_m <= 5.0:
                    try:
                        ego.apply_control(
                            carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0, hand_brake=False)
                        )
                    except Exception:
                        pass
                    episodes_reached += 1
                    LOG.info("  Destination reached! (%.1fm from goal)", goal_dist_m)
                    break

                # Spectator follow (visual debug) - do this after control to keep latency low
                if spectator is not None:
                    yaw_deg = ego_tf.rotation.yaw
                    yaw = math.radians(yaw_deg)
                    if spectator_mode == "fpv":
                        forward = 0.7
                        height = 1.4
                        dx = forward * math.cos(yaw)
                        dy = forward * math.sin(yaw)
                        loc = ego_tf.location + carla.Location(x=dx, y=dy, z=height)
                        rot = carla.Rotation(pitch=0.0, yaw=yaw_deg)
                    else:
                        back = 8.0
                        height = 4.0
                        dx = -back * math.cos(yaw)
                        dy = -back * math.sin(yaw)
                        loc = ego_tf.location + carla.Location(x=dx, y=dy, z=height)
                        rot = carla.Rotation(pitch=-10.0, yaw=yaw_deg)
                    spectator.set_transform(carla.Transform(loc, rot))

                # Ego kinematics (extended)
                acc_world = ego.get_acceleration()
                yaw_rad = math.radians(ego_tf.rotation.yaw)
                c = math.cos(-yaw_rad)
                s = math.sin(-yaw_rad)
                ax_ego = acc_world.x * c - acc_world.y * s
                ay_ego = acc_world.x * s + acc_world.y * c
                curvature = float(yaw_rate / max(speed_mps, 0.5))
                steer_angle_rad = float(math.atan(wheelbase_m * curvature))
                wp = world.get_map().get_waypoint(ego_tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
                speed_limit_mps = float(getattr(wp, "speed_limit", 0.0) or 0.0)
                if speed_limit_mps > 30.0:
                    speed_limit_mps = speed_limit_mps / 3.6
                elif speed_limit_mps <= 0.0:
                    speed_limit_mps = speed_limit_default_mps
                    ep_stats["speed_limit_fallback"] += 1
                try:
                    sun_az = float(world.get_weather().sun_azimuth_angle)
                except Exception:
                    sun_az = 0.0
                ang = math.radians(sun_az % 360.0)
                tod_sin = math.sin(ang)
                tod_cos = math.cos(ang)

                # Heavy feature extraction happens after control to avoid latency
                actors = collect_nearby_actors(world, ego, radius)
                map_polys = extract_lane_centerline_segments(world.get_map(), ego_tf, window_m=window_m, step_m=2.0)
                tls = extract_traffic_light_stoplines(world, ego_tf, window_m=window_m)
                bev_tensor, bev_meta = rasterize_bev(route_xy, map_polys, tls, actors, bev_cfg)
                bev_bytes = encode_bev_to_bytes(bev_tensor)

                # Store frame data after all features are computed
                frame_data = {
                    "frame_id": frame_id,
                    "sim_time": sim_time,
                    "ego_tf": ego_tf,
                    "goal_dist_m": goal_dist_m,
                    "speed_mps": speed_mps,
                    "yaw_rate": yaw_rate,
                    "accel_long": float(ax_ego),
                    "accel_lat": float(ay_ego),
                    "steer_angle_rad": float(steer_angle_rad),
                    "curvature": float(curvature),
                    "speed_limit_mps": float(speed_limit_mps),
                    "command": int(cmd_int),
                    "time_of_day_sin": float(tod_sin),
                    "time_of_day_cos": float(tod_cos),
                    "route_id": 0,
                    "scenario_id": 0,
                    "cmd_int": cmd_int,
                    "route_xy": route_xy,
                    "actors": actors,
                    "map_polys": map_polys,
                    "tls": tls,
                    "bev_meta": bev_meta,
                    "bev_bytes": bev_bytes,
                    "steer_norm": steer_norm,
                    "throttle": throttle,
                    "brake": brake,
                    "gear": gear,
                }

                # Quick action/state sanity: skip frames where control conflicts with kinematics
                action_ok = True
                if speed_mps > 0.5 and brake > 0.5 and throttle > 0.2:
                    action_ok = False  # conflicting accel/brake
                # if speed is near zero but strong throttle with no acceleration could be fine; keep simple
                if not action_ok:
                    ep_stats["skipped_action_mismatch"] += 1
                    ep_frames += 1
                    frame_id += 1
                    continue

                # Buffer current state for future trajectory labels BEFORE queuing
                if recording_ready:
                    futures.add(sim_time, ego_tf, speed_mps)
                    pending.append(frame_data)
                
                frame_id += 1
                ep_frames += 1

            # After episode loop, capture extra horizon ticks to fill futures for tail frames
            for _ in range(horizon_ticks):
                world.tick()
                snapshot = world.get_snapshot()
                sim_time = float(getattr(getattr(snapshot, "timestamp", None), "elapsed_seconds", ep_frames * fixed_dt))
                speed_mps, _ = get_ego_state(ego)
                ego_tf = ego.get_transform()
                futures.add(sim_time, ego_tf, speed_mps)
                ep_frames += 1

            # After episode loop: flush remaining pending frames
            # Process any frames that have sufficient future data
            LOG.info("  Writing %d frames to disk...", len(pending))
            frames_written = 0
            frames_skipped = 0
            
            # For HDF5, batch episode frames
            if backend == "hdf5":
                episode_frames = []
            
            while pending:
                rec = pending.popleft()
                # Check if we have enough future data
                deltas = [future_dt * (i + 1) for i in range(N)]
                samples = futures.sample_future(rec["sim_time"], deltas)
                
                # If we don't have full future trajectory, skip this frame
                # (last few frames of each episode won't have complete futures)
                if not samples or len(samples) < N or any(s is None for s in samples):
                    frames_skipped += 1
                    ep_stats["skipped_future_missing"] += 1
                    continue
                    
                base_tf = rec["ego_tf"]
                wp_future, vel_future = future_waypoints_ego(base_tf, samples)

                # Detect stationary frames but keep some for learning to stop
                max_step = 0.0
                if len(wp_future) >= 2:
                    max_step = max(
                        math.hypot(wp_future[i][0] - wp_future[i - 1][0], wp_future[i][1] - wp_future[i - 1][1])
                        for i in range(1, len(wp_future))
                    )
                total_disp = 0.0
                if wp_future:
                    total_disp = math.hypot(wp_future[-1][0] - wp_future[0][0], wp_future[-1][1] - wp_future[0][1])
                max_speed = max(vel_future) if vel_future else 0.0
                max_forward = max((pt[0] for pt in wp_future), default=0.0)
                
                is_stationary = (max_step < 0.05 and total_disp < 0.2 and max_speed < 0.2)
                
                if is_stationary:
                    consecutive_stationary_count += 1
                    
                    # Decide whether to keep this stationary frame
                    keep_this_stationary = False
                    
                    # Check if near traffic light (always keep these - learning to stop at lights)
                    if keep_near_traffic_light:
                        try:
                            tl = ego.get_traffic_light()
                            if tl is not None:
                                tl_loc = tl.get_transform().location
                                ego_loc = rec["ego_tf"].location
                                tl_dist = math.hypot(ego_loc.x - tl_loc.x, ego_loc.y - tl_loc.y)
                                if tl_dist < traffic_light_radius_m:
                                    keep_this_stationary = True
                        except Exception:
                            pass
                    
                    # Keep some stationary frames based on ratio (but not too many consecutive)
                    if not keep_this_stationary and consecutive_stationary_count <= max_consecutive_stationary:
                        if random.random() < keep_stationary_ratio:
                            keep_this_stationary = True
                    
                    if not keep_this_stationary:
                        frames_skipped += 1
                        ep_stats["skipped_stationary"] += 1
                        continue
                    else:
                        ep_stats["kept_stationary"] += 1
                else:
                    consecutive_stationary_count = 0  # Reset counter when moving
                # Drop trajectories that sit entirely behind ego (likely bad transform/sample)
                if max_forward < -1.0:
                    frames_skipped += 1
                    ep_stats["skipped_behind"] += 1
                    continue

                # Additional future sanity: drop if any step is a huge jump (likely bad sample)
                if len(wp_future) >= 2:
                    max_jump = max(
                        math.hypot(wp_future[i][0] - wp_future[i - 1][0], wp_future[i][1] - wp_future[i - 1][1])
                        for i in range(1, len(wp_future))
                    )
                    if max_jump > 60.0:  # ~>3 car lengths per 0.2s is unreasonable
                        frames_skipped += 1
                        ep_stats["skipped_jump"] += 1
                        continue

                # Align command intent check: keep map-based command but record disagreement
                cmd_from_future = infer_command_from_futures(wp_future)
                rec["exec_command"] = cmd_from_future
                if cmd_from_future is not None and cmd_from_future != rec["command"]:
                    ep_stats["command_disagree"] += 1

                # Per-frame quality metrics
                ep_stats["speed_sum"] += rec["speed_mps"]
                ep_stats["speed_sq_sum"] += rec["speed_mps"] ** 2
                ep_stats["speed_count"] += 1
                cmd_key = int(rec.get("command", 0))
                if cmd_key in ep_stats["cmd_hist"]:
                    ep_stats["cmd_hist"][cmd_key] += 1
                ep_stats["written"] += 1

                # HDF5 backend: batch frames for episode-level write
                if backend == "hdf5":
                    frame_dict = build_hdf5_frame(rec, wp_future, vel_future, K, N, 64)
                    episode_frames.append(frame_dict)
                    frames_written += 1
                    if frames_written % 100 == 0:
                        LOG.info("    Progress: %d frames processed...", frames_written)
                    continue
                
                # Legacy Parquet/DuckDB backends below
                # Frames row
                cur_goal_dist = rec.get("goal_dist_m", None)
                if cur_goal_dist is None:
                    try:
                        cur_goal_dist = distance_to_goal(rec["ego_tf"], goal_tf)
                    except Exception:
                        cur_goal_dist = 0.0
                route_progress_m = max(0.0, float(goal_start_dist_m - cur_goal_dist))
                frames_tbl = table_from_pydict(
                    Schemas.FRAMES,
                    {
                        "run_id": [os.path.basename(run_dir)],
                        "shard_id": [1],
                        "frame_id": [rec["frame_id"]],
                        "sim_time": [rec["sim_time"]],
                        "fixed_dt": [fixed_dt],
                        "future_dt": [future_dt],
                        "map": [world.get_map().name],
                        "weather": [0],
                        "seed": [0],
                        "ego_x_w": [rec["ego_tf"].location.x],
                        "ego_y_w": [rec["ego_tf"].location.y],
                        "ego_yaw_w": [rec["ego_tf"].rotation.yaw],
                        "speed_mps": [rec["speed_mps"]],
                        "yaw_rate": [rec["yaw_rate"]],
                        "road_option": [rec["cmd_int"]],
                        "goal_dist_m": [cur_goal_dist],
                        "at_junction": [1 if world.get_map().get_waypoint(rec["ego_tf"].location).is_junction else 0],
                        "dist_to_next_junction": [0.0],
                        "accel_long": [rec["accel_long"]],
                        "accel_lat": [rec["accel_lat"]],
                        "steer_angle_rad": [rec["steer_angle_rad"]],
                        "steer_norm": [rec["steer_norm"]],
                        "throttle": [rec["throttle"]],
                        "brake": [rec["brake"]],
                        "curvature": [rec["curvature"]],
                        "gear": [rec["gear"]],
                        "speed_limit_mps": [rec["speed_limit_mps"]],
                        "command": [rec["command"]],
                        "time_of_day_sin": [rec["time_of_day_sin"]],
                        "time_of_day_cos": [rec["time_of_day_cos"]],
                        "route_progress_m": [route_progress_m],
                        "route_id": [rec["route_id"]],
                        "scenario_id": [rec["scenario_id"]],
                        "frame_ref": ["ego_t"],
                    },
                )
                frames_w.append_table(frames_tbl)
                frames_rows += 1

                # Route points rows
                if rec["route_xy"]:
                    xs = [float(p[0]) for p in rec["route_xy"]]
                    ys = [float(p[1]) for p in rec["route_xy"]]
                    dxs = []
                    dys = []
                    curvs = []
                    s = [0.0]
                    for i in range(len(xs)):
                        if i == 0:
                            dx = xs[1] - xs[0] if len(xs) > 1 else 0.0
                            dy = ys[1] - ys[0] if len(ys) > 1 else 0.0
                        else:
                            dx = xs[i] - xs[i - 1]
                            dy = ys[i] - ys[i - 1]
                            s.append(s[-1] + math.hypot(dx, dy))
                        dxs.append(float(dx))
                        dys.append(float(dy))
                    s_total = s[-1] if s else 1.0
                    s_frac = [float(si / s_total) if s_total > 0 else 0.0 for si in s]
                    for i in range(len(xs)):
                        if 0 < i < len(xs) - 1:
                            x0, y0 = xs[i - 1], ys[i - 1]
                            x1, y1 = xs[i], ys[i]
                            x2, y2 = xs[i + 1], ys[i + 1]
                            a1x, a1y = x1 - x0, y1 - y0
                            a2x, a2y = x2 - x1, y2 - y1
                            cross = abs(a1x * a2y - a1y * a2x)
                            denom = (math.hypot(a1x, a1y) * math.hypot(a2x, a2y)) + 1e-6
                            curv = cross / denom
                        else:
                            curv = 0.0
                        curvs.append(float(curv))

                    route_tbl = table_from_pydict(
                        Schemas.ROUTE_POINTS,
                        {
                            "run_id": [os.path.basename(run_dir)] * len(rec["route_xy"]),
                            "shard_id": [1] * len(rec["route_xy"]),
                            "frame_id": [rec["frame_id"]] * len(rec["route_xy"]),
                            "idx": list(range(len(rec["route_xy"]))),
                            "x_ego": xs,
                            "y_ego": ys,
                            "dx": dxs,
                            "dy": dys,
                            "curvature": curvs,
                            "s_frac": s_frac,
                        },
                    )
                    route_w.append_table(route_tbl)
                    route_rows += len(rec["route_xy"])

                # Actors rows
                if rec["actors"]:
                    actors_tbl = table_from_pydict(
                        Schemas.ACTORS,
                        {
                            "run_id": [os.path.basename(run_dir)] * len(rec["actors"]),
                            "shard_id": [1] * len(rec["actors"]),
                            "frame_id": [rec["frame_id"]] * len(rec["actors"]),
                            "actor_id": [a["actor_id"] for a in rec["actors"]],
                            "type_id": [a["type_id"] for a in rec["actors"]],
                            "x_ego": [a["x_ego"] for a in rec["actors"]],
                            "y_ego": [a["y_ego"] for a in rec["actors"]],
                            "yaw": [a["yaw"] for a in rec["actors"]],
                            "vx": [a["vx"] for a in rec["actors"]],
                            "vy": [a["vy"] for a in rec["actors"]],
                            "length": [a["length"] for a in rec["actors"]],
                            "width": [a["width"] for a in rec["actors"]],
                        },
                    )
                    actors_w.append_table(actors_tbl)
                    actors_rows += len(rec["actors"])

                    # Object tokens table
                    obj_tbl = table_from_pydict(
                        Schemas.OBJECT_TOKENS,
                        {
                            "run_id": [os.path.basename(run_dir)] * len(rec["actors"]),
                            "shard_id": [1] * len(rec["actors"]),
                            "frame_id": [rec["frame_id"]] * len(rec["actors"]),
                            "idx": list(range(len(rec["actors"]))),
                            "actor_id": [a["actor_id"] for a in rec["actors"]],
                            "type_id": [a["type_id"] for a in rec["actors"]],
                            "x_ego": [a["x_ego"] for a in rec["actors"]],
                            "y_ego": [a["y_ego"] for a in rec["actors"]],
                            "sin_yaw": [a.get("sin_yaw", 0.0) for a in rec["actors"]],
                            "cos_yaw": [a.get("cos_yaw", 1.0) for a in rec["actors"]],
                            "length": [a["length"] for a in rec["actors"]],
                            "width": [a["width"] for a in rec["actors"]],
                            "vx": [a["vx"] for a in rec["actors"]],
                            "vy": [a["vy"] for a in rec["actors"]],
                            "oncoming_flag": [a.get("oncoming_flag", 0) for a in rec["actors"]],
                            "priority_flag": [a.get("priority_flag", 0) for a in rec["actors"]],
                        },
                    )
                    obj_w.append_table(obj_tbl)

                # TL rows
                if rec["tls"]:
                    tls_tbl = table_from_pydict(
                        Schemas.TRAFFIC_LIGHTS,
                        {
                            "run_id": [os.path.basename(run_dir)] * len(rec["tls"]),
                            "shard_id": [1] * len(rec["tls"]),
                            "frame_id": [rec["frame_id"]] * len(rec["tls"]),
                            "tl_id": [t["tl_id"] for t in rec["tls"]],
                            "state": [t["state"] for t in rec["tls"]],
                            "stop_x_ego": [t["stop_x_ego"] for t in rec["tls"]],
                            "stop_y_ego": [t["stop_y_ego"] for t in rec["tls"]],
                        },
                    )
                    tls_w.append_table(tls_tbl)
                    tls_rows += len(rec["tls"])

                # Map polyline rows
                if rec["map_polys"]:
                    maps_tbl = table_from_pydict(
                        Schemas.MAP_SEGMENTS,
                        {
                            "run_id": [os.path.basename(run_dir)] * len(rec["map_polys"]),
                            "shard_id": [1] * len(rec["map_polys"]),
                            "frame_id": [rec["frame_id"]] * len(rec["map_polys"]),
                            "seg_id": [p["seg_id"] for p in rec["map_polys"]],
                            "kind": [p["kind"] for p in rec["map_polys"]],
                            "coords_x": [p["coords_x"] for p in rec["map_polys"]],
                            "coords_y": [p["coords_y"] for p in rec["map_polys"]],
                            "lane_id": [p.get("lane_id", 0) for p in rec["map_polys"]],
                            "lane_type": [p.get("lane_type", 0) for p in rec["map_polys"]],
                            "speed_limit": [p.get("speed_limit", 0.0) for p in rec["map_polys"]],
                        },
                    )
                    maps_w.append_table(maps_tbl)
                    map_rows += len(rec["map_polys"])

                # Futures rows
                if wp_future:
                    fut_tbl = table_from_pydict(
                        Schemas.FUTURES,
                        {
                            "run_id": [os.path.basename(run_dir)] * len(wp_future),
                            "shard_id": [1] * len(wp_future),
                            "frame_id": [rec["frame_id"]] * len(wp_future),
                            "i": list(range(len(wp_future))),
                            "x_ego": [p[0] for p in wp_future],
                            "y_ego": [p[1] for p in wp_future],
                            "v_mps": vel_future,
                        },
                    )
                    fut_w.append_table(fut_tbl)
                    futures_rows += len(wp_future)

                # BEV frame row
                bev_meta = rec["bev_meta"]
                bev_meta_aug = dict(bev_meta)
                bev_meta_aug.update({
                    "ego_x_w": float(rec["ego_tf"].location.x),
                    "ego_y_w": float(rec["ego_tf"].location.y),
                    "ego_yaw_w": float(rec["ego_tf"].rotation.yaw),
                    "frame_ref": "ego_t",
                })
                bev_tbl = table_from_pydict(
                    Schemas.BEV_FRAMES,
                    {
                        "run_id": [os.path.basename(run_dir)],
                        "shard_id": [1],
                        "frame_id": [rec["frame_id"]],
                        "C": [bev_meta_aug["C"]],
                        "H": [bev_meta_aug["H"]],
                        "W": [bev_meta_aug["W"]],
                        "dtype": [bev_meta_aug["dtype"]],
                        "encoding": [bev_meta_aug["encoding"]],
                        "channel_spec": [bev_meta_aug["channel_spec"]],
                        "meters_per_px": [bev_meta_aug["meters_per_px"]],
                        "x_fwd_m": [bev_meta_aug["x_fwd_m"]],
                        "y_left_m": [bev_meta_aug["y_left_m"]],
                        "ego_x_w": [bev_meta_aug["ego_x_w"]],
                        "ego_y_w": [bev_meta_aug["ego_y_w"]],
                        "ego_yaw_w": [bev_meta_aug["ego_yaw_w"]],
                        "spec_version": [bev_meta_aug.get("spec_version", 1)],
                        "norms_version": [bev_meta_aug.get("norms_version", 1)],
                        "frame_ref": [bev_meta_aug["frame_ref"]],
                        "data": [rec["bev_bytes"]],
                    },
                )
                bev_w.append_table(bev_tbl)
                
                # Count written frames and show progress
                frames_written += 1
                if frames_written % 100 == 0:
                    LOG.info("    Progress: %d frames written...", frames_written)
            
            # For HDF5: write episode batch
            if backend == "hdf5" and episode_frames:
                hdf5_writer.append_episode(episode_frames)
                frames_rows += len(episode_frames)
            
            # Per-episode basic distribution metrics
            avg_speed = ep_stats["speed_sum"] / max(1, ep_stats["speed_count"])
            cmd_hist_str = ",".join(f"{k}:{v}" for k, v in sorted(ep_stats["cmd_hist"].items()))

            stationary_kept = ep_stats.get("kept_stationary", 0)
            stationary_total = ep_stats["skipped_stationary"] + stationary_kept
            stationary_ratio = stationary_kept / max(1, stationary_total)
            
            LOG.info(
                "=== Episode %d/%d: Complete (written=%d, skipped=%d | future_missing=%d stationary_skip=%d stationary_kept=%d behind=%d jump=%d action_mismatch=%d cmd_disagree=%d) ===",
                ep + 1,
                episodes,
                frames_written,
                frames_skipped,
                ep_stats["skipped_future_missing"],
                ep_stats["skipped_stationary"],
                stationary_kept,
                ep_stats["skipped_behind"],
                ep_stats["skipped_jump"],
                ep_stats["skipped_action_mismatch"],
                ep_stats["command_disagree"],
            )
            LOG.info(
                "    Avg speed=%.2f m/s, stationary_ratio=%.1f%%, command histogram=%s, speed_limit_fallback_frames=%d",
                avg_speed,
                stationary_ratio * 100,
                cmd_hist_str,
                ep_stats["speed_limit_fallback"],
            )

        # (Writers closed in finally)

    finally:
        try:
            world.apply_settings(original_settings)
            tm.set_synchronous_mode(original_sync)
        except Exception:
            pass
        
        # If DuckDB backend, export to Parquet BEFORE close
        if backend == "duckdb":
            try:
                LOG.info("Exporting DuckDB tables to Parquet...")
                frames_w.export_to_parquet_dir(run_dir, compression=pq_compression)
                route_w.export_to_parquet_dir(run_dir, compression=pq_compression)
                actors_w.export_to_parquet_dir(run_dir, compression=pq_compression)
                tls_w.export_to_parquet_dir(run_dir, compression=pq_compression)
                maps_w.export_to_parquet_dir(run_dir, compression=pq_compression)
                fut_w.export_to_parquet_dir(run_dir, compression=pq_compression)
                obj_w.export_to_parquet_dir(run_dir, compression=pq_compression)
                bev_w.export_to_parquet_dir(run_dir, compression=pq_compression)
                LOG.info("DuckDB export complete!")
            except Exception as e:
                LOG.error("ERROR during DuckDB export: %s", e)
                import traceback
                traceback.print_exc()
        
        # Finalize writers AFTER export (for DuckDB) or immediately (for other backends)
        try:
            if backend == "hdf5":
                hdf5_writer.close()
            else:
                frames_w.close(); route_w.close(); actors_w.close(); tls_w.close(); maps_w.close(); fut_w.close(); obj_w.close(); bev_w.close()
        except Exception as e:
            LOG.warning("Error closing writers: %s", e)
        # Cleanup NPCs and ego
        try:
            destroy_ids = []
            if npc_vehicles:
                destroy_ids.extend([a.id for a in npc_vehicles if a is not None])
            if ego is not None:
                destroy_ids.append(ego.id)
            if destroy_ids:
                client.apply_batch([carla.command.DestroyActor(x) for x in destroy_ids])
        except Exception:
            pass

    # Simple summary
    duration_s = time.time() - start_wall
    LOG.info("=== Collection Complete ===")
    LOG.info("Episodes: %d, reached goal: %d", episodes, episodes_reached)
    LOG.info("Frames written: %d", frames_rows)
    LOG.info("Wall time: %.1fs, ~%.1f fps", duration_s, frames_rows/duration_s if duration_s > 0 else 0)
    LOG.info("Output directory: %s", run_dir)


if __name__ == "__main__":
    main()


