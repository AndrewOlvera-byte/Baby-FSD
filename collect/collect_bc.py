import argparse
import glob
import io
import json
import math
import os
import random
import time
from datetime import datetime
from typing import Dict, List

import yaml
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import zlib

import carla
from agents.navigation.behavior_agent import BehaviorAgent

from data.schema import Schemas, K_ROUTE_POINTS, N_FUTURE_STEPS, FIXED_DELTA_SECONDS, FUTURE_DELTA_SECONDS, ACTOR_RADIUS_METERS, WINDOW_METERS
from data.writer import ParquetShardWriter, ParquetDatasetWriter, DuckDBWriter, table_from_pydict
from carla_utils.route import extract_cmd_and_route, distance_to_goal
from carla_utils.actors import collect_nearby_actors, get_ego_state
from carla_utils.map_vectors import extract_lane_centerline_segments, extract_traffic_light_stoplines
from carla_utils.futures import FuturesBuffer, future_waypoints_ego
from carla_utils.bev import rasterize_bev, encode_bev_to_bytes


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


def spawn_model3(world: carla.World, transform: carla.Transform) -> carla.Vehicle:
    bp = world.get_blueprint_library().find("vehicle.tesla.model3")
    bp.set_attribute("role_name", "hero")
    v = world.try_spawn_actor(bp, transform)
    if not v:
        raise RuntimeError("Failed to spawn ego vehicle")
    return v


def make_run_dir(base_out: str) -> str:
    run_id = datetime.utcnow().strftime("run-%Y%m%d-%H%M%S")
    out = os.path.join(base_out, run_id)
    os.makedirs(out, exist_ok=True)
    return out


def write_meta(out_dir: str, meta: Dict) -> None:
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join("configs", "collect_bc.yaml"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    host = cfg.get("host", "127.0.0.1")
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
    backend = (storage_cfg.get("backend", "duckdb") or "duckdb").lower()

    # Writers per table
    if backend == "duckdb":
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
    agent = None
    goal_tf = None
    goal_start_dist_m = 0.0
    futures = FuturesBuffer(capacity=max(64, int(math.ceil(N * future_dt / fixed_dt)) * 3))

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
                except Exception:
                    pass
            start_tf, end_tf = pick_far_spawn_pair(world)
            ego = spawn_model3(world, start_tf)
            agent = BehaviorAgent(ego, behavior=behavior)
            # Configure agent for better rule following
            try:
                agent.ignore_traffic_lights(False)  # Respect traffic lights
                agent.ignore_stop_signs(False)       # Respect stop signs
                agent.ignore_vehicles(False)         # Avoid collisions
            except Exception:
                pass
            agent.set_destination(end_tf.location)
            # Warm-up: ensure planner has initial state before the control loop
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
            # Reset futures buffer per episode
            futures = FuturesBuffer(capacity=max(64, int(math.ceil(N * future_dt / fixed_dt)) * 3))

            ep_frames = 0
            # pending frame queue for lagged future labels
            # each item holds the extracted features for that frame time
            from collections import deque
            pending = deque()

            while frame_id < max_frames and ep_frames < episode_max_frames:
                world.tick()

                # Keep planner state fresh (matches working visual confirm loop)
                try:
                    agent.update_information()
                except Exception:
                    pass
                
                # If destination reached, stop and end episode
                if agent.done():
                    try:
                        ego.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0, hand_brake=False))
                    except Exception:
                        pass
                    episodes_reached += 1
                    break

                sim_time = (frame_id + 1) * fixed_dt
                
                # Extract current state FIRST (for this frame)
                speed_mps, yaw_rate = get_ego_state(ego)
                ego_tf = ego.get_transform()

                # Route / command (lightweight)
                cmd_int, route_xy = extract_cmd_and_route(agent, ego_tf, K, spacing_m=2.0)

                # Compute and apply control immediately after cheap queries
                ctrl = agent.run_step()
                steer_norm = float(getattr(ctrl, "steer", 0.0))
                throttle = float(getattr(ctrl, "throttle", 0.0))
                brake = float(getattr(ctrl, "brake", 0.0))
                gear = int(getattr(ctrl, "gear", 0))
                ego.apply_control(ctrl)

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

                # NO disk I/O in the control loop! Just collect in memory.
                pending.append(frame_data)
                
                # Buffer current state for future trajectory labels
                futures.add(sim_time, ego_tf, speed_mps)
                
                frame_id += 1
                ep_frames += 1

            # After episode loop: flush remaining pending frames
            # Process any frames that have sufficient future data
            print(f"Episode {ep+1} ended, writing {len(pending)} frames to disk...")
            horizon_s = N * future_dt
            frames_written = 0
            frames_skipped = 0
            while pending:
                rec = pending.popleft()
                # Check if we have enough future data
                deltas = [future_dt * (i + 1) for i in range(N)]
                samples = futures.sample_future(rec["sim_time"], deltas)
                
                # If we don't have full future trajectory, skip this frame
                # (last few frames of each episode won't have complete futures)
                if not samples or len(samples) < N:
                    frames_skipped += 1
                    continue
                    
                base_tf = rec["ego_tf"]
                wp_future, vel_future = future_waypoints_ego(base_tf, samples)

                # Frames row
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
                        "goal_dist_m": [distance_to_goal(rec["ego_tf"], goal_tf)],
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
                if frames_written % 50 == 0:
                    print(f"  Written {frames_written} frames...")
            
            print(f"Episode {ep+1} complete: {frames_written} frames written, {frames_skipped} skipped (no complete futures)")

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
                print("Exporting DuckDB tables to Parquet...")
                frames_w.export_to_parquet_dir(run_dir, compression=pq_compression)
                print("  ✓ frames exported")
                route_w.export_to_parquet_dir(run_dir, compression=pq_compression)
                print("  ✓ route_points exported")
                actors_w.export_to_parquet_dir(run_dir, compression=pq_compression)
                print("  ✓ actors exported")
                tls_w.export_to_parquet_dir(run_dir, compression=pq_compression)
                print("  ✓ traffic_lights exported")
                maps_w.export_to_parquet_dir(run_dir, compression=pq_compression)
                print("  ✓ map_segments exported")
                fut_w.export_to_parquet_dir(run_dir, compression=pq_compression)
                print("  ✓ futures exported")
                obj_w.export_to_parquet_dir(run_dir, compression=pq_compression)
                print("  ✓ object_tokens exported")
                bev_w.export_to_parquet_dir(run_dir, compression=pq_compression)
                print("  ✓ bev_frames exported")
                print("DuckDB export complete!")
            except Exception as e:
                print(f"ERROR during DuckDB export: {e}")
                import traceback
                traceback.print_exc()
        
        # Finalize writers AFTER export (for DuckDB) or immediately (for other backends)
        try:
            frames_w.close(); route_w.close(); actors_w.close(); tls_w.close(); maps_w.close(); fut_w.close(); obj_w.close(); bev_w.close()
        except Exception as e:
            print(f"Warning: Error closing writers: {e}")
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

    # Integrity + summary
    # Check for leftover tmp files (means incomplete shard rotation)
    leftover_tmp = [p for p in os.listdir(run_dir) if p.endswith('.tmp')]
    intact = len(leftover_tmp) == 0

    # Parquet-level validation: try reading footers to get row counts
    def _count_rows_in_file(path: str) -> int:
        try:
            meta = pq.read_metadata(path)
            num = meta.num_rows if meta is not None else 0
            if num:
                return int(num)
            tbl = pq.read_table(path)
            return int(tbl.num_rows)
        except Exception as exc:
            print(f"Warning: failed to read parquet metadata '{path}': {exc}")
            return 0

    def sum_rows(pattern: str) -> int:
        # Backwards-compat scan for flat files in run_dir
        if not os.path.isdir(run_dir):
            return 0
        total = 0
        try:
            entries = os.listdir(run_dir)
        except Exception as exc:
            print(f"Warning: unable to list '{run_dir}' during integrity check: {exc}")
            return 0
        paths = [os.path.join(run_dir, p) for p in entries if p.startswith(pattern) and p.endswith('.parquet')]
        for p in paths:
            total += _count_rows_in_file(p)
        return total

    # New: scan subdirectories used by ParquetDatasetWriter (and DuckDB export)
    def sum_rows_dir(subdir: str) -> int:
        d = os.path.join(run_dir, subdir)
        if not os.path.isdir(d):
            return 0
        total = 0
        pattern = os.path.join(d, "**", "*.parquet")
        for p in glob.glob(pattern, recursive=True):
            total += _count_rows_in_file(p)
        return total

    # Combine flat-file and directory-based totals
    frames_total = sum_rows('frames-') + sum_rows_dir('frames')
    route_total = sum_rows('route_points-') + sum_rows_dir('route_points')
    actors_total = sum_rows('actors-') + sum_rows_dir('actors')
    tls_total = sum_rows('traffic_lights-') + sum_rows_dir('traffic_lights')
    maps_total = sum_rows('map_segments-') + sum_rows_dir('map_segments')
    futures_total = sum_rows('futures-') + sum_rows_dir('futures')
    objects_total = sum_rows('object_tokens-') + sum_rows_dir('object_tokens')
    bev_total = sum_rows('bev_frames-') + sum_rows_dir('bev_frames')

    # DuckDB integrity counts
    duck_counts = {}
    try:
        if backend == "duckdb":
            duck_counts = {
                "frames": frames_w.row_count(),
                "route_points": route_w.row_count(),
                "actors": actors_w.row_count(),
                "traffic_lights": tls_w.row_count(),
                "map_segments": maps_w.row_count(),
                "futures": fut_w.row_count(),
                "object_tokens": obj_w.row_count(),
                "bev_frames": bev_w.row_count(),
            }
    except Exception:
        duck_counts = {}

    # Anomaly checks
    issues = []
    if frames_total <= 0:
        issues.append("no_frames")
    if futures_total % max(1, N) != 0:
        issues.append("futures_not_divisible_by_N")
    # expected lower bound for matured frames (approx)
    horizon_ticks = int(math.ceil((N * future_dt) / max(1e-6, fixed_dt)))
    matured_expected_lower = max(0, frames_rows - episodes * horizon_ticks)
    matured_est = futures_rows // max(1, N)
    if matured_est < max(0, int(0.7 * matured_expected_lower)) and frames_rows > 0:
        issues.append("too_few_matured_frames")
    # route coverage sanity
    if K > 0 and frames_rows > 0:
        route_cov = route_rows / float(K * frames_rows)
        if route_cov < 0.4:
            issues.append("low_route_coverage")
    # actor/tl/map presence if npc requested
    if npc_count > 0 and actors_rows / float(max(1, frames_rows)) < 1.0:
        issues.append("low_actors_density")
    # Per-step schema presence
    if bev_total != frames_total:
        issues.append("bev_count_mismatch")
    if objects_total < actors_total:
        issues.append("object_tokens_missing")

    # Decode a small BEV sample to validate tensor shape and values
    try:
        # Check both flat files and subdirectories
        bev_paths = []
        bev_paths.extend([os.path.join(run_dir, p) for p in os.listdir(run_dir) 
                          if p.startswith('bev_frames-') and p.endswith('.parquet')])
        bev_dir = os.path.join(run_dir, 'bev_frames')
        if os.path.isdir(bev_dir):
            bev_paths.extend([os.path.join(bev_dir, p) for p in os.listdir(bev_dir)
                              if p.endswith('.parquet')])
        bev_paths.sort()
        if bev_paths:
            pf = pq.ParquetFile(bev_paths[0])
            tbl = pf.read_row_group(0, columns=["C", "H", "W", "data"])
            if tbl.num_rows > 0:
                C = int(tbl.column(0)[0].as_py())
                H = int(tbl.column(1)[0].as_py())
                W = int(tbl.column(2)[0].as_py())
                blob = tbl.column(3)[0].as_buffer().to_pybytes()
                raw = zlib.decompress(blob)
                arr = np.load(io.BytesIO(raw), allow_pickle=False)
                if not (arr.shape == (C, H, W)):
                    issues.append("bev_shape_mismatch")
                if not np.isfinite(arr).all():
                    issues.append("bev_contains_nonfinite")
    except Exception:
        issues.append("bev_decode_failed")

    duration_s = time.time() - start_wall
    print(f"Integrity: {'OK' if intact else 'PENDING (tmp files remain)'}")
    print(f"Episodes: {episodes}, reached goal: {episodes_reached}")
    if duck_counts:
        print(f"Frames (mem/db/file): {frames_rows}/{duck_counts.get('frames',0)}/{frames_total}, Routes: {route_rows}/{duck_counts.get('route_points',0)}/{route_total}, Actors: {actors_rows}/{duck_counts.get('actors',0)}/{actors_total}, TLs: {tls_rows}/{duck_counts.get('traffic_lights',0)}/{tls_total}, MapSegs: {map_rows}/{duck_counts.get('map_segments',0)}/{maps_total}, Futures: {futures_rows}/{duck_counts.get('futures',0)}/{futures_total}, ObjTokens: {duck_counts.get('object_tokens',0)}, BEV: {duck_counts.get('bev_frames',0)}")
    else:
        print(f"Frames (mem/file): {frames_rows}/{frames_total}, Routes: {route_rows}/{route_total}, Actors: {actors_rows}/{actors_total}, TLs: {tls_rows}/{tls_total}, MapSegs: {map_rows}/{maps_total}, Futures: {futures_rows}/{futures_total}, ObjTokens: {objects_total}, BEV: {bev_total}")
    print(f"Wall time: {duration_s:.1f}s, ~{(frames_rows/duration_s if duration_s>0 else 0):.1f} fps")
    print(f"Checks: {'OK' if not issues else 'ISSUES -> ' + ','.join(issues)}")


if __name__ == "__main__":
    main()


