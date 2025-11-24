import argparse
import io
import json
import logging
import math
import os
import random
import time
import zlib
from collections import deque
from datetime import datetime
from typing import Dict, List, Tuple

import yaml
import numpy as np

import carla

# Ensure repo root on sys.path so sibling packages resolve
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data.schema import (
    K_ROUTE_POINTS,
    N_FUTURE_STEPS,
    FIXED_DELTA_SECONDS,
    FUTURE_DELTA_SECONDS,
    ACTOR_RADIUS_METERS,
    WINDOW_METERS,
)
from data.hdf5_writer import HDF5EpisodeSetWriter
from data.norms import normalize_ego_vector
from carla_utils.route import map_cmd_and_route, distance_to_goal
from carla_utils.actors import collect_nearby_actors, get_ego_state
from carla_utils.map_vectors import extract_lane_centerline_segments, extract_traffic_light_stoplines
from carla_utils.futures import FuturesBuffer, future_waypoints_ego
from carla_utils.bev import rasterize_bev, encode_bev_to_bytes
from carla_utils.rewards import (
    RewardTracker,
    compute_reward,
    detect_offroad,
    detect_traffic_violation,
)


LOG = logging.getLogger("collect_offline_rl")

ROUTE_PRIME_MIN_TICKS = 15
ROUTE_PRIME_MIN_SPEED_MPS = 1.0


def configure_logging() -> None:
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def set_sync(world: carla.World, tm: carla.TrafficManager, enable: bool, fixed_delta: float, no_rendering: bool) -> None:
    s = world.get_settings()
    s.synchronous_mode = enable
    s.fixed_delta_seconds = fixed_delta if enable else None
    s.no_rendering_mode = no_rendering
    world.apply_settings(s)
    tm.set_synchronous_mode(enable)


def pick_far_spawn_pair(world: carla.World) -> Tuple[carla.Transform, carla.Transform]:
    sps = list(world.get_map().get_spawn_points())
    a = random.choice(sps)
    b = max(sps, key=lambda sp: (sp.location.x - a.location.x) ** 2 + (sp.location.y - a.location.y) ** 2)
    if a == b:
        b = random.choice([sp for sp in sps if sp != a])
    return a, b


def spawn_model3(world: carla.World, transform: carla.Transform, max_retries: int = 10) -> carla.Vehicle:
    bp = world.get_blueprint_library().find("vehicle.tesla.model3")
    bp.set_attribute("role_name", "hero")
    v = world.try_spawn_actor(bp, transform)
    if v:
        return v
    spawn_points = list(world.get_map().get_spawn_points())
    random.shuffle(spawn_points)
    for sp in spawn_points[:max_retries]:
        v = world.try_spawn_actor(bp, sp)
        if v:
            return v
        world.tick()
    raise RuntimeError("Failed to spawn ego vehicle after retries")


def make_run_dir(base_out: str) -> str:
    run_id = datetime.utcnow().strftime("run-%Y%m%d-%H%M%S")
    out = os.path.join(base_out, run_id)
    os.makedirs(out, exist_ok=True)
    return out


def write_meta(out_dir: str, meta: Dict) -> None:
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def build_hdf5_frame(rec: Dict, wp_future: List, vel_future: List, K: int, N: int, M: int) -> Dict:
    """Build a frame dict for HDF5 output (matches BC collector shape)."""
    from types import SimpleNamespace

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

    bev_bytes = rec["bev_bytes"]
    raw = zlib.decompress(bev_bytes)
    bev_arr = np.load(io.BytesIO(raw), allow_pickle=False).astype(np.float32)

    route_xy = rec["route_xy"]
    route_arr = np.zeros((K, 2), dtype=np.float32)
    for i, (x, y) in enumerate(route_xy[:K]):
        route_arr[i] = [x, y]

    objects = rec.get("actors", [])
    objects_arr = np.zeros((M, 11), dtype=np.float32)
    object_mask = np.zeros((M,), dtype=np.float32)
    for i, obj in enumerate(objects[:M]):
        objects_arr[i] = [
            obj["type_id"],
            obj["x_ego"],
            obj["y_ego"],
            obj["sin_yaw"],
            obj["cos_yaw"],
            obj["length"],
            obj["width"],
            obj["vx"],
            obj["vy"],
            obj.get("oncoming_flag", 0),
            obj.get("priority_flag", 0),
        ]
        object_mask[i] = 1.0

    future_xy = np.zeros((N, 2), dtype=np.float32)
    future_v = np.zeros((N,), dtype=np.float32)
    future_mask = np.zeros((N,), dtype=np.float32)
    for i, p in enumerate(wp_future[:N]):
        future_xy[i] = [p[0], p[1]]
        future_mask[i] = 1.0
    for i, v in enumerate(vel_future[:N]):
        future_v[i] = v

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


def inject_control_noise(ctrl: carla.VehicleControl, noise_cfg: Dict) -> Tuple[carla.VehicleControl, bool]:
    prob = float(noise_cfg.get("probability", 0.0))
    if prob <= 0.0 or random.random() > prob:
        return ctrl, False
    steer_std = float(noise_cfg.get("steer_std", 0.0))
    throttle_std = float(noise_cfg.get("throttle_std", 0.0))
    brake_std = float(noise_cfg.get("brake_std", 0.0))

    noisy = carla.VehicleControl(
        steer=float(ctrl.steer + random.gauss(0.0, steer_std)),
        throttle=float(ctrl.throttle + random.gauss(0.0, throttle_std)),
        brake=float(ctrl.brake + random.gauss(0.0, brake_std)),
        hand_brake=bool(getattr(ctrl, "hand_brake", False)),
        reverse=bool(getattr(ctrl, "reverse", False)),
        manual_gear_shift=bool(getattr(ctrl, "manual_gear_shift", False)),
        gear=int(getattr(ctrl, "gear", 0)),
    )
    noisy.steer = max(-1.0, min(1.0, noisy.steer))
    noisy.throttle = max(0.0, min(1.0, noisy.throttle))
    noisy.brake = max(0.0, min(1.0, noisy.brake))
    return noisy, True


def attach_sensors(world: carla.World, ego: carla.Vehicle):
    bp_lib = world.get_blueprint_library()
    col_bp = bp_lib.find("sensor.other.collision")
    lane_bp = bp_lib.find("sensor.other.lane_invasion")

    collision_history: List = []
    lane_history: List = []

    collision_sensor = world.spawn_actor(col_bp, carla.Transform(), attach_to=ego)
    lane_sensor = world.spawn_actor(lane_bp, carla.Transform(), attach_to=ego)

    collision_sensor.listen(lambda event: collision_history.append(event))
    lane_sensor.listen(lambda event: lane_history.append(event))
    return collision_sensor, lane_sensor, collision_history, lane_history


def main():
    parser = argparse.ArgumentParser(description="Offline RL data collector")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/collect_offline_rl.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    configure_logging()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    host = cfg.get("host", "127.0.0.1")
    port = int(cfg.get("port", 2000))
    tm_port = int(cfg.get("tm_port", 8000))
    out_root = cfg.get("out_dir", "data/RL_v1")
    fixed_dt = float(cfg.get("fixed_dt", FIXED_DELTA_SECONDS))
    future_dt = float(cfg.get("future_dt", FUTURE_DELTA_SECONDS))
    K = int(cfg.get("K", K_ROUTE_POINTS))
    N = int(cfg.get("N", N_FUTURE_STEPS))
    radius = float(cfg.get("actor_radius", ACTOR_RADIUS_METERS))
    window_m = float(cfg.get("window_m", WINDOW_METERS))
    no_rendering = bool(cfg.get("no_rendering", False))
    behavior = str(cfg.get("behavior", "normal"))
    wheelbase_m = float(cfg.get("wheelbase_m", 2.875))
    episodes = int(cfg.get("episodes", 10))
    episode_max_frames = int(cfg.get("episode_max_frames", 2000))
    npc_count = int(cfg.get("npc_count", 30))
    max_frames = int(cfg.get("max_frames", 999999))
    bev_cfg = cfg.get("bev", {}) or {}
    reward_cfg = cfg.get("rewards", {}) or {}
    noise_cfg = cfg.get("noise", {}) or {}
    term_cfg = cfg.get("termination", {}) or {}
    storage_cfg = cfg.get("storage", {}) or {}

    client = carla.Client(host, port)
    client.set_timeout(10.0)
    world = client.get_world()
    tm = client.get_trafficmanager(tm_port)
    tm.set_synchronous_mode(True)
    tm.set_hybrid_physics_mode(True)
    tm.set_global_distance_to_leading_vehicle(1.0)
    tm.global_percentage_speed_difference(10.0)

    original_settings = world.get_settings()
    original_sync = original_settings.synchronous_mode

    run_dir = make_run_dir(out_root)
    run_id = os.path.basename(run_dir)

    # HDF5 writer only (offline RL storage)
    episodes_per_set = int(storage_cfg.get("episodes_per_set", 5))
    compression = storage_cfg.get("compression", "lz4")
    chunk_size = int(storage_cfg.get("chunk_size", 100))
    hdf5_writer = HDF5EpisodeSetWriter(
        output_dir=run_dir,
        run_id=run_id,
        episodes_per_set=episodes_per_set,
        K=K,
        N_future=N,
        M=64,
        C=18,
        H=int(round(bev_cfg.get("meters_y", 60.0) / bev_cfg.get("resolution_m", 0.4))),
        W=int(round(bev_cfg.get("meters_x", 80.0) / bev_cfg.get("resolution_m", 0.4))),
        compression=compression,
        chunk_size=chunk_size,
        include_rewards=True,
    )
    LOG.info("Using HDF5 backend (offline RL) with %d episodes per set", episodes_per_set)

    futures = FuturesBuffer(capacity=max(64, int(math.ceil(N * future_dt / fixed_dt)) * 3))
    reward_tracker = RewardTracker()

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
        "bev": bev_cfg,
        "rewards": reward_cfg,
        "noise": noise_cfg,
        "termination": term_cfg,
    }
    write_meta(run_dir, meta)

    npc_vehicles: List[carla.Actor] = []
    ego = None
    goal_tf = None
    goal_start_dist_m = 0.0
    frame_id = 0
    episodes_reached = 0
    start_wall = time.time()

    try:
        set_sync(world, tm, True, fixed_dt, no_rendering)

        def spawn_npc_traffic(world: carla.World, tm_port: int, count: int):
            bps = world.get_blueprint_library().filter("vehicle.*")
            sps = world.get_map().get_spawn_points()
            random.shuffle(sps)
            spawned = []
            for sp in sps[: min(count, len(sps))]:
                vbp = random.choice(bps)
                v = world.try_spawn_actor(vbp, sp)
                if v:
                    v.set_autopilot(True, tm_port)
                    spawned.append(v)
            world.tick()
            return spawned

        for ep in range(episodes):
            if ego is not None:
                try:
                    ego.destroy()
                    for _ in range(3):
                        world.tick()
                except Exception:
                    pass

            start_tf, end_tf = pick_far_spawn_pair(world)
            ego = spawn_model3(world, start_tf)
            ego.set_autopilot(True, tm_port)

            # attach sensors
            col_sensor, lane_sensor, collision_history, lane_history = attach_sensors(world, ego)
            last_collision_count = 0
            last_lane_count = 0

            for _ in range(20):
                world.tick()

            if not npc_vehicles and npc_count > 0:
                npc_vehicles = spawn_npc_traffic(world, tm_port, npc_count)

            goal_tf = end_tf
            try:
                goal_start_dist_m = distance_to_goal(ego.get_transform(), goal_tf)
            except Exception:
                goal_start_dist_m = 0.0

            LOG.info("=== Episode %d/%d: Starting (goal distance: %.1fm) ===", ep + 1, episodes, goal_start_dist_m)

            futures = FuturesBuffer(capacity=max(64, int(math.ceil(N * future_dt / fixed_dt)) * 3))
            pending = deque()
            recording_ready = False
            prime_ticks = 0
            prev_state = None
            prev_goal_dist = goal_start_dist_m
            ep_frames = 0
            termination_reason = None

            while frame_id < max_frames and ep_frames < episode_max_frames:
                world.tick()
                sim_time = (frame_id + 1) * fixed_dt

                speed_mps, yaw_rate = get_ego_state(ego)
                ego_tf = ego.get_transform()

                cmd_int, route_xy, _ = map_cmd_and_route(world, ego_tf, K, spacing_m=2.0, return_plan_stats=True)

                goal_dist_m = 0.0
                if goal_tf is not None:
                    try:
                        goal_dist_m = distance_to_goal(ego_tf, goal_tf)
                    except Exception:
                        goal_dist_m = 0.0

                if not recording_ready:
                    if speed_mps >= ROUTE_PRIME_MIN_SPEED_MPS and route_xy:
                        prime_ticks += 1
                        if prime_ticks >= ROUTE_PRIME_MIN_TICKS:
                            recording_ready = True
                            LOG.info("  Recording started at frame %d (speed=%.1f m/s)", frame_id, speed_mps)
                    else:
                        prime_ticks = 0

                # Control with optional noise injection
                ctrl = ego.get_control()
                noisy_ctrl, noise_injected = inject_control_noise(ctrl, noise_cfg)
                if noise_injected:
                    ego.apply_control(noisy_ctrl)
                steer_norm = float(getattr(noisy_ctrl, "steer", 0.0))
                throttle = float(getattr(noisy_ctrl, "throttle", 0.0))
                brake = float(getattr(noisy_ctrl, "brake", 0.0))
                gear = int(getattr(noisy_ctrl, "gear", 0))

                # Kinematics
                acc_world = ego.get_acceleration()
                yaw_rad = math.radians(ego_tf.rotation.yaw)
                c = math.cos(-yaw_rad)
                s = math.sin(-yaw_rad)
                ax_ego = acc_world.x * c - acc_world.y * s
                ay_ego = acc_world.x * s + acc_world.y * c
                curvature = float(yaw_rate / max(speed_mps, 0.5))
                steer_angle_rad = float(math.atan(wheelbase_m * curvature))
                wp = world.get_map().get_waypoint(
                    ego_tf.location, project_to_road=True, lane_type=carla.LaneType.Driving
                )
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

                # Heavy features after control
                actors = collect_nearby_actors(world, ego, radius)
                map_polys = extract_lane_centerline_segments(world.get_map(), ego_tf, window_m=window_m, step_m=2.0)
                tls = extract_traffic_light_stoplines(world, ego_tf, window_m=window_m)
                bev_tensor, bev_meta = rasterize_bev(route_xy, map_polys, tls, actors, bev_cfg)
                bev_bytes = encode_bev_to_bytes(bev_tensor)

                # Event flags and reward
                collision_events = len(collision_history)
                lane_events = len(lane_history)
                collision_flag = collision_events > last_collision_count
                lane_flag = lane_events > last_lane_count
                last_collision_count = collision_events
                last_lane_count = lane_events
                offroad_flag = detect_offroad(ego_tf, world.get_map())
                violation_flag = detect_traffic_violation(
                    ego_tf, tls, lane_history if lane_flag else [], reward_cfg.get("max_stopline_dist", 8.0)
                )
                route_completed = goal_tf is not None and goal_dist_m <= float(term_cfg.get("completion_goal_dist", 5.0))

                progress_delta = 0.0
                if prev_goal_dist is not None:
                    progress_delta = max(0.0, float(prev_goal_dist - goal_dist_m))
                prev_goal_dist = goal_dist_m

                weights = {
                    "progress": reward_cfg.get("progress_weight", 0.1),
                    "collision": reward_cfg.get("collision_penalty", -5.0),
                    "offroad": reward_cfg.get("offroad_penalty", -2.0),
                    "violation": reward_cfg.get("violation_penalty", -1.0),
                    "comfort": reward_cfg.get("comfort_weight", -0.01),
                    "completion": reward_cfg.get("completion_bonus", 5.0),
                }
                tracker_for_step = reward_tracker if recording_ready else None
                reward_out = compute_reward(
                    ego_state={"accel_long": ax_ego, "accel_lat": ay_ego, "steer_norm": steer_norm},
                    prev_state=prev_state,
                    events={
                        "collision": collision_flag,
                        "offroad": offroad_flag,
                        "traffic_violation": violation_flag,
                        "route_completed": route_completed,
                    },
                    route_progress_delta=progress_delta,
                    dt=fixed_dt,
                    weights=weights,
                    clip_min=reward_cfg.get("clip_min", -10.0),
                    clip_max=reward_cfg.get("clip_max", 10.0),
                    tracker=tracker_for_step,
                )
                prev_state = {"accel_long": ax_ego, "accel_lat": ay_ego, "steer_norm": steer_norm}

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
                    "reward_raw": float(reward_out["raw"]),
                    "reward_clipped": float(reward_out["clipped"]),
                    "reward_normalized": float(reward_out["normalized"]),
                    "reward_progress": float(reward_out["components"]["progress"]),
                    "reward_collision": float(reward_out["components"]["collision"]),
                    "reward_offroad": float(reward_out["components"]["offroad"]),
                    "reward_violation": float(reward_out["components"]["violation"]),
                    "reward_comfort": float(reward_out["components"]["comfort"]),
                    "reward_completion": float(reward_out["components"]["completion"]),
                    "reward_mean": float(reward_out["stats"]["mean"] or 0.0),
                    "reward_std": float(reward_out["stats"]["std"] or 0.0),
                    "noise_injected": bool(noise_injected),
                }

                if recording_ready:
                    pending.append(frame_data)

                futures.add(sim_time, ego_tf, speed_mps)

                frame_id += 1
                ep_frames += 1

                should_stop = False
                if term_cfg.get("collision", True) and collision_flag:
                    termination_reason = "collision"
                    should_stop = True
                elif term_cfg.get("offroad", True) and offroad_flag:
                    termination_reason = "offroad"
                    should_stop = True
                elif term_cfg.get("completion", True) and route_completed:
                    termination_reason = "goal"
                    episodes_reached += 1
                    should_stop = True

                if should_stop:
                    if recording_ready:
                        LOG.info("  Terminating episode due to %s", termination_reason)
                    break

            # Flush pending frames
            LOG.info("  Writing %d frames to disk...", len(pending))
            frames_written = 0
            frames_skipped = 0
            episode_frames = []
            while pending:
                rec = pending.popleft()
                deltas = [future_dt * (i + 1) for i in range(N)]
                samples = futures.sample_future(rec["sim_time"], deltas)
                if not samples or len(samples) < N or any(s is None for s in samples):
                    frames_skipped += 1
                    continue

                base_tf = rec["ego_tf"]
                wp_future, vel_future = future_waypoints_ego(base_tf, samples)
                frame_dict = build_hdf5_frame(rec, wp_future, vel_future, K, N, 64)
                frame_dict.update(
                    {
                        "reward_raw": rec["reward_raw"],
                        "reward_clipped": rec["reward_clipped"],
                        "reward_normalized": rec["reward_normalized"],
                        "reward_progress": rec["reward_progress"],
                        "reward_collision": rec["reward_collision"],
                        "reward_offroad": rec["reward_offroad"],
                        "reward_violation": rec["reward_violation"],
                        "reward_comfort": rec["reward_comfort"],
                        "reward_completion": rec["reward_completion"],
                        "reward_mean": rec["reward_mean"],
                        "reward_std": rec["reward_std"],
                        "noise_injected": rec["noise_injected"],
                    }
                )
                episode_frames.append(frame_dict)
                frames_written += 1

            if episode_frames:
                hdf5_writer.append_episode(episode_frames)

            LOG.info(
                "=== Episode %d/%d complete (%d frames written, %d skipped, termination=%s) ===",
                ep + 1,
                episodes,
                frames_written,
                frames_skipped,
                termination_reason or "none",
            )

            # Cleanup sensors per episode
            try:
                col_sensor.stop()
                lane_sensor.stop()
                col_sensor.destroy()
                lane_sensor.destroy()
            except Exception:
                pass

        # Summary
        duration_s = time.time() - start_wall
        LOG.info("=== Collection Complete ===")
        LOG.info("Episodes: %d, reached goal: %d", episodes, episodes_reached)
        LOG.info("Frames written: %d", frame_id)
        LOG.info("Wall time: %.1fs, ~%.1f fps", duration_s, frame_id / duration_s if duration_s > 0 else 0)
        LOG.info("Output directory: %s", run_dir)

    finally:
        try:
            world.apply_settings(original_settings)
            tm.set_synchronous_mode(original_sync)
        except Exception:
            pass
        try:
            hdf5_writer.close()
        except Exception:
            pass
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


if __name__ == "__main__":
    main()
