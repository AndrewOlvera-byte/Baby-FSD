"""
CARLA Evaluator for real-time model testing.

Connects to CARLA simulator, runs episodes with model policy,
collects metrics, and saves video recordings.
"""

from __future__ import annotations
import os
import time
import math
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

import numpy as np
import torch
import logging

# Optional deps (CARLA only needed when running eval)
try:
    import carla
    CARLA_AVAILABLE = True
    CARLA_IMPORT_ERR = None
except ImportError as e:
    CARLA_AVAILABLE = False
    CARLA_IMPORT_ERR = e

try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False

from src.core.registry import register
from types import SimpleNamespace

from data.norms import (
    normalize_bev,
    normalize_route_points,
    normalize_object_tokens,
    normalize_futures,
    normalize_ego_vector,
    denormalize_futures,
)

LOG = logging.getLogger("carla_eval")

if CARLA_AVAILABLE:
    from carla_utils.actors import collect_nearby_actors, get_ego_state
    from carla_utils.map_vectors import extract_lane_centerline_segments, extract_traffic_light_stoplines
    from carla_utils.route import map_cmd_and_route
    from carla_utils.bev import rasterize_bev


@register("evaluator", "carla_eval")
class CarlaEvaluatorFactory:
    """Factory for CARLA evaluator."""
    
    def build(self, cfg_node, context):
        """
        Build CARLA evaluator from config.
        
        Args:
            cfg_node: Evaluator config node
            context: Dict with model and other context
            
        Returns:
            CarlaEvaluator instance
        """
        if not CARLA_AVAILABLE:
            raise ImportError(
                "carla_eval requires the CARLA Python API (carla==0.9.15) installed in the current environment."
            ) from CARLA_IMPORT_ERR
        return CarlaEvaluator(
            host=str(getattr(cfg_node, "host", "127.0.0.1")),
            port=int(getattr(cfg_node, "port", 2000)),
            tm_port=int(getattr(cfg_node, "tm_port", 8000)),
            n_episodes=int(getattr(cfg_node, "n_episodes", 5)),
            episode_max_steps=int(getattr(cfg_node, "episode_max_steps", 1000)),
            save_video=bool(getattr(cfg_node, "save_video", True)),
            video_fps=int(getattr(cfg_node, "video_fps", 20)),
            town=str(getattr(cfg_node, "town", "Town10HD_Opt")),
            weather=int(getattr(cfg_node, "weather", 0)),
            npc_count=int(getattr(cfg_node, "npc_count", 30)),
            reports_dir=Path(getattr(cfg_node, "reports_dir", "reports")),
            checkpoint_path=str(getattr(cfg_node, "checkpoint_path", "")),
            camera_width=int(getattr(cfg_node, "camera_width", 1280)),
            camera_height=int(getattr(cfg_node, "camera_height", 720)),
            camera_fov=float(getattr(cfg_node, "camera_fov", 90.0)),
            spectator_height=float(getattr(cfg_node, "spectator_height", 12.0)),
            spectator_follow=bool(getattr(cfg_node, "spectator_follow", True)),
            spectator_back=float(getattr(cfg_node, "spectator_back", 8.0)),
            controller_cfg=getattr(cfg_node, "controller", None),
            max_route_distance=float(getattr(cfg_node, "max_route_distance", 300.0)),
            bev_cfg=getattr(cfg_node, "bev", None),
            actor_radius=float(getattr(cfg_node, "actor_radius", 40.0)),
            window_m=float(getattr(cfg_node, "window_m", 60.0)),
        )


class CarlaEvaluator:
    """
    CARLA real-time evaluator.
    
    Features:
    - Spawn ego vehicle with model policy
    - Run N episodes with traffic
    - Record RGB camera frames
    - Compute metrics: route completion, collisions, infractions
    - Save video (MP4) to reports/
    - Reusable for BC, RL, any policy
    
    TODO: Full implementation requires:
    - CARLA Python API connection
    - Vehicle spawning and control
    - Sensor setup (camera, collision, lane invasion)
    - Policy inference loop
    - Video encoding (opencv or imageio)
    - Metrics computation
    """
    
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 2000,
        tm_port: int = 8000,
        n_episodes: int = 5,
        episode_max_steps: int = 1000,
        save_video: bool = True,
        video_fps: int = 20,
        town: str = "Town10HD_Opt",
        weather: int = 0,
        npc_count: int = 30,
        reports_dir: Path = None,
        checkpoint_path: str = "",
        camera_width: int = 1280,
        camera_height: int = 720,
        camera_fov: float = 90.0,
        spectator_height: float = 12.0,
        spectator_follow: bool = True,
        spectator_back: float = 8.0,
        controller_cfg: Any = None,
        max_route_distance: float = 300.0,
        bev_cfg: Optional[Dict[str, Any]] = None,
        actor_radius: float = 40.0,
        window_m: float = 60.0,
    ):
        """
        Initialize CARLA evaluator.
        
        Args:
            host: CARLA server host
            port: CARLA server port
            tm_port: Traffic manager port
            n_episodes: Number of episodes to run
            episode_max_steps: Max steps per episode
            save_video: Whether to save video
            video_fps: Video frame rate
            town: CARLA town/map name
            weather: Weather preset ID
            npc_count: Number of NPC vehicles
            reports_dir: Directory to save outputs
        """
        self.host = host
        self.port = port
        self.tm_port = tm_port
        self.n_episodes = n_episodes
        self.episode_max_steps = episode_max_steps
        self.save_video = save_video
        self.video_fps = video_fps
        self.town = town
        self.weather = weather
        self.npc_count = npc_count
        self.reports_dir = reports_dir or Path("reports")
        self.checkpoint_path = checkpoint_path
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.camera_fov = camera_fov
        self.spectator_height = spectator_height
        self.spectator_follow = spectator_follow
        self.spectator_back = spectator_back
        self.controller_cfg = controller_cfg or {}
        self.max_route_distance = max_route_distance
        self.bev_cfg = bev_cfg or {
            "meters_x": 80.0,
            "meters_y": 60.0,
            "resolution_m": 0.4,
            "lane_width_m": 3.5,
            "route_corridor_width_m": 6.0,
            "centerline_thickness_m": 0.6,
            "drivable_thickness_m": 6.0,
            "actor_radius_m": 1.2,
        }
        self.actor_radius = actor_radius
        self.window_m = window_m
        
        print(f"[CarlaEvaluator] Initialized")
        print(f"  host={host}:{port}, town={town}, episodes={n_episodes}")
        print(f"  save_video={save_video}, npc_count={npc_count}, checkpoint='{checkpoint_path}'")
    
    def __call__(self, model: torch.nn.Module) -> Dict[str, Any]:
        """
        Run evaluation in CARLA simulator.
        
        Args:
            model: Policy model to evaluate
            
        Returns:
            Dict with metrics:
                - route_completion_rate: float [0, 1]
                - collision_rate: float
                - infraction_count: int
                - avg_speed: float (m/s)
                - success_rate: float [0, 1]
                - video_path: str (if save_video=True)
        """
        if not CARLA_AVAILABLE:
            raise ImportError("carla package not available. Install CARLA Python API and ensure it's on PYTHONPATH.")

        # Optional: load checkpoint if provided
        if self.checkpoint_path:
            ckpt_path = Path(self.checkpoint_path).expanduser()
            if not ckpt_path.is_absolute():
                ckpt_path = Path.cwd() / ckpt_path
            if ckpt_path.exists():
                state = torch.load(ckpt_path, map_location="cpu")
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                model.load_state_dict(state, strict=False)
                print(f"[CarlaEvaluator] Loaded checkpoint from {ckpt_path}")
            else:
                print(f"[CarlaEvaluator] WARNING: checkpoint not found at {ckpt_path}, using provided model weights")

        device = next(model.parameters()).device
        model.eval()

        # Connect to CARLA
        client = carla.Client(self.host, self.port)
        client.set_timeout(10.0)
        world = client.load_world(self.town)
        tm = client.get_trafficmanager(self.tm_port)
        tm.set_synchronous_mode(True)

        # Set weather
        weather_presets = {
            0: carla.WeatherParameters.ClearNoon,
            1: carla.WeatherParameters.CloudyNoon,
            2: carla.WeatherParameters.WetNoon,
            3: carla.WeatherParameters.SoftRainNoon,
            4: carla.WeatherParameters.MidRainyNoon,
        }
        world.set_weather(weather_presets.get(self.weather, carla.WeatherParameters.ClearNoon))

        # Enable synchronous mode
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        # Spawn NPCs
        traffic_vehicles = []
        if self.npc_count > 0:
            blueprint_library = world.get_blueprint_library()
            vehicle_blueprints = blueprint_library.filter("vehicle.*")
            spawn_points = world.get_map().get_spawn_points()
            np.random.shuffle(spawn_points)
            for sp in spawn_points[: self.npc_count]:
                bp = np.random.choice(vehicle_blueprints)
                v = world.try_spawn_actor(bp, sp)
                if v:
                    traffic_vehicles.append(v)
            tm.global_percentage_speed_difference(10.0)

        # Prepare reports dir
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        metrics = {
            "eval/route_completion_rate": 0.0,
            "eval/collision_rate": 0.0,
            "eval/infraction_count": 0,
            "eval/avg_speed": 0.0,
            "eval/success_rate": 0.0,
            "eval/episodes_run": 0,
        }

        # Episode loop
        for ep in range(self.n_episodes):
            ep_metrics = self._run_episode(world, tm, model, device, ep_idx=ep)
            # Aggregate simple averages
            for k, v in ep_metrics.items():
                if isinstance(v, (int, float)):
                    metrics[k] = metrics.get(k, 0.0) + float(v)
                else:
                    metrics[k] = v
            metrics["eval/episodes_run"] += 1

        if metrics["eval/episodes_run"] > 0:
            n = float(metrics["eval/episodes_run"])
            for k in ["eval/route_completion_rate", "eval/collision_rate", "eval/infraction_count", "eval/avg_speed", "eval/success_rate"]:
                if k in metrics:
                    metrics[k] /= n

        # Cleanup NPCs
        for v in traffic_vehicles:
            v.destroy()

        # Restore async
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)

        return metrics

    def _run_episode(self, world, tm, model, device, ep_idx: int) -> Dict[str, float]:
        """Run a single CARLA episode."""
        blueprint_library = world.get_blueprint_library()
        vehicle_blueprints = blueprint_library.filter("vehicle.tesla.model3")
        ego_bp = vehicle_blueprints[0] if vehicle_blueprints else blueprint_library.filter("vehicle.*")[0]
        spawn_points = world.get_map().get_spawn_points()
        if len(spawn_points) < 2:
            raise RuntimeError("Not enough spawn points")
        start, goal = spawn_points[0], spawn_points[-1]
        ego = world.try_spawn_actor(ego_bp, start)
        if ego is None:
            raise RuntimeError("Failed to spawn ego")

        # Attach RGB camera (chase)
        cam_bp = blueprint_library.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(self.camera_width))
        cam_bp.set_attribute("image_size_y", str(self.camera_height))
        cam_bp.set_attribute("fov", str(self.camera_fov))
        cam_transform = carla.Transform(carla.Location(x=-6.0, z=self.spectator_height), carla.Rotation(pitch=-10))
        camera = world.spawn_actor(cam_bp, cam_transform, attach_to=ego)
        spectator = world.get_spectator() if self.spectator_follow else None

        # Collision sensor
        col_bp = blueprint_library.find("sensor.other.collision")
        collision_sensor = world.spawn_actor(col_bp, carla.Transform(), attach_to=ego)
        collision_history = []
        collision_sensor.listen(lambda event: collision_history.append(event))

        # Camera frame buffer
        frames: List[np.ndarray] = []
        camera.listen(lambda image: frames.append(self._carla_image_to_numpy(image)))

        # Simple PID controller state
        steer_err_prev = 0.0
        speed_err_prev = 0.0
        steer_cmd_prev = 0.0
        target_speed_prev = float("nan")

        # Main loop
        steps = 0
        avg_speed_accum = 0.0
        success = 0.0
        route_complete = 0.0

        try:
            while steps < self.episode_max_steps:
                world.tick()
                steps += 1

                # Build observations from live world state
                obs = self._build_obs(world, ego, device)

                with torch.no_grad():
                    pred = model(obs)

                # Use first future waypoint for steering target
                # Denormalize full trajectory
                fut_xy, fut_v = denormalize_futures(pred["future_xy"], pred["future_v"])
                waypoints_traj = fut_xy[0].detach().cpu().numpy()  # (N, 2)
                speeds_traj = fut_v[0].detach().cpu().numpy()      # (N,)

                # Speed (current) for logging and control
                vel = ego.get_velocity()
                speed = np.linalg.norm([vel.x, vel.y, vel.z])

                control, steer_err_prev, speed_err_prev, steer_cmd_prev, target_speed_prev, tl_info = self._simple_control(
                    ego,
                    waypoints_traj,
                    speeds_traj,
                    steer_err_prev,
                    speed_err_prev,
                    steer_cmd_prev,
                    target_speed_prev,
                )
                ego.apply_control(control)

                if spectator is not None:
                    ego_tf = ego.get_transform()
                    yaw_deg = ego_tf.rotation.yaw
                    yaw = math.radians(yaw_deg)
                    dx = -self.spectator_back * math.cos(yaw)
                    dy = -self.spectator_back * math.sin(yaw)
                    loc = ego_tf.location + carla.Location(x=dx, y=dy, z=self.spectator_height)
                    rot = carla.Rotation(pitch=-10.0, yaw=yaw_deg)
                    spectator.set_transform(carla.Transform(loc, rot))

                # Per-step logging for visibility
                LOG.info(
                    "[Step %04d] speed=%.2f target=%.2f steer=%.3f throttle=%.3f brake=%.3f tl=%s stop=%s",
                    steps,
                    speed,
                    target_speed_prev if not math.isnan(target_speed_prev) else float("nan"),
                    control.steer,
                    control.throttle,
                    control.brake,
                    tl_info.get("state"),
                    tl_info.get("must_stop"),
                )

                # Metrics
                avg_speed_accum += speed

                if self._reached_goal(ego, goal):
                    success = 1.0
                    route_complete = 1.0
                    break

                if collision_history:
                    break

        finally:
            camera.stop()
            collision_sensor.stop()
            camera.destroy()
            collision_sensor.destroy()
            ego.destroy()

        # Save video
        video_path = None
        if self.save_video and frames:
            if not IMAGEIO_AVAILABLE:
                print("[CarlaEvaluator] WARNING: imageio not installed, skipping video writing")
            else:
                video_path = self.reports_dir / f"carla_eval_ep{ep_idx:03d}.mp4"
                imageio.mimwrite(video_path, frames, fps=self.video_fps, quality=7)

        return {
            "eval/route_completion_rate": route_complete,
            "eval/collision_rate": 1.0 if collision_history else 0.0,
            "eval/infraction_count": float(len(collision_history)),
            "eval/avg_speed": avg_speed_accum / max(1, steps),
            "eval/success_rate": success,
            "eval/video_path": str(video_path) if video_path else "",
        }

    def _carla_image_to_numpy(self, image) -> np.ndarray:
        """Convert CARLA raw image to numpy RGB array."""
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        array = array[:, :, :3][:, :, ::-1]  # BGRA->RGB
        return array

    def _simple_control(
        self,
        ego,
        waypoints_xy: np.ndarray,
        speeds: np.ndarray,
        steer_prev: float,
        speed_err_prev: float,
        steer_cmd_prev: float,
        target_speed_prev: float,
    ) -> Tuple[carla.VehicleControl, float, float, float, float, Dict[str, Any]]:
        """
        Steering/throttle/brake control using a short horizon of waypoints and speeds.
        """
        kp_steer = float(self.controller_cfg.get("steer_kp", 1.5))
        kd_steer = float(self.controller_cfg.get("steer_kd", 0.05))
        kp_th = float(self.controller_cfg.get("throttle_kp", 0.5))
        kd_th = float(self.controller_cfg.get("throttle_kd", 0.1))
        kp_brake = float(self.controller_cfg.get("brake_kp", 0.3))
        lookahead_idx = int(self.controller_cfg.get("lookahead_idx", 2))
        steer_smoothing = float(self.controller_cfg.get("steer_smoothing", 0.5))
        speed_smoothing = float(self.controller_cfg.get("speed_smoothing", 0.7))
        min_target_speed = float(self.controller_cfg.get("min_target_speed_mps", 0.0))

        # Current speed
        vel = ego.get_velocity()
        speed = np.linalg.norm([vel.x, vel.y, vel.z])

        # Steering: weighted average of first few waypoints for smoother anticipation
        steer_err = 0.0
        if isinstance(waypoints_xy, np.ndarray) and waypoints_xy.ndim == 2 and len(waypoints_xy) > 0:
            weights = np.array([0.5, 0.3, 0.2], dtype=np.float32)
            weights = weights[: min(len(weights), len(waypoints_xy))]
            weights = weights / weights.sum()
            weighted_wp = np.sum(waypoints_xy[: len(weights)] * weights[:, None], axis=0)
            # Fallback to specific lookahead index if provided
            if 0 <= lookahead_idx < len(waypoints_xy):
                wp = waypoints_xy[lookahead_idx]
                blended = 0.5 * weighted_wp + 0.5 * wp
                angle = np.arctan2(blended[1], max(1e-3, blended[0]))
            else:
                angle = np.arctan2(weighted_wp[1], max(1e-3, weighted_wp[0]))
            steer_err = angle
        else:
            # Single waypoint fallback
            steer_err = np.arctan2(waypoints_xy[1], max(1e-3, waypoints_xy[0]))

        steer_der = steer_err - steer_prev
        steer = kp_steer * steer_err + kd_steer * steer_der
        steer = float(np.clip(steer, -1.0, 1.0))
        if steer_cmd_prev is None:
            steer_cmd_prev = 0.0
        steer = steer_smoothing * steer_cmd_prev + (1.0 - steer_smoothing) * steer

        # Target speed: prefer model trajectory unless controller override provided
        target_speed_val = self.controller_cfg.get("target_speed_mps", np.nan)
        if target_speed_val is None:
            target_speed_val = np.nan
        try:
            target_speed = float(target_speed_val)
        except Exception:
            target_speed = np.nan
        if np.isnan(target_speed) and isinstance(speeds, np.ndarray) and speeds.size > 0:
            target_speed = float(speeds[0])
        if np.isnan(target_speed):
            target_speed = 6.0  # sane fallback
        if target_speed_prev is None or math.isnan(target_speed_prev):
            target_speed_prev = target_speed
        target_speed = speed_smoothing * target_speed_prev + (1.0 - speed_smoothing) * target_speed
        if min_target_speed > 0.0:
            target_speed = max(target_speed, min_target_speed)

        # Traffic light compliance: stop on red/yellow near stop line
        tl_info: Dict[str, Any] = {"state": None, "must_stop": False}
        try:
            tl = ego.get_traffic_light()
            if tl is not None:
                tl_state = tl.get_state()
                tl_info["state"] = str(tl_state)
                if tl_state in (carla.TrafficLightState.Red, carla.TrafficLightState.Yellow):
                    stop_wps = tl.get_stop_waypoints()
                    stop_loc = stop_wps[0].transform.location if stop_wps else tl.get_transform().location
                    ego_loc = ego.get_location()
                    dist = np.linalg.norm([ego_loc.x - stop_loc.x, ego_loc.y - stop_loc.y, ego_loc.z - stop_loc.z])
                    if dist < 12.0:
                        target_speed = 0.0
                        tl_info["must_stop"] = True
        except Exception:
            pass

        # Speed control
        speed_err = target_speed - speed
        speed_der = speed_err - speed_err_prev
        if speed_err > 0.1:
            throttle = kp_th * speed_err + kd_th * speed_der
            throttle = float(np.clip(throttle, 0.0, 0.75))
            brake = 0.0
        else:
            throttle = 0.0
            brake = kp_brake * abs(speed_err) + kd_th * abs(speed_der)
            brake = float(np.clip(brake, 0.0, 0.5))

        control = carla.VehicleControl(
            throttle=throttle,
            steer=steer,
            brake=brake,
            hand_brake=False,
            reverse=False,
        )
        return control, steer_err, speed_err, steer, target_speed, tl_info

    def _reached_goal(self, ego, goal_transform) -> bool:
        """Check if ego is close to goal."""
        loc = ego.get_location()
        goal_loc = goal_transform.location
        dist = np.linalg.norm([loc.x - goal_loc.x, loc.y - goal_loc.y, loc.z - goal_loc.z])
        return dist < 5.0

    def _build_obs(self, world, ego, device) -> Dict[str, torch.Tensor]:
        """Build normalized observations from CARLA world state."""
        ego_tf = ego.get_transform()

        # Command and route
        cmd_int, route_xy = map_cmd_and_route(world, ego_tf, k_points=32, spacing_m=2.0, lookahead_dist_m=60.0)
        if len(route_xy) < 32:
            # pad with zeros
            route_xy = route_xy + [(0.0, 0.0)] * (32 - len(route_xy))
        route_np = np.array(route_xy[:32], dtype=np.float32)

        # Map vectors and actors
        map_polys = extract_lane_centerline_segments(world.get_map(), ego_tf, window_m=self.window_m, step_m=2.0)
        tls = extract_traffic_light_stoplines(world, ego_tf, window_m=self.window_m)
        actors = collect_nearby_actors(world, ego, radius_m=self.actor_radius)

        bev_np, _ = rasterize_bev(route_xy, map_polys, tls, actors, self.bev_cfg)

        # Objects (truncate/pad to 64, d_obj=11)
        M = 64
        objects_np = np.zeros((M, 11), dtype=np.float32)
        object_mask_np = np.zeros((M,), dtype=np.float32)
        for i, a in enumerate(actors[:M]):
            objects_np[i, 0] = float(a.get("type_id", 0))
            objects_np[i, 1] = float(a.get("x_ego", 0.0))
            objects_np[i, 2] = float(a.get("y_ego", 0.0))
            objects_np[i, 3] = float(a.get("sin_yaw", 0.0))
            objects_np[i, 4] = float(a.get("cos_yaw", 0.0))
            objects_np[i, 5] = float(a.get("length", 0.0))
            objects_np[i, 6] = float(a.get("width", 0.0))
            objects_np[i, 7] = float(a.get("vx", 0.0))
            objects_np[i, 8] = float(a.get("vy", 0.0))
            objects_np[i, 9] = float(a.get("oncoming_flag", 0.0))
            objects_np[i, 10] = float(a.get("priority_flag", 0.0))
            object_mask_np[i] = 1.0

        # Ego vector fields
        speed_mps, yaw_rate = get_ego_state(ego)
        acc_world = ego.get_acceleration()
        yaw_rad = math.radians(ego_tf.rotation.yaw)
        c = math.cos(-yaw_rad)
        s = math.sin(-yaw_rad)
        ax_ego = acc_world.x * c - acc_world.y * s
        ay_ego = acc_world.x * s + acc_world.y * c
        curvature = float(yaw_rate / max(speed_mps, 0.5))
        wheelbase_m = 2.875
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

        ego_row = SimpleNamespace(
            speed_mps=speed_mps,
            yaw_rate=yaw_rate,
            accel_long=ax_ego,
            accel_lat=ay_ego,
            curvature=curvature,
            steer_angle_rad=steer_angle_rad,
            steer_norm=0.0,
            throttle=0.0,
            brake=0.0,
            speed_limit_mps=speed_limit_mps,
            command=cmd_int,
            gear=ego.get_control().gear if hasattr(ego, "get_control") else 0,
            time_of_day_sin=tod_sin,
            time_of_day_cos=tod_cos,
        )

        # Normalize
        ego_vec = normalize_ego_vector(ego_row).unsqueeze(0).to(device)
        bev_t = torch.from_numpy(bev_np).to(device=device, dtype=torch.float32)
        bev_t = normalize_bev(bev_t)
        bev_t = bev_t.unsqueeze(0)
        route_t = torch.from_numpy(route_np).unsqueeze(0).to(device)
        route_t = normalize_route_points(route_t)
        objects_t = torch.from_numpy(objects_np).to(device)
        objects_t = normalize_object_tokens(objects_t).unsqueeze(0)
        object_mask_t = torch.from_numpy(object_mask_np).unsqueeze(0).to(device)

        future_xy = torch.zeros(1, 12, 2, device=device)
        future_v = torch.zeros(1, 12, device=device)
        future_mask = torch.ones(1, 12, device=device)
        future_xy, future_v = normalize_futures(future_xy, future_v)

        return {
            "ego_vec": ego_vec,
            "bev": bev_t,
            "route": route_t,
            "objects": objects_t,
            "object_mask": object_mask_t,
            "future_xy": future_xy,
            "future_v": future_v,
            "future_mask": future_mask,
        }
        
        return metrics
    
    def _connect_carla(self):
        """Connect to CARLA server (TODO)."""
        raise NotImplementedError("CARLA connection not implemented")
    
    def _spawn_ego_vehicle(self, world, start_transform):
        """Spawn ego vehicle (TODO)."""
        raise NotImplementedError("Vehicle spawning not implemented")
    
    def _setup_sensors(self, ego_vehicle):
        """Setup sensors on ego vehicle (TODO)."""
        raise NotImplementedError("Sensor setup not implemented")
    
    def _generate_route(self, world, start, goal):
        """Generate route from start to goal (TODO)."""
        raise NotImplementedError("Route generation not implemented")
    
    def _get_observations(self, ego_vehicle, world, route):
        """Get observations for model (TODO)."""
        raise NotImplementedError("Observation collection not implemented")
    
    def _apply_control(self, ego_vehicle, predictions):
        """Apply model predictions to vehicle control (TODO)."""
        raise NotImplementedError("Control application not implemented")
    
    def _compute_metrics(self, episode_data):
        """Compute metrics from episode data (TODO)."""
        raise NotImplementedError("Metrics computation not implemented")
    
    def _save_video(self, frames, output_path):
        """Save frames as video (TODO)."""
        raise NotImplementedError("Video saving not implemented")

