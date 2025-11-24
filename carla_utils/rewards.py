"""
Reward helpers for offline RL data collection.

Provides per-frame reward computation, simple event detectors, and running
statistics to normalize rewards during collection.
"""

import math
import logging
from typing import Dict, Iterable, Optional, Tuple

try:
    import carla
except ImportError:  # pragma: no cover - handled at runtime
    carla = None  # type: ignore


LOG = logging.getLogger(__name__)

# Default component weights
DEFAULT_WEIGHTS = {
    "progress": 0.1,
    "collision": -5.0,
    "offroad": -2.0,
    "violation": -1.0,
    "comfort": -0.01,
    "completion": 5.0,
}

CLIP_MIN, CLIP_MAX = -10.0, 10.0


class RewardTracker:
    """
    Track running mean/std of rewards using Welford's algorithm.
    """

    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self._m2 = 0.0

    def update(self, value: float) -> Tuple[float, float]:
        """Update tracker with a new reward and return (mean, std)."""
        self.count += 1
        delta = value - self.mean
        self.mean += delta / float(self.count)
        delta2 = value - self.mean
        self._m2 += delta * delta2
        std = math.sqrt(self._m2 / float(self.count)) if self.count > 0 else 0.0
        return self.mean, std

    @property
    def std(self) -> float:
        if self.count == 0:
            return 0.0
        return math.sqrt(self._m2 / float(self.count))


def compute_jerk(accel_prev, accel_curr, dt: float) -> float:
    """Compute jerk magnitude given previous/current acceleration."""
    if accel_prev is None or accel_curr is None or dt <= 0:
        return 0.0

    def _vec(a):
        if isinstance(a, (tuple, list)):
            if len(a) >= 2:
                return float(a[0]), float(a[1])
            return float(a[0]), 0.0
        return float(a), 0.0

    ax_prev, ay_prev = _vec(accel_prev)
    ax_cur, ay_cur = _vec(accel_curr)
    jx = (ax_cur - ax_prev) / dt
    jy = (ay_cur - ay_prev) / dt
    return math.sqrt(jx * jx + jy * jy)


def detect_collision(collision_history: Iterable) -> bool:
    """Return True if collision sensor has reported any collisions."""
    if collision_history is None:
        return False
    try:
        return len(collision_history) > 0
    except Exception:
        return any(True for _ in collision_history)


def detect_offroad(ego_tf, world_map) -> bool:
    """
    Heuristic off-road detection using map waypoints.

    Returns True when the ego is not on a driving lane.
    """
    if ego_tf is None or world_map is None or carla is None:
        return False
    try:
        wp = world_map.get_waypoint(
            ego_tf.location, project_to_road=False, lane_type=carla.LaneType.Any
        )
    except Exception:
        return False
    if wp is None:
        return True
    try:
        return wp.lane_type != carla.LaneType.Driving
    except Exception:
        return False


def detect_traffic_violation(
    ego_tf,
    traffic_lights,
    lane_invasion_history: Optional[Iterable] = None,
    max_stopline_dist: float = 8.0,
) -> bool:
    """
    Detect simple traffic violations (red light or lane invasion).

    Args:
        ego_tf: ego transform
        traffic_lights: list of traffic light dicts with stopline in ego frame
        lane_invasion_history: lane invasion sensor events
        max_stopline_dist: forward distance to consider a stopline
    """
    # Lane invasion already indicates a violation
    try:
        if lane_invasion_history and len(lane_invasion_history) > 0:
            return True
    except Exception:
        pass

    if not traffic_lights or ego_tf is None:
        return False
    for tl in traffic_lights:
        state = int(tl.get("state", 0))
        stop_x = float(tl.get("stop_x_ego", 0.0))
        stop_y = float(tl.get("stop_y_ego", 0.0))

        # Stopline is already in ego frame; consider lines ahead of the vehicle
        if stop_x < 0.0 or stop_x > max_stopline_dist:
            continue
        if abs(stop_y) > max_stopline_dist:
            continue
        # Treat Red or Yellow as needing to stop
        if state in (0, 1):  # Red=0, Yellow=1 in our BEV encoding
            return True
    return False


def compute_reward(
    ego_state: Dict,
    prev_state: Optional[Dict],
    events: Dict,
    route_progress_delta: float,
    dt: float,
    weights: Dict = None,
    clip_min: float = CLIP_MIN,
    clip_max: float = CLIP_MAX,
    tracker: Optional[RewardTracker] = None,
) -> Dict:
    """
    Compute per-frame reward and components.

    Args:
        ego_state: current ego state dict (expects accel_long, accel_lat, steer_norm)
        prev_state: previous ego state dict
        events: flags for collision/offroad/traffic_violation/route_completed
        route_progress_delta: forward progress along the route (meters)
        dt: timestep in seconds
        weights: optional override for component weights
        clip_min/clip_max: bounds for reward clipping
        tracker: optional RewardTracker to maintain running stats
    """
    w = DEFAULT_WEIGHTS.copy()
    if weights:
        w.update(weights)

    collision_flag = bool(events.get("collision", False))
    offroad_flag = bool(events.get("offroad", False))
    violation_flag = bool(events.get("traffic_violation", False))
    completion_flag = bool(events.get("route_completed", False))

    # Kinematics for comfort term
    accel_prev = None
    accel_curr = None
    steer_prev = 0.0
    steer_curr = float(ego_state.get("steer_norm", 0.0))
    if prev_state is not None:
        accel_prev = (
            float(prev_state.get("accel_long", 0.0)),
            float(prev_state.get("accel_lat", 0.0)),
        )
        steer_prev = float(prev_state.get("steer_norm", 0.0))
    accel_curr = (
        float(ego_state.get("accel_long", 0.0)),
        float(ego_state.get("accel_lat", 0.0)),
    )
    jerk = compute_jerk(accel_prev, accel_curr, dt)
    delta_steer = abs(steer_curr - steer_prev)

    comp_progress = w["progress"] * max(0.0, float(route_progress_delta))
    comp_collision = w["collision"] * float(collision_flag)
    comp_offroad = w["offroad"] * float(offroad_flag)
    comp_violation = w["violation"] * float(violation_flag)
    comp_comfort = w["comfort"] * float(abs(jerk) + abs(delta_steer))
    comp_completion = w["completion"] * float(completion_flag)

    raw = comp_progress + comp_collision + comp_offroad + comp_violation + comp_comfort + comp_completion
    clipped = max(clip_min, min(clip_max, raw))

    stats = {"mean": None, "std": None, "count": None}
    normalized = clipped
    if tracker is not None:
        mean, std = tracker.update(clipped)
        stats = {"mean": mean, "std": std, "count": tracker.count}
        if std > 1e-6:
            normalized = (clipped - mean) / std
        else:
            normalized = 0.0

    return {
        "raw": raw,
        "clipped": clipped,
        "normalized": normalized,
        "components": {
            "progress": comp_progress,
            "collision": comp_collision,
            "offroad": comp_offroad,
            "violation": comp_violation,
            "comfort": comp_comfort,
            "completion": comp_completion,
        },
        "stats": stats,
    }
