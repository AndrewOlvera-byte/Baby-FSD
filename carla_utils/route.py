import math
from typing import List, Tuple, Optional

import carla

try:
    from agents.navigation.local_planner import RoadOption
except Exception:
    RoadOption = None  # type: ignore


ROAD_OPTION_TO_INT = {
    "LaneFollow": 0,
    "Left": 1,
    "Right": 2,
    "Straight": 3,
    "ChangeLaneLeft": 4,
    "ChangeLaneRight": 5,
}


def _road_option_to_int(opt) -> int:
    name = str(opt)
    # RoadOption.<Name>
    if name.startswith("RoadOption."):
        name = name.split(".", 1)[1]
    return ROAD_OPTION_TO_INT.get(name, 0)


def _to_ego_frame_xy(world_location: carla.Location, ego_tf: carla.Transform) -> Tuple[float, float]:
    dx = world_location.x - ego_tf.location.x
    dy = world_location.y - ego_tf.location.y
    yaw = math.radians(ego_tf.rotation.yaw)
    cos_y = math.cos(-yaw)
    sin_y = math.sin(-yaw)
    x = dx * cos_y - dy * sin_y
    y = dx * sin_y + dy * cos_y
    return x, y


def resample_route_points_from_plan(
    plan: List[Tuple[carla.Waypoint, object]],
    ego_tf: carla.Transform,
    k_points: int,
    spacing_m: float = 2.0,
) -> List[Tuple[float, float]]:
    if not plan:
        return []

    # Collect world XY along the plan at roughly uniform spacing
    points: List[Tuple[float, float]] = []
    last_l: Optional[carla.Location] = None
    accum = 0.0
    for wp, _ in plan:
        loc = wp.transform.location
        if last_l is None:
            points.append((loc.x, loc.y))
            last_l = loc
            continue
        dx = loc.x - last_l.x
        dy = loc.y - last_l.y
        d = math.hypot(dx, dy)
        accum += d
        if accum >= spacing_m:
            points.append((loc.x, loc.y))
            accum = 0.0
            if len(points) >= k_points:
                break
        last_l = loc

    # Transform to ego frame
    route_xy: List[Tuple[float, float]] = []
    for xw, yw in points[:k_points]:
        x, y = _to_ego_frame_xy(carla.Location(x=xw, y=yw, z=ego_tf.location.z), ego_tf)
        route_xy.append((x, y))
    return route_xy


def extract_cmd_and_route(agent, ego_tf: carla.Transform, k_points: int, spacing_m: float = 2.0):
    # Road option
    try:
        cmd = getattr(agent._local_planner, "target_road_option", None)  # pylint: disable=protected-access
    except Exception:
        cmd = None
    cmd_int = _road_option_to_int(cmd) if cmd is not None else 0

    # Plan
    try:
        plan = agent._local_planner.get_plan()  # type: ignore[attr-defined]
    except Exception:
        plan = []
    route_xy = resample_route_points_from_plan(plan, ego_tf, k_points, spacing_m)
    return cmd_int, route_xy


def distance_to_goal(ego_tf: carla.Transform, goal_tf: carla.Transform) -> float:
    dx = goal_tf.location.x - ego_tf.location.x
    dy = goal_tf.location.y - ego_tf.location.y
    return math.hypot(dx, dy)


