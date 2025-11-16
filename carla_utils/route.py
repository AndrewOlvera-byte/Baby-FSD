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
INT_TO_ROAD_OPTION = {v: k for k, v in ROAD_OPTION_TO_INT.items()}
ROAD_OPTION_TEXT = {
    0: "Continue following lane",
    1: "Prepare for left turn",
    2: "Prepare for right turn",
    3: "Proceed straight through intersection",
    4: "Change lane to the left",
    5: "Change lane to the right",
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


def _normalize_angle_deg(a: float) -> float:
    """Normalize an angle in degrees to the [-180, 180) range."""
    return (a + 180.0) % 360.0 - 180.0


def _infer_cmd_from_yaw(cur_wp: carla.Waypoint, far_wp: carla.Waypoint) -> int:
    """Infer a high-level command (LaneFollow / Left / Right / Straight) from lane yaw change.

    This uses only map lane geometry and junction flags, and does not depend on any
    BehaviorAgent or LocalPlanner internals.
    """
    cur_yaw = cur_wp.transform.rotation.yaw
    far_yaw = far_wp.transform.rotation.yaw
    delta = _normalize_angle_deg(far_yaw - cur_yaw)

    # Outside junctions, treat as LaneFollow
    if not (cur_wp.is_junction or far_wp.is_junction):
        return ROAD_OPTION_TO_INT["LaneFollow"]

    if abs(delta) < 15.0:
        return ROAD_OPTION_TO_INT["Straight"]
    if delta > 0.0:
        return ROAD_OPTION_TO_INT["Left"]
    return ROAD_OPTION_TO_INT["Right"]


def _normalize_plan_entries(raw_plan: List) -> List[Tuple[carla.Waypoint, object]]:
    """Best-effort conversion of planner internals to (Waypoint, option) tuples."""
    normalized: List[Tuple[carla.Waypoint, object]] = []
    for item in raw_plan:
        waypoint = None
        option = None
        if isinstance(item, tuple):
            if item:
                waypoint = item[0]
            if len(item) > 1:
                option = item[1]
        else:
            waypoint = getattr(item, "waypoint", None) or getattr(item, "transform", None)
            option = getattr(item, "road_option", None)
        if waypoint is None:
            continue
        # LocalPlanner keeps carla.Waypoint objects; ensure they expose transform
        if not hasattr(waypoint, "transform"):
            continue
        normalized.append((waypoint, option))
    return normalized


def _extract_active_plan(agent) -> Tuple[List[Tuple[carla.Waypoint, object]], str]:
    planner = getattr(agent, "_local_planner", None)
    if planner is None:
        return [], "missing_planner"

    for attr in ("_waypoints_queue", "_waypoints_buffer"):
        seq = getattr(planner, attr, None)
        if seq:
            plan = _normalize_plan_entries(list(seq))
            if plan:
                return plan, attr

    try:
        plan = getattr(planner, "get_plan", lambda: [])()
        plan = _normalize_plan_entries(list(plan))
        if plan:
            return plan, "global_plan"
    except Exception:
        pass

    return [], "empty"


def _drop_consumed_waypoints(
    plan: List[Tuple[carla.Waypoint, object]],
    ego_tf: carla.Transform,
    min_forward_x: float,
    min_far_x: float,
    lookahead_idx: int,
) -> List[Tuple[carla.Waypoint, object]]:
    if not plan:
        return plan

    lookahead_idx = max(0, lookahead_idx)
    for start in range(len(plan)):
        candidate = plan[start:]
        wp_first = candidate[0][0]
        first_loc = wp_first.transform.location
        first_x, _ = _to_ego_frame_xy(first_loc, ego_tf)
        wp_far = candidate[min(len(candidate) - 1, lookahead_idx)][0]
        far_loc = wp_far.transform.location
        far_x, _ = _to_ego_frame_xy(far_loc, ego_tf)
        if first_x >= min_forward_x and far_x >= min_far_x:
            return candidate
    return plan[-1:]


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


def extract_cmd_and_route(
    agent,
    ego_tf: carla.Transform,
    k_points: int,
    spacing_m: float = 2.0,
    return_plan_stats: bool = False,
):
    # Road option
    try:
        cmd = getattr(agent._local_planner, "target_road_option", None)  # pylint: disable=protected-access
    except Exception:
        cmd = None
    cmd_int = _road_option_to_int(cmd) if cmd is not None else 0

    # Plan (prefer the active queue/buffer before falling back to the global route)
    plan, plan_source = _extract_active_plan(agent)
    plan = _drop_consumed_waypoints(
        plan,
        ego_tf,
        min_forward_x=0.0,
        min_far_x=1.0,
        lookahead_idx=7,
    )
    route_xy = resample_route_points_from_plan(plan, ego_tf, k_points, spacing_m)
    if return_plan_stats:
        first_wp_id = None
        if plan:
            wp = plan[0][0]
            first_wp_id = getattr(getattr(wp, "id", None), "phantom_id", None) or getattr(wp, "id", None)
        return cmd_int, route_xy, {
            "plan_len": len(plan),
            "first_waypoint_id": first_wp_id,
            "plan_source": plan_source,
        }
    return cmd_int, route_xy


def map_cmd_and_route(
    world: carla.World,
    ego_tf: carla.Transform,
    k_points: int,
    spacing_m: float = 2.0,
    lookahead_dist_m: float = 30.0,
    return_plan_stats: bool = False,
):
    """Compute a forward local route and high-level command from the map only.

    This is a BehaviorAgent-free helper that:
      - Traces forward along the lane graph from the ego pose using waypoints.
      - Builds a short local route polyline in ego frame (route_xy).
      - Infers a coarse command (LaneFollow / Left / Right / Straight) from yaw change.
    """
    m = world.get_map()
    wp = m.get_waypoint(
        ego_tf.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if wp is None:
        if return_plan_stats:
            return 0, [], {
                "plan_len": 0,
                "first_waypoint_id": None,
                "plan_source": "map_none",
            }
        return 0, []

    # Collect a chain of lane-following waypoints in world frame.
    plan_wps: List[carla.Waypoint] = [wp]
    cur_wp = wp
    dist_accum = 0.0
    while len(plan_wps) < k_points:
        nxt = cur_wp.next(spacing_m)
        if not nxt:
            break
        cur_wp = nxt[0]
        plan_wps.append(cur_wp)
        dist_accum += spacing_m
        if dist_accum >= lookahead_dist_m:
            break

    # Transform to ego frame
    route_xy: List[Tuple[float, float]] = []
    for w in plan_wps:
        x, y = _to_ego_frame_xy(w.transform.location, ego_tf)
        route_xy.append((x, y))

    # Command from yaw change
    far_idx = min(len(plan_wps) - 1, max(1, int(lookahead_dist_m / spacing_m)) - 1)
    far_wp = plan_wps[far_idx]
    cmd_int = _infer_cmd_from_yaw(wp, far_wp)

    if return_plan_stats:
        first_wp_id = getattr(getattr(wp, "id", None), "phantom_id", None) or getattr(wp, "id", None)
        return cmd_int, route_xy, {
            "plan_len": len(plan_wps),
            "first_waypoint_id": first_wp_id,
            "plan_source": "map",
        }

    return cmd_int, route_xy


def distance_to_goal(ego_tf: carla.Transform, goal_tf: carla.Transform) -> float:
    dx = goal_tf.location.x - ego_tf.location.x
    dy = goal_tf.location.y - ego_tf.location.y
    return math.hypot(dx, dy)


def command_int_to_label(cmd_int: int) -> str:
    return INT_TO_ROAD_OPTION.get(cmd_int, f"Unknown({cmd_int})")


def command_int_to_text(cmd_int: int) -> str:
    return ROAD_OPTION_TEXT.get(cmd_int, command_int_to_label(cmd_int))


