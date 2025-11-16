import math
from typing import Dict, List, Tuple

import carla


def _yaw_rad(deg: float) -> float:
    return math.radians(deg)


def world_to_ego_xy(loc: carla.Location, ego_tf: carla.Transform) -> Tuple[float, float]:
    dx = loc.x - ego_tf.location.x
    dy = loc.y - ego_tf.location.y
    yaw = _yaw_rad(ego_tf.rotation.yaw)
    c = math.cos(-yaw)
    s = math.sin(-yaw)
    x = dx * c - dy * s
    y = dx * s + dy * c
    return x, y


def inside_window(x: float, y: float, half_window_m: float) -> bool:
    return abs(x) <= half_window_m and abs(y) <= half_window_m


def extract_lane_centerline_segments(
    world_map: carla.Map,
    ego_tf: carla.Transform,
    window_m: float = 80.0,
    step_m: float = 2.0,
) -> List[Dict]:
    half = window_m * 0.5
    segments: List[Dict] = []

    wp = world_map.get_waypoint(ego_tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
    if wp is None:
        return segments

    # Follow forward along the lane, collecting points within the window
    coords_x: List[float] = []
    coords_y: List[float] = []
    visited = set()
    cur = wp
    seg_id = 0

    # forward
    for _ in range(400):
        loc = cur.transform.location
        x, y = world_to_ego_xy(loc, ego_tf)
        if not inside_window(x, y, half):
            break
        key = (cur.road_id, cur.section_id, cur.lane_id, int(loc.x * 10), int(loc.y * 10))
        if key in visited:
            break
        visited.add(key)
        coords_x.append(float(x))
        coords_y.append(float(y))
        nxt = cur.next(step_m)
        if not nxt:
            break
        cur = nxt[0]

    if coords_x:
        segments.append({
            "seg_id": int(seg_id),
            "kind": 0,  # lane_centerline
            "coords_x": coords_x,
            "coords_y": coords_y,
            "lane_id": int(wp.lane_id),
            "lane_type": int(wp.lane_type.value if hasattr(wp.lane_type, "value") else 0),
            "speed_limit": float(getattr(wp, "speed_limit", 0.0) or 0.0),
        })
        seg_id += 1

    # backward
    coords_x_b: List[float] = []
    coords_y_b: List[float] = []
    cur = wp
    for _ in range(400):
        prev = cur.previous(step_m)
        if not prev:
            break
        cur = prev[0]
        loc = cur.transform.location
        x, y = world_to_ego_xy(loc, ego_tf)
        if not inside_window(x, y, half):
            break
        key = (cur.road_id, cur.section_id, cur.lane_id, int(loc.x * 10), int(loc.y * 10))
        if key in visited:
            break
        visited.add(key)
        coords_x_b.append(float(x))
        coords_y_b.append(float(y))

    if coords_x_b:
        coords_x_b.reverse(); coords_y_b.reverse()
        segments.insert(0, {
            "seg_id": int(seg_id),
            "kind": 0,
            "coords_x": coords_x_b,
            "coords_y": coords_y_b,
            "lane_id": int(wp.lane_id),
            "lane_type": int(wp.lane_type.value if hasattr(wp.lane_type, "value") else 0),
            "speed_limit": float(getattr(wp, "speed_limit", 0.0) or 0.0),
        })

    return segments


def extract_traffic_light_stoplines(world: carla.World, ego_tf: carla.Transform, window_m: float = 80.0) -> List[Dict]:
    half = window_m * 0.5
    tl_rows: List[Dict] = []
    tls = world.get_actors().filter('traffic.traffic_light*')
    for tl in tls:
        try:
            wps = tl.get_stop_waypoints()
        except Exception:
            wps = []
        if not wps:
            continue
        # Take the first stop line waypoint as representative
        sl = wps[0].transform.location
        x, y = world_to_ego_xy(sl, ego_tf)
        if not inside_window(x, y, half):
            continue
        state = getattr(tl.get_state(), "value", 0) if hasattr(tl, "get_state") else 0
        tl_rows.append({
            "tl_id": int(tl.id),
            "state": int(state),
            "stop_x_ego": float(x),
            "stop_y_ego": float(y),
        })
    return tl_rows


