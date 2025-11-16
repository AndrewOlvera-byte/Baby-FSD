import math
from typing import List, Dict, Tuple

import carla


def _yaw_rad(deg: float) -> float:
    return math.radians(deg)


def _wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def world_to_ego_xy(loc: carla.Location, ego_tf: carla.Transform) -> Tuple[float, float]:
    dx = loc.x - ego_tf.location.x
    dy = loc.y - ego_tf.location.y
    yaw = _yaw_rad(ego_tf.rotation.yaw)
    c = math.cos(-yaw)
    s = math.sin(-yaw)
    x = dx * c - dy * s
    y = dx * s + dy * c
    return x, y


def world_vel_to_ego(v: carla.Vector3D, ego_tf: carla.Transform) -> Tuple[float, float]:
    yaw = _yaw_rad(ego_tf.rotation.yaw)
    c = math.cos(-yaw)
    s = math.sin(-yaw)
    x = v.x * c - v.y * s
    y = v.x * s + v.y * c
    return x, y


def collect_nearby_actors(world: carla.World, ego: carla.Vehicle, radius_m: float) -> List[Dict]:
    ego_tf = ego.get_transform()
    ego_loc = ego_tf.location
    out: List[Dict] = []

    actors = world.get_actors()
    for a in actors:
        # Filter vehicles and walkers only
        if a.type_id.startswith("vehicle."):
            # crude bike classification
            if "bike" in a.type_id or "bicycle" in a.type_id or "motorcycle" in a.type_id:
                type_int = 2
            else:
                type_int = 0
        elif a.type_id.startswith("walker."):
            type_int = 1
        else:
            continue

        loc = a.get_transform().location
        dx = loc.x - ego_loc.x
        dy = loc.y - ego_loc.y
        if dx * dx + dy * dy > radius_m * radius_m:
            continue

        x, y = world_to_ego_xy(loc, ego_tf)
        rel_yaw = 0.0
        if hasattr(a, "get_transform"):
            rel_yaw = _wrap_pi(_yaw_rad(a.get_transform().rotation.yaw) - _yaw_rad(ego_tf.rotation.yaw))

        vx, vy = 0.0, 0.0
        if hasattr(a, "get_velocity"):
            v = a.get_velocity()
            vx, vy = world_vel_to_ego(v, ego_tf)

        length, width = 0.0, 0.0
        if hasattr(a, "bounding_box") and a.bounding_box is not None:
            bb = a.bounding_box
            length = float(bb.extent.x * 2.0)
            width = float(bb.extent.y * 2.0)

        out.append({
            "actor_id": int(a.id),
            "type_id": int(type_int),
            "x_ego": float(x),
            "y_ego": float(y),
            "yaw": float(rel_yaw),
            "sin_yaw": float(math.sin(rel_yaw)),
            "cos_yaw": float(math.cos(rel_yaw)),
            "vx": float(vx),
            "vy": float(vy),
            "length": float(length),
            "width": float(width),
            "oncoming_flag": int(1 if vx < -1.0 else 0),
            "priority_flag": int(1 if type_int == 0 else 0),
        })

    return out


def get_ego_state(ego: carla.Vehicle) -> Tuple[float, float]:
    v = ego.get_velocity()
    speed = math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
    ang = ego.get_angular_velocity()
    yaw_rate = ang.z
    return float(speed), float(yaw_rate)


