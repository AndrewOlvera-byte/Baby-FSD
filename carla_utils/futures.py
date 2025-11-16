import bisect
import math
from collections import deque
from typing import Deque, List, Optional, Tuple

import carla


def yaw_rad(deg: float) -> float:
    return math.radians(deg)


def to_ego_xy(loc: carla.Location, ego_tf: carla.Transform) -> Tuple[float, float]:
    dx = loc.x - ego_tf.location.x
    dy = loc.y - ego_tf.location.y
    yaw = yaw_rad(ego_tf.rotation.yaw)
    c = math.cos(-yaw)
    s = math.sin(-yaw)
    return dx * c - dy * s, dx * s + dy * c


class FuturesBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.times: Deque[float] = deque(maxlen=capacity)
        self.transforms: Deque[carla.Transform] = deque(maxlen=capacity)
        self.speeds: Deque[float] = deque(maxlen=capacity)

    def add(self, sim_time: float, tf: carla.Transform, speed_mps: float) -> None:
        self.times.append(sim_time)
        self.transforms.append(tf)
        self.speeds.append(speed_mps)

    def sample_future(self, base_time: float, deltas: List[float]) -> List[Optional[Tuple[carla.Transform, float]]]:
        # Build a monotonically increasing times list for bisect
        times_list = list(self.times)
        result: List[Optional[Tuple[carla.Transform, float]]] = []
        for dt in deltas:
            target = base_time + dt
            idx = bisect.bisect_left(times_list, target)
            if idx >= len(times_list):
                result.append(None)
                continue
            # take the first time >= target
            # Convert deque to indexable lists on demand
            tfs = list(self.transforms)
            spd = list(self.speeds)
            result.append((tfs[idx], float(spd[idx])))
        return result


def future_waypoints_ego(
    base_ego_tf: carla.Transform,
    samples: List[Optional[Tuple[carla.Transform, float]]],
) -> Tuple[List[Tuple[float, float]], List[float]]:
    wps: List[Tuple[float, float]] = []
    vels: List[float] = []
    for item in samples:
        if item is None:
            break
        tf, v = item
        x, y = to_ego_xy(tf.location, base_ego_tf)
        wps.append((float(x), float(y)))
        vels.append(float(v))
    return wps, vels


