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
        """Return future samples at monotonically increasing indices.

        Using bisect alone can return the same index for several deltas when the
        vehicle is slow or stopped. We walk the buffer forward so each step
        advances by at least one element when available.
        """
        times_list = list(self.times)
        if not times_list:
            return [None] * len(deltas)

        earliest = times_list[0]
        latest = times_list[-1]
        required_latest = base_time + max(deltas)
        # Require the buffer to cover the whole horizon; otherwise skip
        if base_time < earliest or required_latest > latest:
            return [None] * len(deltas)

        tfs = list(self.transforms)
        spd = list(self.speeds)
        result: List[Optional[Tuple[carla.Transform, float]]] = []
        last_idx = -1

        for dt in deltas:
            target = base_time + dt
            idx = bisect.bisect_left(times_list, target)

            # Ensure progression through the buffer
            if idx <= last_idx:
                idx = last_idx + 1

            if idx >= len(times_list):
                result.append(None)
                last_idx = len(times_list)
                continue
            # Enforce that the sampled timestamp is not earlier than target (tolerance = 0)
            if times_list[idx] < target:
                result.append(None)
                last_idx = idx
                continue

            last_idx = idx
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


