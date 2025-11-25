"""
Quick utility to sample CARLA map speed limits and print basic stats.

Usage:
    python tools/inspect_speed_limits.py --host 127.0.0.1 --port 2000

Notes:
- CARLA returns speed_limit in km/h. We also show m/s for convenience.
- Requires a running CARLA server.
"""

import argparse
import random
import statistics

import carla


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="10.0.0.121")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--samples", type=int, default=200, help="Number of waypoints to sample")
    args = ap.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(5.0)
    world = client.get_world()
    m = world.get_map()

    spawn_points = list(m.get_spawn_points())
    if not spawn_points:
        print("No spawn points found on the map.")
        return

    random.shuffle(spawn_points)
    sampled = spawn_points[: args.samples]

    limits_kmh = []
    for sp in sampled:
        wp = m.get_waypoint(sp.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp is None:
            continue
        limit = float(getattr(wp, "speed_limit", 0.0) or 0.0)
        limits_kmh.append(limit)

    if not limits_kmh:
        print("No speed limits retrieved (all zero?).")
        return

    limits_ms = [l / 3.6 for l in limits_kmh]
    uniq = {}
    for l in limits_kmh:
        uniq[l] = uniq.get(l, 0) + 1

    print(f"Sampled {len(limits_kmh)} driving waypoints across {len(spawn_points)} spawn points.")
    print(f"Unique speed limits (km/h): {sorted(uniq.items(), key=lambda x: x[0])}")
    print(
        f"Min/Max km/h: {min(limits_kmh):.1f}/{max(limits_kmh):.1f} | "
        f"Mean/Median km/h: {statistics.mean(limits_kmh):.1f}/{statistics.median(limits_kmh):.1f}"
    )
    print(
        f"Min/Max m/s:  {min(limits_ms):.2f}/{max(limits_ms):.2f} | "
        f"Mean/Median m/s: {statistics.mean(limits_ms):.2f}/{statistics.median(limits_ms):.2f}"
    )


if __name__ == "__main__":
    main()
