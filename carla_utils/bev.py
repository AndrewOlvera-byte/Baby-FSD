import math
import zlib
from typing import Dict, List, Tuple

import numpy as np


def _xy_to_px(x_m: float, y_m: float, meters_x: float, meters_y: float, res_m: float) -> Tuple[int, int]:
    """
    Convert ego-frame meters (x forward, y left) to pixel indices in BEV.
    Origin is ego; +x is up (-row), +y is right (+col) when visualized.
    """
    W = int(round(meters_x / res_m))
    H = int(round(meters_y / res_m))
    cx = W // 2
    cy = H // 2
    # x forward -> rows decrease; y left -> cols decrease
    col = int(round(cx + (y_m / res_m)))
    row = int(round(cy - (x_m / res_m)))
    return row, col


def _draw_point(mask: np.ndarray, r: int, c: int, radius_px: int = 0, value: float = 1.0) -> None:
    H, W = mask.shape
    if radius_px <= 0:
        if 0 <= r < H and 0 <= c < W:
            mask[r, c] = max(mask[r, c], value)
        return
    r0 = max(0, r - radius_px)
    r1 = min(H, r + radius_px + 1)
    c0 = max(0, c - radius_px)
    c1 = min(W, c + radius_px + 1)
    mask[r0:r1, c0:c1] = np.maximum(mask[r0:r1, c0:c1], value)


def _draw_polyline(mask: np.ndarray, pts_rc: List[Tuple[int, int]], thickness_px: int = 1, value: float = 1.0) -> None:
    if len(pts_rc) < 2:
        for r, c in pts_rc:
            _draw_point(mask, r, c, radius_px=max(0, thickness_px - 1), value=value)
        return
    H, W = mask.shape
    for (r0, c0), (r1, c1) in zip(pts_rc[:-1], pts_rc[1:]):
        dr = r1 - r0
        dc = c1 - c0
        steps = max(abs(dr), abs(dc), 1)
        for k in range(steps + 1):
            r = int(round(r0 + dr * (k / steps)))
            c = int(round(c0 + dc * (k / steps)))
            if 0 <= r < H and 0 <= c < W:
                _draw_point(mask, r, c, radius_px=max(0, thickness_px - 1), value=value)


def _polyline_from_xy(xy_m: List[Tuple[float, float]], meters_x: float, meters_y: float, res_m: float) -> List[Tuple[int, int]]:
    return [_xy_to_px(x, y, meters_x, meters_y, res_m) for (x, y) in xy_m]


def _clip(v: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.minimum(np.maximum(v, lo), hi)


def rasterize_bev(
    route_xy: List[Tuple[float, float]],
    map_polys: List[Dict],
    tls: List[Dict],
    actors: List[Dict],
    bev_cfg: Dict,
) -> Tuple[np.ndarray, Dict]:
    """
    Build BEV raster with the following channel order (float32):
      0: drivable_area (binary 0/1)
      1: lane_centerline
      2: lane_boundary_left
      3: lane_boundary_right
      4: stop_line
      5: crosswalk
      6: intersection
      7: route_corridor
      8: route_centerline
      9: traffic_light_red
     10: traffic_light_yellow
     11: traffic_light_green
     12: actors_vehicle
     13: actors_pedestrian
     14: actors_bike
     15: velocity_x
     16: velocity_y
     17: speed_limit_map (m/s)
    """
    meters_x = float(bev_cfg.get("meters_x", 100.0))
    meters_y = float(bev_cfg.get("meters_y", 80.0))
    res_m = float(bev_cfg.get("resolution_m", 0.25))
    lane_width_m = float(bev_cfg.get("lane_width_m", 3.5))
    corridor_width_m = float(bev_cfg.get("route_corridor_width_m", 6.0))
    centerline_thick_m = float(bev_cfg.get("centerline_thickness_m", 0.5))
    drivable_thick_m = float(bev_cfg.get("drivable_thickness_m", 5.0))
    actor_radius_m = float(bev_cfg.get("actor_radius_m", 1.0))

    W = int(round(meters_x / res_m))
    H = int(round(meters_y / res_m))
    C = 18

    bev = np.zeros((C, H, W), dtype=np.float32)

    # Helper: thickness in pixels
    tl_centerline_px = max(1, int(round(centerline_thick_m / res_m)))
    tl_drivable_px = max(1, int(round(drivable_thick_m / res_m)))
    tl_boundary_px = max(1, int(round((lane_width_m * 0.5) / res_m)))
    tl_corridor_px = max(1, int(round((corridor_width_m * 0.5) / res_m)))
    act_radius_px = max(1, int(round(actor_radius_m / res_m)))

    # Lane centerlines and approximate drivable area from map segments (kind==0)
    for seg in map_polys or []:
        if int(seg.get("kind", -1)) != 0:
            continue
        xs = seg.get("coords_x", [])
        ys = seg.get("coords_y", [])
        if not xs or not ys or len(xs) != len(ys):
            continue
        pts_rc = _polyline_from_xy(list(zip(xs, ys)), meters_x, meters_y, res_m)
        _draw_polyline(bev[1], pts_rc, thickness_px=tl_centerline_px, value=1.0)
        _draw_polyline(bev[0], pts_rc, thickness_px=tl_drivable_px, value=1.0)

        # Approximate left/right lane boundaries by offsetting along normals
        # Using simple finite differences per segment
        for (x0, y0), (x1, y1) in zip(list(zip(xs, ys))[:-1], list(zip(xs, ys))[1:]):
            dx = x1 - x0
            dy = y1 - y0
            L = math.hypot(dx, dy)
            if L < 1e-3:
                continue
            nx = -dy / L
            ny = dx / L
            lx0, ly0 = x0 + nx * (lane_width_m * 0.5), y0 + ny * (lane_width_m * 0.5)
            lx1, ly1 = x1 + nx * (lane_width_m * 0.5), y1 + ny * (lane_width_m * 0.5)
            rx0, ry0 = x0 - nx * (lane_width_m * 0.5), y0 - ny * (lane_width_m * 0.5)
            rx1, ry1 = x1 - nx * (lane_width_m * 0.5), y1 - ny * (lane_width_m * 0.5)
            l_rc = _polyline_from_xy([(lx0, ly0), (lx1, ly1)], meters_x, meters_y, res_m)
            r_rc = _polyline_from_xy([(rx0, ry0), (rx1, ry1)], meters_x, meters_y, res_m)
            _draw_polyline(bev[2], l_rc, thickness_px=1, value=1.0)
            _draw_polyline(bev[3], r_rc, thickness_px=1, value=1.0)
        # Speed limit along the segment
        vlim = float(seg.get("speed_limit", 0.0))
        if vlim > 0.0:
            _draw_polyline(bev[17], pts_rc, thickness_px=tl_corridor_px, value=vlim / 3.6 if vlim > 30 else vlim)

    # Traffic light stop lines (approx: mark the stop point with a short bar)
    for tl in tls or []:
        x = float(tl.get("stop_x_ego", 0.0))
        y = float(tl.get("stop_y_ego", 0.0))
        r, c = _xy_to_px(x, y, meters_x, meters_y, res_m)
        _draw_point(bev[4], r, c, radius_px=2, value=1.0)
        st = int(tl.get("state", 0))
        if st == 0:
            _draw_point(bev[9], r, c, radius_px=2, value=1.0)
        elif st == 1:
            _draw_point(bev[10], r, c, radius_px=2, value=1.0)
        elif st == 2:
            _draw_point(bev[11], r, c, radius_px=2, value=1.0)

    # Route corridor and centerline
    if route_xy:
        pts_rc = _polyline_from_xy(route_xy, meters_x, meters_y, res_m)
        _draw_polyline(bev[8], pts_rc, thickness_px=1, value=1.0)
        _draw_polyline(bev[7], pts_rc, thickness_px=tl_corridor_px, value=1.0)

    # Actors footprints + velocity fields
    for a in actors or []:
        ax = float(a.get("x_ego", 0.0))
        ay = float(a.get("y_ego", 0.0))
        r, c = _xy_to_px(ax, ay, meters_x, meters_y, res_m)
        typ = int(a.get("type_id", 0))
        radius = act_radius_px
        if typ == 0:
            _draw_point(bev[12], r, c, radius_px=radius, value=1.0)
        elif typ == 1:
            _draw_point(bev[13], r, c, radius_px=max(1, radius - 1), value=1.0)
        else:
            _draw_point(bev[14], r, c, radius_px=max(1, radius - 1), value=1.0)
        vx = float(a.get("vx", 0.0))
        vy = float(a.get("vy", 0.0))
        if 0 <= r < H and 0 <= c < W:
            bev[15, r, c] = bev[15, r, c] + vx
            bev[16, r, c] = bev[16, r, c] + vy

    # Clip velocity maps to a reasonable range
    bev[15] = _clip(bev[15], -40.0, 40.0)
    bev[16] = _clip(bev[16], -40.0, 40.0)

    meta = {
        "C": C,
        "H": H,
        "W": W,
        "dtype": "float32",
        "encoding": "npy.zlib",
        "channel_spec": "drivable,lane_ctr,lane_l,lane_r,stop,xwalk,inter,route_corr,route_ctr,tl_r,tl_y,tl_g,act_car,act_ped,act_bike,vx,vy,speed_lim",
        "meters_per_px": float(res_m),
        "x_fwd_m": float(meters_x),
        "y_left_m": float(meters_y),
        "spec_version": int(1),
        "norms_version": int(1),
    }
    return bev, meta


def encode_bev_to_bytes(bev: np.ndarray) -> bytes:
    """
    Encode CxHxW float32 array to compressed bytes (npy.zlib).
    """
    assert bev.dtype == np.float32 and bev.ndim == 3
    # np.save to a bytes buffer
    import io

    buf = io.BytesIO()
    np.save(buf, bev, allow_pickle=False)
    raw = buf.getvalue()
    return zlib.compress(raw, level=6)


