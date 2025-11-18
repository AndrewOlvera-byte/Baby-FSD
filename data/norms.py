"""
Normalization functions for BC training.

All normalization constants are hardcoded for generalization across datasets.
"""

import torch
import numpy as np
from typing import Tuple, Union

# ============================================================================
# Normalization Constants
# ============================================================================

# Ego vector normalization (per-field)
V_MAX = 50.0            # m/s (~180 km/h)
ACCEL_MAX = 20.0        # m/s²
YAW_RATE_MAX = 3.0      # rad/s
CURVATURE_MAX = 0.3     # 1/m (tight curve)
STEER_MAX = 1.0         # normalized [-1, 1] already
THROTTLE_MAX = 1.0      # [0, 1] already
BRAKE_MAX = 1.0         # [0, 1] already
SPEED_LIMIT_MAX = 50.0  # m/s

# Spatial normalization (ego frame)
SPATIAL_MAX = 100.0     # meters (route, futures, objects)

# BEV channels (0-11 are binary [0,1], already normalized)
# Channels 15-16 (vx, vy) clipped to [-40, 40] in bev.py
BEV_VEL_MAX = 40.0      # m/s
BEV_SPEED_LIMIT_MAX = 50.0  # m/s (channel 17)


# ============================================================================
# Ego Vector Normalization
# ============================================================================

def normalize_ego_vector(frame_row) -> torch.Tensor:
    """
    Normalize ego vehicle state from frame row.
    
    Args:
        frame_row: pandas DataFrame row or dict-like with ego state fields
        
    Returns:
        tensor: (d_ego,) normalized ego vector
        
    Fields (in order):
        0: speed_mps / V_MAX
        1: yaw_rate / YAW_RATE_MAX
        2: accel_long / ACCEL_MAX
        3: accel_lat / ACCEL_MAX
        4: curvature / CURVATURE_MAX
        5: steer_angle_rad / π (already ~[-1, 1])
        6: steer_norm (already [-1, 1])
        7: throttle (already [0, 1])
        8: brake (already [0, 1])
        9: speed_limit_mps / SPEED_LIMIT_MAX
        10: command / 5.0 (commands are 0-5)
        11: gear / 6.0 (typical -1 to 5)
        12: time_of_day_sin (already [-1, 1])
        13: time_of_day_cos (already [-1, 1])
    """
    ego_vec = torch.tensor([
        float(frame_row.speed_mps) / V_MAX,
        float(frame_row.yaw_rate) / YAW_RATE_MAX,
        float(frame_row.accel_long) / ACCEL_MAX,
        float(frame_row.accel_lat) / ACCEL_MAX,
        float(frame_row.curvature) / CURVATURE_MAX,
        float(frame_row.steer_angle_rad) / np.pi,
        float(frame_row.steer_norm),  # already normalized
        float(frame_row.throttle),     # already normalized
        float(frame_row.brake),        # already normalized
        float(frame_row.speed_limit_mps) / SPEED_LIMIT_MAX,
        float(frame_row.command) / 5.0,
        float(frame_row.gear) / 6.0,
        float(frame_row.time_of_day_sin),
        float(frame_row.time_of_day_cos),
    ], dtype=torch.float32)
    
    return ego_vec


# ============================================================================
# Route Normalization
# ============================================================================

def normalize_route_points(route_xy: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    """
    Normalize route points in ego frame.
    
    Args:
        route_xy: (K, 2) array of (x, y) in ego frame (meters)
        
    Returns:
        tensor: (K, 2) normalized route points in [-1, 1]
    """
    if isinstance(route_xy, np.ndarray):
        route_xy = torch.from_numpy(route_xy)
    
    route_xy = route_xy.float()
    
    # Normalize to [-1, 1] based on SPATIAL_MAX
    route_norm = route_xy / SPATIAL_MAX
    
    # Clip to reasonable range
    route_norm = torch.clamp(route_norm, -1.0, 1.0)
    
    return route_norm


# ============================================================================
# Future Waypoints Normalization
# ============================================================================

def normalize_futures(
    futures_xy: Union[np.ndarray, torch.Tensor],
    futures_v: Union[np.ndarray, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Normalize future waypoints and speeds.
    
    Args:
        futures_xy: (N, 2) array of (x, y) in ego frame (meters)
        futures_v: (N,) array of speeds (m/s)
        
    Returns:
        futures_xy_norm: (N, 2) normalized waypoints in [-1, 1]
        futures_v_norm: (N,) normalized speeds in [0, 1]
    """
    if isinstance(futures_xy, np.ndarray):
        futures_xy = torch.from_numpy(futures_xy)
    if isinstance(futures_v, np.ndarray):
        futures_v = torch.from_numpy(futures_v)
    
    futures_xy = futures_xy.float()
    futures_v = futures_v.float()
    
    # Normalize waypoints to [-1, 1]
    futures_xy_norm = futures_xy / SPATIAL_MAX
    futures_xy_norm = torch.clamp(futures_xy_norm, -1.0, 1.0)
    
    # Normalize speeds to [0, 1]
    futures_v_norm = futures_v / V_MAX
    futures_v_norm = torch.clamp(futures_v_norm, 0.0, 1.0)
    
    return futures_xy_norm, futures_v_norm


# ============================================================================
# Object Tokens Normalization
# ============================================================================

def normalize_object_tokens(tokens: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    """
    Normalize object tokens.
    
    Args:
        tokens: (M, d_obj) array of object features
        
    Feature columns (from schema):
        0: type_id (categorical, don't normalize - will use embedding)
        1: x_ego (meters)
        2: y_ego (meters)
        3: sin_yaw (already [-1, 1])
        4: cos_yaw (already [-1, 1])
        5: length (meters)
        6: width (meters)
        7: vx (m/s)
        8: vy (m/s)
        9: oncoming_flag (binary)
        10: priority_flag (binary)
        
    Returns:
        tensor: (M, d_obj) normalized tokens
    """
    if isinstance(tokens, np.ndarray):
        tokens = torch.from_numpy(tokens)
    
    tokens = tokens.float().clone()
    
    # Normalize spatial positions (columns 1, 2)
    tokens[:, 1:3] = tokens[:, 1:3] / SPATIAL_MAX
    tokens[:, 1:3] = torch.clamp(tokens[:, 1:3], -1.0, 1.0)
    
    # Sin/cos yaw already in [-1, 1] (columns 3, 4)
    
    # Normalize dimensions (columns 5, 6) - typical vehicle ~5m x 2m
    tokens[:, 5] = tokens[:, 5] / 10.0  # length
    tokens[:, 6] = tokens[:, 6] / 5.0   # width
    tokens[:, 5:7] = torch.clamp(tokens[:, 5:7], 0.0, 1.0)
    
    # Normalize velocities (columns 7, 8)
    tokens[:, 7:9] = tokens[:, 7:9] / V_MAX
    tokens[:, 7:9] = torch.clamp(tokens[:, 7:9], -1.0, 1.0)
    
    # Flags already binary (columns 9, 10)
    
    return tokens


# ============================================================================
# BEV Normalization
# ============================================================================

def normalize_bev(bev_tensor: torch.Tensor) -> torch.Tensor:
    """
    Normalize BEV tensor channels.
    
    Args:
        bev_tensor: (C, H, W) BEV tensor with 18 channels
        
    Channel layout:
        0-11: Binary/semantic channels (already [0,1])
        12-14: Actor presence (already [0,1])
        15-16: Velocity fields (vx, vy) clipped to [-40, 40]
        17: Speed limit map (m/s)
        
    Returns:
        tensor: (C, H, W) normalized BEV
    """
    bev_norm = bev_tensor.clone()
    
    # Channels 0-14 already in [0, 1]
    
    # Normalize velocity channels (15, 16)
    bev_norm[15] = bev_norm[15] / BEV_VEL_MAX
    bev_norm[16] = bev_norm[16] / BEV_VEL_MAX
    bev_norm[15:17] = torch.clamp(bev_norm[15:17], -1.0, 1.0)
    
    # Normalize speed limit channel (17)
    bev_norm[17] = bev_norm[17] / BEV_SPEED_LIMIT_MAX
    bev_norm[17] = torch.clamp(bev_norm[17], 0.0, 1.0)
    
    return bev_norm


# ============================================================================
# Denormalization (for inference/visualization)
# ============================================================================

def denormalize_futures(
    futures_xy_norm: torch.Tensor,
    futures_v_norm: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Denormalize future waypoints and speeds back to original scale.
    
    Args:
        futures_xy_norm: (N, 2) normalized waypoints in [-1, 1]
        futures_v_norm: (N,) normalized speeds in [0, 1]
        
    Returns:
        futures_xy: (N, 2) waypoints in meters
        futures_v: (N,) speeds in m/s
    """
    futures_xy = futures_xy_norm * SPATIAL_MAX
    futures_v = futures_v_norm * V_MAX
    
    return futures_xy, futures_v


def denormalize_waypoints(waypoints_norm: torch.Tensor) -> torch.Tensor:
    """
    Denormalize waypoints back to meters.
    
    Args:
        waypoints_norm: (..., 2) normalized waypoints in [-1, 1]
        
    Returns:
        waypoints: (..., 2) waypoints in meters
    """
    return waypoints_norm * SPATIAL_MAX


def denormalize_speed(speed_norm: torch.Tensor) -> torch.Tensor:
    """
    Denormalize speed back to m/s.
    
    Args:
        speed_norm: (...,) normalized speeds in [0, 1]
        
    Returns:
        speed: (...,) speeds in m/s
    """
    return speed_norm * V_MAX

