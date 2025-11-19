import torch
import random
import torchvision.transforms.functional as F
import numpy as np
from typing import Dict

class BCTransform:
    """
    Applies consistent randomization to Ego, BEV, Route, and Futures
    to teach the model how to recover from perturbations.
    """
    def __init__(
        self, 
        augment: bool = True, 
        p_trans: float = 0.5,
        max_rot_deg: float = 20.0,   # +/- 20 degrees rotation
        max_trans_m: float = 1.0,    # +/- 1.0 meter lateral shift
        bev_crop_size: int = 96      # Crop center of BEV (if larger input)
    ):
        self.augment = augment
        self.p_trans = p_trans
        self.max_rot_deg = max_rot_deg
        self.max_trans_m = max_trans_m
        self.bev_crop_size = bev_crop_size

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not self.augment:
            return sample

        # 1. Random Rotation (Simulates heading error)
        rot_deg = 0.0
        if random.random() < self.p_trans:
            rot_deg = random.uniform(-self.max_rot_deg, self.max_rot_deg)

        if rot_deg != 0:
            # Convert angle to radians
            rad = torch.deg2rad(torch.tensor(rot_deg))
            c, s = torch.cos(rad), torch.sin(rad)
            rot_mat = torch.tensor([[c, -s], [s, c]])

            # A. Rotate BEV Image (C, H, W)
            # torchvision expects (..., H, W)
            # angle is in degrees, counter-clockwise
            sample["bev"] = F.rotate(sample["bev"], angle=float(rot_deg))

            # B. Rotate Vector Inputs (Route, Futures, Object Pos)
            # These are (N, 2) xy vectors.
            # Rotation matrix is 2x2.
            # v' = R v
            # If v is (N, 2), we want (v R^T) -> (N, 2)
            
            # Route: (K, 2)
            sample["route"] = sample["route"] @ rot_mat.T
            
            # Futures: (N, 2) - Labels must rotate too!
            sample["future_xy"] = sample["future_xy"] @ rot_mat.T
            
            # Objects: (M, 11)
            # Schema: 0:type, 1:x, 2:y, 3:sin, 4:cos, 5:len, 6:wid, 7:vx, 8:vy, 9:oncoming, 10:priority
            # Rotate positions (x,y) at indices 1,2
            sample["objects"][:, 1:3] = sample["objects"][:, 1:3] @ rot_mat.T
            # Rotate velocities (vx,vy) at indices 7,8
            sample["objects"][:, 7:9] = sample["objects"][:, 7:9] @ rot_mat.T
            
            # Adjust Object Yaw (sin/cos)
            # yaw' = yaw + rot_rad
            # sin(a+b) = sin(a)cos(b) + cos(a)sin(b)
            # cos(a+b) = cos(a)cos(b) - sin(a)sin(b)
            sin_yaw = sample["objects"][:, 3]
            cos_yaw = sample["objects"][:, 4]
            
            new_sin = sin_yaw * c + cos_yaw * s
            new_cos = cos_yaw * c - sin_yaw * s
            
            sample["objects"][:, 3] = new_sin
            sample["objects"][:, 4] = new_cos
        
        # 3. BEV Dropout / Cutout (Forces model to use Route/Ego info)
        if random.random() < 0.2:
            # Randomly zero out a rectangle in BEV
            C, h, w = sample["bev"].shape
            # Random size 10-30
            rh = random.randint(10, 30)
            rw = random.randint(10, 30)
            # Ensure valid range
            if w - rw > 0 and h - rh > 0:
                x = random.randint(0, w - rw)
                y = random.randint(0, h - rh)
                sample["bev"][:, y:y+rh, x:x+rw] = 0.0

        return sample

