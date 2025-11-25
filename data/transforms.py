import torch
import random
import numpy as np
from typing import Dict, Tuple

class BCTransform:
    """
    CPU-side lightweight augmentations for BC data.

    Handles cheap operations (noise, scaling, shifts) that can run efficiently
    in parallel data loading workers.

    Heavy operations (BEV rotation, cutout) moved to GPU via gpu_augmentations.py
    to avoid CPU bottlenecks.

    Best Practice Split:
    - CPU (here): Element-wise ops, noise, scaling, dropout
    - GPU (gpu_augmentations.py): Image transforms (rotation, cutout)
    """
    def __init__(
        self,
        augment: bool = True,
        # Heavy ops moved to GPU (kept for config compatibility)
        p_rot: float = 0.5,
        max_rot_deg: float = 20.0,
        p_bev_cutout: float = 0.2,
        p_bev_channel_drop: float = 0.1,
        # Lightweight CPU ops
        p_drop_bev: float = 0.05,
        p_drop_objects: float = 0.05,
        p_ego_noise: float = 0.1,
        ego_noise_scale: float = 0.02,
        p_speed_scale: float = 0.2,
        speed_scale_range: Tuple[float, float] = (0.8, 1.2),
        p_trajectory_noise: float = 0.3,
        trajectory_noise_std: float = 0.02,
        p_lateral_shift: float = 0.2,
        max_lateral_shift: float = 0.05,
    ):
        self.augment = augment
        # Heavy ops - NOT applied here (moved to GPU)
        self.p_rot = p_rot  # Stored for config, but not used
        self.max_rot_deg = max_rot_deg
        self.p_bev_cutout = p_bev_cutout
        self.p_bev_channel_drop = p_bev_channel_drop
        # Lightweight ops - applied here
        self.p_drop_bev = p_drop_bev
        self.p_drop_objects = p_drop_objects
        self.p_ego_noise = p_ego_noise
        self.ego_noise_scale = ego_noise_scale
        self.p_speed_scale = p_speed_scale
        self.speed_scale_range = speed_scale_range
        self.p_trajectory_noise = p_trajectory_noise
        self.trajectory_noise_std = trajectory_noise_std
        self.p_lateral_shift = p_lateral_shift
        self.max_lateral_shift = max_lateral_shift

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply lightweight CPU augmentations.

        NOTE: Heavy operations (rotation, cutout, channel drop) are now handled
        on GPU in the training loop. This only does cheap element-wise operations.
        """
        if not self.augment:
            return sample

        # 1. Modality dropout (cheap: just zeroing)
        if random.random() < self.p_drop_bev:
            sample["bev"] = torch.zeros_like(sample["bev"])

        if random.random() < self.p_drop_objects:
            sample["objects"] = torch.zeros_like(sample["objects"])
            sample["object_mask"] = torch.zeros_like(sample["object_mask"])

        # 2. Small ego noise (cheap: element-wise addition)
        if random.random() < self.p_ego_noise:
            noise = torch.randn_like(sample["ego_vec"]) * self.ego_noise_scale
            sample["ego_vec"] = torch.clamp(sample["ego_vec"] + noise, -1.0, 1.0)

        # 3. Speed scaling (cheap: element-wise multiplication)
        if random.random() < self.p_speed_scale:
            scale = random.uniform(self.speed_scale_range[0], self.speed_scale_range[1])
            sample["future_v"] = torch.clamp(sample["future_v"] * scale, 0.0, 1.0)
            # Also scale ego speed
            sample["ego_vec"][0] = torch.clamp(sample["ego_vec"][0] * scale, 0.0, 1.0)

        # 4. Trajectory noise (cheap: element-wise addition)
        if random.random() < self.p_trajectory_noise:
            noise = torch.randn_like(sample["future_xy"]) * self.trajectory_noise_std
            sample["future_xy"] = torch.clamp(sample["future_xy"] + noise, -1.0, 1.0)

        # 5. Lateral shift (cheap: element-wise addition)
        if random.random() < self.p_lateral_shift:
            shift = random.uniform(-self.max_lateral_shift, self.max_lateral_shift)
            sample["route"][:, 1] = torch.clamp(sample["route"][:, 1] + shift, -1.0, 1.0)
            sample["future_xy"][:, 1] = torch.clamp(sample["future_xy"][:, 1] + shift, -1.0, 1.0)
            sample["objects"][:, 2] = torch.clamp(sample["objects"][:, 2] - shift, -1.0, 1.0)

        return sample

