"""
GPU-accelerated augmentations using Kornia.

This module handles expensive image operations (rotation, cutout) on GPU
to avoid CPU bottlenecks in data loading workers.

Best Practice:
- Lightweight ops (noise, scaling) run in CPU workers (parallel)
- Heavy ops (image transforms) run on GPU (after batch transfer)
"""

import torch
import torch.nn as nn
import random
from typing import Dict


class GPUBatchAugmentation(nn.Module):
    """
    GPU-accelerated batch augmentation for BC data.

    Applied after batch is moved to GPU, before forward pass.
    Handles expensive image operations that bottleneck CPU workers.
    """

    def __init__(
        self,
        p_rot: float = 0.0,
        enable_rotation: bool = False,
        max_rot_deg: float = 20.0,
        p_bev_cutout: float = 0.2,
        cutout_size_range: tuple = (10, 30),
        p_bev_channel_drop: float = 0.1,
    ):
        """
        Args:
            p_rot: Probability of applying rotation
            max_rot_deg: Maximum rotation angle in degrees
            p_bev_cutout: Probability of applying cutout to BEV
            cutout_size_range: (min, max) size for cutout patches
            p_bev_channel_drop: Probability of dropping a BEV channel
        """
        super().__init__()
        self.p_rot = p_rot
        self.enable_rotation = enable_rotation
        self.max_rot_deg = max_rot_deg
        self.p_bev_cutout = p_bev_cutout
        self.cutout_size_range = cutout_size_range
        self.p_bev_channel_drop = p_bev_channel_drop

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply GPU augmentations to batch.

        Args:
            batch: Dict with tensors already on GPU

        Returns:
            Augmented batch (in-place modification)
        """
        if not self.training:
            return batch

        B = batch["bev"].shape[0]
        device = batch["bev"].device

        # 1. Random rotation (per-sample)
        if self.enable_rotation and self.p_rot > 0.0:
            # Generate random angles for batch
            angles = torch.zeros(B, device=device)
            rot_mask = torch.rand(B, device=device) < self.p_rot
            angles[rot_mask] = (torch.rand(rot_mask.sum(), device=device) * 2 - 1) * self.max_rot_deg

            # Apply rotation to samples that need it
            if rot_mask.any():
                batch = self._apply_rotation_batch(batch, angles, rot_mask)

        # 2. BEV cutout (per-sample, independent)
        if self.p_bev_cutout > 0.0:
            cutout_mask = torch.rand(B, device=device) < self.p_bev_cutout
            if cutout_mask.any():
                batch["bev"] = self._apply_cutout(batch["bev"], cutout_mask)

        # 3. BEV channel dropout (per-sample)
        if self.p_bev_channel_drop > 0.0:
            drop_mask = torch.rand(B, device=device) < self.p_bev_channel_drop
            if drop_mask.any():
                batch["bev"] = self._apply_channel_drop(batch["bev"], drop_mask)

        return batch

    def _apply_rotation_batch(
        self,
        batch: Dict[str, torch.Tensor],
        angles: torch.Tensor,
        rot_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Apply rotation to batch elements with vectorized operations.

        Args:
            batch: Input batch
            angles: Rotation angles in degrees (B,)
            rot_mask: Boolean mask of which samples to rotate (B,)
        """
        B = angles.shape[0]
        device = angles.device

        # Convert angles to radians for samples that need rotation
        rads = torch.deg2rad(angles)  # (B,)
        cos_r = torch.cos(rads)  # (B,)
        sin_r = torch.sin(rads)  # (B,)

        # Build rotation matrices (B, 2, 2)
        rot_matrices = torch.zeros(B, 2, 2, device=device)
        rot_matrices[:, 0, 0] = cos_r
        rot_matrices[:, 0, 1] = -sin_r
        rot_matrices[:, 1, 0] = sin_r
        rot_matrices[:, 1, 1] = cos_r

        # Only rotate samples where rot_mask is True
        # For efficiency, only process rotated samples

        # A. Rotate BEV images using grid_sample (fast on GPU)
        # Only rotate samples that need it
        if rot_mask.any():
            batch["bev"] = self._rotate_bev_images(batch["bev"], angles)

        # B. Rotate vector fields (route, future_xy, objects)
        # Route: (B, K, 2) -> (B, K, 2)
        batch["route"] = torch.bmm(batch["route"], rot_matrices.transpose(1, 2))

        # Future XY: (B, N, 2) -> (B, N, 2)
        batch["future_xy"] = torch.bmm(batch["future_xy"], rot_matrices.transpose(1, 2))

        # Objects: (B, M, 11)
        # Positions (x, y) at indices 1:3
        batch["objects"][:, :, 1:3] = torch.bmm(
            batch["objects"][:, :, 1:3],
            rot_matrices.transpose(1, 2)
        )

        # Velocities (vx, vy) at indices 7:9
        batch["objects"][:, :, 7:9] = torch.bmm(
            batch["objects"][:, :, 7:9],
            rot_matrices.transpose(1, 2)
        )

        # Update object yaw (sin, cos) at indices 3:5
        # yaw' = yaw + rotation
        # sin(a+b) = sin(a)cos(b) + cos(a)sin(b)
        # cos(a+b) = cos(a)cos(b) - sin(a)sin(b)
        sin_yaw = batch["objects"][:, :, 3]  # (B, M)
        cos_yaw = batch["objects"][:, :, 4]  # (B, M)

        # Broadcast cos_r, sin_r from (B,) to (B, M)
        cos_r_expanded = cos_r.unsqueeze(1)  # (B, 1)
        sin_r_expanded = sin_r.unsqueeze(1)  # (B, 1)

        new_sin = sin_yaw * cos_r_expanded + cos_yaw * sin_r_expanded
        new_cos = cos_yaw * cos_r_expanded - sin_yaw * sin_r_expanded

        batch["objects"][:, :, 3] = new_sin
        batch["objects"][:, :, 4] = new_cos

        return batch

    def _rotate_bev_images(self, bev: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        """
        Rotate BEV images using affine_grid + grid_sample (GPU-accelerated).

        Args:
            bev: (B, C, H, W) BEV tensor
            angles: (B,) rotation angles in degrees

        Returns:
            Rotated BEV (B, C, H, W)
        """
        B, C, H, W = bev.shape
        device = bev.device

        # Convert angles to radians
        rads = torch.deg2rad(angles)
        cos_a = torch.cos(rads)
        sin_a = torch.sin(rads)

        # Build affine transformation matrices (B, 2, 3)
        # Rotation matrix around center
        theta = torch.zeros(B, 2, 3, device=device)
        theta[:, 0, 0] = cos_a
        theta[:, 0, 1] = sin_a
        theta[:, 1, 0] = -sin_a
        theta[:, 1, 1] = cos_a
        # No translation (already centered at origin)

        # Generate sampling grid
        grid = torch.nn.functional.affine_grid(theta, bev.size(), align_corners=False)

        # Apply transformation
        rotated = torch.nn.functional.grid_sample(
            bev, grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )

        return rotated

    def _apply_cutout(self, bev: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Apply random cutout to BEV images (GPU-accelerated).

        Args:
            bev: (B, C, H, W) BEV tensor
            mask: (B,) boolean mask of which samples to apply cutout

        Returns:
            BEV with cutout applied
        """
        B, C, H, W = bev.shape
        device = bev.device

        # Get indices of samples to augment
        indices = torch.where(mask)[0]

        for idx in indices:
            # Random cutout size
            rh = random.randint(self.cutout_size_range[0], self.cutout_size_range[1])
            rw = random.randint(self.cutout_size_range[0], self.cutout_size_range[1])

            # Random position
            if W > rw and H > rh:
                x = random.randint(0, W - rw)
                y = random.randint(0, H - rh)
                bev[idx, :, y:y+rh, x:x+rw] = 0.0

        return bev

    def _apply_channel_drop(self, bev: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Drop random channels from BEV (GPU-accelerated).

        Args:
            bev: (B, C, H, W) BEV tensor
            mask: (B,) boolean mask of which samples to drop channels

        Returns:
            BEV with channels dropped
        """
        B, C, H, W = bev.shape

        # Get indices of samples to augment
        indices = torch.where(mask)[0]

        for idx in indices:
            drop_c = random.randint(0, C - 1)
            bev[idx, drop_c] = 0.0

        return bev
