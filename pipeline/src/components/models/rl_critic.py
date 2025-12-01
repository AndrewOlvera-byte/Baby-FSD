import math
import torch
import torch.nn as nn
from typing import Dict, Optional

from src.core.registry import register


class EgoEncoder(nn.Module):
    def __init__(self, d_model: int = 256, ego_dim: int = 14):
        super().__init__()
        self.ego_proj = nn.Linear(ego_dim, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, ego_vec: torch.Tensor) -> torch.Tensor:
        x = self.ego_proj(ego_vec)
        return self.layer_norm(x)


class RouteEncoder(nn.Module):
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.proj = nn.Linear(2, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, route: torch.Tensor) -> torch.Tensor:
        B, N, _ = route.shape
        x = self.proj(route)
        return self.layer_norm(x)


class ObjectEncoder(nn.Module):
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.proj = nn.Linear(11, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, objects: torch.Tensor, objects_mask: torch.Tensor) -> torch.Tensor:
        B, M, _ = objects.shape
        x = self.proj(objects)
        x = self.layer_norm(x)
        x = x.masked_fill(~objects_mask.unsqueeze(-1), 0.0)
        return x


class BEVViTEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 18,
        patch_size: int = 16,
        d_model: int = 256,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model

        self.patch_embed = nn.Conv2d(
            in_channels, d_model, kernel_size=patch_size, stride=patch_size
        )

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=int(d_model * mlp_ratio),
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=depth,
        )

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, bev: torch.Tensor) -> torch.Tensor:
        B, C, H, W = bev.shape
        x = self.patch_embed(bev)
        _, _, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.transformer(x)
        return self.layer_norm(x)


@register("model", "rl_critic")
class RLCritic(nn.Module):
    def __init__(
        self,
        bev_channels: int = 18,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        bev_patch_size: int = 16,
        bev_depth: int = 4,
        bev_heads: int = 8,
        action_dim: int = 3,
        ego_dim: int = 14,
    ):
        super().__init__()

        self.d_model = d_model
        self.action_dim = action_dim

        self.ego_encoder = EgoEncoder(d_model, ego_dim)
        self.route_encoder = RouteEncoder(d_model)
        self.object_encoder = ObjectEncoder(d_model)
        self.bev_encoder = BEVViTEncoder(
            in_channels=bev_channels,
            patch_size=bev_patch_size,
            d_model=d_model,
            depth=bev_depth,
            num_heads=bev_heads,
            dropout=dropout,
        )

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=num_encoder_layers,
        )

        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
        )

        self.q_head = nn.Sequential(
            nn.Linear(d_model * 2, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(
        self, batch: Dict[str, torch.Tensor], action: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        ego_feat = self.ego_encoder(batch["ego_vec"])
        route_feat = self.route_encoder(batch["route"])
        obj_feat = self.object_encoder(batch["objects"], batch["object_mask"])
        bev_feat = self.bev_encoder(batch["bev"])

        ego_feat = ego_feat.unsqueeze(1)

        features = torch.cat([ego_feat, route_feat, obj_feat, bev_feat], dim=1)

        encoded = self.encoder(features)

        state_pooled = encoded[:, 0]

        if action is None:
            action = batch.get("action", None)
            if action is None:
                raise ValueError("Action must be provided either as argument or in batch")

        action_feat = self.action_encoder(action)

        combined = torch.cat([state_pooled, action_feat], dim=-1)

        q_value = self.q_head(combined)

        return q_value.squeeze(-1)
