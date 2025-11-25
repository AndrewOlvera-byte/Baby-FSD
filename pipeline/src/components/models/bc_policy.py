"""
Behavior Cloning Policy Model - Lightweight Query-Based Planning Architecture.

Architecture:
- Input Encoders: Ego MLP, Route MLP, Object MLP (with type embeddings), BEV ViT
- Transformer Encoder: Self-attention across all modality tokens
- Query Decoder: Cross-attention with learnable future query tokens
- Output Heads: Separate MLPs for waypoint (x,y) and speed (v) prediction
"""

from __future__ import annotations
import math
import torch
from torch import nn
from src.core.registry import register


@register("model", "bc_policy")
class BCPolicyFactory:
    """Factory for BC policy model."""
    
    def build(self, cfg, context):
        """
        Build BC policy from config.
        
        Args:
            cfg: Model config node
            context: Dict with any required context
            
        Returns:
            BCPolicy instance
        """
        return BCPolicy(
            d_model=int(getattr(cfg, "d_model", 256)),
            n_heads=int(getattr(cfg, "n_heads", 8)),
            n_layers=int(getattr(cfg, "n_layers", 6)),
            dropout=float(getattr(cfg, "dropout", 0.1)),
            patch_size=list(getattr(cfg, "patch_size", [8, 8])),
            ego_hidden=int(getattr(cfg, "ego_hidden", 128)),
            route_hidden=int(getattr(cfg, "route_hidden", 128)),
            object_hidden=int(getattr(cfg, "object_hidden", 128)),
            n_future_steps=int(getattr(cfg, "n_future_steps", 12)),
            n_object_types=int(getattr(cfg, "n_object_types", 4)),
        )


def build_2d_sinusoidal_position_embedding(n_h: int, n_w: int, d_model: int) -> torch.Tensor:
    """
    Build 2D sinusoidal position embeddings for BEV patches.
    
    Args:
        n_h: Number of patches in height dimension
        n_w: Number of patches in width dimension
        d_model: Model dimension
        
    Returns:
        pos_embed: (n_h * n_w, d_model) position embeddings
    """
    # Create position indices
    h_pos = torch.arange(n_h).float()
    w_pos = torch.arange(n_w).float()
    
    # Create 2D grid
    h_grid, w_grid = torch.meshgrid(h_pos, w_pos, indexing='ij')
    h_grid = h_grid.flatten()  # (n_h * n_w,)
    w_grid = w_grid.flatten()  # (n_h * n_w,)
    
    # Compute position embeddings
    # Split d_model into 4 parts: [sin(h), cos(h), sin(w), cos(w)]
    d_per_dim = d_model // 4
    
    # Frequency bands
    div_term = torch.exp(torch.arange(0, d_per_dim, 2).float() * (-math.log(10000.0) / d_per_dim))
    
    pos_embed = torch.zeros(n_h * n_w, d_model)
    
    # Height embeddings
    pos_embed[:, 0:d_per_dim:2] = torch.sin(h_grid.unsqueeze(1) * div_term)
    pos_embed[:, 1:d_per_dim:2] = torch.cos(h_grid.unsqueeze(1) * div_term)
    
    # Width embeddings
    pos_embed[:, d_per_dim:2*d_per_dim:2] = torch.sin(w_grid.unsqueeze(1) * div_term)
    pos_embed[:, d_per_dim+1:2*d_per_dim:2] = torch.cos(w_grid.unsqueeze(1) * div_term)
    
    # Remaining dimensions (if d_model not divisible by 4, fill with zeros or repeat)
    # Already initialized to zeros, so we're good
    
    return pos_embed


class EgoEncoder(nn.Module):
    """Encode ego state vector to tokens."""
    
    def __init__(self, d_ego: int, d_hidden: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_ego, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_model),
        )
    
    def forward(self, ego_vec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ego_vec: (B, d_ego)
        Returns:
            ego_tokens: (B, 1, d_model)
        """
        tokens = self.mlp(ego_vec)  # (B, d_model)
        return tokens.unsqueeze(1)  # (B, 1, d_model)


class RouteEncoder(nn.Module):
    """Encode route waypoints to tokens."""
    
    def __init__(self, d_hidden: int, d_model: int, max_route_points: int = 64, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_model),
        )
        # Learnable positional encoding for route points
        self.pos_embed = nn.Parameter(torch.randn(1, max_route_points, d_model) * 0.02)
    
    def forward(self, route: torch.Tensor) -> torch.Tensor:
        """
        Args:
            route: (B, K, 2) route waypoints
        Returns:
            route_tokens: (B, K, d_model)
        """
        B, K, _ = route.shape
        tokens = self.mlp(route)  # (B, K, d_model)
        tokens = tokens + self.pos_embed[:, :K, :]  # Add positional encoding
        return tokens


class ObjectEncoder(nn.Module):
    """Encode object tokens with type embeddings."""
    
    def __init__(self, d_obj: int, d_hidden: int, d_model: int, n_object_types: int = 4, dropout: float = 0.1):
        super().__init__()
        # Type embedding (type_id is first feature)
        self.type_embed = nn.Embedding(n_object_types, max(d_model // 2, 32))
        
        # MLP for continuous features (remaining d_obj - 1 features)
        self.mlp = nn.Sequential(
            nn.Linear(d_obj - 1 + max(d_model // 2, 32), d_hidden),
            nn.LayerNorm(d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_model),
        )
    
    def forward(self, objects: torch.Tensor) -> torch.Tensor:
        """
        Args:
            objects: (B, M, d_obj) where first feature is type_id
        Returns:
            object_tokens: (B, M, d_model)
        """
        B, M, _ = objects.shape
        
        # Extract type_id and continuous features
        type_id = objects[:, :, 0].long()  # (B, M)
        continuous_features = objects[:, :, 1:]  # (B, M, d_obj-1)
        
        # Embed type
        type_emb = self.type_embed(type_id)  # (B, M, d_model//4)
        
        # Concatenate and encode
        combined = torch.cat([type_emb, continuous_features], dim=-1)  # (B, M, d_model//4 + d_obj-1)
        tokens = self.mlp(combined)  # (B, M, d_model)
        
        return tokens


class BEVViTEncoder(nn.Module):
    """ViT-style patch encoder for BEV."""
    
    def __init__(self, in_channels: int, d_model: int, patch_size: list, dropout: float = 0.1):
        super().__init__()
        self.patch_size = patch_size
        self.patch_h, self.patch_w = patch_size
        
        # Patch embedding via convolution
        self.patch_embed = nn.Conv2d(
            in_channels, d_model, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Position embeddings will be created dynamically based on input size
        self.register_buffer("pos_embed", None, persistent=False)
    
    def forward(self, bev: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bev: (B, C, H, W) BEV tensor
        Returns:
            bev_tokens: (B, n_patches, d_model)
        """
        B, C, H, W = bev.shape
        
        # Patch embedding
        patches = self.patch_embed(bev)  # (B, d_model, n_h, n_w)
        _, d_model, n_h, n_w = patches.shape
        
        # Flatten patches
        patches = patches.flatten(2).transpose(1, 2)  # (B, n_patches, d_model)
        
        # Add positional embeddings
        if self.pos_embed is None or self.pos_embed.shape[1] != patches.shape[1]:
            pos_embed = build_2d_sinusoidal_position_embedding(n_h, n_w, d_model)
            self.pos_embed = pos_embed.to(patches.device)
        
        patches = patches + self.pos_embed.unsqueeze(0)
        
        # Normalize and dropout
        patches = self.norm(patches)
        patches = self.dropout(patches)
        
        return patches


class BCPolicy(nn.Module):
    """
    Lightweight Query-Based Behavior Cloning Policy.
    
    Architecture:
    1. Input Encoders: Encode ego, route, objects, BEV to token sequences
    2. Modality Embeddings: Add type embeddings to distinguish modalities
    3. Transformer Encoder: Self-attention across all tokens with masking
    4. Query Decoder: Cross-attention with learnable future queries
    5. Output Heads: Predict waypoints and speeds from query outputs
    """
    
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        dropout: float = 0.1,
        patch_size: list = None,
        ego_hidden: int = 128,
        route_hidden: int = 128,
        object_hidden: int = 128,
        n_future_steps: int = 12,
        n_object_types: int = 4,
    ):
        """
        Initialize BC policy.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer encoder/decoder layers
            dropout: Dropout rate
            patch_size: [H, W] patch size for BEV ViT
            ego_hidden: Hidden dim for ego MLP
            route_hidden: Hidden dim for route MLP
            object_hidden: Hidden dim for object MLP
            n_future_steps: Number of future waypoints to predict
            n_object_types: Number of object types for embedding
        """
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.patch_size = patch_size or [8, 8]
        self.n_future_steps = n_future_steps
        
        # Input encoders
        self.ego_encoder = EgoEncoder(d_ego=14, d_hidden=ego_hidden, d_model=d_model, dropout=dropout)
        self.route_encoder = RouteEncoder(d_hidden=route_hidden, d_model=d_model, dropout=dropout)
        self.object_encoder = ObjectEncoder(d_obj=11, d_hidden=object_hidden, d_model=d_model, 
                                           n_object_types=n_object_types, dropout=dropout)
        self.bev_encoder = BEVViTEncoder(in_channels=18, d_model=d_model, 
                                         patch_size=self.patch_size, dropout=dropout)
        
        # Modality type embeddings
        self.modality_embed = nn.Parameter(torch.randn(4, d_model) * 0.02)  # ego, route, object, bev
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Learnable future query tokens
        self.future_queries = nn.Parameter(torch.randn(n_future_steps, d_model) * 0.02)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        
        # Output heads
        self.waypoint_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2)
        )
        
        self.speed_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        # Initialize weights
        self._init_weights()
        
        # Count parameters
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[BCPolicy] Lightweight Query-Based Planning Architecture")
        print(f"  d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}")
        print(f"  n_future_steps={n_future_steps}, patch_size={self.patch_size}")
        print(f"  Total parameters: {n_params:,} ({n_params/1e6:.2f}M)")
    
    def _init_weights(self):
        """Initialize weights with transformer-appropriate scaling for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use smaller gain for transformer layers (better stability)
                nn.init.xavier_uniform_(module.weight, gain=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                # Conv layers (BEV patch embedding)
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                # Embedding layers (object types)
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
        # Initialize learnable parameters (queries, modality embeddings)
        if hasattr(self, 'future_queries'):
            nn.init.normal_(self.future_queries, mean=0.0, std=0.02)
        if hasattr(self, 'modality_embed'):
            nn.init.normal_(self.modality_embed, mean=0.0, std=0.02)
    
    def forward(self, batch):
        """
        Forward pass.
        
        Args:
            batch: Dict with keys:
                - ego_vec: (B, 14) ego state
                - bev: (B, 18, 150, 200) BEV tensor
                - route: (B, K, 2) route waypoints
                - objects: (B, M, 11) object tokens
                - object_mask: (B, M) attention mask for objects
                
        Returns:
            Dict with keys:
                - future_xy: (B, N, 2) predicted waypoints (normalized)
                - future_v: (B, N) predicted speeds (normalized)
        """
        ego_vec = batch["ego_vec"]
        bev = batch["bev"]
        route = batch["route"]
        route_mask_in = batch.get("route_mask", None)
        objects = batch["objects"]
        object_mask = batch["object_mask"]
        
        B = ego_vec.shape[0]
        K = route.shape[1]
        M = objects.shape[1]
        
        # 1. Encode inputs to tokens
        ego_tokens = self.ego_encoder(ego_vec)  # (B, 1, d_model)
        route_tokens = self.route_encoder(route)  # (B, K, d_model)
        object_tokens = self.object_encoder(objects)  # (B, M, d_model)
        bev_tokens = self.bev_encoder(bev)  # (B, n_patches, d_model)
        
        n_patches = bev_tokens.shape[1]
        
        # 2. Add modality type embeddings
        ego_tokens = ego_tokens + self.modality_embed[0:1, :].unsqueeze(0)
        route_tokens = route_tokens + self.modality_embed[1:2, :].unsqueeze(0)
        object_tokens = object_tokens + self.modality_embed[2:3, :].unsqueeze(0)
        bev_tokens = bev_tokens + self.modality_embed[3:4, :].unsqueeze(0)
        
        # 3. Concatenate all tokens
        tokens = torch.cat([ego_tokens, route_tokens, object_tokens, bev_tokens], dim=1)
        # Shape: (B, 1 + K + M + n_patches, d_model)
        
        # 4. Create attention mask for padded tokens
        # PyTorch transformer expects: True = ignore, False = attend
        ego_mask = torch.zeros(B, 1, dtype=torch.bool, device=ego_vec.device)
        if route_mask_in is not None:
            # route_mask is (B, K) with 1 for valid; flip to ignore padded
            route_mask = (route_mask_in == 0).to(dtype=torch.bool, device=ego_vec.device)
        else:
            route_mask = torch.zeros(B, K, dtype=torch.bool, device=ego_vec.device)
        object_attn_mask = (object_mask == 0)  # Flip: 0 = padded, 1 = valid -> True = ignore
        bev_mask = torch.zeros(B, n_patches, dtype=torch.bool, device=ego_vec.device)
        
        src_key_padding_mask = torch.cat([ego_mask, route_mask, object_attn_mask, bev_mask], dim=1)
        # Shape: (B, seq_len)
        
        # 5. Apply transformer encoder
        encoded = self.encoder(tokens, src_key_padding_mask=src_key_padding_mask)
        # Shape: (B, seq_len, d_model)
        
        # 6. Expand future queries to batch size
        queries = self.future_queries.unsqueeze(0).expand(B, -1, -1)  # (B, N, d_model)
        
        # 7. Apply transformer decoder (queries cross-attend to encoded context)
        decoded = self.decoder(
            tgt=queries,
            memory=encoded,
            memory_key_padding_mask=src_key_padding_mask
        )
        # Shape: (B, N, d_model)
        
        # 8. Apply output heads (no direct residual to keep outputs in normalized space)
        future_xy = self.waypoint_head(decoded)  # (B, N, 2)
        future_v = self.speed_head(decoded).squeeze(-1)  # (B, N)
        
        return {
            "future_xy": future_xy,
            "future_v": future_v,
        }

