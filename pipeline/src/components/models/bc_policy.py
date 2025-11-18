"""
Behavior Cloning Policy Model (Scaffold).

Architecture (TO BE IMPLEMENTED):
- Ego MLP: ego_vec -> tokens
- Route MLP: route points -> tokens
- Object MLP: object_tokens -> tokens (with type embeddings)
- BEV ViT: patch embedding (C,H,W) -> tokens with 2D positional encoding
- Encoder: Transformer encoder stack
- Decoder: Query-based decoder with learnable future queries
- Output heads: (x, y, v) predictions per future step
"""

from __future__ import annotations
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


class BCPolicy(nn.Module):
    """
    Behavior Cloning Policy Network.
    
    TODO: IMPLEMENT ARCHITECTURE
    
    Architecture Overview:
    ----------------------
    
    Input Processing:
    1. Ego MLP: ego_vec (d_ego,) -> ego tokens
    2. Route MLP: route points (K, 2) -> route tokens  
    3. Object MLP: object_tokens (M, d_obj) -> object tokens
       - Use nn.Embedding for type_id
       - Concat with continuous features
    4. BEV ViT: (C, H, W) -> BEV tokens
       - Patch embedding (like ViT)
       - 2D sinusoidal positional encoding
    
    Sequence Construction:
    [ego_tokens, route_tokens, object_tokens, bev_tokens]
    - Add type embeddings for each modality
    - Apply attention masking for padded objects
    
    Encoder:
    - Transformer encoder stack (n_layers)
    - Self-attention across all tokens
    - Respect attention masks
    
    Decoder:
    - N learnable query tokens (one per future step)
    - Cross-attend to encoder output
    - Output heads: separate for (x, y) and v
    
    Output:
    - future_xy: (B, N, 2)
    - future_v: (B, N)
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
            n_layers: Number of transformer layers
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
        
        # TODO: Implement input encoders
        # self.ego_encoder = ...
        # self.route_encoder = ...
        # self.object_encoder = ...
        # self.bev_encoder = ...
        
        # TODO: Implement transformer encoder
        # self.encoder = ...
        
        # TODO: Implement decoder with learnable queries
        # self.future_queries = nn.Parameter(...)
        # self.decoder = ...
        
        # TODO: Implement output heads
        # self.waypoint_head = ...
        # self.speed_head = ...
        
        print(f"[BCPolicy] Architecture scaffold created (NOT IMPLEMENTED)")
        print(f"  d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}")
        print(f"  n_future_steps={n_future_steps}")
        print(f"  TODO: Implement forward pass")
    
    def forward(self, batch):
        """
        Forward pass (NOT IMPLEMENTED).
        
        Args:
            batch: Dict with keys:
                - ego_vec: (B, d_ego)
                - bev: (B, C, H, W)
                - route: (B, K, 2)
                - objects: (B, M, d_obj)
                - object_mask: (B, M)
                
        Returns:
            Dict with keys:
                - future_xy: (B, N, 2) predicted waypoints
                - future_v: (B, N) predicted speeds
        """
        # TODO: Implement architecture
        # 
        # 1. Encode inputs to tokens:
        #    - ego_tokens = self.ego_encoder(batch["ego_vec"])
        #    - route_tokens = self.route_encoder(batch["route"])
        #    - object_tokens = self.object_encoder(batch["objects"])
        #    - bev_tokens = self.bev_encoder(batch["bev"])
        #
        # 2. Concatenate sequence with type embeddings:
        #    - tokens = [ego_tokens, route_tokens, object_tokens, bev_tokens]
        #
        # 3. Apply encoder:
        #    - encoded = self.encoder(tokens, attention_mask)
        #
        # 4. Apply decoder with queries:
        #    - decoded = self.decoder(self.future_queries, encoded)
        #
        # 5. Predict outputs:
        #    - future_xy = self.waypoint_head(decoded)
        #    - future_v = self.speed_head(decoded)
        
        raise NotImplementedError(
            "BCPolicy architecture implementation pending. "
            "Please implement the forward pass with ViT BEV encoder, "
            "MLPs for ego/route/objects, transformer encoder/decoder, "
            "and output heads for waypoint and speed prediction."
        )

