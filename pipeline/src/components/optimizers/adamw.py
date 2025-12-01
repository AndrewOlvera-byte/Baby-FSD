from __future__ import annotations
import torch
from src.core.registry import register

@register("optimizer", "adamw")
class AdamWFactory:
    def build(self, cfg_node, context):
        model = context["model"]
        lr = float(cfg_node.lr)
        weight_decay = float(cfg_node.weight_decay)
        fused = bool(getattr(cfg_node, "fused", False))
        
        # Support custom betas for transformer training
        # Default betas=[0.9, 0.999], but transformers often benefit from lower beta2
        betas = getattr(cfg_node, "betas", [0.9, 0.999])
        if isinstance(betas, (list, tuple)) and len(betas) == 2:
            betas = tuple(float(b) for b in betas)
        else:
            betas = (0.9, 0.999)
        
        # Support custom eps
        eps = float(getattr(cfg_node, "eps", 1e-8))
        
        opt_kwargs = dict(
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
        )
        # fused AdamW only on CUDA builds that support it
        if fused and torch.cuda.is_available():
            opt_kwargs["fused"] = True
        opt = torch.optim.AdamW(model.parameters(), **opt_kwargs)
        return opt
