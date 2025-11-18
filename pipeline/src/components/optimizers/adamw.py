from __future__ import annotations
import torch
from src.core.registry import register

@register("optimizer", "adamw")
class AdamWFactory:
    def build(self, cfg_node, context):
        model = context["model"]
        opt = torch.optim.AdamW(model.parameters(), lr=cfg_node.lr, weight_decay=cfg_node.weight_decay)
        return opt
