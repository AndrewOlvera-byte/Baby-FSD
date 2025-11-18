from __future__ import annotations
import torch
from src.core.registry import register

@register("optimizer", "adam")
class AdamFactory:
    def build(self, cfg_node, context):
        model = context["model"]
        opt = torch.optim.Adam(model.parameters(), lr=cfg_node.lr, weight_decay=cfg_node.weight_decay)
        return opt
