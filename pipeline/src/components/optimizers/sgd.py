from __future__ import annotations
import torch
from src.core.registry import register


@register("optimizer", "sgd")
class SGDFactory:
    def build(self, cfg_node, context):
        model = context["model"]
        lr = float(getattr(cfg_node, "lr", 0.1))
        momentum = float(getattr(cfg_node, "momentum", 0.9))
        weight_decay = float(getattr(cfg_node, "weight_decay", 5e-4))
        nesterov = bool(getattr(cfg_node, "nesterov", True))
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                               weight_decay=weight_decay, nesterov=nesterov)


