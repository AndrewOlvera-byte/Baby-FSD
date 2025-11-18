from __future__ import annotations
from typing import Any, Dict
import torch
from torch import nn
from torchvision.models import resnet50
from src.core.registry import register


@register("model", "resnet50_tv")
class ResNet50TorchVisionFactory:
    def build(self, cfg_node, context: Dict[str, Any]):
        num_classes = int(getattr(cfg_node, "num_classes", 10))
        drop = float(getattr(cfg_node, "dropout", 0.0))
        # torchvision's resnet50 does not accept dropout param directly; keep for future use
        model: nn.Module = resnet50(weights=None)
        # adjust classifier
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model


