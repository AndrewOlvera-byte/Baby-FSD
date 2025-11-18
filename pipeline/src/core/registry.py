from __future__ import annotations
from typing import Any, Callable, Dict

REG: Dict[str, Dict[str, Any]] = {
    k: {} for k in [
        "env", "dataset", "datacoll", "model", "loss", "collector",
        "optimizer", "evaluator", "logger", "algo", "trainer"
    ]
}

def register(kind: str, name: str):
    """
    Usage:
        @register("loss", "ppo")
        class PPOLossFactory:
            def build(self, cfg_node, context): ...
    """
    def deco(obj: Any) -> Any:
        if kind not in REG:
            raise KeyError(f"Unknown registry kind '{kind}'")
        if name in REG[kind]:
            raise KeyError(f"Duplicate registration: kind='{kind}', name='{name}'")
        REG[kind][name] = obj
        return obj
    return deco

def get(kind: str, name: str) -> Any:
    return REG[kind][name]

def available(kind: str) -> Dict[str, Any]:
    return dict(REG.get(kind, {}))

def available(kind: str) -> Dict[str, Any]:
    return dict(REG.get(kind, {}))