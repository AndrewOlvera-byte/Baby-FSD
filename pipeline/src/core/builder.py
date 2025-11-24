from __future__ import annotations
from typing import Any, Dict, List
from omegaconf import OmegaConf
from .registry import REG, get
from .trainer_base import BaseTrainer

def _collect_cfg_nodes(cfg: Any, kind: str) -> List[Any]:
    """
    Collect possible config nodes for a given kind (e.g., "trainer", "dataset").
    Includes both flattened and nested experiment shapes so we can merge them
    with the appropriate precedence (experiment overrides base).
    """
    nodes: List[Any] = []

    # Base, flat and nested (e.g., cfg.trainer and cfg.trainer.trainer)
    base = getattr(cfg, kind, None)
    if base is not None:
        nodes.append(base)
        inner_base = getattr(base, kind, None)
        if inner_base is not None:
            nodes.append(inner_base)

    # Experiment-level
    exp = getattr(cfg, "exp", None)
    if exp is not None:
        exp_node = getattr(exp, kind, None)
        if exp_node is not None:
            nodes.append(exp_node)
            inner_exp_node = getattr(exp_node, kind, None)
            if inner_exp_node is not None:
                nodes.append(inner_exp_node)

        # Handle an extra exp.exp nesting if present
        exp2 = getattr(exp, "exp", None)
        if exp2 is not None:
            exp2_node = getattr(exp2, kind, None)
            if exp2_node is not None:
                nodes.append(exp2_node)
                inner_exp2_node = getattr(exp2_node, kind, None)
                if inner_exp2_node is not None:
                    nodes.append(inner_exp2_node)

    return nodes


def _merge_cfg_nodes(cfg: Any, kind: str) -> Any:
    """
    Merge collected nodes for a kind, with later entries taking precedence.
    Precedence order: base (flat → nested) < exp (flat → nested) < exp.exp (flat → nested).
    Returns None if nothing found.
    """
    # Only consider leaf nodes that actually define a component (must have 'name')
    nodes = [n for n in _collect_cfg_nodes(cfg, kind) if n is not None and getattr(n, "name", None) is not None]
    if not nodes:
        return None
    merged = nodes[0]
    for n in nodes[1:]:
        merged = OmegaConf.merge(merged, n)
    return merged

def build_component(kind: str, cfg: Any, context: Dict[str, Any]) -> Any:
    node = _merge_cfg_nodes(cfg, kind)
    if node is None or getattr(node, "name", None) is None:
        return None
    factory = get(kind, node.name)
    if hasattr(factory, "build"):
        return factory().build(node, context)
    return factory(node, context)

def build_trainer(cfg: Any) -> BaseTrainer:
    trainer_node = _merge_cfg_nodes(cfg, "trainer")
    if trainer_node is None or getattr(trainer_node, "name", None) is None:
        raise KeyError("Trainer config not found. Expected 'trainer.name' in the composed config.")
    trainer_name = trainer_node.name
    TrainerCls = get("trainer", trainer_name)

    required = getattr(TrainerCls, "required_components", [])
    context: Dict[str, Any] = {"cfg": cfg}

    for kind in required:
        obj = build_component(kind, cfg, context)
        if obj is not None:
            context[kind] = obj

    kwargs = {k: v for k, v in context.items() if k in required}
    trainer = TrainerCls(cfg, **kwargs)
    return trainer


def build_evaluator(cfg: Any, context: Dict[str, Any] | None = None):
    """
    Build evaluator callable from registry using cfg.evaluator.name if present.
    Returns None if no evaluator configured.
    """
    node = _merge_cfg_nodes(cfg, "evaluator")
    if node is None or getattr(node, "name", None) is None:
        return None
    factory = get("evaluator", node.name)
    ctx: Dict[str, Any] = {"cfg": cfg}
    # optionally pass dataset/model if already built by caller
    if context:
        ctx.update(context)
    elif hasattr(cfg, "_components_context") and isinstance(cfg._components_context, dict):
        ctx.update(cfg._components_context)
    if hasattr(factory, "build"):
        return factory().build(node, ctx)
    return factory(node, ctx)
