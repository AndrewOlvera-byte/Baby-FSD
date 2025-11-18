from __future__ import annotations
from typing import Any, Dict
import torch
from torch.utils.data import DataLoader
from src.core.registry import register


@register("evaluator", "sl_classification")
class SLClassificationEvaluator:
    def build(self, cfg_node, context: Dict[str, Any]):
        ds = context.get("dataset", {}).get("valid", None)
        if ds is None:
            return lambda model: {}
        bs = int(getattr(getattr(context.get("cfg", {}), "trainer", {}), "batch_size", 256))
        dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)

        @torch.no_grad()
        def run(model):
            model.eval()
            correct, total = 0, 0
            for batch in dl:
                batch = {k: (v.cuda(non_blocking=True) if torch.is_tensor(v) and torch.cuda.is_available() else v)
                         for k, v in batch.items()}
                logits = model(batch["images"])
                preds = logits.argmax(dim=1)
                correct += (preds == batch["targets"]).sum().item()
                total += batch["targets"].numel()
            model.train()
            return {"valid/acc": correct / max(1, total)}

        return run


