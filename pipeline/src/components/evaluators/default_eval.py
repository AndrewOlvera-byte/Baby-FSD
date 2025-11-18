from __future__ import annotations
import torch
from torch.utils.data import DataLoader
from src.core.registry import register

@register("evaluator", "default_eval")
class DefaultEvaluatorFactory:
    def build(self, cfg_node, context):
        def _maybe_get(obj, name):
            try:
                return getattr(obj, name)
            except Exception:
                return None
        def _get_dataset_cfg(cfg):
            ds = _maybe_get(cfg, "dataset")
            if ds is not None:
                return ds
            exp = _maybe_get(cfg, "exp")
            if exp is not None:
                ds = _maybe_get(exp, "dataset")
                if ds is not None:
                    return ds
            return None

        valid_ds = context["dataset"]["valid"]
        cfg = context["cfg"]
        ds_cfg = _get_dataset_cfg(cfg)
        bs = int(getattr(ds_cfg, "batch_size", 32)) if ds_cfg is not None else 32
        dl = DataLoader(valid_ds, batch_size=bs, shuffle=False, num_workers=0)
        @torch.no_grad()
        def run(model):
            model.eval()
            total_nll, total_n = 0.0, 0
            for batch in dl:
                batch = {k: (v.cuda(non_blocking=True) if v.__class__.__name__=="Tensor" and torch.cuda.is_available() else v)
                         for k, v in batch.items()}
                dist = model.policy(batch["obs"])
                nll = -dist.log_prob(batch["act"])  # (B,)
                total_nll += nll.sum().item()
                total_n += nll.numel()
            model.train()
            return {"eval/nll_mean": total_nll / max(1,total_n)}
        return run
