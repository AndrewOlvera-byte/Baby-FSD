from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch


@dataclass
class _TopKEntry:
    path: str
    objective: float
    step: int


class CheckpointManager:
    """
    Manages saving checkpoints as plain .pt files and tracking top-k by an objective value.

    Layout (relative to provided outdir):
      - checkpoints/step_{n}.pt
      - topk.json (summary of best K checkpoints)
    """

    def __init__(self, outdir: Path, top_k: int = 5, direction: str = "min"):
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = self.outdir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.topk_path = self.outdir / "topk.json"
        self.top_k = int(top_k)
        direction = str(direction or "min").lower()
        self.reverse = True if direction == "max" else False
        # in-memory cache of topk
        self._topk: List[_TopKEntry] = self._load_topk()

    def _load_topk(self) -> List[_TopKEntry]:
        if not self.topk_path.exists():
            return []
        try:
            with open(self.topk_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            rows = data.get("results", [])
            out: List[_TopKEntry] = []
            for r in rows:
                out.append(_TopKEntry(path=r.get("path", ""), objective=float(r.get("objective", 0.0)), step=int(r.get("step", 0))))
            return out
        except Exception:
            return []

    def _save_topk(self):
        rows = [
            {"path": e.path, "objective": float(e.objective), "step": int(e.step)}
            for e in self._topk
        ]
        payload = {
            "direction": "maximize" if self.reverse else "minimize",
            "top_k": int(self.top_k),
            "results": rows,
        }
        with open(self.topk_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _update_topk(self, path: Path, objective: Optional[float], step: int):
        if objective is None:
            return
        self._topk.append(_TopKEntry(path=str(path), objective=float(objective), step=int(step)))
        # sort and trim
        self._topk.sort(key=lambda e: float(e.objective), reverse=self.reverse)
        if len(self._topk) > self.top_k:
            self._topk = self._topk[: self.top_k]
        self._save_topk()

    def save_step(self, step: int, state: Dict, objective: Optional[float] = None) -> str:
        """
        Save a checkpoint for a given step. `state` should generally be a model.state_dict().
        Returns the file path as string.
        """
        path = self.ckpt_dir / f"step_{int(step)}.pt"
        torch.save(state, path)
        self._update_topk(path, objective, step)
        # Also maintain a convenience symlink/copy for best.pt if the new one is the best
        try:
            if self._topk and str(path) == self._topk[0].path:
                best_path = self.ckpt_dir.parent / "best.pt"
                # Copy to best.pt
                torch.save(state, best_path)
        except Exception:
            pass
        return str(path)

    def save_epoch(self, epoch: int, state: Dict, objective: Optional[float] = None) -> str:
        path = self.ckpt_dir / f"epoch_{int(epoch)}.pt"
        torch.save(state, path)
        self._update_topk(path, objective, epoch)
        return str(path)


