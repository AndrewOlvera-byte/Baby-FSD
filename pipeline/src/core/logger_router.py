from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any
from src.core.run_io import get_run_dir

class BaseLogger:
    def log(self, metrics: Dict[str, Any]): ...
    def artifact(self, path: str, name: str | None = None): ...
    def finish(self): ...
    def should_save(self) -> bool: return True

class OfflineLogger(BaseLogger):
    """Simple JSONL logger in outputs dir."""
    def __init__(self, cfg):
        # Respect Hydra's run directory. Always write under <run_dir>/reports.
        outdir = get_run_dir() / "reports"
        outdir.mkdir(parents=True, exist_ok=True)
        self._path = outdir / "metrics.jsonl"
        self._fh = open(self._path, "a", buffering=1)
        # Respect mode from flat (mode) or nested (modes.mode)
        try:
            policy = getattr(cfg, "mode")
        except Exception:
            policy = None
        if not policy:
            try:
                policy = getattr(getattr(cfg, "modes", {}), "mode", None)
            except Exception:
                policy = None
        self._policy = policy or "train"

        # store a minimal config snapshot for reproducibility
        with open(outdir / "config_snapshot.json", "w") as f:
            try:
                json.dump(dict(cfg), f, indent=2, default=str)
            except Exception:
                f.write("{}")

    def log(self, metrics: Dict[str, Any]):
        self._fh.write(json.dumps(metrics) + "\n")

    def artifact(self, path: str, name: str | None = None):
        # Best-effort copy into reports/ (namespaced) for ease of discovery
        try:
            src = Path(path)
            if not src.exists():
                return
            outdir = Path(self._path).parent
            dst = outdir / (name or src.name)
            if src.resolve() != dst.resolve():
                import shutil
                shutil.copyfile(src, dst)
        except Exception:
            return

    def finish(self):
        self._fh.close()

    def should_save(self) -> bool:
        return self._policy == "train"

def make_logger(cfg) -> BaseLogger:
    # Support flat and nested logger configs
    def _maybe_get(obj, name):
        try:
            return getattr(obj, name)
        except Exception:
            return None
    logger_node = _maybe_get(cfg, "logger") or _maybe_get(_maybe_get(cfg, "exp"), "logger") or {}
    name = getattr(logger_node, "name", None) or getattr(cfg, "mode", None) or "offline"
    if name == "offline" or name == "csv":
        return OfflineLogger(cfg)
    # default: no-op logger
    return BaseLogger()
