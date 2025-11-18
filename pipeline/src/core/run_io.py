from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from omegaconf import OmegaConf


def get_run_dir() -> Path:
    try:
        from hydra.core.hydra_config import HydraConfig
        hc = HydraConfig.get()
        run_dir = getattr(getattr(hc, "runtime", None), "output_dir", None)
        if run_dir:
            return Path(run_dir)
    except Exception:
        pass
    return Path.cwd()


def dump_full_config(cfg, outdir: Path | None = None) -> Dict[str, str]:
    outdir = Path(outdir) if outdir is not None else (get_run_dir() / "reports")
    outdir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}

    cfg_container = OmegaConf.to_container(cfg, resolve=True)
    cfg_path = outdir / "config_full.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_container, f, sort_keys=False, default_flow_style=False, allow_unicode=True)
    paths["config_full"] = str(cfg_path)

    # Write overrides used
    try:
        from hydra.core.hydra_config import HydraConfig
        hc = HydraConfig.get()
        overrides: List[str] = []
        try:
            overrides = list(getattr(getattr(hc, "overrides", {}), "task", []))
        except Exception:
            overrides = []
        over_path = outdir / "overrides.txt"
        with open(over_path, "w", encoding="utf-8") as f:
            for line in overrides:
                f.write(str(line) + "\n")
        paths["overrides"] = str(over_path)
    except Exception:
        # Hydra not initialized
        pass

    return paths


def write_summary(outdir: Path, data: Dict[str, Any], filename: str = "summary.json") -> str:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return str(path)


def update_topk_if_sweep(cfg, objective_value: float, addl: Optional[Dict[str, Any]] = None, k_default: int = 5) -> Optional[str]:
    """If running under a Hydra sweep, append this trial result and recompute top-k file.

    Returns path to the top-k json if updated, else None.
    """
    try:
        from hydra.core.hydra_config import HydraConfig
        hc = HydraConfig.get()
        sweep_dir = Path(getattr(getattr(hc, "sweep", None), "dir", "."))
        if not str(sweep_dir):
            return None
    except Exception:
        return None

    sweep_dir.mkdir(parents=True, exist_ok=True)
    results_path = sweep_dir / "results.jsonl"
    record: Dict[str, Any] = {
        "objective": float(objective_value),
        "run_dir": str(Path.cwd()),
    }
    if addl:
        record.update(addl)

    # Append current result
    with open(results_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    # Read all and compute top-k
    rows: List[Dict[str, Any]] = []
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue

    # Determine direction and top-k
    direction = str(getattr(getattr(cfg, "hpo", {}), "direction", "maximize")).lower()
    k_val = int(getattr(getattr(cfg, "hpo", {}), "top_k", k_default))
    reverse = True if direction == "maximize" else False
    rows_sorted = sorted(rows, key=lambda r: float(r.get("objective", float("nan"))), reverse=reverse)
    topk = rows_sorted[:k_val]

    topk_path = sweep_dir / "topk.json"
    with open(topk_path, "w", encoding="utf-8") as f:
        json.dump({"direction": direction, "top_k": k_val, "results": topk}, f, indent=2)

    return str(topk_path)


