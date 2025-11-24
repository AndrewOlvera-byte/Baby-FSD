import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # force CPU for eval runs

import hydra
from omegaconf import DictConfig
from pathlib import Path
import sys
from pathlib import Path

# Ensure repo root and pipeline/ are on sys.path for module resolution when run from project root
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "pipeline") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "pipeline"))

from src.core.bootstrap import bootstrap
from src.core.builder import build_trainer, build_evaluator
from src.core.run_io import dump_full_config, write_summary, get_run_dir


@hydra.main(config_path="../config", config_name="defaults", version_base=None)
def main(cfg: DictConfig) -> None:
    bootstrap()
    # Build trainer to get components (model, evaluator built via trainer.required_components)
    trainer = build_trainer(cfg)
    model = trainer.components.get("model", None)
    # Build evaluator from cfg if not provided by trainer
    evaluator = build_evaluator(cfg, context={"dataset": trainer.components.get("dataset", None), "model": model})

    run_dir = get_run_dir()
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    dump_full_config(cfg, reports_dir)

    metrics = {}
    if evaluator and model:
        metrics = evaluator(model) or {}
    write_summary(reports_dir, metrics, filename="summary.json")


if __name__ == "__main__":
    main()


