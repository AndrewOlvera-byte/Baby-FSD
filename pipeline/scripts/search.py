import hydra
from omegaconf import DictConfig
from src.core.bootstrap import bootstrap
from src.core.builder import build_trainer
from src.core.run_io import write_summary, update_topk_if_sweep, dump_full_config, get_run_dir
from pathlib import Path


@hydra.main(config_path="../config", config_name="defaults", version_base=None)
def main(cfg: DictConfig) -> float:
    bootstrap()
    trainer = build_trainer(cfg)
    run_dir = get_run_dir()
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    dump_full_config(cfg, reports_dir)
    result = trainer.fit()
    if isinstance(result, (int, float)):
        write_summary(reports_dir, {
            "exp_name": getattr(cfg.exp, "name", ""),
            "mode": getattr(cfg, "mode", ""),
            "objective": float(result),
        }, filename="summary.json")
        update_topk_if_sweep(cfg, float(result))
        return float(result)

    evaluator = trainer.components.get("evaluator", None)
    model = trainer.components.get("model", None)
    objective_key = getattr(getattr(cfg, "hpo", {}), "objective_key", "") or ""
    if evaluator and model and objective_key:
        metrics = evaluator(model) or {}
        if objective_key in metrics:
            val = float(metrics[objective_key])
            write_summary(reports_dir, {
                "exp_name": getattr(cfg.exp, "name", ""),
                "mode": getattr(cfg, "mode", ""),
                "objective_key": objective_key,
                "objective": val,
            }, filename="summary.json")
            update_topk_if_sweep(cfg, val)
            return val
    return 0.0


if __name__ == "__main__":
    main()
