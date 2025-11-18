import hydra
from omegaconf import DictConfig
from src.core.bootstrap import bootstrap
from src.core.builder import build_trainer
from src.core.run_io import write_summary, update_topk_if_sweep, dump_full_config, get_run_dir
from pathlib import Path


@hydra.main(config_path="../config", config_name="defaults", version_base=None)
def main(cfg: DictConfig) -> float:
    # auto-register all trainers/components
    bootstrap()
    # build and run
    trainer = build_trainer(cfg)
    # dump full config and overrides for this run into reports/
    run_dir = get_run_dir()
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    dump_full_config(cfg, reports_dir)
    # ask trainers to return final metric if possible; else evaluate once
    result = trainer.fit()  # may return a float (objective)
    if isinstance(result, (int, float)):
        # write per-run summary for search mode
        write_summary(reports_dir, {
            "exp_name": getattr(cfg.exp, "name", ""),
            "mode": getattr(cfg, "mode", ""),
            "objective": float(result),
        }, filename="summary.json")
        update_topk_if_sweep(cfg, float(result))
        return float(result)

    # Fallback: compute metric via evaluator
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
    # If no metric found, return 0.0 so the trial doesn't crash
    return 0.0


if __name__ == "__main__":
    main()
