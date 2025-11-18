import hydra
from omegaconf import DictConfig
from pathlib import Path
from src.core.bootstrap import bootstrap
from src.core.builder import build_trainer
from src.core.run_io import dump_full_config

@hydra.main(config_path="../config", config_name="defaults", version_base=None)
def main(cfg: DictConfig):
    # auto-register all trainers/components
    bootstrap()
    # dump composed config for debugging overrides (before building)
    dump_full_config(cfg)
    # optional: allow config-only dump without running training
    if bool(getattr(cfg, "only_dump", False)):
        return
    # build the requested trainer from the registry and run
    trainer = build_trainer(cfg)
    trainer.fit()

if __name__ == "__main__":
    main()
