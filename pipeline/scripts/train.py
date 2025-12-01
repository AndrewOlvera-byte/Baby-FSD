import hydra
from omegaconf import DictConfig
from pathlib import Path
from src.core.bootstrap import bootstrap
from src.core.builder import build_trainer
from src.core.run_io import dump_full_config

@hydra.main(config_path="../config", config_name="defaults", version_base=None)
def main(cfg: DictConfig):
    bootstrap()
    dump_full_config(cfg)

    if bool(getattr(cfg, "only_dump", False)):
        return
        
    trainer = build_trainer(cfg)
    trainer.fit()

if __name__ == "__main__":
    main()
