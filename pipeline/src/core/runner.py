from __future__ import annotations
import hydra
from omegaconf import DictConfig
from .builder import build_trainer

@hydra.main(config_path="../../config", config_name="defaults", version_base=None)
def main(cfg: DictConfig):
    trainer = build_trainer(cfg)
    trainer.fit()

if __name__ == "__main__":
    main()
