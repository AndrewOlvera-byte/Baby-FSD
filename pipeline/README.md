# Pipeline Usage

Training pipeline for Behavior Cloning and Offline RL models.

## Prerequisites
- Docker with NVIDIA GPU support (Linux: nvidia-container-toolkit; Windows: WSL2 + Docker Desktop + GPU enabled).
- Python 3.11+ for local runs (optional; Docker handles deps).

## Build Image
Build once for Docker usage:
```bash
# Linux/macOS
BASE_IMAGE=pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime docker compose -f docker/docker-compose.yaml build --no-cache

# Windows PowerShell
$env:BASE_IMAGE="pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime"; docker compose -f docker/docker-compose.yaml build --no-cache
```
Omit `BASE_IMAGE` for default.

## Run Training (Local or Container)
Outputs go to `outputs/${exp.name}/${modes.mode}/${now:%Y%m%d_%H%M%S}/reports/` (config, metrics.jsonl, checkpoints, etc.).

### Local (Host)
```bash
# Default BC training (default experiment)
python -m scripts.train

# BC with custom dataset path
python -m scripts.train exp=bc_train dataset.run_dir=data/BC_v3/run-YYYYMMDD-HHMMSS-wds

# Offline RL training (IQL by default)
python -m scripts.train exp=offline_rl_train dataset.run_dir=data/RL_v1/run-YYYYMMDD-HHMMSS-wds
```

### Container (Recommended for GPU/Isolation)
From project root:
```bash
cd docker
docker-compose up -d trainer  # Runs python scripts/train.py (default: bc_train)
```
Override exp:
```bash
cd docker

# BC training
docker-compose run --rm trainer python -m scripts.train exp=bc_train

# Offline RL training
docker-compose run --rm trainer python -m scripts.train exp=offline_rl_train
```

## Available Experiments

| Experiment | Description |
|------------|-------------|
| `bc_train` | Behavior Cloning (default) |
| `bc_train_small` | BC with smaller batch/epochs for debugging |
| `bc_train_70k` | BC tuned for ~70k samples |
| `offline_rl_train` | Offline RL with IQL/CQL/TD3+BC |

## Evaluate
```bash
# Local - evaluate trained BC model in CARLA
python -m scripts.eval exp=bc_train

# Container
cd docker && docker-compose run --rm eval
```

## Hyperparameter Search (Optuna)
```bash
# Local multirun
python -m scripts.search -m +hpo=optuna exp=bc_train hydra.sweeper.n_trials=50

# Container
cd docker && docker-compose run --rm search
```
Results in per-trial run dirs with topk.json summary.

## Interactive Shell
GPU-enabled shell with mounted source:
```bash
cd docker && docker-compose run --rm trainer bash
```
Inside: `python -m scripts.train exp=bc_train` or `pytest`.

## Monitor Logs
- Local: Watch terminal (progress bars + metrics).
- Container: `docker-compose logs -f trainer` (from docker dir).

## Troubleshooting
- GPU not available? Run `docker-compose run --rm trainer nvidia-smi` to check.
- Interpolation errors? Ensure Hydra overrides are correct (e.g., `exp=...`).
- Dataset not found? Check your `dataset.run_dir` path points to a valid WebDataset directory.
- OOM errors? Reduce `trainer.batch_size` in experiment config.
- Rebuild if deps change: `docker compose build --no-cache`.

For custom experiments, edit `config/exp/*.yaml` and rerun.
