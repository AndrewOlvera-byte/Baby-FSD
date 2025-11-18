# Pipeline V4 Usage

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
# Default YOLO segmentation training (full run)
python -m scripts.train

# Quick YOLO debug pass (1 epoch)
python -m scripts.train exp=yolo_seg_debug

# Explicit full-run selection (same as default)
python -m scripts.train exp=yolo_seg_train
```

### Container (Recommended for GPU/Isolation)
From project root:
```bash
cd docker
docker-compose up -d trainer  # Runs python scripts/train.py (default exp)
```
Override exp:
```bash
cd docker

# YOLO segmentation (debug)
docker-compose run --rm trainer python -m scripts.train exp=yolo_seg_debug

# YOLO segmentation (full run)
docker-compose run --rm trainer python -m scripts.train exp=yolo_seg_train
```

## Evaluate
```bash
# Local
python -m scripts.eval exp=cifar10_eval

# Container
cd docker && docker-compose run --rm eval  # Runs python scripts/eval.py
```

## Hyperparameter Search (Optuna)
```bash
# Local multirun
python -m scripts.search -m +hpo=optuna +hpo_space=space_sl_basic exp=cifar10_train hydra.sweeper.n_trials=50

# Container
cd docker && docker-compose run --rm search  # Example: +hpo=optuna +hpo_space=space_sl_basic
```
Results in per-trial run dirs with topk.json summary.

## Interactive Shell
GPU-enabled shell with mounted source:
```bash
cd docker && docker-compose run --rm trainer bash
```
Inside: `python -m scripts.train exp=quick_debug_sl` or `pytest`.

## Monitor Logs
- Local: Watch terminal (progress bars + metrics).
- Container: `docker-compose logs -f trainer` (from docker dir).

## YOLO Segmentation Training
The pipeline supports YOLOv11 segmentation training with a self-contained wrapper:

### Features
- Self-sufficient trainer (no registry dependencies)
- Wrapper around ultralytics YOLO.train()
- Built-in progress tracking and metrics visualization
- Automatic training history plots (loss/mAP over epochs)
- Optimized for high-resolution images (1920x1080 → 2048px)

### Dataset Setup
Place your YOLO dataset in `data/yolo_dataset_v1/` with structure:
```
data/yolo_dataset_v1/
  ├── data.yaml          # YOLO data config
  ├── images/
  │   ├── train/         # Training images
  │   └── val/           # Validation images
  └── labels/
      ├── train/         # Training labels (.txt)
      └── val/           # Validation labels (.txt)
```

### Configurations
- `yolo_seg_debug`: lightweight smoke test (1 epoch, batch=4, imgsz=1280)
- `yolo_seg_train`: full run (100 epochs, batch=12, imgsz=2048, tuned for 16GB VRAM)
- Tweak hyperparameters directly in `src/trainers/yolo_seg_configs/debug.yaml` or `src/trainers/yolo_seg_configs/train.yaml` (no Hydra edits required)

### Outputs
Training outputs saved to `outputs/${exp.name}/${modes.mode}/${timestamp}/`:
- `reports/training_history.png`: Loss and mAP plots
- `yolo_train/`: YOLO native outputs (weights, results.csv, plots)
- `reports/metrics.jsonl`: Training logs

## Troubleshooting
- GPU not available? Run `docker-compose run --rm trainer nvidia-smi` to check.
- Interpolation errors? Ensure Hydra overrides are correct (e.g., `exp=...`).
- Data missing? CIFAR-10 downloads to `./data` on first run (no copying to outputs).
- YOLO dataset not found? Check `data/yolo_dataset_v1/data.yaml` exists and has correct paths.
- OOM errors? Reduce batch_size in experiment config or use smaller imgsz.
- Rebuild if deps change: `docker compose build --no-cache`.

For custom exps, edit `config/exp/*.yaml` and rerun.