# Baby-FSD 🚗

A research project for training autonomous driving policies using Behavior Cloning (BC) and Offline Reinforcement Learning in CARLA simulator. Built for researchers and hobbyists who want to experiment with end-to-end driving models.

**We welcome all contributions!** Whether it's bug fixes, new features, documentation improvements, or just ideas—open an issue or submit a PR. This is an open-source project and we'd love your help making it better.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Workflow](#workflow)
  - [1. Collecting BC Data](#1-collecting-bc-data)
  - [2. Validating Your Dataset](#2-validating-your-dataset)
  - [3. Converting to WebDataset](#3-converting-to-webdataset)
  - [4. Training a BC Policy](#4-training-a-bc-policy)
  - [5. Collecting Offline RL Data](#5-collecting-offline-rl-data)
  - [6. Training with Offline RL](#6-training-with-offline-rl)
- [Codebase Overview](#codebase-overview)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Baby-FSD implements a complete pipeline for:

1. **Data Collection** — Driving around CARLA with an expert agent, recording observations (BEV, objects, route) and actions (future waypoints, speeds)
2. **Behavior Cloning** — Learning to imitate the expert from collected demonstrations
3. **Offline RL** — Fine-tuning with reward signals (IQL, CQL, TD3+BC) using noisy rollouts

The observation space includes:
- **Ego state** — speed, acceleration, yaw rate, steering, curvature, speed limit, navigation command
- **BEV raster** — 18-channel bird's-eye view with lanes, route, objects, traffic lights
- **Route polyline** — K waypoints ahead in ego frame with curvature info
- **Object tokens** — nearby vehicles/pedestrians with position, velocity, type

The action space is N future waypoints + speeds (trajectory prediction style).

---

## Project Structure

```
Baby-FSD/
├── carla_utils/           # CARLA interaction utilities
│   ├── actors.py          # Spawn/query nearby actors
│   ├── bev.py             # BEV rasterization (18 channels)
│   ├── futures.py         # Future waypoint extraction
│   ├── map_vectors.py     # Lane/traffic light geometry
│   ├── rewards.py         # Reward computation for RL
│   └── route.py           # Route planning and sampling
│
├── collect/               # Data collection scripts
│   ├── collect_bc.py      # Behavior Cloning data collector
│   └── collect_offline_rl.py  # Offline RL data collector (with noise + rewards)
│
├── configs/               # Collection configs (YAML)
│   ├── collect_bc.yaml
│   └── collect_offline_rl.yaml
│
├── data/                  # Data utilities and storage
│   ├── hdf5_writer.py     # Efficient HDF5 episode writer
│   ├── norms.py           # Normalization functions
│   ├── schema.py          # Data schema definitions
│   ├── torch_dataset.py   # PyTorch dataset wrappers
│   └── transforms.py      # Data augmentations
│
├── tools/                 # Inspection and conversion tools
│   ├── convert_hdf5_to_webdataset.py  # HDF5 → WebDataset shards
│   ├── inspect_normalized_hdf5.py     # Validate HDF5 datasets
│   └── inspect_webdataset.py          # Validate WebDataset shards
│
├── pipeline/              # Training pipeline (Hydra-based)
│   ├── config/            # Hydra configs
│   │   ├── exp/           # Experiment configs (bc_train, offline_rl_train, etc.)
│   │   ├── model/         # Model architectures
│   │   └── ...
│   ├── scripts/
│   │   ├── train.py       # Main training entry point
│   │   └── eval.py        # Evaluation script
│   ├── src/
│   │   ├── components/    # Models, datasets, evaluators, etc.
│   │   ├── core/          # Training infrastructure
│   │   └── trainers/      # BC and Offline RL trainers
│   └── docker/            # Docker setup for reproducible runs
│
├── tests/                 # Unit tests
├── requirements.txt       # Python dependencies
└── README.md              # You are here!
```

---

## Getting Started

### Prerequisites

- Python 3.7+ (we use 3.7 for CARLA compatibility, but 3.11+ works for training-only)
- CARLA Simulator 0.9.x (tested with 0.9.13–0.9.15)
- CUDA-capable GPU (for training)

### Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/Baby-FSD.git
cd Baby-FSD

# Create conda environment (recommended)
conda create -n carla-env python=3.7
conda activate carla-env

# Install dependencies
pip install -r requirements.txt

# For training pipeline (can use newer Python)
pip install -r pipeline/requirements.txt
```

### CARLA Setup

1. Download CARLA from https://carla.org/
2. Start the CARLA server:
   ```bash
   # Linux
   ./CarlaUE4.sh -quality-level=Low -RenderOffScreen
   
   # Or with a specific map
   ./CarlaUE4.sh -quality-level=Low -RenderOffScreen -carla-world-port=2000
   ```
3. Make sure the CARLA Python API is in your `PYTHONPATH`

---

## Workflow

Here's the typical workflow from raw data collection to a trained policy:

### 1. Collecting BC Data

First, start CARLA server, then run the collector:

```bash
python collect/collect_bc.py --config configs/collect_bc.yaml
```

**What happens:**
- Spawns a Tesla Model 3 with CARLA's built-in autopilot
- Drives around collecting frames at fixed timesteps
- Records ego state, BEV, route, objects, and future waypoints
- Saves to HDF5 files with LZ4 compression

**Output:** `data/BC_v3/run-YYYYMMDD-HHMMSS/`
- `*.h5` files — Episode sets (5 episodes each by default)
- `meta.json` — Run metadata (config, stats)

**Tip:** For a quick sanity check, edit the config:
```yaml
episodes: 1
episode_max_frames: 400
npc_count: 20
```

### 2. Validating Your Dataset

Before training, always validate your data:

```bash
# Inspect HDF5 files and check for issues
python tools/inspect_normalized_hdf5.py \
    --input_dir data/BC_v3/run-YYYYMMDD-HHMMSS \
    --mode bc \
    --num_steps 100
```

**What to check:**
- No NaN/inf values in any field
- Shapes match expected dimensions
- Reasonable value ranges for normalized data
- No missing frames or corrupt episodes

### 3. Converting to WebDataset

For efficient training with multiple workers, convert HDF5 to WebDataset tar shards:

```bash
python tools/convert_hdf5_to_webdataset.py \
    --input_dir data/BC_v3/run-YYYYMMDD-HHMMSS \
    --output_dir data/BC_v3/run-YYYYMMDD-HHMMSS-wds \
    --samples_per_shard 1000 \
    --num_workers 4
```

**What happens:**
- Reads all HDF5 files
- Validates required fields and data integrity
- Writes tar shards with samples as `.npy` files
- Runs automatic verification on random samples

**Output:** `data/BC_v3/run-YYYYMMDD-HHMMSS-wds/`
- `shard-000000.tar`, `shard-000001.tar`, ...
- `metadata.json` — Dataset metadata (shapes, normalization, total samples)

**Verify the conversion:**
```bash
python tools/inspect_webdataset.py \
    --input_dir data/BC_v3/run-YYYYMMDD-HHMMSS-wds \
    --num_steps 100
```

### 4. Training a BC Policy

With your WebDataset ready, train a behavior cloning policy:

```bash
cd pipeline

# Train with default config
python -m scripts.train exp=bc_train

# Or with custom settings
python -m scripts.train exp=bc_train \
    trainer.epochs=50 \
    trainer.batch_size=64 \
    dataset.run_dir=data/BC_v3/run-YYYYMMDD-HHMMSS-wds
```

**What happens:**
- Loads WebDataset shards with multiple workers
- Trains a policy network (BEV encoder + object transformer + MLP head)
- Predicts N future waypoints + speeds
- Uses cosine LR schedule with warmup

**Output:** `outputs/bc_train/train/YYYYMMDD_HHMMSS/`
- `checkpoints/` — Model checkpoints
- `reports/metrics.jsonl` — Training logs
- `reports/config.yaml` — Full resolved config

### 5. Collecting Offline RL Data

For offline RL, we need diverse data with rewards. This collector adds:
- Action noise injection (stochastic exploration)
- Reward signals (progress, collision, offroad, comfort)
- Episode termination on crashes

```bash
python collect/collect_offline_rl.py --config configs/collect_offline_rl.yaml
```

**Key config options:**
```yaml
# Noise injection for exploration
noise:
  probability: 0.2     # 20% of steps get noise
  steer_std: 0.1
  throttle_std: 0.05

# Reward shaping
rewards:
  progress_weight: 0.1
  collision_penalty: -5.0
  offroad_penalty: -2.0
  completion_bonus: 5.0
```

**Output:** `data/RL_v1/run-YYYYMMDD-HHMMSS/`
- Same structure as BC, plus reward fields

### 6. Training with Offline RL

Fine-tune your BC policy (or train from scratch) with offline RL:

```bash
cd pipeline

python -m scripts.train exp=offline_rl_train \
    dataset.run_dir=data/RL_v1/run-YYYYMMDD-HHMMSS-wds
```

**Supported algorithms:**
- **IQL** (Implicit Q-Learning) — default, conservative, stable
- **CQL** (Conservative Q-Learning) — penalizes OOD actions
- **TD3+BC** — TD3 with BC regularization

**Key hyperparameters:**
```yaml
trainer:
  algorithm: iql
  gamma: 0.99          # Discount factor
  tau: 0.005           # Target network update rate
  expectile: 0.7       # IQL expectile (higher = more conservative)
  temperature: 3.0     # Advantage weighting temperature
```

---

## Codebase Overview

### `carla_utils/` — CARLA Interaction

The glue between CARLA and our data format:

- **`actors.py`** — Spawns ego vehicle, queries nearby actors, extracts state vectors
- **`bev.py`** — Rasterizes 18-channel BEV (lanes, route corridor, objects, traffic lights)
- **`futures.py`** — Extracts future waypoints from expert trajectory for supervision
- **`route.py`** — Route planning, command decoding, polyline sampling
- **`rewards.py`** — Computes dense rewards for offline RL

### `data/` — Data Handling

- **`schema.py`** — Defines data dimensions (K route points, N futures, M objects)
- **`norms.py`** — Normalization functions (crucial for training stability)
- **`hdf5_writer.py`** — Efficient chunked HDF5 writing with LZ4 compression
- **`torch_dataset.py`** — PyTorch dataset classes for training

### `pipeline/` — Training Infrastructure

Built on Hydra for config management:

- **`src/trainers/bc_trainer.py`** — Behavior cloning training loop
- **`src/trainers/offline_rl_trainer.py`** — IQL/CQL/TD3+BC implementations
- **`src/components/models/`** — Policy networks (BEV encoder, object transformer)
- **`config/exp/`** — Experiment configs you can override from CLI

### `tools/` — Utilities

- **`convert_hdf5_to_webdataset.py`** — Format conversion with integrity checks
- **`inspect_*.py`** — Data validation and debugging

---

## Testing

We use pytest for testing. Run the test suite:

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=. --cov-report=html
```

See `TESTING.md` for more details on test organization.

---

## Contributing

**All contributions are welcome!** This is an open-source research project and we appreciate any help:

- 🐛 **Bug reports** — Found something broken? Open an issue!
- ✨ **Feature requests** — Have an idea? Let's discuss it!
- 📝 **Documentation** — Typos, unclear explanations, missing examples
- 🔧 **Code contributions** — Bug fixes, new features, optimizations
- 🧪 **Testing** — More tests are always helpful

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest tests/ -v`)
5. Commit (`git commit -m 'Add amazing feature'`)
6. Push (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Code Style

- We use standard Python conventions (PEP 8)
- Type hints are appreciated but not required
- Add docstrings for public functions
- Keep commits focused and atomic

---

## License

This project is open source. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [CARLA Simulator](https://carla.org/) — The awesome open-source driving simulator
- The autonomous driving research community for inspiration

---

**Questions?** Open an issue or start a discussion. Happy driving! 🚗💨
