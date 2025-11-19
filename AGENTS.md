# Repository Guidelines

## Project Structure & Module Organization
Baby-FSD couples CARLA data collection with policy training. Collectors live in `collect/` and reuse math/BEV helpers from `carla_utils/`, while policy code and registries are under `pipeline/src/` with configs in `pipeline/config/` and runner scripts in `pipeline/scripts/`. Adjust behavior cloning knobs in `configs/collect_bc.yaml`, persist datasets in `data/BC_v2/`, keep experiment crumbs inside `outputs/`, and lean on `tests/` plus `tools/` (`validate_bc_run.py`, `loader_example.py`) for verification.

## Build, Test, and Development Commands
Work inside the `carla-env-3.7` conda environment.
- `pip install -r requirements.txt` and `pip install -r pipeline/requirements.txt` install simulator, data, and pipeline dependencies.
- `python model3_visual_confirm_test.py --host 127.0.0.1 --port 2000 --tm-port 8000 --vehicles 50 --walkers 50` sanity-checks BehaviorAgent control against a live CARLA server.
- `python collect/collect_bc.py --config configs/collect_bc.yaml` records behavior-cloning episodes into `data/BC_v2/run-YYYYmmdd-HHMMSS`.
- `python tools/validate_bc_run.py --root data/BC_v1` replays a dataset to catch BEV or future mismatches before training.
- `pytest tests/ -v` runs everything; narrow with `pytest tests/test_bc_model.py -v` while polishing pipeline modules.

## Coding Style & Naming Conventions
Stick to Python 3.7, 4-space indents, and PEP 8. Functions, files, and config keys stay `snake_case`, constants remain SHOUT_CASE (e.g., `ROUTE_PRIME_MIN_TICKS`), and type hints should match the surrounding modules. Keep logging consistent with `LOG = logging.getLogger(...)`, prefer small helpers over inlined logic, and mirror existing CLI flag names (`--tm-port`, `future_dt`) to keep scripts composable.

## Testing Guidelines
Pytest drives validation. Follow the `tests/test_*.py` naming scheme, generate synthetic HDF5 fixtures where possible, and run `pytest tests/ -v` before pushing. After touching schemas or data loaders, repeat `pytest tests/test_data_loader.py tests/test_torch_dataset.py -v`; for training edits, focus on `tests/test_bc_model.py` and `tests/test_bc_policy.py`. When IO logic changes, capture coverage with `pytest tests/ --cov=data --cov=collect`.

## Commit & Pull Request Guidelines
History favors concise, action-driven messages (`improved data loading using cache`, `perfect collection`). Keep summaries imperative, mention the subsystem (`collect`, `pipeline`, `tools`), and list schema or CARLA version bumps in the body. Pull requests should describe reproduction steps, list key commands (collector, validator, pytest), and attach simulator logs or screenshots for visual features. Link issues or datasets so the next agent can replay your scenario.

## Simulator & Data Tips
Always start a CARLA server (Town10HD_Opt, ports from `configs/collect_bc.yaml`) before running scripts. For smoke tests, set `episodes: 1` and `npc_count: 20`, then restore production values before merging. Watch collector logs for `bev_count_mismatch` or `futures_not_divisible_by_N`, and probe tensors with `python tools/loader_example.py --run_dir <data/BC_v2/...>` before kicking off training.
