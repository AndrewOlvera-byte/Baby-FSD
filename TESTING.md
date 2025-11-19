# Testing Guide for Baby-FSD

## Running Tests

### Prerequisites

**IMPORTANT:** Tests must be run from the `carla-env-3.7` conda environment (not the base environment):

```bash
conda activate carla-env-3.7
```

Verify you're in the correct environment:
```bash
python --version  # Should show Python 3.7.x
```

### Install Test Dependencies

From the project root with the carla-env activated:
```bash
pip install -r requirements.txt
pip install -r pipeline/requirements.txt  # For pipeline components
```

### Run All Tests

From the project root directory:
```bash
pytest tests/ -v
```

### Run Specific Test Suites

**HDF5 Data Loading Tests:**
```bash
pytest tests/test_data_loader.py -v
pytest tests/test_torch_dataset.py -v
```

**BC Model Tests:**
```bash
pytest tests/test_bc_model.py -v
pytest tests/test_bc_policy.py -v
```

### Test Configuration

The `pytest.ini` file configures:
- Python path to include project root
- Test discovery patterns
- Output formatting

### Common Issues

**Issue: "ModuleNotFoundError: No module named 'pyarrow'"**
- **Solution**: You're in the wrong Python environment. Activate `carla-env-3.7` first.

**Issue: "ModuleNotFoundError: No module named 'data'"**
- **Solution**: Run pytest from the project root directory (`C:\Work\ML\Baby-FSD`), not from `tests/` subdirectory.

**Issue: "ModuleNotFoundError: No module named 'components'"**
- **Solution**: This is fixed - the test files now properly add `pipeline/src` to the path.

### Expected Test Results

With the HDF5 refactor complete, you should have:

- ✅ `test_data_loader.py` - Tests HDF5 episode-set writing and reading
- ✅ `test_torch_dataset.py` - Tests PyTorch dataset and dataloader with HDF5
- ✅ `test_bc_model.py` - Tests BC model architecture
- ✅ `test_bc_policy.py` - Tests BC policy model

### Running Tests for HDF5 Data

**After collecting a dataset:**
```bash
# 1. Activate environment
conda activate carla-env-3.7

# 2. Collect a small test dataset (if not done already)
python collect/collect_bc.py --config configs/collect_bc.yaml

# 3. Run data loading tests
pytest tests/test_data_loader.py tests/test_torch_dataset.py -v

# 4. If tests need actual data, update the test to point to your run:
# Edit test_torch_dataset.py and update the path in create_test_hdf5_dataset()
```

### Test Structure

```
tests/
├── test_data_loader.py      # HDF5 writer/reader unit tests (creates synthetic data)
├── test_torch_dataset.py    # PyTorch dataset tests (creates synthetic HDF5 files)
├── test_bc_model.py          # BC model architecture tests (requires pipeline components)
├── test_bc_policy.py         # BC policy model tests (requires pipeline components)
└── verify_training_ready.py  # End-to-end validation script
```

**Note:** The model tests (`test_bc_model.py`, `test_bc_policy.py`) temporarily change to the pipeline directory during import to access the pipeline's component registry. This mimics how the pipeline normally runs.

### Continuous Testing

During development:
```bash
# Watch mode (re-run tests on file changes - requires pytest-watch)
pip install pytest-watch
ptw tests/ -v

# Run with coverage
pip install pytest-cov
pytest tests/ --cov=data --cov=collect -v
```

### Integration Testing

For full pipeline validation:
```bash
# Verify training readiness (requires real collected data)
python tests/verify_training_ready.py
```

## Troubleshooting

### Check Your Environment

```bash
conda activate carla-env-3.7
python -c "import sys; print(sys.executable)"
python -c "import pyarrow; print('pyarrow OK')"
python -c "import h5py; print('h5py OK')"
python -c "import torch; print('torch OK')"
```

All imports should succeed without errors.

### Check pytest.ini

The `pytest.ini` file in the project root should have:
```ini
[pytest]
pythonpath = .
```

This ensures the project root is in the Python path for imports.

### Manual Test Run

If pytest has issues, you can run tests directly:
```bash
cd tests
python test_data_loader.py
python test_torch_dataset.py
```

But pytest is preferred for proper test discovery and reporting.

