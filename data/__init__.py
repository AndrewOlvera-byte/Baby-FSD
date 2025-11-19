"""
BC data collection and loading utilities (v2 - HDF5+LZ4).

Modules:
    schema: Legacy Parquet table schemas (for reference)
    writer: Table writers (HDF5, and legacy Parquet/DuckDB)
    hdf5_writer: HDF5 episode-set writer for BC v2
    torch_dataset: PyTorch Dataset and DataLoader for training (HDF5-backed)
    norms: Normalization functions for training data
    transforms: Data augmentation transforms
"""

from data.schema import Schemas
from data.torch_dataset import BCTrajectoryDataset, create_bc_dataloader
from data.hdf5_writer import HDF5EpisodeSetWriter

__all__ = [
    "Schemas",
    "BCTrajectoryDataset",
    "create_bc_dataloader",
    "HDF5EpisodeSetWriter",
]
