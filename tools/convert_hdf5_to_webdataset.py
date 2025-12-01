"""Convert HDF5 datasets to WebDataset tar shards."""

import argparse
import glob
import heapq
import io
import json
import os
import sys
import tarfile
import threading
import traceback
from datetime import datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import hdf5plugin  # noqa: F401  # registers compression filter
import numpy as np
from tqdm import tqdm

# Required fields for all datasets (BC and RL)
BASE_KEYS = [
    "frame_id",
    "episode_id",
    "ego_vec",
    "bev",
    "route",
    "objects",
    "object_mask",
    "future_xy",
    "future_v",
    "future_mask",
]

# Optional fields for offline RL datasets
# These are automatically detected and included if present in the HDF5 files
OPTIONAL_KEYS = [
    "reward_raw",
    "reward_clipped",
    "reward_normalized",
    "reward_progress",
    "reward_collision",
    "reward_offroad",
    "reward_violation",
    "reward_comfort",
    "reward_completion",
    "reward_mean",
    "reward_std",
    "noise_injected",
    "action_steer",
    "action_throttle",
    "action_brake",
    "route_progress_delta",
    "done",
    "discount",
]


def read_hdf5_sample(f: h5py.File, idx: int, fields: List[str]) -> Dict[str, np.ndarray]:
    """Read a single sample from HDF5 file."""
    sample: Dict[str, np.ndarray] = {}

    for key in fields:
        arr = f[key][idx]

        if key in {"frame_id"}:
            sample[key] = np.array(arr, dtype=np.int64)
        elif key in {"episode_id"}:
            sample[key] = np.array(arr, dtype=np.int16)
        elif key in {"noise_injected", "done"}:
            sample[key] = np.array(arr, dtype=np.bool_)
        else:
            sample[key] = np.array(arr, dtype=np.float32)
        
        # Basic integrity check: forbid NaN/inf to avoid corrupt shards
        if not np.isfinite(sample[key]).all():
            raise ValueError(f"Non-finite values found in field '{key}' at index {idx}")

    return sample


def write_sample_to_tar(tar: tarfile.TarFile, sample_idx: int, sample: Dict[str, np.ndarray]):
    base_name = f"{sample_idx:08d}"
    
    for key, value in sample.items():
        buffer = io.BytesIO()
        np.save(buffer, value, allow_pickle=False)
        buffer.seek(0)
        
        info = tarfile.TarInfo(name=f"{base_name}.{key}.npy")
        info.size = len(buffer.getvalue())
        tar.addfile(info, buffer)


def write_shard_metadata(tar: tarfile.TarFile, shard_idx: int, first_sample_idx: int, 
                        last_sample_idx: int, sample_count: int):
    """Write per-shard metadata to the tar archive."""
    metadata = {
        "shard_idx": shard_idx,
        "first_sample_idx": first_sample_idx,
        "last_sample_idx": last_sample_idx,
        "sample_count": sample_count,
        "created_at": datetime.utcnow().isoformat(),
    }
    
    buffer = io.BytesIO()
    buffer.write(json.dumps(metadata, indent=2).encode('utf-8'))
    buffer.seek(0)
    
    info = tarfile.TarInfo(name="__metadata__.json")
    info.size = len(buffer.getvalue())
    tar.addfile(info, buffer)


def _read_samples_worker(args: Tuple[str, int, int, int, List[str]]) -> List[Tuple[int, Dict[str, np.ndarray]]]:
    file_path, start_idx, end_idx, global_start_idx, fields = args
    file_path = os.path.abspath(os.path.normpath(file_path))
    samples = []

    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"HDF5 file not found in worker: {file_path}")
        if not os.path.isfile(file_path):
            raise ValueError(f"Path is not a file: {file_path}")
        
        with h5py.File(file_path, "r") as f:
            for local_idx in range(start_idx, end_idx):
                global_idx = global_start_idx + (local_idx - start_idx)
                sample = read_hdf5_sample(f, local_idx, fields)
                samples.append((global_idx, sample))
    except Exception as e:
        error_msg = f"[Worker] Error reading {file_path}[{start_idx}:{end_idx}]: {e}\n{traceback.format_exc()}"
        print(error_msg, file=sys.stderr)
        raise
    
    return samples


def convert_hdf5_to_webdataset(
    input_dir: str,
    output_dir: str,
    samples_per_shard: int = 1000,
    shard_pattern: str = "shard-%06d.tar",
    num_workers: int = 0,
) -> Dict[str, int]:
    """
    Convert HDF5 episode-set files to WebDataset tar shards.
    
    Args:
        input_dir: Directory containing .h5 files
        output_dir: Output directory for tar shards
        samples_per_shard: Number of samples per tar shard
        shard_pattern: Format string for shard filenames (should include %d)
        num_workers: Number of worker processes (0 = single-threaded)
        
    Returns:
        Dict with conversion statistics
    """
    # Normalize paths to absolute POSIX paths (handles WSL/Windows path issues)
    input_dir = os.path.abspath(os.path.normpath(input_dir))
    output_dir = os.path.abspath(os.path.normpath(output_dir))
    
    # Convert to Path objects
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Validate input directory
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")
    
    # Validate/cleanup output directory
    if output_dir.exists():
        if output_dir.is_file():
            raise FileExistsError(
                f"Output path exists but is a file, not a directory: {output_dir}\n"
                f"Please remove it or choose a different output directory."
            )
        elif output_dir.is_dir():
            # Check if directory is empty or has existing shards
            existing_files = list(output_dir.glob("shard-*.tar"))
            if existing_files:
                print(f"[WARNING] Output directory already contains {len(existing_files)} shard files.")
                print(f"[WARNING] They will be overwritten. Continuing...")
    else:
        # Create output directory
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise OSError(f"Failed to create output directory {output_dir}: {e}")
    
    # Verify output directory is writable
    if not os.access(output_dir, os.W_OK):
        raise PermissionError(f"Output directory is not writable: {output_dir}")
    
    # Find all HDF5 files
    h5_files = sorted(glob.glob(str(input_dir / "*.h5")))
    if not h5_files:
        raise FileNotFoundError(f"No HDF5 files found in {input_dir}")
    
    print(f"[Convert] Found {len(h5_files)} HDF5 files")
    print(f"[Convert] Input directory: {input_dir}")
    print(f"[Convert] Output directory: {output_dir}")
    
    # Build index of all samples across files
    print("[Convert] Building sample index...")
    file_lengths = []
    file_cumsum = [0]
    
    for fpath in tqdm(h5_files, desc="Indexing files"):
        fpath = os.path.abspath(os.path.normpath(fpath))
        try:
            with h5py.File(fpath, "r") as f:
                available_keys = set(f.keys())
                missing_keys = [key for key in BASE_KEYS if key not in available_keys]
                if missing_keys:
                    raise IOError(f"HDF5 file {fpath} missing required keys: {missing_keys}")

                # Validate objects array has correct shape (11 features)
                if "objects" in available_keys:
                    obj_shape = f["objects"].shape
                    if len(obj_shape) >= 2 and obj_shape[-1] != 11:
                        raise IOError(
                            f"HDF5 file {fpath} has invalid objects array shape: {obj_shape}. "
                            f"Expected 11 features (type_id, x, y, sin_yaw, cos_yaw, length, width, vx, vy, oncoming, priority), "
                            f"but got {obj_shape[-1]} features. This may indicate missing sin_yaw/cos_yaw fields."
                        )

                n_samples = len(f["frame_id"])
                if n_samples == 0:
                    print(f"[WARNING] HDF5 file {fpath} contains 0 samples, skipping")
                    continue

                file_lengths.append(n_samples)
                file_cumsum.append(file_cumsum[-1] + n_samples)
        except Exception as e:
            raise IOError(f"Failed to read HDF5 file {fpath}: {e}")
    
    total_samples = file_cumsum[-1]
    print(f"[Convert] Total samples: {total_samples:,}")
    
    # Read metadata from first file
    fields: List[str] = []
    first_file = os.path.abspath(os.path.normpath(h5_files[0]))
    try:
        with h5py.File(first_file, "r") as f:
            available_keys = set(f.keys())
            fields = [k for k in BASE_KEYS if k in available_keys]
            fields.extend([k for k in OPTIONAL_KEYS if k in available_keys])

            metadata = {
                "K": int(f.attrs.get("K", 32)),
                "N_future": int(f.attrs.get("N_future", 12)),
                "M": int(f.attrs.get("M", 64)),
                "C": int(f.attrs.get("C", 18)),
                "H": int(f.attrs.get("H", 150)),
                "W": int(f.attrs.get("W", 200)),
                "version": f.attrs.get("version", "2.0"),
                "norms_version": int(f.attrs.get("norms_version", 1)),
                "run_id": f.attrs.get("run_id", "").decode() if isinstance(f.attrs.get("run_id", ""), bytes) else f.attrs.get("run_id", ""),
                "fields": fields,
            }
    except Exception as e:
        raise IOError(f"Failed to read metadata from {first_file}: {e}")
    
    # Write metadata file (include conversion settings)
    metadata["samples_per_shard"] = samples_per_shard
    metadata["total_samples"] = total_samples
    metadata["shard_pattern"] = shard_pattern
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[Convert] Wrote metadata to {metadata_path}")
    
    # Convert samples to shards
    print(f"[Convert] Converting {total_samples:,} samples to WebDataset shards...")
    print(f"[Convert] Samples per shard: {samples_per_shard}")
    
    stats = {
        "total_samples": total_samples,
        "total_shards": 0,
        "samples_per_shard": samples_per_shard,
    }
    
    if num_workers > 0:
        # Multi-worker parallel reading
        print(f"[Convert] Using {num_workers} workers for parallel reading")
        
        # Prepare work items: split files into chunks, with correct global offsets
        work_items = []
        
        # Determine chunk size (aim for 2-4 chunks per worker per file)
        for file_idx, h5_path in enumerate(h5_files):
            with h5py.File(h5_path, "r") as f:
                n_samples = len(f["frame_id"])
            
            chunk_size = max(100, n_samples // (num_workers * 2))
            file_start_idx = file_cumsum[file_idx]
            
            for start in range(0, n_samples, chunk_size):
                end = min(start + chunk_size, n_samples)
                global_start_idx = file_start_idx + start
                work_items.append((h5_path, start, end, global_start_idx, fields))
        
        # Stream samples as chunks arrive, write in order using priority queue (memory-efficient)
        print(f"[Convert] Reading {len(work_items)} chunks in parallel (streaming mode)...")
        
        # Thread-safe priority queue for merging chunks in order
        sample_queue = []
        queue_lock = threading.Lock()
        next_expected_idx = 0
        
        # Writer state (single-threaded writing)
        current_shard_idx = 0
        samples_in_current_shard = 0
        current_tar = None
        first_sample_in_shard = None
        
        pbar = tqdm(total=total_samples, desc="Converting samples", unit="samples")
        
        def process_chunk_result(chunk_samples: List[Tuple[int, Dict[str, np.ndarray]]]):
            """Add chunk results to priority queue (thread-safe)."""
            with queue_lock:
                for global_idx, sample in chunk_samples:
                    heapq.heappush(sample_queue, (global_idx, sample))
        
        def write_next_samples():
            """Write samples in order from priority queue (returns count written)."""
            nonlocal current_shard_idx, samples_in_current_shard, current_tar
            nonlocal first_sample_in_shard, next_expected_idx
            
            written = 0
            with queue_lock:
                # Write all consecutive samples we have
                while sample_queue and sample_queue[0][0] == next_expected_idx:
                    global_sample_idx, sample = heapq.heappop(sample_queue)
                    
                    # Open new shard if needed
                    if current_tar is None or samples_in_current_shard >= samples_per_shard:
                        if current_tar is not None:
                            # Write metadata for previous shard before closing
                            write_shard_metadata(
                                current_tar,
                                current_shard_idx - 1,
                                first_sample_in_shard,
                                global_sample_idx - 1,
                                samples_in_current_shard
                            )
                            current_tar.close()
                        
                        shard_name = shard_pattern % current_shard_idx
                        shard_path = output_dir / shard_name
                        current_tar = tarfile.open(str(shard_path), "w")
                        samples_in_current_shard = 0
                        first_sample_in_shard = global_sample_idx
                        current_shard_idx += 1
                        stats["total_shards"] = current_shard_idx
                        
                        tqdm.write(f"[Convert] Writing shard {current_shard_idx-1}: {shard_name}")
                    
                    # Write sample to tar
                    write_sample_to_tar(current_tar, global_sample_idx, sample)
                    samples_in_current_shard += 1
                    next_expected_idx += 1
                    written += 1
                    pbar.update(1)
            
            return written
        
        # Process chunks in parallel, write samples incrementally
        with Pool(processes=num_workers) as pool:
            async_results = pool.imap(_read_samples_worker, work_items)
            
            # Process results as they arrive
            chunks_processed = 0
            for chunk_samples in async_results:
                chunks_processed += 1
                
                # Add samples to priority queue
                process_chunk_result(chunk_samples)
                
                # Try to write samples in order (may write multiple if we have consecutive ones)
                while True:
                    written = write_next_samples()
                    if written == 0:
                        break
        
        # Write any remaining samples in queue (should be in order now)
        while True:
            written = write_next_samples()
            if written == 0:
                break
        
        # Verify we got all samples
        if next_expected_idx != total_samples:
            print(f"[WARNING] Expected {total_samples} samples but only processed {next_expected_idx}")
        
        # Close final shard
        if current_tar is not None:
            # Get last sample index (should be total_samples - 1)
            last_sample_idx = next_expected_idx - 1 if next_expected_idx > 0 else first_sample_in_shard
            write_shard_metadata(
                current_tar,
                current_shard_idx - 1,
                first_sample_in_shard,
                last_sample_idx,
                samples_in_current_shard
            )
            current_tar.close()
        
        pbar.close()
    
    else:
        # Single-threaded (original code)
        current_shard_idx = 0
        samples_in_current_shard = 0
        current_tar = None
        global_sample_idx = 0
        first_sample_in_shard = None
        
        pbar = tqdm(total=total_samples, desc="Converting samples", unit="samples")
        
        for file_idx, h5_path in enumerate(h5_files):
            # Normalize path
            h5_path = os.path.abspath(os.path.normpath(h5_path))
            try:
                with h5py.File(h5_path, "r") as f:
                    n_samples_in_file = len(f["frame_id"])
                    
                    for local_idx in range(n_samples_in_file):
                        # Open new shard if needed
                        if current_tar is None or samples_in_current_shard >= samples_per_shard:
                            if current_tar is not None:
                                # Write metadata for previous shard
                                write_shard_metadata(
                                    current_tar,
                                    current_shard_idx - 1,
                                    first_sample_in_shard,
                                    global_sample_idx - 1,
                                    samples_in_current_shard
                                )
                                current_tar.close()
                            
                            shard_name = shard_pattern % current_shard_idx
                            shard_path = output_dir / shard_name
                            shard_path_str = str(os.path.abspath(os.path.normpath(shard_path)))
                            
                            # Verify parent directory exists
                            shard_path.parent.mkdir(parents=True, exist_ok=True)
                            
                            try:
                                current_tar = tarfile.open(shard_path_str, "w")
                            except Exception as e:
                                raise IOError(f"Failed to open shard file {shard_path_str}: {e}")
                            
                            samples_in_current_shard = 0
                            first_sample_in_shard = global_sample_idx
                            current_shard_idx += 1
                            stats["total_shards"] = current_shard_idx
                            
                            tqdm.write(f"[Convert] Writing shard {current_shard_idx-1}: {shard_name}")
                        
                        # Read sample from HDF5
                        sample = read_hdf5_sample(f, local_idx, fields)
                        
                        # Write to tar
                        write_sample_to_tar(current_tar, global_sample_idx, sample)
                        
                        global_sample_idx += 1
                        samples_in_current_shard += 1
                        pbar.update(1)
            except Exception as e:
                raise IOError(f"Failed to read HDF5 file {h5_path}: {e}")
        
        # Close final shard
        if current_tar is not None:
            write_shard_metadata(
                current_tar,
                current_shard_idx - 1,
                first_sample_in_shard,
                global_sample_idx - 1,
                samples_in_current_shard
            )
            current_tar.close()
        
        pbar.close()
    
    print(f"[Convert] Conversion complete!")
    print(f"[Convert] Created {stats['total_shards']} shards in {output_dir}")
    
    return stats


def read_sample_from_shard(shard_path: str, sample_idx: int, fields: List[str]) -> Optional[Dict[str, np.ndarray]]:
    """
    Read a single sample from a WebDataset shard.
    
    Args:
        shard_path: Path to tar shard file
        sample_idx: Sample index to read
        
    Returns:
        Dict with sample data, or None if not found
    """
    base_name = f"{sample_idx:08d}"
    sample: Dict[str, np.ndarray] = {}
    
    try:
        shard_path = os.path.abspath(os.path.normpath(shard_path))
        with tarfile.open(shard_path, "r") as tar:
            members = {m.name: m for m in tar.getmembers()}
            
            for key in fields:
                member_name = f"{base_name}.{key}.npy"
                if member_name not in members:
                    return None
                
                buffer = tar.extractfile(members[member_name])
                if buffer is None:
                    return None
                
                data_bytes = buffer.read()
                sample[key] = np.load(io.BytesIO(data_bytes))
    except (KeyError, tarfile.TarError, IOError):
        return None
    
    return sample if len(sample) == len(fields) else None


def verify_conversion(
    input_dir: str,
    output_dir: str,
    num_samples_to_check: int = 100,
) -> bool:
    """
    Verify conversion integrity by comparing random samples.
    
    Memory-efficient: Only loads samples being checked, not entire shards.
    
    Args:
        input_dir: Original HDF5 directory
        output_dir: WebDataset shard directory
        num_samples_to_check: Number of random samples to verify
        
    Returns:
        True if verification passes, False otherwise
    """
    print(f"\n[Verify] Checking integrity of {num_samples_to_check} random samples...")
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Load metadata
    metadata_path = output_dir / "metadata.json"
    if not metadata_path.exists():
        print("[Verify] ERROR: metadata.json not found")
        return False
    
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    samples_per_shard = metadata.get("samples_per_shard", 1000)
    fields = metadata.get("fields", BASE_KEYS)
    
    # Build HDF5 index
    h5_files = sorted(glob.glob(str(input_dir / "*.h5")))
    file_cumsum = [0]
    
    for fpath in h5_files:
        fpath = os.path.abspath(os.path.normpath(fpath))
        try:
            with h5py.File(fpath, "r") as f:
                n_samples = len(f["frame_id"])
                file_cumsum.append(file_cumsum[-1] + n_samples)
        except Exception as e:
            print(f"[Verify] ERROR: Failed to read HDF5 file {fpath}: {e}")
            return False
    
    total_samples = file_cumsum[-1]
    
    # Get shard files and sort by shard index
    shard_pattern = metadata.get("shard_pattern", "shard-%06d.tar")
    shard_files_glob = sorted(glob.glob(str(output_dir / "shard-*.tar")))
    
    if not shard_files_glob:
        print("[Verify] ERROR: No shard files found")
        return False
    
    # Extract shard indices and sort
    import re
    shard_files = []
    for fpath in shard_files_glob:
        # Extract number from filename (e.g., "shard-000001.tar" -> 1)
        match = re.search(r'shard-(\d+)\.tar', Path(fpath).name)
        if match:
            shard_idx = int(match.group(1))
            shard_files.append((shard_idx, fpath))
    
    shard_files.sort(key=lambda x: x[0])
    shard_files = [fpath for _, fpath in shard_files]
    
    if not shard_files:
        print("[Verify] ERROR: Could not parse shard indices from filenames")
        return False
    
    # Check random samples
    import random
    random.seed(42)
    np.random.seed(42)
    
    samples_to_check = sorted(random.sample(range(total_samples), min(num_samples_to_check, total_samples)))
    
    errors = []
    for sample_idx in tqdm(samples_to_check, desc="Verifying samples"):
        # Find which shard contains this sample
        shard_idx = sample_idx // samples_per_shard
        
        if shard_idx >= len(shard_files):
            errors.append(f"Sample {sample_idx}: Shard index {shard_idx} out of range")
            continue
        
        shard_path = shard_files[shard_idx]
        
        # Read from HDF5
        file_idx = np.searchsorted(file_cumsum, sample_idx, side="right") - 1
        local_h5_idx = sample_idx - file_cumsum[file_idx]
        
        with h5py.File(h5_files[file_idx], "r") as f:
            h5_sample = read_hdf5_sample(f, local_h5_idx, fields)
        
        # Read from WebDataset shard
        wds_sample = read_sample_from_shard(shard_path, sample_idx, fields)
        
        if wds_sample is None:
            errors.append(f"Sample {sample_idx}: Not found in shard {shard_idx}")
            continue
        
        # Compare all fields
        for key in h5_sample.keys():
            if key not in wds_sample:
                errors.append(f"Sample {sample_idx}: Missing key '{key}'")
                continue
            
            h5_arr = h5_sample[key]
            wds_arr = wds_sample[key]
            
            # Check shape
            if h5_arr.shape != wds_arr.shape:
                errors.append(f"Sample {sample_idx}, key '{key}': Shape mismatch {h5_arr.shape} vs {wds_arr.shape}")
                continue
            
            if key in ["frame_id", "episode_id"]:
                if h5_arr.dtype != wds_arr.dtype:
                    errors.append(f"Sample {sample_idx}, key '{key}': Dtype mismatch {h5_arr.dtype} vs {wds_arr.dtype}")
                    continue
                # For integer fields, require exact match
                if not np.array_equal(h5_arr, wds_arr):
                    errors.append(f"Sample {sample_idx}, key '{key}': Value mismatch (integers must match exactly)")
                    continue
            elif key in ["noise_injected", "done"]:
                if h5_arr.dtype != wds_arr.dtype:
                    errors.append(f"Sample {sample_idx}, key '{key}': Dtype mismatch {h5_arr.dtype} vs {wds_arr.dtype}")
                    continue
                if not np.array_equal(h5_arr, wds_arr):
                    errors.append(f"Sample {sample_idx}, key '{key}': Value mismatch (bool fields must match)")
            else:
                # Check values (allow small floating point differences for float arrays)
                max_diff = np.abs(h5_arr.astype(np.float32) - wds_arr.astype(np.float32)).max()
                if max_diff > 1e-5:
                    errors.append(f"Sample {sample_idx}, key '{key}': Max diff = {max_diff:.2e}")
    
    if errors:
        print(f"\n[Verify] FAILED: Found {len(errors)} errors:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
        return False
    else:
        print(f"\n[Verify] SUCCESS: All {num_samples_to_check} samples verified correctly!")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert HDF5 BC datasets to WebDataset tar shards",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory containing .h5 files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for WebDataset tar shards",
    )
    parser.add_argument(
        "--samples_per_shard",
        type=int,
        default=1000,
        help="Number of samples per tar shard",
    )
    parser.add_argument(
        "--shard_pattern",
        type=str,
        default="shard-%06d.tar",
        help="Format string for shard filenames (must include %%d)",
    )
    parser.add_argument(
        "--skip_verify",
        action="store_true",
        help="Skip integrity verification after conversion",
    )
    parser.add_argument(
        "--verify_samples",
        type=int,
        default=100,
        help="Number of random samples to verify (0 = verify all)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of worker processes for parallel reading (0 = single-threaded, default: 0)",
    )
    
    args = parser.parse_args()
    # Determine number of workers
    num_workers = args.num_workers
    if num_workers == 0:
        num_workers = 0  # Single-threaded
    elif num_workers < 0:
        num_workers = max(1, cpu_count() + num_workers)  # Negative = CPU count + num_workers
    
    print("=" * 70)
    print("HDF5 to WebDataset Converter")
    print("=" * 70)
    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Samples per shard: {args.samples_per_shard}")
    print(f"Workers: {num_workers if num_workers > 0 else 'single-threaded'}")
    print("=" * 70)
    
    # Convert
    stats = convert_hdf5_to_webdataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        samples_per_shard=args.samples_per_shard,
        shard_pattern=args.shard_pattern,
        num_workers=num_workers,
    )
    
    # Verify
    if not args.skip_verify:
        verify_samples = args.verify_samples
        if verify_samples == 0:
            verify_samples = stats["total_samples"]
        
        success = verify_conversion(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            num_samples_to_check=verify_samples,
        )
        
        if not success:
            print("\n[ERROR] Integrity verification failed!")
            sys.exit(1)
    
    print("\n" + "=" * 70)
    print("Conversion complete and verified!")
    print("=" * 70)
    print(f"Total samples: {stats['total_samples']:,}")
    print(f"Total shards: {stats['total_shards']}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
