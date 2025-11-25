"""
Inspect and print normalized values from HDF5 BC dataset.

Reads the first N steps from HDF5 files, normalizes them using norms.py,
and prints statistics and sample values for inspection.

Usage:
    python tools/inspect_normalized_hdf5.py \
        --input_dir data/BC_v2/run-YYYYMMDD-HHMMSS \
        --num_steps 100
"""

import os
import sys
import argparse
import glob
from pathlib import Path
from typing import Dict, List
import numpy as np

# Add parent directory to path to allow imports from data module
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import h5py
    import hdf5plugin  # Register LZ4 compression filter
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False
    print("Error: h5py and hdf5plugin are required. Install with: pip install h5py hdf5plugin")
    sys.exit(1)

import torch
from data.norms import (
    normalize_route_points,
    normalize_futures,
    normalize_object_tokens,
    normalize_bev,
)


def read_hdf5_sample(f: h5py.File, idx: int) -> Dict[str, np.ndarray]:
    """
    Read a single sample from HDF5 file.
    
    Args:
        f: Open HDF5 file handle
        idx: Sample index within file
        
    Returns:
        Dict with all sample data as numpy arrays
    """
    return {
        "frame_id": f["frame_id"][idx],
        "episode_id": f["episode_id"][idx],
        "ego_vec": f["ego_vec"][idx].astype(np.float32),
        "bev": f["bev"][idx].astype(np.float32),
        "route": f["route"][idx].astype(np.float32),
        "objects": f["objects"][idx].astype(np.float32),
        "object_mask": f["object_mask"][idx].astype(np.float32),
        "future_xy": f["future_xy"][idx].astype(np.float32),
        "future_v": f["future_v"][idx].astype(np.float32),
        "future_mask": f["future_mask"][idx].astype(np.float32),
    }


def normalize_sample(sample: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    """
    Normalize a sample using norms.py functions.
    
    Args:
        sample: Dict with numpy arrays
        
    Returns:
        Dict with normalized torch tensors
    """
    # Convert to torch tensors
    ego_vec = torch.from_numpy(sample["ego_vec"].astype(np.float32))
    bev = torch.from_numpy(sample["bev"].astype(np.float32))
    route = torch.from_numpy(sample["route"].astype(np.float32))
    objects = torch.from_numpy(sample["objects"].astype(np.float32))
    object_mask = torch.from_numpy(sample["object_mask"].astype(np.float32))
    future_xy = torch.from_numpy(sample["future_xy"].astype(np.float32))
    future_v = torch.from_numpy(sample["future_v"].astype(np.float32))
    future_mask = torch.from_numpy(sample["future_mask"].astype(np.float32))
    
    # Normalize (ego_vec is already normalized during collection)
    bev_norm = normalize_bev(bev)
    route_norm = normalize_route_points(route)
    objects_norm = normalize_object_tokens(objects)
    future_xy_norm, future_v_norm = normalize_futures(future_xy, future_v)
    
    return {
        "frame_id": sample["frame_id"],
        "episode_id": sample["episode_id"],
        "ego_vec": ego_vec,
        "bev": bev_norm,
        "route": route_norm,
        "objects": objects_norm,
        "object_mask": object_mask,
        "future_xy": future_xy_norm,
        "future_v": future_v_norm,
        "future_mask": future_mask,
    }


def print_statistics(name: str, tensor: torch.Tensor, indent: int = 0):
    """Print statistics for a tensor."""
    indent_str = "  " * indent
    if tensor.numel() == 0:
        print(f"{indent_str}{name}: empty")
        return
    
    print(f"{indent_str}{name}:")
    print(f"{indent_str}  shape: {tuple(tensor.shape)}")
    print(f"{indent_str}  min: {tensor.min().item():.6f}")
    print(f"{indent_str}  max: {tensor.max().item():.6f}")
    print(f"{indent_str}  mean: {tensor.float().mean().item():.6f}")
    print(f"{indent_str}  std: {tensor.float().std().item():.6f}")
    
    # Print sample values for small tensors
    if tensor.numel() <= 20:
        print(f"{indent_str}  values: {tensor.tolist()}")
    elif tensor.dim() == 1 and tensor.numel() <= 50:
        print(f"{indent_str}  values: {tensor.tolist()}")
    elif tensor.dim() == 2 and tensor.shape[0] <= 5:
        print(f"{indent_str}  values (first {tensor.shape[0]} rows):")
        for i in range(tensor.shape[0]):
            print(f"{indent_str}    [{i}]: {tensor[i].tolist()}")


def inspect_hdf5_dataset(input_dir: str, num_steps: int = 100):
    """
    Read and normalize first N steps from HDF5 dataset.
    
    Args:
        input_dir: Directory containing .h5 files
        num_steps: Number of steps to read and inspect
    """
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Find all HDF5 files
    h5_files = sorted(glob.glob(str(input_dir / "*.h5")))
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found in {input_dir}")
    
    print("=" * 80)
    print("HDF5 Normalized Values Inspector")
    print("=" * 80)
    print(f"Input directory: {input_dir}")
    print(f"Found {len(h5_files)} HDF5 file(s)")
    print(f"Reading first {num_steps} steps")
    print("=" * 80)
    
    samples_read = 0
    normalized_samples = []
    
    # Read samples from files
    for file_idx, h5_path in enumerate(h5_files):
        if samples_read >= num_steps:
            break
        
        print(f"\nReading from file {file_idx + 1}/{len(h5_files)}: {Path(h5_path).name}")
        
        with h5py.File(h5_path, "r") as f:
            n_samples = len(f["frame_id"])
            print(f"  File contains {n_samples} samples")
            
            # Read up to num_steps samples
            samples_to_read = min(n_samples, num_steps - samples_read)
            
            for local_idx in range(samples_to_read):
                sample = read_hdf5_sample(f, local_idx)
                normalized = normalize_sample(sample)
                normalized_samples.append(normalized)
                samples_read += 1
                
                if samples_read >= num_steps:
                    break
    
    print(f"\n{'=' * 80}")
    print(f"Read {len(normalized_samples)} samples")
    print(f"{'=' * 80}\n")
    
    # Print statistics for all samples
    print("GLOBAL STATISTICS (across all samples):")
    print("-" * 80)
    
    # Aggregate statistics
    ego_vecs = torch.stack([s["ego_vec"] for s in normalized_samples])
    bevs = torch.stack([s["bev"] for s in normalized_samples])
    routes = torch.stack([s["route"] for s in normalized_samples])
    objects = torch.stack([s["objects"] for s in normalized_samples])
    future_xys = torch.stack([s["future_xy"] for s in normalized_samples])
    future_vs = torch.stack([s["future_v"] for s in normalized_samples])
    
    print_statistics("ego_vec", ego_vecs)
    print()
    print_statistics("bev", bevs)
    print()
    print_statistics("route", routes)
    print()
    print_statistics("objects", objects)
    print()
    print_statistics("future_xy", future_xys)
    print()
    print_statistics("future_v", future_vs)
    
    # Distribution analysis for key metrics
    print(f"\n{'=' * 80}")
    print("DISTRIBUTION ANALYSIS:")
    print(f"{'=' * 80}\n")
    
    # Extract key metrics from ego_vec
    speeds = ego_vecs[:, 0]  # speed_mps/V_MAX
    yaw_rates = ego_vecs[:, 1]  # yaw_rate/YAW_RATE_MAX
    curvatures = ego_vecs[:, 4]  # curvature/CURVATURE_MAX
    commands = ego_vecs[:, 10]  # command/5.0
    
    # Speed distribution
    print("Speed Distribution (normalized, 0-1):")
    print(f"  Min: {speeds.min().item():.6f} ({speeds.min().item() * 50.0:.2f} m/s)")
    print(f"  Max: {speeds.max().item():.6f} ({speeds.max().item() * 50.0:.2f} m/s)")
    print(f"  Mean: {speeds.mean().item():.6f} ({speeds.mean().item() * 50.0:.2f} m/s)")
    print(f"  Std: {speeds.std().item():.6f} ({speeds.std().item() * 50.0:.2f} m/s)")
    print(f"  Median: {torch.median(speeds).item():.6f} ({torch.median(speeds).item() * 50.0:.2f} m/s)")
    
    # Speed bins
    speed_bins = torch.linspace(0, 1, 11)  # 0.0 to 1.0 in 0.1 increments
    speed_hist = torch.histc(speeds, bins=10, min=0, max=1)
    print("  Histogram (0.0-1.0 in 0.1 bins):")
    for i in range(10):
        count = int(speed_hist[i].item())
        pct = (count / len(speeds)) * 100
        bar = "█" * int(pct / 2)  # Visual bar
        print(f"    [{speed_bins[i]:.1f}-{speed_bins[i+1]:.1f}): {count:4d} ({pct:5.1f}%) {bar}")
    print()
    
    # Curvature distribution
    print("Curvature Distribution (normalized, -1 to 1):")
    print(f"  Min: {curvatures.min().item():.6f} ({curvatures.min().item() * 1.0:.4f} 1/m)")
    print(f"  Max: {curvatures.max().item():.6f} ({curvatures.max().item() * 1.0:.4f} 1/m)")
    print(f"  Mean: {curvatures.mean().item():.6f} ({curvatures.mean().item() * 1.0:.4f} 1/m)")
    print(f"  Std: {curvatures.std().item():.6f} ({curvatures.std().item() * 1.0:.4f} 1/m)")
    print(f"  Median: {torch.median(curvatures).item():.6f} ({torch.median(curvatures).item() * 1.0:.4f} 1/m)")
    print(f"  Abs mean: {curvatures.abs().mean().item():.6f} ({curvatures.abs().mean().item() * 1.0:.4f} 1/m)")
    
    # Curvature bins (centered around 0)
    curv_bins = torch.linspace(-1, 1, 11)  # -1.0 to 1.0 in 0.2 increments
    curv_hist = torch.histc(curvatures, bins=10, min=-1, max=1)
    print("  Histogram (-1.0 to 1.0 in 0.2 bins):")
    for i in range(10):
        count = int(curv_hist[i].item())
        pct = (count / len(curvatures)) * 100
        bar = "█" * int(pct / 2)
        print(f"    [{curv_bins[i]:.1f}-{curv_bins[i+1]:.1f}): {count:4d} ({pct:5.1f}%) {bar}")
    print()
    
    # Command distribution
    print("Command Distribution:")
    # Commands are stored as normalized (command/5.0), so multiply by 5 to get actual command
    commands_int = (commands * 5.0).round().long()
    unique_commands, counts = torch.unique(commands_int, return_counts=True)
    command_names = {
        0: "LaneFollow",
        1: "Left", 
        2: "Right",
        3: "Straight",
        4: "ChangeLaneLeft",
        5: "ChangeLaneRight"
    }
    print("  Command counts:")
    for cmd_int, count in zip(unique_commands.tolist(), counts.tolist()):
        cmd_name = command_names.get(cmd_int, f"Unknown({cmd_int})")
        pct = (count / len(commands)) * 100
        bar = "█" * int(pct / 2)
        print(f"    {cmd_int} ({cmd_name:20s}): {count:4d} ({pct:5.1f}%) {bar}")
    print()
    
    # Yaw rate distribution
    print("Yaw Rate Distribution (normalized, -1 to 1):")
    print(f"  Min: {yaw_rates.min().item():.6f} ({yaw_rates.min().item() * 10.0:.4f} rad/s)")
    print(f"  Max: {yaw_rates.max().item():.6f} ({yaw_rates.max().item() * 10.0:.4f} rad/s)")
    print(f"  Mean: {yaw_rates.mean().item():.6f} ({yaw_rates.mean().item() * 10.0:.4f} rad/s)")
    print(f"  Std: {yaw_rates.std().item():.6f} ({yaw_rates.std().item() * 10.0:.4f} rad/s)")
    print(f"  Abs mean: {yaw_rates.abs().mean().item():.6f} ({yaw_rates.abs().mean().item() * 10.0:.4f} rad/s)")
    print()
    
    # Future trajectory variation analysis
    print("Future Trajectory Variation Analysis:")
    future_variations_x = []
    future_variations_y = []
    future_variations_speed = []
    
    for sample in normalized_samples:
        future_xy = sample["future_xy"]
        future_v = sample["future_v"]
        
        if future_xy.shape[0] > 1:
            # Calculate step-to-step differences
            diffs_x = future_xy[1:, 0] - future_xy[:-1, 0]
            diffs_y = future_xy[1:, 1] - future_xy[:-1, 1]
            diffs_v = future_v[1:] - future_v[:-1]
            
            future_variations_x.append(diffs_x.abs().mean().item())
            future_variations_y.append(diffs_y.abs().mean().item())
            future_variations_speed.append(diffs_v.abs().mean().item())
    
    if future_variations_x:
        var_x_tensor = torch.tensor(future_variations_x)
        var_y_tensor = torch.tensor(future_variations_y)
        var_v_tensor = torch.tensor(future_variations_speed)
        
        print(f"  Mean step-to-step variation in X: {var_x_tensor.mean().item():.6f}")
        print(f"  Mean step-to-step variation in Y: {var_y_tensor.mean().item():.6f}")
        print(f"  Mean step-to-step variation in speed: {var_v_tensor.mean().item():.6f}")
        print(f"  Samples with very low X variation (<0.0001): {(var_x_tensor < 0.0001).sum().item()}/{len(var_x_tensor)}")
        print(f"  Samples with very low Y variation (<0.00001): {(var_y_tensor < 0.00001).sum().item()}/{len(var_y_tensor)}")
    print()
    
    # Print detailed sample values for first few samples
    print(f"\n{'=' * 80}")
    print(f"DETAILED SAMPLE VALUES (first 5 samples):")
    print(f"{'=' * 80}\n")
    
    for sample_idx in range(min(5, len(normalized_samples))):
        sample = normalized_samples[sample_idx]
        print(f"Sample {sample_idx + 1}:")
        print(f"  frame_id: {sample['frame_id']}")
        print(f"  episode_id: {sample['episode_id']}")
        print()
        
        print("  ego_vec (already normalized during collection):")
        ego_vec = sample["ego_vec"]
        ego_vec_names = [
            "speed_mps/V_MAX", "yaw_rate/YAW_RATE_MAX", "accel_long/ACCEL_MAX",
            "accel_lat/ACCEL_MAX", "curvature/CURVATURE_MAX", "steer_angle_rad/π",
            "steer_norm", "throttle", "brake", "speed_limit_mps/SPEED_LIMIT_MAX",
            "command/5.0", "gear/6.0", "time_of_day_sin", "time_of_day_cos"
        ]
        for i, name in enumerate(ego_vec_names):
            print(f"    [{i:2d}] {name:30s}: {ego_vec[i].item():.6f}")
        print()
        
        print("  route (normalized):")
        route = sample["route"]
        print(f"    shape: {tuple(route.shape)}")
        print(f"    first 5 waypoints:")
        for i in range(min(5, route.shape[0])):
            print(f"      [{i}]: x={route[i, 0].item():.6f}, y={route[i, 1].item():.6f}")
        print()
        
        print("  future_xy (normalized):")
        future_xy = sample["future_xy"]
        print(f"    shape: {tuple(future_xy.shape)}")
        print(f"    first 5 waypoints:")
        for i in range(min(5, future_xy.shape[0])):
            print(f"      [{i}]: x={future_xy[i, 0].item():.6f}, y={future_xy[i, 1].item():.6f}")
        print()
        
        print("  future_v (normalized):")
        future_v = sample["future_v"]
        print(f"    shape: {tuple(future_v.shape)}")
        print(f"    values: {future_v.tolist()}")
        print()
        
        print("  objects (normalized, first 3 valid objects):")
        objects = sample["objects"]
        object_mask = sample["object_mask"]
        valid_objects = torch.nonzero(object_mask > 0.5, as_tuple=False).squeeze(-1)
        print(f"    shape: {tuple(objects.shape)}")
        print(f"    valid objects: {len(valid_objects)}")
        if len(valid_objects) > 0:
            for i, obj_idx in enumerate(valid_objects[:3]):
                obj = objects[obj_idx]
                print(f"    object {obj_idx}:")
                print(f"      type_id: {obj[0].item():.0f}")
                print(f"      x_ego: {obj[1].item():.6f}")
                print(f"      y_ego: {obj[2].item():.6f}")
                print(f"      sin_yaw: {obj[3].item():.6f}")
                print(f"      cos_yaw: {obj[4].item():.6f}")
                print(f"      length: {obj[5].item():.6f}")
                print(f"      width: {obj[6].item():.6f}")
                print(f"      vx: {obj[7].item():.6f}")
                print(f"      vy: {obj[8].item():.6f}")
                print(f"      oncoming_flag: {obj[9].item():.0f}")
                print(f"      priority_flag: {obj[10].item():.0f}")
        print()
        
        print("  bev (normalized, channel statistics):")
        bev = sample["bev"]
        print(f"    shape: {tuple(bev.shape)}")
        print(f"    channel statistics:")
        for c in range(bev.shape[0]):
            channel = bev[c]
            print(f"      channel {c:2d}: min={channel.min().item():.6f}, "
                  f"max={channel.max().item():.6f}, mean={channel.float().mean().item():.6f}")
        print()
        
        print("-" * 80)
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Inspect normalized values from HDF5 BC dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory containing .h5 files",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=100,
        help="Number of steps to read and inspect",
    )
    
    args = parser.parse_args()
    
    if not HDF5_AVAILABLE:
        print("Error: h5py is required. Install with: pip install h5py hdf5plugin")
        sys.exit(1)
    
    try:
        inspect_hdf5_dataset(args.input_dir, args.num_steps)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

