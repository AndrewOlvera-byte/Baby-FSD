"""
Lightweight validator for HDF5 BC runs.

Loads a few batches through the BC DataLoader to catch schema/shape/range issues.
"""

import argparse
import sys
import torch


def validate(run_dir: str, batch_size: int, num_batches: int, num_workers: int,
             future_horizon: int, route_points: int, max_objects: int) -> None:
    try:
        from data.torch_dataset import create_bc_dataloader
    except ImportError as e:
        print(f"Import error: {e}. Ensure h5py is installed in the trainer env.")
        sys.exit(1)

    print(f"[validate_bc_run] Loading run_dir={run_dir}")
    loader = create_bc_dataloader(
        run_dir=run_dir,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        future_horizon=future_horizon,
        route_points=route_points,
        max_objects=max_objects,
        pin_memory=False,
        persistent_workers=False,
    )

    total = len(loader.dataset)
    print(f"[validate_bc_run] Dataset size: {total} samples")

    seen = 0
    for bidx, batch in enumerate(loader):
        print(f"\nBatch {bidx} shapes:")
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k:12s} {tuple(v.shape)} "
                      f"range=[{v.min().item():.3f}, {v.max().item():.3f}]")

        # Basic mask checks
        obj_mask_vals = torch.unique(batch["object_mask"])
        fut_mask_vals = torch.unique(batch["future_mask"])
        print(f"  object_mask unique: {obj_mask_vals.tolist()}")
        print(f"  future_mask unique: {fut_mask_vals.tolist()}")

        # Sanity: all tensors finite
        for k, v in batch.items():
            if isinstance(v, torch.Tensor) and not torch.isfinite(v).all():
                raise ValueError(f"{k} contains NaN/Inf")

        seen += batch["ego_vec"].shape[0]
        if bidx + 1 >= num_batches:
            break

    print(f"\n[validate_bc_run] OK: inspected {min(seen, total)} / {total} samples")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Path to BC run directory (contains .h5 files)")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_batches", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--future_horizon", type=int, default=12)
    ap.add_argument("--route_points", type=int, default=32)
    ap.add_argument("--max_objects", type=int, default=64)
    args = ap.parse_args()

    validate(
        run_dir=args.run_dir,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        num_workers=args.num_workers,
        future_horizon=args.future_horizon,
        route_points=args.route_points,
        max_objects=args.max_objects,
    )


if __name__ == "__main__":
    main()
