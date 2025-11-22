"""
Test script for BCPolicy model.

Run from project root: pytest tests/test_bc_model.py

Verifies:
- Model instantiation
- Parameter count
- Forward pass with correct shapes
- Output dimensions match expectations
"""

import sys
import os

# Ensure pipeline is on sys.path so `src.*` resolves
ROOT = os.path.dirname(os.path.dirname(__file__))
PIPELINE_DIR = os.path.join(ROOT, "pipeline")
if PIPELINE_DIR not in sys.path:
    sys.path.insert(0, PIPELINE_DIR)

# Bootstrap the pipeline registry first
from src.core.bootstrap import bootstrap
bootstrap()

import torch
from src.components.models.bc_policy import BCPolicy
from tests.utils.model_checks import (
    assert_no_nan_inf,
    assert_grad_norms_reasonable,
    step_loss_decreases,
)


def test_bc_policy():
    """Test BCPolicy model with expected data shapes."""
    print("=" * 80)
    print("Testing BCPolicy Model")
    print("=" * 80)

    # GPU-only: require CUDA and build everything on device to avoid CPU/GPU mismatch
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("CUDA is required for BCPolicy test (GPU-only)")
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    
    # Model config (matching bc_policy.yaml)
    cfg = {
        "d_model": 256,
        "n_heads": 8,
        "n_layers": 6,
        "dropout": 0.1,
        "patch_size": [8, 8],
        "ego_hidden": 128,
        "route_hidden": 128,
        "object_hidden": 128,
        "n_future_steps": 12,
        "n_object_types": 4,
    }
    
    print("\n1. Instantiating model...")
    model = BCPolicy(**cfg).to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n2. Parameter count: {n_params:,} ({n_params/1e6:.2f}M)")
    
    # Check parameter count
    if n_params < 15e6:
        print(f"   [OK] Lightweight model with {n_params/1e6:.2f}M parameters (below 15M target)")
        print(f"   [OK] This is good for fast iteration on small BC data and future RL")
    elif 15e6 <= n_params <= 30e6:
        print(f"   [OK] Parameter count in target range (15-30M)")
    else:
        print(f"   [WARN] Parameter count above target (30M)")
    
    # Create dummy batch matching dataloader output
    print("\n3. Creating dummy batch...")
    batch_size = 1  # keep small to reduce GPU mem during test
    
    # Create objects with proper type_id (integer 0-3)
    objects = torch.randn(batch_size, 64, 11, device=device)
    objects[:, :, 0] = torch.randint(0, cfg["n_object_types"], (batch_size, 64), device=device).float()  # Valid type_ids
    
    batch = {
        "ego_vec": torch.randn(batch_size, 14, device=device),
        "bev": torch.randn(batch_size, 18, 150, 200, device=device),
        "route": torch.randn(batch_size, 32, 2, device=device),
        "objects": objects,
        "object_mask": torch.ones(batch_size, 64, device=device),  # All valid objects
    }
    
    # Set some objects to be padded (mask = 0)
    batch["object_mask"][:, 50:] = 0  # Last 14 objects are padded
    
    print(f"   Batch shapes:")
    for key, val in batch.items():
        if isinstance(val, torch.Tensor):
            print(f"     {key}: {tuple(val.shape)}")
    
    # Forward pass
    print("\n4. Running forward pass...")
    model.eval()
    with torch.no_grad():
        outputs = model(batch)
    
    print(f"   Output shapes:")
    print(f"     future_xy: {tuple(outputs['future_xy'].shape)}")
    print(f"     future_v: {tuple(outputs['future_v'].shape)}")
    
    # Verify output shapes
    expected_xy_shape = (batch_size, cfg["n_future_steps"], 2)
    expected_v_shape = (batch_size, cfg["n_future_steps"])
    
    assert outputs["future_xy"].shape == expected_xy_shape, \
        f"Expected future_xy shape {expected_xy_shape}, got {outputs['future_xy'].shape}"
    assert outputs["future_v"].shape == expected_v_shape, \
        f"Expected future_v shape {expected_v_shape}, got {outputs['future_v'].shape}"
    
    print("   [OK] Output shapes are correct")
    
    print("\n5. Verifying outputs...")
    xy_min, xy_max = outputs["future_xy"].min().item(), outputs["future_xy"].max().item()
    v_min, v_max = outputs["future_v"].min().item(), outputs["future_v"].max().item()
    
    print(f"   future_xy range: [{xy_min:.3f}, {xy_max:.3f}]")
    print(f"   future_v range: [{v_min:.3f}, {v_max:.3f}]")
    print(f"   (Note: Outputs will be trained to match normalized data in [-1,1] or [0,1])")
    
    # Test gradient flow
    print("\n6. Testing gradient flow...")
    model.train()
    outputs_train = model(batch)
    
    # Dummy loss
    loss = outputs_train["future_xy"].sum() + outputs_train["future_v"].sum()
    loss.backward()
    
    # Check gradients exist
    has_grads = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    n_grads = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
    n_total = sum(1 for p in model.parameters() if p.requires_grad)
    
    print(f"   [OK] Gradients computed for {n_grads}/{n_total} parameter groups")
    assert has_grads, "Expected gradients on trainable params"
    assert_no_nan_inf(outputs_train, context="BCPolicy forward")
    assert_grad_norms_reasonable(model.parameters(), context="BCPolicy backward")

    # Tiny optimization loop: ensure loss can decrease on synthetic data
    def simple_loss_fn(out, tgt):
        return (out["future_xy"] - tgt["future_xy"]).pow(2).mean() + (out["future_v"] - tgt["future_v"]).pow(2).mean()

    init_loss, final_loss = step_loss_decreases(model, batch, simple_loss_fn, steps=10, lr=1e-3, device=device)
    print(f"   [OK] Loss decreased: {init_loss:.4f} -> {final_loss:.4f}")
    assert final_loss <= init_loss * 0.8, "Expected training loss to decrease in mini loop"
    
    # Compute some architecture stats
    print("\n7. Architecture statistics...")
    n_h, n_w = 150 // 8, 200 // 8
    n_bev_tokens = n_h * n_w
    n_ego_tokens = 1
    n_route_tokens = 32
    n_object_tokens = 64
    total_seq_len = n_ego_tokens + n_route_tokens + n_object_tokens + n_bev_tokens
    
    print(f"   BEV patches: {n_h} x {n_w} = {n_bev_tokens} tokens")
    print(f"   Total sequence length: {n_ego_tokens} (ego) + {n_route_tokens} (route) + {n_object_tokens} (objects) + {n_bev_tokens} (bev) = {total_seq_len} tokens")
    print(f"   Encoder processes {total_seq_len} context tokens with self-attention")
    print(f"   Decoder uses {cfg['n_future_steps']} query tokens with cross-attention")
    
    print("\n" + "=" * 80)
    print("All tests passed! [SUCCESS]")
    print("=" * 80)
    print("\nModel is ready for BC training!")
    print(f"  - {n_params/1e6:.2f}M parameters (lightweight, efficient)")
    print(f"  - Query-based planning architecture (SOTA-aligned)")
    print(f"  - Handles batch size {batch_size}")
    print(f"  - Predicts {cfg['n_future_steps']} future waypoints and speeds")
    print(f"  - Suitable for scaling to larger BC datasets and RL fine-tuning")


if __name__ == "__main__":
    test_bc_policy()

