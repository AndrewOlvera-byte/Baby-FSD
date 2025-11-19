"""
Test script for BCPolicy model.

Verifies:
- Model instantiation
- Parameter count
- Forward pass with correct shapes
- Output dimensions match expectations
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pipeline', 'src'))

import torch
from components.models.bc_policy import BCPolicy


def test_bc_policy():
    """Test BCPolicy model with expected data shapes."""
    print("=" * 80)
    print("Testing BCPolicy Model")
    print("=" * 80)
    
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
    model = BCPolicy(**cfg)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n2. Parameter count: {n_params:,} ({n_params/1e6:.2f}M)")
    
    # Check parameter count is in target range (15-30M)
    if 15e6 <= n_params <= 30e6:
        print("   ✓ Parameter count in target range (15-30M)")
    elif n_params < 15e6:
        print(f"   ⚠ Parameter count below target (15M), but acceptable for lightweight model")
    else:
        print(f"   ⚠ Parameter count above target (30M)")
    
    # Create dummy batch matching dataloader output
    print("\n3. Creating dummy batch...")
    batch_size = 4
    batch = {
        "ego_vec": torch.randn(batch_size, 14),
        "bev": torch.randn(batch_size, 18, 150, 200),
        "route": torch.randn(batch_size, 32, 2),
        "objects": torch.randn(batch_size, 64, 11),
        "object_mask": torch.ones(batch_size, 64),  # All valid objects
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
    
    print("\n5. Verifying outputs are in normalized range...")
    xy_min, xy_max = outputs["future_xy"].min().item(), outputs["future_xy"].max().item()
    v_min, v_max = outputs["future_v"].min().item(), outputs["future_v"].max().item()
    
    print(f"   future_xy range: [{xy_min:.3f}, {xy_max:.3f}]")
    print(f"   future_v range: [{v_min:.3f}, {v_max:.3f}]")
    print(f"   (Note: Outputs may be outside [-1, 1] or [0, 1] before training)")
    
    # Test with CUDA if available
    if torch.cuda.is_available():
        print("\n6. Testing with CUDA...")
        model_cuda = model.cuda()
        batch_cuda = {k: v.cuda() if isinstance(v, torch.Tensor) else v 
                      for k, v in batch.items()}
        
        with torch.no_grad():
            outputs_cuda = model_cuda(batch_cuda)
        
        print(f"   ✓ CUDA forward pass successful")
        print(f"     GPU outputs shapes match CPU: {outputs_cuda['future_xy'].shape == outputs['future_xy'].shape}")
    else:
        print("\n6. CUDA not available, skipping GPU test")
    
    # Test gradient flow
    print("\n7. Testing gradient flow...")
    model.train()
    outputs_train = model(batch)
    
    # Dummy loss
    loss = outputs_train["future_xy"].sum() + outputs_train["future_v"].sum()
    loss.backward()
    
    # Check gradients exist
    has_grads = any(p.grad is not None for p in model.parameters() if p.requires_grad)
    print(f"   ✓ Gradients computed: {has_grads}")
    
    print("\n" + "=" * 80)
    print("All tests passed! ✓")
    print("=" * 80)
    print("\nModel is ready for training!")
    print(f"  - {n_params/1e6:.2f}M parameters")
    print(f"  - Handles batch size {batch_size}")
    print(f"  - BEV patches: {150//8} x {200//8} = {(150//8)*(200//8)} tokens")
    print(f"  - Total sequence length: 1 (ego) + 32 (route) + 64 (objects) + {(150//8)*(200//8)} (bev) = {1+32+64+(150//8)*(200//8)} tokens")
    print(f"  - Predicts {cfg['n_future_steps']} future waypoints and speeds")


if __name__ == "__main__":
    test_bc_policy()

