"""
Example usage of PyTorch DataLoader for BC trajectories.

This demonstrates how to load and iterate through batched BC data for training.
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.torch_dataset import create_bc_dataloader


def main():
    # Find a BC run directory
    base_dir = os.path.join("data", "BC_v1")
    
    if not os.path.isdir(base_dir):
        print(f"No BC data found at {base_dir}")
        print("Run collection first: python -m collect.collect_bc --config configs/collect_bc.yaml")
        return
    
    # Find first run
    run_dirs = [
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if d.startswith("run-") and os.path.isdir(os.path.join(base_dir, d))
    ]
    
    if not run_dirs:
        print("No run directories found")
        return
    
    run_dir = run_dirs[0]
    print(f"Loading data from: {run_dir}")
    
    # Create dataloader
    loader = create_bc_dataloader(
        run_dir=run_dir,
        batch_size=8,
        shuffle=True,
        num_workers=2,  # Use multiple workers for faster loading
        future_horizon=12,
        route_points=32,
    )
    
    print(f"\nDataset size: {len(loader.dataset)} samples")
    print(f"Number of batches: {len(loader)}")
    
    # Iterate through a few batches
    print("\n=== Sample Batches ===")
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= 3:
            break
        
        print(f"\nBatch {batch_idx + 1}:")
        print(f"  BEV shape: {batch['bev'].shape}")
        print(f"  Route shape: {batch['route'].shape}")
        print(f"  Futures shape: {batch['futures'].shape}")
        print(f"  Futures speed shape: {batch['futures_speed'].shape}")
        print(f"  Control shape: {batch['control'].shape}")
        print(f"  State shape: {batch['state'].shape}")
        
        # Show sample values from first item in batch
        print(f"  Sample control (steer, throttle, brake): {batch['control'][0].tolist()}")
        print(f"  Sample state (speed, yaw_rate, ...): {batch['state'][0][:2].tolist()}")
        print(f"  BEV value range: [{batch['bev'].min():.3f}, {batch['bev'].max():.3f}]")
    
    print("\n✓ DataLoader working correctly!")
    
    # Example: Simple training loop skeleton
    print("\n=== Training Loop Example ===")
    print("""
    # Pseudo-code for training:
    
    model = MyBCModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(num_epochs):
        for batch in loader:
            # Forward pass
            pred_control = model(
                bev=batch['bev'],
                route=batch['route'],
                state=batch['state'],
            )
            
            # Loss (e.g., MSE for continuous control)
            loss = F.mse_loss(pred_control, batch['control'])
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    """)


if __name__ == "__main__":
    main()

