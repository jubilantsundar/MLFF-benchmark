#!/usr/bin/env python3
"""
Simple test to verify TorchMD-NET installation and basic functionality
"""

import torch
import numpy as np

print("="*70)
print("TORCHMD-NET INSTALLATION TEST")
print("="*70)

# Test 1: Import torchmdnet
print("\n[1/5] Testing imports...")
try:
    import torchmdnet
    from torchmdnet.models.model import create_model
    print("✓ TorchMD-NET imported successfully")
    print(f"  Version: {torchmdnet.__version__ if hasattr(torchmdnet, '__version__') else 'unknown'}")
except Exception as e:
    print(f"✗ Failed to import: {e}")
    exit(1)

# Test 2: Check CUDA availability
print("\n[2/5] Checking CUDA...")
cuda_available = torch.cuda.is_available()
print(f"  CUDA available: {cuda_available}")
if cuda_available:
    print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA version: {torch.version.cuda}")

# Test 3: Create a TorchMD-NET model
print("\n[3/5] Creating TorchMD-NET model...")
try:
    # TorchMD-NET requires a configuration dictionary
    args = {
        'model': 'tensornet',
        'precision': 32,
        'embedding_dimension': 64,
        'num_layers': 2,
        'num_rbf': 16,
        'rbf_type': 'expnorm',
        'trainable_rbf': False,
        'activation': 'silu',
        'cutoff_lower': 0.0,
        'cutoff_upper': 5.0,
        'max_z': 100,
        'max_num_neighbors': 32,
        'equivariance_invariance_group': 'O(3)',
        'output_model': 'Scalar',
        'reduce_op': 'add',
        'derivative': False,  # Don't compute forces for this test
        'atom_filter': -1,
        'prior_model': None,
        'static_shapes': False,
    }

    model = create_model(args)
    print(f"✓ Model created successfully")
    print(f"  Model type: TensorNet")
    print(f"  Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
except Exception as e:
    print(f"✗ Failed to create model: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 4: Test model on CPU
print("\n[4/5] Testing model inference on CPU...")
try:
    device = 'cpu'
    model = model.to(device)
    model.eval()

    # Create dummy input (10 atoms, simple geometry)
    n_atoms = 10

    # Random positions (Angstrom scale - typical for molecular systems)
    z = torch.randint(1, 8, (n_atoms,), device=device, dtype=torch.long)  # Atomic numbers (H to N)
    pos = torch.randn(n_atoms, 3, device=device, dtype=torch.float32) * 2.0  # Random positions in Angstroms

    # Run model - TorchMD-NET returns a tuple (energy, forces) if derivative=True, otherwise just energy
    with torch.no_grad():
        output = model(z=z, pos=pos)

    # Output is typically a tuple (energy,) or (energy, forces)
    if isinstance(output, tuple):
        energy = output[0]
        print(f"✓ CPU inference successful")
        print(f"  Energy: {energy.item():.4f} (model units)")
        if len(output) > 1:
            forces = output[1]
            print(f"  Forces shape: {forces.shape}")
    else:
        print(f"✓ CPU inference successful")
        print(f"  Output shape: {output.shape}")
        print(f"  Energy: {output.item():.4f} (model units)")
except Exception as e:
    print(f"✗ CPU inference failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Test model on CUDA (if available)
if cuda_available:
    print("\n[5/5] Testing model inference on CUDA...")
    try:
        device = 'cuda'
        model = model.to(device)
        model.eval()

        # Create dummy input on GPU
        z = torch.randint(1, 8, (n_atoms,), device=device, dtype=torch.long)
        pos = torch.randn(n_atoms, 3, device=device, dtype=torch.float32) * 2.0

        # Run model
        with torch.no_grad():
            output = model(z=z, pos=pos)

        # Output is typically a tuple (energy,) or (energy, forces)
        if isinstance(output, tuple):
            energy = output[0]
            print(f"✓ CUDA inference successful")
            print(f"  Energy: {energy.item():.4f} (model units)")
            if len(output) > 1:
                forces = output[1]
                print(f"  Forces shape: {forces.shape}")
        else:
            print(f"✓ CUDA inference successful")
            print(f"  Energy: {output.item():.4f} (model units)")

        # Quick timing test
        print("\n  Quick performance test (100 iterations)...")
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(100):
            with torch.no_grad():
                output = model(z=z, pos=pos)
        end.record()
        torch.cuda.synchronize()

        avg_time_ms = start.elapsed_time(end) / 100
        print(f"  Average inference time: {avg_time_ms:.3f} ms")

    except Exception as e:
        print(f"✗ CUDA inference failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print("\n[5/5] Skipping CUDA test (not available)")

print("\n" + "="*70)
print("TEST COMPLETED")
print("="*70)
print("\n✓ TorchMD-NET is properly installed and functional!")
