#!/usr/bin/env python3
"""
TorchMD-NET + OpenMM Integration Demo
Demonstrates how to use TorchMD-NET as a force provider in OpenMM simulations

Note: This uses an untrained model for demonstration. In production,
you would load a pre-trained checkpoint.
"""

# CRITICAL: Import torch BEFORE OpenMM to avoid library conflicts
import torch
from torchmdnet.models.model import create_model
from openmmtorch import TorchForce

# Now import OpenMM
try:
    from openmm import *
    from openmm.app import *
    from openmm.unit import *
except ImportError:
    from simtk.openmm import *
    from simtk.openmm.app import *
    from simtk.unit import *

import sys
import numpy as np


class TorchMDNetWrapper(torch.nn.Module):
    """
    Wrapper that adapts TorchMD-NET for use with OpenMM via TorchForce.

    OpenMM provides positions in nanometers and expects energies in kJ/mol.
    TorchMD-NET expects positions in Angstroms and outputs energies in eV.
    """

    def __init__(self, atomic_numbers, model_args):
        super(TorchMDNetWrapper, self).__init__()

        # Store atomic numbers as a buffer (non-trainable parameter)
        self.register_buffer('atomic_numbers',
                           torch.tensor(atomic_numbers, dtype=torch.long))

        # Create TorchMD-NET model
        # In production, you would load a trained model here
        self.model = create_model(model_args)
        self.model.eval()  # Set to evaluation mode

    def forward(self, positions):
        """
        Forward pass for OpenMM integration.

        Args:
            positions: Tensor of shape (n_atoms, 3) in nanometers

        Returns:
            energy: Scalar tensor in kJ/mol
        """
        # Convert positions from nanometers to Angstroms
        positions_angstrom = positions * 10.0

        # Ensure correct dtype
        positions_angstrom = positions_angstrom.to(torch.float32)

        # Run TorchMD-NET model
        # Output is tuple (energy,) in eV
        output = self.model(z=self.atomic_numbers, pos=positions_angstrom)
        energy_ev = output[0]  # First element is energy

        # Convert energy from eV to kJ/mol
        # 1 eV = 96.4916 kJ/mol
        energy_kj_mol = energy_ev * 96.4916

        return energy_kj_mol


def create_torchmdnet_system(pdb_file, platform_name='CPU', save_model_script=False):
    """
    Create an OpenMM system with TorchMD-NET forces.

    Args:
        pdb_file: Path to PDB file
        platform_name: 'CPU' or 'CUDA'
        save_model_script: If True, save TorchScript model to file

    Returns:
        simulation: OpenMM Simulation object with TorchMD-NET forces
        wrapper: The TorchMDNetWrapper model
    """
    print(f"\n{'='*70}")
    print("TorchMD-NET + OpenMM Integration")
    print(f"{'='*70}\n")

    # Load structure
    print(f"Loading structure: {pdb_file}")
    pdb = PDBFile(pdb_file)
    topology = pdb.topology
    positions = pdb.positions

    n_atoms = topology.getNumAtoms()
    n_residues = topology.getNumResidues()
    print(f"System: {n_atoms} atoms, {n_residues} residues\n")

    # Extract atomic numbers from topology
    atomic_numbers = []
    for atom in topology.atoms():
        atomic_numbers.append(atom.element.atomic_number)

    print(f"Atomic numbers range: {min(atomic_numbers)}-{max(atomic_numbers)}")
    print(f"Unique elements: {len(set(atomic_numbers))}\n")

    # Configure TorchMD-NET model
    print("Creating TorchMD-NET model...")
    model_args = {
        'model': 'tensornet',
        'precision': 32,
        'embedding_dimension': 128,
        'num_layers': 2,
        'num_rbf': 32,
        'rbf_type': 'expnorm',
        'trainable_rbf': False,
        'activation': 'silu',
        'cutoff_lower': 0.0,
        'cutoff_upper': 5.0,  # 5 Angstrom cutoff
        'max_z': max(atomic_numbers) + 10,
        'max_num_neighbors': 64,
        'equivariance_invariance_group': 'O(3)',
        'output_model': 'Scalar',
        'reduce_op': 'add',
        'derivative': False,  # OpenMM computes forces via autograd
        'atom_filter': -1,
        'prior_model': None,
        'static_shapes': False,
    }

    # Create wrapper
    wrapper = TorchMDNetWrapper(atomic_numbers, model_args)
    param_count = sum([p.numel() for p in wrapper.parameters()])
    print(f"✓ Model created ({param_count:,} parameters)")
    print(f"  Architecture: TensorNet")
    print(f"  Cutoff: {model_args['cutoff_upper']} Å")
    print(f"  Embedding dim: {model_args['embedding_dimension']}\n")

    # Convert to TorchScript for OpenMM compatibility
    print("Converting to TorchScript...")
    wrapper_scripted = torch.jit.script(wrapper)

    if save_model_script:
        model_file = 'torchmdnet_wrapper.pt'
        wrapper_scripted.save(model_file)
        print(f"✓ Model saved to {model_file}\n")
    else:
        print("✓ TorchScript conversion complete\n")

    # Create OpenMM system
    print("Creating OpenMM system...")
    system = System()

    # Add particles
    for atom in topology.atoms():
        system.addParticle(atom.element.mass)

    # Add TorchMD-NET force
    torch_force = TorchForce(wrapper_scripted)
    system.addForce(torch_force)
    print(f"✓ Added TorchForce to system\n")

    # Create integrator
    temperature = 300 * kelvin
    friction = 1.0 / picosecond
    timestep = 2.0 * femtosecond
    integrator = LangevinMiddleIntegrator(temperature, friction, timestep)

    # Create simulation
    platform = Platform.getPlatformByName(platform_name)
    properties = {}
    if platform_name == 'CUDA':
        properties['Precision'] = 'mixed'

    simulation = Simulation(topology, system, integrator, platform, properties)
    simulation.context.setPositions(positions)

    print(f"✓ Simulation created")
    print(f"  Platform: {platform_name}")
    print(f"  Temperature: {temperature}")
    print(f"  Timestep: {timestep}")

    return simulation, wrapper


def run_demo_simulation(pdb_file, platform='CPU', n_steps=10):
    """
    Run a short demonstration simulation with TorchMD-NET forces.

    Args:
        pdb_file: Path to PDB file
        platform: 'CPU' or 'CUDA'
        n_steps: Number of MD steps to run
    """
    # Create system
    simulation, wrapper = create_torchmdnet_system(pdb_file, platform)

    # Add reporter
    print(f"\n{'='*70}")
    print(f"Running {n_steps} MD steps with TorchMD-NET forces")
    print(f"{'='*70}\n")

    simulation.reporters.append(
        StateDataReporter(sys.stdout, 1, step=True,
                         potentialEnergy=True, temperature=True,
                         speed=True)
    )

    # Run simulation
    simulation.step(n_steps)

    print(f"\n{'='*70}")
    print("Simulation Complete!")
    print(f"{'='*70}\n")

    # Get final state
    state = simulation.context.getState(getEnergy=True, getPositions=True)
    final_energy = state.getPotentialEnergy()

    print(f"Final potential energy: {final_energy}")
    print(f"\nNote: This energy is from an UNTRAINED model.")
    print(f"For meaningful simulations, use a model trained on appropriate data.")

    return simulation


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='TorchMD-NET + OpenMM Integration Demo'
    )
    parser.add_argument('pdb_file', help='Input PDB file')
    parser.add_argument('--platform', default='CPU', choices=['CPU', 'CUDA'],
                       help='Simulation platform (default: CPU)')
    parser.add_argument('--steps', type=int, default=10,
                       help='Number of MD steps (default: 10)')
    parser.add_argument('--save-model', action='store_true',
                       help='Save TorchScript model to file')

    args = parser.parse_args()

    # Run demo
    run_demo_simulation(args.pdb_file, args.platform, args.steps)


if __name__ == '__main__':
    main()
