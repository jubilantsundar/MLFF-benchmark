# TorchMD-NET + OpenMM Integration

Complete setup, testing, and benchmarking suite for using TorchMD-NET machine learning forcefields with OpenMM molecular dynamics simulations.

## 🎯 Overview

This repository contains a fully functional integration of TorchMD-NET (a state-of-the-art neural network forcefield) with OpenMM for running molecular dynamics simulations. It includes comprehensive benchmarks comparing traditional empirical forcefields (AMBER14) with the ML forcefield framework.

## ✨ Features

- **Complete TorchMD-NET Installation**: Compiled from source with CUDA support
- **OpenMM Integration**: Seamless integration using OpenMM-Torch plugin
- **Benchmark Suite**: Compare traditional vs ML forcefields
- **Unit Conversion**: Automatic handling of nm↔Å and kJ/mol↔eV
- **Multi-Platform**: Support for both CPU and CUDA execution
- **TorchScript**: JIT compilation for optimized performance

## 📊 Benchmark Results

Performance comparison on RNA hairpin structure (700 atoms, 1000 MD steps):

| Forcefield   | Platform | Performance (ns/day) | Speedup |
|--------------|----------|----------------------|---------|
| AMBER14      | CPU      | 133 ns/day          | 1.0x    |
| AMBER14      | CUDA     | **2,201 ns/day**    | 16.6x   |
| TorchMD-NET  | CPU      | 137 ns/day          | 1.0x    |
| TorchMD-NET  | CUDA     | **760 ns/day**      | 5.5x    |

See [`TorchMD-NET_Benchmark_Report.pdf`](TorchMD-NET_Benchmark_Report.pdf) for detailed visual analysis.

## 🚀 Quick Start

### Prerequisites

- CUDA-capable GPU (tested on Tesla T4)
- Python 3.13+
- PyTorch 2.7+ with CUDA
- OpenMM with OpenMM-Torch plugin

### Installation Test

```bash
# Test TorchMD-NET installation
python3 test_torchmdnet.py
```

### Run Demo Simulation

```bash
# CPU simulation (10 steps)
python3 torchmdnet_openmm_integration.py hairpin.pdb --platform CPU --steps 10

# CUDA simulation (100 steps)
python3 torchmdnet_openmm_integration.py hairpin.pdb --platform CUDA --steps 100
```

### Run Benchmarks

```bash
# Compare traditional and ML forcefields
python3 comprehensive_ff_benchmark.py hairpin.pdb --output-dir my_benchmark
```

## 📁 Repository Structure

```
.
├── README.md                              # This file
├── TorchMD-NET_Benchmark_Report.pdf       # Visual benchmark report (6 pages)
├── TORCHMDNET_SETUP_COMPLETE.md           # Complete setup documentation
├── TORCHMDNET_BENCHMARK_ANALYSIS.md       # Detailed benchmark analysis
├── test_torchmdnet.py                     # Installation validation tests
├── torchmdnet_openmm_integration.py       # OpenMM integration demo
├── comprehensive_ff_benchmark.py          # Multi-forcefield benchmark suite
├── generate_benchmark_pdf.py              # PDF report generator
├── hairpin.pdb                            # Example RNA hairpin structure
└── hairpin_ml_benchmark_final/            # Benchmark results
    ├── benchmark_results.json
    └── benchmark_summary.txt
```

## 🔧 System Requirements

### Hardware
- **GPU**: NVIDIA GPU with CUDA support (tested on Tesla T4)
- **RAM**: 16GB+ recommended
- **Storage**: 5GB for installation

### Software
- **OS**: Linux (tested on Ubuntu)
- **CUDA**: 12.6+
- **PyTorch**: 2.7.1+
- **TorchMD-NET**: 2.4.12
- **OpenMM**: Latest version
- **OpenMM-Torch**: 1.5+

## 📖 Documentation

### Quick Reference
- [`README.md`](README.md) - This file (quick start & overview)
- [`TORCHMDNET_SETUP_COMPLETE.md`](TORCHMDNET_SETUP_COMPLETE.md) - Complete setup guide
- [`TORCHMDNET_BENCHMARK_ANALYSIS.md`](TORCHMDNET_BENCHMARK_ANALYSIS.md) - Detailed analysis
- [`TorchMD-NET_Benchmark_Report.pdf`](TorchMD-NET_Benchmark_Report.pdf) - Visual summary

### Key Scripts

#### `test_torchmdnet.py`
Validates TorchMD-NET installation:
- Model creation
- CPU and CUDA inference
- Performance testing

#### `torchmdnet_openmm_integration.py`
Demonstrates OpenMM integration:
- Creates TorchMD-NET wrapper
- Sets up OpenMM simulation
- Runs MD with ML forces

Usage:
```bash
python3 torchmdnet_openmm_integration.py <pdb_file> [--platform CPU|CUDA] [--steps N]
```

#### `comprehensive_ff_benchmark.py`
Compares traditional and ML forcefields:
- AMBER14 (traditional)
- TorchMD-NET (ML framework)
- CPU and CUDA platforms

Usage:
```bash
python3 comprehensive_ff_benchmark.py <pdb_file> [--output-dir DIR]
```

## 🧬 Example: RNA Hairpin

The repository includes a test RNA hairpin structure (700 atoms, 43 residues) used for all benchmarks.

```bash
# Quick 10-step simulation
python3 torchmdnet_openmm_integration.py hairpin.pdb --platform CUDA --steps 10
```

Expected output:
```
TorchMD-NET + OpenMM Integration
System: 700 atoms, 43 residues
Model created (756,865 parameters)
Platform: CUDA
Running 10 MD steps...
✓ Simulation Complete!
```

## ⚙️ Technical Details

### Model Architecture
- **Type**: TensorNet (equivariant graph neural network)
- **Parameters**: ~750,000
- **Embedding Dimension**: 128
- **Layers**: 2
- **Cutoff**: 5.0 Å
- **Equivariance**: O(3) symmetry

### Integration Method
TorchMD-NET is integrated via OpenMM-Torch plugin:
1. Create TorchMD-NET model as `torch.nn.Module`
2. Wrap with unit conversion (nm↔Å, kJ/mol↔eV)
3. Compile to TorchScript
4. Add as `TorchForce` to OpenMM system
5. OpenMM computes forces via automatic differentiation

### Import Order (Critical!)
```python
# CORRECT: Import torch BEFORE OpenMM
import torch
from torchmdnet.models.model import create_model
from openmm import *  # Import OpenMM after torch

# WRONG: Will cause library conflicts
from openmm import *
import torch  # Too late!
```

## 🐛 Troubleshooting

### Import Error: Symbol not found
**Symptom**: `undefined symbol: _ZNK5torch8profiler...`

**Solution**: Import PyTorch before OpenMM (see "Import Order" above)

### Generator Type Error
**Symptom**: `object of type 'generator' has no len()`

**Solution**: Use list comprehension instead of generator:
```python
# Use this:
sum([p.numel() for p in model.parameters()])

# Not this:
sum(p.numel() for p in model.parameters())
```

### CUDA Out of Memory
**Solution**:
- Reduce `embedding_dimension`
- Reduce `num_layers`
- Reduce `max_num_neighbors`
- Use smaller system

## 📝 Notes

### Current Status
✅ **Framework Operational**: All components installed and tested
✅ **Integration Complete**: TorchMD-NET works with OpenMM
✅ **Benchmarks Run**: Performance data collected

⚠️ **Untrained Models**: Current benchmarks use untrained models for framework testing. Production use requires trained models on appropriate QM datasets (SPICE, ANI, etc.).

### Next Steps for Production

1. **Train Model**: Use TorchMD-NET training pipeline on relevant dataset
2. **Load Checkpoint**: Replace untrained model with trained checkpoint
3. **Validate**: Compare with QM reference calculations
4. **Deploy**: Run production simulations

Example training:
```bash
cd /path/to/torchmd-net
torchmd-train --conf examples/TensorNet-SPICE.yaml
```

## 📚 References

### TorchMD-NET
- Paper: [arXiv:2202.02541](https://arxiv.org/abs/2202.02541)
- GitHub: [https://github.com/torchmd/torchmd-net](https://github.com/torchmd/torchmd-net)

### OpenMM
- Website: [http://openmm.org/](http://openmm.org/)
- Docs: [http://docs.openmm.org/](http://docs.openmm.org/)

### OpenMM-Torch
- GitHub: [https://github.com/openmm/openmm-torch](https://github.com/openmm/openmm-torch)

## 📄 License

This benchmark suite is provided as-is for research and educational purposes.

- TorchMD-NET: MIT License
- OpenMM: MIT License

## 🙏 Acknowledgments

- TorchMD-NET developers for the excellent ML forcefield framework
- OpenMM team for the simulation engine
- OpenMM-Torch developers for the PyTorch integration

## 📧 Contact

For issues and questions:
- Open an issue on GitHub
- Check documentation in `TORCHMDNET_SETUP_COMPLETE.md`

---

**Status**: ✅ Fully Operational (2025-11-17)

**Tested System**: Tesla T4 GPU, CUDA 12.6, PyTorch 2.7.1, TorchMD-NET 2.4.12
