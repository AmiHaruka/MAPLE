# MAPLE

**MA**chine Learning **P**otential for **L**andscape **E**xploration (**MAPLE**)

MAPLE is a powerful computational chemistry toolkit that leverages machine learning potentials for efficient structure optimization, transition state searching, and reaction pathway analysis.

## Overview

MAPLE integrates state-of-the-art machine learning models (AIMNet2, ANI) with classical optimization algorithms to enable:

- **Structure Optimization**: LBFGS, RFO algorithms for finding energy minima
- **Transition State Search**: NEB (Nudged Elastic Band), CI-NEB, String Method, Dimer Method
- **Reaction Path Analysis**: IRC (Intrinsic Reaction Coordinate) calculations
- **Vibrational Analysis**: Frequency calculations with mass-weighted Hessian

The software is designed for computational chemists who need fast, accurate quantum-mechanical calculations without the computational cost of traditional ab initio methods.

## Installation

### Requirements

- Python ≥ 3.9
- CUDA-capable GPU (recommended for ML models)

### Install MAPLE

```bash
# Clone the repository
git clone https://github.com/ClickFF/maple.git
cd maple

# Install in development mode
pip install -e .
```

### Install Dependencies

MAPLE requires several scientific computing and machine learning packages:

```bash
# Core dependencies (required)
conda install -c conda-forge numpy scipy matplotlib ase

# PyTorch (adjust for your CUDA version)
# For CUDA 11.8:
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CPU only:
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Machine learning potentials
pip install fairchem-core  # For AIMNet2 and ANI models
```

## Quick Start

### Basic Usage

```bash
# Run optimization
maple input.inp

# Specify output file
maple input.inp output.out
```

### Example Input File

```
#model=ANI-1xnr
#ts(method=neb,refine=nebts)
#device=gpu0

C  0.000  0.000  0.000
H  1.089  0.000  0.000
H -0.363  1.028  0.000
H -0.363 -0.514  0.890
H -0.363 -0.514 -0.890
```


## Citation

If you use MAPLE in your research, please cite:

```
https://github.com/ClickFF/MAPLE
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes with clear commit messages
4. Submit a pull request

## Support

For questions, bug reports, or feature requests:
- Open an issue on GitHub
- Contact: [Contact information]

## Acknowledgments

MAPLE leverages several excellent open-source projects:
- ASE (Atomic Simulation Environment)
- PyTorch
- AIMNet2 / ANI models
- fair-chem

---

**Version**: 0.1.0  
**Status**: Active Development  
**Last Updated**: December 2025