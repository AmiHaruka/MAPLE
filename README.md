# **MA**chine-learning **P**otential for **L**andscape **E**xploration (**MAPLE**)

![MAPLE Concept](./maple.jpg)

MAPLE is a **machine-learning force-field (MLFF)–native molecular modeling platform**
designed for efficient and scalable exploration of potential energy landscapes, with a
particular focus on **geometry optimization, transition-state localization, and reaction
pathway analysis** under GPU-accelerated and parallel workflows.

---

## Core Capabilities

### Geometry Optimization
- Energy minimization using **batch-parallel L-BFGS**
- Saddle-point optimization via **partitioned RFO variants (DS-P-RFO)**

### Transition-State and Reaction Path Search
- **NEB / CI-NEB** for minimum-energy pathway construction
- String-based and dimer-style TS localization (where applicable)
- Stable handling of non-equilibrium and reactive structures

### Reaction Pathway Analysis
- **Intrinsic Reaction Coordinate (IRC)** calculations
- Extraction and refinement of highest-energy images (HEI)
- Mechanistic descriptor analysis along reaction coordinates

### Vibrational and Hessian Analysis
- Mass-weighted Hessian construction
- Harmonic frequency analysis
- Imaginary-mode validation for transition-state characterization

---

## Machine-Learning Force Fields

MAPLE interfaces with multiple **general-purpose reactive ML force fields**, including:

- **ANI family**
- **AIMNet2 family**
- **Universal MLFFs (e.g., UMA)**

These models are treated as **interchangeable computational backends**.  
MAPLE itself remains **model-agnostic**, focusing on algorithmic robustness, physical
consistency, and scalable execution.

> MAPLE does **not** conceptually depend on any specific MLFF ecosystem.  
> Model providers and implementations are modular and replaceable by design.

---

## Installation

### Requirements

- Python ≥ 3.9
- CUDA-capable GPU strongly recommended for production workloads

### Install MAPLE

```bash
git clone https://github.com/ClickFF/MAPLE.git
cd MAPLE
pip install -e .

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
- Contact: xuw74@pitt.edu

---

**Version**: 0.1.0  
**Status**: Active Development  
**Last Updated**: December 2025
