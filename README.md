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

## Key Features

- **Fast ML-driven calculations**: Orders of magnitude faster than DFT
- **Multiple optimization algorithms**: LBFGS, RFO for various optimization scenarios
- **Comprehensive TS methods**: NEB, CI-NEB, String, Dimer for transition state searching
- **Flexible calculator support**: AIMNet2, ANI models, easily extensible
- **Command-line interface**: Simple `maple input.inp` execution
- **Trajectory tracking**: Automatic generation of optimization trajectories

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

# Use shell redirection
maple input.inp > custom_output.txt
```

### Example Input File

```
# Structure optimization with LBFGS
TASK = opt
CALCULATOR = aimnet2
OPT_ALGORITHM = lbfgs
MAX_STEPS = 100

GEOMETRY
C  0.000  0.000  0.000
H  1.089  0.000  0.000
H -0.363  1.028  0.000
H -0.363 -0.514  0.890
H -0.363 -0.514 -0.890
END
```

## Supported Calculations

### 1. Structure Optimization (`TASK = opt`)

Find energy minima using:
- **LBFGS**: Limited-memory BFGS for efficient optimization
- **RFO**: Rational Function Optimization for challenging cases

**Example**: See `example/opt/lbfgs/inp1.inp`

### 2. Transition State Search (`TASK = ts`)

Locate transition states using:

- **NEB (Nudged Elastic Band)**: Connect reactant and product with elastic band
- **CI-NEB**: Climbing Image NEB for accurate TS geometry
- **String Method**: Evolve string in collective variable space
- **Dimer Method**: Find TS from single initial structure

**Example**: See `example/ts/neb/inp2.inp`, `example/ts/neb/inp3.inp`

### 3. IRC Calculations (`TASK = irc`)

Follow reaction coordinate from transition state to reactants/products:
- **GS (Gonzalez-Schlegel)**: Second-order IRC method

**Example**: See `example/irc/gs/inp1.inp`

### 4. Frequency Analysis (`TASK = freq`)

Calculate vibrational frequencies and thermodynamic properties:
- Mass-weighted Hessian
- IR intensities
- Zero-point energy, enthalpies, entropies

**Example**: See `example/freq/mw/inp1.inp`

### 5. Coordinate Scan (`TASK = scan`)

Explore potential energy surface along specified coordinates:
- Relaxed scans
- Constrained optimizations

**Example**: See `example/scan/exo.inp`

## Calculators

MAPLE supports multiple machine learning potential calculators:

### AIMNet2 Calculator (`CALCULATOR = aimnet2`)

High-accuracy neural network potential trained on large quantum chemical datasets.
- Good transferability across chemical space
- Fast inference on GPU

### ANI Calculator (`CALCULATOR = ani`)

ANI (Accurate Neural Network Interaction) potentials for organic molecules.
- Excellent for C, H, N, O systems
- Multiple model versions available

## Output Files

MAPLE generates several output files:

- `*.out`: Main output with energies, gradients, optimization progress
- `*_opt.xyz`: Final optimized structure
- `*_traj.xyz`: Full optimization trajectory
- `*_mep.xyz`: Minimum energy path (for NEB/String)
- `*_ts.xyz`: Transition state structure
- `*_hei.xyz`: Highest energy image along path

## Advanced Features

### Dynamic Spring Constants (NEB)

MAPLE implements ORCA-style dynamic spring constants for improved NEB convergence:
```
k_i = k_min + (k_max - k_min) × exp(-k_decay × ΔE_i / ΔE_max)
```

### Convergence Criteria

Flexible convergence thresholds:
- Energy change tolerance
- Force/gradient thresholds
- Maximum displacement criteria

### Trajectory Analysis

All calculations save complete trajectories for post-processing and visualization.

## Project Structure

```
maple/
├── maple/
│   ├── function/
│   │   ├── calculator/          # ML potential calculators
│   │   ├── dispatcher/          # Task dispatchers (opt, ts, irc, freq)
│   │   │   ├── optimization/    # Optimization algorithms
│   │   │   ├── ts/              # Transition state methods
│   │   │   ├── irc/             # IRC implementations
│   │   │   └── frequency/       # Frequency calculations
│   │   ├── read/                # Input file parsers
│   │   └── utility/             # Helper functions
│   ├── cli.py                   # Command-line interface
│   └── main.py                  # Main entry point
├── example/                     # Example input files
└── README.md
```

## License

```
BSD 3-Clause License

Copyright (c) 2024, ClickFF

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

### Important Notice

**For Academic Use Only**: This software is licensed under Creative Commons Attribution (CC BY) for academic and research purposes.

**Commercial Use Prohibited**: Commercial use requires explicit permission from the copyright holder. Contact the author for licensing inquiries.

**All Rights Reserved**: The copyright holder reserves all rights not explicitly granted by the BSD 3-Clause License and CC BY terms.

## Citation

If you use MAPLE in your research, please cite:

```
[Citation information to be added]
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
- fair-chem (Open Catalyst Project)

---

**Version**: 0.1.0  
**Status**: Active Development  
**Last Updated**: December 2024