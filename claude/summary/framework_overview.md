# MAPLE Framework Overview

## Project Identity

**MAPLE** = **MAchine-learning Potential for Landscape Exploration**

A Python-based computational chemistry toolkit developed at the University of Pittsburgh that leverages machine learning potentials for efficient molecular simulations.

**Version**: 0.1.0 (Active Development)
**License**: BSD 3-Clause + Creative Commons BY (academic use only)
**Repository**: University of Pittsburgh computational chemistry group

---

## Technology Stack

### Core Dependencies
- **Python** >= 3.9
- **PyTorch** >= 2.0 (CUDA or CPU support)
- **ASE** (Atomic Simulation Environment) - Interface for atomistic simulations
- **NumPy** - Numerical computing
- **SciPy** - Scientific algorithms
- **fairchem-core** - Machine learning potential models

### Supported ML Potentials
| Potential | Target Systems | Description |
|-----------|---------------|-------------|
| **UMA** | Universal Materials | General-purpose materials potential |
| **AIMNet2** | Organic molecules | General organic chemistry (CHNO+) |
| **ANI** | CHNO molecules | Accurate Neural Network potential for organic systems |
| **MACE** | Universal | Message-passing equivariant architecture |

### Additional Corrections
- **DFT-D4**: Dispersion correction for long-range interactions
- **GBSA**: Generalized Born Surface Area for implicit solvation
- **QEq**: Charge equilibration methods

---

## Project Structure

```
/home/user/software/MAPLE/
├── maple/                          # Main package (95 Python files)
│   ├── main.py                     # CLI entry point
│   ├── __init__.py
│   └── function/
│       ├── engine.py               # Core orchestrator
│       │
│       ├── read/                   # Input parsing subsystem
│       │   ├── input_reader.py     # Main input parser
│       │   ├── command_control.py  # Hierarchical parameter parser
│       │   ├── header/             # Header directive parsing
│       │   │   ├── header.py
│       │   │   ├── device.py
│       │   │   ├── jobtype.py
│       │   │   └── model.py
│       │   └── filereader/         # Coordinate file readers
│       │       ├── xyzreader.py
│       │       └── xyztrajreader.py
│       │
│       ├── calculator/             # ML potential calculators
│       │   ├── set_calculator.py   # Calculator factory
│       │   ├── aimnet/             # AIMNet2 implementation
│       │   │   └── aimnet2_calculator.py
│       │   ├── ani/                # ANI implementation
│       │   │   └── ani_calculator.py
│       │   ├── mace/               # MACE implementation
│       │   │   └── mace_calculator.py
│       │   ├── uma/                # UMA implementation
│       │   │   └── uma_calculator.py
│       │   └── extra_correction/   # Additional physics
│       │       ├── charge/         # Charge corrections (QEq)
│       │       │   ├── qeq.py
│       │       │   └── eeq.py
│       │       └── solvent/        # Solvent models (GBSA)
│       │           └── gbsa.py
│       │
│       ├── dispatcher/             # Job dispatchers (43 files)
│       │   ├── dispatcher.py       # Main job router
│       │   ├── jobABC.py           # Abstract base class for all jobs
│       │   │
│       │   ├── optimization/       # Geometry optimization
│       │   │   ├── opt.py
│       │   │   └── algorithm/
│       │   │       ├── lbfgs.py
│       │   │       ├── rfo.py
│       │   │       ├── sd.py
│       │   │       └── diis.py
│       │   │
│       │   ├── ts/                 # Transition state methods
│       │   │   ├── ts.py
│       │   │   └── algorithm/
│       │   │       ├── neb.py      # Nudged Elastic Band (1517 lines)
│       │   │       ├── PRFO.py     # Partitioned RFO
│       │   │       ├── dimer.py    # Dimer method
│       │   │       ├── string.py   # Growing String Method
│       │   │       ├── afir.py     # AFIR method
│       │   │       ├── autoneb.py  # Automated NEB
│       │   │       └── ...
│       │   │
│       │   ├── irc/                # Intrinsic reaction coordinate
│       │   │   ├── irc.py
│       │   │   └── algorithm/
│       │   │       └── gs.py       # Gonzalez-Schlegel method
│       │   │
│       │   ├── frequency/          # Frequency analysis
│       │   │   └── freq.py
│       │   │
│       │   ├── scan/               # PES scanning
│       │   │   └── scan.py
│       │   │
│       │   ├── sp/                 # Single point energy
│       │   │   └── sp.py
│       │   │
│       │   └── hessian/            # Hessian calculations
│       │       └── hessian.py
│       │
│       ├── utility/                # Utility classes
│       │   ├── molecules.py        # Multi-structure container
│       │   └── ...
│       │
│       ├── timer/                  # Performance timing
│       │   └── timer.py
│       │
│       └── units/                  # Unit conversions
│           └── units.py            # Hartree/Angstrom conversions
│
├── example/                        # Test cases and examples
│   ├── opt/                        # Optimization examples
│   ├── ts/                         # TS search examples
│   ├── irc/                        # IRC examples
│   ├── freq/                       # Frequency examples
│   ├── scan/                       # Scan examples
│   ├── sp/                         # Single point examples
│   └── test.py                     # Test runner script
│
├── claude/                         # Documentation (this directory)
│   ├── summary/                    # Code summaries
│   ├── knowledge/                  # Collaboration notes
│   └── plan/                       # Development plans
│
├── ARCHITECTURE.md                 # Detailed architecture docs
├── README.md                       # User documentation
├── pyproject.toml                  # Package configuration
├── LICENSE                         # BSD 3-Clause + CC BY
├── freq_traj.py                    # Utility: frequency visualization
└── out2xyz.py                      # Utility: output conversion
```

---

## Main Components

### 1. Entry Point and Engine

**[main.py](../maple/main.py)**
- CLI interface with argument parsing
- Supports direct execution: `maple input.inp`
- Built-in test cases (8 examples)
- Version and help information

**[engine.py](../maple/function/engine.py)**
- Core orchestrator that coordinates:
  - Input file reading and parsing
  - Calculator setup (ML potential selection)
  - Job dispatching to appropriate handler
  - Output management

### 2. Input System

**[InputReader](../maple/function/read/input_reader.py)**
- Parses input files with three sections:
  1. Header directives (model, job type, device)
  2. Coordinates (inline or external file reference)
  3. Command control block (optional parameters)
- Case-insensitive parsing
- Comment support (`#` for comments)

**[CommandControl](../maple/function/read/command_control.py)**
- Hierarchical parameter management
- Three-tier architecture: global, job-level, method-level
- Alias support for parameter names
- Type conversion and validation
- Nested dictionary structure

**File Readers**
- **XYZReader**: Single structure from XYZ file
- **XYZTrajReader**: Multiple structures from trajectory file

### 3. Calculator System

**[SetCalculator](../maple/function/calculator/set_calculator.py)**
- Factory pattern for ML potential initialization
- Device management (CPU, GPU0, GPU1, etc.)
- Automatic model download and caching
- Calculator attachment to ASE Atoms objects

**Calculator Implementations**
Each ML potential has dedicated calculator:
- [AIMNet2Calculator](../maple/function/calculator/aimnet/aimnet2_calculator.py)
- [ANICalculator](../maple/function/calculator/ani/ani_calculator.py)
- [MACECalculator](../maple/function/calculator/mace/mace_calculator.py)
- [UMACalculator](../maple/function/calculator/uma/uma_calculator.py)

**Extra Corrections**
- **D4 Dispersion**: DFT-D4 correction for van der Waals interactions
- **GBSA Solvation**: Implicit solvent effects
- **Charge Methods**: QEq, EEq for partial charge assignment

### 4. Job Dispatcher Architecture

**[JobABC](../maple/function/dispatcher/jobABC.py)** - Abstract base class providing:
- `_init_params()`: Case-insensitive parameter initialization
- `_lower_keys()`: Recursive dictionary key lowercasing
- `_select_subdict()`: Nested parameter extraction
- `log_info()`, `log_error()`: File-based logging
- Abstract `run()` method for algorithm implementation

**Supported Job Types**

| Job Type | Command | Methods | Purpose |
|----------|---------|---------|---------|
| **opt** | `#opt` | LBFGS, RFO, SD, DIIS | Geometry optimization to local minimum |
| **ts** | `#ts` | NEB, PRFO, Dimer, String, AFIR, AutoNEB | Transition state search |
| **irc** | `#irc` | GS | Intrinsic reaction coordinate following |
| **freq** | `#freq` | MW | Vibrational frequency analysis |
| **scan** | `#scan` | Constrained opt | Potential energy surface scanning |
| **sp** | `#sp` | Direct | Single point energy/forces calculation |
| **hessian** | `#hessian` | Finite diff | Hessian matrix calculation |

---

## Execution Flow

```
User Command: maple input.inp
        ↓
┌───────────────────┐
│    main.py        │  Parse command line arguments
│  (CLI Entry)      │  Load input file path
└─────────┬─────────┘
          ↓
┌───────────────────┐
│    engine()       │  Core orchestration function
└─────────┬─────────┘
          ↓
┌───────────────────┐
│  InputReader      │  Parse input file:
│                   │  - Headers (#model, #jobtype, #device)
│                   │  - Coordinates (XYZ inline or file)
│                   │  - Command control block (parameters)
└─────────┬─────────┘
          ↓
┌───────────────────┐
│  SetCalculator    │  Initialize ML potential:
│                   │  - Select model (UMA/AIMNet2/ANI/MACE)
│                   │  - Configure device (GPU/CPU)
│                   │  - Apply corrections (D4, GBSA)
│                   │  - Attach to Atoms object
└─────────┬─────────┘
          ↓
┌───────────────────┐
│   Dispatcher      │  Route to job handler:
│                   │  - Set convergence thresholds
│                   │  - Extract job-specific parameters
│                   │  - Instantiate job class
└─────────┬─────────┘
          ↓
    ┌─────┴─────┬──────┬──────┬──────┬──────┐
    │           │      │      │      │      │
┌───▼───┐  ┌───▼──┐ ┌─▼──┐ ┌─▼───┐ ┌▼───┐ ┌▼──┐
│  opt  │  │  ts  │ │irc │ │freq │ │scan│ │sp │
│       │  │      │ │    │ │     │ │    │ │   │
│LBFGS  │  │ NEB  │ │ GS │ │ MW  │ │Cstr│ │E+F│
│ RFO   │  │PRFO  │ │    │ │     │ │Opt │ │   │
│ SD    │  │Dimer │ │    │ │     │ │    │ │   │
│ DIIS  │  │String│ │    │ │     │ │    │ │   │
│       │  │ AFIR │ │    │ │     │ │    │ │   │
└───┬───┘  └───┬──┘ └─┬──┘ └──┬──┘ └─┬──┘ └┬──┘
    │          │      │       │      │     │
    └──────────┴──────┴───────┴──────┴─────┘
                       ↓
              ┌────────────────┐
              │  Output Files  │
              │  - Trajectories│
              │  - Optimized   │
              │  - Results     │
              └────────────────┘
```

---

## Input File Format

### Basic Structure

```
# Header directives (case-insensitive)
#model=<uma|aimnet2|ani|mace>
#<jobtype>(method=<method>)
#device=<gpu0|gpu1|cpu>

# Coordinates (inline)
C   0.000   0.000   0.000
H   1.089   0.000   0.000
O  -1.200   0.000   0.000

# OR external file reference
XYZ /path/to/structure.xyz

# Optional command control block
$command_control
  level = medium

  opt
    algorithm = lbfgs
    max_steps = 500
  end

  neb
    n_images = 12
    k_max = 0.3
    refine = cineb
  end
$end
```

### Header Directives

**Model Selection**
```
#model=uma         # Universal materials potential
#model=aimnet2     # AIMNet2 for organic molecules
#model=ani         # ANI for CHNO systems
#model=mace        # MACE universal potential
```

**Job Type**
```
#opt                    # Geometry optimization (default: LBFGS)
#opt(method=rfo)        # With specific method
#ts(method=neb)         # Transition state search
#irc                    # IRC calculation
#freq                   # Frequency analysis
#scan                   # PES scan
#sp                     # Single point
```

**Device**
```
#device=gpu0       # First GPU
#device=cpu        # CPU only
```

### Command Control Parameters

**Global Level**
```
$command_control
  level = medium              # Convergence: extratight/tight/medium/loose/extraloose/superloose
  max_steps = 500            # Maximum optimization steps
  output = output_name       # Output file prefix
$end
```

**Job-Specific**
```
$command_control
  opt
    algorithm = lbfgs        # Algorithm selection
    lbfgs_memory = 20       # History size
  end

  neb
    n_images = 12           # Number of NEB images
    k_max = 0.3             # Spring constant
    refine = cineb          # Post-refinement
  end
$end
```

---

## Convergence Thresholds (Gaussian-style)

| Level | f_max (Eh/Å) | f_rms (Eh/Å) | dp_max (Å) | dp_rms (Å) |
|-------|-------------|-------------|-----------|-----------|
| **extratight** | 2.5e-5 | 1.7e-5 | 1.0e-4 | 6.7e-5 |
| **tight** | 5.0e-5 | 3.3e-5 | 2.0e-4 | 1.3e-4 |
| **medium** (default) | 1.5e-4 | 1.0e-4 | 6.0e-4 | 4.0e-4 |
| **loose** | 2.5e-4 | 1.7e-4 | 1.0e-3 | 6.7e-4 |
| **extraloose** | 1.0e-3 | 6.7e-4 | 4.0e-3 | 2.7e-3 |
| **superloose** | 2.5e-3 | 1.7e-3 | 1.0e-2 | 6.7e-3 |

Convergence criteria:
- `f_max`: Maximum force component
- `f_rms`: Root mean square force
- `dp_max`: Maximum displacement
- `dp_rms`: RMS displacement

---

## Output Files

### Naming Convention
Files use prefix from `output` parameter (or input filename) + suffix:

| Suffix | Description | Job Types |
|--------|-------------|-----------|
| `_opt.xyz` | Optimized structure | opt |
| `_traj.xyz` | Optimization trajectory | opt, ts (PRFO/Dimer) |
| `_mep.xyz` | Minimum energy path | ts (NEB) |
| `_hei.xyz` | Highest energy image | ts (NEB) |
| `_cineb_mep.xyz` | CI-NEB refined path | ts (NEB with refine=cineb) |
| `_ts.xyz` | Transition state | ts (all methods) |
| `_irc_fwd.xyz` | IRC forward path | irc |
| `_irc_bwd.xyz` | IRC backward path | irc |
| `_freq.out` | Frequency results | freq |
| `_scan.xyz` | Scan trajectory | scan |
| `_image_N_traj.xyz` | Per-image trajectory | ts (NEB) |

### Output Format (XYZ)
```
<number_of_atoms>
Energy: -123.456789 Hartree
C   0.000   0.000   0.000
H   1.089   0.000   0.000
...
```

---

## Key Features

### 1. Multiple ML Potentials
- Unified interface for UMA, AIMNet2, ANI, MACE
- Automatic model downloading
- Device selection (CPU/GPU)

### 2. Comprehensive Job Types
- **Optimization**: Local minima search
- **Transition States**: 8 different algorithms
- **IRC**: Reaction pathway following
- **Frequencies**: Vibrational analysis
- **Scanning**: PES exploration

### 3. Advanced Algorithms
- **NEB**: IDPP interpolation, dynamic springs, CINEB refinement
- **PRFO**: Trust region, mode following
- **Dimer**: Minimum-mode following
- **String**: Growing string with reparameterization
- **AFIR**: Artificial force for reaction discovery

### 4. Flexible Parameter System
- Three-tier hierarchy (global → job → method)
- Case-insensitive parsing
- Alias support
- Type validation

### 5. Quality Output
- XYZ trajectory files
- Energy profiles
- Detailed logging
- Convergence information

### 6. Built-in Examples
8 test cases demonstrating all features:
1. LBFGS optimization
2. RFO optimization
3. NEB transition state
4. String method
5. Dimer method
6. IRC calculation
7. Frequency analysis
8. PES scanning

---

## Installation and Usage

### Installation
```bash
cd /home/user/software/MAPLE
pip install -e .              # Full installation
pip install -e ".[minimal]"   # Minimal dependencies
pip install -e ".[dev]"       # Development mode
```

### Basic Usage
```bash
# Run with input file
maple input.inp

# Run built-in test
maple --test 1       # LBFGS optimization test

# Show help
maple --help

# Show version
maple --version
```

### Example Workflow
```bash
# 1. Create input file
cat > ethane_opt.inp <<EOF
#model=aimnet2
#opt(method=lbfgs)
#device=gpu0

C   0.000   0.000   0.000
C   1.500   0.000   0.000
H   -0.5   -0.9   0.0
H   -0.5    0.9   0.0
H   -0.5    0.0   0.9
H    2.0   -0.9   0.0
H    2.0    0.9   0.0
H    2.0    0.0   0.9
EOF

# 2. Run optimization
maple ethane_opt.inp

# 3. Check results
cat ethane_opt_opt.xyz      # Optimized structure
cat ethane_opt_traj.xyz     # Trajectory
```

---

## Software Architecture Highlights

### Design Patterns
1. **Factory Pattern**: SetCalculator for ML potential creation
2. **Template Method**: JobABC defines algorithm interface
3. **Strategy Pattern**: Dispatcher routes to algorithms
4. **Dataclass Pattern**: Type-safe parameters with defaults

### Code Quality
- Modular architecture with clear separation
- Extensive use of type hints (Python 3.9+)
- Dataclasses for parameter management
- Comprehensive docstrings
- Reusable utility functions

### Extensibility
- Easy to add new ML potentials (implement calculator interface)
- Easy to add new job types (inherit from JobABC)
- Easy to add new algorithms (register in dispatcher)
- Pluggable corrections (D4, GBSA)

---

## Dependencies Summary

**Required**
- torch, ase, numpy, scipy, fairchem-core

**Optional**
- dftd4, pyscf-dftd4 (for D4 dispersion)
- ml-collections (for AIMNet2)
- CUDA toolkit (for GPU acceleration)

**Development**
- pytest, black, mypy, flake8

---

## License and Attribution

**License**: BSD 3-Clause + Creative Commons Attribution (CC BY)
- Free for academic and research use with attribution
- Commercial use requires separate licensing
- Developed at University of Pittsburgh

**Citation**: Please cite MAPLE in publications using this software.

---

## Summary

MAPLE is a production-quality computational chemistry toolkit that:
- Bridges ML potentials with traditional quantum chemistry workflows
- Provides 8+ algorithms for geometry optimization and TS search
- Offers Gaussian-style input format for user familiarity
- Implements state-of-the-art methods (NEB, PRFO, Dimer, String, AFIR)
- Maintains clean, extensible architecture
- Supports multiple hardware backends (CPU/GPU)

**Target Users**: Computational chemists, materials scientists, machine learning researchers working on molecular simulations.
