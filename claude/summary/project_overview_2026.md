# MAPLE Project Summary - February 2026

**Document Generated**: February 6, 2026
**Repository**: /home/user/software/MAPLE
**Version**: 0.1.0
**Current Branch**: enhance
**Active Development Focus**: Transition State Search Module

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Project Overview](#project-overview)
3. [Architecture](#architecture)
4. [Transition State Module - Detailed Analysis](#transition-state-module---detailed-analysis)
5. [Recent Development Activity](#recent-development-activity)
6. [Technical Implementation Details](#technical-implementation-details)
7. [Usage Examples](#usage-examples)
8. [Development Guidelines](#development-guidelines)

---

## Executive Summary

**MAPLE** (MAchine-learning Potential for Landscape Exploration) is a computational chemistry toolkit developed for efficient molecular simulations using machine learning potentials. The project provides a comprehensive suite of quantum chemistry algorithms—including geometry optimization, transition state searching, and reaction pathway analysis—powered by modern ML potentials (AIMNet2, ANI, MACE, UMA).

### Current Development Status (Feb 2026)
- **Active Branch**: `enhance`
- **Recent Major Changes**:
  - ✅ Dimer method bug fix (commit: 2d0758b)
  - ✅ AFIR method removed (commit: 4486881)
  - ✅ CINEB (Climbing Image NEB) refinement and debugging (commit: 238b89b)
  - ✅ SD, CG, DIIS optimization methods added (commit: 353044c)
  - ✅ IRC improvements and rigid scan support (commit: 1211979)
- **Primary Development Area**: Transition State search algorithms
- **Code Quality**: Well-structured, ~4,000+ LOC in TS module alone

---

## Project Overview

### 1.1 Core Capabilities

| Category | Methods Available |
|----------|------------------|
| **Geometry Optimization** | LBFGS, RFO, SD, CG, DIIS |
| **Transition State Search** | NEB, CI-NEB, PRFO, Dimer, String (GSM), AutoNEB |
| **Reaction Paths** | IRC (Gonzalez-Schlegel method) |
| **Analysis** | Frequency (mass-weighted), PES Scan, Single Point |
| **ML Potentials** | AIMNet2, ANI (1x/1ccx/1xnr/2x), MACE, UMA |
| **Physical Corrections** | DFT-D4 dispersion, GBSA solvation, QEq/EEq charges |

### 1.2 Key Features

✨ **Production-Ready Algorithms**:
- Robust NEB implementation with IDPP interpolation, Kabsch alignment, dynamic spring constants
- Advanced PRFO with trust-region control and mode-following
- Efficient Dimer method with HVP callback support
- Growing String Method with adaptive growth and endpoint refinement
- AutoNEB for automated multi-step reaction pathway exploration

🚀 **Performance Optimizations**:
- L-BFGS optimizer with memory-efficient two-loop recursion
- Batch computation support for ML potentials
- Projected force calculations for path optimization
- Smart interpolation strategies (IDPP vs linear)

🔧 **Developer-Friendly**:
- Clean OOP design with `JobABC` base class
- Dataclass-based parameter management
- Case-insensitive parameter handling
- Extensive logging and trajectory output

### 1.3 Installation & Requirements

```bash
# Clone repository
git clone https://github.com/ClickFF/maple.git
cd maple

# Install in development mode
pip install -e .

# Dependencies
pip install numpy scipy ase
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install fairchem-core  # For MACE and UMA potentials
```

**System Requirements**:
- Python >= 3.9 (tested with 3.12.7)
- PyTorch >= 2.0
- CUDA-capable GPU (recommended for ML potentials)

---

## Architecture

### 2.1 High-Level Structure

```
maple/
├── main.py                     # CLI entry point (maple command)
├── function/
│   ├── engine.py               # Core execution engine
│   ├── read/                   # Input file parsing
│   │   ├── input_reader.py     # Main parser
│   │   ├── command_control.py  # Parameter extraction
│   │   └── filereader/         # XYZ readers
│   ├── calculator/             # ML potential wrappers
│   │   ├── set_calculator.py   # Factory pattern
│   │   ├── aimnet/             # AIMNet2 implementation
│   │   ├── ani/                # ANI implementation
│   │   ├── mace/               # MACE implementation
│   │   ├── uma/                # UMA implementation
│   │   └── extra_correction/   # D4, GBSA corrections
│   └── dispatcher/             # Job routing & algorithms
│       ├── dispatcher.py       # Main router
│       ├── jobABC.py           # Abstract base class
│       ├── optimization/       # Geometry optimization
│       ├── ts/                 # ⭐ Transition State methods
│       ├── irc/                # IRC calculations
│       ├── frequency/          # Vibrational analysis
│       ├── scan/               # PES scanning
│       └── sp/                 # Single point energy
```

### 2.2 Execution Flow

```
User Input File (*.inp)
        ↓
main.py (CLI entry)
        ↓
engine.py
        ↓
┌───────┴────────┐
│  InputReader   │ → Parse #headers, $command_control, XYZ
└────────┬───────┘
         ↓
┌───────┴────────┐
│ SetCalculator  │ → Initialize ML potential (ANI/AIMNet2/MACE/UMA)
└────────┬───────┘
         ↓
┌───────┴────────┐
│  Dispatcher    │ → Route to job type
└────────┬───────┘
         ↓
    ┌────┴────┐
    │ TS Job  │ → method='neb'|'prfo'|'dimer'|'string'|'autoneb'
    └─────────┘

Example routing:
#ts(method=neb,refine=nebts)
    ↓
TransitionState.run()
    ↓
NEB.__init__(atoms_or_molecules, paras)
    ↓
NEB.run()
    ↓
Output files: *_mep.xyz, *_hei.xyz, *_traj.xyz
```

### 2.3 Parameter System Architecture

MAPLE uses a three-layer parameter system:

```
Input File ($command_control block)
        ↓
Python Dict (case-insensitive keys)
        ↓
JobABC._init_params(ParamsClass, paras, aliases)
        ↓
Dataclass Instance (typed, validated)
```

**Example**:
```python
# Input file:
$command_control
  level = medium
  neb
    n_images = 12
    k_max = 0.3
    refine = cineb
  end
$end

# Becomes:
paras = {
    'level': 'medium',
    'neb': {
        'n_images': 12,
        'k_max': 0.3,
        'refine': 'cineb'
    }
}

# Initialized as:
params = NEBParams(
    n_images=12,
    k_max=0.3,
    refine='cineb',
    # ... other defaults
)
```

---

## Transition State Module - Detailed Analysis

### 3.1 Module Overview

**Location**: `maple/function/dispatcher/ts/`

The TS module is the **most comprehensive and actively developed** component of MAPLE, containing sophisticated implementations of multiple transition state search methods. Total codebase: **~4,000+ lines** across 9 algorithm files.

### 3.2 File Structure

```
ts/
├── ts.py                    # Main dispatcher (120 lines)
├── __init__.py
└── algorithm/
    ├── __init__.py          # Exports: NEB, PRFO, Dimer, GSM, AutoNEB
    ├── neb.py               # 🔥 NEB + CI-NEB (1,517 lines)
    ├── PRFO.py              # Partitioned RFO (793 lines)
    ├── dimer.py             # Dimer method (521 lines)
    ├── string.py            # Growing String Method (1,100+ lines)
    ├── autoneb.py           # Automated NEB (1,200+ lines)
    ├── newton.py            # Newton optimizer (150 lines)
    ├── BPRFO.py             # Bofill-PRFO variant (850 lines)
    └── logger.py            # Logging utilities
```

**Note**: AFIR (afir.py, descafir.py) was recently **removed** in commit 4486881 (Feb 2026).

### 3.3 Algorithm Implementations

#### 3.3.1 NEB (Nudged Elastic Band)

**File**: `algorithm/neb.py` (1,517 lines) - **Most comprehensive implementation**

**Status**: ✅ Mature, recently debugged for CINEB (commit 238b89b)

**Key Features**:
- ✅ **IDPP Interpolation**: Image Dependent Pair Potential for robust initial paths
- ✅ **Kabsch Alignment**: Rigid-body alignment with RMSD reporting
- ✅ **Improved Tangent**: Energy-weighted tangent calculation (Henkelman-Jónsson)
- ✅ **Dynamic Spring Constants**: ORCA-style adaptive springs
  - `k = k_max - (k_max - k_min) * exp(-decay * ΔE / ΔE_max)`
  - Weaker springs near barrier, stronger in flat regions
- ✅ **Climbing Image NEB (CI-NEB)**: Optional refinement mode
- ✅ **L-BFGS Optimizer**: Custom driver with two-loop recursion
- ✅ **NEBTS Refinement**: Automatic PRFO refinement of TS from HEI

**Algorithm Flow**:

```
NEB.run()
│
├─[1] Input Processing
│    ├── Validate >= 2 images (reactant + product)
│    ├── Kabsch align all to reactant
│    └── Check if interpolation needed
│
├─[2] Path Interpolation
│    ├── Compute inter-image distances
│    ├── Determine insertion points (prioritize largest gaps)
│    ├── Linear interpolation
│    └── IDPP smoothing (if ifidpp=1)
│         └── Minimize: E_IDPP = Σ w_ij * (1/d_ij - 1/d_ij^target)²
│
├─[3] Optional Endpoint Optimization
│    └── If initial_opt=True: minimize R₀ and Rₙ
│
├─[4] Main NEB Optimization (L-BFGS on projected forces)
│    │
│    └── For each iteration:
│         ├── Compute energies & forces for all images
│         ├── Calculate improved tangents τᵢ
│         ├── Compute projected NEB forces:
│         │    F_NEB = F_perp + F_spring
│         │    F_perp = F - (F·τ)τ
│         │    F_spring = k(|Rᵢ₊₁-Rᵢ| - |Rᵢ-Rᵢ₋₁|)τ
│         ├── L-BFGS step: d = -H⁻¹ F_NEB
│         ├── Step size limiting
│         ├── Update positions: Rᵢ ← Rᵢ + α·d
│         ├── Update L-BFGS history
│         └── Check convergence:
│              max(|F_NEB|) < f_max_th AND
│              RMS(F_NEB) < f_rms_th
│
├─[5] Optional CI-NEB Refinement (if refine='cineb' or 'nebts')
│    ├── Identify highest energy image (HEI)
│    ├── Freeze HEI index
│    ├── Apply climbing force: F_CI = F - 2(F·τ)τ
│    ├── Dual convergence check:
│    │    - Regular images: standard NEB convergence
│    │    - Climbing image: separate tighter thresholds
│    └── Output: *_cineb_mep.xyz, *_cineb_hei.xyz
│
└─[6] NEBTS PRFO Refinement (if refine='nebts')
     ├── Extract HEI from CI-NEB
     ├── Run PRFO optimization on HEI
     └── Output: *_nebts_ts.xyz, *_nebts_mep.xyz
```

**Parameters** (`NEBParams` dataclass):

```python
@dataclass
class NEBParams:
    # Path setup
    n_images: int = 10                # Number of internal images
    ifidpp: int = 1                   # Use IDPP (1) or linear (0)
    initial_opt: bool = False         # Optimize endpoints first

    # Spring constants
    k_min: float = 0.03               # Minimum spring (barrier region)
    k_max: float = 0.3                # Maximum spring (flat region)
    use_dynamic_k: bool = True        # ORCA-style dynamic springs
    k_decay: float = 0.5              # Decay factor

    # Optimization
    max_iter: int = 500
    lbfgs_m: int = 20                 # L-BFGS memory size
    step0: float = 2e-2               # Initial step length

    # Convergence (regular NEB)
    neb_f_max_th: float = 9.5e-3      # Max force threshold
    neb_f_rms_th: float = 5e-3        # RMS force threshold

    # CI-NEB refinement
    refine: Optional[str] = None      # 'cineb' or 'nebts' or None
    cineb_f_max_th: float = 1e-2      # CI-NEB max force
    cineb_f_rms_th: float = 1e-2      # CI-NEB RMS force
    cilbfgs_m: int = 20               # CI-NEB L-BFGS memory
    cistep0: float = 5e-3             # CI-NEB step length
```

**Output Files**:
- `*_mep.xyz` - Minimum Energy Path (all images)
- `*_hei.xyz` - Highest Energy Image (TS guess)
- `*_image_traj.xyz` - Full optimization trajectory
- `*_cineb_mep.xyz` - CI-NEB refined path
- `*_cineb_hei.xyz` - CI-NEB refined TS
- `*_cineb_traj.xyz` - CI-NEB optimization trajectory
- `*_nebts_ts.xyz` - PRFO-refined TS structure
- `*_nebts_mep.xyz` - Path with refined TS

**Recent Changes**:
- ✅ CINEB debugging and cleanup (commit 238b89b, Feb 2026)
- ✅ Improved tangent calculation
- ✅ Better convergence criteria handling

---

#### 3.3.2 PRFO (Partitioned Rational Function Optimization)

**File**: `algorithm/PRFO.py` (793 lines)

**Purpose**: Refine TS guess to true saddle point (1st order, not 0th/2nd)

**Status**: ✅ Production-ready, recently redesigned (commit ac2dcb7)

**Key Features**:
- ✅ **Dual-shift RFO**: Separate shifts for maximization (mode-following) and minimization
- ✅ **Trust Region**: Adaptive radius based on model agreement
- ✅ **Mass-Weighted Coordinates**: Proper treatment of molecular vibrations
- ✅ **Hessian Update**: SR1 or BFGS update schemes
- ✅ **Mode-Following**: Track and follow the TS eigenvector

**Algorithm**:

```
PRFO.run()
│
├─[1] Initial Hessian
│    ├── Compute exact Hessian (finite difference)
│    └── Or use model Hessian (Lindh guess)
│
├─[2] For each iteration:
│    ├── Diagonalize Hessian: H = V Λ V^T
│    ├── Identify TS mode (most negative eigenvalue)
│    ├── Partition space:
│    │    - Maximize along TS mode (λ < 0)
│    │    - Minimize along all others (λ > 0)
│    ├── Compute RFO step:
│    │    δ = (H - λ*I)⁻¹ g
│    ├── Apply trust radius:
│    │    If |δ| > r_trust: δ ← δ * r_trust / |δ|
│    ├── Update geometry: x ← x + δ
│    ├── Update Hessian (SR1 or BFGS)
│    └── Adjust trust radius based on:
│         ΔE_actual / ΔE_predicted
│
└─[3] Convergence Check
     ├── |g| < g_threshold
     ├── |δ| < step_threshold
     └── Exactly 1 negative eigenvalue
```

**Parameters** (`PRFOParams`):

```python
@dataclass
class PRFOParams:
    max_iter: int = 100
    trust_radius: float = 0.3         # Initial trust radius (Angstrom)
    trust_radius_min: float = 0.05
    trust_radius_max: float = 0.5
    hessian_update: str = 'sr1'       # 'sr1' or 'bfgs'
    recalc_hessian_every: int = 10    # Recompute exact Hessian
    f_max_th: float = 3e-3            # Convergence: max force
    f_rms_th: float = 2e-3            # Convergence: RMS force
    dp_max_th: float = 4e-3           # Convergence: max displacement
    dp_rms_th: float = 3e-3           # Convergence: RMS displacement
```

**Output Files**:
- `*_prfo_ts.xyz` - Optimized TS structure
- `*_prfo_traj.xyz` - Optimization trajectory

**Typical Usage**:
- Refine NEB HEI to true saddle point
- Optimize user-provided TS guess
- Chain with frequency calculation to verify TS (1 imaginary freq)

---

#### 3.3.3 Dimer Method

**File**: `algorithm/dimer.py` (521 lines)

**Purpose**: Find TS without knowing reaction endpoint

**Status**: ✅ Production-ready, bug fix applied (commit 2d0758b, Feb 2026 - **most recent**)

**Key Features**:
- ✅ **Minimum-Mode Following**: Rotates dimer to find lowest curvature direction
- ✅ **Translation with Flip**: Ascends along negative curvature mode
- ✅ **HVP Callback Support**: Can use autograd for Hessian-vector product (no Δ tuning)
- ✅ **Finite Difference Fallback**: Robust numerical curvature estimation
- ✅ **Trust-Radius Control**: Adaptive step size management

**Dimer Concept**:

```
     R₁ ●─────n─────● R₂
            │
            ● R₀ (midpoint)

R₁ = R₀ + δ*n
R₂ = R₀ - δ*n
δ = 0.005 Å (dimer distance)

Curvature: κ = n^T H n ≈ (F₁ - F₂)·n / (2δ)
```

**Algorithm**:

```
Dimer.run()
│
└── For each iteration:
     │
     ├─[Rotation Phase] Minimize curvature κ = n^T H n
     │  │
     │  └── For rot_iter in range(rot_max_iter):
     │       ├── Compute F₁ at R₀ + δ*n
     │       ├── Compute F₂ at R₀ - δ*n
     │       ├── Estimate Hn:
     │       │    Hn ≈ (F₁ - F₂) / (2δ)  (finite diff)
     │       │    OR use hvp_fn(R₀, n)     (autograd)
     │       ├── Rotation force: F_rot = Hn - (Hn·n)n
     │       ├── Update dimer axis: n ← n + α * F_rot
     │       ├── Normalize: n ← n / |n|
     │       └── Check convergence: |F_rot| < rot_f_max_th
     │
     └─[Translation Phase] Move along dimer axis
          ├── Compute κ = (F₁ - F₂)·n / (2δ)
          ├── If κ < 0: F_eff = F₀ - 2(F₀·n)n  (flip to ascend)
          ├── Else:     F_eff = F₀              (descend)
          ├── Compute step: δR = α * F_eff
          ├── Apply step limit: |δR| < step_max
          ├── Update: R₀ ← R₀ + δR
          └── Check convergence: |F₀| < f_max_th
```

**Parameters** (`DimerParams`):

```python
@dataclass
class DimerParams:
    # Rotation
    use_hvp: bool = False             # Use HVP callback?
    delta: float = 0.005               # Dimer distance (Å)
    rot_max_iter: int = 5             # Rotation iterations per step
    rot_alpha: float = 0.5            # Rotation step factor
    rot_f_max_th: float = 1e-3        # Rotation convergence
    rot_f_rms_th: float = 5e-4

    # Translation
    step0: float = 0.2                # Initial step scaling
    step_max: float = 0.15            # Max displacement (Å)
    max_iter: int = 200

    # Convergence
    f_max_th: float = 3e-3
    f_rms_th: float = 2e-3
```

**Output Files**:
- `*_dimer_ts.xyz` - Optimized TS structure
- `*_dimer_traj.xyz` - Full trajectory (each step writes R₀)

**Recent Bug Fix (Feb 2026)**:
The most recent commit (2d0758b) fixed a bug in the Dimer implementation. Exact nature of the fix should be checked in the commit diff.

---

#### 3.3.4 String Method (GSM - Growing String Method)

**File**: `algorithm/string.py` (1,100+ lines)

**Purpose**: Build reaction path from endpoints via adaptive growth

**Status**: ✅ Production-ready

**Key Features**:
- ✅ **Two-Ended Growth**: Grow string from both reactant and product
- ✅ **Adaptive Strategy**: Pause slower side, refine endpoints, adapt step size
- ✅ **Constrained Relaxation**: Project forces onto tangent-orthogonal subspace
- ✅ **Smart Merging**: Detect close nodes and merge strings
- ✅ **Re-sampling**: Fixed-spacing interpolation after merge
- ✅ **CI-STRING Refinement**: Climbing image variant
- ✅ **STRINGTS**: Automatic PRFO refinement

**Algorithm**:

```
GSM.run()
│
├─[1] Initialization
│    ├── Start with R (reactant) and P (product)
│    └── Kabsch align P to R
│
├─[2] Growth Phase
│    │
│    └── While not merged:
│         ├── Grow from R: R_new = R_tip + Δs * τ
│         ├── Grow from P: P_new = P_tip + Δs * τ
│         ├── Relax new nodes (LBFGS on projected forces):
│         │    F_proj = F - (F·τ)τ
│         ├── Check merge condition: |R_tip - P_tip| < d_merge
│         └── Adaptive policy:
│              - Pause side with higher energy tip
│              - Detect endpoint descent → optimize endpoint
│              - Adapt step size based on force/energy
│
├─[3] Re-sampling
│    └── Interpolate to fixed n_images with equal spacing
│
├─[4] MEP Relaxation
│    └── Projected L-BFGS on full path (like NEB but no springs)
│
└─[5] Optional Refinements
     ├── CI-STRING: Climb HEI to TS
     └── STRINGTS: Run PRFO on HEI
```

**Parameters** (`GSMParams`):

```python
@dataclass
class GSMParams:
    # Growth
    n_images: int = 9                 # Target images after merge
    max_growth_iter: int = 50
    growth_step: float = 0.5          # Initial step size (Å)
    d_merge: float = 0.3              # Merge distance threshold

    # Relaxation
    relax_max_iter: int = 10          # Per-node relaxation
    relax_f_th: float = 0.05

    # MEP optimization
    mep_max_iter: int = 100
    mep_f_max_th: float = 1e-2
    mep_f_rms_th: float = 5e-3

    # Refinement
    refine: Optional[str] = None      # 'cistring' or 'stringts'
```

**Output Files**:
- `*_gsm_grow_final.xyz` - Final grown path before re-sampling
- `*_gsm_mep.xyz` - Relaxed MEP
- `*_gsm_hei.xyz` - Highest energy image
- `*_stringts_ts.xyz` - PRFO-refined TS (if refine='stringts')

---

#### 3.3.5 AutoNEB (Automated Multi-Step NEB)

**File**: `algorithm/autoneb.py` (1,200+ lines)

**Purpose**: Explore complex reaction pathways with multiple intermediates and TSs automatically

**Status**: ✅ V1 implemented (commit f5160f7)

**Key Features**:
- ✅ **Adaptive Image Insertion**: Add images where gaps are too large
- ✅ **Local Minima Detection**: Identify intermediates along path
- ✅ **Path Splitting**: Recursively create sub-paths for each segment
- ✅ **Endpoint Optimization**: Update endpoints if lower-energy structures found
- ✅ **Binary Tree Management**: Organize multiple paths hierarchically
- ✅ **Global Refinement**: Final NEB pass over all discovered paths

**Workflow**:

```
AutoNEB.run()
│
├─[1] Initial Path (R → P)
│    └── Run NEB on full path
│
├─[2] Detection Loop
│    │
│    └── For each path:
│         ├── Check for local minima:
│         │    E_i < E_{i-1} AND E_i < E_{i+1}
│         │    AND ΔE > min_e_drop
│         ├── If found: Split path at minimum
│         │    ├── New path 1: R → intermediate
│         │    └── New path 2: intermediate → P
│         └── Recursively process sub-paths
│
├─[3] Adaptive Insertion
│    └── If |R_i - R_{i+1}| > ang_max:
│         └── Insert new image and re-optimize
│
├─[4] Endpoint Updates
│    └── If neighbor has E < endpoint:
│         └── Update endpoint and re-optimize
│
└─[5] Global Refinement
     ├── Concatenate all path segments
     ├── Run final NEB with looser convergence
     └── Output global MEP
```

**Parameters** (`AutoNEBParams`):

```python
@dataclass
class AutoNEBParams:
    # Basic NEB settings
    n_images: int = 20
    k_min: float = 0.03
    k_max: float = 0.3
    max_iter: int = 256

    # Adaptive insertion
    ang_max: float = 0.3              # Max spacing (Å)
    ang_iter: int = 20                # Check every N iterations

    # Path splitting
    path_iter: int = 50               # Check every N iterations
    min_e_drop: float = 0.001         # Minima detection threshold (Eh)

    # Endpoint optimization
    ep_iter: int = 30
    ep_e_drop: float = 0.0005         # Endpoint update threshold (Eh)

    # Recursion control
    max_depth: int = 5
    max_paths: int = 10

    # Final refinement
    do_final_refine: bool = True
    final_refine_max_iter: int = 200
```

**Output Files**:
- `*_autoneb_tree.json` - Path tree structure (JSON)
- `*_autoneb_path_<id>_mep.xyz` - Individual path MEPs
- `*_autoneb_global_mep.xyz` - Combined global MEP
- `*_autoneb_intermediates.xyz` - All detected intermediates
- `*_autoneb_ts_list.xyz` - All transition states

**Use Cases**:
- Multi-step reactions (e.g., aldol condensation with multiple intermediates)
- Complex rearrangements with hidden intermediates
- Exploratory pathway searches

---

### 3.4 Method Comparison & Selection Guide

| Method | When to Use | Requires | Strengths | Limitations |
|--------|-------------|----------|-----------|-------------|
| **NEB** | Know R and P structures | R + P | Robust, full MEP, well-tested | Needs good initial/final states |
| **PRFO** | Refine TS guess | TS guess | Fast, precise, validates TS | Needs good starting point |
| **Dimer** | Unknown endpoint, have TS region | TS guess | No need for P, mode-following | Can be slower than NEB |
| **String** | R + P, adaptive growth desired | R + P | Handles difficult cases, adaptive | More complex than NEB |
| **AutoNEB** | Complex multi-step reactions | R + P | Finds intermediates automatically | Computationally expensive |

**Recommended Workflow**:

```
Simple reaction (known R, P):
    NEB → (optional) NEBTS → Frequency

Refine TS guess:
    PRFO → Frequency

Unknown product:
    Dimer → Frequency

Complex reaction:
    AutoNEB → (manual inspection) → PRFO (selected TSs) → Frequency

Chain NEB → PRFO:
    #ts(method=neb,refine=nebts)
    Automatically runs PRFO on CI-NEB HEI
```

---

## Recent Development Activity

### 4.1 Recent Commits (Feb 2026)

| Date | Commit | Description | Impact |
|------|--------|-------------|--------|
| Feb 2026 | 2d0758b | **Dimer bug fix** | 🔧 Fixed bug in dimer.py |
| Feb 2026 | 4486881 | **Remove AFIR** | ❌ Removed afir.py and descafir.py |
| Feb 2026 | 238b89b | **Clean & CINEB debug** | ✅ Improved CINEB implementation |
| Jan 2026 | 353044c | **SD CG DIIS** | ✨ Added steepest descent, conjugate gradient, DIIS optimizers |
| Jan 2026 | 1211979 | **IRC improvements** | ✨ Enhanced IRC + rigid scan support |
| Jan 2026 | f5160f7 | **AutoNEB v1** | ✨ First version of automated NEB |
| Jan 2026 | ac2dcb7 | **PRFO redesign** | 🔄 Redesigned PRFO algorithm |

### 4.2 Development Trends

**Active Development Areas**:
1. ✅ **Transition State Methods** (primary focus)
   - NEB refinements (CINEB, NEBTS)
   - Dimer improvements
   - AutoNEB implementation
2. ✅ **Optimization Algorithms**
   - Additional optimizers (SD, CG, DIIS)
   - Better convergence criteria
3. ✅ **Reaction Path Analysis**
   - IRC enhancements
   - Rigid scan support

**Recently Deprecated**:
- ❌ **AFIR Method** (removed Feb 2026) - Artificial Force Induced Reaction
  - Reason: Likely superseded by AutoNEB or maintenance burden

### 4.3 Code Quality Metrics

```bash
# TS Module Statistics
Total Lines: ~4,000+
Files: 9 Python files
Algorithms: 6 methods (NEB, PRFO, Dimer, String, AutoNEB, Newton)
Test Coverage: Examples in example/ts/ for all methods

# Largest implementations:
neb.py:       1,517 lines (most comprehensive)
autoneb.py:   1,200+ lines
string.py:    1,100+ lines
BPRFO.py:       850 lines
PRFO.py:        793 lines
dimer.py:       521 lines (recently bug-fixed)
```

---

## Technical Implementation Details

### 5.1 Common Utilities

All TS algorithms share common utilities:

```python
# Type conversion (handles numpy, torch, lists)
def to_numpy_f64(x):
    """Convert any input to float64 numpy array"""

# Vector flattening
def vec1d(x, n_expected=None):
    """Flatten to 1D and optionally validate size"""

# Rigid alignment
def kabsch_align(P, Q):
    """Align Q onto P using Kabsch algorithm
    Returns: Q_aligned, RMSD, rotation_matrix, translation
    """

# XYZ I/O
def write_xyz(filename, atoms, energy=None):
    """Write single or multi-frame XYZ"""

def write_all_images_xyz(filename, images, energies, iteration):
    """Append trajectory frame"""
```

### 5.2 L-BFGS Implementation

Custom L-BFGS driver used by NEB and AutoNEB:

```python
class LBFGSDriver:
    """Limited-memory BFGS optimizer with two-loop recursion"""

    def __init__(self, m=20, step0=0.02):
        self.m = m                    # Memory size
        self.step0 = step0            # Initial step
        self.S_hist = []              # Position differences
        self.Y_hist = []              # Gradient differences
        self.rho_hist = []            # 1 / (y · s)

    def two_loop(self, g):
        """Compute search direction d = H⁻¹ g using two-loop recursion"""
        q = g.copy()
        alpha = []

        # Backward loop
        for i in reversed(range(len(self.S_hist))):
            α_i = self.rho_hist[i] * np.dot(self.S_hist[i], q)
            q -= α_i * self.Y_hist[i]
            alpha.append(α_i)

        # Initial Hessian scaling
        if len(self.S_hist) > 0:
            s = self.S_hist[-1]
            y = self.Y_hist[-1]
            gamma = np.dot(s, y) / np.dot(y, y)
            q *= gamma

        # Forward loop
        for i in range(len(self.S_hist)):
            β = self.rho_hist[i] * np.dot(self.Y_hist[i], q)
            q += self.S_hist[i] * (alpha[-(i+1)] - β)

        return -q  # Search direction

    def update(self, x_old, x_new, g_old, g_new):
        """Update history with new step"""
        s = x_new - x_old
        y = g_new - g_old
        rho = 1.0 / np.dot(s, y)

        self.S_hist.append(s)
        self.Y_hist.append(y)
        self.rho_hist.append(rho)

        # Keep only last m entries
        if len(self.S_hist) > self.m:
            self.S_hist.pop(0)
            self.Y_hist.pop(0)
            self.rho_hist.pop(0)
```

### 5.3 Convergence Criteria

MAPLE uses Gaussian-style convergence thresholds:

| Level | f_max | f_rms | dp_max | dp_rms | Use Case |
|-------|-------|-------|--------|--------|----------|
| extratight | 0.00030 | 0.00020 | 0.00030 | 0.00020 | High-precision TS |
| tight | 0.00085 | 0.00055 | 0.00110 | 0.00075 | Publication quality |
| **medium** | 0.00285 | 0.00190 | 0.00315 | 0.00210 | **Default** |
| loose | 0.00380 | 0.00250 | 0.00600 | 0.00400 | Initial exploration |
| extraloose | 0.00755 | 0.00500 | 0.01200 | 0.00800 | Rough scans |

Units: Eh/Angstrom for forces, Angstrom for displacements

**Convergence Check**:
```python
converged = (max_force < f_max_th) and \
            (rms_force < f_rms_th) and \
            (max_disp < dp_max_th) and \
            (rms_disp < dp_rms_th)
```

### 5.4 Input File Format

Standard MAPLE input for TS search:

```
#model=<model_name>          # ML potential
#ts(method=<method>,<opts>)  # TS method with options
#device=<device>             # gpu0, gpu1, cpu

XYZ <path_to_reactant.xyz>   # Reactant structure (for NEB/String/AutoNEB)
XYZ <path_to_product.xyz>    # Product structure

# Alternative: inline coordinates
C   x   y   z
H   x   y   z
...

# Optional: parameter block
$command_control
  level = medium              # Convergence level
  neb                         # Method-specific params
    n_images = 12
    k_max = 0.3
    refine = nebts            # Chain with PRFO
  end
$end
```

**Available Models**:
- `ANI-1xnr` - ANI (CHNO molecules, fastest)
- `aimnet2` - AIMNet2 (general organic)
- `uma` - UMA (universal materials)
- `maceoff23m` - MACE (high accuracy)

**TS Methods**:
- `neb` - Nudged Elastic Band
- `prfo` - Partitioned RFO
- `dimer` - Dimer method
- `string` - Growing String Method
- `autoneb` - Automated multi-step NEB

---

## Usage Examples

### 6.1 Basic NEB Calculation

```bash
# Input file: neb_example.inp
#model=ANI-1xnr
#ts(method=neb)
#device=gpu0

XYZ /path/to/reactant.xyz
XYZ /path/to/product.xyz
```

```bash
# Run
maple neb_example.inp

# Output files:
# neb_example_mep.xyz        - Minimum energy path
# neb_example_hei.xyz        - Highest energy image (TS guess)
# neb_example_image_traj.xyz - Full optimization trajectory
```

### 6.2 NEB with CI-NEB Refinement and PRFO

```bash
#model=ANI-1xnr
#ts(method=neb,refine=nebts)
#device=gpu0

XYZ /path/to/reactant.xyz
XYZ /path/to/product.xyz

$command_control
  level = tight
  neb
    n_images = 15
    k_max = 0.3
    refine = nebts    # Run CI-NEB then PRFO
  end
$end
```

**Workflow**:
1. Regular NEB converges
2. CI-NEB refines HEI to TS
3. PRFO optimizes TS to saddle point
4. Outputs: `*_nebts_ts.xyz` (final TS)

### 6.3 Dimer Method

```bash
#model=ANI-1xnr
#ts(method=dimer)
#device=gpu0

# Provide TS guess (single structure)
C   0.877   0.783  -0.163
H   0.691   1.554  -0.922
O   0.530   0.976   1.123
...

$command_control
  dimer
    delta = 0.005
    rot_max_iter = 5
    max_iter = 150
  end
$end
```

### 6.4 AutoNEB for Complex Reactions

```bash
#model=aimnet2
#ts(method=autoneb)
#device=gpu0

XYZ reactant.xyz
XYZ product.xyz

$command_control
  autoneb
    n_images = 20
    max_depth = 5
    min_e_drop = 0.001
    do_final_refine = True
  end
$end
```

**Output**:
- JSON tree showing detected intermediates
- Individual path MEPs
- Global concatenated MEP
- All TS structures

### 6.5 Command-Line Test Cases

```bash
# Built-in test cases
maple --test 1   # LBFGS optimization
maple --test 2   # NEB transition state
maple --test 3   # String method
maple --test 4   # Dimer method
maple --test 5   # RFO optimization
maple --test 6   # IRC
maple --test 7   # Frequency
maple --test 8   # PES Scan
```

---

## Development Guidelines

### 7.1 Adding a New TS Algorithm

**Step 1**: Create parameter dataclass

```python
# In algorithm/new_method.py
from dataclasses import dataclass

@dataclass
class NewMethodParams:
    max_iter: int = 100
    step_size: float = 0.1
    convergence_th: float = 1e-3
    # ... more parameters
```

**Step 2**: Implement algorithm class

```python
from ...jobABC import JobABC

class NewMethod(JobABC):
    def __init__(self, atoms, output, paras=None):
        super().__init__(output)
        self.atoms = atoms

        # Initialize parameters using base class method
        self.params = self._init_params(
            NewMethodParams,
            paras,
            aliases=('newmethod', 'NewMethod', 'new_method')
        )

    def run(self):
        """Main algorithm implementation"""
        self.log_info("Starting NewMethod optimization")

        # Algorithm code here
        for iteration in range(self.params.max_iter):
            # ... optimization steps

            if converged:
                self.log_info(f"Converged at iteration {iteration}")
                break

        # Write output
        write_xyz(f"{self.output}_ts.xyz", self.atoms)
```

**Step 3**: Register in dispatcher

```python
# In algorithm/__init__.py
from .new_method import NewMethod

# In ts/ts.py
elif self.method == 'newmethod':
    from .algorithm import NewMethod
    method_obj = NewMethod(
        atoms=self.atoms,
        output=self.output,
        paras=self.params
    )
    method_obj.run()
```

**Step 4**: Add test case

```python
# In example/ts/newmethod/
# Create inp1.inp with example input
```

### 7.2 Parameter Naming Conventions

Follow these conventions for consistency:

```python
# ✅ Good
max_iter: int            # Maximum iterations
f_max_th: float          # Force maximum threshold
n_images: int            # Number of images
use_dynamic_k: bool      # Boolean flags start with "use_" or "is_"

# ❌ Avoid
maxIter                  # No camelCase
MaxIter                  # No PascalCase
f_max_threshold          # Abbreviate to "_th"
verbosity                # Use "verbose" instead
```

### 7.3 Logging Best Practices

```python
# Use JobABC logging methods
self.log_info([
    "="*60,
    f"Iteration {iter}",
    f"Energy: {energy:.8f} Eh",
    f"Max Force: {max_f:.6f} Eh/Ang",
    "="*60
])

# Error handling
try:
    result = some_operation()
except Exception as e:
    self.log_error(f"Operation failed: {str(e)}")
    raise
```

### 7.4 Testing Checklist

Before committing TS algorithm changes:

- [ ] Algorithm converges on test case
- [ ] Output files are generated correctly
- [ ] Parameters are properly documented (docstrings)
- [ ] Logging is informative but not excessive
- [ ] Convergence criteria match literature
- [ ] Memory usage is reasonable (no leaks)
- [ ] Works with all supported ML potentials
- [ ] Input validation handles edge cases

---

## Appendix: Mathematical Formulas

### A.1 NEB Force Projection

**Improved Tangent** (Henkelman-Jónsson, 2000):

```
If E_{i+1} > E_i > E_{i-1}:
    τ_i = R_{i+1} - R_i

Else if E_{i+1} < E_i < E_{i-1}:
    τ_i = R_i - R_{i-1}

Else:
    ΔE_+ = max(|E_{i+1} - E_i|, |E_{i-1} - E_i|) if E_{i+1} > E_{i-1}
    ΔE_- = min(|E_{i+1} - E_i|, |E_{i-1} - E_i|) otherwise

    τ_i = ΔE_+ * (R_{i+1} - R_i) + ΔE_- * (R_i - R_{i-1})

τ_i ← τ_i / |τ_i|  (normalize)
```

**NEB Projected Force**:

```
F_NEB^i = F_true^⊥ + F_spring^∥

F_true^⊥ = F_true - (F_true · τ_i) τ_i

F_spring^∥ = k_i * (|R_{i+1} - R_i| - |R_i - R_{i-1}|) τ_i
```

**Climbing Image Force**:

```
F_CI^i = F_true - 2(F_true · τ_i) τ_i
```

**Dynamic Spring Constant** (ORCA-style):

```
k_i = k_max - (k_max - k_min) * exp(-k_decay * ΔE_i / ΔE_max)

where:
    E_ref = max(E_{i-1}, E_{i+1})
    ΔE_i = max(E_i - E_ref, 0)
    ΔE_max = max{ΔE_j} over all images j
```

### A.2 IDPP (Image Dependent Pair Potential)

**Energy Function**:

```
E_IDPP = Σ_{i<j} w_ij * (1/d_ij - 1/d_ij^target)^2

where:
    d_ij = |R_i - R_j|  (actual pair distance)
    d_ij^target = interpolated distance between endpoints
    w_ij = (1 / d_ij^target^p)^2  (weight, p=2 typically)
```

**Force**:

```
F_IDPP^k = -∂E_IDPP/∂R_k = Σ_j F_kj

F_kj = -4 * w_ij * (1/d_ij - 1/d_ij^target) *
       (R_k - R_j) / d_ij^3
```

### A.3 RFO (Rational Function Optimization)

**Augmented Hessian**:

```
H_aug = [ H    g  ]
        [ g^T  0  ]
```

**RFO Step**:

```
Solve eigenvalue problem:
    (H - λI) δ = -g

For minimization: λ < λ_min (most negative eigenvalue)
For maximization: λ > λ_max (most positive eigenvalue)
```

**Trust Radius Enforcement**:

```
If |δ| > r_trust:
    δ ← δ * r_trust / |δ|
```

**Model Agreement Ratio**:

```
ρ = ΔE_actual / ΔE_predicted

If ρ > 0.75:  r_trust ← min(2 * r_trust, r_max)
If ρ < 0.25:  r_trust ← max(0.5 * r_trust, r_min)
```

### A.4 Dimer Curvature

**Finite Difference Estimate**:

```
κ = n^T H n ≈ (F_1 - F_2) · n / (2δ)

where:
    R_1 = R_0 + δ*n
    R_2 = R_0 - δ*n
    F_1 = Force at R_1
    F_2 = Force at R_2
    δ = dimer distance (0.005 Å typically)
```

**Rotation Force**:

```
F_rot = Hn - (Hn · n)n
```

**Translation Force** (when κ < 0):

```
F_trans = F_0 - 2(F_0 · n)n  (flip perpendicular component)
```

---

## Summary

MAPLE is a mature, actively developed computational chemistry toolkit with a **world-class implementation of transition state search algorithms**. The TS module represents **4,000+ lines of production-quality code**, featuring:

✅ **6 distinct TS methods** (NEB, PRFO, Dimer, String, AutoNEB)
✅ **Advanced features**: IDPP, Kabsch alignment, dynamic springs, CI-NEB, adaptive growth
✅ **Recent improvements**: Dimer bug fix, CINEB debugging, AutoNEB v1, PRFO redesign
✅ **Clean architecture**: JobABC base class, dataclass parameters, comprehensive logging
✅ **Production-ready**: Extensive examples, test cases, detailed output files

The project is **actively maintained** (most recent commit: Feb 2026) with continuous improvements to TS algorithms. The enhance branch shows ongoing development with a focus on robustness and new features.

**For TS development**: The codebase follows excellent software engineering practices with clear separation of concerns, reusable utilities, and comprehensive documentation. New algorithms can be easily integrated following the established patterns.

---

**Document Version**: 1.0
**Last Updated**: February 6, 2026
**Author**: Development Team
**Repository**: https://github.com/ClickFF/MAPLE
