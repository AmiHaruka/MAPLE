# **MA**chine-learning **P**otential for **L**andscape **E**xploration (**MAPLE**)

![MAPLE Concept](./maple.jpg)

MAPLE is a machine-learning-potential-native computational chemistry toolkit for
geometry optimization, transition-state search, reaction-path analysis, molecular
dynamics, and related post-processing workflows.

## Core Capabilities

| Category | Methods |
|----------|---------|
| **Optimization** | L-BFGS, RFO, SD, CG, SD-CG, GDIIS |
| **Transition State** | NEB, CI-NEB, P-RFO, Dimer, String/GSM, AutoNEB |
| **Reaction Path** | IRC with GS, LQA, HPC, EulerPC |
| **Dynamics** | NVE, NVT, NPT |
| **Analysis** | Frequency, PES Scan, Single Point |
| **ML Potentials** | ANI, AIMNet2, MACE, MACEPol, UMA |
| **Extras** | D4 dispersion, explicit solvent cluster builder, experimental GB-polar SP energy correction, UMA/FAIR-Chem-backed PBC, restart files, DCD output |

## Installation

### Requirements

- Python >= 3.10
- CUDA-capable GPU recommended for production workloads

MAPLE separates dependencies into three groups:

| Group | Installed by `pip install -e .` | Purpose |
|-------|----------------------------------|---------|
| Core | Yes | Base MAPLE runtime and general scientific I/O |
| Optional tools | Only when explicitly requested | Plotting and development tools |
| External ML runtimes | No | User-selected PyTorch/CUDA and FAIR-Chem stacks |

Core dependencies declared by MAPLE:

| Package | Minimum version | Notes |
|---------|-----------------|-------|
| `ase` | `>=3.22` | Atomic structures, calculators, I/O |
| `numpy` | `>=1.20` | Numerical arrays |
| `scipy` | `>=1.7` | Scientific routines |

External runtime dependencies that users install manually:

| Package | Version boundary | Required for | Why MAPLE does not auto-install it |
|---------|------------------|--------------|------------------------------------|
| `torch` | `>=2.0` | ANI, AIMNet2, MACE-OFF, MACE-O-MOL, MACE-Polar, UMA | PyTorch wheels must match the user's CUDA/CPU runtime and should be selected from the official PyTorch index. |
| `fairchem-core` | FAIR-Chem release with UMA support; tested locally with `2.19.0` | UMA and FAIR-Chem-backed/PBC workflows | FAIR-Chem may impose its own compatible PyTorch/runtime constraints, so install it after the matching PyTorch wheel. |

### Install MAPLE

```bash
git clone https://github.com/ClickFF/MAPLE.git
cd MAPLE
pip install -e .
```

### Install Dependencies

The `pip install -e .` command above installs only MAPLE's core dependencies.
Install PyTorch separately for your hardware. Examples:

```bash
# PyTorch example: CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CPU-only PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Install FAIR-Chem only if you need UMA or FAIR-Chem-backed/PBC workflows:

```bash
pip install fairchem-core
```

Model checkpoint boundary:

- MAPLE auto-downloads only the model files hosted at https://huggingface.co/Wayne7815/MAPLE_models.
- Auto-downloads use a pinned HuggingFace revision by default; set `MAPLE_MODEL_REVISION` only when intentionally refreshing model assets.
- Backend-specific or local checkpoints, such as MACE-Polar `.pt` files, must be present in `maple/function/calculator/model/` or supplied through an explicit model path.
- UMA checkpoints are resolved through an explicit path, a local `maple/function/calculator/model/uma-*.pt` file, or FAIR-Chem's official model-loading path.

PBC boundary:

- PBC support is currently available through UMA/FAIR-Chem-backed workflows only.
- ANI, AIMNet2, MACE-OFF, MACE-O-MOL, and MACE-Polar are molecular no-PBC wrappers in MAPLE and fail fast when periodic atoms are supplied.
- AIMNet2 `coulomb_method=ewald` is disabled until validated cell/PBC/MIC inputs and reference tests exist; use `simple` or `dsf`.
- UMA stress/virial requests are rejected until MAPLE validates stress-unit conversion.

### Batch acceleration contract

The batch evaluators preserve the selected model, interaction rules, and
energy/force units. Batch and single evaluations are not guaranteed to be
bitwise identical. This branch also changes some optimization, Hessian, and
transition-state contracts; it is not a transparent scheduling-only update.

- Native batching is used only when all structures share one calculator,
  carry no unsupported constraints/PBC/solvent state, and the backend declares
  `supports_batch_energy_forces=True`. Unsupported cases use the validated
  single-structure path with the ASE calculator cache restored afterward.
- Every batch result is checked for requested fields, batch cardinality,
  per-structure atom shapes, exact integer padding, and finite E/F/H values.
  Unknown properties and ambiguous checkpoint output layouts fail instead of
  being truncated or guessed.
- ANI batches identical atomic-number sequences; D4, implicit solvent, and PBC
  remain sequential. AIMNet2 batching accepts the padded per-molecule output
  contract only when the checkpoint SHA256 matches the validated packaged
  `aimnet2.pt`/`aimnet2nse.pt` identities; custom or changed checkpoints remain
  sequential. AIMNet2 `simple` Coulomb always uses every non-self atom pair
  within each molecule; only `dsf` uses a finite LR cutoff. The closed-shell
  checkpoint rejects `mult != 1`, while AIMNet2-NSE receives the required
  per-molecule multiplicity tensor. MACE-OFF and MACE-O-MOL use disconnected no-PBC graphs;
  MACE-Polar stays sequential. UMA native batching is limited to neutral
  closed-shell molecular inputs; charged/open-shell inputs retain the
  single-structure FAIR-Chem path.
- `batch_size`, `path_batch_size`, `scan_batch_size`, `fd_batch_size`, and
  `hvp_batch_size` bound evaluation chunks. Unset evaluator limits default to
  `auto`; request `all` explicitly to attempt the whole workload at once.
  `auto` never rounds above its memory estimate, and recognized CUDA OOMs
  retry in ordered half-size chunks. Rigid scans flush bounded chunks instead
  of retaining the full grid of `Atoms` objects.
- Multi-structure optimization invokes the new BatchLBFGS path explicitly by
  supplying multiple structures. It rejects constraints/empty structures,
  rejects non-finite search state, applies per-structure Armijo backtracking
  plus step clipping, records failure status by original structure index, and
  does not write `_opt.xyz` for non-finite, rejected-step, or max-iteration
  failures. The generic ASE-compatible adapter still crosses a CPU NumPy
  boundary; this is a compatibility path rather than an end-to-end GPU claim.

Batch evaluation does not certify a transition state. NEB/PRFO convergence
produces `_ts_candidate.xyz`; max-iteration PRFO termination produces only
`_prfo_unconverged.xyz` and fails closed. Production first-order-saddle claims
still require an independent frequency check (exactly one imaginary mode) and
forward/reverse IRC endpoint validation. Experimental `BatchPRFO` remains
runtime-disabled. Scalar `PRFO` now requires an explicit rigid-symmetry
contract when the calculator does not declare one: use
`rigid_symmetry=free_molecule` only for a validated isolated, rigid-motion-
invariant PES, or `rigid_symmetry=cartesian_external` for a laboratory-frame
potential. The same option is forwarded to `nebts` refinement; unknown
symmetry fails closed rather than guessing from geometry.

### Changed scientific and output contracts

- PRFO uses a restricted-step P-RFO update rather than the earlier dual-shift
  step. A geometry candidate now requires converged forces, exactly one
  significant negative **physical-space** Hessian mode (internal for a free
  molecule, full Cartesian for a laboratory-frame PES), resolved physical
  curvature,
  and a small *unrestricted current-point* Newton correction in Cartesian
  coordinates. A small previous or trust-limited step alone is not a proximity
  test. These checks do not replace independent frequency and IRC validation.
- `rigid_symmetry=auto` rejects calculators without an explicit
  `rigid_body_invariant` capability. No packaged backend currently declares
  this capability; even bundled AIMNet2 checkpoints did not pass the frozen
  E/F/H covariance screen for automatic admission. Existing PRFO and TS
  refinement inputs therefore need a scientifically justified explicit
  `rigid_symmetry` setting until their backend is admitted. Unknown or
  laboratory-frame potentials must not be assumed to be free molecules.
- `get_hessian()` defaults to `constraint_mode='fixed_cartesian'`. It supports
  `FixAtoms` and `FixCartesian` but rejects nonlinear `FixInternals` rather than
  return a misleading Cartesian matrix. `raw_cartesian` is an explicit
  unconstrained derivative, **not** a constrained RFO Hessian. Relaxed RFO
  scans are unsupported; use an explicit `#scan(method=lbfgs,mode=relaxed)`
  input or a rigid scan. The former RFO methanol-scan output is retained only
  as a historical artifact under `examples/scan/rfo/`.
- Numerical Hessians retain a hard default antisymmetry diagnostic. Its
  dimensionless scaled residual is `max|H-Hᵀ| / max(1, max|H|)`, with H
  expressed in Hartree/Å² and the scale floor equal to 1 Hartree/Å². A small
  residual does not prove accuracy. There is no automatic float32 threshold
  relaxation. Inspect step-size and reference-curvature evidence before using
  the explicit warning/ignore policies.
- PRFO and refinements write `_prfo_ts_candidate`, `_nebts_ts_candidate`, or
  `_stringts_ts_candidate` instead of the former `_ts` filenames. No success
  alias is written. Single-point trajectories flush completed frame windows in
  input order; on a later failure the valid prefix remains in the output,
  without a success summary.

## Quick Start

### Command Line

```bash
maple input.inp
maple input.inp output.out
maple --version
maple md nve
```

### Minimal Example

```text
#model=uma(size=uma-s-1p1)
#opt(method=lbfgs)
#device=gpu0

0 1
C   -0.748   0.014   0.025
C    0.748  -0.014  -0.025
O    1.170   0.016   1.330
H   -1.155  -0.888  -0.460
H   -1.096   0.888  -0.530
H   -1.155   0.049   1.065
H    1.148  -0.912   0.457
H    1.096   0.869   0.513
H    0.802   0.842   1.742
```
### External coordinates:
```
#model=uma(size=uma-s-1p1)
#opt(method=lbfgs)
#device=gpu0

XYZ 0 1 /path/to/molecule.xyz
```
TIPS:  Charge and spin multiplicity are supported only in the **OMOL task** mode of the **UMA** model and in the **AIMNet2 / AIMNet2-NSE** models.
## Input Overview

### Header Keywords

```text
#model=<model>
#<task>(options)
#device=<device>
```

### Common Tasks

| Header | Description |
|--------|-------------|
| `#opt(method=lbfgs)` | Geometry optimization |
| `#sp` | Single-point energy |
| `#ts(method=neb)` | Transition-state search |
| `#freq` | Frequency analysis |
| `#irc(method=gs)` | Intrinsic reaction coordinate |
| `#scan(method=lbfgs)` | PES scan |
| `#md(mdp=nvt.mdp)` | Molecular dynamics |

### UMA Options

`#model=uma(...)` accepts the following keys (all optional):

| Key | Values | Default | Notes |
|-----|--------|---------|-------|
| `size` | `uma-s-1p1`, `uma-s-1p2`, `uma-m-1p1` | `uma-s-1p1` | Checkpoint variant |
| `task` | `omol`, `omat`, `oc20`, `odac`, `omc`, `oc22`, `oc25` | `omol` for non-periodic systems | Periodic UMA requires an explicit periodic task such as `omat`, `oc20`, `oc22`, `oc25`, `omc`, or `odac` |
| `inference` | `default`, `turbo` | `default` | `turbo` accelerates fixed-composition GPU workloads (NEB / TS / freq); ignored on CPU |

UMA follows FAIR-Chem's charge/spin contract: charge and spin multiplicity are
honored only for the `omol` molecular head.  Non-`omol` tasks reject non-neutral
`charge`/`mult`/`spin` metadata instead of silently ignoring charged or
open-shell input.  As a MAPLE fail-closed policy extension, `task=omol` is
also rejected for PBC systems; choose a periodic task such as `omat`, `oc20`,
`odac`, `omc`, `oc22`, or `oc25` instead.

### Coordinates

Inline coordinates:

```text
#model=uma(size=uma-s-1p1,task=omol,inference=default)
#sp
#device=gpu0

0 1
C   0.000   0.000   0.000
H   1.089   0.000   0.000
...
```

External coordinates:

```text
#model=uma(size=uma-s-1p1,task=omol,inference=default)
#sp
#device=gpu0

XYZ 0 1 /path/to/molecule.xyz
```

Multi-structure jobs such as NEB accept multiple `XYZ` records.<br>
TIPS:  Charge and spin multiplicity are supported only in the **OMOL task** mode of the **UMA** model

MAPLE supports custom explicit-solvent PDB templates; see the
[solvent documentation](https://www.maplechem.org/functions/solvent.html)
for usage guidance.

## Documentation

- Website: https://www.maplechem.org/
- Release history: https://github.com/ClickFF/MAPLE/releases
- Architecture notes: [ARCHITECTURE.md](ARCHITECTURE.md)
- Authoring a calculator backend: [maple/function/calculator/AUTHORING.md](maple/function/calculator/AUTHORING.md)

## Citation

```text
https://github.com/ClickFF/MAPLE
```

## Contributing

1. Fork the repository.
2. Create a feature branch.
3. Make changes with clear commits.
4. Open a pull request.

## Acknowledgments

- [ASE](https://wiki.fysik.dtu.dk/ase/)
- [PyTorch](https://pytorch.org/)
- [AIMNet2](https://github.com/isayevlab/AIMNet2)
- [FAIR-Chem](https://github.com/FAIR-Chem/fairchem)

**Version**: 0.1.4<br>
**Status**: Active Development<br>
**Updated**: May 2026<br>
