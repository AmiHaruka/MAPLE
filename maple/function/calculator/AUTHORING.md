# Authoring MAPLE calculators

MAPLE backends are registered classes that satisfy a narrow protocol. This
document is the contract a new calculator must honor and the migration path
for downstream code that depended on the legacy entry points.

## What MAPLE expects from a calculator

A backend is any class that exposes the public surface below; it does **not**
have to inherit `CalcABC`. UMA, for example, extends third-party
`FAIRChemCalculator` and satisfies the protocol by attribute presence.

**Public methods:**
- `calculate(self, atoms, properties, system_changes)` — ASE entry point.
- `get_hessian(self, atoms, delta=0.002)` — returns a `(3N, 3N) np.ndarray`
  in Hartree / Å². `CalcABC` provides a default that dispatches on
  `self.hessian`.
- `get_hvp(self, atoms, n)` — optional; only required if Dimer-mode TS will
  run on the backend. `CalcABC.get_hvp` raises `NotImplementedError` by
  default — there is **no** shared autograd default, because the forward
  shape differs per backend. `ANICalculator` implements it for ANI's
  `self.model(species, coords)` shape; other backends must override it
  before Dimer-mode TS can run on them.

**Class attributes** (read by `SetCalculator` before the class is
instantiated):

| Attribute | Type | Purpose |
|---|---|---|
| `MODEL_NAMES` | `tuple[str, ...]` | Registry routing keys, lowercase. |
| `MODEL_ENERGY_UNIT` | `'eV'` or `'hartree'` | Declares the backend's `calculate()` energy unit. `_finalize_results` converts to Hartree based on this — **do not multiply by `EV2HARTREE` yourself**. |
| `SUPPORTED_HESSIAN_MODES` | `tuple[str, ...]` | Subset of `('analytic', 'numerical')`. |
| `SUPPORTS_CHARGE_MULT` | `bool` | True if the backend honors `atoms.info['charge']` / `atoms.info['mult']`. |
| `CHECKPOINT_FILENAME` | `dict[str, str] \| None` | Per-name filename for HuggingFace auto-download. `None` if no auto-download. |
| `REQUIRES_LOCAL_MODEL_FILE` | `bool` | Fallback when `CHECKPOINT_FILENAME` does not cover the requested name. |

## Minimum runnable subclass

```python
from maple.function.calculator import CalcABC, register_calculator
import numpy as np


@register_calculator
class FooCalculator(CalcABC):
    MODEL_NAMES = ('foo2x', 'foo1ccx')
    MODEL_ENERGY_UNIT = 'eV'           # or 'hartree' — declare honestly per backend
    SUPPORTED_HESSIAN_MODES = ('analytic', 'numerical')
    SUPPORTS_CHARGE_MULT = False
    CHECKPOINT_FILENAME = {'foo2x': 'foo2x.pt', 'foo1ccx': 'foo1ccx.pt'}
    REQUIRES_LOCAL_MODEL_FILE = False

    implemented_properties = ['energy', 'free_energy', 'forces']

    @classmethod
    def build_kwargs_from_options(cls, model, options, *, resolved_model_path=None):
        # Translate input-header options + factory-resolved path into ctor kwargs.
        return {}

    def __init__(self, device, model, *, implicit='none', solvent='none', **backend_kwargs):
        super().__init__()
        self.device = device
        self.hessian = 'analytic'
        self.implicit_solv_init(implicit=implicit, solvent=solvent)
        # ...load self.model, etc...

    def calculate(self, atoms=None, properties=['energy'], system_changes=None):
        super().calculate(atoms, properties, system_changes)
        energy, forces = self._forward(atoms)      # private, backend-internal
        self._finalize_results(atoms, energy=energy, forces=forces)

    def _analytic_hessian(self, atoms) -> np.ndarray:
        # backend autograd; return np.ndarray (3N, 3N) in Hartree / Å²
        ...
```

## Unit contract

- Set `MODEL_ENERGY_UNIT` honestly. ANI 's TorchScript model returns Hartree
  natively, so ANI declares `'hartree'` and `_finalize_results` skips the
  conversion. Every other shipped backend declares `'eV'`.
- `_finalize_results` is the **only** path that converts the
  `calculate()` energy / forces flow to Hartree. **Custom calculators must
  not multiply by `EV2HARTREE` themselves** for the `calculate()` path.
- The Hessian path is independent: `_analytic_hessian` must return
  Hartree / Å² directly (each eV-native backend multiplies by `EV2HARTREE`
  inside its analytic method); the numerical path inherits Hartree via
  `numerical_hessian_from_atoms`, which calls back into `calculate()`.

## Implicit solvent

- `_finalize_results` is the only place that adds the GBSA correction.
  **Custom calculators do not call `implicit_solv_energy_and_force()`
  directly** for the `calculate()` flow. If a backend needs special
  handling, override `_finalize_results` rather than duplicating the
  solvent path.
- Solvent setup happens via `init_implicit_solvent(calc, implicit,
  solvent, device)`. `CalcABC.__init__` does not call this for you in the
  current release; subclasses still invoke `self.implicit_solv_init(...)`
  inside their own `__init__`.

## Hessian

- `self.hessian` selects `'analytic'` or `'numerical'`. Numerical falls
  through to the shared `numerical_hessian_from_atoms` helper for free.
- Analytic Hessian with implicit solvent is unsupported and raises
  `NotImplementedError` from `CalcABC.get_hessian`. Document the
  limitation in any backend-specific notes.

## Charge / multiplicity

- `atoms.info['charge']` and `atoms.info['mult']` carry the values.
- Set `SUPPORTS_CHARGE_MULT = True` if the backend honors them; otherwise
  `SetCalculator` warns the user that the values will be ignored.

## Backend-specific kwargs

- Override `build_kwargs_from_options(cls, model, options, *,
  resolved_model_path=None)` to translate input-header `model_options`
  plus the factory-resolved checkpoint path into ctor kwargs. **Do not
  call back into `SetCalculator` from a backend method.** The factory
  passes `resolved_model_path` for backends that need a local file.

## HVP override

- `CalcABC.get_hvp` raises `NotImplementedError`. Override it if Dimer-mode
  TS will run on this model. `ANICalculator` is the only shipped backend
  with an implementation (autograd over the `(species, coords)` forward);
  the rest fail loudly rather than misread a differently-shaped forward.

## Backend capability matrix

What the shipped backends actually support today. The hand-wrapped MACE
backends build a **no-PBC** radius graph (zero `cell`/`shifts`), so they are
molecule-only regardless of any periodic claim elsewhere. UMA is the only
backend that switches tasks for periodic input.

| Backend (names) | PBC | charge/mult | Hessian | Implicit solvent | D4 | HVP (Dimer) |
|---|---|---|---|---|---|---|
| ANI (`ani2x/1x/1ccx/1xnr`) | no | no | analytic + numerical | yes | yes | yes |
| AIMNet2 (`aimnet2`, `aimnet2nse`) | no | yes | analytic + numerical | yes | no | no |
| MACE-OFF (`maceoff23s/m/l`, `egret`) | no | no | analytic + numerical | yes | no | no |
| MACE-omol (`maceomol`) | no | no | analytic + numerical | yes | no | no |
| MACE-POLAR (`macepols/m/l`) | no, no external field | yes (`spin = mult − 1`) | analytic + numerical | yes | no | no |
| UMA (`uma`) | yes (auto `omol`/`omat`) | yes (`spin = mult`) | numerical only | yes | no | no |

`spin` semantics differ on purpose: MACE-POLAR's traced interface takes the
number of unpaired electrons (`mult − 1`), UMA's FAIR-Chem path takes the
spin multiplicity (`mult`). Confirm against the specific checkpoint before
trusting open-shell results — neither encoding is verified here.

Open question (observed, unresolved): on H₂O a `q=0 → +1` change moves the
energy by ~0.5 Ha for AIMNet2 and MACE-POLAR but only ~5e-5 Ha for UMA
(uma-s-1p1, omol). Charge *is* reaching the FAIR-Chem model (the response is
nonzero, and the default `a2g` carries `r_data_keys=['spin','charge']`), so
this is a charge/spin-handling question for the UMA checkpoint, not a missing
wire-up in MAPLE. Verify UMA charged/open-shell energetics before relying on
them.

## Plug-in discovery — three layers

| Layer | Mechanism | Status |
|---|---|---|
| 1 | Input header `#model=mymodel(module=my_lab.maple_plugin, model_path=/tmp/m.pt)` | Implemented |
| 2 | Env var `MAPLE_CALCULATOR_PLUGINS=my_lab.maple_plugin,other.plugin` | Implemented (loaded once per process at first `_build_calculator`) |
| 3 | Python entry points: `[project.entry-points."maple.calculators"]` | Preview only; not implemented |

The recommended `pyproject.toml` shape for Layer 3, so packagers can prepare
ahead of time:

```toml
[project.entry-points."maple.calculators"]
my_lab = "my_lab.maple_plugin"
```

## Public vs private API

- **Public**: `calculate`, `get_hessian`, `get_hvp`, the six class
  attributes above.
- **Private** (do not depend on from outside the calculator): `_forward_energy`,
  `_build_inputs`, `_analytic_hessian`, `self.model`.
- **Banned**: `calc.get_energy(...)` (removed from every backend), reading
  `self.results` before `calculate()` has been invoked.

## Migration from `calc.get_energy(...)`

Downstream code that previously called `calc.get_energy(atoms)` must
switch to ASE's standard pattern:

```python
atoms.calc = calc
energy = atoms.get_potential_energy()
```

`get_energy` carried three different signatures across the shipped
backends; no single contract was honest. The ASE accessor reads
`calc.results['energy']` after `calculate()` runs, which is now the only
public energy entry point.

## Registration collisions

`@register_calculator` raises `ValueError` on a duplicate name rather than
silently overwriting. If you see this error, the registry already has a
class registered under one of your `MODEL_NAMES` entries — a collision is
a real bug, not a feature.

## UMA is a documented exception

UMA does not inherit `CalcABC` because it already extends
`FAIRChemCalculator`. It satisfies the protocol via attribute presence
and registers via `@register_calculator`. Do not write
`isinstance(calc, CalcABC)` anywhere in the dispatcher — the calculator
surface is duck-typed by design.
