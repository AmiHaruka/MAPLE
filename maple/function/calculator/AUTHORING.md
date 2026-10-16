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
- `get_hvp(self, atoms, n)` — optional analytic path for Dimer.
  Returns the 3-tuple `(Hn, forces, energy)` as
  torch tensors: `Hn` is the Hessian–vector product `H·n` flattened to `(3N,)`,
  `forces` is the `(3N,)` force vector, and `energy` is a scalar — all in
  Hartree units (energy in Hartree, forces in Hartree/Å, `H` in Hartree/Å²).
  Dimer prefers this path; an absent method or `NotImplementedError` selects
  a joint batch of positive displacement, negative displacement and center.
  ASE constraints also select force differences to include their energy and
  force adjustments. Other calculation errors propagate. `CalcABC.get_hvp` raises
  `NotImplementedError` by default — there is **no** shared autograd default,
  because the forward shape differs per backend. ANI, AIMNet2, the MACE
  calculators and exported FENNIX implement it through their own native inputs.
- `calculate_batch(self, atoms_list, properties=None, batch_size=None)` —
  `CalcABC` supplies ordered grouping, chunking and sequential fallback.
  Energy is `[B]`; forces, charges and Hessians are per-member lists; fixed
  shapes are stacked. This call preserves ASE's previous single-structure state.
  Dispatcher helpers also compose MAPLE's UMA and single-structure plug-ins without
  this method. A separately implemented `get_hessian` remains usable for batch H requests.

SP, BatchLBFGS, BatchPRFO and ready path images consume joint properties through
`_batch.py`. Algorithms retain coordinates, optimizer history and acceptance
state. Output functions receive existing E/F and never invoke a calculator.
See the [dispatcher integration record](model/ORIGNAL/docs/function_docs/dispatcher-batch.md)
for input grouping, member filenames and performance validation.

**Class attributes** (read by `SetCalculator` before the class is
instantiated):

| Attribute | Type | Purpose |
|---|---|---|
| `MODEL_NAMES` | `tuple[str, ...]` | Registry routing keys, lowercase. |
| `MODEL_ENERGY_UNIT` | `'eV'` or `'hartree'` | Declares the native model energy unit. Public E/F/H/HVP use Hartree and Å; pass `unit='hartree'` explicitly if supplying already converted E/F to `_finalize_results`. |
| `SUPPORTED_HESSIAN_MODES` | `tuple[str, ...]` | Subset of `('analytic', 'numerical')`. |
| `SUPPORTS_CHARGE_MULT` | `bool` | True if the backend honors `atoms.info['charge']` / `atoms.info['mult']`. |
| `SUPPORTS_PBC` | `bool` | True only when the backend constructs a validated periodic graph / neighbor list. `SetCalculator` and `CalcABC.calculate()` reject periodic atoms for false values. |
| `CHECKPOINT_FILENAME` | `dict[str, str] \| None` | Per-name filename for HuggingFace auto-download. `None` if no auto-download. |
| `REQUIRES_LOCAL_MODEL_FILE` | `bool` | Fallback when `CHECKPOINT_FILENAME` does not cover the requested name. |
| `OPTION_KEYS` | `tuple[str, ...] \| None` | Supported backend-specific `model_options`. Shipped backends set this so typos fail loudly; `None` keeps legacy plug-ins permissive. |
| `MODEL_PATH_OPTION` | `str \| None` | Constructor kwarg that consumes an explicit user `model_path` (`'model_path'` or `'checkpoint_path'`). `None` rejects an explicit `model_path` **only when** the constructor also exposes no `model_path`/`checkpoint_path` parameter — `SetCalculator` falls back to inspecting the ctor signature for support detection. The resolved path is still delivered through `build_kwargs_from_options(..., resolved_model_path=...)`, so legacy plug-ins that omit `MODEL_PATH_OPTION` must route that value into their constructor kwargs explicitly. |

Current AIMNet2 uses `OPTION_KEYS=('lr', 'lr_cutoff', 'dtype')` and configures
the exported model directly. Family-specific options are validated against
the calculator's `OPTION_KEYS`.

## Shared implementation and family ownership

`calculator_base.py` owns the ASE protocol, cache state and result conventions.
Builtin names stay in `set_calculator.py`'s `_BUILTIN_NAME_TO_MODULE`, each
family's `MODEL_NAMES`, and `CommandControl.SUPPORTED_MODELS`. A regression
check keeps these declarations consistent; concrete backends are imported only
when selected. The base registry still handles classes and external plug-ins.
`_inputs.py` provides shared input validation and native tensor/graph preparation;
family calculators select its fields and configuration. These files belong to
the calculator's shared foundation; native packing stays outside `CalcABC`.
Adapter-selected periodic-cell checks run in `_validate_geometry`, before
neighbor construction and device transfer, within the existing geometry pass.

ANI, AIMNet2 and the three MACE calculators use `_torch.py` for artifact loading,
derivatives and Tensor result splitting. MACE constructors handle their buffer
dtype conversion. FENNIX uses `_jax.py` for standard export loading, saved-VJP
derivatives, compiled coordinate/cell strain and native array result splitting;
family code binds native fields and model calls. `compile_strain` accepts field
names, and `array_properties` accepts already computed arrays without calling
the model. Property results reuse base unit conversion. `_jax_inputs.py` retains BIO1 preprocessing.
`CalcABC` does not import either backend at module load.
`_batch.py` supplies the existing multi-structure composition behind
`calculate_batch`; algorithm iteration and task scheduling stay with consumers.

Torch loading requires boolean `supports_native_batch` and
`supports_native_pbc_batch` fields in the PT. Concrete calculators read those
fields in `_calculate_batch`; they do not copy them onto the calculator.
JAX loading checks the export's target platforms and, with `require_vjp=True`,
its saved first VJP. FENNIX requests that VJP for its validated derivative paths.
Standard `model.call` checks input trees, dtypes and shape constraints.

`CalcABC._calculate_batch` returns `NotImplemented` by default. Concrete methods
return a property dictionary for native execution or `NotImplemented` before
forward when the current request needs single-structure composition. Exceptions
propagate; only explicit allocation failures reduce the chunk size. ANI keeps
external D4 on the single-structure route, as does GBSA for the complete energy.
UMA currently has no MAPLE private batch method and uses its existing `calculate`.

Private batch methods receive the requested property names and return those keys.
The public batch helper preserves distinct `energy` and `free_energy` values;
`force_consistent=True` requests `free_energy` together with forces. Current
static MLIPs expose the same complete potential under both ASE energy fields,
matching `_finalize_results`. Their Torch/JAX property tools convert the energy
array once and supply either or both requested fields without another forward.

`mace-les-off` uses the existing MACE calculator with a standalone PT and its
native LES long-range energy. It supports vacuum and full three-dimensional PBC;
partial PBC is rejected by the input builder. The LES export backports MACE
PR #1478's positions/cell strain wiring. Its native stress and virial include
the long-range derivatives and use the ordinary MACE property path.
Latent charges remain internal model quantities; no physical `charges` or BEC
property is supplied for this model. SP keeps its existing E/F output.

`_batch_key(atoms)` describes input compatibility for the current request. Its
default is `()`; AIMNet2 and FENNIX separate gas and periodic inputs, POLAR
groups by PBC mask, and periodic ANI also groups by cell. MACE graphs can combine different
cells. The public batch helper uses these keys without model-family labels,
restores input order, and preserves ASE state on success or error.

FENNIX supports `fennix-bio-1s` and `fennix-bio-1m`, both using `.jax` artifacts.
Native `.fnx` weights are offline export sources; the custom checkpoint runtime
and generic `fennol` model entry have been removed. `ORIGNAL` is the user's
personal wrapper-export maintenance directory and is outside the runtime
dependency graph.

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
    SUPPORTS_PBC = False               # fail fast on periodic atoms unless validated
    CHECKPOINT_FILENAME = {'foo2x': 'foo2x.pt', 'foo1ccx': 'foo1ccx.pt'}
    REQUIRES_LOCAL_MODEL_FILE = False
    OPTION_KEYS = ('foo_mode',)
    MODEL_PATH_OPTION = 'model_path'

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
        atoms = super().calculate(atoms, properties, system_changes)
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
- Raw model E/F enter `_finalize_results` in `MODEL_ENERGY_UNIT`, or with an
  explicit `unit`. The base `_convert_properties(result, source_unit=...)`
  converts native property dictionaries without modifying the inputs or
  moving Tensors. `_convert_energy_force_units` uses the same conversion.
  Torch property resolution and FENNIX call this boundary once;
  `_store_properties` finalizes their results with `unit='hartree'`.
  Batch results do not enter ASE's `self.results`.
- Stress uses eV/Å³; virial uses Hartree. Keep these conversions separate from
  E/F conversion. NumPy output is FP64; the model retains its artifact precision.
- The Hessian path is independent: `_analytic_hessian` must return
  Hartree / Å² directly. A Hessian obtained through `get_hessian` is added
  after converting other native properties, so it is not converted twice.
  Native eV Hessians use `_convert_properties`. The numerical path inherits Hartree from displaced
  forces, using `calculate_batch` where available and the ASE protocol otherwise.
  The existing `hessian_via_double_autograd` helper remains available to
  external plug-ins, including the bundled calculator example.

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
- `None`, `none`, `null`, `false`, `0`, and empty strings normalize to
  `none`. `implicit='gbsa'` requires a real solvent name such as
  `solvent='water'`; MAPLE fails early instead of looking for `None.dat`.

## Hessian

- `self.hessian` selects `'analytic'` or `'numerical'`. Numerical falls
  through to the shared `numerical_hessian_from_atoms` helper for free.
- `numerical_hessian_from_atoms` restores the calculator's pre-call
  `results` before returning, so standalone `calc.get_hessian(atoms)` does
  not leave `results` pointing at the final displaced geometry.
- Analytic Hessian with implicit solvent is unsupported and raises
  `NotImplementedError` from `CalcABC.get_hessian`. Document the
  limitation in any backend-specific notes.

## Periodic boundary conditions and stress

- `SUPPORTS_PBC = False` is the default. If any `atoms.pbc` component is
  true, `SetCalculator` rejects the model before construction and
  `CalcABC.calculate()` rejects direct backend use. This prevents molecular
  wrappers from silently treating periodic systems as isolated clusters.
- AIMNet2's hand wrapper is no-PBC. `coulomb_method='ewald'` is disabled
  until cell/PBC/MIC inputs and reference tests exist; use `simple` or `dsf`.
- The hand-wrapped MACE backends are no-PBC and do not provide validated
  stress/virial output. Use UMA or a backend-native periodic calculator for
  periodic stress/NPT workflows.
- UMA delegates PBC graph construction to FAIR-Chem and is the only shipped
  backend with `SUPPORTS_PBC = True`; MAPLE currently rejects UMA
  `stress`/`virial` requests until unit conversion is validated. Periodic UMA
  calculations must set an explicit FAIR-Chem `task=` (`omat`, `oc20`, `oc22`,
  `oc25`, `omc`, or `odac`); MAPLE no longer silently maps every periodic
  system to `omat`, and rejects `task='omol'` with periodic atoms.

## Charge / multiplicity

- `atoms.info['charge']` and `atoms.info['mult']` carry the values. Backends
  that require integer charge/spin inputs must reject non-integer values rather
  than truncating them.
- Set `SUPPORTS_CHARGE_MULT = True` if the backend honors them; otherwise
  `SetCalculator` warns the user that the values will be ignored.
- MACE-POLAR rejects non-integer `mult` before converting multiplicity to the
  model's unpaired-electron `spin = mult - 1` input.
- UMA `omol` charged/open-shell inputs are passed through to FAIR-Chem and
  emit a warning until MAPLE has accepted golden numerical tolerances for
  those states.
- UMA non-`omol` tasks reject non-default `charge`/`mult` because FAIR-Chem's
  current calculator contract only uses charge/spin for the `omol` head.

## Backend-specific kwargs

- Override `build_kwargs_from_options(cls, model, options, *,
  resolved_model_path=None)` to translate input-header `model_options`
  plus the factory-resolved checkpoint path into ctor kwargs. **Do not
  call back into `SetCalculator` from a backend method.** The factory
  passes `resolved_model_path` for backends that need a local file.
- Set `OPTION_KEYS` for every shipped backend-specific option. Unknown keys
  are rejected before model construction so a misspelled production config
  cannot be silently ignored.
- If explicit `model_path` is supported, set `MODEL_PATH_OPTION`. If a user
  passes `model_path` to a backend that does not support it, MAPLE raises a
  configuration error instead of silently falling back to a default checkpoint.

## HVP override

- `CalcABC.get_hvp` raises `NotImplementedError`. Override it only if this
  model will use the HVP-enabled Dimer path (`use_hvp=True`); regular Dimer can
  fall back to finite-difference forces. Torch adapters differentiate the complete
  energy; exported FENNIX differentiates its native forces through the saved VJP.
- Unlike the `calculate()` path, `get_hvp` does **not** route through
  `_finalize_results`, so the backend converts units itself: an eV-native
  backend must apply `EV2HARTREE` inside `get_hvp` and return Hartree-unit
  tensors. ANI is Hartree-native, so its implementation needs no conversion.
- ANI HVP rejects implicit-solvent runs because solvent HVP is not implemented;
  returning gas-phase HVP in that mode would be a silent mixed-model result.

## Backend capability matrix

Torch adapters execute complete TorchScript artifacts without native model
libraries. Periodic and native-batch capabilities follow the qualified artifact
and its input conditions. Explicit model paths must satisfy the same contract.

| Backend (names) | PBC | charge/mult | Hessian | Implicit solvent | D4 | HVP (Dimer) |
|---|---|---|---|---|---|---|
| ANI | yes, per artifact | no | analytic + numerical | molecular energy only | molecular only; no extra D4 on ANI2DR | yes; no implicit solvent |
| AIMNet2 | 3D PBC; auto/simple/dsf/ewald/pme | per checkpoint; spin conditioning requires NSE | analytic + numerical | molecular energy only | native dispersion follows artifact | yes |
| MACE-OFF / EGRET | yes | per artifact conditioning | analytic + numerical | molecular energy only | no extra option | yes |
| MACE-OMOL | yes | native categorical conditioning | analytic + numerical | molecular energy only | no extra option | yes |
| MACE-POLAR | yes; field/phi supported | multiplicity converted once in PT | analytic + numerical | molecular energy only | no extra option | yes |
| FENNIX-BIO1 S/M (`.jax`) | 3D PBC | charge; mult=1 | analytic + numerical | molecular energy only | native terms follow artifact | yes |
| UMA (`uma`) | yes; non-PBC auto `omol`; PBC requires explicit non-`omol` task; stress rejected | `omol` only (`spin = mult`); non-`omol` rejects non-default charge/mult | numerical only | yes | no | no |

MACE-POLAR receives MAPLE multiplicity and converts it once inside the PT.
UMA's existing FAIR-Chem route receives multiplicity directly. Preserve each
artifact/runtime's conditioning convention.

Observed on a local uma-s-1p1 checkpoint: direct FAIR-Chem and the MAPLE UMA
wrapper agree for H₂O `q=0/+1/-1` and `mult=3`, so charge/spin reaches
FAIR-Chem through MAPLE. The `q=0 → +1` same-geometry energy change is small
(~4.6e-5 Ha) for that checkpoint, so keep a warning and verify charged/open-
shell energetics against FAIR-Chem/reference calculations before relying on
them for production chemistry.

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

- **Public**: `calculate`, `calculate_batch`, `get_hessian`, `get_hvp`, the protocol class
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


### Workflow Hessian consumers

`_batch.hessian_one(calc, atoms, batch_size=None)` retains the existing `get_hessian(atoms)` protocol and sends explicitly chunked numerical Hessians through `calculate_batch`. The numeric branch retains calculator PBC/solvent guards even when every atom is fixed. Freq accepts a supplied Hessian for analysis without another model call; IRC retains each method's existing propagation sequence. Output helpers consume supplied E/F and resolve electronic-state metadata through `_inputs._state` without inference.
