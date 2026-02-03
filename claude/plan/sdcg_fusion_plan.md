# SDCG Fusion Optimizer Plan

## Overview
Fuse Steepest Descent (SD) and Conjugate Gradient (CG) into a single phased optimizer `SDCG`, with optional GDIIS acceleration. SD provides robust initial descent; CG (PRP+ variant) provides faster convergence near the minimum.

## File Changes

### 1. NEW: `maple/function/dispatcher/optimization/algorithm/SDCG.py`
The main implementation file, replacing SD.py as the active optimizer.

**SDCGParams dataclass:**
```python
@dataclass
class SDCGParams:
    # General
    max_step: float = 0.2
    max_iter: int = 256
    write_traj: bool = False
    traj_every: int = 1
    verbose: int = 1

    # Phase control
    sd_enabled: bool = True         # Enable SD phase
    cg_enabled: bool = True         # Enable CG phase
    sd_max_iter: int = 50           # Max SD iterations before forced CG switch
    cg_switch_fmax: float = 0.0     # Switch to CG when max_f < threshold (0.0 = auto)

    # CG parameters
    cg_restart_threshold: float = 0.2  # Powell restart: |f_k·f_{k-1}| >= threshold * ||f_k||^2
    cg_beta_method: str = "prp+"       # CG variant (prp+ only for now)

    # GDIIS parameters
    diis_enabled: bool = True
    diis_store_every: int = 5
    diis_min_snapshots: int = 3
    diis_memory: int = 6
```

**SDCG class (inherits JobABC):**
- Phase state machine: `_phase = "sd"` or `"cg"`
- SD phase: identical to current SD.py logic (step = max_step * forces, clip)
- CG phase: PRP+ conjugate gradient with Powell restart
- GDIIS: works in both phases, reset on phase transition
- Barzilai-Borwein step scale estimation (carried over from SD)

**PRP+ CG Algorithm:**
```
d_0 = f_0  (first CG direction = steepest descent)
For k >= 1:
    beta = max(dot(f_k, f_k - f_{k-1}) / dot(f_{k-1}, f_{k-1}), 0)  # PRP+

    # Powell restart condition
    if |dot(f_k, f_{k-1})| >= 0.2 * dot(f_k, f_k):
        beta = 0  # restart to SD direction

    d_k = f_k + beta * d_{k-1}

    # Step size: use max_step scaling (same as SD), clip per-atom
    step = max_step * d_k / max(|d_k|)  # normalize so max displacement = max_step
    x_{k+1} = x_k + step
```

**SD → CG Transition Logic:**
1. Auto mode (`cg_switch_fmax = 0.0`): threshold = `0.5 * initial_max_f` (computed at iter 0)
2. Explicit mode: user sets `cg_switch_fmax` to desired value
3. Iteration-based: always switch after `sd_max_iter` SD steps
4. Transition triggers on whichever condition is met first
5. On transition: reset GDIIS, log phase change, start CG from current forces

**User Control Modes:**
- `#opt(method=sd)` → `sd_enabled=True, cg_enabled=False` (SD only)
- `#opt(method=cg)` → `sd_enabled=False, cg_enabled=True` (CG only)
- `#opt(method=sdcg)` → `sd_enabled=True, cg_enabled=True` (default fusion)
- `diis_enabled=false` → disable GDIIS in any mode

### 2. MODIFY: `maple/function/dispatcher/optimization/algorithm/SD.py`
Convert to backward-compatibility stub:
```python
from .SDCG import SDCG, SDCGParams

# Backward compatibility aliases
SD = SDCG
SDParams = SDCGParams
```
Keep the legacy function-based `DIIS()` reference in DIIS.py unchanged.

### 3. MODIFY: `maple/function/dispatcher/optimization/algorithm/__init__.py`
```python
from .LBFGS import LBFGS, LBFGSParams
from .RFO import RFO, RFOParams
from .DIIS import DIIS, DIISAccelerator, DIISParams, OptimizationStorage
from .SDCG import SDCG, SDCGParams
from .SD import SD, SDParams  # backward compat
```

### 4. MODIFY: `maple/function/dispatcher/optimization/optimization.py`
Add routing for `sdcg` and `cg`, update `sd` routing:
```python
elif method in ('sd', 'sdcg', 'cg'):
    from .algorithm import SDCG
    opt = SDCG(self.atoms, output=self.output, paras=self.commandcontrol)
    return opt.run()
```

### 5. MODIFY: `maple/function/read/command_control.py`
Add `"sdcg"` to IMPLEMENTATION_MAP:
```python
"opt": {"lbfgs", "rfo", "sd", "cg", "sdcg", ""},
```

### 6. NEW: `example/opt/sdcg/inp1.inp`
Test input for SDCG fusion mode (same molecule as SD tests):
```
#model=uma
#opt(method=sdcg)
#device=gpu0
```

### 7. NEW: `example/opt/cg/inp1.inp`
Test input for CG-only mode:
```
#model=uma
#opt(method=cg)
#device=gpu0
```

## Implementation Details

### Phase Transition in `run()`:
```python
# In the main loop:
if self._phase == "sd" and self.params.cg_enabled:
    should_switch = False
    if sd_iter_count >= self.params.sd_max_iter:
        should_switch = True
    if max_f < cg_switch_threshold:
        should_switch = True
    if should_switch:
        self._switch_to_cg(iteration)
```

### CG Step Computation:
```python
def _cg_step(self, forces: np.ndarray) -> np.ndarray:
    if self._cg_prev_forces is None:
        # First CG step = SD direction
        direction = forces.copy()
    else:
        # PRP+ beta
        df = forces - self._cg_prev_forces
        beta = np.dot(forces.ravel(), df.ravel()) / max(np.dot(self._cg_prev_forces.ravel(), self._cg_prev_forces.ravel()), 1e-20)
        beta = max(beta, 0.0)  # PRP+ clamp

        # Powell restart
        if abs(np.dot(forces.ravel(), self._cg_prev_forces.ravel())) >= self.params.cg_restart_threshold * np.dot(forces.ravel(), forces.ravel()):
            beta = 0.0

        direction = forces + beta * self._cg_prev_direction

    # Normalize and scale
    max_disp = np.abs(direction).max()
    if max_disp > 0:
        step = self.params.max_step * direction / max_disp
    else:
        step = direction
    step = self._clip_step(step)

    # Store for next iteration
    self._cg_prev_forces = forces.copy()
    self._cg_prev_direction = direction.copy()

    return step
```

### GDIIS Integration:
- Same `_try_diis_acceleration()` method as current SD.py
- Called in both SD and CG phases at `diis_store_every` intervals
- On phase transition: `self.diis.reset()` + `self._sd_step_counter = 0`
- Step validation identical: reject if energy rises AND force increases

## Files Touched (Summary)
| File | Action |
|------|--------|
| `algorithm/SDCG.py` | CREATE - main implementation |
| `algorithm/SD.py` | MODIFY - backward-compat stub |
| `algorithm/__init__.py` | MODIFY - add SDCG imports |
| `optimization/optimization.py` | MODIFY - add routing |
| `read/command_control.py` | MODIFY - add "sdcg" |
| `example/opt/sdcg/inp1.inp` | CREATE - test input |
| `example/opt/cg/inp1.inp` | CREATE - test input |

## Verification
1. Run SDCG default mode: `example/opt/sdcg/inp1.inp` - should show SD phase then CG phase transition
2. Run SD-only mode: `example/opt/sd/inp1.inp` with `method=sd` - should behave identically to current SD
3. Run CG-only mode: `example/opt/cg/inp1.inp` - should skip SD phase entirely
4. Run with DIIS disabled: add `diis_enabled=false` - should show no GDIIS attempts
5. Check .out files for phase transition logging, convergence, and trajectory output
