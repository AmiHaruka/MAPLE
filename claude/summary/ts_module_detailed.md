# MAPLE Transition State (TS) Module - Detailed Technical Analysis

## Table of Contents
1. [Module Structure](#module-structure)
2. [Architecture Overview](#architecture-overview)
3. [Algorithms Deep Dive](#algorithms-deep-dive)
4. [Mathematical Foundations](#mathematical-foundations)
5. [Integration Points](#integration-points)
6. [Design Patterns](#design-patterns)

---

## Module Structure

### File Organization

```
maple/function/dispatcher/ts/
├── ts.py                    # Main dispatcher/router (157 lines)
├── __init__.py             # Module exports
└── algorithm/              # Algorithm implementations
    ├── __init__.py        # Algorithm exports
    ├── neb.py             # Nudged Elastic Band (1,517 lines)
    ├── PRFO.py            # Partitioned RFO (793 lines)
    ├── dimer.py           # Dimer method (521 lines)
    ├── string.py          # Growing String Method (GSM)
    ├── afir.py            # DSAFIR (Double-Sphere AFIR)
    ├── descafir.py        # Descent from AFIR
    ├── newton.py          # Newton optimizer
    ├── BPRFO.py           # Bofill-PRFO variant
    ├── autoneb.py         # Automated multi-step NEB
    └── logger.py          # Logging utilities
```

### Lines of Code Distribution
- **NEB**: 1,517 lines (most comprehensive)
- **PRFO**: 793 lines
- **Dimer**: 521 lines
- **String/AFIR**: ~400 lines each
- Total TS module: ~4,000+ lines

---

## Architecture Overview

### Class Hierarchy

```
JobABC (Abstract Base Class)
  ↑
  │ (inherits)
  │
  ├── NEB                # Nudged Elastic Band
  ├── PRFO               # Partitioned RFO
  ├── Dimer              # Dimer method
  ├── GSM                # Growing String Method
  ├── DSAFIR             # Double-Sphere AFIR
  ├── DescAFIR           # Descent from AFIR
  ├── BPRFO              # Bofill-PRFO
  └── AutoNEB            # Automated NEB pipeline
```

### JobABC Base Class

**Location**: [maple/function/dispatcher/jobABC.py](../../maple/function/dispatcher/jobABC.py)

**Key Responsibilities**:
- Parameter initialization from nested dictionaries
- Case-insensitive parameter handling
- Logging infrastructure
- Abstract interface enforcement

**Key Methods**:
```python
class JobABC(ABC):
    def __init__(self, output: str):
        self.output = output
        self.logger = self._setup_logger()

    def _init_params(self, ParamsClass, paras, aliases=None):
        """Initialize dataclass parameters with case-insensitive matching"""
        # 1. Lower all keys in nested dict
        # 2. Select relevant subdict using aliases
        # 3. Instantiate dataclass with extracted values
        # 4. Return typed parameter object

    @abstractmethod
    def run(self):
        """Algorithm implementation - must be overridden"""
        pass

    def log_info(self, msg):
        """Write info message to log file"""

    def log_error(self, msg):
        """Write error message to log file"""
```

### TransitionState Dispatcher

**Location**: [ts.py:14-157](../../maple/function/dispatcher/ts/ts.py#L14-L157)

```python
class TransitionState:
    def __init__(self, output, atoms, params, method='newton'):
        self.output = output
        self.atoms = atoms      # Can be Atoms, Molecules, or List[Atoms]
        self.params = params    # Nested dict from command_control
        self.method = method.lower()

    def run(self):
        """Route to specific algorithm based on method"""
        if self.method == 'neb':
            return NEB(self.output, self.atoms, self.params).run()
        elif self.method == 'prfo':
            return PRFO(self.output, self.atoms, self.params).run()
        elif self.method == 'dimer':
            return Dimer(self.output, self.atoms, self.params).run()
        # ... other methods
```

**Supported Methods**:
1. `newton` - Newton-Raphson optimizer
2. `prfo` - Partitioned Rational Function Optimization
3. `bprfo` - Bofill-PRFO variant
4. `neb` - Nudged Elastic Band
5. `string` - Growing String Method
6. `dimer` - Dimer method
7. `afir` - Artificial Force Induced Reaction
8. `descafir` - Descent from AFIR
9. `autoneb` - Automated multi-step NEB

---

## Algorithms Deep Dive

### 1. NEB (Nudged Elastic Band)

**Location**: [algorithm/neb.py](../../maple/function/dispatcher/ts/algorithm/neb.py)

**Purpose**: Find minimum energy path (MEP) between reactant and product configurations.

#### Parameters (NEBParams)

```python
@dataclass
class NEBParams:
    # Path setup
    n_images: int = 11                  # Number of images (excluding endpoints)

    # Spring constants
    k_min: float = 0.01                 # Minimum spring constant (Eh/Angstrom²)
    k_max: float = 0.30                 # Maximum spring constant
    k_decay: float = 1.0                # Decay rate for dynamic springs
    use_dynamic_k: bool = True          # Enable dynamic spring constants

    # Optimization
    algorithm: str = 'lbfgs'            # Optimizer (lbfgs/fire)
    max_steps: int = 500                # Maximum optimization steps
    fmax: float = 0.05                  # Force convergence (Eh/Angstrom)

    # Initial path
    use_idpp: bool = True               # Use IDPP smoothing
    idpp_maxstep: float = 0.2           # IDPP max step size
    idpp_fmax: float = 0.1              # IDPP convergence

    # Refinement
    refine: Optional[str] = None        # 'cineb', 'nebts', or None
    cineb_fmax: float = 0.02            # CI-NEB convergence
    nebts_fmax: float = 0.02            # NEBTS convergence

    # Advanced
    use_improved_tangent: bool = True   # Henkelman-Jonsson tangent
    climb: bool = False                 # Start with climbing image
    parallel: bool = False              # Parallel image evaluation
    remove_rotation: bool = True        # Kabsch alignment
```

#### Core Algorithm

**Step 1: Initial Path Generation**
```python
def _generate_initial_path(self, reactant, product, n_images):
    """
    Create initial path using linear interpolation + optional IDPP smoothing
    """
    # 1. Linear interpolation
    images = [reactant.copy()]
    for i in range(1, n_images + 1):
        alpha = i / (n_images + 1)
        interpolated = reactant.copy()
        interpolated.positions = (1-alpha)*reactant.positions + alpha*product.positions
        images.append(interpolated)
    images.append(product.copy())

    # 2. IDPP smoothing (if enabled)
    if self.params.use_idpp:
        images = self._run_idpp_smoothing(images)

    return images
```

**Step 2: IDPP (Image Dependent Pair Potential)**
```python
def _run_idpp_smoothing(self, images):
    """
    Optimize path using pair-distance potential
    V_IDPP = Σ(i<j) (1/d_ij - 1/d_ij^target)²

    where d_ij^target is linearly interpolated between endpoints
    """
    # Target distances from linear interpolation
    target_distances = self._compute_target_distances(images)

    # Optimize internal images to match target distances
    for idx in range(1, len(images)-1):
        atoms = images[idx]
        optimizer = LBFGS(atoms)
        optimizer.run(fmax=self.params.idpp_fmax)

    return images
```

**Step 3: NEB Force Calculation**
```python
def neb_forces(images, energies, k_springs, improved_tangent=True):
    """
    Compute NEB projected forces for all images

    F_NEB = F_true⊥ + F_spring∥

    where:
      F_true⊥ = F_true - (F_true · τ)τ     # Perpendicular component
      F_spring∥ = k(|R_{i+1}-R_i| - |R_i-R_{i-1}|)τ  # Parallel component
      τ = unit tangent vector
    """
    neb_forces = []
    max_force = 0.0
    hei_idx = -1
    max_energy = -float('inf')

    for i in range(1, len(images)-1):  # Skip endpoints
        # Calculate tangent
        if improved_tangent:
            tau = improved_tangent(
                images[i-1].positions,
                images[i].positions,
                images[i+1].positions,
                energies[i-1], energies[i], energies[i+1]
            )
        else:
            tau = (images[i+1].positions - images[i-1].positions)

        tau = tau / np.linalg.norm(tau)

        # True force (from calculator)
        F_true = images[i].get_forces()

        # Perpendicular component
        F_true_perp = F_true - np.dot(F_true.ravel(), tau.ravel()) * tau

        # Spring force (parallel)
        R_forward = images[i+1].positions - images[i].positions
        R_backward = images[i].positions - images[i-1].positions
        dist_forward = np.linalg.norm(R_forward)
        dist_backward = np.linalg.norm(R_backward)

        k = k_springs[i]
        F_spring_par = k * (dist_forward - dist_backward) * tau

        # Combined NEB force
        F_neb = F_true_perp + F_spring_par
        neb_forces.append(F_neb)

        # Track highest energy image
        if energies[i] > max_energy:
            max_energy = energies[i]
            hei_idx = i
            max_force = np.max(np.abs(F_neb))

    return neb_forces, max_force, hei_idx
```

**Step 4: Improved Tangent (Henkelman-Jonsson)**
```python
def improved_tangent(Rm1, R, Rp1, Em1, E, Ep1):
    """
    Energy-weighted tangent for better barrier description

    Cases:
    1. E_{i+1} > E_i > E_{i-1}  → τ = R_{i+1} - R_i     (uphill)
    2. E_{i+1} < E_i < E_{i-1}  → τ = R_i - R_{i-1}     (downhill)
    3. Mixed                     → τ = ΔE_max·τ_max + ΔE_min·τ_min (blended)
    """
    tau_plus = Rp1 - R
    tau_minus = R - Rm1

    dE_plus = abs(Ep1 - E)
    dE_minus = abs(E - Em1)

    if Ep1 > E > Em1:  # Ascending
        return tau_plus / np.linalg.norm(tau_plus)
    elif Ep1 < E < Em1:  # Descending
        return tau_minus / np.linalg.norm(tau_minus)
    else:  # Mixed - blend based on energy differences
        dE_max = max(dE_plus, dE_minus)
        dE_min = min(dE_plus, dE_minus)

        if Ep1 > Em1:
            tau = dE_max * tau_plus + dE_min * tau_minus
        else:
            tau = dE_min * tau_plus + dE_max * tau_minus

        return tau / np.linalg.norm(tau)
```

**Step 5: Dynamic Spring Constants**
```python
def compute_dynamic_k(energies, k_min, k_max, k_decay):
    """
    ORCA-style dynamic spring constants

    k_i = k_max - (k_max - k_min) * exp(-k_decay * ΔE_i / ΔE_max)

    - Near barrier (high ΔE): k → k_min (softer, allows climbing)
    - Flat regions (low ΔE): k → k_max (stiffer, prevents drift)
    """
    E_max = max(energies)
    E_min = min(energies)
    dE_ref = E_max - E_min

    k_springs = []
    for E in energies:
        dE = E - E_min
        ratio = dE / dE_ref if dE_ref > 1e-6 else 0.0
        k = k_max - (k_max - k_min) * np.exp(-k_decay * ratio)
        k_springs.append(k)

    return k_springs
```

**Step 6: Climbing Image NEB (CI-NEB)**
```python
def cineb_forces(images, energies, k_spring):
    """
    Climbing Image NEB for highest energy image

    F_CI = F_true - 2(F_true · τ)τ

    Inverts parallel component to push upward along MEP
    """
    hei_idx = np.argmax(energies[1:-1]) + 1  # Highest energy image

    cineb_forces = []
    for i in range(1, len(images)-1):
        if i == hei_idx:
            # Climbing image: invert parallel force
            tau = improved_tangent(...)
            F_true = images[i].get_forces()
            F_par = np.dot(F_true.ravel(), tau.ravel()) * tau
            F_ci = F_true - 2 * F_par  # Inversion
            cineb_forces.append(F_ci)
        else:
            # Regular NEB force
            cineb_forces.append(regular_neb_force(i))

    return cineb_forces, hei_idx
```

**Step 7: Optimization Loop**
```python
def run(self):
    """Main NEB execution"""
    # 1. Generate initial path
    images = self._generate_initial_path(reactant, product, n_images)

    # 2. Regular NEB optimization
    for step in range(self.params.max_steps):
        energies = [img.get_potential_energy() for img in images]
        k_springs = compute_dynamic_k(energies, k_min, k_max, k_decay)
        forces, fmax, hei_idx = neb_forces(images, energies, k_springs)

        if fmax < self.params.fmax:
            break

        # Update positions using L-BFGS
        self.optimizer.step(forces)

    # 3. CI-NEB refinement (if requested)
    if self.params.refine == 'cineb':
        for step in range(cineb_steps):
            forces, hei_idx = cineb_forces(images, energies)
            if max_force < self.params.cineb_fmax:
                break
            self.optimizer.step(forces)

    # 4. TS optimization with PRFO (if requested)
    if self.params.refine == 'nebts':
        ts_atoms = images[hei_idx].copy()
        PRFO(self.output, ts_atoms, {...}).run()

    # 5. Write output
    self._write_mep(images)
    self._write_hei(images[hei_idx])
```

#### Key Features
- **IDPP smoothing**: Prevents unphysical initial paths
- **Dynamic springs**: Adapts to local energy landscape
- **Improved tangents**: Energy-weighted for accurate barriers
- **CI-NEB**: Accurate TS location without extra images
- **NEBTS**: Direct TS optimization from highest image
- **Kabsch alignment**: Removes rigid body rotation

---

### 2. PRFO (Partitioned Rational Function Optimization)

**Location**: [algorithm/PRFO.py](../../maple/function/dispatcher/ts/algorithm/PRFO.py)

**Purpose**: Optimize transition states using rational function approximation with trust radius control.

#### Parameters (PRFOParams)

```python
@dataclass
class PRFOParams:
    max_steps: int = 500                # Maximum iterations
    trust_radius: float = 0.2           # Initial trust radius (Angstrom)
    trust_min: float = 0.05             # Minimum trust radius
    trust_max: float = 0.5              # Maximum trust radius
    eta_shrink: float = 0.75            # Threshold for shrinking
    eta_expand: float = 1.75            # Threshold for expanding
    fmax: float = 0.0001                # Convergence criterion (Eh/Angstrom)
    use_mass_weighted: bool = True      # Mass-weighted coordinates
    mode_type: str = 'auto'             # 'auto', 'ts', 'min'
    target_mode: Optional[int] = None   # Mode to follow (0=lowest)
    hessian_update: str = 'bofill'      # 'bofill', 'powell', 'bfgs'
    recalc_hessian_every: int = 10      # Hessian recalculation frequency
```

#### Core Algorithm

**RFO Step Function**
```python
def prfo_step(H, g, is_ts=True, target_mode=None, trust_radius=0.2):
    """
    Partitioned Rational Function Optimization step

    Dual-shift RFO: separate treatment for uphill and downhill modes

    Standard RFO:
        min/max  (g·s + 0.5·s·H·s) / (1 + s·s)

    Lagrangian form:
        (H - μI)s = -g

    Partitioning:
        - Minus set (uphill): maximize along this eigenvector
        - Plus set (downhill): minimize along these eigenvectors
    """
    # 1. Diagonalize Hessian
    eigenvalues, eigenvectors = np.linalg.eigh(H)

    # 2. Identify mode to follow
    if target_mode is None:
        if is_ts:
            # Follow most negative mode
            target_mode = 0 if eigenvalues[0] < 0 else None

    # 3. Partition eigenspaces
    if target_mode is not None:
        minus_indices = [target_mode]  # Uphill mode
        plus_indices = [i for i in range(len(eigenvalues)) if i != target_mode]
    else:
        minus_indices = []
        plus_indices = list(range(len(eigenvalues)))

    # 4. Solve for each subspace with independent μ
    s_minus = solve_rfo_subspace(
        eigenvalues[minus_indices],
        eigenvectors[:, minus_indices],
        g,
        trust_radius,
        maximize=True  # σ = -1
    )

    s_plus = solve_rfo_subspace(
        eigenvalues[plus_indices],
        eigenvectors[:, plus_indices],
        g,
        trust_radius,
        maximize=False  # σ = +1
    )

    # 5. Combine steps
    s = s_minus + s_plus

    return s
```

**Bisection for Trust Radius**
```python
def solve_rfo_subspace(eigenvalues, eigenvectors, gradient, R, maximize):
    """
    Solve (H - μI)s = -g subject to ||s|| = R using bisection

    RFO equation in eigenspace:
        s_i = -g_i / (λ_i - μ)

    Trust radius constraint:
        Σ_i [g_i / (λ_i - μ)]² = R²

    Bisection searches for μ that satisfies constraint
    """
    # Project gradient onto eigenspace
    g_eigen = eigenvectors.T @ gradient

    # Bisection bounds
    if maximize:
        mu_lower = eigenvalues.min() - 10.0
        mu_upper = eigenvalues.min() - 1e-6
    else:
        mu_lower = eigenvalues.min() - 10.0
        mu_upper = eigenvalues.max() + 10.0

    # Bisection loop
    for iteration in range(100):
        mu = (mu_lower + mu_upper) / 2

        # Compute step in eigenspace
        s_eigen = -g_eigen / (eigenvalues - mu)
        step_norm = np.linalg.norm(s_eigen)

        if abs(step_norm - R) < 1e-6:
            break

        if step_norm > R:
            if maximize:
                mu_upper = mu
            else:
                mu_lower = mu
        else:
            if maximize:
                mu_lower = mu
            else:
                mu_upper = mu

    # Transform back to Cartesian
    s_cart = eigenvectors @ s_eigen
    return s_cart
```

**Trust Radius Adaptation**
```python
def update_trust_radius(trust_radius, rho, on_boundary):
    """
    Adapt trust radius based on model agreement

    rho = (E_actual - E_old) / (E_model - E_old)

    Decision rules:
    - rho < 0.75:  Poor agreement → reject step, shrink radius
    - rho > 1.75 and on_boundary:  Good agreement → expand radius
    - Otherwise: accept step, keep radius
    """
    if rho < 0.75:
        # Poor prediction - shrink
        trust_radius *= 0.5
        accept = False
    elif rho > 1.75 and on_boundary:
        # Excellent prediction and hitting boundary - expand
        trust_radius *= 1.5
        accept = True
    else:
        # Acceptable - maintain
        accept = True

    # Enforce bounds
    trust_radius = np.clip(trust_radius, trust_min, trust_max)

    return trust_radius, accept
```

**Mass-Weighted Coordinates**
```python
def to_mass_weighted(positions, masses):
    """
    Transform to mass-weighted coordinates
    q_MW = D @ q_Cart where D = diag(1/√m_i)

    Advantage: physically meaningful trust radius
    """
    sqrt_masses = np.sqrt(masses)
    D = np.diag(1.0 / sqrt_masses)
    q_mw = D @ positions.ravel()
    return q_mw, D

def transform_hessian_mw(H_cart, D):
    """Transform Hessian to mass-weighted coordinates"""
    return D @ H_cart @ D

def transform_gradient_mw(g_cart, D):
    """Transform gradient to mass-weighted coordinates"""
    return D @ g_cart
```

**Main Loop**
```python
def run(self):
    """Main PRFO optimization"""
    atoms = self.atoms.copy()

    # 1. Calculate initial Hessian
    H = atoms.calc.get_hessian(atoms)

    # 2. Mass-weighted transformation
    if self.params.use_mass_weighted:
        D = get_mass_weighted_matrix(atoms)
        H = D @ H @ D

    trust_radius = self.params.trust_radius

    # 3. Optimization loop
    for step in range(self.params.max_steps):
        # Calculate gradient
        E = atoms.get_potential_energy()
        g = -atoms.get_forces().ravel()

        if self.params.use_mass_weighted:
            g = D @ g

        # Check convergence
        if np.max(np.abs(g)) < self.params.fmax:
            break

        # Compute PRFO step
        s = prfo_step(H, g, is_ts=True, trust_radius=trust_radius)

        # Trial step
        atoms_trial = atoms.copy()
        if self.params.use_mass_weighted:
            s_cart = np.linalg.inv(D) @ s
        else:
            s_cart = s
        atoms_trial.positions += s_cart.reshape(-1, 3)

        # Evaluate trial
        E_trial = atoms_trial.get_potential_energy()

        # Model energy
        E_model = E + np.dot(g, s) + 0.5 * np.dot(s, H @ s)

        # Trust radius update
        rho = (E_trial - E) / (E_model - E)
        on_boundary = (np.linalg.norm(s) > 0.95 * trust_radius)
        trust_radius, accept = update_trust_radius(trust_radius, rho, on_boundary)

        if accept:
            atoms = atoms_trial
            # Hessian update (Bofill/Powell/BFGS)
            H = update_hessian(H, s, g_new - g, method=self.params.hessian_update)

        # Recalculate Hessian periodically
        if step % self.params.recalc_hessian_every == 0:
            H = atoms.calc.get_hessian(atoms)
            if self.params.use_mass_weighted:
                H = D @ H @ D

    # 4. Write output
    self._write_ts(atoms)
    return atoms
```

#### Key Features
- **Dual-shift partitioning**: Independent treatment of uphill/downhill modes
- **Trust radius control**: Adaptive step sizing based on model quality
- **Mass-weighted coordinates**: Physically meaningful distances
- **Mode tracking**: Follows specific eigenvector across iterations
- **Hessian updates**: Bofill/Powell/BFGS for reduced cost
- **Robust convergence**: Rejects bad steps automatically

---

### 3. Dimer Method

**Location**: [algorithm/dimer.py](../../maple/function/dispatcher/ts/algorithm/dimer.py)

**Purpose**: Minimum-mode following for transition state optimization without Hessian.

#### Parameters (DimerParams)

```python
@dataclass
class DimerParams:
    max_steps: int = 500                # Total iterations
    fmax: float = 0.0001                # Convergence (Eh/Angstrom)
    dimer_dist: float = 0.01            # Dimer separation (Angstrom)
    max_rot_iter: int = 10              # Rotation iterations per step
    trial_angle: float = 5.0            # Initial rotation angle (degrees)
    use_hvp: bool = False               # Use autograd HVP
    translation_method: str = 'cg'      # 'cg', 'lbfgs', 'fire'
    rotation_conv: float = 0.001        # Rotation convergence
    remove_translation: bool = True     # Project out translation
    remove_rotation: bool = True        # Project out rotation
```

#### Core Algorithm

**Dimer Concept**
```
    R1 •----o----• R2
           R0
           (midpoint)

where:
  R1 = R0 + δ·n    (image 1)
  R2 = R0 - δ·n    (image 2)
  δ = dimer_dist / 2
  n = dimer orientation (unit vector)
```

**Curvature Calculation**
```python
def compute_curvature(atoms, n, delta):
    """
    Compute curvature κ = n^T H n using finite differences

    κ = (F1 - F2) · n / (2δ)

    where F1, F2 are forces at R1, R2

    Alternatively, with HVP:
    κ = n^T (∇²E) n  (direct from autograd)
    """
    if use_hvp:
        Hn = atoms.calc.get_hvp(atoms, n)
        kappa = np.dot(n.ravel(), Hn.ravel())
    else:
        # Finite difference
        R0 = atoms.positions.copy()

        # Image 1: R0 + δn
        atoms.positions = R0 + delta * n
        F1 = atoms.get_forces()

        # Image 2: R0 - δn
        atoms.positions = R0 - delta * n
        F2 = atoms.get_forces()

        # Restore midpoint
        atoms.positions = R0

        # Curvature
        kappa = np.dot((F1 - F2).ravel(), n.ravel()) / (2 * delta)

    return kappa
```

**Rotation Phase**
```python
def rotate_dimer(atoms, n, delta, max_iter):
    """
    Minimize curvature by rotating dimer orientation

    F_rot = (I - nn^T) Hn
         = Hn - (Hn·n)n

    Update: n ← n - α·F_rot, then normalize
    """
    for rot_iter in range(max_iter):
        # Calculate curvature and rotational force
        kappa, Hn = compute_curvature_and_hvp(atoms, n, delta)

        # Rotational force (perpendicular to n)
        F_rot = Hn - np.dot(Hn.ravel(), n.ravel()) * n
        F_rot_norm = np.linalg.norm(F_rot)

        # Check convergence
        if F_rot_norm < self.params.rotation_conv:
            break

        # Update orientation (gradient descent)
        alpha = self.params.trial_angle * np.pi / 180  # radians
        n_new = n - alpha * F_rot / F_rot_norm
        n_new = n_new / np.linalg.norm(n_new)

        # Check improvement
        kappa_new = compute_curvature(atoms, n_new, delta)
        if kappa_new < kappa:
            n = n_new
        else:
            alpha *= 0.5  # Backtracking

    return n, kappa
```

**Translation Phase**
```python
def dimer_forces(atoms, n, kappa):
    """
    Compute effective force for translation

    F_eff = F_perp - F_par    (if κ < 0, uphill mode)
         = F_perp              (if κ ≥ 0, no mode)

    where:
      F_perp = F - (F·n)n       # perpendicular
      F_par = (F·n)n            # parallel
    """
    F = atoms.get_forces()
    F_par = np.dot(F.ravel(), n.ravel()) * n
    F_perp = F - F_par

    if kappa < 0:
        # Negative mode exists - invert parallel component
        F_eff = F_perp - F_par
    else:
        # Positive definite - standard minimization
        F_eff = F_perp

    return F_eff
```

**Main Loop**
```python
def run(self):
    """Main dimer optimization"""
    atoms = self.atoms.copy()

    # 1. Initialize dimer orientation (random or from Hessian)
    if self.params.initial_mode_vector is None:
        n = random_unit_vector(len(atoms) * 3)
    else:
        n = self.params.initial_mode_vector

    # 2. Optimization loop
    for step in range(self.params.max_steps):
        # Rotation phase: find minimum mode
        n, kappa = self.rotate_dimer(atoms, n, self.params.dimer_dist)

        # Translation phase: move along effective force
        F_eff = self.dimer_forces(atoms, n, kappa)

        # Check convergence
        fmax = np.max(np.abs(F_eff))
        if fmax < self.params.fmax:
            break

        # Update positions
        if self.params.translation_method == 'lbfgs':
            self.optimizer.step(F_eff)
        elif self.params.translation_method == 'cg':
            self.cg_optimizer.step(F_eff)
        elif self.params.translation_method == 'fire':
            self.fire_optimizer.step(F_eff)

        # Log progress
        self.log_info(f"Step {step}: E={E:.6f} Eh, kappa={kappa:.6f}, fmax={fmax:.6f}")

    # 3. Write output
    self._write_ts(atoms)
    return atoms
```

#### Key Features
- **No Hessian required**: Uses only force evaluations
- **Minimum-mode following**: Automatically finds negative eigenvalue direction
- **Rotation/translation separation**: Two-phase algorithm
- **HVP support**: Optional autograd for efficiency
- **Rigid body removal**: Projects out external modes

---

### 4. GSM (Growing String Method)

**Location**: [algorithm/string.py](../../maple/function/dispatcher/ts/algorithm/string.py)

**Purpose**: Grow reaction path from both ends until they meet, then refine.

#### Parameters (GSMParams)

```python
@dataclass
class GSMParams:
    max_nodes: int = 20                 # Maximum path nodes
    growth_step: float = 0.05           # Growth step size (Angstrom)
    max_growth_iter: int = 100          # Growth iterations
    fmax_growth: float = 0.05           # Growth convergence
    reparam_every: int = 5              # Reparameterization frequency
    use_climbing: bool = True           # CI-STRING after growth
    fmax_climbing: float = 0.02         # Climbing convergence
    refine_ts: bool = False             # PRFO refinement
```

#### Core Algorithm

**Growth Strategy**
```python
def grow_string(self):
    """
    Two-ended adaptive growth

    1. Add new node at each end
    2. Relax nodes perpendicular to tangent
    3. Reparameterize to equal arclength
    4. Repeat until paths meet
    """
    # Initialize with endpoints
    nodes = [reactant, product]

    while len(nodes) < self.params.max_nodes:
        # Add nodes at both ends
        node_new_front = self._add_node_front(nodes)
        node_new_back = self._add_node_back(nodes)

        nodes.insert(1, node_new_front)  # After reactant
        nodes.insert(-1, node_new_back)  # Before product

        # Relax internal nodes (perpendicular to tangent)
        for i in range(1, len(nodes)-1):
            tau = self._compute_tangent(nodes, i)
            self._relax_perpendicular(nodes[i], tau)

        # Reparameterize to equal arclength
        if step % self.params.reparam_every == 0:
            nodes = self._reparameterize(nodes)

        # Check for convergence
        if self._check_convergence(nodes):
            break

    return nodes
```

**Perpendicular Relaxation**
```python
def _relax_perpendicular(self, node, tau):
    """
    Optimize node in hyperplane perpendicular to tangent

    F_perp = F - (F · τ)τ
    """
    for iter in range(max_iter):
        F = node.get_forces()
        F_perp = F - np.dot(F.ravel(), tau.ravel()) * tau

        if np.max(np.abs(F_perp)) < fmax:
            break

        # Gradient descent
        node.positions += alpha * F_perp
```

**Reparameterization**
```python
def _reparameterize(self, nodes):
    """
    Redistribute nodes to equal arclength spacing

    1. Compute cumulative arclength
    2. Interpolate to uniform spacing
    3. Return new nodes
    """
    # Compute cumulative distances
    distances = [0.0]
    for i in range(1, len(nodes)):
        dist = np.linalg.norm(nodes[i].positions - nodes[i-1].positions)
        distances.append(distances[-1] + dist)

    total_length = distances[-1]
    target_spacing = total_length / (len(nodes) - 1)

    # Interpolate to uniform spacing
    new_nodes = [nodes[0]]  # Keep reactant
    for i in range(1, len(nodes)-1):
        target_dist = i * target_spacing
        # Find bracketing nodes and interpolate
        new_node = self._interpolate_at_distance(nodes, distances, target_dist)
        new_nodes.append(new_node)
    new_nodes.append(nodes[-1])  # Keep product

    return new_nodes
```

#### Key Features
- **Two-ended growth**: Efficient path finding
- **Perpendicular relaxation**: Energy minimization constrained to path
- **Reparameterization**: Equal arclength spacing
- **CI-STRING**: Climbing image after growth
- **Kabsch alignment**: Rigid body removal

---

### 5. DSAFIR (Double-Sphere AFIR)

**Location**: [algorithm/afir.py](../../maple/function/dispatcher/ts/algorithm/afir.py)

**Purpose**: Automatic reaction path discovery using artificial forces.

#### Parameters (DSAFIRParams)

```python
@dataclass
class DSAFIRParams:
    gamma: float = 100.0                # AFIR parameter (kJ/mol)
    n_models: int = 10                  # Number of reaction models
    max_cycles: int = 10                # LQA cycles
    lqa_conv: float = 1.0               # LQA convergence (kJ/mol)
    use_gradient: bool = True           # Use gradient-based LQA
    n_pathways: int = 1                 # Number of pathways to find
```

#### Core Algorithm

**AFIR Force**
```python
def afir_force(atoms, gamma, fragment_A, fragment_B):
    """
    Artificial Force Induced Reaction

    F_AFIR = F_true + F_artificial

    F_artificial = -γ · ∇ρ(r_AB)

    where ρ(r) = (Σ_i∈A Σ_j∈B 1/r_ij)^{-1}

    Effect: pushes fragments together to induce reactions
    """
    F_true = atoms.get_forces()

    # Compute artificial potential
    rho = 0.0
    grad_rho = np.zeros_like(atoms.positions)

    for i in fragment_A:
        for j in fragment_B:
            r_ij = atoms.positions[j] - atoms.positions[i]
            dist = np.linalg.norm(r_ij)

            rho += 1.0 / dist

            # Gradient of 1/r_ij
            grad_1_over_r = -r_ij / dist**3
            grad_rho[i] += grad_1_over_r
            grad_rho[j] -= grad_1_over_r

    # Artificial force
    F_artificial = -gamma * grad_rho / rho**2

    return F_true + F_artificial
```

**LQA (Local Quadratic Approximation)**
```python
def lqa_optimization(atoms, gamma, fragment_A, fragment_B):
    """
    Optimize with AFIR force, then find approximate TS

    1. Optimize with F_AFIR
    2. Approximate barrier top using energy profile
    3. Return approximate TS structure
    """
    trajectory = []

    # Optimize with AFIR
    for step in range(max_steps):
        F_afir = afir_force(atoms, gamma, fragment_A, fragment_B)
        atoms.positions += alpha * F_afir

        E = atoms.get_potential_energy()
        trajectory.append((atoms.copy(), E))

        if converged(F_afir):
            break

    # Find maximum energy structure (approximate TS)
    energies = [E for _, E in trajectory]
    max_idx = np.argmax(energies)
    approximate_ts = trajectory[max_idx][0]

    return approximate_ts, trajectory
```

#### Key Features
- **Automatic reaction discovery**: No initial TS guess needed
- **Multi-pathway search**: Finds multiple reaction channels
- **LQA integration**: Efficient barrier crossing
- **Fragment definition**: Flexible reaction site specification

---

## Mathematical Foundations

### 1. Projection Operators

**Parallel Projection**
```
F_∥ = (F · n)n
```

**Perpendicular Projection**
```
F_⊥ = F - F_∥ = F - (F · n)n
```

**Used in**: NEB, Dimer translation, GSM relaxation

### 2. Kabsch Alignment

**Purpose**: Remove rigid body rotation between structures

**Algorithm**:
```python
def kabsch_align(P, Q):
    """
    Find optimal rotation matrix R that minimizes ||P - RQ||

    1. Center both structures
    2. Compute covariance matrix C = P^T Q
    3. SVD: C = UΣV^T
    4. Rotation: R = VU^T (with det correction)
    """
    # Center
    P_center = P - P.mean(axis=0)
    Q_center = Q - Q.mean(axis=0)

    # Covariance
    C = P_center.T @ Q_center

    # SVD
    U, S, Vt = np.linalg.svd(C)

    # Rotation matrix
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1, 1, np.sign(d)])
    R = Vt.T @ D @ U.T

    # Apply rotation
    Q_aligned = (R @ Q_center.T).T + P.mean(axis=0)

    # RMSD
    rmsd = np.sqrt(np.mean((P - Q_aligned)**2))

    return Q_aligned, rmsd, R
```

### 3. Mass-Weighted Coordinates

**Transformation**:
```
q_MW = D @ q_Cart
where D = diag(1/√m_1, 1/√m_1, 1/√m_1, 1/√m_2, 1/√m_2, 1/√m_2, ...)
```

**Hessian in MW coordinates**:
```
H_MW = D @ H_Cart @ D
```

**Gradient in MW coordinates**:
```
g_MW = D @ g_Cart
```

**Advantage**: Trust radius in MW coordinates is physically meaningful (all coordinates have same units).

### 4. RFO Method

**Standard RFO**:
```
Minimize/Maximize: (g·s + 0.5·s·H·s) / (1 + s·s)
```

**Lagrangian form**:
```
(H - μI)s = -g
```

**In eigenspace**:
```
s_i = -g_i / (λ_i - μ)
```

**Trust radius constraint**:
```
Σ_i [g_i / (λ_i - μ)]² = R²
```

### 5. L-BFGS Update

**Two-loop recursion**:
```python
def lbfgs_two_loop(s_history, y_history, grad):
    """
    Compute H^{-1} @ grad using stored {s, y} pairs

    s_k = x_{k+1} - x_k       (position difference)
    y_k = g_{k+1} - g_k       (gradient difference)
    """
    q = grad.copy()
    alphas = []

    # Backward loop
    for s, y in reversed(list(zip(s_history, y_history))):
        rho = 1.0 / np.dot(y, s)
        alpha = rho * np.dot(s, q)
        q = q - alpha * y
        alphas.append(alpha)

    # Initial Hessian approximation
    if len(s_history) > 0:
        s_last = s_history[-1]
        y_last = y_history[-1]
        gamma = np.dot(y_last, s_last) / np.dot(y_last, y_last)
        r = gamma * q
    else:
        r = q

    # Forward loop
    for (s, y), alpha in zip(s_history, reversed(alphas)):
        rho = 1.0 / np.dot(y, s)
        beta = rho * np.dot(y, r)
        r = r + s * (alpha - beta)

    return r  # Approximate H^{-1} @ grad
```

---

## Integration Points

### 1. Calculator Interface (ASE)

All TS algorithms interact with ML potentials through ASE `Calculator` interface:

```python
# Energy
E = atoms.get_potential_energy(force_consistent=True)

# Forces (negative gradient)
F = atoms.get_forces()  # Shape: (N_atoms, 3), units: Eh/Angstrom

# Hessian (PRFO only)
H = atoms.calc.get_hessian(atoms)  # Shape: (3N, 3N), units: Eh/Angstrom²

# Hessian-vector product (Dimer with HVP)
Hn, forces, energy = atoms.calc.get_hvp(atoms, n)
```

### 2. Convergence Thresholds

Set by `Dispatcher` before job creation:

```python
# Gaussian-style thresholds attached to Atoms
atoms.f_max_th = 1.5e-4    # Eh/Angstrom
atoms.f_rms_th = 1.0e-4
atoms.dp_max_th = 6.0e-4   # Angstrom
atoms.dp_rms_th = 4.0e-4
```

Checked in algorithms:
```python
def check_convergence(atoms, forces, displacements):
    f_max = np.max(np.abs(forces))
    f_rms = np.sqrt(np.mean(forces**2))
    dp_max = np.max(np.abs(displacements))
    dp_rms = np.sqrt(np.mean(displacements**2))

    converged = (
        f_max < atoms.f_max_th and
        f_rms < atoms.f_rms_th and
        dp_max < atoms.dp_max_th and
        dp_rms < atoms.dp_rms_th
    )
    return converged
```

### 3. Parameter Flow

```
Input File (command_control)
    ↓
Dispatcher.set_threshold()
    ↓
TransitionState(output, atoms, params, method)
    ↓
Algorithm._init_params(ParamsClass, paras, ("alias1", "alias2"))
    ↓
DataClass instance (type-safe, case-insensitive)
```

Example:
```python
# In input file
$command_control
  neb
    N_Images = 12       # Case-insensitive
    K_MAX = 0.3
    Refine = CINEB
  end
$end

# In NEB.__init__
self.params = self._init_params(NEBParams, paras, ("neb",))

# Result: NEBParams(n_images=12, k_max=0.3, refine='cineb', ...)
```

### 4. Output Files

Generated by each algorithm:

```python
def _write_trajectory(self, images, suffix='_traj.xyz'):
    """Write XYZ trajectory"""
    filename = self.output + suffix
    with open(filename, 'w') as f:
        for atoms in images:
            f.write(f"{len(atoms)}\n")
            E = atoms.get_potential_energy()
            f.write(f"Energy: {E:.8f} Hartree\n")
            for symbol, pos in zip(atoms.symbols, atoms.positions):
                f.write(f"{symbol:2s} {pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f}\n")
```

Naming convention:
- NEB: `{output}_mep.xyz`, `{output}_hei.xyz`, `{output}_cineb_mep.xyz`
- PRFO: `{output}_prfo_ts.xyz`, `{output}_prfo_traj.xyz`
- Dimer: `{output}_dimer_ts.xyz`, `{output}_dimer_traj.xyz`
- GSM: `{output}_string_path.xyz`, `{output}_string_ts.xyz`

---

## Design Patterns

### 1. Template Method Pattern

`JobABC` defines the algorithm interface:
```python
class JobABC(ABC):
    def _init_params(self, ParamsClass, paras, aliases):
        # Template for parameter initialization
        pass

    @abstractmethod
    def run(self):
        # Subclasses must implement
        pass
```

### 2. Strategy Pattern

`TransitionState` dispatcher routes to algorithms:
```python
class TransitionState:
    def run(self):
        if self.method == 'neb':
            return NEB(...).run()
        elif self.method == 'prfo':
            return PRFO(...).run()
        # ... etc
```

### 3. Dataclass Pattern

Type-safe parameters with defaults:
```python
@dataclass
class NEBParams:
    n_images: int = 11
    k_max: float = 0.3
    # ... etc
```

### 4. Factory Pattern (Implicit)

Calculator creation handled by `SetCalculator`:
```python
# Not in TS module, but used by all algorithms
calc = SetCalculator(model='aimnet2', device='gpu0')
atoms.calc = calc
```

### 5. Utility Functions

Pure functions for geometric operations:
```python
# No side effects, reusable
kabsch_align(P, Q) -> (Q_aligned, rmsd, R, t)
improved_tangent(Rm1, R, Rp1, Em1, E, Ep1) -> tau
compute_dynamic_k(energies, k_min, k_max, k_decay) -> k_springs
```

---

## Summary

The TS module is a sophisticated computational chemistry toolkit with:

### Technical Strengths
- **8 distinct algorithms** covering diverse TS search strategies
- **Production-quality implementations** (NEB: 1,517 lines with comprehensive features)
- **Robust geometric handling** (Kabsch, IDPP, improved tangents)
- **Advanced optimization** (L-BFGS, RFO, trust regions, mode following)
- **Clean architecture** (JobABC base class, dataclass parameters)
- **Flexible parameter system** (case-insensitive, hierarchical, alias support)

### Mathematical Sophistication
- Energy-weighted tangents (Henkelman-Jonsson)
- Dynamic spring constants (ORCA-style)
- Partitioned RFO with dual-shift
- Mass-weighted coordinates for physically meaningful trust radii
- Kabsch alignment for rigid body removal
- IDPP smoothing for realistic initial paths

### Software Engineering
- Type-safe parameters using dataclasses
- Case-insensitive user-friendly input
- Comprehensive logging
- Trajectory output for visualization
- Reusable utility functions
- Clear separation of concerns

### Algorithm Completeness
| Feature | NEB | PRFO | Dimer | GSM | AFIR |
|---------|-----|------|-------|-----|------|
| Hessian-free | ✓ | ✗ | ✓ | ✓ | ✓ |
| No initial path | ✗ | ✗ | ✗ | ✗ | ✓ |
| Accurate TS | ✓✓ | ✓✓✓ | ✓✓ | ✓ | ✓ |
| Multiple TS | ✗ | ✗ | ✗ | ✗ | ✓ |
| Path info | ✓✓✓ | ✗ | ✗ | ✓✓ | ✓ |

The module represents state-of-the-art computational chemistry methodology implemented with excellent software practices.
