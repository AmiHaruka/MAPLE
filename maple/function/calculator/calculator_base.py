from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import ase.calculators.calculator

if TYPE_CHECKING:
    import torch


_REGISTRY: dict[str, type] = {}


def register_calculator(cls):
    """Register a calculator class under each name in cls.MODEL_NAMES.

    Raises ValueError on duplicate registration to surface accidental name
    collisions instead of silently overwriting.
    """
    for raw_name in cls.MODEL_NAMES:
        name = raw_name.lower()
        existing = _REGISTRY.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"Calculator name '{name}' is already registered to {existing.__name__}; "
                f"refusing to overwrite with {cls.__name__}."
            )
        _REGISTRY[name] = cls
    return cls


def get_registered_calculator(name: str) -> type:
    return _REGISTRY[name.lower()]


def import_calculator_plugin(module_path: str) -> None:
    import importlib
    importlib.import_module(module_path)


def load_calculator_plugins_from_env() -> None:
    import os
    raw = os.environ.get('MAPLE_CALCULATOR_PLUGINS', '')
    for entry in (s.strip() for s in raw.split(',') if s.strip()):
        import_calculator_plugin(entry)


EV2HARTREE = 1.0 / 27.211386245988


def _convert_energy_force_units(energy, forces, *, source_unit):
    """Convert backend (energy, forces) to Hartree and Hartree/Å.

    Backends declare MODEL_ENERGY_UNIT honestly. Hartree is a no-op; eV
    multiplies through by EV2HARTREE.
    """
    if source_unit == 'hartree':
        return energy, forces
    if source_unit == 'eV':
        energy_ha = energy * EV2HARTREE
        forces_ha = forces * EV2HARTREE if forces is not None else None
        return energy_ha, forces_ha
    raise ValueError(
        f"Unknown source_unit: {source_unit!r}; expected 'eV' or 'hartree'."
    )


def init_implicit_solvent(calc, implicit, solvent, device):
    """Shared implicit-solvent initializer.

    Usable by CalcABC subclasses and duck-typed calculators (UMA) so the
    GBSA/QEq construction lives in one place.
    """
    if implicit == 'gbsa' and solvent != 'none':
        from .extra_correction import GBSA, QEqTorch

        calc.solvent_correction = GBSA(solvent=solvent, device=device)
        calc.chargecalc = QEqTorch(device=device)
    else:
        calc.solvent_correction = None


def numerical_hessian_from_atoms(calc, atoms, delta=0.002):
    """Numerical Hessian via central finite difference on forces.

    Polymorphic: works for any calculator with the ASE protocol
    (calc.calculate(atoms, properties=['forces'], system_changes=...) writes
    calc.results['forces'] as a (N, 3) ndarray). Returns float64 ndarray of
    shape (3N, 3N).
    """
    from ase.constraints import FixAtoms
    from ase.calculators.calculator import all_changes

    N = len(atoms)
    pos0 = atoms.get_positions().copy()
    fixed = {
        i
        for c in getattr(atoms, 'constraints', []) or []
        if isinstance(c, FixAtoms)
        for i in c.get_indices()
    }
    movable = [i for i in range(N) if i not in fixed]

    H = np.zeros((3 * N, 3 * N), dtype=np.float64)
    if not movable:
        return H

    def force_at(positions):
        at = atoms.copy()
        at.set_positions(positions)
        if getattr(atoms, 'constraints', None):
            at.set_constraint(atoms.constraints)
        calc.calculate(at, properties=['forces'], system_changes=all_changes)
        return np.asarray(calc.results['forces'], dtype=np.float64)

    for a in movable:
        for k in range(3):
            row = 3 * a + k
            pos_p = pos0.copy(); pos_p[a, k] += delta
            Fp = force_at(pos_p)
            pos_m = pos0.copy(); pos_m[a, k] -= delta
            Fm = force_at(pos_m)
            H[row, :] = (-(Fp - Fm) / (2.0 * delta)).reshape(-1)

    return H


class CalcABC(ase.calculators.calculator.Calculator):
    # Protocol attributes — each subclass overrides what's relevant.
    MODEL_NAMES: tuple = ()
    MODEL_ENERGY_UNIT: str = 'eV'
    SUPPORTED_HESSIAN_MODES: tuple = ('numerical',)
    SUPPORTS_CHARGE_MULT: bool = False
    CHECKPOINT_FILENAME: dict | None = None
    REQUIRES_LOCAL_MODEL_FILE: bool = False

    def __init__(self):
        super().__init__()

    @classmethod
    def build_kwargs_from_options(cls, model, model_options, *, resolved_model_path=None):
        """Translate input-header options into ctor kwargs. Backends override."""
        return {}

    def _finalize_results(self, atoms, *, energy, forces=None, hessian=None, unit=None):
        """Single entry: unit conversion + implicit-solvent + write self.results.

        Backends pass the pure model outputs (in the unit declared by
        MODEL_ENERGY_UNIT). This method converts to Hartree, then optionally
        adds the implicit-solvent correction, then writes self.results.
        """
        source_unit = unit if unit is not None else self.MODEL_ENERGY_UNIT
        energy_ha, forces_ha = _convert_energy_force_units(
            energy, forces, source_unit=source_unit
        )

        if getattr(self, 'solvent_correction', None) is not None:
            solvent_energy, solvent_force = self.implicit_solv_energy_and_force(atoms)
            se = solvent_energy.item() if hasattr(solvent_energy, 'item') else float(solvent_energy)
            energy_ha = energy_ha + se
            if forces_ha is not None and solvent_force is not None:
                sf = (
                    solvent_force.detach().cpu().numpy()
                    if hasattr(solvent_force, 'detach')
                    else np.asarray(solvent_force)
                )
                forces_ha = forces_ha + sf

        self.results['energy'] = float(energy_ha)
        self.results['free_energy'] = float(energy_ha)
        if forces_ha is not None:
            self.results['forces'] = forces_ha
        if hessian is not None:
            self.results['hessian'] = hessian

    def get_hessian(self, atoms, delta: float = 0.002):
        """Dispatch on self.hessian. Subclasses may override for backend autograd."""
        mode = getattr(self, 'hessian', self.SUPPORTED_HESSIAN_MODES[0])
        if mode == 'analytic':
            if getattr(self, 'solvent_correction', None) is not None:
                raise NotImplementedError(
                    'Analytic Hessian with implicit solvent is not supported. '
                    "Set hessian='numerical'."
                )
            return np.asarray(self._analytic_hessian(atoms))
        if mode == 'numerical':
            return numerical_hessian_from_atoms(self, atoms, delta)
        raise ValueError(f"Unknown hessian mode: {mode!r}")

    def _analytic_hessian(self, atoms):
        """Backend autograd Hessian. Override in subclasses that can autodiff."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement analytic Hessian; "
            "set hessian='numerical' or override _analytic_hessian."
        )


    def log_error(self, error_message: str) -> None:
        """
        Logs error messages to the output file.

        Args:
            error_message: The error message to log.
        """
        with open(self.output, 'a') as file:
            file.write(f"ERROR: {error_message}\n")

    def log_info(self, info_message: list) -> None:
        """
        Logs info messages to the output file.

        Args:
            info_message: The info message to log.
        """
        with open(self.output, 'a') as file:
            for info in info_message:   
                file.write(f"{info}")

    def get_hvp(self, atoms, n: np.ndarray):
        """
        Compute Hessian-vector product Hn for the given atoms and direction n using autograd.
        Args:
            atoms (ase.Atoms): system
            n (np.ndarray): direction vector, shape (3N,)
        Returns:
            Hn (torch.Tensor): Hessian-vector product (3N,) on same device/dtype
            forces (torch.Tensor): forces (3N,) on same device/dtype
            energy (torch.Tensor): scalar total energy
        """
        import torch

        # 1. prepare coordinates with grad enabled
        coords = torch.tensor(
            atoms.get_positions(),
            dtype=self.dtype,
            device=self.device,
            requires_grad=True
        ).unsqueeze(0)

        # 2. atomic numbers
        species = torch.tensor(
            atoms.get_atomic_numbers(),
            dtype=torch.long,
            device=self.device
        ).unsqueeze(0)

        # 3. forward pass → energy
        energy = self.model(species, coords)[0]
        if self.d4:
            energy += self.dftd4(species, coords)

        # 4. compute gradient (forces = -grad V)
        grad = torch.autograd.grad(energy, coords, create_graph=True)[0].squeeze(0)  # shape (N,3)
        grad_vec = grad.view(-1)  # (3N,)

        # 5. Hessian-vector product: grad(grad·n)
        n_tensor = torch.tensor(n, dtype=self.dtype, device=self.device)
        hvp = torch.autograd.grad(
            grad_vec @ n_tensor, coords, retain_graph=True
        )[0].squeeze(0).view(-1)  # (3N,)

        # 6. Forces (already computed, negative gradient)
        forces = -grad_vec

        return hvp, forces, energy

    def implicit_solv_init(self, implicit: str, solvent: str):

        if implicit == "gbsa" and solvent != 'none':

            # GBSA solvent correction and QEq charge calculator
            from .extra_correction import GBSA
            from .extra_correction import QEqTorch

            self.solvent_correction = GBSA(solvent=solvent, device=self.device)
        
            self.chargecalc = QEqTorch(device=self.device)
        else:
            self.solvent_correction = None
    
    def implicit_solv_energy(self, atoms: ase.Atoms) -> torch.Tensor:
        """
        Compute implicit solvent correction energy if applicable.

        Args:
            atoms (ase.Atoms): Atomic structure.

        Returns:
            torch.Tensor: Implicit solvent correction energy in Hartree.
        """
        atoms.atomic_charges = self.chargecalc(atoms)
        solvent_energy,_ = self.solvent_correction.get_energy(atoms)
        return solvent_energy

    def implicit_solv_energy_and_force(self, atoms: ase.Atoms) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute implicit solvent correction energy and forces if applicable.

        Args:
            atoms (ase.Atoms): Atomic structure.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Implicit solvent correction energy in Hartree and forces in Hartree/Å.
        """
        atoms.atomic_charges = self.chargecalc(atoms)
        solvent_energy, solvent_forces = self.solvent_correction.get_energy_and_force(atoms)
        return solvent_energy, solvent_forces