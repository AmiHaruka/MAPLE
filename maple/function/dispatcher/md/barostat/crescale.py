"""
C-rescale (stochastic cell rescaling) barostat for NPT molecular dynamics.

C-rescale is the pressure analogue of V-rescale: it augments Berendsen-like
cell relaxation with the stochastic volume term of Bernetti & Bussi (2020).

Algorithm (Bernetti & Bussi, 2020):
    The volume V is rescaled stochastically each step.  The new volume is
    drawn from the conditional distribution:

        d ln(V) = β * (dt/τ_P) * (P - P_target)
                  + sqrt(2 * k_B * T * β * dt / (V * τ_P)) * W

    where W ~ N(0, 1) is a Wiener noise term. The driving pressure combines
    configurational pressure with the target-temperature ideal contribution.

    Positions and cell are scaled isotropically by μ = (V_new / V)^(1/3).
    Velocities are left unchanged (Berendsen convention).

Notes:
    - Intended for NPT sampling; validate ensemble statistics for each setup.
    - Isotropic scaling only; anisotropic tensors not yet supported.
    - Pressure is computed from the virial theorem. In this implementation,
      calculator stress supplies the configurational pressure and is required.
      Current kinetic pressure is returned for logging but does not drive the
      stochastic cell update.

Reference:
    Bernetti & Bussi, J. Chem. Phys. 153, 114107 (2020).
"""

import numpy as np
from ase import Atoms
from typing import Optional

from ..utils import (
    HARTREE_TO_EV,
    KELVIN_TO_HARTREE,
    EV_PER_ANG3_TO_BAR,
    DEFAULT_COMPRESSIBILITY,
    compute_configurational_pressure,
    compute_instantaneous_pressure,
)


class CRescaleBarostat:
    """
    Stochastic cell rescaling barostat (C-rescale).

    Isotropically rescales cell and atomic positions for NPT sampling.
    """

    def __init__(
        self,
        atoms: Atoms,
        pressure: float,
        temperature: float,
        tau_p: float,
        timestep: float,
        compressibility: float = DEFAULT_COMPRESSIBILITY,
        rng: Optional[np.random.Generator] = None,
        n_dof: int | None = None,
    ):
        """
        Parameters
        ----------
        atoms : ase.Atoms
            Molecular system (must have a periodic cell)
        pressure : float
            Target pressure in bar
        temperature : float
            Target temperature in Kelvin (needed for the noise term)
        tau_p : float
            Pressure relaxation time in fs.
            Larger τ_P = weaker coupling.  Recommended: 2000–5000 fs for MLP.
        timestep : float
            MD timestep in fs
        compressibility : float
            Isothermal compressibility in 1/bar (default: water ~4.5e-5)
        rng : np.random.Generator, optional
            Random number generator for reproducibility
        n_dof : int, optional
            Translational degrees of freedom: ``3N`` or ``3N - 3`` when
            center-of-mass motion is removed. Defaults to ``3N``.
        """
        if atoms.constraints:
            raise ValueError(
                "C-rescale does not support constrained degrees of freedom"
            )
        if n_dof is None:
            n_dof = 3 * len(atoms)
        allowed_n_dof = {3 * len(atoms), 3 * len(atoms) - 3}
        if not isinstance(n_dof, (int, np.integer)) or n_dof not in allowed_n_dof:
            raise ValueError(
                f"C-rescale n_dof must be 3N or 3N-3; got {n_dof} for N={len(atoms)}"
            )

        self.atoms = atoms
        self.pressure_target = pressure          # bar
        self.temperature = temperature           # K
        self.tau_p = tau_p                       # fs
        self.timestep = timestep                 # fs
        self.compressibility = compressibility   # 1/bar
        self.rng = rng if rng is not None else np.random.default_rng()
        self.n_dof = int(n_dof)

        # Warning flag: emit stress-unavailable warning at most once per instance
        self._stress_warned = False

        # Deterministic prefactor: β * dt / τ_P  (dimensionless)
        self._det_prefactor = compressibility * timestep / tau_p

        # Stochastic noise prefactor (dimensionless, multiplied by 1/√V later):
        #   d ln(V)|_noise = sqrt(2 k_B T β dt / (τ_P V)) * W
        # We precompute sqrt(2 k_B T β dt / τ_P) in units of √Å³:
        #   k_B T in eV = T * KELVIN_TO_HARTREE * HARTREE_TO_EV
        #   β in Å³/eV  = compressibility * EV_PER_ANG3_TO_BAR
        #   → product: [eV * Å³/eV * 1] = Å³  ✓
        kT_ev = temperature * KELVIN_TO_HARTREE * HARTREE_TO_EV      # eV
        self._mean_kinetic_pressure_volume = (
            (self.n_dof / 3.0) * kT_ev * EV_PER_ANG3_TO_BAR
        )  # bar Å³
        beta_ang3_per_ev = compressibility * EV_PER_ANG3_TO_BAR      # Å³/eV
        self._noise_prefactor = np.sqrt(
            2.0 * kT_ev * beta_ang3_per_ev * (timestep / tau_p)
        )   # units: √Å³

    def get_pressure(self, velocities: np.ndarray) -> float:
        """
        Compute instantaneous pressure in bar via the virial theorem.

        P = (2*KE + W) / (3*V)

        Parameters
        ----------
        velocities : np.ndarray
            Current velocities in atomic units, shape (N_atoms, 3)

        Returns
        -------
        float
            Instantaneous pressure in bar
        """
        pressure, self._stress_warned = compute_instantaneous_pressure(
            self.atoms, velocities, self._stress_warned, self.__class__.__name__
        )
        return pressure

    def apply(self, velocities: np.ndarray) -> float:
        """
        Apply one C-rescale barostat step: stochastically rescale cell.

        The log-volume change has deterministic and stochastic terms:

            d ln(V) = β*(dt/τ_P)*(P - P_target) + noise * W / sqrt(V)

        Cell and positions are scaled isotropically by μ = (V_new/V)^(1/3).
        Velocities are not modified.

        Parameters
        ----------
        velocities : np.ndarray
            Current velocities in atomic units

        Returns
        -------
        float
            Instantaneous pressure before rescaling (bar), for logging
        """
        volume = self.atoms.get_volume()   # Å³
        configurational_pressure = compute_configurational_pressure(self.atoms)
        pressure = self.get_pressure(velocities)
        driving_pressure = (
            configurational_pressure
            + self._mean_kinetic_pressure_volume / volume
        )

        # Deterministic log-volume term: β*(dt/τ_P)*(P - P_target)
        # so that P < P_target shrinks the cell and P > P_target expands it.
        depsilon_det = self._det_prefactor * (
            driving_pressure - self.pressure_target
        )

        # Stochastic log-volume term: _noise_prefactor [√Å³] / sqrt(V [Å³]) * W
        #                = sqrt(2 k_B T β dt / (τ_P V)) * W  (dimensionless)
        w = self.rng.standard_normal()
        depsilon_stoch = self._noise_prefactor / np.sqrt(volume) * w
        depsilon = depsilon_det + depsilon_stoch
        if not np.isfinite(depsilon):
            raise FloatingPointError("C-rescale produced a non-finite log-volume step")
        with np.errstate(over="raise", under="raise", invalid="raise"):
            mu = float(np.exp(depsilon / 3.0))

        # Rescale cell and positions isotropically
        self.atoms.set_cell(self.atoms.get_cell().array * mu, scale_atoms=True)

        return pressure
