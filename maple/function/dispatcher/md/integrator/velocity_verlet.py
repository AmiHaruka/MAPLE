"""
Velocity Verlet integrator for molecular dynamics.

The Velocity Verlet algorithm is a symplectic integrator that preserves
phase space volume, making it ideal for Hamiltonian dynamics.

Algorithm:
    1. v(t+dt/2) = v(t) + F(t)/m * dt/2     [half-step velocity]
    2. r(t+dt) = r(t) + v(t+dt/2) * dt      [full-step position]
    3. Calculate F(t+dt)                     [new forces]
    4. v(t+dt) = v(t+dt/2) + F(t+dt)/m * dt/2  [final velocity]

Advantages:
    - Symplectic (preserves energy in NVE)
    - Time-reversible
    - Second-order accurate
"""

import numpy as np
from ase import Atoms

# Import unit conversions from utils
from ..utils import FS_TO_AU, AMU_TO_AU, HA_PER_ANG_TO_AU, BOHR_TO_ANGSTROM


class VelocityVerlet:
    """
    Velocity Verlet integrator for classical MD.

    This integrator is the foundation for all MD ensemble simulations (NVE, NVT, NPT).
    It can be used standalone or split into half-steps for thermostat/barostat integration.

    Attributes
    ----------
    atoms : ase.Atoms
        The molecular system
    timestep : float
        Integration timestep in atomic units
    masses : np.ndarray
        Atomic masses in atomic units
    """

    def __init__(self, atoms: Atoms, timestep: float):
        """
        Initialize Velocity Verlet integrator.

        Parameters
        ----------
        atoms : ase.Atoms
            Molecular system with attached calculator
        timestep : float
            Time step in femtoseconds
        """
        self.atoms = atoms
        self.timestep_fs = timestep
        self.timestep = timestep * FS_TO_AU  # Convert fs → atomic units

        # Cache masses (avoid repeated ASE calls)
        self.masses = atoms.get_masses() * AMU_TO_AU  # Convert to atomic units

    def step(self, velocities: np.ndarray,
             forces: np.ndarray = None) -> tuple:
        """
        Perform a full Velocity Verlet integration step.

        This is the standalone step for NVE dynamics.  For maximum efficiency
        call with the forces cached from the previous step so that only one
        force evaluation is needed per step (standard Velocity Verlet caching):

            forces = atoms.get_forces() * HA_PER_ANG_TO_AU  # t=0
            for each step:
                velocities, forces = integrator.step(velocities, forces)

        If *forces* is None a fresh evaluation is performed for the first
        B-step (two force evaluations per step, 2× slower).

        Parameters
        ----------
        velocities : np.ndarray
            Current atomic velocities in atomic units, shape (N_atoms, 3).
        forces : np.ndarray, optional
            Forces at the current positions in a.u. (Ha/Bohr), shape (N_atoms, 3).
            If None, forces are computed from the calculator.

        Returns
        -------
        (velocities, forces) : tuple[np.ndarray, np.ndarray]
            Updated velocities and forces at the new positions (both in a.u.).
            The returned forces can be passed directly to the next call.
        """
        dt = self.timestep
        masses = self.masses[:, np.newaxis]  # Shape: (N_atoms, 1)

        # B1: Half-step velocity update with current forces
        if forces is None:
            forces = self.atoms.get_forces() * HA_PER_ANG_TO_AU  # Ha/Å → Ha/Bohr
        velocities = velocities + 0.5 * forces / masses * dt

        # A: Full-step position update (v in a.u., dt in a.u. → displacement in Bohr → Å)
        positions = self.atoms.get_positions()
        positions += velocities * dt * BOHR_TO_ANGSTROM
        self.atoms.set_positions(positions)
        if any(self.atoms.pbc):
            self.atoms.wrap()

        # Compute forces at new positions
        forces = self.atoms.get_forces() * HA_PER_ANG_TO_AU  # Ha/Å → Ha/Bohr

        # B2: Final half-step velocity update
        velocities = velocities + 0.5 * forces / masses * dt

        return velocities, forces

    def half_step_v(self, velocities: np.ndarray) -> np.ndarray:
        """
        Perform half-step velocity update: v(t+dt/2) = v(t) + F(t)/m * dt/2

        Used for split integration with thermostats/barostats.

        Parameters
        ----------
        velocities : np.ndarray
            Current velocities

        Returns
        -------
        np.ndarray
            Half-step velocities
        """
        dt = self.timestep
        masses = self.masses[:, np.newaxis]

        forces = self.atoms.get_forces() * HA_PER_ANG_TO_AU  # Ha/Å → Ha/Bohr (a.u.)
        velocities += 0.5 * forces / masses * dt

        return velocities

    def full_step_r(self, velocities: np.ndarray):
        """
        Perform full-step position update: r(t+dt) = r(t) + v(t+dt/2) * dt

        Modifies atoms.positions in-place.

        Parameters
        ----------
        velocities : np.ndarray
            Half-step velocities
        """
        dt = self.timestep

        positions = self.atoms.get_positions()                   # Å
        positions += velocities * dt * BOHR_TO_ANGSTROM          # Å
        self.atoms.set_positions(positions)
        if any(self.atoms.pbc):
            self.atoms.wrap()

    def complete_step_v(self, velocities: np.ndarray) -> np.ndarray:
        """
        Complete velocity update: v(t+dt) = v(t+dt/2) + F(t+dt)/m * dt/2

        Call after position update and force recalculation.

        Parameters
        ----------
        velocities : np.ndarray
            Half-step velocities

        Returns
        -------
        np.ndarray
            Full-step velocities
        """
        dt = self.timestep
        masses = self.masses[:, np.newaxis]

        forces = self.atoms.get_forces() * HA_PER_ANG_TO_AU  # Ha/Å → Ha/Bohr (a.u.)
        velocities += 0.5 * forces / masses * dt

        return velocities

    def split_step(self, velocities: np.ndarray) -> np.ndarray:
        """
        Perform B-A half-steps for thermostat insertion.

        Applies the first half-step velocity update and the full-step position
        update. The caller applies the thermostat (O-step) before calling
        complete_step_v() for the second B half-step.

        Parameters
        ----------
        velocities : np.ndarray
            Current velocities

        Returns
        -------
        np.ndarray
            Half-step velocities (after first B, before O-step)
        """
        velocities_half = self.half_step_v(velocities)
        self.full_step_r(velocities_half)
        return velocities_half
