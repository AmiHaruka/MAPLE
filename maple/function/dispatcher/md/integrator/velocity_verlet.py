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
from typing import Tuple

# Import unit conversions from utils
from ..utils import FS_TO_AU, AMU_TO_AU


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

    def step(self, velocities: np.ndarray) -> np.ndarray:
        """
        Perform a full Velocity Verlet integration step.

        This is the standard all-in-one step for NVE dynamics.

        Parameters
        ----------
        velocities : np.ndarray
            Current atomic velocities in atomic units
            Shape: (N_atoms, 3)

        Returns
        -------
        np.ndarray
            Updated velocities in atomic units
        """
        dt = self.timestep
        masses = self.masses[:, np.newaxis]  # Shape: (N_atoms, 1)

        # Step 1: Half-step velocity update
        forces = self.atoms.get_forces()  # ← Calculator call (expensive!)
        velocities += 0.5 * forces / masses * dt

        # Step 2: Full-step position update
        positions = self.atoms.get_positions()
        positions += velocities * dt
        self.atoms.set_positions(positions)

        # Step 3: Recalculate forces at new positions
        forces = self.atoms.get_forces()  # ← Calculator call again

        # Step 4: Final half-step velocity update
        velocities += 0.5 * forces / masses * dt

        return velocities

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

        forces = self.atoms.get_forces()
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

        positions = self.atoms.get_positions()
        positions += velocities * dt
        self.atoms.set_positions(positions)

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

        forces = self.atoms.get_forces()
        velocities += 0.5 * forces / masses * dt

        return velocities

    def split_step(self, velocities: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform split integration for thermostat insertion.

        This allows thermostats to be applied between the two velocity half-steps:
            v(t+dt/2) = v(t) + F(t)/m * dt/2
            r(t+dt) = r(t) + v(t+dt/2) * dt
            # ← Apply thermostat here
            v(t+dt) = v(t+dt/2) + F(t+dt)/m * dt/2

        Parameters
        ----------
        velocities : np.ndarray
            Current velocities

        Returns
        -------
        velocities_half : np.ndarray
            Half-step velocities (before thermostat)
        positions_new : np.ndarray
            Updated positions
        """
        # First half-step
        velocities_half = self.half_step_v(velocities)

        # Position update
        self.full_step_r(velocities_half)

        positions_new = self.atoms.get_positions()

        return velocities_half, positions_new

    def get_current_forces(self) -> np.ndarray:
        """
        Get forces at current positions.

        Returns
        -------
        np.ndarray
            Forces in eV/Å (ASE units)
        """
        return self.atoms.get_forces()

    def get_potential_energy(self) -> float:
        """
        Get potential energy at current positions.

        Returns
        -------
        float
            Potential energy (uses calculator's units, typically eV)
        """
        return self.atoms.get_potential_energy(force_consistent=True)


class IntegratorBase:
    """
    Base class for MD integrators (future extension).

    This provides a common interface for different integration schemes:
    - Velocity Verlet
    - Leapfrog
    - Verlet (original)
    - Higher-order integrators
    """

    def __init__(self, atoms: Atoms, timestep: float):
        """
        Initialize base integrator.

        Parameters
        ----------
        atoms : ase.Atoms
            Molecular system
        timestep : float
            Time step in femtoseconds
        """
        self.atoms = atoms
        self.timestep_fs = timestep
        self.timestep = timestep * FS_TO_AU
        self.masses = atoms.get_masses() * AMU_TO_AU

    def step(self, velocities: np.ndarray) -> np.ndarray:
        """
        Perform one integration step.

        Must be implemented by subclasses.

        Parameters
        ----------
        velocities : np.ndarray
            Current velocities

        Returns
        -------
        np.ndarray
            Updated velocities
        """
        raise NotImplementedError("Subclasses must implement step()")


# Convenience function for standalone usage
def integrate_nve(
    atoms: Atoms,
    velocities: np.ndarray,
    timestep: float,
    n_steps: int
) -> Tuple[list, list, list]:
    """
    Simple standalone NVE integration for testing.

    Parameters
    ----------
    atoms : ase.Atoms
        Molecular system with calculator
    velocities : np.ndarray
        Initial velocities in atomic units
    timestep : float
        Time step in femtoseconds
    n_steps : int
        Number of integration steps

    Returns
    -------
    trajectory : list of ase.Atoms
        Atomic configurations at each step
    energies : list of float
        Total energies at each step
    velocities_traj : list of np.ndarray
        Velocities at each step
    """
    integrator = VelocityVerlet(atoms, timestep)

    trajectory = []
    energies = []
    velocities_traj = []

    for step in range(n_steps):
        # Integrate
        velocities = integrator.step(velocities)

        # Record
        trajectory.append(atoms.copy())

        potential = integrator.get_potential_energy()
        from ..utils import calculate_kinetic_energy
        kinetic = calculate_kinetic_energy(atoms, velocities)
        total = kinetic + potential

        energies.append(total)
        velocities_traj.append(velocities.copy())

    return trajectory, energies, velocities_traj
