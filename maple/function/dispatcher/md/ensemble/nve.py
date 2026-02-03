"""
NVE (Microcanonical) ensemble implementation.

Constant:
    - N: Number of particles
    - V: Volume
    - E: Total energy

Uses Velocity Verlet integrator for symplectic time evolution.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from ase import Atoms

from ...jobABC import JobABC
from maple.function.timer import timer

from ..integrator.velocity_verlet import VelocityVerlet
from ..utils import (
    calculate_temperature,
    calculate_kinetic_energy,
    initialize_velocities
)
from ..logger import MDLogger


@dataclass
class NVEParams:
    """Parameters for NVE simulation."""
    timestep: float = 0.5           # fs
    steps: int = 10000              # total steps
    temperature: float = 300.0      # K (for velocity initialization)
    traj_every: int = 10            # trajectory write frequency
    log_every: int = 100            # log output frequency
    verbose: int = 1                # 0=concise, 1=detailed
    init_velocities: bool = True    # whether to initialize velocities
    remove_com: bool = True         # remove center-of-mass motion
    random_seed: Optional[int] = None  # random seed for reproducibility


class NVE(JobABC):
    """
    NVE (microcanonical) ensemble simulation.

    Inherits from JobABC to integrate with MAPLE dispatcher.
    """

    def __init__(self, output: str, atoms: Atoms, paras: Optional[dict] = None):
        """
        Initialize NVE ensemble.

        Args:
            output: Output file path
            atoms: ASE Atoms object with calculator
            paras: Parameters dictionary from CommandControl
        """
        super().__init__(output)

        if atoms.calc is None:
            raise ValueError("Atoms object must have a calculator attached")

        self.atoms = atoms

        # Initialize params from dict
        self.params = self._init_params(NVEParams, paras, ("md", "MD", "nve", "NVE"))

        # Initialize components
        self.integrator = VelocityVerlet(atoms, self.params.timestep)
        self.logger = MDLogger(
            output_path=output,
            log_every=self.params.log_every,
            traj_every=self.params.traj_every
        )

    def run(self):
        """
        Execute NVE simulation.
        """
        with timer("MD Simulation (NVE)"):
            # Log parameters
            self._log_parameters()

            # Initialize velocities
            if self.params.init_velocities:
                velocities = self._initialize_velocities()
            else:
                # User should provide velocities in atoms.arrays['velocities']
                if 'velocities' not in self.atoms.arrays:
                    raise ValueError("init_velocities=False but no velocities found in atoms.arrays")
                velocities = self.atoms.arrays['velocities']

            # Run simulation
            final_velocities = self._run_simulation(velocities)

            # Store final velocities (optional, for restart)
            self.atoms.arrays['velocities'] = final_velocities

    def _log_parameters(self):
        """Log NVE parameters to output."""
        self.log_info([
            "\n" + "="*80 + "\n",
            f"{'NVE MD PARAMETERS':^80}\n",
            "="*80 + "\n",
            f"Ensemble:           NVE (microcanonical)\n",
            f"Timestep:           {self.params.timestep:.3f} fs\n",
            f"Total steps:        {self.params.steps}\n",
            f"Initial temp:       {self.params.temperature:.2f} K\n",
            f"\nOutput frequencies:\n",
            f"  Log every:        {self.params.log_every} steps\n",
            f"  Traj every:       {self.params.traj_every} steps\n",
            f"\nVelocity init:      {self.params.init_velocities}\n",
            f"Remove COM motion:  {self.params.remove_com}\n",
        ])

        if self.params.random_seed is not None:
            self.log_info([f"Random seed:        {self.params.random_seed}\n"])

        self.log_info(["="*80 + "\n"])

    def _initialize_velocities(self) -> np.ndarray:
        """
        Initialize velocities from Maxwell-Boltzmann distribution.

        Returns:
            velocities: Velocity array (Å/fs)
        """
        self.log_info([
            f"\nInitializing velocities at {self.params.temperature:.2f} K...\n"
        ])

        # Create RNG if seed provided
        rng = np.random.default_rng(self.params.random_seed) if self.params.random_seed else None

        # Initialize velocities
        velocities = initialize_velocities(
            atoms=self.atoms,
            temperature=self.params.temperature,
            remove_com=self.params.remove_com,
            rng=rng
        )

        # Verify temperature
        actual_temp = calculate_temperature(self.atoms, velocities)
        self.log_info([
            f"Initial temperature: {actual_temp:.2f} K\n"
        ])

        return velocities

    def _run_simulation(self, velocities: np.ndarray) -> np.ndarray:
        """
        Run NVE simulation.

        Args:
            velocities: Initial velocities (Å/fs)

        Returns:
            Final velocities (Å/fs)
        """
        # Start logging
        self.logger.start_simulation(
            ensemble='nve',
            timestep=self.params.timestep,
            n_steps=self.params.steps,
            temperature=self.params.temperature,
            atoms=self.atoms
        )

        self.logger.log_main(["\nStarting NVE simulation...\n\n"])

        # Current velocities
        v = velocities.copy()

        # Main MD loop
        for step in range(1, self.params.steps + 1):
            # Integrate one step
            v = self.integrator.step(v)

            # Calculate thermodynamic quantities
            current_time = step * self.params.timestep
            temperature = calculate_temperature(self.atoms, v)
            kinetic_energy = calculate_kinetic_energy(self.atoms, v)
            potential_energy = self.atoms.get_potential_energy()  # eV
            total_energy = kinetic_energy + potential_energy * self.logger.eV2Hartree

            # Log data
            self.logger.log_step(
                step=step,
                time=current_time,
                temperature=temperature,
                kinetic_energy=kinetic_energy,
                potential_energy=potential_energy,
                total_energy=total_energy,
                atoms=self.atoms,
                velocities=v
            )

        # Finalize
        self.logger.end_simulation()
        self.logger.log_main(["\nNVE simulation completed successfully.\n"])

        return v
