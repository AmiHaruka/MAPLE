"""
NVT (canonical) ensemble implementation.

Supports two thermostat algorithms:
    - langevin: Langevin dynamics (Leimkuhler & Matthews, AMRX 2013)
                Strong coupling; per-atom stochastic force.
                Good for equilibration or when strong damping is wanted.
    - v-rescale: Stochastic velocity rescaling (Bussi et al., 2007)
                 Correct canonical ensemble; global velocity scaling.
                 Less perturbation to dynamics; preferred for production.

Integration loop (Velocity Verlet with midstep thermostat):
    B: half-step velocity with conservative force  (cached from previous step)
    A: full-step position update + PBC wrap
    O: thermostat step (Langevin O-U step  OR  V-rescale global rescaling)
    B: half-step velocity with newly computed forces (cached for next step)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from ase import Atoms

from ...jobABC import JobABC
from maple.function.timer import timer

from ..integrator.velocity_verlet import VelocityVerlet
from ..thermostat.langevin import LangevinThermostat
from ..thermostat.vrescale import VRescaleThermostat
from ..utils import (
    calculate_temperature,
    calculate_kinetic_energy,
    initialize_velocities,
    HA_PER_ANG_TO_AU,
)
from ..rst_io import get_rng_state_hex, restore_rng_from_hex
from ..logger import MDLogger


@dataclass
class NVTParams:
    """
    Parameters for NVT (canonical) ensemble simulation.

    All defaults are grounded in published standards for ML potentials
    and major MD software (GROMACS, AMBER, NAMD, LAMMPS).  Inline
    citations are provided next to each field.
    """
    # ------------------------------------------------------------------
    # Timestep: 0.1 fs
    # Smaller timestep for ML potentials improves energy conservation.
    # Refs: Zhang et al. (2018) Phys. Rev. Lett. 120, 143001 (DeePMD);
    #       Batatia et al. (2022) NeurIPS 35, 11423 (MACE).
    # ------------------------------------------------------------------
    timestep: float = 0.1           # fs

    # ------------------------------------------------------------------
    # Total steps: 100000 × 0.1 fs = 10 ps
    # Standard default simulation length for ML-MD runs.
    # Refs: GROMACS Lemkul tutorial; AMBER Tutorial 1.
    # ------------------------------------------------------------------
    steps: int = 100000             # steps (= 10 ps at 0.1 fs/step)

    # ------------------------------------------------------------------
    # Reference temperature
    # 300 K: standard ambient condition used across all major MD tutorials.
    # ------------------------------------------------------------------
    temperature:     float = 300.0        # K

    # ------------------------------------------------------------------
    # Thermostat algorithm
    # Langevin is the recommended default for ML potentials and is the
    # default in AMBER (ntt=3), NAMD, OpenMM (LangevinMiddleIntegrator),
    # LAMMPS (fix langevin), MACE (ase.md.langevin), and DeePMD-kit.
    #
    # Theoretical basis: Langevin dynamics are governed by the
    # fluctuation-dissipation theorem (Kubo 1966), which guarantees
    # the Boltzmann distribution as the stationary state.
    # Unlike Nose-Hoover, Langevin is ergodic by construction — each
    # DOF receives independent stochastic perturbations at every step,
    # preventing trapping in quasi-periodic orbits.
    #
    # Thermostat comparison:
    #   Langevin    — correct canonical ensemble; ergodic; per-atom noise;
    #                 recommended for ML potentials and biomolecular NVT.
    #                 Slightly damps dynamical properties (diffusion,
    #                 viscosity) — use small γ for transport calculations.
    #   Refs: Leimkuhler & Matthews (2013) Appl. Math. Res. eXpress 2013, 34–56;
    #         Schneider & Stoll (1978) Phys. Rev. B 17, 1302;
    #         Basconi & Shirts (2013) JCTC 9, 2887.
    #
    #   V-rescale   — correct canonical ensemble for kinetic energy
    #                 (Bussi et al. 2007); global rescaling only; ergodicity
    #                 in configuration space not rigorously proven; weaker
    #                 perturbation, preserves dynamics better than Langevin.
    #                 Default in GROMACS (since v4.5).
    #   Refs: Bussi, Donadio & Parrinello (2007) J. Chem. Phys. 126, 014101.
    #
    #   Nose-Hoover — deterministic, time-reversible; correct for large
    #                 ergodic systems.  Non-ergodic for small/harmonic
    #                 systems (Legoll et al. 2007).  Not implemented here.
    #   Refs: Nosé (1984) J. Chem. Phys. 81, 511;
    #         Hoover (1985) Phys. Rev. A 31, 1695;
    #         Martyna et al. (1992) J. Chem. Phys. 97, 2635 (chains).
    #
    #   Berendsen   — NOT canonical; suppresses KE fluctuations; produces
    #                 wrong ensemble.  Use only for rapid pre-equilibration.
    #   Ref: Berendsen et al. (1984) J. Chem. Phys. 81, 3684.
    #        Basconi & Shirts (2013) JCTC 9, 2887 (analysis).
    # ------------------------------------------------------------------
    thermostat:      str   = 'langevin'   # [AMBER ntt=3; NAMD; OpenMM; MACE; DeePMD-kit]

    # ------------------------------------------------------------------
    # Langevin friction coefficient  (used only when thermostat='langevin')
    # 0.001 1/fs = 1 ps⁻¹: balances fast sampling with realistic dynamics.
    # Lower values (~0.1 ps⁻¹) preserve dynamics; higher (~10 ps⁻¹) give
    # faster but over-damped equilibration.
    # Refs: Leimkuhler & Matthews (2013) Appl. Math. Res. eXpress 2013, 34–56;
    #       AMBER: gamma_ln = 1 ps⁻¹ (Case et al. 2023 Tutorial 1);
    #       NAMD UG §2.6: langevinDamping = 1 ps⁻¹ for production.
    # ------------------------------------------------------------------
    friction:        float = 0.001        # 1/fs = 1 ps⁻¹  [Leimkuhler & Matthews 2013; AMBER; NAMD]

    # ------------------------------------------------------------------
    # V-rescale temperature coupling time  (used only when thermostat='v-rescale')
    # 100 fs: GROMACS built-in default and Lemkul tutorial value.
    # Bussi et al. (2007) validate the algorithm at tau_t = 0.1 ps; it is
    # correct for any tau_t > 0.  LAMMPS Nose-Hoover Tdamp equivalent: 0.1 ps.
    # Refs: Bussi, Donadio & Parrinello (2007) J. Chem. Phys. 126, 014101;
    #       GROMACS Reference Manual 2024, mdp-options (tau_t default = 0.1 ps);
    #       LAMMPS fix nvt docs: "Tdamp of 100 time units is reasonable"
    #         → metal units: 100 × 0.001 ps = 0.1 ps = 100 fs.
    # ------------------------------------------------------------------
    tau_t:           float = 100.0        # fs  [Bussi 2007; GROMACS Manual 2024; LAMMPS fix nvt]

    # ------------------------------------------------------------------
    # Output frequencies
    #
    # GROMACS/AMBER defaults (nstxout=500×2fs=1ps) target classical FF
    # speeds of 100–1000 ns/day.  ML potentials are ~1000–3000× slower;
    # a typical ML-NVT run is 10–100 ps.
    #
    # Target: 100–1000 frames per 10 ps.
    #   traj_every = 100 steps × 0.1 fs/step = 10 fs = 0.01 ps/frame
    #   10 ps → 1000 frames  ✓
    #
    # Refs: Stocker et al. (2022) Mach. Learn.: Sci. Technol. 3, 045010 —
    #         GNN-MD benchmarks, 10–100 ps runs with ps-scale trajectory output.
    #       Kovács et al. (2023) J. Chem. Phys. 159, 044118 — MACE evaluation
    #         with dense per-step output for monitoring convergence.
    # ------------------------------------------------------------------
    traj_every:      int   = 100          # steps (= 10 fs = 0.01 ps at 0.1 fs/step)
    log_every:       int   = 100          # steps (= 10 fs; dense logging is cheap vs ML force eval)
    # ------------------------------------------------------------------
    # Trajectory format: xyz (text) or dcd (binary)
    # DCD binary format is ~3-4x smaller than XYZ and faster to read/write.
    # Ref: CHARMM documentation; VMD molfile plugin.
    # ------------------------------------------------------------------
    traj_format:     str   = "xyz"        # "xyz" (text, default) or "dcd" (binary)

    verbose:         int   = 1            # 0=off, 1=GROMACS-style progress, 2=verbose
    debug:           bool  = False

    init_velocities: bool  = True
    restart:          bool  = False
    load_state:       bool  = False
    rst_file:         str   = ""           # Path to RST checkpoint file (explicit source for restart/load_state)
    rst_every:        int   = 1000
    remove_com:       bool  = True
    remove_rotation:  bool  = False
    random_seed: Optional[int] = None


class NVT(JobABC):
    """
    NVT (canonical) ensemble simulation.

    Integrates with the MAPLE dispatcher via JobABC.
    """

    _THERMOSTAT_CHOICES = {'langevin', 'v-rescale'}

    def __init__(self, output: str, atoms: Atoms, paras: Optional[dict] = None):
        super().__init__(output)

        if atoms.calc is None:
            raise ValueError("Atoms object must have a calculator attached")

        self.atoms = atoms
        self.params = self._init_params(NVTParams, paras, ("md", "MD", "nvt", "NVT"))

        if self.params.thermostat not in self._THERMOSTAT_CHOICES:
            raise ValueError(
                f"Unknown thermostat '{self.params.thermostat}'. "
                f"Choose from: {self._THERMOSTAT_CHOICES}"
            )

        # Warn if user set Langevin-specific params but chose v-rescale (or vice versa)
        if self.params.thermostat == 'v-rescale' and paras and 'friction' in (paras or {}):
            self.log_info([
                "\n*** WARNING: 'friction' parameter was specified but thermostat is 'v-rescale'.\n"
                "    The friction parameter is only used by the Langevin thermostat.\n"
                "    If you intended Langevin dynamics, add: thermostat=langevin\n\n"
            ])
        if self.params.thermostat == 'langevin' and paras and 'tau_t' in (paras or {}):
            self.log_info([
                "\n*** WARNING: 'tau_t' parameter was specified but thermostat is 'langevin'.\n"
                "    The tau_t parameter is only used by the V-rescale thermostat.\n\n"
            ])

        self._rng = (np.random.default_rng(self.params.random_seed)
                     if self.params.random_seed is not None
                     else np.random.default_rng())

        if self.params.thermostat == 'langevin':
            self.thermostat = LangevinThermostat(
                atoms,
                temperature=self.params.temperature,
                friction=self.params.friction,
                timestep=self.params.timestep,
                rng=self._rng,
            )
        else:  # v-rescale
            self.thermostat = VRescaleThermostat(
                atoms,
                temperature=self.params.temperature,
                tau_t=self.params.tau_t,
                timestep=self.params.timestep,
                rng=self._rng,
            )

        self.logger = MDLogger(
            output_path=output,
            log_every=self.params.log_every,
            traj_every=self.params.traj_every,
            traj_format=self.params.traj_format,
            verbose=self.params.verbose,
            debug=self.params.debug,
        )

    def run(self):
        """Execute NVT simulation."""
        with timer("MD Simulation (NVT)"):
            self._log_parameters()

            if self.params.load_state:
                if self.params.init_velocities and self.params.debug:
                    self.log_info([
                        "\nload_state=True: ignoring init_velocities and using coordinates/velocities from RST.\n"
                    ])
                result = self.logger.restart_simulation(
                    ensemble='nvt',
                    timestep=self.params.timestep,
                    n_steps=self.params.steps,
                    temperature=self.params.temperature,
                    atoms=self.atoms,
                    rst_file=self.params.rst_file if self.params.rst_file else None,
                    load_state=True,
                )
                self.atoms, velocities, step_offset = result
                if self.logger.resumed_rng_state is not None:
                    restore_rng_from_hex(self._rng, self.logger.resumed_rng_state)
                remaining = self.params.steps
            elif self.params.restart:
                if self.params.init_velocities and self.params.debug:
                    self.log_info([
                        "\nrestart=True: ignoring init_velocities and using coordinates/velocities from RST.\n"
                    ])
                result = self.logger.restart_simulation(
                    ensemble='nvt',
                    timestep=self.params.timestep,
                    n_steps=self.params.steps,
                    temperature=self.params.temperature,
                    atoms=self.atoms,
                    rst_file=self.params.rst_file if self.params.rst_file else None,
                    load_state=False,
                )
                if result is None:   # already completed
                    return
                self.atoms, velocities, step_offset = result
                # Restore RNG state for deterministic continuation
                if self.logger.resumed_rng_state is not None:
                    restore_rng_from_hex(self._rng, self.logger.resumed_rng_state)
                remaining = self.params.steps - step_offset
            else:
                if 'velocities' in self.atoms.arrays and self.params.init_velocities:
                    velocities = self.atoms.arrays['velocities']
                    t_check = calculate_temperature(self.atoms, velocities)
                    self.log_info([
                        f"\nVelocities loaded from input file "
                        f"(T = {t_check:.2f} K); skipping random initialisation.\n"
                    ])
                elif self.params.init_velocities:
                    velocities = self._initialize_velocities()
                else:
                    if 'velocities' not in self.atoms.arrays:
                        raise ValueError(
                            "init_velocities=False, "
                            "but no velocities found in atoms.arrays"
                        )
                    velocities = self.atoms.arrays['velocities']
                step_offset = 0
                remaining   = self.params.steps
                source = "input_xyz" if 'velocities' in self.atoms.arrays and not self.params.init_velocities else ("input_xyz" if 'velocities' in self.atoms.arrays and self.params.init_velocities else "init_velocities")
                self.logger.log_debug_initial_state(self.atoms, velocities, mode=source, effective_step=step_offset)

            final_velocities = self._run_simulation(velocities,
                                                    step_offset=step_offset,
                                                    n_steps=remaining)
            self.atoms.arrays['velocities'] = final_velocities

    def _log_parameters(self):
        """Log NVT parameters to output."""
        lines = [
            "\n" + "=" * 80 + "\n",
            f"{'NVT MD PARAMETERS':^80}\n",
            "=" * 80 + "\n",
            f"Ensemble:           NVT (canonical)\n",
            f"Thermostat:         {self.params.thermostat}\n",
            f"Timestep:           {self.params.timestep:.3f} fs\n",
            f"Total steps:        {self.params.steps}\n",
            f"Temperature:        {self.params.temperature:.2f} K\n",
        ]
        if self.params.thermostat == 'langevin':
            lines.append(f"Friction (γ):       {self.params.friction:.4f} 1/fs\n")
        else:
            lines.append(f"τ_T:                {self.params.tau_t:.1f} fs\n")
        lines += [
            f"\nOutput frequencies:\n",
            f"  Log every:        {self.params.log_every} steps\n",
            f"  Traj every:       {self.params.traj_every} steps\n",
            f"\nVelocity init:      {self.params.init_velocities}\n",
            f"Restart mode:       {self.params.restart}\n",
            f"Load-state mode:    {self.params.load_state}\n",
            f"RST every:          {self.params.rst_every} steps\n",
            f"Remove COM motion:  {self.params.remove_com}\n",
        ]
        if self.params.random_seed is not None:
            lines.append(f"Random seed:        {self.params.random_seed}\n")
        lines.append("=" * 80 + "\n")
        self.log_info(lines)

    def _initialize_velocities(self) -> np.ndarray:
        """Initialize velocities from Maxwell-Boltzmann distribution."""
        self.log_info([f"\nInitializing velocities at {self.params.temperature:.2f} K...\n"])
        velocities = initialize_velocities(
            atoms=self.atoms,
            temperature=self.params.temperature,
            remove_com=self.params.remove_com,
            remove_rotation=self.params.remove_rotation,
            rng=self._rng,
        )
        actual_temp = calculate_temperature(self.atoms, velocities)
        self.log_info([f"Initial temperature: {actual_temp:.2f} K\n"])
        return velocities

    def _run_simulation(self, velocities: np.ndarray,
                        step_offset: int = 0, n_steps: int = None) -> np.ndarray:
        """
        Run NVT simulation.

        Integration scheme depends on the thermostat:

        Langevin — BAOAB splitting (Leimkuhler & Matthews, AMRX 2013):
            B(dt/2) → A(dt/2) → O(OU-step) → A(dt/2) → B(dt/2)
            The OU step is an exact analytical propagator whose detailed
            balance does not depend on the input velocity distribution,
            so mid-step placement is optimal (highest configurational
            accuracy among splittings).

        V-rescale — VV + post-step rescaling (Bussi et al., JCP 2007):
            B(dt/2) → A(dt) → force eval → B(dt/2) → rescale(v_full)
            Bussi Eq. A7 assumes the input kinetic energy K is drawn from
            the canonical chi²(N_f) distribution.  Full-step velocities
            satisfy this; half-step velocities carry an O(dt) bias from
            the B kick, which breaks detailed balance of the rescaling
            step and contaminates the conserved energy H̃.  Placing the
            rescale after a complete VV step — as GROMACS does — restores
            exact detailed balance and makes H̃ a reliable measure of
            integration quality.
        """
        if n_steps is None:
            n_steps = self.params.steps

        self.logger.start_simulation(
            ensemble='nvt',
            timestep=self.params.timestep,
            n_steps=n_steps,
            temperature=self.params.temperature,
            atoms=self.atoms,
            step_offset=step_offset,
        )
        self.logger.log_main([
            f"\nStarting NVT simulation ({self.params.thermostat})...\n\n"
        ])

        integrator = VelocityVerlet(self.atoms, self.params.timestep)
        masses_1d = integrator.masses
        total_mass = masses_1d.sum()
        v = velocities.copy()

        # Cache forces at t=0; reused as first B-step forces each cycle.
        forces = self.atoms.get_forces() * HA_PER_ANG_TO_AU  # Ha/Å → a.u.

        # V-rescale conserved energy — discrete form of Bussi 2007 Eq. 15:
        #   H̃_N = H_N − Σ_{k=0}^{N-1} ΔW_k
        # where ΔW_k = (α²_k − 1)·K_k is the energy injected by the thermostat
        # at step k.  H̃ only accumulates the Verlet integration error.
        # Its drift measures timestep accuracy, analogous to TE drift in NVE.
        # Only valid for NVT with V-rescale; Langevin has no analogous quantity.
        is_vrescale = self.params.thermostat == 'v-rescale'
        w_bath = 0.0

        for step in range(1, n_steps + 1):

            if is_vrescale:
                # VV + COM projection + post-step rescale (Bussi 2007 / GROMACS scheme):
                #   1. Full Velocity Verlet step (forces cached across steps)
                #   2. For isolated systems, project out numerical COM drift so the
                #      thermostat acts in the intended 3N-3 subspace
                #   3. Rescale full-step velocities with V-rescale Eq. A7
                #
                # H̃ must track all non-Hamiltonian energy changes applied after the
                # Verlet step. Therefore we include both the COM-projection kinetic
                # energy change and the thermostat work in w_bath.
                v, forces = integrator.step(v, forces)
                ke_before_proj = calculate_kinetic_energy(self.atoms, v)
                delta_w_com = 0.0
                if not any(self.atoms.pbc):
                    p_com = np.sum(masses_1d[:, np.newaxis] * v, axis=0)
                    v -= p_com / total_mass
                    ke_after_proj = calculate_kinetic_energy(self.atoms, v)
                    delta_w_com = ke_after_proj - ke_before_proj
                v, delta_w = self.thermostat.apply(v)
                w_bath += delta_w_com + delta_w
            else:
                # BAOAB splitting (Leimkuhler & Matthews 2013):
                #   B: half-kick  A(dt/2): half-position  O: thermostat
                #   A(dt/2): half-position  B: half-kick
                v_half = integrator.split_step(v, forces)
                v_therm = self.thermostat.apply(v_half)
                v, forces = integrator.complete_split_step(v_therm)

            abs_step         = step_offset + step
            current_time     = abs_step * self.params.timestep
            temperature      = calculate_temperature(self.atoms, v)
            kinetic_energy   = calculate_kinetic_energy(self.atoms, v)
            potential_energy = self.atoms.get_potential_energy()  # Ha

            # Conserved energy: H̃ = H − Σ ΔW (V-rescale only)
            conserved = (kinetic_energy + potential_energy - w_bath) if is_vrescale else None

            self.logger.log_step(
                step=abs_step,
                time=current_time,
                temperature=temperature,
                kinetic_energy=kinetic_energy,
                potential_energy=potential_energy,
                total_energy=kinetic_energy + potential_energy,
                atoms=self.atoms,
                velocities=v,
                rng_state=get_rng_state_hex(self._rng),
                rst_every=self.params.rst_every,
                conserved_energy=conserved,
            )

        self.logger.end_simulation(
            atoms=self.atoms,
            final_velocities=v,
            rng_state=get_rng_state_hex(self._rng)
        )
        self.logger.log_main(["\nNVT simulation completed successfully.\n"])
        return v
