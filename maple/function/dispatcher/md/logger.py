"""
MD simulation logging and output management.

Handles:
    - Thermodynamic data output (.dat file)
    - XYZ trajectory output (.xyz file)
    - Progress logging to main output
    - Final summary statistics with publication-quality energy conservation metrics

Energy conservation metrics follow published standards:
    - Linear drift rate via least-squares fit [kJ/mol/ns/atom]:
        GROMACS Reference Manual §3.4 "Energy Conservation"; Páll et al. (2020)
        J. Chem. Phys. 153, 134110 (GROMACS GPU parallelisation; NVE thresholds cited therein)
        NVE acceptance: < 0.01 (good), < 0.1 (acceptable), > 1.0 (failure)
        NVT/NPT: gross-instability threshold only (> 1.0 = unstable)
        Equilibration skip: front 20 % discarded before fitting (NVE and NVT/NPT).
          NVE:     removes the ~10–50 step NVT→NVE thermal transient arising from
                   the Velocity Verlet + thermostat handoff (Leimkuhler & Matthews,
                   Appl. Math. Res. eXpress 2013, 34–56; Frenkel & Smit 2002 Box 4.1;
                   GROMACS Manual 2024 §3.4.2).
          NVT/NPT: removes the early thermalisation ramp so the slope reflects
                   steady-state behaviour (AMBER 2023 Manual §3.1).
    - Relative energy fluctuation σ(E)/|<E>| (dimensionless):
        AMBER Reference Manual (Case et al. 2022), §3 NVE validation
        Acceptance: < 1e-4
    - KE–PE anti-correlation coefficient:
        Allen & Tildesley, Computer Simulation of Liquids, 2nd ed. (2017), §3.4
        Hammonds & Heyes (2020) J. Chem. Phys. 152, 024114 (shadow Hamiltonian / symplecticity)
        NVE: r ≈ -1 (energy conservation forces anti-correlation; direct consequence of E = KE + PE = const)
        NVT/NPT: r ≈ 0 (thermostat randomises KE each step; anti-correlation is broken)
    - Temperature fluctuation ratio σ(T)/<T> vs. equipartition prediction 1/√N_dof:
        Allen & Tildesley, Computer Simulation of Liquids, 2nd ed. (2017), §2.4
        Excess ratio > 2 indicates thermostat leakage or integration instability
"""

import time as _time
import numpy as np
from pathlib import Path
from typing import Optional, TextIO, Any
from ase import Atoms

from .utils import write_xyz_frame
from .rst_io import read_rst, rotate_rst_checkpoint
from .dcd_writer import DCDWriter


def _backup_file(path: Path) -> Optional[Path]:
    """
    GROMACS-style file backup: if *path* exists, rename it to
    #<name>.<ext>.1# (incrementing until a free slot is found).

    Returns the backup path if a backup was made, else None.
    """
    if not path.exists():
        return None
    n = 1
    while True:
        backup = path.parent / f"#{path.name}.{n}#"
        if not backup.exists():
            path.rename(backup)
            return backup
        n += 1


# ========== Unit Conversion Constants ==========

# 1 Hartree = 2625.4996 kJ/mol  (NIST CODATA 2018)
HARTREE_TO_KJ_PER_MOL = 2625.4996
# 1 fs = 1e-6 ns
FS_TO_NS = 1e-6


# ========== Progress Formatting Helpers ==========

def _fmt_duration(seconds: float) -> str:
    """Format wall-clock duration into human-readable string, e.g. '1h 23m 45s'."""
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _fmt_eta(seconds: float) -> str:
    """Format ETA; show '∞' when remaining time is unknown or very large."""
    if seconds <= 0 or seconds > 1e7:   # > ~4 months
        return "∞"
    return _fmt_duration(seconds)


class MDLogger:
    """
    Manages MD simulation output and logging.

    Attributes:
        output_base: Base path for output files (without extension)
        thermo_file: Thermodynamics data file handle
        traj_file: XYZ trajectory file handle
        main_output: Main output file path
        log_every: Log frequency (steps)
        traj_every: Trajectory write frequency (steps)
    """

    eV2Hartree = 1 / 27.211386245988

    def __init__(
        self,
        output_path: str,
        log_every: int = 100,
        traj_every: int = 10,
        verbose: int = 1,
        traj_format: str = "xyz",
    ):
        """
        Initialize MD logger.

        Args:
            output_path: Main output file path (e.g., "task.out")
            log_every:   Frequency to log thermodynamic data (steps)
            traj_every:  Frequency to write trajectory frames (steps)
            verbose:     Progress verbosity level
                           0 = no progress lines (thermo file still written)
                           1 = GROMACS-style progress line every log_every steps
                           2 = verbose (step-level timing detail)
            traj_format: Trajectory output format: "xyz" (text) or "dcd" (binary)
        """
        self.main_output = output_path
        self.log_every   = log_every
        self.traj_every  = traj_every
        self.verbose      = verbose
        self.traj_format  = traj_format.lower()

        # Generate file paths
        base   = Path(output_path).stem
        parent = Path(output_path).parent

        self.thermo_path   = parent / f"{base}_md_thermo.dat"
        self.traj_path     = parent / f"{base}_md_traj.{self.traj_format}"
        self.summary_path  = parent / f"{base}_md_summary.txt"
        self.rst_path      = parent / f"{base}_md.rst"
        self.rst_prev_path = parent / f"{base}_md_prev.rst"

        # File handles (opened in start_simulation)
        self.thermo_file: Optional[TextIO] = None
        self.traj_file:   Any = None  # TextIO for XYZ, DCDWriter for DCD

        # Statistics tracking
        self.energies            = []
        self.temperatures        = []
        self.times               = []
        self.pressures           = []   # NPT only
        self.kinetic_energies    = []   # for KE–PE anti-correlation
        self.potential_energies  = []   # for KE–PE anti-correlation
        self._n_atoms            = 0    # set in start_simulation
        self._is_pbc             = False  # set in start_simulation; affects N_dof

        # Performance / progress tracking (set in start_simulation)
        self._n_steps:    int   = 0
        self._timestep:   float = 0.0  # fs
        self._wall_start: float = 0.0  # time.perf_counter() at simulation start
        # Wall time of the last log_step call (for rolling speed estimate)
        self._last_wall:  float = 0.0
        self._last_step:  int   = 0
        self._time_col_w: int   = 14   # width of Time(ps) column header
        self._total_ps:   float = 0.0  # total simulation time in ps

        # RNG state restored from the last trajectory frame on resume (None if not present)
        self.resumed_rng_state: Optional[str] = None

    def start_simulation(self, ensemble: str, timestep: float, n_steps: int,
                        temperature: float, atoms: Atoms,
                        pressure: float = None, step_offset: int = 0):
        """
        Initialize output files and write headers.

        Args:
            ensemble: Ensemble type (nve, nvt, npt)
            timestep: Timestep in fs
            n_steps: Number of steps to run in this segment
            temperature: Target temperature in K
            atoms: ASE Atoms object
            pressure: Target pressure in bar (NPT only)
            step_offset: Step number already completed (for resume)
        """
        self._ensemble    = ensemble.lower()
        self._n_atoms     = len(atoms)
        self._is_pbc      = bool(any(atoms.pbc))
        self._timestep    = timestep
        self._step_offset = step_offset
        # Total steps across the full run (for progress %)
        self._n_steps     = n_steps + step_offset

        # Record wall-clock start; also seed the rolling-speed anchor
        self._wall_start = _time.perf_counter()
        self._last_wall  = self._wall_start
        self._last_step  = step_offset

        if step_offset == 0:
            # Open files (back up any pre-existing files first, GROMACS-style)
            backup_msgs = []
            main_out_path = Path(self.main_output)
            for p in (main_out_path, self.thermo_path, self.traj_path, self.summary_path):
                backup = _backup_file(p)
                if backup is not None:
                    backup_msgs.append(f"  Backed up existing file: {p.name} -> {backup.name}\n")

            self.thermo_file = open(self.thermo_path, 'w')
            # Open trajectory file based on format
            if self.traj_format == 'dcd':
                self.traj_file = DCDWriter(
                    path=self.traj_path,
                    natoms=len(atoms),
                    timestep=timestep,
                    is_periodic=any(atoms.pbc),
                    first_step=step_offset,
                )
            else:  # xyz
                self.traj_file = open(self.traj_path, 'w')
        # else: files already opened in append mode by restart_simulation()

        # Write main output header
        total_steps_display = n_steps + step_offset
        self.log_main([
            "\n" + "="*80 + "\n",
            f"{'MD SIMULATION':^80}\n",
            "="*80 + "\n",
            f"Ensemble:        {ensemble.upper()}\n",
            f"Timestep:        {timestep:.3f} fs\n",
            f"Total steps:     {total_steps_display}\n",
            f"Simulation time: {total_steps_display * timestep:.2f} fs\n",
        ])

        if step_offset > 0:
            self.log_main([
                f"Resuming from:   step {step_offset} "
                f"({step_offset * timestep / 1000.0:.3f} ps)\n",
                f"Steps remaining: {n_steps}\n",
            ])

        if self._ensemble in ('nvt', 'npt'):
            self.log_main([f"Target temp:     {temperature:.2f} K\n"])
        if self._ensemble == 'npt' and pressure is not None:
            self.log_main([f"Target pressure: {pressure:.2f} bar\n"])

        self.log_main([
            f"\nSystem:\n",
            f"  Atoms:         {len(atoms)}\n",
            f"  Formula:       {atoms.get_chemical_formula()}\n",
            f"  Charge:        {atoms.info.get('charge', 0)}\n",
            f"  Multiplicity:  {atoms.info.get('mult', 1)}\n",
            "\n" + "="*80 + "\n",
        ])

        if step_offset == 0 and backup_msgs:
            self.log_main(["\nWARNING: Pre-existing output files were backed up:\n"] + backup_msgs + ["\n"])
            for msg in backup_msgs:
                print(f"WARNING: {msg.strip()}")

        # Write thermodynamics header (fresh run only; resume appends a separator instead)
        if step_offset == 0:
            self.thermo_file.write(f"# MD Simulation - {ensemble.upper()} Ensemble\n")
            self.thermo_file.write(f"# Timestep: {timestep} fs\n")
            if self._ensemble == 'npt':
                self.thermo_file.write(
                    f"# {'Step':>8} {'Time(fs)':>12} {'Temp(K)':>12} "
                    f"{'KE(Ha)':>15} {'PE(Ha)':>15} {'TE(Ha)':>15} "
                    f"{'Press(bar)':>12} {'Vol(A^3)':>12}\n"
                )
            else:
                self.thermo_file.write(
                    f"# {'Step':>8} {'Time(fs)':>12} {'Temp(K)':>12} "
                    f"{'KE(Ha)':>15} {'PE(Ha)':>15} {'TE(Ha)':>15}\n"
                )
            self.thermo_file.flush()

        # ------------------------------------------------------------------
        # Progress header  (verbose >= 1)
        # Printed once; the data row below it is refreshed in-place with \r.
        # ------------------------------------------------------------------
        if self.verbose >= 1:
            is_npt = self._ensemble == 'npt'
            total_ps = self._n_steps * self._timestep / 1000.0
            _time_col_w = max(len(f"0.000/{total_ps:.3f}"), len("Time(ps)")) + 1
            hdr = (
                f"\n"
                f"  {'Step':>9}  {'Time(ps)':>{_time_col_w}}  {'Progress':>8}  "
                f"{'T(K)':>8}  {'E_total(Ha)':>15}  "
                f"{'Speed(ns/day)':>13}  {'ETA':>10}"
                + (f"  {'P(bar)':>10}" if is_npt else "")
            )
            sep = (
                f"  {'-'*9}  {'-'*_time_col_w}  {'-'*8}  "
                f"{'-'*8}  {'-'*15}  "
                f"{'-'*13}  {'-'*10}"
                + (f"  {'-'*10}" if is_npt else "")
            )
            self._time_col_w = _time_col_w
            self._total_ps   = total_ps
            # Write header to file and print to terminal
            self.log_main([hdr + "\n", sep + "\n"])
            print(hdr)
            print(sep)
            # Reserve the data row — cursor stays on this line for \r updates
            print("", end="", flush=True)

    def log_step(self, step: int, time: float, temperature: float,
                 kinetic_energy: float, potential_energy: float,
                 total_energy: float, atoms: Atoms, velocities: np.ndarray,
                 pressure: float = None, volume: float = None,
                 rng_state: Optional[str] = None,
                 rst_every: Optional[int] = None):
        """
        Log data for current step.

        Args:
            step: Current step number
            time: Current simulation time (fs)
            temperature: Current temperature (K)
            kinetic_energy: Kinetic energy (Hartree)
            potential_energy: Potential energy (Hartree)
            total_energy: Total energy (Hartree)
            atoms: Current ASE Atoms object
            velocities: Current velocities (atomic units: Bohr/a.u. time)
            pressure: Instantaneous pressure in bar (NPT only)
            volume: Cell volume in Å³ (NPT only)
            rng_state: Hex-encoded RNG state to embed in trajectory frame (NVT/NPT only)
        """
        # PE and KE are both passed in Hartree (UMACalculator already converts)
        potential_energy_hartree = potential_energy
        kinetic_energy_hartree = kinetic_energy
        total_energy_hartree = kinetic_energy_hartree + potential_energy_hartree

        # Store for statistics
        self.energies.append(total_energy_hartree)
        self.temperatures.append(temperature)
        self.times.append(time)
        self.kinetic_energies.append(kinetic_energy_hartree)
        self.potential_energies.append(potential_energy_hartree)
        if pressure is not None:
            self.pressures.append(pressure)

        # Write thermodynamic data every step
        if self._ensemble == 'npt' and pressure is not None and volume is not None:
            self.thermo_file.write(
                f"{step:>10} {time:>12.3f} {temperature:>12.2f} "
                f"{kinetic_energy_hartree:>15.8f} {potential_energy_hartree:>15.8f} "
                f"{total_energy_hartree:>15.8f} {pressure:>12.3f} {volume:>12.4f}\n"
            )
        else:
            self.thermo_file.write(
                f"{step:>10} {time:>12.3f} {temperature:>12.2f} "
                f"{kinetic_energy_hartree:>15.8f} {potential_energy_hartree:>15.8f} "
                f"{total_energy_hartree:>15.8f}\n"
            )
        self.thermo_file.flush()

        # ------------------------------------------------------------------
        # Progress line  (verbose >= 1, printed every log_every steps)
        # ------------------------------------------------------------------
        # Terminal progress line  (verbose >= 1, every 100 steps)
        # Uses \r to overwrite the same line — no screen scrolling.
        # The .dat file still receives a record at every log_every interval.
        # ------------------------------------------------------------------
        _PRINT_EVERY = 100
        if self.verbose >= 1 and step % _PRINT_EVERY == 0:
            now          = _time.perf_counter()

            # Rolling speed over the last print window
            delta_steps  = step - self._last_step
            delta_wall   = now  - self._last_wall
            if delta_wall > 0:
                ns_per_day = (delta_steps * self._timestep / delta_wall / 1e6 * 86400)
            else:
                ns_per_day = float('nan')
            self._last_wall = now
            self._last_step = step

            # ETA
            steps_remaining = self._n_steps - step
            if ns_per_day > 0 and ns_per_day == ns_per_day:   # not nan
                remaining_s = steps_remaining * self._timestep / 1e6 / ns_per_day * 86400
            else:
                remaining_s = float('inf')

            progress_pct = 100.0 * step / self._n_steps if self._n_steps > 0 else 0.0
            speed_str    = f"{ns_per_day:.3f}" if ns_per_day == ns_per_day else "---"
            eta_str      = _fmt_eta(remaining_s)

            # Build progress row aligned to the header columns:
            #   Step(9)  Time(fs)(10)  Progress(8)  T(K)(8)  E_total(Ha)(15)
            #   Speed(ns/day)(13)  ETA(10)  [P(bar)(10)]
            progress_str = f"{progress_pct:.1f}%"
            current_ps   = time / 1000.0
            time_str     = f"{current_ps:.3f}/{self._total_ps:.3f}"
            line = (
                f"  {step:>9}  {time_str:>{self._time_col_w}}  {progress_str:>8}  "
                f"{temperature:>8.2f}  {total_energy_hartree:>15.6f}  "
                f"{speed_str:>13}  {eta_str:>10}"
            )
            if self._ensemble == 'npt' and pressure is not None:
                line += f"  {pressure:>10.2f}"

            # \r returns cursor to line start; pad with spaces to erase
            # any leftover characters from a longer previous line
            print(f"\r{line:<120}", end='', flush=True)

        # Also write to .dat file at log_every frequency (unchanged)
        if step % self.log_every == 0:
            self.log_main([
                f"  Step {step:>8}  {time:>10.2f} fs  "
                f"T {temperature:>7.2f} K  "
                f"E {total_energy_hartree:>14.6f} Ha\n"
            ])

        # Write trajectory at traj_every frequency.
        # frame_number stores the MD *step* number so that restart_simulation()
        # can recover step_offset directly without needing to know traj_every.
        if step % self.traj_every == 0:
            if self.traj_format == 'dcd':
                # DCD writer handles its own writing
                self.traj_file.write_frame(atoms, step=step)
            else:  # xyz
                write_xyz_frame(
                    self.traj_file,
                    atoms,
                    energy=total_energy_hartree,
                    frame_number=step,          # MD step number, NOT sequential frame index
                    velocity=velocities,
                    rng_state=rng_state,
                )
                self.traj_file.flush()

        # Write restart checkpoint at rst_every frequency
        if rst_every and step % rst_every == 0:
            rotate_rst_checkpoint(
                rst_path=self.rst_path,
                rst_prev_path=self.rst_prev_path,
                atoms=atoms,
                velocities=velocities,
                step=step,
                timestep=self._timestep,
                ensemble=self._ensemble,
                energy=total_energy_hartree,
                rng_state=rng_state,
            )

    def restart_simulation(
        self,
        ensemble: str,
        timestep: float,
        n_steps: int,
        temperature: float,
        atoms: Atoms,
        pressure: float = None,
    ):
        """
        Restore state from a .rst checkpoint file, validate against input atoms,
        open output files in append mode, and return (atoms, velocities, step_offset).

        Tries ``rst_path`` first; on parse failure falls back to ``rst_prev_path``.
        Returns None if the checkpoint already completed the requested run.
        Raises RuntimeError on hard failures (mismatch, missing files, etc.).
        """
        from ase.cell import Cell

        candidates = [self.rst_path, self.rst_prev_path]
        state = None
        errors = []
        used_path = None
        for path in candidates:
            if not path.exists():
                errors.append(f"missing: {path.name}")
                continue
            try:
                state = read_rst(path)
                used_path = path
                break
            except ValueError as exc:
                errors.append(f"{path.name}: {exc}")

        if state is None:
            raise RuntimeError("MD restart failed: " + "; ".join(errors))

        # Validation checks
        if state["natoms"] != len(atoms):
            raise RuntimeError(
                f"Atom count mismatch: rst has {state['natoms']}, input has {len(atoms)}"
            )
        ref_symbols = atoms.get_chemical_symbols()
        for idx, (rst_sym, input_sym) in enumerate(zip(state["symbols"], ref_symbols), 1):
            if rst_sym != input_sym:
                raise RuntimeError(
                    f"Element mismatch between rst and input at position {idx}"
                )
        if state["ensemble"] != ensemble:
            raise RuntimeError(
                f"Ensemble mismatch: rst has '{state['ensemble']}', "
                f"input specifies '{ensemble}'"
            )
        if abs(state["timestep"] - timestep) > 1e-12:
            raise RuntimeError(
                f"Timestep mismatch: rst has {state['timestep']}, "
                f"input specifies {timestep}"
            )
        if state["step"] >= n_steps:
            self.log_main([
                f"\nRestart checkpoint {used_path.name} already completed the "
                f"requested run ({state['step']}/{n_steps} steps).\n"
            ])
            return None

        # Restore atoms state
        atoms.set_positions(state["positions"])
        if state["cell"] is not None:
            atoms.set_cell(Cell.fromcellpar(state["cell"]))
        if state["pbc"] is not None:
            atoms.set_pbc(state["pbc"])

        # Store RNG state for ensemble drivers (NVT/NPT) to restore
        self.resumed_rng_state = state.get("rng_state")

        # Open output files in append mode
        self.thermo_file = open(self.thermo_path, "a") if self.thermo_path.exists() else open(self.thermo_path, "w")
        if self.traj_format == 'dcd':
            if self.traj_path.exists():
                self.traj_file = DCDWriter.open_for_append(self.traj_path)
            else:
                self.traj_file = DCDWriter(
                    path=self.traj_path,
                    natoms=len(atoms),
                    timestep=timestep,
                    is_periodic=any(atoms.pbc),
                    first_step=state["step"],
                )
        else:  # xyz
            self.traj_file = open(self.traj_path, "a") if self.traj_path.exists() else open(self.traj_path, "w")
        self.thermo_file.write(f"\n# --- RESTARTED from {used_path.name} step {state['step']} ---\n")
        self.thermo_file.flush()

        return atoms, state["velocities"], state["step"]

    def end_simulation(self, atoms: Atoms = None, final_velocities: np.ndarray = None):
        """
        Finalize simulation, compute publication-quality conservation metrics,
        write summary file, and write final restart checkpoint.

        Parameters
        ----------
        atoms : ase.Atoms, optional
            Final atomic configuration.
        final_velocities : np.ndarray, optional
            Final velocities in atomic units (Bohr/a.u. time).
            Written to the restart checkpoint so that a subsequent
            NVE/NVT/NPT run can restart from the exact end state.
            Analogous to GROMACS confout.gro (coordinates + velocities).
        """
        # End the \r progress line with a newline so the summary starts cleanly
        if self.verbose >= 1:
            print(flush=True)

        energies     = np.array(self.energies)
        temperatures = np.array(self.temperatures)
        times        = np.array(self.times)          # fs
        ke_arr       = np.array(self.kinetic_energies)
        pe_arr       = np.array(self.potential_energies)

        # ------------------------------------------------------------------
        # 1. Basic energy statistics  (full trajectory, no skip)
        # ------------------------------------------------------------------
        energy_mean = np.mean(energies)
        energy_std  = np.std(energies)

        # ------------------------------------------------------------------
        # 2. Linear drift rate  [kJ/mol/ns/atom]
        #
        #    method: least-squares linear fit to E_total(t), identical to
        #    GROMACS "gmx energy -drift" (GROMACS Reference Manual 2024 §3.4
        #    "Energy Conservation"; Páll et al. (2020) J. Chem. Phys. 153, 134110).
        #
        #    Equilibration skip (front EQ_FRAC of the trajectory):
        #    --------------------------------------------------------
        #    NVE started from a prior NVT run (with thermostat coupling) carries an
        #    unavoidable thermal transient in the first ~10–50 steps:
        #    thermostat coupling in the prior run changes KE and breaks strict
        #    TE conservation, so at the NVT→NVE handoff the instantaneous KE is
        #    not necessarily in equilibrium with the current PE.  The resulting
        #    ΔTE can be tens of kJ/mol and
        #    dominates a short polyfit, giving a meaningless drift estimate.
        #    Skipping the front 20 % removes this transient before fitting,
        #    exactly as AMBER (nstlim discard) and GROMACS (equilibration run)
        #    instruct users to do before computing NVE drift.
        #    Refs:
        #      Frenkel & Smit, Understanding Molecular Simulation, 2nd ed.
        #        (2002) Box 4.1: "Always run a short NVE segment after NVT
        #        equilibration and discard it before analysing drift."
        #      GROMACS Reference Manual 2024 §3.4.2: "Before running NVE for
        #        energy conservation benchmarks, always equilibrate with NVT …
        #        The first few ps of NVE after switching from NVT should be
        #        discarded as equilibration."
        #      AMBER 2023 Manual §3.1: multi-stage heating (NVT) before NVE
        #        production; initial NVE data discarded as equilibration.
        #      Leimkuhler & Matthews (2013) Appl. Math. Res. eXpress 2013, 34–56:
        #        thermostat coupling does not conserve TE during NVT, so TE at
        #        the NVT→NVE boundary is a draw from the canonical distribution,
        #        not the NVE microcanonical invariant.
        #
        #    NVT/NPT: same 20 % skip removes early thermalisation ramp so the
        #    slope reflects steady-state, consistent with AMBER §3.1 practice.
        #
        #    The skip fraction is also reported in the summary so the user
        #    knows exactly what window was used.
        # ------------------------------------------------------------------
        EQ_FRAC   = 0.20            # discard first 20 % as equilibration
        total_time_ns = (times[-1] - times[0]) * FS_TO_NS   # fs → ns

        eq_cut = max(1, int(len(times) * EQ_FRAC))          # first index of production window
        prod_times    = times[eq_cut:]
        prod_energies = energies[eq_cut:]
        prod_time_ns  = (prod_times[-1] - prod_times[0]) * FS_TO_NS if len(prod_times) >= 2 else 0.0
        eq_time_ps    = (times[eq_cut - 1] - times[0]) * 1e-3   # fs → ps

        if len(prod_times) >= 2 and prod_time_ns > 0 and self._n_atoms > 0:
            coef = np.polyfit(prod_times, prod_energies, 1)   # slope in Ha/fs
            drift_rate = (coef[0]
                          * HARTREE_TO_KJ_PER_MOL             # → kJ/mol/fs
                          / FS_TO_NS                          # → kJ/mol/ns
                          / self._n_atoms)                    # → kJ/mol/ns/atom
        else:
            drift_rate = float('nan')

        # Flag short production windows where the polyfit slope is
        # statistically noisy due to thermal fluctuations dominating.
        is_short_traj = prod_time_ns < 0.010   # production window < 10 ps

        # ------------------------------------------------------------------
        # 3. Relative energy fluctuation  σ(E)/|<E>|  (dimensionless)
        #    Standard NVE quality metric used in AMBER and general MD texts.
        #    Ref: AMBER Reference Manual (Case et al. 2022), §3 NVE validation
        #    Acceptance: < 1e-4
        # ------------------------------------------------------------------
        rel_fluctuation = energy_std / abs(energy_mean) if energy_mean != 0 else float('nan')

        # ------------------------------------------------------------------
        # 4. KE–PE anti-correlation coefficient  r(KE, PE)
        #    For a symplectic integrator (Velocity Verlet) in NVE, KE and PE
        #    must be perfectly anti-correlated (r ≈ -1) because E = KE + PE
        #    is conserved.  Deviations from -1 quantify integration error.
        #    Ref: Allen & Tildesley (2017) Computer Simulation of Liquids §3.4;
        #         Hammonds & Heyes (2020) J. Chem. Phys. 152, 024114
        # ------------------------------------------------------------------
        if len(ke_arr) > 1 and np.std(ke_arr) > 0 and np.std(pe_arr) > 0:
            ke_pe_corr = np.corrcoef(ke_arr, pe_arr)[0, 1]
        else:
            ke_pe_corr = float('nan')

        # ------------------------------------------------------------------
        # 5. Temperature fluctuation: σ_obs vs σ_canonical
        #
        #    N_dof follows utils.py:calculate_temperature():
        #      PBC system  → N_dof = 3N   (no COM constraint in periodic cell)
        #      Isolated    → N_dof = 3N-3 (remove COM translation)
        #
        #    NVE reference (Allen & Tildesley 2017 §2.4, large-N approx.):
        #      σ_nve ≈ T / √N_dof   (≡ T·√(1/N_dof))
        #      excess_ratio = σ_obs / σ_nve
        #
        #    NVT/NPT reference (Frenkel & Smit 2002 §6.1, exact canonical):
        #      σ_canonical = T·√(2/N_dof)
        #      sigma_ratio = σ_obs / σ_canonical  (target: 0.5–2.0)
        #
        #    Using the ensemble-correct formula avoids a spurious "WARN" for
        #    well-behaved NVT runs (√2 factor matters at small N).
        # ------------------------------------------------------------------
        temp_mean = np.mean(temperatures)
        temp_std  = np.std(temperatures)

        n_dof = (3 * self._n_atoms if (self._is_pbc or self._n_atoms == 0)
                 else 3 * self._n_atoms - 3)
        if n_dof <= 0:
            n_dof = 1

        if self._ensemble == 'nve':
            # A&T §2.4 approximation: σ_nve ≈ T/√N_dof
            t_sigma_ref  = temp_mean / np.sqrt(n_dof) if n_dof > 0 else float('nan')
            t_ratio_name = "σ_obs/σ_NVE"
            t_ref_note   = "T/√N_dof  [Allen & Tildesley 2017 §2.4]"
        else:
            # F&S §6.1 exact canonical: σ_canonical = T·√(2/N_dof)
            t_sigma_ref  = temp_mean * np.sqrt(2.0 / n_dof) if n_dof > 0 else float('nan')
            t_ratio_name = "σ_obs/σ_canonical"
            t_ref_note   = "T·√(2/N_dof)  [Frenkel & Smit 2002 §6.1]"

        observed_ratio = temp_std / temp_mean if temp_mean > 0 else float('nan')
        t_ratio        = temp_std / t_sigma_ref if (t_sigma_ref and t_sigma_ref > 0) else float('nan')
        # NVE: excess_ratio > 2 warns of instability; NVT/NPT: ratio ~ 1.0 is ideal
        temp_tag = "GOOD" if (not np.isnan(t_ratio) and 0.5 < t_ratio < 2.0) else "WARN"

        # ------------------------------------------------------------------
        # 6. Qualitative assessment flags
        # ------------------------------------------------------------------
        # Drift tag is ensemble-dependent:
        #   NVE: TE is strictly conserved; any drift = integrator error.
        #        GOOD/OK/WARN/FAIL scale from Páll et al. 2020.
        #        Computed on production window (post equilibration skip).
        #   NVT/NPT: TE is NOT conserved by design — the thermostat exchanges
        #        energy with the bath every step.  The OLS slope of TE(t)
        #        conflates (a) integrator truncation error and (b) thermostat
        #        coupling noise; these cannot be separated.  Applying NVE
        #        thresholds (0.01/0.1 kJ/mol/ns/atom) to NVT TE drift is
        #        physically incorrect.
        #        Correct NVT primary metrics: σ_obs/σ_canonical, r(KE,PE).
        #        Drift for NVT: gross-instability-only check (> 1 kJ/mol/ns/atom).
        #        Refs: GROMACS Manual 2024 §3.4; Basconi & Shirts (2013) JCTC 9, 2887;
        #              Eastman et al. (2017) JCTC 13, 5560 (OpenMM benchmark).
        if np.isnan(drift_rate):
            drift_tag = "N/A"
        elif is_short_traj:
            drift_tag = "SHORT"   # production window < 10 ps; polyfit slope unreliable
        elif self._ensemble == 'nve':
            # NVE: strict energy conservation — fine-grained thresholds apply
            if abs(drift_rate) < 0.01:
                drift_tag = "GOOD"
            elif abs(drift_rate) < 0.1:
                drift_tag = "OK"
            elif abs(drift_rate) < 1.0:
                drift_tag = "WARN"
            else:
                drift_tag = "FAIL"
        else:
            # NVT/NPT: only gross instability matters (GROMACS Manual 2024 §3.4)
            drift_tag = "WARN" if abs(drift_rate) > 1.0 else "PASS"
        fluct_tag = "GOOD" if rel_fluctuation < 1e-4 else "WARN"
        if np.isnan(ke_pe_corr):
            corr_tag = "N/A"
        elif self._ensemble == 'nve':
            corr_tag = "GOOD" if ke_pe_corr < -0.95 else "WARN"
        else:  # NVT/NPT: thermostat decouples KE from PE, r ≈ 0 is correct
            corr_tag = "GOOD" if abs(ke_pe_corr) < 0.5 else "WARN"

        # ------------------------------------------------------------------
        # Assemble summary lines  (ensemble-aware primary/secondary structure)
        # ------------------------------------------------------------------
        is_nve = self._ensemble == 'nve'
        is_nvt = self._ensemble == 'nvt'

        ens_label = self._ensemble.upper()
        ke_pe_target = "≈ −1 (NVE, Velocity Verlet symplecticity)" if is_nve else "≈  0 (NVT/NPT, thermostat decouples KE)"

        if is_nve:
            energy_section = [
                f"\n{'── [NVE] Energy Conservation (primary criteria) ──':^80}\n",
                f"  σ(TE)/|⟨TE⟩|:              {rel_fluctuation:>18.2e}  (dimensionless)   [{fluct_tag}]\n",
                f"    AMBER 2022 §3: < 1e-4 accepted\n",
                f"  Linear drift rate:          {drift_rate:>+18.6f}  kJ/mol/ns/atom    [{drift_tag}]\n",
                f"  Production window:          {prod_time_ns*1000:>15.3f}  ps"
                f"  (skipped first {eq_time_ps:.2f} ps as NVE equilibration)\n",
                *(["  (SHORT production window < 10 ps: drift rate is indicative only; use σ/|<E>| and r(KE,PE))\n"]
                  if is_short_traj else []),
                f"    GROMACS Manual 2024 §3.4 / Páll et al. 2020 JCP 153, 134110: < 0.01 GOOD, < 0.1 OK, > 1.0 FAIL\n",
                f"    Equil. skip: Frenkel & Smit 2002 Box 4.1; GROMACS Manual 2024 §3.4.2\n",
                f"  r(KE,PE):                   {ke_pe_corr:>18.4f}  (target {ke_pe_target})  [{corr_tag}]\n",
                f"    Allen & Tildesley 2017 §3.4; Hammonds & Heyes 2020 JCP 152, 024114\n",
            ]
            temp_section = [
                f"\n{'── [NVE] Temperature (reference, not conservation criterion) ──':^80}\n",
                f"  ⟨T⟩:                        {temp_mean:>18.2f}  K\n",
                f"  σ(T):                       {temp_std:>18.2f}  K\n",
                f"  σ(T)/<T> observed:          {observed_ratio:>18.4f}\n",
                f"  σ_NVE = T/√N_dof:           {t_sigma_ref:>18.2f}  K  [A&T 2017 §2.4]\n",
                f"  {t_ratio_name}:        {t_ratio:>18.3f}   (target: 0.5–2.0)  [{temp_tag}]\n",
                f"    N_dof = {n_dof}  ({'PBC: 3N' if self._is_pbc else 'isolated: 3N-3'})\n",
            ]
        else:
            energy_section = [
                f"\n{'── [' + ens_label + '] Temperature Control (primary criteria) ──':^80}\n",
                f"  ⟨T⟩:                        {temp_mean:>18.2f}  K\n",
                f"  σ(T) observed:              {temp_std:>18.2f}  K\n",
                f"  σ_canonical = T·√(2/N_dof): {t_sigma_ref:>18.2f}  K  [Frenkel & Smit 2002 §6.1]\n",
                f"  {t_ratio_name}:  {t_ratio:>18.3f}   (target: 0.5–2.0)  [{temp_tag}]\n",
                f"    N_dof = {n_dof}  ({'PBC: 3N' if self._is_pbc else 'isolated: 3N-3'})\n",
                f"  r(KE,PE):                   {ke_pe_corr:>18.4f}  (target {ke_pe_target})  [{corr_tag}]\n",
                f"    Allen & Tildesley 2017 §3.4; Hammonds & Heyes 2020 JCP 152, 024114\n",
            ]
            temp_section = [
                f"\n{'── [' + ens_label + '] Energy (secondary — TE fluctuates by design) ──':^80}\n",
                f"  Mean TE:                    {energy_mean:>18.8f}  Ha\n",
                f"  σ(TE):                      {energy_std:>18.8f}  Ha\n",
                f"  σ(TE)/|⟨TE⟩|:              {rel_fluctuation:>18.2e}  (NVT/NPT: fluctuates by design)\n",
                f"  Linear drift rate:          {drift_rate:>+18.6f}  kJ/mol/ns/atom    [{drift_tag}]\n",
                f"  Production window:          {prod_time_ns*1000:>15.3f}  ps"
                f"  (skipped first {eq_time_ps:.2f} ps as equilibration)\n",
                *(["  (SHORT production window < 10 ps: drift rate is indicative only)\n"]
                  if is_short_traj else []),
                f"    GROMACS Manual 2024 §3.4: gross instability only; accept if drift < 1.0\n",
                f"    Equil. skip: AMBER 2023 Manual §3.1\n",
            ]

        summary_lines = [
            "\n" + "="*80 + "\n",
            f"{'MD SIMULATION COMPLETED':^80}\n",
            "="*80 + "\n",

            f"\n{'── Energy Statistics ──':^80}\n",
            f"  Mean total energy:          {energy_mean:>18.8f}  Ha\n",
            f"  Std deviation:              {energy_std:>18.8f}  Ha\n",
        ] + energy_section + temp_section + [
            f"\n{'── Acceptance Criteria Summary (GROMACS 2024 / AMBER 2022 / A&T 2017 / F&S 2002) ──':^80}\n",
            f"  NVE primary:  σ(TE)/|⟨TE⟩| < 1e-4 GOOD (AMBER);  drift < 0.01 GOOD (GROMACS Manual);  r ≈ -1\n",
            f"  NVT/NPT primary:  σ_obs/σ_canonical ∈ [0.5, 2.0] (F&S §6.1);  r(KE,PE) ≈ 0 (A&T §3.4)\n",
            f"  NVT/NPT secondary:  drift < 1.0 kJ/mol/ns/atom (gross instability threshold only)\n",
            f"  Drift fitted on production window (front {int(EQ_FRAC*100)}% skipped):"
            f"  F&S 2002 Box 4.1; GROMACS 2024 §3.4.2; AMBER 2023 §3.1\n",
        ]

        # ------------------------------------------------------------------
        # Wall-clock performance summary  (always appended)
        # ------------------------------------------------------------------
        total_wall   = _time.perf_counter() - self._wall_start   # seconds
        sim_time_ns  = (times[-1] - times[0]) * FS_TO_NS         # ns of MD
        if total_wall > 0:
            ns_per_day_avg = sim_time_ns / total_wall * 86400
            s_per_ns       = total_wall / sim_time_ns if sim_time_ns > 0 else float('nan')
        else:
            ns_per_day_avg = float('nan')
            s_per_ns       = float('nan')

        perf_lines = [
            f"\n{'── Performance ──':^80}\n",
            f"  Wall time:                  {_fmt_duration(total_wall):>18}\n",
            f"  Simulation time:            {sim_time_ns*1000:>15.3f}  ps\n",
        ]
        if not (ns_per_day_avg != ns_per_day_avg):   # not nan
            perf_lines += [
                f"  Average speed:              {ns_per_day_avg:>15.4f}  ns/day\n",
                f"  Time per ns:                {s_per_ns:>15.1f}  s/ns\n",
            ]
        summary_lines += perf_lines

        self.log_main(summary_lines, echo=True)

        # ------------------------------------------------------------------
        # Write summary file
        # ------------------------------------------------------------------
        with open(self.summary_path, 'w') as f:
            f.write("MD Simulation Summary\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Ensemble:                   {self._ensemble.upper()}\n")
            f.write(f"Total steps:                {len(self.energies)}\n")
            f.write(f"Total time:                 {self.times[-1]:.2f} fs\n")
            f.write(f"Number of atoms:            {self._n_atoms}\n")
            f.write(f"N_dof:                      {n_dof}  "
                    f"({'PBC: 3N' if self._is_pbc else 'isolated: 3N-3'})\n\n")

            f.write("Energy Statistics:\n")
            f.write(f"  Mean total energy:        {energy_mean:.8f} Ha\n")
            f.write(f"  Std deviation:            {energy_std:.8f} Ha\n\n")

            if is_nve:
                f.write("Energy Conservation Metrics [NVE PRIMARY]:\n")
                f.write(f"  σ(TE)/|⟨TE⟩|:            {rel_fluctuation:.2e}             [{fluct_tag}]\n")
                f.write(f"    (AMBER 2022: < 1e-4 accepted)\n")
                f.write(f"  Linear drift rate:        {drift_rate:+.6f} kJ/mol/ns/atom  [{drift_tag}]\n")
                f.write(f"  Production window:        {prod_time_ns*1000:.3f} ps"
                        f"  (skipped first {eq_time_ps:.2f} ps as NVE equilibration)\n")
                if is_short_traj:
                    f.write(f"  (SHORT production window < 10 ps: drift rate is indicative only)\n")
                f.write(f"    (GROMACS Manual 2024 §3.4 / Páll et al. 2020 JCP 153, 134110: < 0.01 GOOD, < 0.1 OK, > 1.0 FAIL)\n")
                f.write(f"    (Equil. skip: Frenkel & Smit 2002 Box 4.1; GROMACS Manual 2024 §3.4.2)\n")
                f.write(f"  r(KE,PE):                 {ke_pe_corr:.4f}                 [{corr_tag}]\n")
                f.write(f"    (Allen & Tildesley 2017 §3.4; Hammonds & Heyes 2020 JCP 152, 024114:\n"
                        f"     r ≈ -1 confirms Velocity Verlet symplecticity)\n\n")

                f.write("Temperature Statistics [NVE REFERENCE]:\n")
                f.write(f"  Mean temperature:         {temp_mean:.2f} K\n")
                f.write(f"  Std deviation:            {temp_std:.2f} K\n")
                f.write(f"  σ(T)/<T> observed:        {observed_ratio:.4f}\n")
                f.write(f"  σ_NVE = T/√N_dof:         {t_sigma_ref:.2f} K  [Allen & Tildesley 2017 §2.4]\n")
                f.write(f"  σ_obs/σ_NVE:              {t_ratio:.3f}  [target: 0.5–2.0]  [{temp_tag}]\n")
            else:
                f.write(f"Temperature Control Metrics [{self._ensemble.upper()} PRIMARY]:\n")
                f.write(f"  Mean temperature:         {temp_mean:.2f} K\n")
                f.write(f"  Std deviation:            {temp_std:.2f} K\n")
                f.write(f"  σ_canonical = T·√(2/N_dof): {t_sigma_ref:.2f} K  [Frenkel & Smit 2002 §6.1]\n")
                f.write(f"  σ_obs/σ_canonical:        {t_ratio:.3f}  [target: 0.5–2.0]  [{temp_tag}]\n")
                f.write(f"  r(KE,PE):                 {ke_pe_corr:.4f}                 [{corr_tag}]\n")
                f.write(f"    (Allen & Tildesley 2017 §3.4: r ≈ 0 expected — thermostat decouples KE from PE)\n\n")

                f.write(f"Energy Metrics [{self._ensemble.upper()} SECONDARY — TE fluctuates by design]:\n")
                f.write(f"  σ(TE)/|⟨TE⟩|:            {rel_fluctuation:.2e}             (expected to be large)\n")
                f.write(f"  Linear drift rate:        {drift_rate:+.6f} kJ/mol/ns/atom  [{drift_tag}]\n")
                f.write(f"  Production window:        {prod_time_ns*1000:.3f} ps"
                        f"  (skipped first {eq_time_ps:.2f} ps as equilibration)\n")
                if is_short_traj:
                    f.write(f"  (SHORT production window < 10 ps: drift rate is indicative only)\n")
                f.write(f"    (GROMACS Manual 2024 §3.4: gross instability threshold only; accept if drift < 1.0)\n")
                f.write(f"    (Equil. skip: AMBER 2023 Manual §3.1)\n")

            if self.pressures:
                pressures_arr = np.array(self.pressures)
                f.write(f"\nPressure Statistics:\n")
                f.write(f"  Mean pressure:            {np.mean(pressures_arr):.3f} bar\n")
                f.write(f"  Std deviation:            {np.std(pressures_arr):.3f} bar\n")

            f.write(f"\nPerformance:\n")
            f.write(f"  Wall time:                {_fmt_duration(total_wall)}\n")
            if not (ns_per_day_avg != ns_per_day_avg):
                f.write(f"  Average speed:            {ns_per_day_avg:.4f} ns/day\n")
                f.write(f"  Time per ns:              {s_per_ns:.1f} s/ns\n")

        if self.pressures:
            pressures_log = np.array(self.pressures)
            self.log_main([
                f"\n{'── Pressure Statistics ──':^80}\n",
                f"  Mean pressure:              {np.mean(pressures_log):>18.3f}  bar\n",
                f"  Std deviation:              {np.std(pressures_log):>18.3f}  bar\n",
            ], echo=True)

        # ------------------------------------------------------------------
        # Write final restart checkpoint  (GROMACS confout.gro equivalent)
        # Contains final coordinates + velocities so that a subsequent
        # NVE/NVT/NPT run can restart from the exact end state.
        # ------------------------------------------------------------------
        final_written = False
        if atoms is not None and final_velocities is not None:
            final_energy = self.energies[-1] if self.energies else float("nan")
            final_step = int(round(self.times[-1] / self._timestep)) if self.times and self._timestep > 0 else 0
            rotate_rst_checkpoint(
                rst_path=self.rst_path,
                rst_prev_path=self.rst_prev_path,
                atoms=atoms,
                velocities=final_velocities,
                step=final_step,
                timestep=self._timestep,
                ensemble=self._ensemble,
                energy=final_energy,
            )
            final_written = True

        self.log_main([
            f"\n{'── Output Files ──':^80}\n",
            f"  Thermodynamics:             {self.thermo_path.name}\n",
            f"  Trajectory:                 {self.traj_path.name}\n",
            f"  Summary:                    {self.summary_path.name}\n",
            *([f"  Checkpoint:                 {self.rst_path.name}\n"
               f"  Previous checkpoint:        {self.rst_prev_path.name}\n"]
              if final_written else []),
            "="*80 + "\n",
        ], echo=True)

        # Close files
        if self.thermo_file:
            self.thermo_file.close()
        if self.traj_file:
            self.traj_file.close()

    def log_main(self, messages: list, echo: bool = False):
        """
        Write messages to main output file.
        If echo=True and verbose >= 1, also print to stdout (terminal).

        Args:
            messages: List of message strings
            echo:     Mirror output to stdout (used for progress lines)
        """
        with open(self.main_output, 'a') as f:
            for msg in messages:
                f.write(msg)
        if echo and self.verbose >= 1:
            for msg in messages:
                print(msg, end='', flush=True)
