from typing import Iterator, List, Optional, Union
from dataclasses import dataclass
from contextlib import ExitStack
import math

from ase import Atoms

from ...calculator._batch_eval import (
    EnergyEvaluator,
    PathEvaluator,
    energy_forces_one,
    shared_calculator,
    structures_have_constraints,
    supports_batch_calculation,
)
from ...calculator._batch_utils import preserve_calculator_state
from ..jobABC import JobABC
from maple.function.timer import timer

_AUTO_TRAJECTORY_WINDOW = 32

@dataclass
class SPParams:
    """Parameters for Single Point calculation."""
    verbose: int = 0  # 0=coordinates+energy+charge/mult, 1=+gradients

class SinglePoint(JobABC):

    def __init__(self, output: str, atoms: Union[Atoms, List[Atoms]],
                 paras: Optional[dict] = None):
        super().__init__(output)
        self.atoms = atoms
        self.is_trajectory = isinstance(atoms, list)

        # Initialize params
        self.params = self._init_params(SPParams, paras, ("sp", "SP"))
        self.verbose = self.params.verbose

    def run(self):
        if self.is_trajectory:
            self._run_trajectory()
        else:
            self._run_single()

    def _run_single(self):
        """Original single-point calculation logic."""
        with timer("Single Point Energy Calculation"):
            energy = self.atoms.get_potential_energy()
            self.log_info(self._single_energy_lines(energy))

    def _single_energy_lines(self, energy: float) -> list:
        """Return single-structure SP result lines for the selected verbosity."""
        lines = ["\n"]
        lines.extend(self._charge_mult_lines(self.atoms))
        lines.append(f"Energy: {energy:.10f} Hartree\n")
        if self.verbose >= 1:
            lines.extend(self._gradient_lines(self.atoms))
        return lines

    def _charge_mult_lines(self, atoms: Atoms) -> list:
        """Return charge and multiplicity metadata lines for SP output."""
        charge = atoms.info.get('charge', 0)
        mult = atoms.info.get('mult', 1)
        return [f"Charge: {charge}, Multiplicity: {mult}\n"]

    def _gradient_lines(self, atoms: Atoms, forces=None) -> list:
        """Return per-atom energy gradients for detailed SP output."""
        if forces is None:
            forces = atoms.get_forces()
        gradients = -forces
        symbols = atoms.get_chemical_symbols()
        lines = [
            "\nGradients (Hartree/Angstrom):\n",
            "  Gradient = -Force\n",
            "  Atom  El"
            "        Gx              Gy              Gz\n",
        ]
        for i, (sym, gradient) in enumerate(zip(symbols, gradients), start=1):
            lines.append(
                f"  {i:<4} {sym:<2}"
                f" {gradient[0]:>15.8f} {gradient[1]:>15.8f} {gradient[2]:>15.8f}\n"
            )
        return lines

    def _trajectory_frame_lines(
        self,
        idx: int,
        atoms_frame: Atoms,
        energy_hartree: float,
        forces=None,
    ) -> list:
        """Return trajectory-frame SP result lines for the selected verbosity."""
        lines = [
            f"\n{('Frame ' + str(idx)):=^80}\n",
        ]
        lines.extend(self._charge_mult_lines(atoms_frame))
        lines.extend([
            f"Energy: {energy_hartree:.10f} Hartree\n\n",
            "Coordinates (Angstrom):\n",
        ])
        symbols = atoms_frame.get_chemical_symbols()
        positions = atoms_frame.get_positions()
        for i, (sym, pos) in enumerate(zip(symbols, positions), start=1):
            lines.append(f"  {i:<4} {sym:<2} {pos[0]:>15.8f} {pos[1]:>15.8f} {pos[2]:>15.8f}\n")
        if self.verbose >= 1:
            lines.extend(self._gradient_lines(atoms_frame, forces=forces))
        lines.append("=" * 80 + "\n")
        return lines

    def _trajectory_chunks(self, stack: ExitStack):
        """Yield validated E/F for bounded, output-ordered frame windows."""
        calc = shared_calculator(self.atoms)
        if (
            calc is None
            or not supports_batch_calculation(calc)
            or structures_have_constraints(self.atoms)
        ):
            seen_calculators = set()
            for start, atoms_frame in enumerate(self.atoms):
                frame_calc = getattr(atoms_frame, "calc", None)
                if frame_calc is None:
                    raise ValueError(
                        "Single-point trajectory frame has no calculator."
                    )
                calc_id = id(frame_calc)
                if calc_id not in seen_calculators:
                    seen_calculators.add(calc_id)
                    stack.enter_context(preserve_calculator_state(frame_calc))
                if self.verbose >= 1:
                    energy, force = energy_forces_one(
                        frame_calc, atoms_frame, force_consistent=False,
                    )
                    yield start, [atoms_frame], [energy], [force]
                else:
                    energy = float(atoms_frame.get_potential_energy())
                    if not math.isfinite(energy):
                        raise FloatingPointError(
                            "Single-point trajectory returned a non-finite energy"
                        )
                    yield start, [atoms_frame], [energy], None
            return

        batch_size = getattr(calc, "path_batch_size", None)
        if self.verbose >= 1:
            evaluator = PathEvaluator(calc, batch_size=batch_size)
        else:
            evaluator = EnergyEvaluator(calc, batch_size=batch_size)
        window = evaluator.batch_size
        if window == "auto":
            # Keep the host-side result buffer finite even when CUDA has room
            # for the entire trajectory. The evaluator can still split/back off
            # further according to the existing native batch policy.
            caps = [
                value for value in (
                    getattr(calc, "auto_path_batch_cap", None),
                    getattr(calc, "auto_batch_hard_cap", None),
                )
                if isinstance(value, int) and not isinstance(value, bool) and value > 0
            ]
            window = min([_AUTO_TRAJECTORY_WINDOW, *caps])
        elif window == "all":
            window = max(1, len(self.atoms))

        for start in range(0, len(self.atoms), window):
            frames = self.atoms[start:start + window]
            if self.verbose >= 1:
                energies, forces = evaluator.energy_forces(frames)
            else:
                energies = evaluator.energies(frames)
                forces = None
            yield start, frames, energies, forces

    def _run_trajectory(self):
        """Process multiple independent structures."""
        with timer("Single Point Energy Calculation (Trajectory)"):
            n_frames = len(self.atoms)
            self.log_info([
                f"\nProcessing {n_frames} structures from trajectory...\n",
                "=" * 80 + "\n",
            ])

            processed = 0
            energy_min = math.inf
            energy_max = -math.inf
            with ExitStack() as stack:
                for start, frames, energies, forces in self._trajectory_chunks(stack):
                    lines = []
                    for offset, (atoms_frame, energy) in enumerate(
                        zip(frames, energies)
                    ):
                        force = None if forces is None else forces[offset]
                        lines.extend(self._trajectory_frame_lines(
                            start + offset + 1, atoms_frame, float(energy), force,
                        ))
                    self.log_info(lines)
                    processed += len(frames)
                    if len(energies):
                        energy_min = min(energy_min, min(energies))
                        energy_max = max(energy_max, max(energies))
            self.log_info(self._trajectory_summary_lines(
                processed, energy_min, energy_max,
            ))

    def _trajectory_summary_lines(
        self, count: int, energy_min: float, energy_max: float,
    ) -> Iterator[str]:
        yield f"\n{' SUMMARY ':=^80}\n"
        yield f"Total frames processed: {count}\n"
        if not count:
            yield "No structures to summarize.\n"
            yield "=" * 80 + "\n"
            return
        yield (
            f"Energy range: {energy_min:.10f} to "
            f"{energy_max:.10f} Hartree\n"
        )
        energy_span = energy_max - energy_min
        yield (
            f"Energy span: {energy_span:.10f} Hartree "
            f"({energy_span * 627.509:.4f} kcal/mol)\n"
        )
        yield "=" * 80 + "\n"
