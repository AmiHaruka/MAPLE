import os
from typing import List, Optional

from ase import Atoms
from ase.constraints import FixInternals

from ..jobABC import JobABC


def write_scan_xyz(filename: str,
                   atoms_list: List[Atoms],
                   coords_list: List[List[float]],
                   energies: List[float]):
    """Write relaxed scan XYZ file."""
    with open(filename, "w") as f:
        for i, (at, coord, e) in enumerate(zip(atoms_list, coords_list, energies)):
            pos = at.get_positions()
            symbols = at.get_chemical_symbols()

            f.write(f"{len(symbols)}\n")
            coord_str = "[" + ", ".join(f"{v:.4f}" for v in coord) + "]"
            f.write(f"Scanning combination {i+1}: {coord_str}  Energy = {e:.10f}\n")

            for s, (x, y, z) in zip(symbols, pos):
                f.write(f"{s:2s} {x: .10f} {y: .10f} {z: .10f}\n")


class Scan(JobABC):
    """
    N-dimensional relaxed scan (supports 1D, 2D, 3D).
    Uses hierarchical continuous scanning strategy.
    """

    def __init__(self, output: str, atoms: Atoms, method: str = "lbfgs", 
                 constraints: Optional[list] = None):
        super().__init__(output)
        self.atoms = atoms
        self.initial_calc = atoms.calc
        self.method = method.upper()
        self.output = output
        
        if constraints is None:
            raise ValueError("Constraints must be provided for scan.")
        self.constraints = self._convert_constraints(constraints)
        
        # Save initial custom threshold attributes
        self._threshold_attrs = ["f_max_th", "f_rms_th", "dp_max_th", "dp_rms_th"]
        self._initial_thresholds = {}
        for attr in self._threshold_attrs:
            self._initial_thresholds[attr] = getattr(self.atoms, attr, 1e10)

    def _convert_constraints(self, original_constraints: list) -> list:
        """Normalize constraint definitions."""
        converted = []
        for c in original_constraints:
            if len(c) == 4:     # distance: [a1, a2, step, steps]
                a1, a2, step, steps = c
                converted.append({
                    "type": "distance",
                    "atoms": [a1, a2],
                    "step": step,
                    "steps": steps
                })
            elif len(c) == 5:   # angle: [a1, a2, a3, step, steps]
                a1, a2, a3, step, steps = c
                converted.append({
                    "type": "angle",
                    "atoms": [a1, a2, a3],
                    "step": step,
                    "steps": steps
                })
            elif len(c) == 6:   # dihedral: [a1, a2, a3, a4, step, steps]
                a1, a2, a3, a4, step, steps = c
                converted.append({
                    "type": "dihedral",
                    "atoms": [a1, a2, a3, a4],
                    "step": step,
                    "steps": steps
                })
            else:
                raise ValueError(f"Unsupported constraint format: {c}")
        return converted

    def _generate_scan_values(self) -> List[List[float]]:
        """Generate scan grid values for each dimension."""
        scan_values = []
        for con in self.constraints:
            ctype = con["type"]
            step = con["step"]
            steps = con["steps"]
            # Convert atom indices to 0-based
            atoms_idx = [a - 1 for a in con["atoms"]]

            if ctype == "distance":
                initial = self.atoms.get_distance(*atoms_idx)
            elif ctype == "angle":
                initial = self.atoms.get_angle(*atoms_idx)
            elif ctype == "dihedral":
                initial = self.atoms.get_dihedral(*atoms_idx)
            else:
                raise ValueError(f"Unknown constraint type: {ctype}")

            values = [initial + i * step for i in range(steps + 1)]
            scan_values.append(values)
        return scan_values

    def _build_fix_internals(self, current_values: List[float]) -> FixInternals:
        """Build FixInternals constraint for given values."""
        bonds, angles, dihedrals = [], [], []

        for idx, con in enumerate(self.constraints):
            ctype = con["type"]
            atoms_idx = [a - 1 for a in con["atoms"]]  # convert to 0-based
            val = current_values[idx]

            if ctype == "distance":
                bonds.append([val, atoms_idx])
            elif ctype == "angle":
                angles.append([val, atoms_idx])
            elif ctype == "dihedral":
                dihedrals.append([val, atoms_idx])

        return FixInternals(
            bonds=bonds if bonds else None,
            angles_deg=angles if angles else None,
            dihedrals_deg=dihedrals if dihedrals else None
        )

    def _safe_copy(self, atoms: Atoms) -> Atoms:
        """
        Create a deep copy of Atoms with calculator and custom attributes restored.
        """
        new_atoms = atoms.copy()
        new_atoms.info = dict(atoms.info)
        
        # Restore calculator
        new_atoms.calc = atoms.calc if atoms.calc is not None else self.initial_calc
        
        # Restore custom threshold attributes
        for attr, val in self._initial_thresholds.items():
            setattr(new_atoms, attr, val)

        return new_atoms

    def _print_progress(self, idx: int, total: int, coord: List[float]):
        """Print progress header before calling optimizer."""
        coord_str = "[" + ", ".join(f"{v:.2f}" for v in coord) + "]"
        self.log_info("\n")
        self.log_info("-" * 70)
        self.log_info(f"\n            Scanning combination {idx}/{total}: {coord_str}\n")


    def _apply_constraints(self, atoms: Atoms, coord: List[float]) -> Atoms:
        """
        Apply FixInternals constraint to atoms (in-place modification).
        Returns the same atoms object with constraint applied.
        """
        constraint = self._build_fix_internals(coord)
        atoms.set_constraint(constraint)  # in-place
        
        # Ensure calculator remains valid
        if atoms.calc is None:
            atoms.calc = self.initial_calc
        
        return atoms

    def _run_optimizer(self, atoms: Atoms) -> Atoms:
        """Run geometry optimization."""
        if self.method == "LBFGS":
            from malepso.function.dispatcher.optimization.algorithm import LBFGS
            run = LBFGS(atoms, output=self.output, paras={"verbose": 0})
            return run.run()
        else:
            raise ValueError(f"Only LBFGS is supported for scan, got: {self.method}")

    def _record_result(self, atoms: Atoms, coord: List[float],
                       results: list, coords_list: list, energies: list):
        """Record a scan point result."""
        res = self._safe_copy(atoms)
        results.append(res)
        coords_list.append(coord[:])  # copy coordinate list
        energies.append(float(res.get_potential_energy(force_consistent=True)))

    def _scan_1d(self, scan_values: List[List[float]]):
        """Execute 1D scan."""
        x_values = scan_values[0]
        results, coords_list, energies = [], [], []
        
        atoms_current = self._safe_copy(self.atoms)

        for xv in x_values:
            coord = [xv]
            self._current_index += 1
            self._print_progress(self._current_index, self._total_combinations, coord)
            atoms_current = self._apply_constraints(atoms_current, coord)
            atoms_current = self._run_optimizer(atoms_current)
            self._record_result(atoms_current, coord, results, coords_list, energies)

        return results, coords_list, energies

    def _scan_2d(self, scan_values: List[List[float]]):
        """Execute 2D scan using hierarchical strategy."""
        x_values, y_values = scan_values[0], scan_values[1]
        results, coords_list, energies = [], [], []
        grid_xy = {}

        # Step 1: scan along X (y = y0)
        atoms_current = self._safe_copy(self.atoms)
        for ix, xv in enumerate(x_values):
            coord = [xv, y_values[0]]
            self._current_index += 1
            self._print_progress(self._current_index, self._total_combinations, coord)

            atoms_current = self._apply_constraints(atoms_current, coord)
            atoms_current = self._run_optimizer(atoms_current)
            
            grid_xy[(ix, 0)] = self._safe_copy(atoms_current)
            self._record_result(atoms_current, coord, results, coords_list, energies)

        # Step 2: for each X, scan along Y
        for ix, xv in enumerate(x_values):
            atoms_current = self._safe_copy(grid_xy[(ix, 0)])
            
            for iy in range(1, len(y_values)):
                coord = [xv, y_values[iy]]
                self._current_index += 1
                self._print_progress(self._current_index, self._total_combinations, coord)
                atoms_current = self._apply_constraints(atoms_current, coord)
                atoms_current = self._run_optimizer(atoms_current)
                
                grid_xy[(ix, iy)] = self._safe_copy(atoms_current)
                self._record_result(atoms_current, coord, results, coords_list, energies)

        return results, coords_list, energies

    def _scan_3d(self, scan_values: List[List[float]]):
        """Execute 3D scan using hierarchical strategy."""
        x_values, y_values, z_values = scan_values[0], scan_values[1], scan_values[2]
        results, coords_list, energies = [], [], []
        grid_xy = {}

        # Step 1: scan along X (y=y0, z=z0)
        atoms_current = self._safe_copy(self.atoms)
        for ix, xv in enumerate(x_values):
            coord = [xv, y_values[0], z_values[0]]
            self._current_index += 1
            self._print_progress(self._current_index, self._total_combinations, coord)

            atoms_current = self._apply_constraints(atoms_current, coord)
            atoms_current = self._run_optimizer(atoms_current)
            
            grid_xy[(ix, 0)] = self._safe_copy(atoms_current)
            self._record_result(atoms_current, coord, results, coords_list, energies)

        # Step 2: scan along Y (z=z0) for each X
        for ix, xv in enumerate(x_values):
            atoms_current = self._safe_copy(grid_xy[(ix, 0)])
            
            for iy in range(1, len(y_values)):
                coord = [xv, y_values[iy], z_values[0]]
                self._current_index += 1
                self._print_progress(self._current_index, self._total_combinations, coord)
                atoms_current = self._apply_constraints(atoms_current, coord)
                atoms_current = self._run_optimizer(atoms_current)
                
                grid_xy[(ix, iy)] = self._safe_copy(atoms_current)
                self._record_result(atoms_current, coord, results, coords_list, energies)

        # Step 3: scan along Z for each (X, Y)
        for ix, xv in enumerate(x_values):
            for iy, yv in enumerate(y_values):
                atoms_current = self._safe_copy(grid_xy[(ix, iy)])
                
                for iz, zv in enumerate(z_values):
                    # Skip z=z0 (already computed)
                    if iz == 0:
                        continue
                    
                    coord = [xv, yv, zv]
                    self._current_index += 1
                    self._print_progress(self._current_index, self._total_combinations, coord)
                    atoms_current = self._apply_constraints(atoms_current, coord)
                    atoms_current = self._run_optimizer(atoms_current)
                    self._record_result(atoms_current, coord, results, coords_list, energies)

        return results, coords_list, energies

    def run_scan(self):
        """Main scan entry point."""
        scan_values = self._generate_scan_values()
        dim = len(scan_values)

        total = 1
        for values in scan_values:
            total *= len(values)
        self._total_combinations = total
        self._current_index = 0

        if dim == 1:
            results, coords_list, energies = self._scan_1d(scan_values)
        elif dim == 2:
            results, coords_list, energies = self._scan_2d(scan_values)
        elif dim == 3:
            results, coords_list, energies = self._scan_3d(scan_values)
        else:
            raise ValueError(f"Only 1D, 2D, 3D scans are supported, got {dim}D")

        base, _ = os.path.splitext(self.output)
        write_scan_xyz(base + "_scan_final.xyz", results, coords_list, energies)

    def run(self):
        """JobABC interface."""
        self.run_scan()
