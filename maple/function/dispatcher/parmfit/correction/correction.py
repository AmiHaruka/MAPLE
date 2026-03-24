import os
from copy import deepcopy
from dataclasses import dataclass, field
from math import degrees, isclose
from typing import Optional

import numpy as np
from ase import Atoms

from ...jobABC import JobABC
from ...optimization.algorithm import LBFGS, LBFGSParams
from ...scan.scan import Scan
from ..utils.mSeminario import apply_mseminario
from ..utils.mmcalc import build_mm_topology_cache
from ..utils.outputparm import write_gromacs_files
from ..utils.parm import Angle, Bond, Dihedral, FourierTerm, Improper, Nonbond
from ..utils.readparm import CorrectionParameterSet, build_correction_parameter_set
from ..utils.torsionfit import (
    TorsionFitParams,
    apply_fitted_torsion,
    apply_global_delta,
    build_torsion_fit_params,
    build_global_torsion_problem,
    extract_global_delta,
    fit_torsion_scan,
    format_torsion_stage1_lines,
    format_torsion_stage2_lines,
    normalize_center_bond,
    read_scan_xyz,
    refine_torsion_scans_global,
    representative_dihedral_for_center_bond,
    resolve_torsion_center_bonds,
)

from maple.function.timer import timer


MEDIUM_THRESHOLDS = {
    "f_max_th": 0.00285,
    "f_rms_th": 0.00190,
    "dp_max_th": 0.00315,
    "dp_rms_th": 0.00210,
}
SECTION_WIDTH = 108
REPORT_FLOAT_TOL = 1.0e-10


@dataclass
class CorrectionParams:
    mol2: str = ""
    frcmod: str = ""
    vibrational_scaling: float = 1.0
    torsion: TorsionFitParams = field(default_factory=TorsionFitParams)

    def get(self, key: str, default=None):
        return getattr(self, key, default)

def to_f64(x):
    """Convert input to float64 numpy array or scalar."""
    if isinstance(x, np.ndarray):
        return x.astype(np.float64, copy=False)
    try:
        import torch
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
            return x.astype(np.float64, copy=False)
    except Exception:
        pass
    if np.isscalar(x):
        return float(x)
    return np.asarray(x, dtype=np.float64)

class _SilentLBFGS(LBFGS):
    """Local silent LBFGS runner used only by parmfit correction."""

    def __init__(self, atoms: Atoms, output: str, paras: Optional[dict] = None):
        JobABC.__init__(self, output)
        self.atoms = atoms
        self.params = self._init_params(LBFGSParams, paras, ("lbfgs", "LBFGS", "opt"))
        self.params.write_traj = False
        self.params.verbose = 0
        self.S: list[np.ndarray] = []
        self.Y: list[np.ndarray] = []
        self.rhos: list[float] = []
        self.converged = False

    def _update_metrics(self, step_cart: np.ndarray, forces: np.ndarray) -> None:
        atoms = self.atoms
        atoms.max_dp = float(np.max(np.abs(step_cart)))
        atoms.rms_dp = float(np.sqrt((step_cart ** 2).sum() / step_cart.size * 3.0))
        atoms.max_f = float(np.max(np.abs(forces)))
        atoms.rms_f = float(np.sqrt((forces ** 2).sum() / step_cart.size * 3.0))

    def run(self):
        atoms = self.atoms
        positions = np.asarray(atoms.get_positions(), dtype=float)
        forces = np.asarray(atoms.get_forces(), dtype=float)

        iteration = 0
        while iteration < self.params.max_iter:
            step_flat = self._two_loop(forces.reshape(-1))
            step = self._clip_step(step_flat.reshape(forces.shape))

            positions_old = positions.copy()
            forces_old = forces.copy()

            atoms.set_positions(positions + step)
            positions = np.asarray(atoms.get_positions(), dtype=float)
            forces = np.asarray(atoms.get_forces(), dtype=float)

            self._update_history((positions - positions_old).reshape(-1), (forces - forces_old).reshape(-1))
            self._update_metrics(step, forces)
            iteration += 1

            if (
                atoms.max_f <= atoms.f_max_th
                and atoms.rms_f <= atoms.f_rms_th
                and atoms.max_dp <= atoms.dp_max_th
                and atoms.rms_dp <= atoms.dp_rms_th
            ):
                self.converged = True
                return atoms

        self.converged = False
        return atoms


class Correction(JobABC):
    def __init__(self, output: str, atoms: Atoms, params: Optional[dict] = None):
        super().__init__(output)
        self.atoms = atoms
        self.raw_params = params if params is not None else {}
        self.params = self._init_params(CorrectionParams, self.raw_params, ("parmfit", "correction"))
        self.params.torsion = build_torsion_fit_params(self.raw_params)
        self.result: Optional[CorrectionParameterSet] = None
        self.stage0_result: Optional[CorrectionParameterSet] = None
        self.stage1_result: Optional[CorrectionParameterSet] = None
        self.torsion_fit_reports: list[dict] = []
        self.torsion_fit_warnings: list[str] = []
        self.torsion_refine_reports: list[dict] = []
        self.torsion_scan_data: dict[tuple[int, int], dict] = {}
        self.torsion_scan_xyz: dict[tuple[int, int], str] = {}
        self.gromacs_export: dict | None = None

    @property
    def torsion_stage0_result(self) -> Optional[CorrectionParameterSet]:
        return self.stage0_result

    @property
    def torsion_stage1_result(self) -> Optional[CorrectionParameterSet]:
        return self.stage1_result

    @property
    def torsion_fit_results(self) -> list[dict]:
        return self.torsion_fit_reports

    @property
    def torsion_refine_cycles(self) -> list[dict]:
        return self.torsion_refine_reports

    def run(self):
        with timer("Parmfit correction"):
            self.result = None
            self.stage0_result = None
            self.stage1_result = None
            self.torsion_fit_reports = []
            self.torsion_fit_warnings = []
            self.torsion_refine_reports = []
            self.torsion_scan_data = {}
            self.torsion_scan_xyz = {}
            self.gromacs_export = None

            original_result = self._build_parameter_set()
            report_base_result = deepcopy(original_result)
            self._validate_runtime_requirements()
            self._run_geometry_optimization()
            self._run_hessian_and_mseminario()
            self._run_stage1_torsion_fit()
            self._run_stage2_torsion_refine()
            self._export_gromacs()
            self._write_final_parameter_report(report_base_result)

    def _stage_lines(self, message: str) -> list[str]:
        return [f"{message}\n"]

    def _build_parameter_set(self) -> CorrectionParameterSet:
        self.result = build_correction_parameter_set(self.atoms, self.params.mol2, self.params.frcmod)
        self.log_info(self._summary_lines())
        return deepcopy(self.result)

    def _run_geometry_optimization(self) -> None:
        self.log_info(self._stage_lines("[Stage 0] Running silent LBFGS geometry optimization..."))
        optimizer = _SilentLBFGS(atoms=self.atoms, output=self.output, paras=self.raw_params)
        optimizer.run()
        if not optimizer.converged:
            raise RuntimeError(
                f"Silent LBFGS did not converge within {optimizer.params.max_iter} iterations."
            )
        self.log_info(
            self._stage_lines(
                f"          Silent LBFGS converged within {optimizer.params.max_iter} allowed iterations.",
            )
        )

    def _run_hessian_and_mseminario(self) -> None:
        self.log_info(self._stage_lines("          Evaluating Hessian and applying mSeminario bond/angle correction..."))
        assert self.result is not None
        hessian = self._get_hessian_cart()
        apply_mseminario(
            self.atoms,
            hessian,
            self.result.bonds,
            self.result.angles,
            self.params.vibrational_scaling,
        )
        self.stage0_result = deepcopy(self.result)
        self.log_info(
            self._stage_lines(
                f"          Updated {len(self.result.bonds)} bond terms and {len(self.result.angles)} angle terms.",
            )
        )

    def _run_stage1_torsion_fit(self) -> None:
        self.log_info(self._stage_lines("[Stage 1] Running stage-1 torsion scan fitting..."))
        self._run_torsion_fit_workflow(self.params.torsion)
        self.log_info(format_torsion_stage1_lines(self.params.torsion, self.torsion_fit_reports, self.torsion_fit_warnings))

    def _run_stage2_torsion_refine(self) -> None:
        self.log_info(self._stage_lines("\n[Stage 2] Running stage-2 global torsion refinement..."))
        if self.params.torsion.enabled and self.params.torsion.refine_rounds > 0 and self.torsion_fit_reports:
            self._run_torsion_refine_workflow(self.params.torsion)
        self.log_info(format_torsion_stage2_lines(self.params.torsion, self.torsion_fit_reports, self.torsion_refine_reports))

    def _export_gromacs(self) -> None:
        self.log_info(self._stage_lines("[Final] Exporting final GROMACS topology and coordinates..."))
        assert self.result is not None
        gromacs_top, gromacs_gro, gromacs_meta = write_gromacs_files(
            self.result,
            self.atoms,
            os.path.splitext(self.output)[0],
        )
        self.gromacs_export = {
            "top": gromacs_top,
            "gro": gromacs_gro,
            "warnings": list(gromacs_meta.get("warnings", [])),
            "omitted_counts": dict(gromacs_meta.get("omitted_counts", {})),
            "written_sections": list(gromacs_meta.get("written_sections", [])),
        }

    def _write_final_parameter_report(self, original_result: CorrectionParameterSet) -> None:
        assert self.result is not None
        self.log_info(self._stage_lines("        Writing final corrected parameter report..."))
        self.log_info(self._parameter_report_lines(original_result, self.result))

    def _validate_runtime_requirements(self) -> None:
        if getattr(self.atoms, "calc", None) is None:
            raise ValueError("Correction requires atoms.calc to be attached.")
        for method in ("get_forces", "get_potential_energy"):
            if not hasattr(self.atoms, method):
                raise ValueError(f"Correction requires atoms.{method}().")
        if not hasattr(self.atoms.calc, "get_hessian"):
            raise ValueError("Correction requires calculator.get_hessian(atoms).")
        missing = [key for key in MEDIUM_THRESHOLDS if getattr(self.atoms, key, None) is None]
        if missing:
            raise ValueError(
                "Correction requires LBFGS thresholds on atoms: " + ", ".join(missing)
            )

    def _get_hessian_cart(self) -> np.ndarray:
        """
        Get Cartesian Hessian (3N x 3N) from the calculator.

        Assumes atoms.calc implements get_hessian(atoms) and returns 3N x 3N
        or shape (1, 3N, 3N).
        """
        H = self.atoms.calc.get_hessian(self.atoms)
        H = to_f64(H)
        if H.ndim == 3 and H.shape[0] == 1:
            H = H[0]
        if H.ndim != 2 or H.shape[0] != H.shape[1]:
            raise ValueError(f"Hessian must be square 2D, got shape {H.shape}")
        return H

    def _summary(self) -> dict:
        assert self.result is not None
        return {
            "bonds": len(self.result.bonds),
            "angles": len(self.result.angles),
            "dihedrals": len(self.result.dihedrals),
            "impropers": len(self.result.impropers),
            "nonbonds": len(self.result.nonbonds),
            "unmatched_bonds": len(self.result.unmatched_bonds),
            "unmatched_angles": len(self.result.unmatched_angles),
            "unmatched_dihedrals": len(self.result.unmatched_dihedrals),
            "unmatched_impropers": len(self.result.unmatched_impropers),
            "unmatched_nonbonds": len(self.result.unmatched_nonbonds),
        }

    def _lbfgs_value(self, key: str, default):
        for alias in ("opt", "lbfgs", "LBFGS"):
            section = self.raw_params.get(alias)
            if isinstance(section, dict) and key in section:
                return section[key]
        return self.raw_params.get(key, default)

    def _scan_output_paths(self, center_bond: tuple[int, int]) -> tuple[str, str]:
        center = normalize_center_bond(center_bond)
        base, _ = os.path.splitext(self.output)
        prefix = f"{base}_torsionfit_{center[0]}-{center[1]}"
        return prefix + ".out", prefix + "_scan_final.xyz"

    def _copy_atoms_for_scan(self) -> Atoms:
        atoms = self.atoms.copy()
        atoms.calc = self.atoms.calc
        for key in MEDIUM_THRESHOLDS:
            setattr(atoms, key, getattr(self.atoms, key))
        return atoms

    def _run_center_bond_scan(
        self,
        center_bond: tuple[int, int],
        representative_dihedral: tuple[int, int, int, int],
        params: TorsionFitParams,
    ) -> str:
        scan_output, scan_xyz = self._scan_output_paths(center_bond)
        with open(scan_output, "w", encoding="utf-8"):
            pass

        scan_params = {
            "mode": "relaxed",
            "opt": {
                "max_iter": int(self._lbfgs_value("max_iter", 256)),
                "memory": int(self._lbfgs_value("memory", 5)),
                "curvature": float(self._lbfgs_value("curvature", 70.0)),
                "max_step": float(self._lbfgs_value("max_step", 0.2)),
                "write_traj": False,
                "verbose": 0,
            },
        }
        scan_atoms = self._copy_atoms_for_scan()
        constraint = [*representative_dihedral, params.scan_step_deg, params.scan_steps]
        scanner = Scan(
            output=scan_output,
            atoms=scan_atoms,
            method="lbfgs",
            constraints=[constraint],
            params=scan_params,
        )
        scanner.run()
        if not os.path.isfile(scan_xyz):
            raise FileNotFoundError(f"Torsion scan did not produce the expected xyz file: {scan_xyz}")
        return scan_xyz

    def _run_torsion_fit_workflow(self, params: TorsionFitParams) -> None:
        self.torsion_fit_reports = []
        self.torsion_fit_warnings = []
        self.torsion_scan_data = {}
        self.torsion_scan_xyz = {}
        self.stage1_result = None
        if not params.enabled:
            return
        assert self.result is not None
        assert self.stage0_result is not None

        stage0_result = deepcopy(self.stage0_result)
        stage0_cache = build_mm_topology_cache(stage0_result)
        center_bonds, self.torsion_fit_warnings = resolve_torsion_center_bonds(
            stage0_result,
            params,
            topology_cache=stage0_cache,
        )

        if not center_bonds:
            self.result = stage0_result
            self.stage1_result = deepcopy(stage0_result)
            return

        for center_bond in center_bonds:
            representative = representative_dihedral_for_center_bond(
                stage0_result,
                center_bond,
                topology_cache=stage0_cache,
            )
            scan_xyz = self._run_center_bond_scan(center_bond, representative.atoms, params)
            scan_data = read_scan_xyz(scan_xyz)
            fit_result = fit_torsion_scan(
                scan_data,
                stage0_result,
                center_bond,
                topology_cache=stage0_cache,
            )
            self.torsion_fit_reports.append(fit_result)
            self.torsion_scan_data[fit_result["center_bond"]] = scan_data
            self.torsion_scan_xyz[fit_result["center_bond"]] = scan_xyz

        stage1_result = deepcopy(stage0_result)
        for fit_result in self.torsion_fit_reports:
            stage1_result = apply_fitted_torsion(fit_result, stage1_result)

        self.stage1_result = deepcopy(stage1_result)
        self.result = stage1_result

    def _run_torsion_refine_workflow(self, params: TorsionFitParams) -> None:
        self.torsion_refine_reports = []
        assert self.result is not None
        assert self.stage0_result is not None
        assert self.stage1_result is not None

        if params.refine_rounds <= 0 or not self.torsion_fit_reports:
            return

        stage0_result = deepcopy(self.stage0_result)
        stage1_result = deepcopy(self.stage1_result)
        stage0_cache = build_mm_topology_cache(stage0_result)
        if params.refine_rounds > 0 and self.torsion_fit_reports:
            problem = build_global_torsion_problem(
                stage0_result,
                [result["center_bond"] for result in self.torsion_fit_reports],
                self.torsion_scan_data,
                topology_cache=stage0_cache,
            )
            delta_init = extract_global_delta(problem, stage1_result)
            delta_final, cycles = refine_torsion_scans_global(
                problem,
                delta_init=delta_init,
                max_sweeps=params.refine_rounds,
                max_block_iter=params.refine_max_iter,
                tol=params.refine_tol,
            )
            self.torsion_refine_reports = cycles
            self.result = apply_global_delta(problem, delta_final)

    def _summary_lines(self) -> list[str]:
        assert self.result is not None
        summary = self._summary()
        torsion = self.params.torsion
        return [
            "\n",
            "=" * 70 + "\n",
            "Parmfit Correction Summary".center(70) + "\n",
            "=" * 70 + "\n",
            f"mol2:              {self.params.mol2}\n",
            f"frcmod:            {self.params.frcmod}\n",
            f"Topology atoms:    {len(self.result.mol2.atoms)}\n",
            f"Topology bonds:    {len(self.result.mol2.bonds)}\n",
            f"torsionfit:        {'enabled' if torsion.enabled else 'disabled'}\n",
            f"Bond terms:        {summary['bonds']} (unmatched: {summary['unmatched_bonds']})\n",
            f"Angle terms:       {summary['angles']} (unmatched: {summary['unmatched_angles']})\n",
            f"Dihedral terms:    {summary['dihedrals']} (unmatched: {summary['unmatched_dihedrals']})\n",
            f"Improper terms:    {summary['impropers']} (unmatched: {summary['unmatched_impropers']})\n",
            f"Nonbond terms:     {summary['nonbonds']} (unmatched: {summary['unmatched_nonbonds']})\n",
            f"scan grid:         {torsion.scan_step_deg:.4f} deg x {torsion.scan_steps} steps\n",
            f"refinement:        {torsion.refine_rounds} rounds, max_iter={torsion.refine_max_iter}, tol={torsion.refine_tol:.6g}\n",
            (
                "center bonds:      auto-select non-ring center bonds with proper torsions\n"
                if torsion.center_bonds is None
                else f"center bonds:      {list(torsion.center_bonds)}\n"
            ),
            "=" * 70 + "\n",
        ]

    def _parameter_report_lines(
        self,
        original_result: CorrectionParameterSet,
        corrected_result: CorrectionParameterSet,
    ) -> list[str]:
        lines = [
            "\n",
            "=" * SECTION_WIDTH + "\n",
            "Parmfit Correction Parameter Report".center(SECTION_WIDTH) + "\n",
            "=" * SECTION_WIDTH + "\n",
            f"mol2:   {self.params.mol2}\n",
            f"frcmod: {self.params.frcmod}\n",
            "Display convention: left = original assigned parameter | right = corrected system parameter\n",
            "Only changed force-field items are shown below.\n",
            "\n",
        ]
        changed_sections = 0
        for section_lines in (
            self._bond_lines(original_result.bonds, corrected_result.bonds),
            self._angle_lines(original_result.angles, corrected_result.angles),
            self._dihedral_lines(original_result.dihedrals, corrected_result.dihedrals),
            self._improper_lines(original_result.impropers, corrected_result.impropers),
            self._nonbond_lines(original_result.nonbonds, corrected_result.nonbonds),
        ):
            if section_lines:
                changed_sections += 1
                lines.extend(section_lines)
        if changed_sections == 0:
            lines.append("No parameter changes were detected.\n")
        return lines

    def _section_header(self, title: str) -> list[str]:
        return [
            "\n",
            "-" * SECTION_WIDTH + "\n",
            title.center(SECTION_WIDTH) + "\n",
            "-" * SECTION_WIDTH + "\n",
        ]

    @staticmethod
    def _validate_paired_lengths(name: str, old_items: list, new_items: list) -> None:
        if len(old_items) != len(new_items):
            raise ValueError(
                f"Correction report cannot align {name}: {len(old_items)} original entries vs {len(new_items)} new entries."
            )

    @staticmethod
    def _format_atoms(atoms: tuple[int, ...]) -> str:
        return "(" + ",".join(str(atom) for atom in atoms) + ")"

    @staticmethod
    def _format_types(atom_types: tuple[str, ...]) -> str:
        return "-".join(atom_types)

    @staticmethod
    def _format_float(value: Optional[float], precision: int = 6) -> str:
        if value is None:
            return "NA"
        return f"{float(value):.{precision}f}"

    @staticmethod
    def _format_angle_deg(value: Optional[float]) -> str:
        if value is None:
            return "NA"
        return f"{degrees(float(value)):.4f}"

    def _format_bond_side(self, bond: Bond) -> str:
        return f"k={self._format_float(bond.kBond)}  r={self._format_float(bond.rEq)}"

    def _format_angle_side(self, angle: Angle) -> str:
        return f"k={self._format_float(angle.kTheta)}  theta={self._format_angle_deg(angle.thetaEq)}"

    def _format_term_side(self, term: Optional[FourierTerm]) -> str:
        if term is None:
            return "k=NA  n=NA  phase=NA"
        return (
            f"k={self._format_float(term.kPhi)}  "
            f"n={self._format_float(term.period, 3)}  "
            f"phase={self._format_angle_deg(term.phase)}"
        )

    def _format_nonbond_side(self, nonbond: Nonbond) -> str:
        return (
            f"q={self._format_float(nonbond.charge)}  "
            f"rmin/2={self._format_float(nonbond.rmin_half)}  "
            f"eps={self._format_float(nonbond.epsilon)}"
        )

    @staticmethod
    def _close_float(left: Optional[float], right: Optional[float], tol: float = REPORT_FLOAT_TOL) -> bool:
        if left is None and right is None:
            return True
        if left is None or right is None:
            return False
        return isclose(float(left), float(right), rel_tol=0.0, abs_tol=tol)

    def _bond_changed(self, old_bond: Bond, new_bond: Bond) -> bool:
        return not (
            self._close_float(old_bond.kBond, new_bond.kBond)
            and self._close_float(old_bond.rEq, new_bond.rEq)
        )

    def _angle_changed(self, old_angle: Angle, new_angle: Angle) -> bool:
        return not (
            self._close_float(old_angle.kTheta, new_angle.kTheta)
            and self._close_float(old_angle.thetaEq, new_angle.thetaEq)
        )

    def _term_changed(self, old_term: Optional[FourierTerm], new_term: Optional[FourierTerm]) -> bool:
        if old_term is None and new_term is None:
            return False
        if old_term is None or new_term is None:
            return True
        return not (
            self._close_float(old_term.kPhi, new_term.kPhi)
            and self._close_float(old_term.period, new_term.period)
            and self._close_float(old_term.phase, new_term.phase)
        )

    def _nonbond_changed(self, old_nonbond: Nonbond, new_nonbond: Nonbond) -> bool:
        return not (
            self._close_float(old_nonbond.charge, new_nonbond.charge)
            and self._close_float(old_nonbond.rmin_half, new_nonbond.rmin_half)
            and self._close_float(old_nonbond.epsilon, new_nonbond.epsilon)
        )

    def _bond_lines(self, old_bonds: list[Bond], new_bonds: list[Bond]) -> list[str]:
        self._validate_paired_lengths("bonds", old_bonds, new_bonds)
        rows: list[str] = []
        for old_bond, new_bond in zip(old_bonds, new_bonds):
            if not self._bond_changed(old_bond, new_bond):
                continue
            label = (
                f"{self._format_atoms(new_bond.atoms):<16} "
                f"{self._format_types(new_bond.atom_types):<18}"
            )
            rows.append(
                f"{label}{self._format_bond_side(old_bond):<34} | \t {self._format_bond_side(new_bond):<34}\n"
            )
        return self._section_header("BONDS") + rows if rows else []

    def _angle_lines(self, old_angles: list[Angle], new_angles: list[Angle]) -> list[str]:
        self._validate_paired_lengths("angles", old_angles, new_angles)
        rows: list[str] = []
        for old_angle, new_angle in zip(old_angles, new_angles):
            if not self._angle_changed(old_angle, new_angle):
                continue
            label = (
                f"{self._format_atoms(new_angle.atoms):<16} "
                f"{self._format_types(new_angle.atom_types):<24}"
            )
            rows.append(
                f"{label}{self._format_angle_side(old_angle):<36} | \t {self._format_angle_side(new_angle):<36}\n"
            )
        return self._section_header("ANGLES") + rows if rows else []

    def _dihedral_lines(self, old_dihedrals: list[Dihedral], new_dihedrals: list[Dihedral]) -> list[str]:
        self._validate_paired_lengths("dihedrals", old_dihedrals, new_dihedrals)
        rows: list[str] = []
        for old_dihedral, new_dihedral in zip(old_dihedrals, new_dihedrals):
            n_terms = max(len(old_dihedral.terms), len(new_dihedral.terms), 1)
            for term_index in range(n_terms):
                old_term = old_dihedral.terms[term_index] if term_index < len(old_dihedral.terms) else None
                new_term = new_dihedral.terms[term_index] if term_index < len(new_dihedral.terms) else None
                if not self._term_changed(old_term, new_term):
                    continue
                label = (
                    f"{self._format_atoms(new_dihedral.atoms):<18} "
                    f"{self._format_types(new_dihedral.atom_types):<26} "
                    f"term={term_index + 1:<2}"
                )
                rows.append(
                    f"{label}{self._format_term_side(old_term):<34} |  {self._format_term_side(new_term):<34}\n"
                )
        return self._section_header("DIHEDRALS") + rows if rows else []

    def _improper_lines(self, old_impropers: list[Improper], new_impropers: list[Improper]) -> list[str]:
        self._validate_paired_lengths("impropers", old_impropers, new_impropers)
        rows: list[str] = []
        for old_improper, new_improper in zip(old_impropers, new_impropers):
            n_terms = max(len(old_improper.terms), len(new_improper.terms), 1)
            for term_index in range(n_terms):
                old_term = old_improper.terms[term_index] if term_index < len(old_improper.terms) else None
                new_term = new_improper.terms[term_index] if term_index < len(new_improper.terms) else None
                if not self._term_changed(old_term, new_term):
                    continue
                label = (
                    f"{self._format_atoms(new_improper.atoms):<18} "
                    f"{self._format_types(new_improper.atom_types):<26} "
                    f"term={term_index + 1:<2}"
                )
                rows.append(
                    f"{label}{self._format_term_side(old_term):<34} |  {self._format_term_side(new_term):<34}\n"
                )
        return self._section_header("IMPROPERS") + rows if rows else []

    def _nonbond_lines(self, old_nonbonds: list[Nonbond], new_nonbonds: list[Nonbond]) -> list[str]:
        self._validate_paired_lengths("nonbonds", old_nonbonds, new_nonbonds)
        rows: list[str] = []
        for old_nonbond, new_nonbond in zip(old_nonbonds, new_nonbonds):
            if not self._nonbond_changed(old_nonbond, new_nonbond):
                continue
            label = f"atom={new_nonbond.atom:<4d} {new_nonbond.atom_type:<12}"
            rows.append(
                f"{label}{self._format_nonbond_side(old_nonbond):<42} |  {self._format_nonbond_side(new_nonbond):<42}\n"
            )
        return self._section_header("NONBONDS") + rows if rows else []
