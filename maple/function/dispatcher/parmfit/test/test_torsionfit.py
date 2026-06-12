import importlib
import os
from pathlib import Path
import math
import sys
import types

import numpy as np
import pytest

try:
    from ase import Atoms
except ModuleNotFoundError:
    ase_stub = types.ModuleType("ase")
    ase_stub.__path__ = []  # type: ignore[attr-defined]

    class Atoms:  # type: ignore[override]
        def __init__(self, symbols, positions):
            self.symbols = list(symbols)
            self.positions = np.asarray(positions, dtype=float)
            self.info = {}
            self.calc = None
            self.constraint = None

        def __len__(self):
            return len(self.symbols)

        def copy(self):
            atoms = Atoms(self.symbols[:], self.positions.copy())
            atoms.info = dict(self.info)
            atoms.calc = self.calc
            atoms.constraint = self.constraint
            return atoms

        def get_positions(self):
            return np.asarray(self.positions, dtype=float)

        def set_positions(self, positions):
            self.positions = np.asarray(positions, dtype=float)

        def get_chemical_symbols(self):
            return self.symbols[:]

        def set_constraint(self, constraint):
            self.constraint = constraint

    ase_stub.Atoms = Atoms
    sys.modules["ase"] = ase_stub


class _StubFixInternals:
    def __init__(self, bonds=None, angles_deg=None, dihedrals_deg=None):
        self.bonds = bonds
        self.angles_deg = angles_deg
        self.dihedrals_deg = dihedrals_deg


class _StubNeighborList:
    def __init__(self, cutoffs, self_interaction=False, bothways=True):
        del cutoffs, self_interaction, bothways
        self.atoms = None

    def update(self, atoms):
        self.atoms = atoms

    def get_neighbors(self, index):
        del index
        return np.asarray([], dtype=int), np.asarray([], dtype=int)


def _stub_natural_cutoffs(atoms):
    return [1.0] * len(atoms)


ase_module = sys.modules.get("ase")
if ase_module is not None and not hasattr(ase_module, "__path__"):
    ase_module.__path__ = []  # type: ignore[attr-defined]

if "ase.constraints" not in sys.modules:
    ase_constraints_stub = types.ModuleType("ase.constraints")
    ase_constraints_stub.FixInternals = _StubFixInternals
    sys.modules["ase.constraints"] = ase_constraints_stub
    if ase_module is not None:
        ase_module.constraints = ase_constraints_stub

if "ase.neighborlist" not in sys.modules:
    ase_neighborlist_stub = types.ModuleType("ase.neighborlist")
    ase_neighborlist_stub.NeighborList = _StubNeighborList
    ase_neighborlist_stub.natural_cutoffs = _stub_natural_cutoffs
    sys.modules["ase.neighborlist"] = ase_neighborlist_stub
    if ase_module is not None:
        ase_module.neighborlist = ase_neighborlist_stub

if not hasattr(Atoms, "get_forces"):
    def _get_forces(self):
        if getattr(self, "calc", None) is None:
            raise AttributeError("atoms.calc is required")
        return self.calc.get_forces(self)

    Atoms.get_forces = _get_forces  # type: ignore[attr-defined]

if not hasattr(Atoms, "get_potential_energy"):
    def _get_potential_energy(self, force_consistent=True):
        if getattr(self, "calc", None) is None:
            raise AttributeError("atoms.calc is required")
        return self.calc.get_potential_energy(self, force_consistent=force_consistent)

    Atoms.get_potential_energy = _get_potential_energy  # type: ignore[attr-defined]

if not hasattr(Atoms, "get_distance"):
    def _get_distance(self, i, j):
        positions = np.asarray(self.get_positions(), dtype=float)
        return float(np.linalg.norm(positions[j] - positions[i]))

    Atoms.get_distance = _get_distance  # type: ignore[attr-defined]

if not hasattr(Atoms, "get_angle"):
    def _get_angle(self, i, j, k):
        positions = np.asarray(self.get_positions(), dtype=float)
        v1 = positions[i] - positions[j]
        v2 = positions[k] - positions[j]
        cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cosine = float(np.clip(cosine, -1.0, 1.0))
        return float(np.degrees(np.arccos(cosine)))

    Atoms.get_angle = _get_angle  # type: ignore[attr-defined]

if not hasattr(Atoms, "get_dihedral"):
    def _get_dihedral(self, i, j, k, l):
        positions = np.asarray(self.get_positions(), dtype=float)
        p0, p1, p2, p3 = positions[[i, j, k, l]]
        b0 = -(p1 - p0)
        b1 = p2 - p1
        b2 = p3 - p2
        b1 /= np.linalg.norm(b1)
        v = b0 - np.dot(b0, b1) * b1
        w = b2 - np.dot(b2, b1) * b1
        x_val = np.dot(v, w)
        y_val = np.dot(np.cross(b1, v), w)
        return float(np.degrees(np.arctan2(y_val, x_val)))

    Atoms.get_dihedral = _get_dihedral  # type: ignore[attr-defined]

if not hasattr(Atoms, "set_distance"):
    def _set_distance(self, i, j, value, fix=0, mask=None):
        del i, j, value, fix, mask

    Atoms.set_distance = _set_distance  # type: ignore[attr-defined]

if not hasattr(Atoms, "set_angle"):
    def _set_angle(self, i, j, k, value, mask=None):
        del i, j, k, value, mask

    Atoms.set_angle = _set_angle  # type: ignore[attr-defined]

if not hasattr(Atoms, "set_dihedral"):
    def _set_dihedral(self, i, j, k, l, value, mask=None):
        del i, j, k, l, value, mask

    Atoms.set_dihedral = _set_dihedral  # type: ignore[attr-defined]

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maple.function.dispatcher.parmfit.utils.TorsionFit import (
    TorsionFitParams,
    TorsionScanRuntime,
    build_torsion_fit_params,
    format_torsion_fit_report,
    format_torsion_stage1_lines,
    format_torsion_stage2_lines,
    normalize_center_bond,
    read_scan_xyz,
    run_torsion_workflow,
)
from maple.function.dispatcher.parmfit.utils.TorsionFit.topology import (
    apply_center_bond_terms,
    apply_fitted_torsion,
    center_bond_group_atoms,
    enumerate_fittable_center_bonds,
    is_ring_center_bond,
    representative_dihedral_for_center_bond,
    resolve_torsion_center_bonds,
)
from maple.function.dispatcher.parmfit.utils.TorsionFit.fit import fit_torsion_scan
from maple.function.dispatcher.parmfit.utils.TorsionFit import ensemble as torsion_ensemble_module
from maple.function.dispatcher.parmfit.utils.TorsionFit.basis import _group_center_bond_dihedrals, build_global_torsion_problem
from maple.function.dispatcher.parmfit.utils.TorsionFit.stage2 import (
    evaluate_global_refit_objective,
    refine_torsion_scans_global,
)
from maple.function.dispatcher.parmfit.utils.TorsionFit.records import (
    TorsionFitReport,
    TorsionEnsembleResult,
    TorsionRefineCycle,
    TorsionScanData,
    TorsionWorkflowResult,
)
from maple.function.dispatcher.parmfit.utils.TorsionFit.scanio import HARTREE_TO_KCAL_MOL, read_scan_xyz as read_scan_xyz_from_scanio
from maple.function.dispatcher.parmfit.utils import runtime as runtime_module
from maple.function.dispatcher.parmfit.utils.Scan import read_scan_final_atoms, run_silent_scan
from maple.function.dispatcher.parmfit.utils.Scan.engine import SilentScanEngine
from maple.function.dispatcher.parmfit.utils.Scan.models import SilentScanResult
from maple.function.dispatcher.parmfit.utils.mechanics import build_mm_topology_cache, evaluate_mm_energy
from maple.function.dispatcher.parmfit.utils.readparm import (
    CorrectionParameterSet,
    Dihedral,
    FourierTerm,
    FrcmodDB,
    Mol2Atom,
    Mol2Bond,
    Mol2Topology,
    Nonbond,
)


def _make_atoms(symbols, positions):
    return Atoms(symbols=symbols, positions=positions)


def _make_parameter_set(atom_types, bond_graph, *, dihedrals=None, nonbonds=None, bond_types=None):
    nonbond_list = list(nonbonds or [])
    charges = {entry.atom: entry.charge for entry in nonbond_list}
    mol2_atoms = [
        Mol2Atom(atom_id=index, name=f"A{index}", atom_type=atom_type, charge=charges.get(index, 0.0))
        for index, atom_type in enumerate(atom_types, start=1)
    ]
    type_by_bond = {
        normalize_center_bond(tuple(bond)): str(bond_type)
        for bond, bond_type in (bond_types or {}).items()
    }
    mol2_bonds = [
        Mol2Bond(bond_id=index, atom1=i, atom2=j, bond_type=type_by_bond.get(normalize_center_bond((i, j)), "1"))
        for index, (i, j) in enumerate(bond_graph, start=1)
    ]
    adjacency = {index: set() for index in range(1, len(atom_types) + 1)}
    for i, j in bond_graph:
        adjacency[i].add(j)
        adjacency[j].add(i)
    mol2 = Mol2Topology(
        atoms=mol2_atoms,
        bonds=mol2_bonds,
        id_to_index={index: index for index in range(1, len(atom_types) + 1)},
        adjacency=adjacency,
    )
    return CorrectionParameterSet(
        mol2=mol2,
        frcmod=FrcmodDB(),
        bonds=[],
        angles=[],
        dihedrals=list(dihedrals or []),
        impropers=[],
        nonbonds=list(nonbond_list),
        unmatched_bonds=[],
        unmatched_angles=[],
        unmatched_dihedrals=[],
        unmatched_impropers=[],
        unmatched_nonbonds=[],
    )


def _fitted_mm_profile_rmse_for_test(
    scan_data: TorsionScanData,
    parameter_set: CorrectionParameterSet,
    center_bond: tuple[int, int],
    fitted_terms: list[list[FourierTerm]],
) -> tuple[float, np.ndarray, np.ndarray]:
    updated_parameter_set = apply_center_bond_terms(parameter_set, center_bond, fitted_terms)
    cache = build_mm_topology_cache(updated_parameter_set)
    qm_rel = np.asarray(scan_data.qm_rel, dtype=float)
    mm_refit_total = np.asarray(
        [evaluate_mm_energy(atoms, updated_parameter_set, topology_cache=cache).total for atoms in scan_data.frames],
        dtype=float,
    )
    mm_refit_rel = mm_refit_total - mm_refit_total[int(scan_data.ref_idx)]
    residual_after = qm_rel - mm_refit_rel
    rmse = float(np.sqrt(np.mean(residual_after**2))) if residual_after.size else 0.0
    return rmse, mm_refit_rel, residual_after


def _four_atom_frame(phi_deg: float) -> Atoms:
    rad = math.radians(phi_deg)
    positions = [
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        (math.cos(rad), math.sin(rad), 1.0),
    ]
    return _make_atoms(["C", "C", "C", "C"], positions)


def _five_atom_frame(phi_deg: float) -> Atoms:
    rad = math.radians(phi_deg)
    positions = [
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        (math.cos(rad), math.sin(rad), 1.0),
        (0.0, 1.0, 0.0),
    ]
    return _make_atoms(["C", "C", "C", "C", "C"], positions)


def test_ensemble_masked_rotor_keeps_complete_mobile_r_group_side():
    parameter_set = _make_parameter_set(
        atom_types=["c", "c", "c", "c"],
        bond_graph=[(1, 2), (2, 3), (3, 4)],
    )

    rotor = torsion_ensemble_module._center_bond_rotor(
        parameter_set,
        (2, 3),
        mobile_atoms={3, 4},
    )

    assert rotor == (3, 3, 2, 3)

    atoms = _make_atoms(
        ["C", "C", "C", "C"],
        [
            (-1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
        ],
    )
    candidate = torsion_ensemble_module._apply_trial_offsets(
        atoms,
        parameter_set,
        {rotor: 90.0},
        mobile_atoms={3, 4},
    )

    before = np.asarray(atoms.get_positions(), dtype=float)
    after = np.asarray(candidate.get_positions(), dtype=float)
    assert np.allclose(after[[0, 1]], before[[0, 1]])
    assert np.allclose(after[2], before[2])
    assert not np.allclose(after[3], before[3])


def test_ensemble_masked_rotor_rejects_partial_mobile_fragment():
    parameter_set = _make_parameter_set(
        atom_types=["c", "c", "c", "c"],
        bond_graph=[(1, 2), (2, 3), (3, 4)],
    )

    rotor = torsion_ensemble_module._center_bond_rotor(
        parameter_set,
        (2, 3),
        mobile_atoms={3},
    )

    assert rotor is None


def test_ensemble_rotor_skips_terminal_h_but_keeps_terminal_chlorine():
    h_parameter_set = _make_parameter_set(
        atom_types=["c", "h"],
        bond_graph=[(1, 2)],
    )
    cl_parameter_set = _make_parameter_set(
        atom_types=["c", "cl"],
        bond_graph=[(1, 2)],
    )

    assert torsion_ensemble_module._center_bond_rotor(h_parameter_set, (1, 2), mobile_atoms=None) is None
    assert torsion_ensemble_module._center_bond_rotor(cl_parameter_set, (1, 2), mobile_atoms=None) is not None


def _shift_positions(atoms: Atoms, dx: float) -> np.ndarray:
    positions = np.asarray(atoms.get_positions(), dtype=float).copy()
    positions[:, 0] += dx
    return positions


def _eight_atom_two_fragment_frame(phi_a_deg: float, phi_b_deg: float) -> Atoms:
    frag_a = _four_atom_frame(phi_a_deg)
    frag_b = _four_atom_frame(phi_b_deg)
    positions = np.vstack([frag_a.get_positions(), _shift_positions(frag_b, 10.0)])
    return _make_atoms(["C"] * 8, positions)


def _write_scan_xyz(path: Path, frames):
    with path.open("w", encoding="utf-8") as handle:
        for index, (angle_deg, energy_hartree, atoms) in enumerate(frames, start=1):
            positions = atoms.get_positions()
            handle.write(f"{len(atoms)}\n")
            handle.write(
                f"Scanning combination {index}/{len(frames)}: [{angle_deg:.4f}]  Energy = {energy_hartree:.10f}\n"
            )
            for symbol, (x, y, z) in zip(atoms.get_chemical_symbols(), positions):
                handle.write(f"{symbol:2s} {x: .10f} {y: .10f} {z: .10f}\n")


def _read_scan_comments(path: Path) -> list[str]:
    comments: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        while True:
            natoms_line = handle.readline()
            if natoms_line == "":
                break
            if not natoms_line.strip():
                continue
            natoms = int(natoms_line.strip())
            comments.append(handle.readline().strip())
            for _ in range(natoms):
                handle.readline()
    return comments


def _fake_center_bond_scan(output: str, center_bond: tuple[int, int], representative_dihedral: tuple[int, int, int, int]) -> str:
    del representative_dihedral
    prefix = Path(output)
    center = normalize_center_bond(center_bond)
    xyz_path = prefix.parent / f"{prefix.stem}_work/torsionfit/{prefix.stem}_torsionfit_{center[0]}-{center[1]}_scan_final.xyz"
    xyz_path.parent.mkdir(parents=True, exist_ok=True)
    if center == (2, 3) and prefix.stem not in {"shared_stage2", "filtered"}:
        phis = [0.0, 60.0, 120.0, 180.0, -120.0, -60.0]
        frames = []
        for phi in phis:
            atoms = _four_atom_frame(phi)
            energy_kcal = 1.75 * (1.0 + math.cos(math.radians(phi)))
            frames.append((phi, energy_kcal / HARTREE_TO_KCAL_MOL, atoms))
    elif center == (2, 3):
        phis = [0.0, 90.0, 180.0, -90.0]
        frames = []
        for phi in phis:
            atoms = _eight_atom_two_fragment_frame(phi, 0.0)
            energy_kcal = 1.40 * (1.0 + math.cos(math.radians(phi)))
            frames.append((phi, energy_kcal / HARTREE_TO_KCAL_MOL, atoms))
    else:
        phis = [0.0, 60.0, 120.0, 180.0, -120.0, -60.0]
        frames = []
        for phi in phis:
            atoms = _eight_atom_two_fragment_frame(0.0, phi)
            energy_kcal = 0.90 * (1.0 + math.cos(math.radians(phi)))
            frames.append((phi, energy_kcal / HARTREE_TO_KCAL_MOL, atoms))
    _write_scan_xyz(xyz_path, frames)
    return str(xyz_path)


class FakeScan:
    def __init__(self, output, atoms, method="lbfgs", constraints=None, params=None):
        self.output = output
        self.atoms = atoms
        self.method = method
        self.constraints = constraints or []
        self.params = params or {}

    def run(self):
        base, _ = os.path.splitext(self.output)
        xyz_path = Path(base + "_scan_final.xyz")
        dihedral = tuple(self.constraints[0][:4])
        if len(self.atoms) == 4:
            phis = [0.0, 60.0, 120.0, 180.0, -120.0, -60.0]
            frames = []
            for phi in phis:
                atoms = _four_atom_frame(phi)
                energy_kcal = 1.75 * (1.0 + math.cos(math.radians(phi)))
                frames.append((phi, energy_kcal / HARTREE_TO_KCAL_MOL, atoms))
        elif dihedral[1:3] == (2, 3):
            phis = [0.0, 90.0, 180.0, -90.0]
            frames = []
            for phi in phis:
                atoms = _eight_atom_two_fragment_frame(phi, 0.0)
                energy_kcal = 1.40 * (1.0 + math.cos(math.radians(phi)))
                frames.append((phi, energy_kcal / HARTREE_TO_KCAL_MOL, atoms))
        else:
            phis = [0.0, 60.0, 120.0, 180.0, -120.0, -60.0]
            frames = []
            for phi in phis:
                atoms = _eight_atom_two_fragment_frame(0.0, phi)
                energy_kcal = 0.90 * (1.0 + math.cos(math.radians(phi)))
                frames.append((phi, energy_kcal / HARTREE_TO_KCAL_MOL, atoms))
        _write_scan_xyz(xyz_path, frames)


def test_read_scan_xyz_parses_single_coordinate_and_energy(tmp_path: Path):
    path = tmp_path / "scan.xyz"
    atoms = _four_atom_frame(60.0)
    _write_scan_xyz(path, [(60.0, 1.5, atoms), (120.0, 2.0, atoms)])

    data = read_scan_xyz_from_scanio(str(path))

    assert isinstance(data, TorsionScanData)
    assert len(data.frames) == 2
    assert data.angles_deg.tolist() == pytest.approx([60.0, 120.0])
    assert data.qm_hartree[0] == pytest.approx(1.5)
    assert data.qm_kcal[0] == pytest.approx(1.5 * HARTREE_TO_KCAL_MOL)


def test_torsionfit_public_api_exposes_param_helpers():
    params = build_torsion_fit_params(
        {
            "torsionfit": True,
            "torsion_steps": 12,
            "torsion_bonds": [(3, 2)],
        }
    )

    assert isinstance(params, TorsionFitParams)
    assert params.enabled
    assert params.torsion_step_deg == pytest.approx(30.0)
    assert params.torsion_steps == 12
    assert params.center_bonds == ((2, 3),)
    assert normalize_center_bond((4, 1)) == (1, 4)


def test_torsionfit_params_use_dataclass_defaults_when_keys_are_missing():
    params = build_torsion_fit_params({})

    assert isinstance(params, TorsionFitParams)
    assert params.enabled
    assert params.torsion_steps == 36
    assert params.torsion_step_deg == pytest.approx(10.0)
    assert params.backend == "cgbs"
    assert params.constraint_mode == "projected"
    assert params.stage1_weights is True
    assert params.torsion_ensemble is False
    assert params.torsion_ensemble_ratio == pytest.approx(0.3)
    assert params.torsion_ensemble_weight == pytest.approx(0.5)


def test_torsionfit_params_accept_backend_and_constraint_mode():
    params = build_torsion_fit_params(
        {
            "torsionfit": True,
            "backend": "cgws",
            "constraint_mode": "projected",
            "stage1_weights": "false",
        }
    )

    assert params.backend == "cgws"
    assert params.constraint_mode == "projected"
    assert params.stage1_weights is False


def test_torsionfit_params_accept_optional_ensemble_target_options():
    params = build_torsion_fit_params(
        {
            "torsion_ensemble": "true",
            "torsion_ensemble_size": 12,
            "torsion_ensemble_ratio": 0.35,
            "torsion_ensemble_weight": 0.35,
            "torsion_ensemble_seed": 7,
        }
    )

    assert params.torsion_ensemble is True
    assert params.torsion_ensemble_ratio == pytest.approx(0.35)
    assert params.torsion_ensemble_weight == pytest.approx(0.35)
    assert not hasattr(params, "torsion_ensemble_size")
    assert not hasattr(params, "torsion_ensemble_seed")


def test_torsionfit_params_use_report_debug_field_name():
    params = build_torsion_fit_params({"report_debug": "true"})

    assert params.report_debug is True


def test_torsionfit_params_reject_invalid_backend():
    with pytest.raises(ValueError, match="backend must be one of"):
        build_torsion_fit_params({"backend": "cg_ws"})


def test_torsionfit_params_reject_invalid_constraint_mode():
    with pytest.raises(ValueError, match="constraint_mode must be one of"):
        build_torsion_fit_params({"constraint_mode": "projection"})


def test_torsionfit_params_reject_non_positive_torsion_steps():
    with pytest.raises(ValueError, match="torsion_steps must be a positive integer"):
        build_torsion_fit_params({"torsion_steps": 0})


def test_torsionfit_params_ignore_legacy_scan_config_shapes():
    params = build_torsion_fit_params(
        {
            "torsion": {
                "torsion_steps": 36,
                "torsion_bonds": [(3, 2)],
            },
            "torsion_scan_step": 10.0,
            "torsion_scan_steps": 36,
            "scan_steps": 18,
            "scan_step_deg": 20.0,
            "center_bonds": [(4, 5)],
            "torsion_center_bonds": [(6, 7)],
        }
    )

    assert params.torsion_steps == 36
    assert params.torsion_step_deg == pytest.approx(10.0)
    assert params.center_bonds is None


def test_torsionfit_import_path_resolves_to_the_package() -> None:
    module = importlib.import_module("maple.function.dispatcher.parmfit.utils.TorsionFit")

    assert module.__file__ is not None
    assert module.__file__.endswith("__init__.py")
    assert module.run_torsion_workflow is run_torsion_workflow
    assert module.TorsionFitParams is TorsionFitParams
    assert module.read_scan_xyz is read_scan_xyz
    assert not hasattr(module, "fit_torsion_scan")


def test_torsionfit_root_exports_only_stable_workflow_surface() -> None:
    stage1 = "".join(format_torsion_stage1_lines(TorsionFitParams(enabled=False), []))
    stage2 = "".join(format_torsion_stage2_lines(TorsionFitParams(enabled=False), []))

    assert "disabled" in stage1
    assert "disabled" in stage2


def test_torsion_scan_runtime_to_scan_params_carries_backend_and_constraint_mode():
    runtime = TorsionScanRuntime(
        max_iter=5,
        memory=7,
        curvature=12.0,
        max_step=0.15,
        backend="cgbs",
        constraint_mode="projected",
    )

    params = runtime.to_scan_params()

    assert params["backend"] == "cgbs"
    assert params["constraint_mode"] == "projected"
    assert params["opt"]["max_iter"] == 5
    assert params["opt"]["max_step"] == pytest.approx(0.15)


def test_runtime_force_and_energy_helpers_raise_value_error_without_calculator():
    atoms = _four_atom_frame(0.0)

    with pytest.raises(ValueError, match="atoms.calc"):
        runtime_module.get_forces(atoms)
    with pytest.raises(ValueError, match="atoms.calc"):
        runtime_module.get_potential_energy(atoms)


def test_fit_torsion_scan_regularizes_single_term_delta(tmp_path: Path):
    k_true = 1.75
    k_orig = 0.50
    phis = [0.0, 60.0, 120.0, 180.0, -120.0, -60.0]
    frames = []
    for phi in phis:
        atoms = _four_atom_frame(phi)
        energy_kcal = k_true * (1.0 + math.cos(math.radians(phi)))
        frames.append((phi, energy_kcal / HARTREE_TO_KCAL_MOL, atoms))

    path = tmp_path / "single.xyz"
    _write_scan_xyz(path, frames)
    scan_data = read_scan_xyz(str(path))
    parameter_set = _make_parameter_set(
        atom_types=["c", "c", "c", "c"],
        bond_graph=[(1, 2), (2, 3), (3, 4)],
        dihedrals=[
            Dihedral(
                atoms=(1, 2, 3, 4),
                atom_types=("c", "c", "c", "c"),
                terms=[FourierTerm(kPhi=k_orig, period=1.0, phase=0.0)],
            )
        ],
        nonbonds=[Nonbond(atom=index, atom_type="c", charge=0.0) for index in range(1, 5)],
    )

    before_rmse, _, _ = _fitted_mm_profile_rmse_for_test(
        scan_data,
        parameter_set,
        center_bond=(2, 3),
        fitted_terms=[[FourierTerm(kPhi=k_orig, period=1.0, phase=0.0)]],
    )
    result = fit_torsion_scan(scan_data, parameter_set, center_bond=(2, 3))

    assert isinstance(result, TorsionFitReport)
    fitted_k = result.terms.fitted_terms[0][0].kPhi
    assert k_orig < fitted_k < k_true
    assert result.terms.delta_kphi[0] == pytest.approx(fitted_k - k_orig)
    assert result.metrics.rmse < before_rmse
    assert result.metrics.mae < before_rmse


def test_fit_torsion_scan_shares_typed_terms_within_center_bond(tmp_path: Path):
    k_shared = 1.0
    phis = [0.0, 60.0, 120.0, 180.0, -120.0, -60.0]
    frames = []
    for phi in phis:
        atoms = _five_atom_frame(phi)
        phi1 = math.radians(phi)
        phi2 = math.radians(phi - 90.0)
        energy_kcal = k_shared * (1.0 + math.cos(phi1)) + k_shared * (1.0 + math.cos(phi2))
        frames.append((phi, energy_kcal / HARTREE_TO_KCAL_MOL, atoms))

    path = tmp_path / "independent.xyz"
    _write_scan_xyz(path, frames)
    scan_data = read_scan_xyz(str(path))
    parameter_set = _make_parameter_set(
        atom_types=["c", "c", "c", "c", "c"],
        bond_graph=[(1, 2), (2, 3), (3, 4), (2, 5)],
        dihedrals=[
            Dihedral(
                atoms=(1, 2, 3, 4),
                atom_types=("c", "c", "c", "c"),
                terms=[FourierTerm(kPhi=0.25, period=1.0, phase=0.0)],
            ),
            Dihedral(
                atoms=(5, 2, 3, 4),
                atom_types=("c", "c", "c", "c"),
                terms=[FourierTerm(kPhi=0.25, period=1.0, phase=0.0)],
            ),
        ],
        nonbonds=[Nonbond(atom=index, atom_type="c", charge=0.0) for index in range(1, 6)],
    )

    before_rmse, _, _ = _fitted_mm_profile_rmse_for_test(
        scan_data,
        parameter_set,
        center_bond=(2, 3),
        fitted_terms=[[FourierTerm(kPhi=0.25, period=1.0, phase=0.0)], [FourierTerm(kPhi=0.25, period=1.0, phase=0.0)]],
    )
    result = fit_torsion_scan(scan_data, parameter_set, center_bond=(3, 2))

    assert isinstance(result, TorsionFitReport)
    assert len(result.target_dihedrals) == 2
    assert result.metrics.rmse < before_rmse
    assert len(result.curves.angles_deg) == len(phis)
    assert len(result.terms.shared_groups) == 1
    assert len(result.terms.shared_groups[0].instances) == 2
    assert result.terms.fitted_terms[0][0].kPhi == pytest.approx(result.terms.fitted_terms[1][0].kPhi)
    after_rmse, _, _ = _fitted_mm_profile_rmse_for_test(
        scan_data,
        parameter_set,
        center_bond=(2, 3),
        fitted_terms=result.terms.fitted_terms,
    )
    assert after_rmse == pytest.approx(result.metrics.rmse)

    updated = apply_fitted_torsion(result, parameter_set)
    cache = build_mm_topology_cache(updated)
    assert len(cache.proper_by_center_bond[(2, 3)]) == 2
    report = "".join(format_torsion_fit_report(result))
    assert "Center bond: (2, 3)" in report
    assert "effective_rank" in report


def test_torsion_shared_groups_use_one_hop_environment_without_splitting_hydrogens():
    parameter_set = _make_parameter_set(
        atom_types=["ca", "ca", "oh", "ho", "ca", "i", "ca", "ha", "hn"],
        bond_graph=[(1, 2), (2, 3), (3, 4), (5, 2), (7, 2), (1, 6), (5, 8), (7, 9)],
        dihedrals=[
            Dihedral(
                atoms=(1, 2, 3, 4),
                atom_types=("ca", "ca", "oh", "ho"),
                terms=[FourierTerm(kPhi=0.25, period=2.0, phase=np.pi)],
            ),
            Dihedral(
                atoms=(5, 2, 3, 4),
                atom_types=("ca", "ca", "oh", "ho"),
                terms=[FourierTerm(kPhi=0.25, period=2.0, phase=np.pi)],
            ),
            Dihedral(
                atoms=(7, 2, 3, 4),
                atom_types=("ca", "ca", "oh", "ho"),
                terms=[FourierTerm(kPhi=0.25, period=2.0, phase=np.pi)],
            ),
        ],
    )

    groups = _group_center_bond_dihedrals(parameter_set.dihedrals, parameter_set=parameter_set)

    assert len(groups) == 2
    assert sorted(len(group.instances) for group in groups) == [1, 2]
    assert any({instance[0] for instance in group.instances} == {5, 7} for group in groups)


def test_fit_torsion_scan_admits_all_canonical_periods_per_group(tmp_path: Path):
    k1 = 1.25
    k2 = 0.75
    k_orig = 0.10
    phis = [0.0, 60.0, 120.0, 180.0, -120.0, -60.0]
    frames = []
    for phi in phis:
        atoms = _five_atom_frame(phi)
        phi1 = math.radians(phi)
        phi2 = math.radians(phi - 90.0)
        energy_kcal = k1 * (1.0 + math.cos(phi1)) + k2 * (1.0 + math.cos(2.0 * phi2))
        frames.append((phi, energy_kcal / HARTREE_TO_KCAL_MOL, atoms))

    path = tmp_path / "mixed_terms.xyz"
    _write_scan_xyz(path, frames)
    scan_data = read_scan_xyz(str(path))
    parameter_set = _make_parameter_set(
        atom_types=["c", "c", "c", "c", "c"],
        bond_graph=[(1, 2), (2, 3), (3, 4), (2, 5)],
        dihedrals=[
            Dihedral(
                atoms=(1, 2, 3, 4),
                atom_types=("c", "c", "c", "c"),
                terms=[FourierTerm(kPhi=k_orig, period=1.0, phase=0.0)],
            ),
            Dihedral(
                atoms=(5, 2, 3, 4),
                atom_types=("c", "c", "c", "c"),
                terms=[FourierTerm(kPhi=k_orig, period=2.0, phase=0.0)],
            ),
        ],
        nonbonds=[Nonbond(atom=index, atom_type="c", charge=0.0) for index in range(1, 6)],
    )

    before_rmse, _, _ = _fitted_mm_profile_rmse_for_test(
        scan_data,
        parameter_set,
        center_bond=(2, 3),
        fitted_terms=[
            [FourierTerm(kPhi=k_orig, period=1.0, phase=0.0)],
            [FourierTerm(kPhi=k_orig, period=2.0, phase=0.0)],
        ],
    )
    result = fit_torsion_scan(scan_data, parameter_set, center_bond=(2, 3))

    assert result.metrics.rmse < before_rmse
    assert len(result.terms.shared_groups) == 1
    assert result.terms.shared_groups[0].active_slots[0] == "k1"
    existing_slots = {f"k{int(term.period)}" for term in result.terms.shared_groups[0].original_terms}
    new_slots = {slot for slot in result.terms.shared_groups[0].active_slots if slot not in existing_slots}
    assert new_slots == {"k3", "k4"}
    assert result.terms.shared_groups[0].frozen_non_template_slots == ()
    periods_first = sorted(int(term.period) for term in result.terms.fitted_terms[0])
    periods_second = sorted(int(term.period) for term in result.terms.fitted_terms[1])
    assert periods_first == [1, 2, 3, 4]
    assert periods_second == [1, 2, 3, 4]
    shared_template_periods = [int(term.period) for term in result.terms.shared_groups[0].fitted_terms]
    assert shared_template_periods == [1, 2, 3, 4]
    shared_existing_periods = {int(term.period) for term in result.terms.shared_groups[0].original_terms}
    shared_new_periods = {period for period in shared_template_periods if period not in shared_existing_periods}
    assert shared_new_periods == {3, 4}
    fitted_k1 = next(term.kPhi for term in result.terms.fitted_terms[0] if int(term.period) == 1)
    assert result.metrics.rmse < before_rmse
    assert k_orig < fitted_k1 < k1
    after_rmse, _, _ = _fitted_mm_profile_rmse_for_test(
        scan_data,
        parameter_set,
        center_bond=(2, 3),
        fitted_terms=result.terms.fitted_terms,
    )
    assert after_rmse == pytest.approx(result.metrics.rmse)


def test_fit_torsion_scan_keeps_low_gain_canonical_slots_active(tmp_path: Path):
    k_true = 1.35
    phis = [0.0, 60.0, 120.0, 180.0, -120.0, -60.0]
    frames = []
    for phi in phis:
        atoms = _four_atom_frame(phi)
        energy_kcal = k_true * (1.0 + math.cos(math.radians(phi)))
        frames.append((phi, energy_kcal / HARTREE_TO_KCAL_MOL, atoms))

    path = tmp_path / "no_extra_slot.xyz"
    _write_scan_xyz(path, frames)
    scan_data = read_scan_xyz(str(path))
    parameter_set = _make_parameter_set(
        atom_types=["c", "c", "c", "c"],
        bond_graph=[(1, 2), (2, 3), (3, 4)],
        dihedrals=[
            Dihedral(
                atoms=(1, 2, 3, 4),
                atom_types=("c", "c", "c", "c"),
                terms=[FourierTerm(kPhi=0.3, period=1.0, phase=0.0)],
            )
        ],
        nonbonds=[Nonbond(atom=index, atom_type="c", charge=0.0) for index in range(1, 5)],
    )

    result = fit_torsion_scan(scan_data, parameter_set, center_bond=(2, 3))

    periods = [int(term.period) for term in result.terms.fitted_terms[0]]
    assert periods == [1, 2, 3, 4]
    fitted_k1 = next(term.kPhi for term in result.terms.fitted_terms[0] if int(term.period) == 1)
    assert fitted_k1 > 0.3


def test_fit_torsion_scan_preserves_period6_as_frozen_noncanonical_term(tmp_path: Path):
    k_true = 1.10
    phis = [0.0, 60.0, 120.0, 180.0, -120.0, -60.0]
    frames = []
    for phi in phis:
        atoms = _four_atom_frame(phi)
        energy_kcal = k_true * (1.0 + math.cos(math.radians(phi)))
        frames.append((phi, energy_kcal / HARTREE_TO_KCAL_MOL, atoms))

    path = tmp_path / "frozen_non_template.xyz"
    _write_scan_xyz(path, frames)
    scan_data = read_scan_xyz(str(path))
    parameter_set = _make_parameter_set(
        atom_types=["c", "c", "c", "c"],
        bond_graph=[(1, 2), (2, 3), (3, 4)],
        dihedrals=[
            Dihedral(
                atoms=(1, 2, 3, 4),
                atom_types=("c", "c", "c", "c"),
                terms=[
                    FourierTerm(kPhi=0.20, period=1.0, phase=0.0),
                    FourierTerm(kPhi=0.35, period=6.0, phase=0.0),
                ],
            )
        ],
        nonbonds=[Nonbond(atom=index, atom_type="c", charge=0.0) for index in range(1, 5)],
    )

    result = fit_torsion_scan(scan_data, parameter_set, center_bond=(2, 3))

    periods = [int(term.period) for term in result.terms.fitted_terms[0]]
    assert 1 in periods
    assert 6 in periods
    assert next(term.kPhi for term in result.terms.fitted_terms[0] if int(term.period) == 6) == pytest.approx(0.35)
    assert len(result.terms.shared_groups) == 1
    assert "k6 phase=0" in result.terms.shared_groups[0].frozen_non_template_slots
    report = "".join(format_torsion_fit_report(result))
    assert "frozen_non_template: k6 phase=0" in report

    updated = apply_fitted_torsion(result, parameter_set)
    updated_periods = [int(term.period) for term in updated.dihedrals[0].terms]
    assert 1 in updated_periods
    assert 6 in updated_periods
    assert next(term.kPhi for term in updated.dihedrals[0].terms if int(term.period) == 6) == pytest.approx(0.35)


def test_center_bond_helpers_filter_ring_and_track_group_atoms():
    chain_parameter_set = _make_parameter_set(
        atom_types=["c", "c", "c", "c", "c"],
        bond_graph=[(1, 2), (2, 3), (3, 4), (2, 5)],
        dihedrals=[
            Dihedral(
                atoms=(1, 2, 3, 4),
                atom_types=("c", "c", "c", "c"),
                terms=[FourierTerm(kPhi=0.25, period=1.0, phase=0.0)],
            ),
            Dihedral(
                atoms=(5, 2, 3, 4),
                atom_types=("c", "c", "c", "c"),
                terms=[FourierTerm(kPhi=0.25, period=1.0, phase=0.0)],
            ),
        ],
        nonbonds=[Nonbond(atom=index, atom_type="c", charge=0.0) for index in range(1, 6)],
    )

    assert enumerate_fittable_center_bonds(chain_parameter_set) == [(2, 3)]
    assert resolve_torsion_center_bonds(chain_parameter_set, TorsionFitParams(enabled=True)) == ([(2, 3)], [])
    assert not is_ring_center_bond(chain_parameter_set, (2, 3))
    assert representative_dihedral_for_center_bond(chain_parameter_set, (3, 2)).atoms == (1, 2, 3, 4)
    assert center_bond_group_atoms(chain_parameter_set, (2, 3)) == (1, 2, 3, 4, 5)

    ring_parameter_set = _make_parameter_set(
        atom_types=["c", "c", "c", "c"],
        bond_graph=[(1, 2), (2, 3), (3, 4), (4, 1)],
        dihedrals=[
            Dihedral(
                atoms=(1, 2, 3, 4),
                atom_types=("c", "c", "c", "c"),
                terms=[FourierTerm(kPhi=0.25, period=1.0, phase=0.0)],
            )
        ],
        nonbonds=[Nonbond(atom=index, atom_type="c", charge=0.0) for index in range(1, 5)],
    )

    assert is_ring_center_bond(ring_parameter_set, (2, 3))
    assert enumerate_fittable_center_bonds(ring_parameter_set) == []
    assert enumerate_fittable_center_bonds(ring_parameter_set, include_ring=True) == [(2, 3)]


def _make_ch152_like_rotatable_parameter_set() -> CorrectionParameterSet:
    bond_graph = [
        (1, 2),
        (1, 3),
        (1, 4),
        (4, 5),
        (5, 6),
        (5, 7),
        (5, 8),
        (4, 9),
        (9, 10),
        (10, 11),
        (10, 12),
        (10, 13),
        (9, 14),
        (14, 15),
        (14, 16),
    ]
    dihedrals = [
        Dihedral((2, 1, 4, 5), ("c", "c", "c", "c"), [FourierTerm(0.25, 1.0, 0.0)]),
        Dihedral((1, 4, 5, 6), ("c", "c", "c", "c"), [FourierTerm(0.25, 1.0, 0.0)]),
        Dihedral((1, 4, 9, 10), ("c", "c", "c", "c"), [FourierTerm(0.25, 1.0, 0.0)]),
        Dihedral((4, 9, 10, 11), ("c", "c", "c", "c"), [FourierTerm(0.25, 1.0, 0.0)]),
        Dihedral((4, 9, 14, 15), ("c", "c", "c", "c"), [FourierTerm(0.25, 1.0, 0.0)]),
    ]
    return _make_parameter_set(
        atom_types=["c"] * 16,
        bond_graph=bond_graph,
        bond_types={(1, 4): "2", (9, 14): "2"},
        dihedrals=dihedrals,
        nonbonds=[Nonbond(atom=index, atom_type="c", charge=0.0) for index in range(1, 17)],
    )


def test_center_bond_helpers_filter_non_single_mol2_bonds():
    parameter_set = _make_ch152_like_rotatable_parameter_set()

    assert enumerate_fittable_center_bonds(parameter_set) == [(4, 5), (4, 9), (9, 10)]
    assert resolve_torsion_center_bonds(parameter_set, TorsionFitParams(enabled=True)) == (
        [(4, 5), (4, 9), (9, 10)],
        [],
    )


def test_explicit_non_rotatable_center_bond_is_rejected():
    parameter_set = _make_ch152_like_rotatable_parameter_set()

    with pytest.raises(ValueError, match=r"not rotatable.*bond_type=2"):
        resolve_torsion_center_bonds(
            parameter_set,
            TorsionFitParams(enabled=True, center_bonds=((1, 4),)),
        )


def test_stage1_local_initializers_are_order_independent_across_center_bonds(tmp_path: Path):
    k_true_a = 1.4
    k_true_b = 0.9
    k_orig = 0.2
    phis_a = [0.0, 90.0, 180.0, -90.0]
    phis_b = [0.0, 60.0, 120.0, 180.0, -120.0, -60.0]
    frames_a = []
    frames_b = []
    for phi in phis_a:
        atoms = _eight_atom_two_fragment_frame(phi, 0.0)
        energy_kcal = k_true_a * (1.0 + math.cos(math.radians(phi)))
        frames_a.append((phi, energy_kcal / HARTREE_TO_KCAL_MOL, atoms))
    for phi in phis_b:
        atoms = _eight_atom_two_fragment_frame(0.0, phi)
        energy_kcal = k_true_b * (1.0 + math.cos(math.radians(phi)))
        frames_b.append((phi, energy_kcal / HARTREE_TO_KCAL_MOL, atoms))

    path_a = tmp_path / "scan_a.xyz"
    path_b = tmp_path / "scan_b.xyz"
    _write_scan_xyz(path_a, frames_a)
    _write_scan_xyz(path_b, frames_b)
    scan_a = read_scan_xyz(str(path_a))
    scan_b = read_scan_xyz(str(path_b))
    parameter_set = _make_parameter_set(
        atom_types=["c"] * 8,
        bond_graph=[(1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8)],
        dihedrals=[
            Dihedral(atoms=(1, 2, 3, 4), atom_types=("c",) * 4, terms=[FourierTerm(kPhi=k_orig, period=1.0, phase=0.0)]),
            Dihedral(atoms=(5, 6, 7, 8), atom_types=("c",) * 4, terms=[FourierTerm(kPhi=k_orig, period=1.0, phase=0.0)]),
        ],
        nonbonds=[Nonbond(atom=index, atom_type="c", charge=0.0) for index in range(1, 9)],
    )

    fit_a = fit_torsion_scan(scan_a, parameter_set, center_bond=(2, 3))
    fit_b = fit_torsion_scan(scan_b, parameter_set, center_bond=(6, 7))

    forward = apply_fitted_torsion(fit_b, apply_fitted_torsion(fit_a, parameter_set))
    reverse = apply_fitted_torsion(fit_a, apply_fitted_torsion(fit_b, parameter_set))

    assert forward.dihedrals[0].terms[0].kPhi == pytest.approx(reverse.dihedrals[0].terms[0].kPhi)
    assert forward.dihedrals[1].terms[0].kPhi == pytest.approx(reverse.dihedrals[1].terms[0].kPhi)


def test_fit_scan_xyz_wrapper_matches_direct_fit(tmp_path: Path):
    k_true = 1.2
    phis = [0.0, 90.0, 180.0, -90.0]
    frames = []
    for phi in phis:
        atoms = _four_atom_frame(phi)
        energy_kcal = k_true * (1.0 + math.cos(math.radians(phi)))
        frames.append((phi, energy_kcal / HARTREE_TO_KCAL_MOL, atoms))

    path = tmp_path / "wrapper.xyz"
    _write_scan_xyz(path, frames)
    scan_data = read_scan_xyz(str(path))
    parameter_set = _make_parameter_set(
        atom_types=["c", "c", "c", "c"],
        bond_graph=[(1, 2), (2, 3), (3, 4)],
        dihedrals=[
            Dihedral(
                atoms=(1, 2, 3, 4),
                atom_types=("c", "c", "c", "c"),
                terms=[FourierTerm(kPhi=0.10, period=1.0, phase=0.0)],
            )
        ],
        nonbonds=[Nonbond(atom=index, atom_type="c", charge=0.0) for index in range(1, 5)],
    )

    direct = fit_torsion_scan(scan_data, parameter_set, center_bond=(2, 3))
    wrapped = fit_torsion_scan(read_scan_xyz_from_scanio(str(path)), parameter_set, center_bond=(2, 3))

    assert wrapped.terms.fitted_terms[0][0].kPhi == pytest.approx(direct.terms.fitted_terms[0][0].kPhi)
    assert wrapped.curves.mm_stage0_rel[0] == pytest.approx(direct.curves.mm_stage0_rel[0])


def test_local_fitted_mm_profile_helper_matches_true_curve(tmp_path: Path):
    k_true = 1.75
    phis = [0.0, 60.0, 120.0, 180.0, -120.0, -60.0]
    frames = []
    for phi in phis:
        atoms = _four_atom_frame(phi)
        energy_kcal = k_true * (1.0 + math.cos(math.radians(phi)))
        frames.append((phi, energy_kcal / HARTREE_TO_KCAL_MOL, atoms))

    path = tmp_path / "objective.xyz"
    _write_scan_xyz(path, frames)
    scan_data = read_scan_xyz(str(path))
    parameter_set = _make_parameter_set(
        atom_types=["c", "c", "c", "c"],
        bond_graph=[(1, 2), (2, 3), (3, 4)],
        dihedrals=[
            Dihedral(
                atoms=(1, 2, 3, 4),
                atom_types=("c", "c", "c", "c"),
                terms=[FourierTerm(kPhi=0.5, period=1.0, phase=0.0)],
            )
        ],
        nonbonds=[Nonbond(atom=index, atom_type="c", charge=0.0) for index in range(1, 5)],
    )

    fitted_terms = [[FourierTerm(kPhi=k_true, period=1.0, phase=0.0)]]
    rmse, mm_refit_rel, residual_after = _fitted_mm_profile_rmse_for_test(
        scan_data,
        parameter_set,
        center_bond=(2, 3),
        fitted_terms=fitted_terms,
    )

    qm_rel = np.asarray(scan_data.qm_rel, dtype=float)
    assert rmse == pytest.approx(0.0, abs=5.0e-8)
    assert mm_refit_rel == pytest.approx(qm_rel)
    assert residual_after == pytest.approx(np.zeros_like(qm_rel), abs=5.0e-8)


def test_refine_torsion_scans_global_improves_true_rmse_from_perturbed_start(tmp_path: Path):
    k_true = 1.75
    phis = [0.0, 60.0, 120.0, 180.0, -120.0, -60.0]
    frames = []
    for phi in phis:
        atoms = _four_atom_frame(phi)
        energy_kcal = k_true * (1.0 + math.cos(math.radians(phi)))
        frames.append((phi, energy_kcal / HARTREE_TO_KCAL_MOL, atoms))

    path = tmp_path / "refine.xyz"
    _write_scan_xyz(path, frames)
    scan_data = read_scan_xyz(str(path))
    parameter_set = _make_parameter_set(
        atom_types=["c", "c", "c", "c"],
        bond_graph=[(1, 2), (2, 3), (3, 4)],
        dihedrals=[
            Dihedral(
                atoms=(1, 2, 3, 4),
                atom_types=("c", "c", "c", "c"),
                terms=[FourierTerm(kPhi=0.25, period=1.0, phase=0.0)],
            )
        ],
        nonbonds=[Nonbond(atom=index, atom_type="c", charge=0.0) for index in range(1, 5)],
    )

    before_rmse, _, _ = _fitted_mm_profile_rmse_for_test(
        scan_data,
        parameter_set,
        center_bond=(2, 3),
        fitted_terms=[[FourierTerm(kPhi=0.25, period=1.0, phase=0.0)]],
    )
    problem = build_global_torsion_problem(parameter_set, [(2, 3)], {(2, 3): scan_data})
    before_eval = evaluate_global_refit_objective(problem, np.zeros(2 * len(problem.term_paths), dtype=float))
    vector_init = np.zeros(2 * len(problem.term_paths), dtype=float)
    vector_final, cycles = refine_torsion_scans_global(
        problem,
        delta_init=vector_init,
        enabled=True,
        max_block_iter=20,
        tol=1.0e-8,
    )
    after_eval = evaluate_global_refit_objective(problem, vector_final)

    assert cycles[-1].accepted_blocks == 1
    assert before_rmse > 0.0
    assert after_eval.global_rmse < before_eval.global_rmse


def test_refine_torsion_scans_global_keeps_already_optimal_terms(tmp_path: Path):
    k_true = 1.75
    phis = [0.0, 60.0, 120.0, 180.0, -120.0, -60.0]
    frames = []
    for phi in phis:
        atoms = _four_atom_frame(phi)
        energy_kcal = k_true * (1.0 + math.cos(math.radians(phi)))
        frames.append((phi, energy_kcal / HARTREE_TO_KCAL_MOL, atoms))

    path = tmp_path / "optimal.xyz"
    _write_scan_xyz(path, frames)
    scan_data = read_scan_xyz(str(path))
    parameter_set = _make_parameter_set(
        atom_types=["c", "c", "c", "c"],
        bond_graph=[(1, 2), (2, 3), (3, 4)],
        dihedrals=[
            Dihedral(
                atoms=(1, 2, 3, 4),
                atom_types=("c", "c", "c", "c"),
                terms=[FourierTerm(kPhi=k_true, period=1.0, phase=0.0)],
            )
        ],
        nonbonds=[Nonbond(atom=index, atom_type="c", charge=0.0) for index in range(1, 5)],
    )

    problem = build_global_torsion_problem(parameter_set, [(2, 3)], {(2, 3): scan_data})
    vector_init = np.zeros(2 * len(problem.term_paths), dtype=float)
    vector_final, cycles = refine_torsion_scans_global(
        problem,
        delta_init=vector_init,
        enabled=True,
        max_block_iter=5,
        tol=1.0e-8,
    )
    before_eval = evaluate_global_refit_objective(problem, vector_init)
    after_eval = evaluate_global_refit_objective(problem, vector_final)

    assert vector_final.tolist() == pytest.approx(vector_init.tolist())
    assert cycles[-1].accepted_blocks == 0
    assert before_eval.global_rmse == pytest.approx(0.0, abs=5.0e-8)
    assert after_eval.global_rmse == pytest.approx(0.0, abs=5.0e-8)


def test_run_torsion_workflow_reports_disabled_state():
    atoms = _four_atom_frame(0.0)
    parameter_set = _make_parameter_set(
        atom_types=["c", "c", "c", "c"],
        bond_graph=[(1, 2), (2, 3), (3, 4)],
        dihedrals=[
            Dihedral(
                atoms=(1, 2, 3, 4),
                atom_types=("c", "c", "c", "c"),
                terms=[FourierTerm(kPhi=0.5, period=1.0, phase=0.0)],
            )
        ],
        nonbonds=[Nonbond(atom=index, atom_type="c", charge=0.0) for index in range(1, 5)],
    )

    result = run_torsion_workflow(
        atoms=atoms,
        output="disabled.out",
        parameter_set=parameter_set,
        params=TorsionFitParams(enabled=False),
        runtime=TorsionScanRuntime(max_iter=5, memory=5, curvature=70.0, max_step=0.2),
    )

    assert result.stage1_parameter_set is None
    assert result.refine_cycles == []
    assert result.center_bonds == []
    assert result.final_parameter_set.dihedrals[0].terms[0].kPhi == pytest.approx(0.5)


def test_run_torsion_workflow_handles_no_eligible_center_bonds():
    atoms = _four_atom_frame(0.0)
    parameter_set = _make_parameter_set(
        atom_types=["c", "c", "c", "c"],
        bond_graph=[(1, 2), (2, 3), (3, 4), (4, 1)],
        dihedrals=[
            Dihedral(
                atoms=(1, 2, 3, 4),
                atom_types=("c", "c", "c", "c"),
                terms=[FourierTerm(kPhi=0.5, period=1.0, phase=0.0)],
            )
        ],
        nonbonds=[Nonbond(atom=index, atom_type="c", charge=0.0) for index in range(1, 5)],
    )

    result = run_torsion_workflow(
        atoms=atoms,
        output="ring.out",
        parameter_set=parameter_set,
        params=TorsionFitParams(enabled=True, refine_rounds=0),
        runtime=TorsionScanRuntime(max_iter=5, memory=5, curvature=70.0, max_step=0.2),
    )

    assert result.stage1_parameter_set is not None
    assert result.center_bonds == []
    assert result.refine_cycles == []
    assert result.final_parameter_set.dihedrals[0].terms[0].kPhi == pytest.approx(0.5)


def test_run_torsion_workflow_runs_stage1_with_internal_silent_scan(tmp_path: Path, monkeypatch):
    atoms = _four_atom_frame(0.0)
    parameter_set = _make_parameter_set(
        atom_types=["c", "c", "c", "c"],
        bond_graph=[(1, 2), (2, 3), (3, 4)],
        dihedrals=[
            Dihedral(
                atoms=(1, 2, 3, 4),
                atom_types=("c", "c", "c", "c"),
                terms=[FourierTerm(kPhi=0.5, period=1.0, phase=0.0)],
            )
        ],
        nonbonds=[Nonbond(atom=index, atom_type="c", charge=0.0) for index in range(1, 5)],
    )

    monkeypatch.setattr(
        "maple.function.dispatcher.parmfit.utils.TorsionFit.workflow._run_center_bond_scan",
        lambda atoms, output, params, runtime, center_bond, representative_dihedral: _fake_center_bond_scan(output, center_bond, representative_dihedral),
    )

    result = run_torsion_workflow(
        atoms=atoms,
        output=str(tmp_path / "shared_stage1.out"),
        parameter_set=parameter_set,
        params=TorsionFitParams(enabled=True, torsion_steps=5, refine_rounds=0),
        runtime=TorsionScanRuntime(max_iter=5, memory=5, curvature=70.0, max_step=0.2),
    )

    assert len(result.scan_xyz) == 1
    assert Path(next(iter(result.scan_xyz.values()))).is_file()
    assert next(iter(result.scan_xyz.values())).endswith("_work/torsionfit/shared_stage1_torsionfit_2-3_scan_final.xyz")
    assert result.final_parameter_set.dihedrals[0].terms[0].kPhi > 0.5


def test_run_torsion_workflow_routes_loss_mode(monkeypatch):
    atoms = _four_atom_frame(0.0)
    parameter_set = _make_parameter_set(
        atom_types=["c", "c", "c", "c"],
        bond_graph=[(1, 2), (2, 3), (3, 4)],
        dihedrals=[
            Dihedral(
                atoms=(1, 2, 3, 4),
                atom_types=("c", "c", "c", "c"),
                terms=[FourierTerm(kPhi=0.5, period=1.0, phase=0.0)],
            )
        ],
        nonbonds=[Nonbond(atom=index, atom_type="c", charge=0.0) for index in range(1, 5)],
    )
    scan_data = TorsionScanData(
        angles_deg=np.asarray([0.0, 60.0], dtype=float),
        qm_hartree=np.asarray([0.0, 0.0], dtype=float),
        qm_kcal=np.asarray([0.0, 0.0], dtype=float),
        frames=[_four_atom_frame(0.0), _four_atom_frame(60.0)],
        source_path="fake.xyz",
        ref_idx=0,
        qm_rel=np.asarray([0.0, 0.0], dtype=float),
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "maple.function.dispatcher.parmfit.utils.TorsionFit.workflow.resolve_torsion_center_bonds",
        lambda parameter_set, params, topology_cache=None: ([(2, 3)], []),
    )
    monkeypatch.setattr(
        "maple.function.dispatcher.parmfit.utils.TorsionFit.workflow.representative_dihedral_for_center_bond",
        lambda parameter_set, center_bond, topology_cache=None: types.SimpleNamespace(atoms=(1, 2, 3, 4)),
    )
    monkeypatch.setattr(
        "maple.function.dispatcher.parmfit.utils.TorsionFit.workflow._run_center_bond_scan",
        lambda *args, **kwargs: "fake.xyz",
    )
    monkeypatch.setattr("maple.function.dispatcher.parmfit.utils.TorsionFit.workflow.read_scan_xyz", lambda path: scan_data)

    def fake_run_loss_mode(
        *,
        base_parameter_set,
        original_parameter_set,
        center_bonds,
        scan_data_map,
        scan_xyz_map,
        scan_mm_orig_rel_map=None,
        params,
        topology_cache=None,
        log_info=None,
        ensemble_result=None,
    ):
        del topology_cache, original_parameter_set, scan_mm_orig_rel_map, log_info
        captured["center_bonds"] = tuple(center_bonds)
        captured["scan_keys"] = tuple(scan_data_map)
        captured["ensemble_result"] = ensemble_result
        return TorsionWorkflowResult(
            stage1_parameter_set=base_parameter_set,
            final_parameter_set=base_parameter_set,
            refine_cycles=[],
            scan_xyz=scan_xyz_map,
            center_bonds=list(center_bonds),
            warnings=[],
        )

    monkeypatch.setattr("maple.function.dispatcher.parmfit.utils.TorsionFit.workflow.run_loss_mode", fake_run_loss_mode)

    run_torsion_workflow(
        atoms=atoms,
        output="loss_route.out",
        parameter_set=parameter_set,
        params=TorsionFitParams(enabled=True, torsion_ensemble=False),
        runtime=TorsionScanRuntime(max_iter=5, memory=5, curvature=70.0, max_step=0.2),
    )

    assert captured["center_bonds"] == ((2, 3),)
    assert captured["scan_keys"] == ((2, 3),)
    assert captured["ensemble_result"] is None


def test_run_torsion_workflow_passes_optional_ensemble_provider(monkeypatch):
    atoms = _four_atom_frame(0.0)
    parameter_set = _make_parameter_set(
        atom_types=["c", "c", "c", "c"],
        bond_graph=[(1, 2), (2, 3), (3, 4)],
        dihedrals=[
            Dihedral(
                atoms=(1, 2, 3, 4),
                atom_types=("c", "c", "c", "c"),
                terms=[FourierTerm(kPhi=0.5, period=1.0, phase=0.0)],
            )
        ],
        nonbonds=[Nonbond(atom=index, atom_type="c", charge=0.0) for index in range(1, 5)],
    )
    scan_data = TorsionScanData(
        angles_deg=np.asarray([0.0, 60.0], dtype=float),
        qm_hartree=np.asarray([0.0, 0.0], dtype=float),
        qm_kcal=np.asarray([0.0, 0.0], dtype=float),
        frames=[_four_atom_frame(0.0), _four_atom_frame(60.0)],
        source_path="fake.xyz",
        ref_idx=0,
        qm_rel=np.asarray([0.0, 0.0], dtype=float),
    )
    ensemble = TorsionEnsembleResult(xyz_paths={(2, 3): "ensemble.xyz"})
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "maple.function.dispatcher.parmfit.utils.TorsionFit.workflow.resolve_torsion_center_bonds",
        lambda parameter_set, params, topology_cache=None: ([(2, 3)], []),
    )
    monkeypatch.setattr(
        "maple.function.dispatcher.parmfit.utils.TorsionFit.workflow.representative_dihedral_for_center_bond",
        lambda parameter_set, center_bond, topology_cache=None: types.SimpleNamespace(atoms=(1, 2, 3, 4)),
    )
    monkeypatch.setattr(
        "maple.function.dispatcher.parmfit.utils.TorsionFit.workflow._run_center_bond_scan",
        lambda *args, **kwargs: "fake.xyz",
    )
    monkeypatch.setattr("maple.function.dispatcher.parmfit.utils.TorsionFit.workflow.read_scan_xyz", lambda path: scan_data)
    monkeypatch.setattr(
        "maple.function.dispatcher.parmfit.utils.TorsionFit.workflow.build_torsion_local_ensemble",
        lambda **kwargs: ensemble,
    )

    def fake_run_loss_mode(**kwargs):
        captured.update(kwargs)
        return TorsionWorkflowResult(
            stage1_parameter_set=kwargs["base_parameter_set"],
            final_parameter_set=kwargs["base_parameter_set"],
            scan_xyz=kwargs["scan_xyz_map"],
        )

    monkeypatch.setattr("maple.function.dispatcher.parmfit.utils.TorsionFit.workflow.run_loss_mode", fake_run_loss_mode)

    result = run_torsion_workflow(
        atoms=atoms,
        output="ensemble_route.out",
        parameter_set=parameter_set,
        params=TorsionFitParams(enabled=True, torsion_ensemble=True),
        runtime=TorsionScanRuntime(max_iter=5, memory=5, curvature=70.0, max_step=0.2),
    )

    assert captured["ensemble_result"] is ensemble
    assert result.ensemble_xyz == {(2, 3): "ensemble.xyz"}


def test_run_torsion_workflow_does_not_scan_non_rotatable_mol2_bonds(monkeypatch):
    atoms = _make_atoms(["C"] * 16, [(float(index), 0.0, 0.0) for index in range(16)])
    parameter_set = _make_ch152_like_rotatable_parameter_set()
    scan_data = TorsionScanData(
        angles_deg=np.asarray([0.0], dtype=float),
        qm_hartree=np.asarray([0.0], dtype=float),
        qm_kcal=np.asarray([0.0], dtype=float),
        frames=[atoms],
        source_path="fake.xyz",
        ref_idx=0,
        qm_rel=np.asarray([0.0], dtype=float),
    )
    scanned: list[tuple[int, int]] = []
    captured: dict[str, object] = {}

    def fake_scan(atoms, output, params, runtime, center_bond, representative_dihedral):
        del atoms, output, params, runtime, representative_dihedral
        center = normalize_center_bond(center_bond)
        scanned.append(center)
        return f"{center[0]}-{center[1]}.xyz"

    def fake_run_loss_mode(
        *,
        base_parameter_set,
        original_parameter_set,
        center_bonds,
        scan_data_map,
        scan_xyz_map,
        scan_mm_orig_rel_map=None,
        params,
        topology_cache=None,
        log_info=None,
        ensemble_result=None,
    ):
        del original_parameter_set, scan_mm_orig_rel_map, params, topology_cache, log_info, ensemble_result
        captured["center_bonds"] = tuple(center_bonds)
        captured["scan_keys"] = tuple(scan_data_map)
        return TorsionWorkflowResult(
            stage1_parameter_set=base_parameter_set,
            final_parameter_set=base_parameter_set,
            refine_cycles=[],
            scan_xyz=scan_xyz_map,
            center_bonds=list(center_bonds),
            warnings=[],
        )

    monkeypatch.setattr("maple.function.dispatcher.parmfit.utils.TorsionFit.workflow._run_center_bond_scan", fake_scan)
    monkeypatch.setattr("maple.function.dispatcher.parmfit.utils.TorsionFit.workflow.read_scan_xyz", lambda path: scan_data)
    monkeypatch.setattr(
        "maple.function.dispatcher.parmfit.utils.TorsionFit.workflow.evaluate_mm_energy",
        lambda *args, **kwargs: types.SimpleNamespace(total=0.0),
    )
    monkeypatch.setattr("maple.function.dispatcher.parmfit.utils.TorsionFit.workflow.run_loss_mode", fake_run_loss_mode)

    run_torsion_workflow(
        atoms=atoms,
        output="rotatable_route.out",
        parameter_set=parameter_set,
        params=TorsionFitParams(enabled=True),
        runtime=TorsionScanRuntime(max_iter=5, memory=5, curvature=70.0, max_step=0.2),
    )

    assert scanned == [(4, 5), (4, 9), (9, 10)]
    assert captured["center_bonds"] == ((4, 5), (4, 9), (9, 10))
    assert captured["scan_keys"] == ((4, 5), (4, 9), (9, 10))


def test_run_torsion_workflow_passes_backend_and_constraint_mode_to_silent_scan(monkeypatch, tmp_path: Path):
    atoms = _four_atom_frame(0.0)
    parameter_set = _make_parameter_set(
        atom_types=["c", "c", "c", "c"],
        bond_graph=[(1, 2), (2, 3), (3, 4)],
        dihedrals=[
            Dihedral(
                atoms=(1, 2, 3, 4),
                atom_types=("c", "c", "c", "c"),
                terms=[FourierTerm(kPhi=0.5, period=1.0, phase=0.0)],
            )
        ],
        nonbonds=[Nonbond(atom=index, atom_type="c", charge=0.0) for index in range(1, 5)],
    )
    captured: dict[str, object] = {}

    def fake_run_silent_scan(*, output, atoms, constraints, params=None, method="lbfgs", constraint_mode="fixinternals"):
        del atoms, constraints
        captured["output"] = output
        captured["params"] = params
        captured["method"] = method
        captured["constraint_mode"] = constraint_mode
        xyz_path = Path(str(Path(output).with_suffix("")) + "_scan_final.xyz")
        _write_scan_xyz(
            xyz_path,
            [(0.0, 0.0, _four_atom_frame(0.0)), (60.0, 0.0, _four_atom_frame(60.0))],
        )
        return SilentScanResult(output_path=output, xyz_path=str(xyz_path))

    monkeypatch.setattr("maple.function.dispatcher.parmfit.utils.TorsionFit.workflow.run_silent_scan", fake_run_silent_scan)

    run_torsion_workflow(
        atoms=atoms,
        output=str(tmp_path / "shared_stage1_backend.out"),
        parameter_set=parameter_set,
        params=TorsionFitParams(enabled=True, torsion_steps=5, refine_rounds=0),
        runtime=TorsionScanRuntime(
            max_iter=5,
            memory=5,
            curvature=70.0,
            max_step=0.2,
            backend="cgws",
            constraint_mode="projected",
        ),
    )

    assert captured["method"] == "cgws"
    assert captured["constraint_mode"] == "projected"
    assert captured["params"]["backend"] == "cgws"
    assert captured["params"]["constraint_mode"] == "projected"


def test_silent_scan_projected_requires_relaxed_mode(tmp_path: Path):
    atoms = _four_atom_frame(0.0)
    atoms.calc = types.SimpleNamespace(
        get_forces=lambda atoms: np.zeros_like(atoms.get_positions(), dtype=float),
        get_potential_energy=lambda atoms, force_consistent=True: 0.0,
    )
    for attr, value in runtime_module.MEDIUM_THRESHOLDS.items():
        setattr(atoms, attr, value)

    with pytest.raises(ValueError, match="projected"):
        run_silent_scan(
            output=str(tmp_path / "projected_invalid.out"),
            atoms=atoms,
            constraints=[
                [1, 2, 3, 4, 30.0, 4],
            ],
            params={"mode": "rigid", "opt": {"max_iter": 2, "max_step": 0.1}},
            method="cgws",
            constraint_mode="projected",
        )


def test_silent_scan_is_exposed_only_through_scan_api():
    assert not hasattr(runtime_module, "SilentScan")


def test_silent_scan_projected_supports_2d_scan(tmp_path: Path, monkeypatch):
    atoms = _five_atom_frame(0.0)
    atoms.calc = types.SimpleNamespace(
        get_forces=lambda atoms: np.zeros_like(atoms.get_positions(), dtype=float),
        get_potential_energy=lambda atoms, force_consistent=True: 0.0,
    )
    for attr, value in runtime_module.MEDIUM_THRESHOLDS.items():
        setattr(atoms, attr, value)

    monkeypatch.setattr(SilentScanEngine, "_generate_scan_values", lambda self: [[0.0, 30.0], [10.0, 20.0, 30.0]])
    monkeypatch.setattr(SilentScanEngine, "_apply_rigid_geometry", lambda self, atoms, coord: atoms)
    monkeypatch.setattr(SilentScanEngine, "_run_projected_optimizer", lambda self, atoms: atoms)

    result = run_silent_scan(
        output=str(tmp_path / "projected_2d.out"),
        atoms=atoms,
        constraints=[
            [1, 2, 3, 4, 30.0, 1],
            [1, 2, 3, 10.0, 2],
        ],
        params={"mode": "relaxed", "opt": {"max_iter": 2, "max_step": 0.1}},
        method="cgws",
        constraint_mode="projected",
    )

    comments = _read_scan_comments(Path(result.xyz_path))
    assert len(comments) == 6
    assert "[0.0000, 10.0000]" in comments[0]
    assert "[30.0000, 30.0000]" in comments[-1]
    assert len(read_scan_final_atoms(result.xyz_path)) == len(atoms)


def test_silent_scan_projected_supports_3d_scan(tmp_path: Path, monkeypatch):
    atoms = _five_atom_frame(0.0)
    atoms.calc = types.SimpleNamespace(
        get_forces=lambda atoms: np.zeros_like(atoms.get_positions(), dtype=float),
        get_potential_energy=lambda atoms, force_consistent=True: 0.0,
    )
    for attr, value in runtime_module.MEDIUM_THRESHOLDS.items():
        setattr(atoms, attr, value)

    monkeypatch.setattr(SilentScanEngine, "_generate_scan_values", lambda self: [[0.0, 30.0], [10.0], [20.0, 40.0]])
    monkeypatch.setattr(SilentScanEngine, "_apply_rigid_geometry", lambda self, atoms, coord: atoms)
    monkeypatch.setattr(SilentScanEngine, "_run_projected_optimizer", lambda self, atoms: atoms)

    result = run_silent_scan(
        output=str(tmp_path / "projected_3d.out"),
        atoms=atoms,
        constraints=[
            [1, 2, 0.1, 1],
            [1, 2, 3, 10.0, 0],
            [1, 2, 3, 4, 20.0, 1],
        ],
        params={"mode": "relaxed", "opt": {"max_iter": 2, "max_step": 0.1}},
        method="cgws",
        constraint_mode="projected",
    )

    comments = _read_scan_comments(Path(result.xyz_path))
    assert len(comments) == 4
    assert "[0.0000, 10.0000, 20.0000]" in comments[0]
    assert "[30.0000, 10.0000, 40.0000]" in comments[-1]


def test_silent_scan_projected_passes_all_active_constraints_to_optimizer(tmp_path: Path, monkeypatch):
    atoms = _five_atom_frame(0.0)
    atoms.calc = types.SimpleNamespace(
        get_forces=lambda atoms: np.zeros_like(atoms.get_positions(), dtype=float),
        get_potential_energy=lambda atoms, force_consistent=True: 0.0,
    )
    for attr, value in runtime_module.MEDIUM_THRESHOLDS.items():
        setattr(atoms, attr, value)

    captured: dict[str, object] = {}

    class FakeCGWS:
        def __init__(self, atoms, output, paras=None):
            del output
            captured["opt"] = dict(paras["opt"])
            self.atoms = atoms

        def run(self):
            return self.atoms

    monkeypatch.setattr("maple.function.dispatcher.parmfit.utils.Scan.engine.CGWS", FakeCGWS)
    monkeypatch.setattr(SilentScanEngine, "_generate_scan_values", lambda self: [[0.0], [10.0], [20.0]])
    monkeypatch.setattr(SilentScanEngine, "_apply_rigid_geometry", lambda self, atoms, coord: atoms)

    run_silent_scan(
        output=str(tmp_path / "projected_constraints.out"),
        atoms=atoms,
        constraints=[
            [1, 2, 0.1, 0],
            [1, 2, 3, 10.0, 0],
            [1, 2, 3, 4, 20.0, 0],
        ],
        params={"mode": "relaxed", "opt": {"max_iter": 2, "max_step": 0.1}},
        method="cgws",
        constraint_mode="projected",
    )

    assert captured["opt"]["bond_constraints"] == [(1, 2)]
    assert captured["opt"]["angle_constraints"] == [(1, 2, 3)]
    assert captured["opt"]["torsion_constraints"] == [(1, 2, 3, 4)]


def test_silent_scan_projected_rejects_above_3d(tmp_path: Path, monkeypatch):
    atoms = _five_atom_frame(0.0)
    atoms.calc = types.SimpleNamespace(
        get_forces=lambda atoms: np.zeros_like(atoms.get_positions(), dtype=float),
        get_potential_energy=lambda atoms, force_consistent=True: 0.0,
    )
    for attr, value in runtime_module.MEDIUM_THRESHOLDS.items():
        setattr(atoms, attr, value)

    monkeypatch.setattr(SilentScanEngine, "_generate_scan_values", lambda self: [[0.0], [10.0], [20.0], [30.0]])

    with pytest.raises(ValueError, match="1D, 2D, 3D"):
        run_silent_scan(
            output=str(tmp_path / "projected_4d.out"),
            atoms=atoms,
            constraints=[
                [1, 2, 0.1, 0],
                [1, 2, 3, 10.0, 0],
                [1, 2, 3, 4, 20.0, 0],
                [2, 3, 4, 5, 30.0, 0],
            ],
            params={"mode": "relaxed", "opt": {"max_iter": 2, "max_step": 0.1}},
            method="cgws",
            constraint_mode="projected",
        )


def test_run_torsion_workflow_executes_stage2_global_refine(tmp_path: Path, monkeypatch):
    atoms = _eight_atom_two_fragment_frame(0.0, 0.0)
    parameter_set = _make_parameter_set(
        atom_types=["c"] * 8,
        bond_graph=[(1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8)],
        dihedrals=[
            Dihedral(atoms=(1, 2, 3, 4), atom_types=("c",) * 4, terms=[FourierTerm(kPhi=0.2, period=1.0, phase=0.0)]),
            Dihedral(atoms=(5, 6, 7, 8), atom_types=("c",) * 4, terms=[FourierTerm(kPhi=0.2, period=1.0, phase=0.0)]),
        ],
        nonbonds=[Nonbond(atom=index, atom_type="c", charge=0.0) for index in range(1, 9)],
    )

    monkeypatch.setattr(
        "maple.function.dispatcher.parmfit.utils.TorsionFit.workflow._run_center_bond_scan",
        lambda atoms, output, params, runtime, center_bond, representative_dihedral: _fake_center_bond_scan(output, center_bond, representative_dihedral),
    )

    result = run_torsion_workflow(
        atoms=atoms,
        output=str(tmp_path / "shared_stage2.out"),
        parameter_set=parameter_set,
        params=TorsionFitParams(
            enabled=True,
            torsion_steps=5,
            refine_rounds=2,
            refine_max_iter=5,
            refine_tol=1.0e-8,
        ),
        runtime=TorsionScanRuntime(max_iter=5, memory=5, curvature=70.0, max_step=0.2),
    )

    assert result.center_bonds == [(2, 3), (6, 7)]
    assert len(result.refine_cycles) >= 1
    assert isinstance(result.refine_cycles[0], TorsionRefineCycle)
    assert result.refine_cycles[0].accepted_blocks >= 1
    assert result.final_parameter_set.dihedrals[0].terms[0].kPhi != pytest.approx(0.2)
    assert result.final_parameter_set.dihedrals[1].terms[0].kPhi != pytest.approx(0.2)


def test_run_torsion_workflow_applies_ncaa_style_center_bond_filter(tmp_path: Path, monkeypatch):
    atoms = _eight_atom_two_fragment_frame(0.0, 0.0)
    parameter_set = _make_parameter_set(
        atom_types=["c"] * 8,
        bond_graph=[(1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8)],
        dihedrals=[
            Dihedral(atoms=(1, 2, 3, 4), atom_types=("c",) * 4, terms=[FourierTerm(kPhi=0.2, period=1.0, phase=0.0)]),
            Dihedral(atoms=(5, 6, 7, 8), atom_types=("c",) * 4, terms=[FourierTerm(kPhi=0.2, period=1.0, phase=0.0)]),
        ],
        nonbonds=[Nonbond(atom=index, atom_type="c", charge=0.0) for index in range(1, 9)],
    )

    monkeypatch.setattr(
        "maple.function.dispatcher.parmfit.utils.TorsionFit.workflow._run_center_bond_scan",
        lambda atoms, output, params, runtime, center_bond, representative_dihedral: _fake_center_bond_scan(output, center_bond, representative_dihedral),
    )

    result = run_torsion_workflow(
        atoms=atoms,
        output=str(tmp_path / "filtered.out"),
        parameter_set=parameter_set,
        params=TorsionFitParams(enabled=True, torsion_steps=5, refine_rounds=0),
        runtime=TorsionScanRuntime(max_iter=5, memory=5, curvature=70.0, max_step=0.2),
        center_bond_filter=lambda bond: bond == (2, 3),
    )

    assert result.center_bonds == [(2, 3)]
    assert list(result.scan_xyz) == [(2, 3)]
    assert result.final_parameter_set.dihedrals[0].terms[0].kPhi != pytest.approx(0.2)
    assert result.final_parameter_set.dihedrals[1].terms[0].kPhi == pytest.approx(0.2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
