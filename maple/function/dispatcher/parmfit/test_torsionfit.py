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

    class Atoms:  # type: ignore[override]
        def __init__(self, symbols, positions):
            self.symbols = list(symbols)
            self.positions = np.asarray(positions, dtype=float)
            self.info = {}
            self.calc = None

        def __len__(self):
            return len(self.symbols)

        def copy(self):
            atoms = Atoms(self.symbols[:], self.positions.copy())
            atoms.info = dict(self.info)
            atoms.calc = self.calc
            return atoms

        def get_positions(self):
            return np.asarray(self.positions, dtype=float)

        def set_positions(self, positions):
            self.positions = np.asarray(positions, dtype=float)

        def get_chemical_symbols(self):
            return self.symbols[:]

    ase_stub.Atoms = Atoms
    sys.modules["ase"] = ase_stub

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maple.function.dispatcher.parmfit.utils.mmcalc import build_mm_topology_cache
from maple.function.dispatcher.parmfit.utils.parm import Dihedral, FourierTerm, Nonbond
from maple.function.dispatcher.parmfit.utils.readparm import (
    CorrectionParameterSet,
    FrcmodDB,
    Mol2Atom,
    Mol2Bond,
    Mol2Topology,
)
from maple.function.dispatcher.parmfit.utils.torsionfit import (
    HARTREE_TO_KCAL_MOL,
    apply_center_bond_terms,
    apply_fitted_torsion,
    build_global_torsion_problem,
    center_bond_group_atoms,
    enumerate_fittable_center_bonds,
    evaluate_global_refit_objective,
    evaluate_refit_objective,
    extract_global_delta,
    fit_torsion_scan,
    fit_scan_xyz_to_center_bond,
    format_torsion_fit_report,
    is_ring_center_bond,
    read_scan_xyz,
    refine_torsion_scan,
    refine_torsion_scans_global,
    representative_dihedral_for_center_bond,
)


def _make_atoms(symbols, positions):
    return Atoms(symbols=symbols, positions=positions)


def _make_parameter_set(atom_types, bond_graph, *, dihedrals=None, nonbonds=None):
    nonbond_list = list(nonbonds or [])
    charges = {entry.atom: entry.charge for entry in nonbond_list}
    mol2_atoms = [
        Mol2Atom(atom_id=index, name=f"A{index}", atom_type=atom_type, charge=charges.get(index, 0.0))
        for index, atom_type in enumerate(atom_types, start=1)
    ]
    mol2_bonds = [
        Mol2Bond(bond_id=index, atom1=i, atom2=j, bond_type="1")
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


def _four_atom_frame(phi_deg: float) -> Atoms:
    rad = math.radians(180.0 + phi_deg)
    positions = [
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        (math.cos(rad), math.sin(rad), 1.0),
    ]
    return _make_atoms(["C", "C", "C", "C"], positions)


def _five_atom_frame(phi_deg: float) -> Atoms:
    rad = math.radians(180.0 + phi_deg)
    positions = [
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        (math.cos(rad), math.sin(rad), 1.0),
        (0.0, 1.0, 0.0),
    ]
    return _make_atoms(["C", "C", "C", "C", "C"], positions)


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


def test_read_scan_xyz_parses_single_coordinate_and_energy(tmp_path: Path):
    path = tmp_path / "scan.xyz"
    atoms = _four_atom_frame(60.0)
    _write_scan_xyz(path, [(60.0, 1.5, atoms), (120.0, 2.0, atoms)])

    data = read_scan_xyz(str(path))

    assert len(data["frames"]) == 2
    assert data["angles_deg"].tolist() == pytest.approx([60.0, 120.0])
    assert data["qm_hartree"][0] == pytest.approx(1.5)
    assert data["qm_kcal"][0] == pytest.approx(1.5 * HARTREE_TO_KCAL_MOL)


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

    before_rmse, _, _ = evaluate_refit_objective(
        scan_data,
        parameter_set,
        center_bond=(2, 3),
        fitted_terms=[[FourierTerm(kPhi=k_orig, period=1.0, phase=0.0)]],
    )
    result = fit_torsion_scan(scan_data, parameter_set, center_bond=(2, 3))

    fitted_k = result["fitted_terms"][0][0].kPhi
    assert k_orig < fitted_k < k_true
    assert result["delta_kphi"][0] == pytest.approx(fitted_k - k_orig)
    assert result["rmse"] < before_rmse
    assert result["mae"] < before_rmse


def test_fit_torsion_scan_recovers_independent_terms_for_multiple_dihedrals(tmp_path: Path):
    k1 = 1.5
    k2 = 0.5
    phis = [0.0, 60.0, 120.0, 180.0, -120.0, -60.0]
    frames = []
    for phi in phis:
        atoms = _five_atom_frame(phi)
        phi1 = math.radians(phi)
        phi2 = math.radians(phi - 90.0)
        energy_kcal = k1 * (1.0 + math.cos(phi1)) + k2 * (1.0 + math.cos(phi2))
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

    before_rmse, _, _ = evaluate_refit_objective(
        scan_data,
        parameter_set,
        center_bond=(2, 3),
        fitted_terms=[[FourierTerm(kPhi=0.25, period=1.0, phase=0.0)], [FourierTerm(kPhi=0.25, period=1.0, phase=0.0)]],
    )
    result = fit_torsion_scan(scan_data, parameter_set, center_bond=(3, 2))

    assert len(result["target_dihedrals"]) == 2
    assert result["rmse"] < before_rmse
    assert len(result["angles_deg"]) == len(phis)
    after_rmse, _, _ = evaluate_refit_objective(
        scan_data,
        parameter_set,
        center_bond=(2, 3),
        fitted_terms=result["fitted_terms"],
    )
    assert after_rmse == pytest.approx(result["rmse"])

    updated = apply_fitted_torsion(result, parameter_set)
    cache = build_mm_topology_cache(updated)
    assert len(cache.proper_by_center_bond[(2, 3)]) == 2
    report = "".join(format_torsion_fit_report(result))
    assert "Center bond: (2, 3)" in report
    assert "RMSE:" in report


def test_fit_torsion_scan_handles_misaligned_terms_independently(tmp_path: Path):
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

    before_rmse, _, _ = evaluate_refit_objective(
        scan_data,
        parameter_set,
        center_bond=(2, 3),
        fitted_terms=[
            [FourierTerm(kPhi=k_orig, period=1.0, phase=0.0)],
            [FourierTerm(kPhi=k_orig, period=2.0, phase=0.0)],
        ],
    )
    result = fit_torsion_scan(scan_data, parameter_set, center_bond=(2, 3))

    fitted_k1 = result["fitted_terms"][0][0].kPhi
    fitted_k2 = result["fitted_terms"][1][0].kPhi
    assert result["rmse"] < before_rmse
    assert k_orig < fitted_k1 < k1
    assert fitted_k2 > k_orig
    assert fitted_k2 == pytest.approx(k2, abs=3.0e-2)
    after_rmse, _, _ = evaluate_refit_objective(
        scan_data,
        parameter_set,
        center_bond=(2, 3),
        fitted_terms=result["fitted_terms"],
    )
    assert after_rmse == pytest.approx(result["rmse"])


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


def test_global_objective_weights_each_scan_equally(tmp_path: Path):
    k_orig = 0.2
    frames_a = []
    frames_b = []
    for phi in [0.0, 180.0]:
        atoms = _eight_atom_two_fragment_frame(phi, 0.0)
        energy_kcal = 1.4 * (1.0 + math.cos(math.radians(phi)))
        frames_a.append((phi, energy_kcal / HARTREE_TO_KCAL_MOL, atoms))
    for phi in [0.0, 60.0, 120.0, 180.0, -120.0, -60.0]:
        atoms = _eight_atom_two_fragment_frame(0.0, phi)
        energy_kcal = 0.9 * (1.0 + math.cos(math.radians(phi)))
        frames_b.append((phi, energy_kcal / HARTREE_TO_KCAL_MOL, atoms))

    path_a = tmp_path / "weighted_a.xyz"
    path_b = tmp_path / "weighted_b.xyz"
    _write_scan_xyz(path_a, frames_a)
    _write_scan_xyz(path_b, frames_b)
    scan_map = {
        (2, 3): read_scan_xyz(str(path_a)),
        (6, 7): read_scan_xyz(str(path_b)),
    }
    parameter_set = _make_parameter_set(
        atom_types=["c"] * 8,
        bond_graph=[(1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8)],
        dihedrals=[
            Dihedral(atoms=(1, 2, 3, 4), atom_types=("c",) * 4, terms=[FourierTerm(kPhi=k_orig, period=1.0, phase=0.0)]),
            Dihedral(atoms=(5, 6, 7, 8), atom_types=("c",) * 4, terms=[FourierTerm(kPhi=k_orig, period=1.0, phase=0.0)]),
        ],
        nonbonds=[Nonbond(atom=index, atom_type="c", charge=0.0) for index in range(1, 9)],
    )

    problem = build_global_torsion_problem(parameter_set, [(2, 3), (6, 7)], scan_map)
    objective = evaluate_global_refit_objective(problem, np.zeros(len(problem["term_paths"]), dtype=float))
    expected = np.mean([value ** 2 for value in objective["per_scan_rmse"].values()])
    assert objective["data_loss"] == pytest.approx(expected)

    fit_a = fit_torsion_scan(scan_map[(2, 3)], parameter_set, center_bond=(2, 3))
    fit_b = fit_torsion_scan(scan_map[(6, 7)], parameter_set, center_bond=(6, 7))
    stage1 = apply_fitted_torsion(fit_b, apply_fitted_torsion(fit_a, parameter_set))
    delta_init = extract_global_delta(problem, stage1)
    before = evaluate_global_refit_objective(problem, delta_init)
    delta_final, cycles = refine_torsion_scans_global(problem, delta_init, max_sweeps=2, max_block_iter=5, tol=1.0e-8)
    after = evaluate_global_refit_objective(problem, delta_final)
    assert cycles
    assert after["total_loss"] <= before["total_loss"] + 1.0e-12


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
    wrapped = fit_scan_xyz_to_center_bond(str(path), parameter_set, center_bond=(2, 3))

    assert wrapped["fitted_terms"][0][0].kPhi == pytest.approx(direct["fitted_terms"][0][0].kPhi)
    assert wrapped["orig_mm_rel"][0] == pytest.approx(direct["orig_mm_rel"][0])


def test_evaluate_refit_objective_matches_true_mm_curve(tmp_path: Path):
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
    rmse, mm_refit_rel, residual_after = evaluate_refit_objective(
        scan_data,
        parameter_set,
        center_bond=(2, 3),
        fitted_terms=fitted_terms,
    )

    qm_rel = np.asarray(scan_data["qm_rel"], dtype=float)
    assert rmse == pytest.approx(0.0, abs=5.0e-8)
    assert mm_refit_rel == pytest.approx(qm_rel)
    assert residual_after == pytest.approx(np.zeros_like(qm_rel), abs=5.0e-8)


def test_refine_torsion_scan_improves_true_rmse_from_perturbed_start(tmp_path: Path):
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

    before_rmse, _, _ = evaluate_refit_objective(
        scan_data,
        parameter_set,
        center_bond=(2, 3),
        fitted_terms=[[FourierTerm(kPhi=0.25, period=1.0, phase=0.0)]],
    )
    result = refine_torsion_scan(
        scan_data,
        parameter_set,
        center_bond=(2, 3),
        cycle=1,
        max_iter=20,
        tol=1.0e-8,
    )

    assert result["accepted"]
    assert result["rmse_after"] < result["rmse_before"]
    assert result["rmse_before"] == pytest.approx(before_rmse)
    updated = apply_center_bond_terms(parameter_set, (2, 3), result["terms_after"])
    after_rmse, _, _ = evaluate_refit_objective(
        scan_data,
        updated,
        center_bond=(2, 3),
        fitted_terms=result["terms_after"],
    )
    assert after_rmse == pytest.approx(result["rmse_after"])


def test_refine_torsion_scan_rejects_already_optimal_terms(tmp_path: Path):
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

    result = refine_torsion_scan(
        scan_data,
        parameter_set,
        center_bond=(2, 3),
        cycle=1,
        max_iter=5,
        tol=1.0e-8,
    )

    assert not result["accepted"]
    assert result["rmse_before"] == pytest.approx(0.0, abs=5.0e-8)
    assert result["rmse_after"] == pytest.approx(0.0, abs=5.0e-8)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
