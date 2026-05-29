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

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maple.function.dispatcher.parmfit.test.helpers.sample_test import HARTREE_TO_KCAL_MOL, run_conformer_benchmark
from maple.function.dispatcher.parmfit.utils.mechanics import dihedral_radians
from maple.function.dispatcher.parmfit.utils.readparm import (
    CorrectionParameterSet,
    Dihedral,
    FrcmodDB,
    FourierTerm,
    Mol2Atom,
    Mol2Bond,
    Mol2Topology,
    Nonbond,
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
        nonbonds=list(nonbonds or []),
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


def _three_atom_frame() -> Atoms:
    positions = [(0.0, 0.0, 0.0), (1.2, 0.0, 0.0), (2.4, 0.0, 0.0)]
    return _make_atoms(["C", "C", "O"], positions)


def _chain_parameter_set(k_mm: float, phase_mm: float = 0.0) -> CorrectionParameterSet:
    return _make_parameter_set(
        atom_types=["c", "c", "c", "c"],
        bond_graph=[(1, 2), (2, 3), (3, 4)],
        dihedrals=[
            Dihedral(
                atoms=(1, 2, 3, 4),
                atom_types=("c", "c", "c", "c"),
                terms=[FourierTerm(kPhi=k_mm, period=1.0, phase=phase_mm)],
            )
        ],
        nonbonds=[Nonbond(atom=index, atom_type="c", charge=0.0) for index in range(1, 5)],
    )


def _no_rotor_parameter_set() -> CorrectionParameterSet:
    return _make_parameter_set(
        atom_types=["c", "c", "o"],
        bond_graph=[(1, 2), (2, 3)],
        dihedrals=[],
        nonbonds=[Nonbond(atom=index, atom_type="c", charge=0.0) for index in range(1, 4)],
    )


def _torsion_deg(atoms: Atoms) -> float:
    return float(np.degrees(dihedral_radians(atoms.get_positions(), 1, 2, 3, 4)))


class FakeTorsionCalculator:
    def __init__(self, k_qm: float, phase_qm: float = 0.0):
        self.k_qm = float(k_qm)
        self.phase_qm = float(phase_qm)

    def get_potential_energy(self, atoms, force_consistent=True):
        phi = dihedral_radians(atoms.get_positions(), 1, 2, 3, 4)
        energy_kcal = self.k_qm * (1.0 + math.cos(phi - self.phase_qm))
        return energy_kcal / HARTREE_TO_KCAL_MOL


class FakeConstantCalculator:
    def __init__(self, energy_hartree: float):
        self.energy_hartree = float(energy_hartree)

    def get_potential_energy(self, atoms, force_consistent=True):
        return self.energy_hartree


def test_run_conformer_benchmark_is_reproducible_and_changes_dihedral():
    atoms = _four_atom_frame(180.0)
    atoms.calc = FakeTorsionCalculator(k_qm=2.0, phase_qm=0.0)
    parameter_set = _chain_parameter_set(k_mm=1.0, phase_mm=0.0)

    confs_a, mlp_a, mm_a, rmse_a = run_conformer_benchmark(
        atoms,
        parameter_set,
        n_samples=18,
        k=1,
        keep_top_n=3,
        random_seed=7,
    )
    confs_b, mlp_b, mm_b, rmse_b = run_conformer_benchmark(
        atoms,
        parameter_set,
        n_samples=18,
        k=1,
        keep_top_n=3,
        random_seed=7,
    )

    assert len(confs_a) == len(confs_b) == 2
    assert mlp_a == pytest.approx(mlp_b)
    assert mm_a == pytest.approx(mm_b)
    assert rmse_a == pytest.approx(rmse_b)

    observed = sorted(round(((_torsion_deg(conf) + 360.0) % 360.0), 3) for conf in confs_a)
    assert observed == pytest.approx([180.0, 300.0], abs=1.0e-2)


def test_run_conformer_benchmark_deduplicates_and_keeps_lowest_mlp():
    atoms = _four_atom_frame(180.0)
    atoms.calc = FakeTorsionCalculator(k_qm=2.0, phase_qm=0.0)
    parameter_set = _chain_parameter_set(k_mm=1.0, phase_mm=0.0)

    dedup_confs, _, _, _ = run_conformer_benchmark(
        atoms,
        parameter_set,
        n_samples=40,
        k=1,
        keep_top_n=10,
        random_seed=3,
    )
    kept_confs, mlp_rel, mm_rel, rmse = run_conformer_benchmark(
        atoms,
        parameter_set,
        n_samples=40,
        k=1,
        keep_top_n=1,
        random_seed=3,
    )

    assert len(dedup_confs) == 2
    assert len(kept_confs) == 1
    assert ((_torsion_deg(kept_confs[0]) + 360.0) % 360.0) == pytest.approx(180.0, abs=1.0e-2)
    assert mlp_rel.tolist() == pytest.approx([0.0])
    assert mm_rel.tolist() == pytest.approx([0.0])
    assert rmse == pytest.approx(0.0)


def test_run_conformer_benchmark_uses_common_mlp_reference_for_mm():
    atoms = _four_atom_frame(180.0)
    atoms.calc = FakeTorsionCalculator(k_qm=2.0, phase_qm=0.0)
    parameter_set = _chain_parameter_set(k_mm=1.0, phase_mm=math.pi)

    _, mlp_rel, mm_rel, rmse = run_conformer_benchmark(
        atoms,
        parameter_set,
        n_samples=40,
        k=1,
        keep_top_n=3,
        random_seed=11,
    )

    assert mlp_rel.tolist() == pytest.approx([0.0, 3.0], abs=1.0e-6)
    assert mm_rel.tolist() == pytest.approx([0.0, -1.5], abs=1.0e-6)
    assert rmse == pytest.approx(math.sqrt(10.125), abs=1.0e-6)


def test_run_conformer_benchmark_handles_excluded_or_missing_rotatable_bonds():
    atoms = _four_atom_frame(180.0)
    atoms.calc = FakeTorsionCalculator(k_qm=2.0, phase_qm=0.0)
    parameter_set = _chain_parameter_set(k_mm=1.0, phase_mm=0.0)

    confs, mlp_rel, mm_rel, rmse = run_conformer_benchmark(
        atoms,
        parameter_set,
        n_samples=20,
        k=1,
        keep_top_n=5,
        exclude_bonds=[(2, 3)],
        random_seed=5,
    )

    assert len(confs) == 1
    assert ((_torsion_deg(confs[0]) + 360.0) % 360.0) == pytest.approx(0.0, abs=1.0e-2)
    assert mlp_rel.tolist() == pytest.approx([0.0])
    assert mm_rel.tolist() == pytest.approx([0.0])
    assert rmse == pytest.approx(0.0)

    atoms_no_rotor = _three_atom_frame()
    atoms_no_rotor.calc = FakeConstantCalculator(0.123)
    no_rotor_ps = _no_rotor_parameter_set()
    confs2, mlp_rel2, mm_rel2, rmse2 = run_conformer_benchmark(
        atoms_no_rotor,
        no_rotor_ps,
        n_samples=8,
        k=1,
        keep_top_n=5,
        random_seed=5,
    )

    assert len(confs2) == 1
    assert mlp_rel2.tolist() == pytest.approx([0.0])
    assert mm_rel2.tolist() == pytest.approx([0.0])
    assert rmse2 == pytest.approx(0.0)
