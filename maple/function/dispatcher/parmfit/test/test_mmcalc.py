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
    masses = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999}
    atomic_numbers = {"H": 1, "C": 6, "N": 7, "O": 8}

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

        def get_masses(self):
            return np.asarray([masses[symbol] for symbol in self.symbols], dtype=float)

        def get_chemical_symbols(self):
            return self.symbols[:]

        def get_atomic_numbers(self):
            return np.asarray([atomic_numbers[symbol] for symbol in self.symbols], dtype=int)

    ase_stub.Atoms = Atoms
    sys.modules["ase"] = ase_stub

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maple.function.dispatcher.parmfit.utils.mechanics import (
    COULOMB_KCAL_ANG_E2,
    SCNB,
    SCEE,
    angle_radians,
    build_mm_topology_cache,
    dihedral_radians,
    distance_angstrom,
    evaluate_mm_energy,
)
from maple.function.dispatcher.parmfit.utils.readparm import (
    Angle,
    Bond,
    CorrectionParameterSet,
    Dihedral,
    FourierTerm,
    FrcmodDB,
    Improper,
    Mol2Atom,
    Mol2Bond,
    Mol2Topology,
    Nonbond,
    build_correction_parameter_set,
)


def _make_atoms(symbols, positions):
    return Atoms(symbols=symbols, positions=positions)


def _make_parameter_set(
    atom_types,
    bond_graph,
    *,
    bonds=None,
    angles=None,
    dihedrals=None,
    impropers=None,
    nonbonds=None,
):
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
        bonds=list(bonds or []),
        angles=list(angles or []),
        dihedrals=list(dihedrals or []),
        impropers=list(impropers or []),
        nonbonds=list(nonbond_list),
        unmatched_bonds=[],
        unmatched_angles=[],
        unmatched_dihedrals=[],
        unmatched_impropers=[],
        unmatched_nonbonds=[],
    )


def test_geometry_helpers_return_expected_values():
    positions = np.asarray(
        [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (2.0, 1.0, 1.0),
            (2.0, 1.0, -1.0),
        ],
        dtype=float,
    )

    assert distance_angstrom(positions, 1, 2) == pytest.approx(1.0)
    assert angle_radians(positions, 1, 2, 3) == pytest.approx(math.pi / 2.0)
    assert dihedral_radians(positions, 1, 2, 3, 4) == pytest.approx(-math.pi / 4.0)
    assert dihedral_radians(positions, 1, 2, 3, 5) == pytest.approx(math.pi / 4.0)


def test_bond_angle_proper_and_improper_energies_are_accumulated():
    atoms = _make_atoms(
        symbols=["C", "C", "C", "C"],
        positions=[
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (2.0, 1.0, 1.0),
        ],
    )
    parameter_set = _make_parameter_set(
        atom_types=["c", "c", "c", "c"],
        bond_graph=[(1, 2), (2, 3), (3, 4)],
        bonds=[Bond(atoms=(1, 2), atom_types=("c", "c"), kBond=100.0, rEq=1.1)],
        angles=[Angle(atoms=(1, 2, 3), atom_types=("c", "c", "c"), kTheta=2.0, thetaEq=math.pi / 3.0)],
        dihedrals=[
            Dihedral(
                atoms=(1, 2, 3, 4),
                atom_types=("c", "c", "c", "c"),
                terms=[
                    FourierTerm(kPhi=1.0, period=1.0, phase=0.0),
                    FourierTerm(kPhi=0.5, period=2.0, phase=math.pi),
                ],
            )
        ],
        impropers=[
            Improper(
                atoms=(1, 2, 3, 4),
                atom_types=("c", "c", "c", "c"),
                terms=[FourierTerm(kPhi=0.25, period=1.0, phase=0.0)],
            )
        ],
        nonbonds=[
            Nonbond(atom=1, atom_type="c", charge=0.0),
            Nonbond(atom=2, atom_type="c", charge=0.0),
            Nonbond(atom=3, atom_type="c", charge=0.0),
            Nonbond(atom=4, atom_type="c", charge=0.0),
        ],
    )

    result = evaluate_mm_energy(atoms, parameter_set)

    expected_bond = 100.0 * (1.0 - 1.1) ** 2
    expected_angle = 2.0 * (math.pi / 2.0 - math.pi / 3.0) ** 2
    phi = -math.pi / 4.0
    expected_proper = (
        1.0 * (1.0 + math.cos(phi))
        + 0.5 * (1.0 + math.cos(2.0 * phi - math.pi))
    )
    expected_improper = 0.25 * (1.0 + math.cos(phi))

    assert result.bond == pytest.approx(expected_bond)
    assert result.angle == pytest.approx(expected_angle)
    assert result.proper == pytest.approx(expected_proper)
    assert result.improper == pytest.approx(expected_improper)
    assert result.vdw == pytest.approx(0.0)
    assert result.elec == pytest.approx(0.0)
    assert result.total == pytest.approx(expected_bond + expected_angle + expected_proper + expected_improper)


def test_nonbond_exclusions_and_14_scaling_follow_amber_rules():
    atoms = _make_atoms(
        symbols=["C", "C", "C", "C", "C"],
        positions=[
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (3.0, 0.0, 0.0),
            (4.0, 0.0, 0.0),
        ],
    )
    parameter_set = _make_parameter_set(
        atom_types=["c", "c", "c", "c", "c"],
        bond_graph=[(1, 2), (2, 3), (3, 4), (4, 5)],
        nonbonds=[
            Nonbond(atom=1, atom_type="c", charge=1.0, rmin_half=1.0, epsilon=1.0),
            Nonbond(atom=2, atom_type="c", charge=0.0, rmin_half=1.0, epsilon=0.0),
            Nonbond(atom=3, atom_type="c", charge=0.0, rmin_half=1.0, epsilon=0.0),
            Nonbond(atom=4, atom_type="c", charge=1.0, rmin_half=1.0, epsilon=1.0),
            Nonbond(atom=5, atom_type="c", charge=1.0, rmin_half=1.0, epsilon=1.0),
        ],
    )

    cache = build_mm_topology_cache(parameter_set)
    result = evaluate_mm_energy(atoms, parameter_set, topology_cache=cache)

    vdw_14 = ((2.0 / 3.0) ** 12 - 2.0 * (2.0 / 3.0) ** 6) / SCNB
    elec_14 = (COULOMB_KCAL_ANG_E2 / 3.0) / SCEE
    vdw_15 = (0.5 ** 12 - 2.0 * 0.5 ** 6)
    elec_15 = COULOMB_KCAL_ANG_E2 / 4.0

    assert (1, 2) in cache.excluded_12
    assert (1, 3) in cache.excluded_13
    assert (1, 4) in cache.scaled_14
    assert (1, 5) not in cache.scaled_14

    assert result.vdw == pytest.approx(vdw_14 + vdw_15)
    assert result.elec == pytest.approx(elec_14 + elec_15)
    assert result.total == pytest.approx(result.vdw + result.elec)


def test_zero_proper_center_bond_skips_only_target_propers():
    atoms = _make_atoms(
        symbols=["C", "C", "C", "C", "C", "C"],
        positions=[
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (2.0, 1.0, 1.0),
            (0.0, 1.0, 0.0),
            (2.0, 0.0, -1.0),
        ],
    )
    parameter_set = _make_parameter_set(
        atom_types=["c", "c", "c", "c", "c", "c"],
        bond_graph=[(1, 2), (2, 3), (3, 4), (2, 5), (3, 6)],
        dihedrals=[
            Dihedral(
                atoms=(1, 2, 3, 4),
                atom_types=("c", "c", "c", "c"),
                terms=[FourierTerm(kPhi=1.0, period=1.0, phase=0.0)],
            ),
            Dihedral(
                atoms=(5, 2, 3, 6),
                atom_types=("c", "c", "c", "c"),
                terms=[FourierTerm(kPhi=2.0, period=2.0, phase=0.0)],
            ),
        ],
        impropers=[
            Improper(
                atoms=(1, 2, 3, 4),
                atom_types=("c", "c", "c", "c"),
                terms=[FourierTerm(kPhi=0.75, period=1.0, phase=0.0)],
            )
        ],
        nonbonds=[Nonbond(atom=index, atom_type="c", charge=0.0) for index in range(1, 7)],
    )

    full = evaluate_mm_energy(atoms, parameter_set)
    zeroed = evaluate_mm_energy(atoms, parameter_set, zero_proper_center_bond=(3, 2))

    assert full.proper > 0.0
    assert zeroed.proper == pytest.approx(0.0)
    assert zeroed.improper == pytest.approx(full.improper)
    assert zeroed.bond == pytest.approx(full.bond)
    assert zeroed.angle == pytest.approx(full.angle)
    assert zeroed.vdw == pytest.approx(full.vdw)
    assert zeroed.elec == pytest.approx(full.elec)
    assert zeroed.total == pytest.approx(full.total - full.proper)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
