from __future__ import annotations

from pathlib import Path
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

        def get_positions(self):
            return np.asarray(self.positions, dtype=float)

        def get_chemical_symbols(self):
            return self.symbols[:]

    ase_stub.Atoms = Atoms
    sys.modules["ase"] = ase_stub

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maple.function.dispatcher.parmfit.utils.outputparm import write_amber_files, write_gromacs_files
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
    parse_frcmod,
)
from maple.function.dispatcher.parmfit.test.helpers.parameter_audit import gromacs_row_by_atoms, gromacs_section_rows


def _make_atoms(symbols, positions):
    return Atoms(symbols=symbols, positions=positions)


def _make_parameter_set() -> CorrectionParameterSet:
    mol2 = Mol2Topology(
        atoms=[
            Mol2Atom(1, "C1", "ca", -0.10),
            Mol2Atom(2, "C2", "ca", 0.05),
            Mol2Atom(3, "C3", "ca", 0.05),
            Mol2Atom(4, "C4", "ca", 0.00),
        ],
        bonds=[
            Mol2Bond(1, 1, 2, "1"),
            Mol2Bond(2, 2, 3, "1"),
            Mol2Bond(3, 3, 4, "1"),
        ],
        id_to_index={1: 1, 2: 2, 3: 3, 4: 4},
        adjacency={1: {2}, 2: {1, 3}, 3: {2, 4}, 4: {3}},
    )
    return CorrectionParameterSet(
        mol2=mol2,
        frcmod=FrcmodDB(mass_params={"ca": 12.01}),
        bonds=[Bond(atoms=(1, 2), atom_types=("ca", "ca"), kBond=100.0, rEq=1.50)],
        angles=[Angle(atoms=(1, 2, 3), atom_types=("ca", "ca", "ca"), kTheta=10.0, thetaEq=np.pi / 3.0)],
        dihedrals=[
            Dihedral(
                atoms=(1, 2, 3, 4),
                atom_types=("ca", "ca", "ca", "ca"),
                terms=[FourierTerm(kPhi=1.5, period=2.0, phase=np.pi)],
            )
        ],
        impropers=[
            Improper(
                atoms=(1, 2, 3, 4),
                atom_types=("ca", "ca", "ca", "ca"),
                terms=[FourierTerm(kPhi=1.1, period=2.0, phase=np.pi)],
            )
        ],
        nonbonds=[
            Nonbond(atom=1, atom_type="ca", charge=-0.10, rmin_half=1.50, epsilon=0.10),
            Nonbond(atom=2, atom_type="ca", charge=0.05, rmin_half=1.50, epsilon=0.10),
            Nonbond(atom=3, atom_type="ca", charge=0.05, rmin_half=1.50, epsilon=0.10),
            Nonbond(atom=4, atom_type="ca", charge=0.00, rmin_half=1.50, epsilon=0.10),
        ],
        unmatched_bonds=[],
        unmatched_angles=[],
        unmatched_dihedrals=[],
        unmatched_impropers=[],
        unmatched_nonbonds=[],
    )


def _make_split_parameter_set() -> CorrectionParameterSet:
    mol2 = Mol2Topology(
        atoms=[
            Mol2Atom(1, "C1", "ca", -0.10),
            Mol2Atom(2, "C2", "ca", 0.05),
            Mol2Atom(3, "C3", "ca", 0.05),
            Mol2Atom(4, "C4", "ca", 0.00),
            Mol2Atom(5, "C5", "ca", 0.00),
        ],
        bonds=[
            Mol2Bond(1, 1, 2, "1"),
            Mol2Bond(2, 2, 3, "1"),
            Mol2Bond(3, 3, 4, "1"),
            Mol2Bond(4, 4, 5, "1"),
        ],
        id_to_index={index: index for index in range(1, 6)},
        adjacency={1: {2}, 2: {1, 3}, 3: {2, 4}, 4: {3, 5}, 5: {4}},
    )
    return CorrectionParameterSet(
        mol2=mol2,
        frcmod=FrcmodDB(mass_params={"ca": 12.01}),
        bonds=[
            Bond(atoms=(1, 2), atom_types=("ca", "ca"), kBond=100.0, rEq=1.50),
            Bond(atoms=(2, 3), atom_types=("ca", "ca"), kBond=101.0, rEq=1.51),
        ],
        angles=[
            Angle(atoms=(1, 2, 3), atom_types=("ca", "ca", "ca"), kTheta=10.0, thetaEq=np.pi / 3.0),
            Angle(atoms=(2, 3, 4), atom_types=("ca", "ca", "ca"), kTheta=11.0, thetaEq=np.pi / 2.0),
        ],
        dihedrals=[
            Dihedral(
                atoms=(1, 2, 3, 4),
                atom_types=("ca", "ca", "ca", "ca"),
                terms=[FourierTerm(kPhi=1.5, period=2.0, phase=np.pi)],
            ),
            Dihedral(
                atoms=(2, 3, 4, 5),
                atom_types=("ca", "ca", "ca", "ca"),
                terms=[FourierTerm(kPhi=2.5, period=3.0, phase=0.0)],
            ),
        ],
        impropers=[
            Improper(
                atoms=(1, 2, 3, 4),
                atom_types=("ca", "ca", "ca", "ca"),
                terms=[FourierTerm(kPhi=1.1, period=2.0, phase=np.pi)],
            )
        ],
        nonbonds=[
            Nonbond(atom=index, atom_type="ca", charge=0.01 * index, rmin_half=1.50, epsilon=0.10)
            for index in range(1, 6)
        ],
        unmatched_bonds=[],
        unmatched_angles=[],
        unmatched_dihedrals=[],
        unmatched_impropers=[],
        unmatched_nonbonds=[],
    )


def _write_input_mol2(path: Path, parameter_set: CorrectionParameterSet) -> None:
    lines = [
        "@<TRIPOS>MOLECULE",
        "TEST",
        f"{len(parameter_set.mol2.atoms)} {len(parameter_set.mol2.bonds)} 0 0 0",
        "SMALL",
        "USER_CHARGES",
        "@<TRIPOS>ATOM",
    ]
    for atom in parameter_set.mol2.atoms:
        lines.append(
            f"{atom.atom_id} {atom.name} {float(atom.atom_id):.4f} 0.0000 0.0000 {atom.atom_type} 1 RES {atom.charge:.4f}"
        )
    lines.append("@<TRIPOS>BOND")
    for bond in parameter_set.mol2.bonds:
        lines.append(f"{bond.bond_id} {bond.atom1} {bond.atom2} {bond.bond_type}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def test_parse_frcmod_reads_mass_section(tmp_path: Path):
    frcmod = tmp_path / "demo.frcmod"
    frcmod.write_text(
        "\n".join(
            [
                "REMARK demo",
                "",
                "MASS",
                "CT  12.010  0.360",
                "ca  12.010  0.360",
                "",
                "BOND",
                "",
                "ANGLE",
                "",
                "DIHE",
                "",
                "IMPROPER",
                "",
                "NONBON",
                "CT  1.9080  0.1094",
                "",
            ]
        ),
        encoding="utf-8",
    )

    parsed = parse_frcmod(str(frcmod))

    assert parsed.mass_params["CT"] == pytest.approx(12.01)
    assert parsed.mass_params["ca"] == pytest.approx(12.01)


def test_write_gromacs_files_writes_expected_sections_and_units(tmp_path: Path):
    parameter_set = _make_parameter_set()
    atoms = _make_atoms(
        ["C", "C", "C", "C"],
        [
            (0.0, 0.0, 0.0),
            (1.5, 0.0, 0.0),
            (3.0, 0.0, 0.0),
            (4.5, 1.0, 0.0),
        ],
    )

    top_path, gro_path, meta = write_gromacs_files(parameter_set, atoms, str(tmp_path / "demo"))

    assert Path(top_path).is_file()
    assert Path(gro_path).is_file()
    assert meta["omitted_counts"]["bonds"] == 0
    assert meta["omitted_counts"]["angles"] == 0
    assert meta["omitted_counts"]["dihedrals"] == 0
    assert meta["omitted_counts"]["impropers"] == 0

    top_text = Path(top_path).read_text(encoding="utf-8")
    gro_text = Path(gro_path).read_text(encoding="utf-8")

    assert "[ defaults ]" in top_text
    assert "[ atomtypes ]" in top_text
    assert "[ pairs ]" in top_text
    assert "[ dihedrals ]" in top_text
    assert "0.15000" in top_text
    assert "83680.000000" in top_text
    assert "60.000000" in top_text
    assert "83.680000" in top_text
    assert "6.2760000" in top_text
    assert "4.6024000" in top_text
    assert "0.26726962" in top_text
    assert "0.4184000" in top_text
    assert "    1MOL" in gro_text
    assert "    4" in gro_text.splitlines()[1]

    bond_row = gromacs_row_by_atoms(top_path, "bonds", (1, 2))
    angle_row = gromacs_row_by_atoms(top_path, "angles", (1, 2, 3))
    proper_row = gromacs_row_by_atoms(top_path, "dihedrals", (1, 2, 3, 4))
    improper_row = [row for row in gromacs_section_rows(top_path, "dihedrals") if row[:4] == ["1", "2", "3", "4"] and row[4] == "4"][0]
    assert float(bond_row[3]) == pytest.approx(0.15000)
    assert float(bond_row[4]) == pytest.approx(2.0 * 100.0 * 4.184 * 100.0)
    assert float(angle_row[4]) == pytest.approx(60.0)
    assert float(angle_row[5]) == pytest.approx(2.0 * 10.0 * 4.184)
    assert proper_row[4] == "1"
    assert float(proper_row[5]) == pytest.approx(180.0)
    assert float(proper_row[6]) == pytest.approx(1.5 * 4.184)
    assert proper_row[7] == "2"
    assert improper_row[4] == "4"
    assert float(improper_row[6]) == pytest.approx(1.1 * 4.184)


def test_write_gromacs_files_omits_missing_parameter_rows_with_warnings(tmp_path: Path):
    parameter_set = _make_parameter_set()
    parameter_set.bonds[0].kBond = None
    parameter_set.nonbonds[0].rmin_half = None
    atoms = _make_atoms(
        ["C", "C", "C", "C"],
        [
            (0.0, 0.0, 0.0),
            (1.5, 0.0, 0.0),
            (3.0, 0.0, 0.0),
            (4.5, 1.0, 0.0),
        ],
    )

    top_path, gro_path, meta = write_gromacs_files(parameter_set, atoms, str(tmp_path / "missing"))

    assert Path(top_path).is_file()
    assert Path(gro_path).is_file()
    assert meta["omitted_counts"]["atomtypes"] == 1
    assert meta["omitted_counts"]["bonds"] == 1
    assert len(meta["warnings"]) >= 2

    top_text = Path(top_path).read_text(encoding="utf-8")
    assert "omitted from [ atomtypes ]" in top_text
    assert "omitted from [ bonds ]" in top_text


def test_write_amber_files_rewrites_atom_types_and_mass_section(tmp_path: Path):
    parameter_set = _make_parameter_set()
    input_mol2 = tmp_path / "input.mol2"
    input_mol2.write_text(
        "\n".join(
            [
                "@<TRIPOS>MOLECULE",
                "TEST",
                "4 3 0 0 0",
                "SMALL",
                "USER_CHARGES",
                "@<TRIPOS>ATOM",
                "1 C1 1.0000 0.0000 0.0000 ca 1 RES -0.1000",
                "2 C2 0.0000 0.0000 0.0000 ca 1 RES 0.0500",
                "3 C3 0.0000 0.0000 1.0000 ca 1 RES 0.0500",
                "4 C4 -1.0000 0.0000 1.0000 ca 1 RES 0.0000",
                "@<TRIPOS>BOND",
                "1 1 2 1",
                "2 2 3 1",
                "3 3 4 1",
                "",
            ]
        ),
        encoding="utf-8",
    )

    maple_mol2, maple_frcmod = write_amber_files(parameter_set, str(input_mol2), str(tmp_path / "demo"))

    mol2_text = Path(maple_mol2).read_text(encoding="utf-8")
    frcmod_text = Path(maple_frcmod).read_text(encoding="utf-8")

    assert Path(maple_mol2).is_file()
    assert Path(maple_frcmod).is_file()
    assert "1 C1 1.0000 0.0000 0.0000 Z0 1 RES -0.1000" in mol2_text
    assert "2 C2 0.0000 0.0000 0.0000 Z1 1 RES 0.0500" in mol2_text
    assert "3 C3 0.0000 0.0000 1.0000 Z2 1 RES 0.0500" in mol2_text
    assert "4 C4 -1.0000 0.0000 1.0000 Z3 1 RES 0.0000" in mol2_text
    assert "MASS\nZ0      12.010\nZ1      12.010\nZ2      12.010\nZ3      12.010\n" in frcmod_text
    assert "Z0-Z1" in frcmod_text
    assert "Z0  " in frcmod_text


def test_write_amber_files_splits_same_old_type_instances_into_maple_terms(tmp_path: Path):
    parameter_set = _make_split_parameter_set()
    input_mol2 = tmp_path / "input.mol2"
    _write_input_mol2(input_mol2, parameter_set)

    _maple_mol2, maple_frcmod = write_amber_files(parameter_set, str(input_mol2), str(tmp_path / "split"))

    parsed = parse_frcmod(maple_frcmod)
    assert set(parsed.mass_params) == {"Z0", "Z1", "Z2", "Z3", "Z4"}
    assert set(parsed.nonbond_params) == {"Z0", "Z1", "Z2", "Z3", "Z4"}
    assert parsed.bond_params[("Z0", "Z1")] == pytest.approx((100.0, 1.50))
    assert parsed.bond_params[("Z1", "Z2")] == pytest.approx((101.0, 1.51))
    assert parsed.angle_params[("Z0", "Z1", "Z2")][0] == pytest.approx(10.0)
    assert parsed.angle_params[("Z1", "Z2", "Z3")][0] == pytest.approx(11.0)
    assert parsed.dihedral_params[("Z0", "Z1", "Z2", "Z3")][0].kPhi == pytest.approx(1.5)
    assert parsed.dihedral_params[("Z1", "Z2", "Z3", "Z4")][0].kPhi == pytest.approx(2.5)
    assert all("ca" not in atom_type for key in parsed.dihedral_params for atom_type in key)


def test_write_amber_files_rejects_incomplete_remapped_terms(tmp_path: Path):
    parameter_set = _make_parameter_set()
    parameter_set.bonds[0].kBond = None
    input_mol2 = tmp_path / "input.mol2"
    _write_input_mol2(input_mol2, parameter_set)

    with pytest.raises(ValueError, match="BOND.*1-2.*kBond/rEq"):
        write_amber_files(parameter_set, str(input_mol2), str(tmp_path / "incomplete"))


def test_write_amber_files_rejects_missing_mass_or_nonbond_for_maple_type(tmp_path: Path):
    parameter_set = _make_parameter_set()
    parameter_set.frcmod.mass_params.clear()
    input_mol2 = tmp_path / "input.mol2"
    _write_input_mol2(input_mol2, parameter_set)

    with pytest.raises(ValueError, match="MASS.*ca"):
        write_amber_files(parameter_set, str(input_mol2), str(tmp_path / "missing_mass"))

    parameter_set = _make_parameter_set()
    parameter_set.nonbonds[0].epsilon = None
    _write_input_mol2(input_mol2, parameter_set)

    with pytest.raises(ValueError, match="NONBON.*atom 1"):
        write_amber_files(parameter_set, str(input_mol2), str(tmp_path / "missing_nonbon"))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
