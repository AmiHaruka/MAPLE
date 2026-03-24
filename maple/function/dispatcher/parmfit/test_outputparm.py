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

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maple.function.dispatcher.parmfit.utils.outputparm import write_gromacs_files
from maple.function.dispatcher.parmfit.utils.parm import Angle, Bond, Dihedral, FourierTerm, Improper, Nonbond
from maple.function.dispatcher.parmfit.utils.readparm import CorrectionParameterSet, FrcmodDB, Mol2Atom, Mol2Bond, Mol2Topology


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
        frcmod=FrcmodDB(),
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
