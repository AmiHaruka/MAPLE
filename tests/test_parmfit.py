from pathlib import Path
import importlib.util
import sys
from types import SimpleNamespace
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maple.function.dispatcher.dispatcher import Dispatcher
from maple.function.dispatcher.parmfit import Parmfit
from maple.function.dispatcher.parmfit.utils.readparm import (
    build_correction_parameter_set,
    parse_frcmod,
    parse_mol2,
)

_cc_spec = importlib.util.spec_from_file_location(
    "test_command_control_module", ROOT / "maple/function/read/command_control.py"
)
assert _cc_spec is not None and _cc_spec.loader is not None
_cc_module = importlib.util.module_from_spec(_cc_spec)
_cc_spec.loader.exec_module(_cc_module)
CommandControl = _cc_module.CommandControl


MOL2_TEXT = """@<TRIPOS>MOLECULE
TEST
5 4 0 0 0
SMALL
USER_CHARGES

@<TRIPOS>ATOM
1 A1 0.0 0.0 0.0 c1 1 MOL 0.10
2 A2 0.0 0.0 0.0 n2 1 MOL -0.20
3 A3 0.0 0.0 0.0 c3 1 MOL 0.30
4 A4 0.0 0.0 0.0 o4 1 MOL -0.40
5 A5 0.0 0.0 0.0 h5 1 MOL 0.20
@<TRIPOS>BOND
1 1 2 1
2 2 3 1
3 3 4 2
4 3 5 1
"""


FRCMOD_TEXT = """Remark line goes here
MASS
c1 12.010 0.0
n2 14.010 0.0
c3 12.010 0.0
o4 16.000 0.0
h5 1.008 0.0

BOND
c1-n2  300.00  1.45
n2-c3  350.00  1.32
c3-o4  500.00  1.23
c3-h5  250.00  1.09

ANGLE
c1-n2-c3  50.00  120.00
n2-c3-o4  70.00  123.00
n2-c3-h5  35.00  109.00
o4-c3-h5  45.00  110.00

DIHE
o4-c3-n2-c1  1  0.50  180.0  2.0
o4-c3-n2-c1  1  1.25    0.0  3.0
h5-c3-n2-X   1  0.80  180.0  1.0

IMPROPER
X-o4-c3-h5  1.10  180.0  2.0

NONBON
c1  1.90  0.10
n2  1.82  0.17
c3  1.91  0.11
o4  1.66  0.21
h5  1.20  0.02
"""


def _write_fixture_files(tmp_path: Path) -> tuple[Path, Path]:
    mol2_path = tmp_path / "fixture.mol2"
    frcmod_path = tmp_path / "fixture.frcmod"
    mol2_path.write_text(MOL2_TEXT, encoding="utf-8")
    frcmod_path.write_text(FRCMOD_TEXT, encoding="utf-8")
    return mol2_path, frcmod_path


def _build_atoms() -> Atoms:
    return Atoms(
        symbols=["C", "N", "C", "O", "H"],
        positions=[
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (3.0, 0.0, 0.0),
            (2.0, 1.0, 0.0),
        ],
    )


def test_command_control_accepts_parmfit():
    cc = CommandControl.from_settings(
        ["#model=aimnet2", "#parmfit(method=correction,mol2=a.mol2,frcmod=b.frcmod)"]
    )
    assert cc.task == "parmfit"
    assert cc.params["method"] == "correction"
    assert cc.params["mol2"] == "a.mol2"
    assert cc.params["frcmod"] == "b.frcmod"


def test_mol2_and_frcmod_parsing_and_assignment(tmp_path: Path):
    mol2_path, frcmod_path = _write_fixture_files(tmp_path)
    mol2 = parse_mol2(str(mol2_path))
    frcmod = parse_frcmod(str(frcmod_path))

    assert len(mol2.atoms) == 5
    assert len(mol2.bonds) == 4
    assert mol2.atoms[0].atom_type == "c1"
    assert ("o4", "c3", "n2", "c1") in frcmod.dihedral_params
    assert len(frcmod.dihedral_params[("o4", "c3", "n2", "c1")]) == 2
    assert frcmod.nonbond_params["o4"] == (1.66, 0.21)

    result = build_correction_parameter_set(_build_atoms(), str(mol2_path), str(frcmod_path))

    assert len(result.bonds) == 4
    assert len(result.angles) == 4
    assert len(result.dihedrals) == 2
    assert len(result.impropers) == 1
    assert len(result.nonbonds) == 5
    assert not result.unmatched_bonds
    assert not result.unmatched_angles
    assert not result.unmatched_dihedrals
    assert not result.unmatched_nonbonds
    assert any(len(dihedral.terms) == 2 for dihedral in result.dihedrals)
    assert result.impropers[0].atom_types[2] == "c3"


def test_parmfit_resolves_paths_without_input_reader_changes(tmp_path: Path):
    mol2_path, frcmod_path = _write_fixture_files(tmp_path)
    input_path = tmp_path / "job.inp"
    input_path.write_text("#parmfit(method=correction,mol2=fixture.mol2,frcmod=fixture.frcmod)\n", encoding="utf-8")
    output_path = tmp_path / "job.out"
    output_path.write_text("", encoding="utf-8")

    parmfit = Parmfit(
        output=str(output_path),
        atoms=_build_atoms(),
        method="correction",
        params={"method": "correction", "mol2": "fixture.mol2", "frcmod": "fixture.frcmod"},
        extra={"input_path": str(input_path)},
    )
    parmfit.run()

    assert parmfit.params["mol2"] == str(mol2_path)
    assert parmfit.params["frcmod"] == str(frcmod_path)
    assert parmfit.atoms.info["parmfit"]["method"] == "correction"


def test_dispatcher_enters_parmfit_branch(tmp_path: Path):
    _write_fixture_files(tmp_path)
    output_path = tmp_path / "dispatch.out"
    output_path.write_text("", encoding="utf-8")
    commandcontrol = SimpleNamespace(
        params={"method": "correction", "mol2": "fixture.mol2", "frcmod": "fixture.frcmod"}
    )

    dispatcher = Dispatcher()
    dispatcher(
        commandcontrol,
        "parmfit",
        _build_atoms(),
        str(output_path),
        extra={"parmfit": {"input_path": str(tmp_path / "job.inp")}},
    )

    content = output_path.read_text(encoding="utf-8")
    assert "Parmfit Correction Summary" in content


def test_correction_requires_aux_files(tmp_path: Path):
    output_path = tmp_path / "missing.out"
    output_path.write_text("", encoding="utf-8")
    parmfit = Parmfit(
        output=str(output_path),
        atoms=_build_atoms(),
        method="correction",
        params={"method": "correction", "mol2": "fixture.mol2"},
        extra={"input_path": str(tmp_path / "job.inp")},
    )

    with pytest.raises(ValueError, match="requires the following file inputs"):
        parmfit.run()
