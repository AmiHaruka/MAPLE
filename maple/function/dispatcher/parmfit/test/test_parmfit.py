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

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maple.function.dispatcher.dispatcher import Dispatcher
from maple.function.dispatcher.parmfit import Parmfit
from maple.function.dispatcher.parmfit.utils.readparm import (
    build_correction_parameter_set,
    parse_frcmod,
    parse_mol2,
)
from maple.function.read.filereader import PDBReader
from maple.function.read.filereader.parmfit_reader import ParmfitReader
from maple.function.read.input_reader import InputReader
from maple.function.utility import Molecules

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


PDB_TEXT = """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.400   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.100   1.200   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.700   2.300   0.000  1.00  0.00           O
END
"""


PDB_NMR_TEXT = """MODEL        1
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.400   0.000   0.000  1.00  0.00           C
TER
ENDMDL
MODEL        2
ATOM      1  N   ALA A   1       0.100   0.200   0.300  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.500   0.200   0.300  1.00  0.00           C
CONECT    1    2
ENDMDL
END
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


def test_command_control_keeps_parmfit_input_inline_params_raw():
    cc = CommandControl.from_settings(
        ["#parmfit(input=recipe.parmfit,target=INLINE,cmo=-1 6)"]
    )

    assert cc.params["input"] == "recipe.parmfit"
    assert cc.params["target"] == "INLINE"
    assert cc.params["cmo"] == "-1 6"


def test_parmfit_reader_reads_key_value_config_relative_paths(tmp_path: Path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    mol2_path = config_dir / "lig.mol2"
    cfmol2_path = config_dir / "HEM_ff.mol2"
    mol2_path.write_text("@<TRIPOS>MOLECULE\nLIG\n", encoding="utf-8")
    cfmol2_path.write_text("@<TRIPOS>MOLECULE\nHEM\n", encoding="utf-8")
    config_path = config_dir / "recipe.parmfit"
    config_path.write_text(
        "# comment\nmethod=correction\nmol2=lig.mol2\ncfmol2=HEM_ff.mol2\nnproc=8\n",
        encoding="utf-8",
    )

    params = ParmfitReader(str(config_path))

    assert params["method"] == "correction"
    assert params["mol2"] == str(mol2_path.resolve())
    assert params["cfmol2"] == str(cfmol2_path.resolve())
    assert params["nproc"] == 8


def test_parmfit_reader_accepts_plain_text_config_without_parmfit_suffix(tmp_path: Path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    mol2_path = config_dir / "lig.mol2"
    mol2_path.write_text("@<TRIPOS>MOLECULE\nLIG\n", encoding="utf-8")
    config_path = config_dir / "recipe.txt"
    config_path.write_text("method=correction\nmol2=lig.mol2\n", encoding="utf-8")

    params = ParmfitReader(str(config_path))

    assert params["method"] == "correction"
    assert params["mol2"] == str(mol2_path.resolve())


def test_parmfit_reader_rejects_frcmod_key(tmp_path: Path):
    config_path = tmp_path / "bad.parmfit"
    config_path.write_text("method=correction\nfrcmod=lig.frcmod\n", encoding="utf-8")

    with pytest.raises(ValueError, match="frcmod"):
        ParmfitReader(str(config_path))


def test_input_reader_keeps_parmfit_external_config_as_raw_params(tmp_path: Path):
    pdb_path = tmp_path / "fixture.pdb"
    pdb_path.write_text(PDB_TEXT, encoding="utf-8")
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    cfmol2_path = config_dir / "HEM_ff.mol2"
    cfmol2_path.write_text("@<TRIPOS>MOLECULE\nHEM\n", encoding="utf-8")
    config_path = config_dir / "p450.parmfit"
    config_path.write_text(
        "\n".join(
            [
                "# parmfit recipe",
                "method=abinitio",
                "target=A462",
                "cmo=-1 6 3",
                "cfmol2=HEM_ff.mol2",
                "nproc=8",
                "mem=24",
                "route=Guess=Read",
                "prom=ff14sb",
                "watm=tip3p",
                "bonded=seminario",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    input_path = tmp_path / "job.inp"
    input_path.write_text(
        "#model=uma(size=uma-s-1p1)\n#parmfit(input=configs/p450.parmfit)\n#device=gpu0\n\nPDB fixture.pdb\n",
        encoding="utf-8",
    )

    reader = InputReader()
    atoms = reader(str(input_path), str(tmp_path / "job.out"))

    assert atoms.get_chemical_symbols() == ["H"]
    params = reader.command_control.params
    assert params["method"] == "abinitio"
    assert params["input"] == "configs/p450.parmfit"
    assert "target" not in params
    assert "cmo" not in params
    assert "cfmol2" not in params
    assert params["pdb"] == str(pdb_path.resolve())


def test_parmfit_external_config_ignores_inline_parmfit_values(tmp_path: Path, monkeypatch):
    pdb_path = tmp_path / "fixture.pdb"
    pdb_path.write_text(PDB_TEXT, encoding="utf-8")
    config_path = tmp_path / "recipe.parmfit"
    config_path.write_text("method=abinitio\ntarget=FROMFILE\ncmo=0 1\n", encoding="utf-8")
    input_path = tmp_path / "job.inp"
    input_path.write_text(
        "#parmfit(input=recipe.parmfit,target=INLINE,cmo=-1 6)\n\nPDB fixture.pdb\n",
        encoding="utf-8",
    )

    reader = InputReader()
    atoms = reader(str(input_path), str(tmp_path / "job.out"))
    observed = {}

    class FakeAbinitio:
        def __init__(self, output, atoms, params):
            del output, atoms
            observed["params"] = dict(params)

        def run(self):
            return "ok"

    import maple.function.dispatcher.parmfit.abinitio as abinitio_package

    monkeypatch.setattr(abinitio_package, "Abinitio", FakeAbinitio)

    result = Parmfit(
        output=str(tmp_path / "job.out"),
        atoms=atoms,
        method=reader.command_control.params.get("method"),
        params=reader.command_control.params,
        extra={"input_path": str(input_path)},
    ).run()

    assert result == "ok"
    assert observed["params"]["target"] == "FROMFILE"
    assert observed["params"]["cmo"] == "0 1"
    assert observed["params"]["pdb"] == str(pdb_path.resolve())
    assert "input" not in observed["params"]


def test_parmfit_external_config_rejects_pdb_key(tmp_path: Path):
    pdb_path = tmp_path / "fixture.pdb"
    pdb_path.write_text(PDB_TEXT, encoding="utf-8")
    config_path = tmp_path / "bad.parmfit"
    config_path.write_text("method=abinitio\npdb=fixture.pdb\ntarget=A1\n", encoding="utf-8")
    input_path = tmp_path / "job.inp"
    input_path.write_text("#parmfit(input=bad.parmfit)\n\nPDB fixture.pdb\n", encoding="utf-8")

    reader = InputReader()
    atoms = reader(str(input_path), str(tmp_path / "job.out"))

    with pytest.raises(ValueError, match="PDB <path> block"):
        Parmfit(
            output=str(tmp_path / "job.out"),
            atoms=atoms,
            method=reader.command_control.params.get("method"),
            params=reader.command_control.params,
            extra={"input_path": str(input_path)},
        ).run()


def test_parmfit_external_correction_config_requires_method_hint(tmp_path: Path, monkeypatch):
    mol2_path, _frcmod_path = _write_fixture_files(tmp_path)
    config_path = tmp_path / "correction.parmfit"
    config_path.write_text("method=correction\nmol2=fixture.mol2\n", encoding="utf-8")
    input_path = tmp_path / "job.inp"
    input_path.write_text(
        "#parmfit(method=correction,input=correction.parmfit)\n\n0 1\nH 0.0 0.0 0.0\n",
        encoding="utf-8",
    )
    observed = {}

    class FakeCorrection:
        def __init__(self, output, atoms, params):
            del output, atoms
            observed["params"] = dict(params)

        def run(self):
            return "ok"

    import maple.function.dispatcher.parmfit.correction as correction_package

    monkeypatch.setattr(correction_package, "Correction", FakeCorrection)

    reader = InputReader()
    atoms = reader(str(input_path), str(tmp_path / "job.out"))
    result = Parmfit(
        output=str(tmp_path / "job.out"),
        atoms=atoms,
        method=reader.command_control.params.get("method"),
        params=reader.command_control.params,
        extra={"input_path": str(input_path)},
    ).run()

    assert result == "ok"
    assert observed["params"]["method"] == "correction"
    assert observed["params"]["mol2"] == str(mol2_path.resolve())
    assert "frcmod" not in observed["params"]


def test_parmfit_external_config_rejects_method_mismatch(tmp_path: Path):
    pdb_path = tmp_path / "fixture.pdb"
    pdb_path.write_text(PDB_TEXT, encoding="utf-8")
    mol2_path = tmp_path / "fixture.mol2"
    mol2_path.write_text("@<TRIPOS>MOLECULE\nLIG\n", encoding="utf-8")
    config_path = tmp_path / "correction.parmfit"
    config_path.write_text("method=correction\nmol2=fixture.mol2\n", encoding="utf-8")
    input_path = tmp_path / "job.inp"
    input_path.write_text("#parmfit(input=correction.parmfit)\n\nPDB fixture.pdb\n", encoding="utf-8")

    reader = InputReader()
    atoms = reader(str(input_path), str(tmp_path / "job.out"))

    with pytest.raises(ValueError, match="does not match"):
        Parmfit(
            output=str(tmp_path / "job.out"),
            atoms=atoms,
            method=reader.command_control.params.get("method"),
            params=reader.command_control.params,
            extra={"input_path": str(input_path)},
        ).run()


def test_input_reader_injects_abinitio_pdb_from_pdb_block(tmp_path: Path):
    pdb_path = tmp_path / "fixture.pdb"
    pdb_path.write_text(PDB_TEXT, encoding="utf-8")
    input_path = tmp_path / "job.inp"
    input_path.write_text(
        "#parmfit(method=abinitio,target=A1,cmo=0 1)\n\nPDB fixture.pdb\n",
        encoding="utf-8",
    )

    reader = InputReader()
    atoms = reader(str(input_path), str(tmp_path / "job.out"))

    assert len(atoms) == 1
    assert atoms.get_chemical_symbols() == ["H"]
    assert reader.command_control.params["pdb"] == str(pdb_path.resolve())
    assert "_pdb_source" not in reader.command_control.params


def test_abinitio_pdb_line_charge_mult_does_not_override_cmo(tmp_path: Path, monkeypatch):
    pdb_path = tmp_path / "fixture.pdb"
    pdb_path.write_text(PDB_TEXT, encoding="utf-8")
    input_path = tmp_path / "job.inp"
    input_path.write_text(
        "#parmfit(method=abinitio,target=A1,cmo=-1 6)\n\nPDB 0 1 FIXTURE.PDB\n",
        encoding="utf-8",
    )
    observed = {}

    class FakeAbinitio:
        def __init__(self, output, atoms, params):
            del output
            observed["atoms"] = atoms
            observed["params"] = dict(params)

        def run(self):
            return "ok"

    import maple.function.dispatcher.parmfit.abinitio as abinitio_package

    monkeypatch.setattr(abinitio_package, "Abinitio", FakeAbinitio)

    reader = InputReader()
    atoms = reader(str(input_path), str(tmp_path / "job.out"))
    assert reader.command_control.params["pdb"] == str(pdb_path.resolve())
    assert "charge" not in atoms.info
    assert "mult" not in atoms.info

    result = Parmfit(
        output=str(tmp_path / "job.out"),
        atoms=atoms,
        method="abinitio",
        params=reader.command_control.params,
    ).run()

    assert result == "ok"
    assert observed["atoms"].info["charge"] == -1
    assert observed["atoms"].info["mult"] == 6
    assert observed["atoms"].info["spin"] == 2.5
    assert observed["params"]["pdb"] == str(pdb_path.resolve())


def test_pdb_reader_reads_single_model_atoms(tmp_path: Path):
    pdb_path = tmp_path / "fixture.pdb"
    pdb_path.write_text(PDB_TEXT, encoding="utf-8")

    atoms = PDBReader("PDB fixture.pdb", base_dir=str(tmp_path))

    assert atoms.get_chemical_symbols() == ["N", "C", "C", "O"]
    np.testing.assert_allclose(atoms.get_positions()[1], [1.4, 0.0, 0.0])


def test_pdb_reader_stores_charge_and_multiplicity(tmp_path: Path):
    pdb_path = tmp_path / "fixture.pdb"
    pdb_path.write_text(PDB_TEXT, encoding="utf-8")

    atoms = PDBReader("PDB -1 2 fixture.pdb", base_dir=str(tmp_path))

    assert atoms.info["charge"] == -1
    assert atoms.info["mult"] == 2
    assert atoms.info["spin"] == 0.5


def test_pdb_reader_rejects_trailing_charge_multiplicity(tmp_path: Path):
    pdb_path = tmp_path / "fixture.pdb"
    pdb_path.write_text(PDB_TEXT, encoding="utf-8")

    with pytest.raises(ValueError, match="PDB <charge> <mult> <path>"):
        PDBReader("fixture.pdb -1 2", base_dir=str(tmp_path))


def test_pdb_reader_reads_multi_model_as_molecules(tmp_path: Path):
    pdb_path = tmp_path / "nmr.pdb"
    pdb_path.write_text(PDB_NMR_TEXT, encoding="utf-8")

    molecules = PDBReader("PDB 1 3 nmr.pdb", base_dir=str(tmp_path))

    assert isinstance(molecules, Molecules)
    assert len(molecules.multiatoms) == 2
    assert molecules.multiatoms[0].get_chemical_symbols() == ["N", "C"]
    assert molecules.multiatoms[0].info["charge"] == 1
    assert molecules.multiatoms[1].info["mult"] == 3
    assert molecules.multiatoms[1].info["spin"] == 1.0
    np.testing.assert_allclose(molecules.multiatoms[1].get_positions()[0], [0.1, 0.2, 0.3])


def test_input_reader_reads_pdb_multi_model_for_regular_task(tmp_path: Path):
    pdb_path = tmp_path / "nmr.pdb"
    pdb_path.write_text(PDB_NMR_TEXT, encoding="utf-8")
    input_path = tmp_path / "job.inp"
    input_path.write_text("#sp\n\nPDB 0 1 nmr.pdb\n", encoding="utf-8")

    atoms = InputReader()(str(input_path), str(tmp_path / "job.out"))

    assert isinstance(atoms, Molecules)
    assert len(atoms.multiatoms) == 2
    assert atoms.multiatoms[0].get_chemical_symbols() == ["N", "C"]
    assert atoms.multiatoms[0].info["charge"] == 0
    assert atoms.multiatoms[1].info["mult"] == 1
    np.testing.assert_allclose(atoms.multiatoms[1].get_positions()[1], [1.5, 0.2, 0.3])


def test_parmfit_abinitio_requires_pdb_path(tmp_path: Path):
    with pytest.raises(ValueError, match="requires a PDB block"):
        Parmfit(
            output=str(tmp_path / "missing_pdb.out"),
            atoms=_build_atoms(),
            method="abinitio",
            params={"method": "abinitio", "target": "A1", "cmo": "0 1"},
        ).run()


def test_parmfit_abinitio_accepts_runtime_pdb_param(tmp_path: Path, monkeypatch):
    pdb_path = tmp_path / "runtime.pdb"
    pdb_path.write_text(PDB_TEXT, encoding="utf-8")
    observed = {}

    class FakeAbinitio:
        def __init__(self, output, atoms, params):
            del output
            observed["atoms"] = atoms
            observed["params"] = dict(params)

        def run(self):
            return "ok"

    import maple.function.dispatcher.parmfit.abinitio as abinitio_package

    monkeypatch.setattr(abinitio_package, "Abinitio", FakeAbinitio)

    result = Parmfit(
        output=str(tmp_path / "runtime.out"),
        atoms=_build_atoms(),
        method="abinitio",
        params={"method": "abinitio", "pdb": str(pdb_path), "target": "A1", "cmo": "0 1"},
    ).run()

    assert result == "ok"
    assert observed["params"]["pdb"] == str(pdb_path)
    assert observed["atoms"].info["charge"] == 0
    assert observed["atoms"].info["mult"] == 1


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


def test_unmatched_improper_candidates_are_tracked_but_not_exported(tmp_path: Path):
    mol2_path = tmp_path / "star.mol2"
    frcmod_path = tmp_path / "star.frcmod"
    mol2_path.write_text(
        """@<TRIPOS>MOLECULE
STAR
5 4 0 0 0
SMALL
USER_CHARGES
@<TRIPOS>ATOM
1 C1 0.0 0.0 0.0 c3 1 MOL 0.0
2 H2 1.0 0.0 0.0 h1 1 MOL 0.0
3 H3 0.0 1.0 0.0 h1 1 MOL 0.0
4 H4 0.0 0.0 1.0 h1 1 MOL 0.0
5 H5 -1.0 0.0 0.0 h1 1 MOL 0.0
@<TRIPOS>BOND
1 1 2 1
2 1 3 1
3 1 4 1
4 1 5 1
""",
        encoding="utf-8",
    )
    frcmod_path.write_text(
        """Remark line goes here
MASS
c3 12.010 0.0
h1 1.008 0.0

BOND
c3-h1  300.00  1.09

ANGLE
h1-c3-h1  35.00  109.50

DIHE

IMPROPER

NONBON
c3  1.91  0.11
h1  1.20  0.02
""",
        encoding="utf-8",
    )
    atoms = Atoms(
        symbols=["C", "H", "H", "H", "H"],
        positions=[
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (-1.0, 0.0, 0.0),
        ],
    )

    result = build_correction_parameter_set(atoms, str(mol2_path), str(frcmod_path))

    assert result.impropers == []
    assert len(result.unmatched_impropers) == 4


def test_parmfit_resolves_paths_without_input_reader_changes(tmp_path: Path, monkeypatch):
    mol2_path, frcmod_path = _write_fixture_files(tmp_path)
    _patch_parmchk2_from_fixture(monkeypatch, frcmod_path)
    _patch_correction_run(monkeypatch)
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
    assert parmfit.params["frcmod"] == "fixture.frcmod"
    assert parmfit.atoms.info["parmfit"]["method"] == "correction"


def test_dispatcher_enters_parmfit_branch(tmp_path: Path, monkeypatch):
    _mol2_path, frcmod_path = _write_fixture_files(tmp_path)
    _patch_parmchk2_from_fixture(monkeypatch, frcmod_path)
    _patch_correction_run(monkeypatch)
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
    assert "PARMFIT CORRECTION RESULT" in content


def _patch_parmchk2_from_fixture(monkeypatch, frcmod_path: Path) -> None:
    def fake_run_parmchk2(input_file, cfg, ifmol2, workdir):
        del input_file, ifmol2
        output_path = Path(workdir) / f"{cfg['residue_name']}.frcmod"
        output_path.write_text(frcmod_path.read_text(encoding="utf-8"), encoding="utf-8")
        return SimpleNamespace(frcmod_path=str(output_path))

    monkeypatch.setattr(
        "maple.function.dispatcher.parmfit.correction.parameters.interface.run_parmchk2",
        fake_run_parmchk2,
    )


def _patch_correction_run(monkeypatch) -> None:
    def fake_run(self):
        self.atoms.info["parmfit"] = {"method": "correction"}
        Path(self.output).write_text("PARMFIT CORRECTION RESULT\n", encoding="utf-8")
        return SimpleNamespace()

    monkeypatch.setattr(
        "maple.function.dispatcher.parmfit.correction.correction.Correction.run",
        fake_run,
    )


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
