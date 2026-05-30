from pathlib import Path

import pytest
from ase import Atoms

from maple.function.engine import engine as MapleEngine
from maple.function.read.command_control import CommandControl
from maple.function.read.post_process.explicit_solvent.solvate import (
    DATA_DIR,
    ExplicitSolv,
    WATER_MOLAR_MASS_G_MOL,
    _mass_density_g_ml,
)


def _water_records(residue_id: int, x: float) -> list[str]:
    return [
        f"HETATM    {3 * residue_id - 2:1d}  O   HOH  {residue_id:4d}      "
        f"{x:6.3f}   5.000   5.000  1.00  0.00           O",
        f"HETATM    {3 * residue_id - 1:1d}  H1  HOH  {residue_id:4d}      "
        f"{x + 0.957:6.3f}   5.000   5.000  1.00  0.00           H",
        f"HETATM    {3 * residue_id:1d}  H2  HOH  {residue_id:4d}      "
        f"{x - 0.239:6.3f}   5.927   5.000  1.00  0.00           H",
        "TER",
    ]


def _write_water_box_and_density(path: Path, molecules: int = 2) -> float:
    lines = [
        "CRYST1   20.000   20.000   20.000  90.00  90.00  90.00 P 1           1",
    ]
    for idx in range(molecules):
        lines.extend(_water_records(idx + 1, 5.0 + 10.0 * idx))
    lines.append("END")
    lines.append("")
    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    return _mass_density_g_ml(molecules, WATER_MOLAR_MASS_G_MOL, 20.0 ** 3)


def _gromacs_water_record(serial: int, atom_name: str, residue_id: int, pos) -> str:
    return (
        f"HETATM{serial:5d} {atom_name:<4s} SOL A{residue_id:4d}    "
        f"{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}  1.00  0.00"
    )


def _solvent_molecule_count(atoms: Atoms) -> int:
    molecule_ids = atoms.arrays["maple_molecule_id"]
    return len({int(mid) for mid in molecule_ids.tolist() if mid >= 0})


def test_command_control_accepts_custom_solvent_pdb_path(tmp_path):
    out = tmp_path / "parse.out"

    cc = CommandControl.from_settings(
        ["#solv(explicit=water,solvent_pdb=Boxes/WaterFixed.PDB)"],
        output_path=str(out),
    )

    assert cc.get("solv")["solvent_pdb"] == "Boxes/WaterFixed.PDB"


@pytest.mark.parametrize(
    "setting",
    [
        "#solv(explicit=water,solvent_pdb=)",
        "#solv(explicit=water,solvent_pdb=1)",
    ],
)
def test_command_control_rejects_invalid_solvent_pdb_path(setting, tmp_path):
    out = tmp_path / "parse.out"

    with pytest.raises(ValueError, match="solvent_pdb must be a non-empty path string"):
        CommandControl.from_settings([setting], output_path=str(out))


def test_command_control_rejects_solvent_pdb_for_implicit_solvation(tmp_path):
    out = tmp_path / "parse.out"

    with pytest.raises(ValueError, match="Explicit-solvent options"):
        CommandControl.from_settings(
            [
                "#solv(method=gbsa,implicit=water,experimental=true,"
                "solvent_pdb=box.pdb)"
            ],
            output_path=str(out),
        )


def test_custom_water_pdb_requires_explicit_density_when_default_mismatches(tmp_path):
    box_dir = tmp_path / "boxes"
    box_dir.mkdir()
    pdb_path = box_dir / "waterbox.pdb"
    template_density = _write_water_box_and_density(pdb_path)
    output = tmp_path / "solv.out"
    solute = Atoms("He", positions=[[0.0, 0.0, 0.0]])

    with pytest.raises(ValueError) as excinfo:
        ExplicitSolv(
            solute,
            params={
                "explicit": "water",
                "solvent_pdb": "boxes/waterbox.pdb",
                "radius": 1.0,
                "seed": 0,
            },
            device=None,
            output=str(output),
            base_dir=str(tmp_path),
        )

    message = str(excinfo.value)
    assert "Custom solvent_pdb template" in message
    assert "density" in message
    assert str(pdb_path) in message
    assert f"density={template_density:.4f}" in message
    assert "number=<int>" in message


def test_custom_water_pdb_adds_solvent_with_matching_explicit_density(tmp_path):
    box_dir = tmp_path / "boxes"
    box_dir.mkdir()
    pdb_path = box_dir / "waterbox.pdb"
    template_density = _write_water_box_and_density(pdb_path)
    output = tmp_path / "solv.out"
    solute = Atoms("He", positions=[[100.0, 100.0, 100.0]])

    atoms = ExplicitSolv(
        solute,
        params={
            "explicit": "water",
            "solvent_pdb": "boxes/waterbox.pdb",
            "density": template_density,
            "number": 1,
            "radius": 15.0,
            "randomize": False,
            "seed": 0,
        },
        device=None,
        output=str(output),
        base_dir=str(tmp_path),
    )

    assert len(atoms) == 4
    assert _solvent_molecule_count(atoms) == 1
    assert "Solvent template PDB:" in output.read_text(encoding="utf-8")


def test_custom_water_pdb_allows_number_to_bypass_default_density_mismatch(tmp_path):
    box_dir = tmp_path / "boxes"
    box_dir.mkdir()
    pdb_path = box_dir / "waterbox.pdb"
    _write_water_box_and_density(pdb_path)
    output = tmp_path / "solv.out"
    solute = Atoms("He", positions=[[100.0, 100.0, 100.0]])

    atoms = ExplicitSolv(
        solute,
        params={
            "explicit": "water",
            "solvent_pdb": "boxes/waterbox.pdb",
            "number": 1,
            "radius": 15.0,
            "randomize": False,
        },
        device=None,
        output=str(output),
        base_dir=str(tmp_path),
    )

    assert len(atoms) == 4
    assert _solvent_molecule_count(atoms) == 1
    assert "Target source: explicit molecule count (number=1)" in output.read_text(
        encoding="utf-8"
    )


def test_density_scale_changes_custom_template_target_count(tmp_path):
    box_dir = tmp_path / "boxes"
    box_dir.mkdir()
    pdb_path = box_dir / "waterbox.pdb"
    template_density = _write_water_box_and_density(pdb_path)
    output = tmp_path / "solv.out"
    solute = Atoms("He", positions=[[100.0, 100.0, 100.0]])

    ExplicitSolv(
        solute,
        params={
            "explicit": "water",
            "solvent_pdb": "boxes/waterbox.pdb",
            "density": template_density,
            "density_scale": 2.0,
            "shape": "cube",
            "box_size": 20.0,
            "randomize": False,
        },
        device=None,
        output=str(output),
        base_dir=str(tmp_path),
    )

    assert "Target solvent molecules: 4" in output.read_text(encoding="utf-8")


def test_custom_template_explicit_wrong_density_error_names_user_file(tmp_path):
    box_dir = tmp_path / "boxes"
    box_dir.mkdir()
    pdb_path = box_dir / "waterbox.pdb"
    _write_water_box_and_density(pdb_path)

    with pytest.raises(ValueError) as excinfo:
        ExplicitSolv(
            Atoms("He", positions=[[0.0, 0.0, 0.0]]),
            params={
                "explicit": "water",
                "solvent_pdb": "boxes/waterbox.pdb",
                "density": 1.0,
                "radius": 1.0,
            },
            device=None,
            output=str(tmp_path / "solv.out"),
            base_dir=str(tmp_path),
        )

    message = str(excinfo.value)
    assert "differs from requested density" in message
    assert str(pdb_path) in message


def test_builtin_water_template_still_resolves_and_builds_without_custom_gate(tmp_path):
    output = tmp_path / "solv.out"

    atoms = ExplicitSolv(
        Atoms("He", positions=[[100.0, 100.0, 100.0]]),
        params={
            "explicit": "water",
            "number": 1,
            "radius": 15.0,
            "randomize": False,
        },
        device=None,
        output=str(output),
    )

    assert len(atoms) == 4
    assert _solvent_molecule_count(atoms) == 1
    assert f"Solvent template PDB: {DATA_DIR / 'water.pdb'}" in output.read_text(
        encoding="utf-8"
    )


def test_custom_water_pdb_rejects_bad_residue_grouping(tmp_path):
    pdb_path = tmp_path / "bad_water.pdb"
    records = [
        "CRYST1   20.000   20.000   20.000  90.00  90.00  90.00 P 1           1",
    ]
    records.extend(line.replace("   2", "   1") for line in _water_records(1, 5.0))
    records.extend(line.replace("   2", "   1") for line in _water_records(2, 15.0))
    records.extend(["END", ""])
    pdb_path.write_text("\n".join(records), encoding="utf-8")

    with pytest.raises(ValueError, match="one water molecule per residue") as excinfo:
        ExplicitSolv(
            Atoms("He", positions=[[0.0, 0.0, 0.0]]),
            params={"explicit": "water", "solvent_pdb": str(pdb_path), "number": 0},
            device=None,
            output=str(tmp_path / "solv.out"),
        )

    assert str(pdb_path) in str(excinfo.value)


def test_custom_template_rejects_mixed_residue_formulas(tmp_path):
    pdb_path = tmp_path / "mixed.pdb"
    pdb_path.write_text(
        "\n".join(
            [
                "CRYST1   20.000   20.000   20.000  90.00  90.00  90.00 P 1           1",
                "HETATM    1  C   SOL     1       5.000   5.000   5.000  1.00  0.00           C",
                "TER",
                "HETATM    2  O   SOL     2      15.000   5.000   5.000  1.00  0.00           O",
                "TER",
                "END",
                "",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not a homogeneous pure-solvent template"):
        ExplicitSolv(
            Atoms("He", positions=[[0.0, 0.0, 0.0]]),
            params={"explicit": "custom", "solvent_pdb": str(pdb_path), "number": 0},
            device=None,
            output=str(tmp_path / "solv.out"),
        )


def test_custom_template_rejects_disconnected_molecules_inside_one_residue(tmp_path):
    pdb_path = tmp_path / "two_methanes_one_residue.pdb"
    pdb_path.write_text(
        "\n".join(
            [
                "CRYST1   30.000   30.000   30.000  90.00  90.00  90.00 P 1           1",
                "HETATM    1  C   MET     1       5.000   5.000   5.000  1.00  0.00           C",
                "HETATM    2  H1  MET     1       5.629   5.629   5.629  1.00  0.00           H",
                "HETATM    3  H2  MET     1       4.371   4.371   5.629  1.00  0.00           H",
                "HETATM    4  H3  MET     1       4.371   5.629   4.371  1.00  0.00           H",
                "HETATM    5  H4  MET     1       5.629   4.371   4.371  1.00  0.00           H",
                "HETATM    6  C   MET     1      20.000  20.000  20.000  1.00  0.00           C",
                "HETATM    7  H1  MET     1      20.629  20.629  20.629  1.00  0.00           H",
                "HETATM    8  H2  MET     1      19.371  19.371  20.629  1.00  0.00           H",
                "HETATM    9  H3  MET     1      19.371  20.629  19.371  1.00  0.00           H",
                "HETATM   10  H4  MET     1      20.629  19.371  19.371  1.00  0.00           H",
                "TER",
                "END",
                "",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="multiple covalent components") as excinfo:
        ExplicitSolv(
            Atoms("He", positions=[[0.0, 0.0, 0.0]]),
            params={"explicit": "custom", "solvent_pdb": str(pdb_path), "number": 1},
            device=None,
            output=str(tmp_path / "solv.out"),
        )

    assert str(pdb_path) in str(excinfo.value)


def test_custom_template_allows_boundary_wrapped_single_residue_molecule(tmp_path):
    pdb_path = tmp_path / "wrapped_water.pdb"
    template_density = _mass_density_g_ml(1, WATER_MOLAR_MASS_G_MOL, 20.0 ** 3)
    pdb_path.write_text(
        "\n".join(
            [
                "CRYST1   20.000   20.000   20.000  90.00  90.00  90.00 P 1           1",
                "HETATM    1  O   HOH     1      19.800   5.000   5.000  1.00  0.00           O",
                "HETATM    2  H1  HOH     1       0.500   5.000   5.000  1.00  0.00           H",
                "HETATM    3  H2  HOH     1      19.561   5.927   5.000  1.00  0.00           H",
                "TER",
                "END",
                "",
            ]
        ),
        encoding="utf-8",
    )

    atoms = ExplicitSolv(
        Atoms("He", positions=[[100.0, 100.0, 100.0]]),
        params={
            "explicit": "water",
            "solvent_pdb": str(pdb_path),
            "density": template_density,
            "number": 1,
            "radius": 15.0,
            "randomize": False,
        },
        device=None,
        output=str(tmp_path / "solv.out"),
    )

    assert len(atoms) == 4
    assert _solvent_molecule_count(atoms) == 1


def test_custom_template_accepts_gromacs_water_without_element_column(tmp_path):
    pdb_path = tmp_path / "gromacs_water_no_elements.pdb"
    pdb_path.write_text(
        "\n".join(
            [
                "CRYST1   20.000   20.000   20.000  90.00  90.00  90.00 P 1           1",
                _gromacs_water_record(1, "OW", 1, (5.000, 5.000, 5.000)),
                _gromacs_water_record(2, "HW1", 1, (5.957, 5.000, 5.000)),
                _gromacs_water_record(3, "HW2", 1, (4.761, 5.927, 5.000)),
                "TER",
                "END",
                "",
            ]
        ),
        encoding="utf-8",
    )

    atoms = ExplicitSolv(
        Atoms("He", positions=[[100.0, 100.0, 100.0]]),
        params={
            "explicit": "water",
            "solvent_pdb": str(pdb_path),
            "number": 1,
            "radius": 15.0,
            "randomize": False,
        },
        device=None,
        output=str(tmp_path / "solv.out"),
    )

    assert len(atoms) == 4
    assert _solvent_molecule_count(atoms) == 1


def test_custom_template_cryst1_error_names_user_file(tmp_path):
    pdb_path = tmp_path / "no_cryst1.pdb"
    pdb_path.write_text(
        "\n".join(
            [
                "HETATM    1  O   HOH     1       0.000   0.000   0.000  1.00  0.00           O",
                "HETATM    2  H1  HOH     1       0.957   0.000   0.000  1.00  0.00           H",
                "HETATM    3  H2  HOH     1      -0.239   0.927   0.000  1.00  0.00           H",
                "TER",
                "END",
                "",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as excinfo:
        ExplicitSolv(
            Atoms("He", positions=[[0.0, 0.0, 0.0]]),
            params={"explicit": "water", "solvent_pdb": str(pdb_path), "number": 0},
            device=None,
            output=str(tmp_path / "solv.out"),
        )

    assert "has no CRYST1 cell" in str(excinfo.value)
    assert str(pdb_path) in str(excinfo.value)


def test_engine_resolves_custom_solvent_pdb_relative_to_input_file(tmp_path, monkeypatch):
    box_dir = tmp_path / "boxes"
    box_dir.mkdir()
    pdb_path = box_dir / "waterbox.pdb"
    template_density = _write_water_box_and_density(pdb_path)
    input_path = tmp_path / "custom_solv.inp"
    output_path = tmp_path / "custom_solv.out"
    input_path.write_text(
        "\n".join(
            [
                "#sp",
                (
                    "#solv(explicit=water,solvent_pdb=boxes/waterbox.pdb,"
                    f"density={template_density},number=1,radius=15,randomize=false)"
                ),
                "#device=cpu",
                "",
                "He 100.0 100.0 100.0",
                "",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        MapleEngine,
        "_mlp_initiator",
        lambda self, model, device: setattr(self, "calulator", None),
    )
    monkeypatch.setattr(
        MapleEngine,
        "_jobtype_dispatcher",
        lambda self, commandcontrol, jobtype, atoms, output, extra=None: None,
    )

    runner = MapleEngine()
    runner(str(input_path), str(output_path))

    assert len(runner.atoms) == 4
    assert _solvent_molecule_count(runner.atoms) == 1
    assert f"Solvent template PDB: {pdb_path}" in output_path.read_text(
        encoding="utf-8"
    )
