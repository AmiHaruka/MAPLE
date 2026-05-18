from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maple.function.dispatcher.parmfit.test.helpers.addcap_test import addcap
from maple.function.dispatcher.parmfit.test.helpers.recognize_test import recognize
from maple.function.dispatcher.parmfit.utils.context import extract_cluster, find_unique_residue
from maple.function.dispatcher.parmfit.utils.structure import parse_selector, read_pdb


def _pdb_atom(
    record: str,
    serial: int,
    name: str,
    resname: str,
    chain: str,
    resseq: int,
    x: float,
    y: float,
    z: float,
    element: str,
    icode: str = "",
) -> str:
    return (
        f"{record:<6}{serial:5d} {name:>4s} {resname:>3s} {chain:1s}{resseq:4d}{icode:1s}   "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{0.0:6.2f}          {element:>2s}\n"
    )


def _write_text(path: Path, text: str) -> str:
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_parse_selector_supports_chain_and_residue_name_forms() -> None:
    assert parse_selector("A1") == {"chain": "A", "resname": None, "resseq": 1, "icode": ""}
    assert parse_selector("A1A") == {"chain": "A", "resname": None, "resseq": 1, "icode": "A"}
    assert parse_selector("_1") == {"chain": "_", "resname": None, "resseq": 1, "icode": ""}
    assert parse_selector("SER1") == {"chain": None, "resname": "SER", "resseq": 1, "icode": ""}
    assert parse_selector("ALA336") == {"chain": None, "resname": "ALA", "resseq": 336, "icode": ""}
    assert parse_selector("SER1A") == {"chain": None, "resname": "SER", "resseq": 1, "icode": "A"}


def test_recognize_handles_insertion_code(tmp_path: Path) -> None:
    pdb = "".join(
        [
            _pdb_atom("ATOM", 1, "N", "GLY", "A", 11, 0.0, 0.0, 0.0, "N"),
            _pdb_atom("ATOM", 2, "CA", "GLY", "A", 11, 1.4, 0.0, 0.0, "C"),
            _pdb_atom("ATOM", 3, "C", "GLY", "A", 11, 2.4, 1.0, 0.0, "C"),
            _pdb_atom("ATOM", 4, "O", "GLY", "A", 11, 3.5, 1.0, 0.0, "O"),
            _pdb_atom("ATOM", 5, "N", "SER", "A", 11, 4.0, 0.0, 0.0, "N", icode="A"),
            _pdb_atom("ATOM", 6, "CA", "SER", "A", 11, 5.4, 0.0, 0.0, "C", icode="A"),
            _pdb_atom("ATOM", 7, "C", "SER", "A", 11, 6.4, 1.0, 0.0, "C", icode="A"),
            _pdb_atom("ATOM", 8, "O", "SER", "A", 11, 7.5, 1.0, 0.0, "O", icode="A"),
            "END\n",
        ]
    )
    path = _write_text(tmp_path / "icode.pdb", pdb)

    result = recognize(path, target="A11A")

    assert result["target"]["resseq"] == 11
    assert result["target"]["icode"] == "A"
    assert result["target"]["resname"] == "SER"


def test_residue_name_selector_finds_no_chain_residue_and_context(tmp_path: Path) -> None:
    pdb = "".join(
        [
            _pdb_atom("ATOM", 1, "N", "GLY", "", 1, 0.0, 0.0, 0.0, "N"),
            _pdb_atom("ATOM", 2, "CA", "GLY", "", 1, 1.4, 0.0, 0.0, "C"),
            _pdb_atom("ATOM", 3, "C", "GLY", "", 1, 2.4, 1.0, 0.0, "C"),
            _pdb_atom("ATOM", 4, "O", "GLY", "", 1, 2.4, 2.2, 0.0, "O"),
            _pdb_atom("ATOM", 5, "N", "SER", "", 2, 3.65, 1.0, 0.0, "N"),
            _pdb_atom("ATOM", 6, "CA", "SER", "", 2, 5.05, 1.0, 0.0, "C"),
            _pdb_atom("ATOM", 7, "C", "SER", "", 2, 6.05, 2.0, 0.0, "C"),
            _pdb_atom("ATOM", 8, "O", "SER", "", 2, 6.05, 3.2, 0.0, "O"),
            _pdb_atom("ATOM", 9, "CB", "SER", "", 2, 5.05, -0.2, 0.0, "C"),
            _pdb_atom("ATOM", 10, "N", "ALA", "", 3, 7.30, 2.0, 0.0, "N"),
            _pdb_atom("ATOM", 11, "CA", "ALA", "", 3, 8.70, 2.0, 0.0, "C"),
            _pdb_atom("ATOM", 12, "C", "ALA", "", 3, 9.70, 3.0, 0.0, "C"),
            _pdb_atom("ATOM", 13, "O", "ALA", "", 3, 9.70, 4.2, 0.0, "O"),
            "END\n",
        ]
    )
    path = _write_text(tmp_path / "no_chain.pdb", pdb)

    structure = read_pdb(path)
    target = find_unique_residue(structure, "SER2")
    recognized = recognize(path, target="SER2")
    capped = addcap(path, target="SER2", rn="NAA")

    assert target["chain"] == "_"
    assert target["resname"] == "SER"
    assert recognized["target"]["chain"] == "_"
    assert recognized["prev_residue"]["resname"] == "GLY"
    assert recognized["next_residue"]["resname"] == "ALA"
    assert [residue["resname"] for residue in capped["assembled_residues"]] == ["ACE", "NAA", "NME"]


def test_residue_name_selector_does_not_match_wrong_residue_name(tmp_path: Path) -> None:
    pdb = "".join(
        [
            _pdb_atom("ATOM", 1, "N", "SER", "", 2, 0.0, 0.0, 0.0, "N"),
            _pdb_atom("ATOM", 2, "CA", "SER", "", 2, 1.4, 0.0, 0.0, "C"),
            _pdb_atom("ATOM", 3, "C", "SER", "", 2, 2.4, 1.0, 0.0, "C"),
            _pdb_atom("ATOM", 4, "O", "SER", "", 2, 3.5, 1.0, 0.0, "O"),
            "END\n",
        ]
    )
    path = _write_text(tmp_path / "wrong_resname.pdb", pdb)

    with pytest.raises(ValueError, match="THR2"):
        find_unique_residue(read_pdb(path), "THR2")


def test_residue_name_selector_preserves_multiple_match_error(tmp_path: Path) -> None:
    pdb = "".join(
        [
            _pdb_atom("ATOM", 1, "N", "SER", "A", 2, 0.0, 0.0, 0.0, "N"),
            _pdb_atom("ATOM", 2, "CA", "SER", "A", 2, 1.4, 0.0, 0.0, "C"),
            _pdb_atom("ATOM", 3, "C", "SER", "A", 2, 2.4, 1.0, 0.0, "C"),
            _pdb_atom("ATOM", 4, "O", "SER", "A", 2, 3.5, 1.0, 0.0, "O"),
            _pdb_atom("ATOM", 5, "N", "SER", "B", 2, 0.0, 3.0, 0.0, "N"),
            _pdb_atom("ATOM", 6, "CA", "SER", "B", 2, 1.4, 3.0, 0.0, "C"),
            _pdb_atom("ATOM", 7, "C", "SER", "B", 2, 2.4, 4.0, 0.0, "C"),
            _pdb_atom("ATOM", 8, "O", "SER", "B", 2, 3.5, 4.0, 0.0, "O"),
            "END\n",
        ]
    )
    path = _write_text(tmp_path / "duplicate_resname.pdb", pdb)

    with pytest.raises(ValueError, match="matched multiple residues"):
        find_unique_residue(read_pdb(path), "SER2")


def test_cluster_extracts_nearby_residue_water_ion_and_ligand(tmp_path: Path) -> None:
    pdb = "".join(
        [
            _pdb_atom("ATOM", 1, "N", "ALA", "A", 11, 0.0, 0.0, 0.0, "N"),
            _pdb_atom("ATOM", 2, "CA", "ALA", "A", 11, 1.4, 0.0, 0.0, "C"),
            _pdb_atom("ATOM", 3, "C", "ALA", "A", 11, 2.4, 1.0, 0.0, "C"),
            _pdb_atom("ATOM", 4, "O", "ALA", "A", 11, 3.5, 1.0, 0.0, "O"),
            _pdb_atom("ATOM", 5, "CB", "ALA", "A", 11, 1.4, -1.2, 0.0, "C"),
            _pdb_atom("HETATM", 6, "O", "HOH", "A", 201, 1.4, 0.0, 3.0, "O"),
            _pdb_atom("HETATM", 7, "ZN", "ZN", "A", 202, 0.0, 3.0, 0.0, "ZN"),
            _pdb_atom("HETATM", 8, "C1", "LIG", "B", 153, 0.0, 0.0, 3.5, "C"),
            _pdb_atom("HETATM", 9, "O1", "LIG", "B", 153, 0.0, 0.0, 4.7, "O"),
            _pdb_atom("ATOM", 10, "N", "GLY", "A", 50, 20.0, 0.0, 0.0, "N"),
            _pdb_atom("ATOM", 11, "CA", "GLY", "A", 50, 21.4, 0.0, 0.0, "C"),
            _pdb_atom("ATOM", 12, "C", "GLY", "A", 50, 22.4, 1.0, 0.0, "C"),
            _pdb_atom("ATOM", 13, "O", "GLY", "A", 50, 23.5, 1.0, 0.0, "O"),
            "END\n",
        ]
    )
    path = _write_text(tmp_path / "cluster.pdb", pdb)

    result = extract_cluster(path, target="A11", cutoff=4.0)
    kinds = {residue["kind"] for residue in result["environment_residues"]}
    keys = {(residue["chain"], residue["resseq"]) for residue in result["environment_residues"]}

    assert kinds == {"water", "ion", "ligand"}
    assert ("A", 50) not in keys


def test_addcap_caps_non_peptide_target_with_hydrogen(tmp_path: Path) -> None:
    pdb = "".join(
        [
            _pdb_atom("HETATM", 1, "C1", "LIG", "A", 1, 0.0, 0.0, 0.0, "C"),
            _pdb_atom("HETATM", 2, "O1", "LIG", "A", 1, 1.2, 0.0, 0.0, "O"),
            _pdb_atom("HETATM", 3, "C1", "BRG", "A", 2, -1.4, 0.0, 0.0, "C"),
            "END\n",
        ]
    )
    path = _write_text(tmp_path / "ligand.pdb", pdb)

    result = addcap(path, target="A1", rn="LIGX")

    assert result["caps"] == []
    assert result["target"]["resname"] == "LIGX"
    assert len(result["assembled_residues"]) == 1
    assert len(result["target"]["atoms"]) > 2


def test_addcap_uses_ideal_caps_for_isolated_peptide_fragment(tmp_path: Path) -> None:
    pdb = "".join(
        [
            _pdb_atom("ATOM", 1, "N", "GLY", "A", 11, 0.0, 0.0, 0.0, "N"),
            _pdb_atom("ATOM", 2, "CA", "GLY", "A", 11, 1.4, 0.0, 0.0, "C"),
            _pdb_atom("ATOM", 3, "C", "GLY", "A", 11, 2.4, 1.0, 0.0, "C"),
            _pdb_atom("ATOM", 4, "O", "GLY", "A", 11, 3.5, 1.0, 0.0, "O"),
            "END\n",
        ]
    )
    path = _write_text(tmp_path / "single_residue.pdb", pdb)

    recognized = recognize(path, target="A11")
    capped = addcap(path, target="A11", rn="GLX")

    assert recognized["prev_residue"] is None
    assert recognized["next_residue"] is None
    assert [residue["resname"] for residue in capped["caps"]] == ["ACE", "NME"]
    assert [residue["resname"] for residue in capped["assembled_residues"]] == ["ACE", "GLX", "NME"]


def test_addcap_treats_missing_left_neighbor_as_open_end(tmp_path: Path) -> None:
    pdb = "".join(
        [
            _pdb_atom("ATOM", 1, "N", "GLY", "A", 11, 0.0, 0.0, 0.0, "N"),
            _pdb_atom("ATOM", 2, "CA", "GLY", "A", 11, 1.4, 0.0, 0.0, "C"),
            _pdb_atom("ATOM", 3, "C", "GLY", "A", 11, 2.4, 1.0, 0.0, "C"),
            _pdb_atom("ATOM", 4, "O", "GLY", "A", 11, 3.5, 1.0, 0.0, "O"),
            _pdb_atom("ATOM", 5, "N", "ALA", "A", 12, 3.65, 0.1, 0.0, "N"),
            _pdb_atom("ATOM", 6, "CA", "ALA", "A", 12, 5.0, 0.1, 0.0, "C"),
            _pdb_atom("ATOM", 7, "C", "ALA", "A", 12, 6.0, 1.1, 0.0, "C"),
            _pdb_atom("ATOM", 8, "O", "ALA", "A", 12, 7.1, 1.1, 0.0, "O"),
            "END\n",
        ]
    )
    path = _write_text(tmp_path / "open_left.pdb", pdb)

    recognized = recognize(path, target="A11")
    capped = addcap(path, target="A11", rn="GLX")

    assert recognized["prev_residue"] is None
    assert recognized["next_residue"] is not None
    assert [residue["resname"] for residue in capped["caps"]] == ["ACE", "NME"]
    assert [residue["resname"] for residue in capped["assembled_residues"]] == ["ACE", "GLX", "NME"]

