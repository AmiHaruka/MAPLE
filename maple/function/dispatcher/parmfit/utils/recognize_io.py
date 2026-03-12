#!/usr/bin/env python3
"""
Use the current recognize parser to:
1) parse a PDB file into Structure
2) export Structure as JSON
3) rebuild a PDB file from Structure
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    # Package mode
    from .recognize import Atom, Residue, ResidueKey, Structure, parse_pdb
except ImportError:
    # Script mode
    from recognize import Atom, Residue, ResidueKey, Structure, parse_pdb


def key_to_dict(key: ResidueKey) -> dict[str, Any]:
    return {
        "model": key.model,
        "chain": key.chain,
        "resseq": key.resseq,
        "icode": key.icode,
        "resname": key.resname,
    }


def atom_to_dict(atom: Atom) -> dict[str, Any]:
    return {
        "serial": atom.serial,
        "record": atom.record,
        "name": atom.name,
        "altloc": atom.altloc,
        "x": atom.x,
        "y": atom.y,
        "z": atom.z,
        "occ": atom.occ,
        "bfac": atom.bfac,
        "element": atom.element,
        "charge": atom.charge,
        "residue_key": key_to_dict(atom.residue_key) if atom.residue_key else None,
    }


def residue_to_dict(residue: Residue) -> dict[str, Any]:
    atoms = sorted(residue.atoms.values(), key=lambda a: a.serial)
    return {
        "key": key_to_dict(residue.key),
        "file_order": residue.file_order,
        "ter_after": residue.ter_after,
        "parent_std": residue.parent_std,
        "prev_parent_std": residue.prev_parent_std,
        "next_parent_std": residue.next_parent_std,
        "ccd_type": residue.ccd_type,
        "atoms": [atom_to_dict(atom) for atom in atoms],
    }


def structure_to_dict(structure: Structure) -> dict[str, Any]:
    links = []
    for rk1, a1, rk2, a2 in structure.links:
        links.append(
            {
                "left_residue": key_to_dict(rk1),
                "left_atom": a1,
                "right_residue": key_to_dict(rk2),
                "right_atom": a2,
            }
        )

    conect = {
        str(root): sorted(neighbors)
        for root, neighbors in sorted(structure.conect.items(), key=lambda kv: kv[0])
        if neighbors
    }

    serial_to_residue = {
        str(serial): key_to_dict(key)
        for serial, key in sorted(structure.serial_to_residue.items(), key=lambda kv: kv[0])
    }

    return {
        "residue_count": len(structure.residues),
        "residues": [residue_to_dict(residue) for residue in structure.residues],
        "links": links,
        "conect": conect,
        "serial_to_residue": serial_to_residue,
    }


def _atom_name_field(atom: Atom) -> str:
    # Keep output stable and simple: right align to the standard 4-char atom-name field.
    return f"{atom.name[:4]:>4}"


def format_atom_line(atom: Atom, residue: Residue) -> str:
    chain = "" if residue.key.chain == "_" else residue.key.chain
    icode = residue.key.icode or " "
    altloc = atom.altloc or " "
    element = (atom.element or "").strip()
    charge = (atom.charge or "").strip()

    line = (
        f"{atom.record:<6}{atom.serial:>5} "
        f"{_atom_name_field(atom)}"
        f"{altloc:1}"
        f"{residue.key.resname:>3} "
        f"{chain:1}"
        f"{residue.key.resseq:>4}"
        f"{icode:1}   "
        f"{atom.x:>8.3f}"
        f"{atom.y:>8.3f}"
        f"{atom.z:>8.3f}"
        f"{atom.occ:>6.2f}"
        f"{atom.bfac:>6.2f}"
        f"          "
        f"{element:>2}"
        f"{charge:>2}"
    )
    return line.ljust(80)


def format_link_line(rk1: ResidueKey, a1: str, rk2: ResidueKey, a2: str) -> str:
    chain1 = "" if rk1.chain == "_" else rk1.chain
    chain2 = "" if rk2.chain == "_" else rk2.chain
    icode1 = rk1.icode or " "
    icode2 = rk2.icode or " "
    line = (
        f"LINK        {a1:>4} {rk1.resname:>3} {chain1:1}{rk1.resseq:>4}{icode1:1}"
        f"                 {a2:>4} {rk2.resname:>3} {chain2:1}{rk2.resseq:>4}{icode2:1}"
    )
    return line.ljust(80)


def write_structure_pdb(structure: Structure, out_path: str) -> None:
    lines: list[str] = []

    model_ids = sorted({residue.key.model for residue in structure.residues})
    multi_model = len(model_ids) > 1

    for model in model_ids:
        model_residues = [res for res in structure.residues if res.key.model == model]
        if multi_model:
            lines.append(f"MODEL     {model:>4}".ljust(80))

        for residue in model_residues:
            atoms = sorted(residue.atoms.values(), key=lambda atom: atom.serial)
            for atom in atoms:
                lines.append(format_atom_line(atom, residue))
            if residue.ter_after:
                lines.append("TER".ljust(80))

        if multi_model:
            lines.append("ENDMDL".ljust(80))

    for rk1, a1, rk2, a2 in structure.links:
        lines.append(format_link_line(rk1, a1, rk2, a2))

    for root in sorted(structure.conect.keys()):
        neighbors = sorted(structure.conect[root])
        if not neighbors:
            continue
        for i in range(0, len(neighbors), 4):
            group = neighbors[i : i + 4]
            line = f"CONECT{root:>5}" + "".join(f"{nbr:>5}" for nbr in group)
            lines.append(line.ljust(80))

    lines.append("END".ljust(80))
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_default_paths(input_pdb: str) -> tuple[str, str]:
    src = Path(input_pdb)
    stem = src.stem
    parent = src.parent
    return (
        str(parent / f"{stem}.recognized.json"),
        str(parent / f"{stem}.rebuild.pdb"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse PDB using recognize.py, export JSON, and rebuild PDB."
    )
    parser.add_argument("input_pdb", help="Input PDB file path")
    parser.add_argument(
        "--json-out",
        dest="json_out",
        default=None,
        help="Output JSON file path (default: <input>.recognized.json)",
    )
    parser.add_argument(
        "--pdb-out",
        dest="pdb_out",
        default=None,
        help="Rebuilt PDB file path (default: <input>.rebuild.pdb)",
    )
    parser.add_argument(
        "--keep-altloc",
        default="A",
        help="Keep this altloc letter and blank altloc entries (default: A)",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indent size (default: 2)",
    )
    args = parser.parse_args()

    json_default, pdb_default = build_default_paths(args.input_pdb)
    json_out = args.json_out or json_default
    pdb_out = args.pdb_out or pdb_default

    structure = parse_pdb(args.input_pdb, keep_altloc=args.keep_altloc)

    data = structure_to_dict(structure)
    Path(json_out).write_text(json.dumps(data, indent=args.indent), encoding="utf-8")

    write_structure_pdb(structure, pdb_out)

    print(f"Input PDB: {args.input_pdb}")
    print(f"Residues recognized: {len(structure.residues)}")
    print(f"JSON output: {json_out}")
    print(f"Rebuilt PDB: {pdb_out}")


if __name__ == "__main__":
    main()

