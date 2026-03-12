#!/usr/bin/env python3
"""
Test helper for addcap:
1) parse a PDB file
2) locate target residue by chain + residue id (and optional model/icode)
3) build ACE/NME capped fragment
4) export capped fragment as JSON and PDB
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .addcap import CappedFragment, build_capped_fragment
    from .recognize import Atom, ResidueKey, Structure, parse_pdb
except ImportError:
    from addcap import CappedFragment, build_capped_fragment
    from recognize import Atom, ResidueKey, Structure, parse_pdb


def key_to_dict(key: ResidueKey | None) -> dict[str, Any] | None:
    if key is None:
        return None
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
        "residue_key": key_to_dict(atom.residue_key),
    }


def context_to_dict(ctx: dict[str, Any]) -> dict[str, Any]:
    out = dict(ctx)
    for k in ("target_key", "prev_key", "next_key"):
        if k in out:
            out[k] = key_to_dict(out[k])
    return out


def fragment_to_dict(fragment: CappedFragment) -> dict[str, Any]:
    return {
        "atom_count": len(fragment.atoms),
        "bond_count": len(fragment.bonds),
        "ace_mode": fragment.ace_mode,
        "nme_mode": fragment.nme_mode,
        "context": context_to_dict(fragment.context),
        "residue_groups": {k: sorted(v) for k, v in fragment.residue_groups.items()},
        "atoms": [atom_to_dict(atom) for atom in sorted(fragment.atoms, key=lambda a: a.serial)],
        "bonds": [list(b) for b in sorted(fragment.bonds)],
    }


def _atom_name_field(atom: Atom) -> str:
    return f"{atom.name[:4]:>4}"


def format_atom_line(atom: Atom) -> str:
    if atom.residue_key is None:
        raise ValueError(f"Atom {atom.serial} has no residue_key; cannot write PDB.")
    key = atom.residue_key
    chain = "" if key.chain == "_" else key.chain
    icode = key.icode or " "
    altloc = atom.altloc or " "
    element = (atom.element or "").strip()
    charge = (atom.charge or "").strip()

    line = (
        f"{atom.record:<6}{atom.serial:>5} "
        f"{_atom_name_field(atom)}"
        f"{altloc:1}"
        f"{key.resname:>3} "
        f"{chain:1}"
        f"{key.resseq:>4}"
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


def write_fragment_pdb(fragment: CappedFragment, out_path: str) -> None:
    atom_by_serial = {atom.serial: atom for atom in fragment.atoms}
    lines: list[str] = []

    # Keep fragment order explicit for readability.
    ordered_groups = ("ACE", "TARGET", "NME")
    for i, group_name in enumerate(ordered_groups):
        serials = sorted(fragment.residue_groups.get(group_name, []))
        for serial in serials:
            atom = atom_by_serial[serial]
            lines.append(format_atom_line(atom))
        if i < len(ordered_groups) - 1:
            lines.append("TER".ljust(80))

    # Write connectivity from fragment bonds.
    neigh: dict[int, list[int]] = {}
    for a, b in sorted(fragment.bonds):
        neigh.setdefault(a, []).append(b)
        neigh.setdefault(b, []).append(a)
    for root in sorted(neigh.keys()):
        nbrs = sorted(set(neigh[root]))
        for i in range(0, len(nbrs), 4):
            group = nbrs[i : i + 4]
            line = f"CONECT{root:>5}" + "".join(f"{n:>5}" for n in group)
            lines.append(line.ljust(80))

    lines.append("END".ljust(80))
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _normalize_chain(chain: str) -> str:
    return chain if chain else "_"


def locate_target_index(
    structure: Structure,
    chain: str,
    resseq: int,
    model: int | None,
    icode: str,
) -> int:
    chain_norm = _normalize_chain(chain)
    icode_norm = icode.strip()
    matches: list[int] = []
    for i, residue in enumerate(structure.residues):
        key = residue.key
        if key.chain != chain_norm:
            continue
        if key.resseq != resseq:
            continue
        if model is not None and key.model != model:
            continue
        if icode_norm and key.icode != icode_norm:
            continue
        matches.append(i)

    if not matches:
        raise ValueError(
            f"No residue found for chain={chain_norm!r}, resseq={resseq}, "
            f"model={model if model is not None else '*'}, icode={icode_norm!r}"
        )
    if len(matches) > 1:
        candidates = []
        for idx in matches[:8]:
            key = structure.residues[idx].key
            candidates.append(
                f"idx={idx} model={key.model} chain={key.chain} "
                f"resseq={key.resseq} icode={key.icode!r} resname={key.resname}"
            )
        candidate_text = "; ".join(candidates)
        raise ValueError(
            "Multiple residues matched. Please specify --model and/or --icode. "
            f"Candidates: {candidate_text}"
        )
    return matches[0]


def build_default_paths(input_pdb: str, chain: str, resseq: int, model: int | None, icode: str) -> tuple[str, str]:
    src = Path(input_pdb)
    stem = src.stem
    parent = src.parent
    chain_label = chain if chain else "_"
    model_label = model if model is not None else "auto"
    icode_label = icode if icode else "none"
    tag = f"cap_{chain_label}_{resseq}_m{model_label}_i{icode_label}"
    return (
        str(parent / f"{stem}.{tag}.json"),
        str(parent / f"{stem}.{tag}.pdb"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build ACE/NME capped fragment from input PDB and target residue."
    )
    parser.add_argument("input_pdb", help="Input PDB file path")
    parser.add_argument("chain", help="Target chain id (use '_' for blank chain)")
    parser.add_argument("resseq", type=int, help="Target residue sequence number")
    parser.add_argument(
        "--model",
        type=int,
        default=None,
        help="Target model id (default: auto match)",
    )
    parser.add_argument(
        "--icode",
        default="",
        help="Target insertion code (default: empty)",
    )
    parser.add_argument(
        "--keep-altloc",
        default="A",
        help="Keep this altloc letter and blank altloc entries (default: A)",
    )
    parser.add_argument(
        "--json-out",
        dest="json_out",
        default=None,
        help="Output JSON file path (default: auto naming)",
    )
    parser.add_argument(
        "--pdb-out",
        dest="pdb_out",
        default=None,
        help="Output capped PDB file path (default: auto naming)",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indent size (default: 2)",
    )
    args = parser.parse_args()

    json_default, pdb_default = build_default_paths(
        args.input_pdb, args.chain, args.resseq, args.model, args.icode
    )
    json_out = args.json_out or json_default
    pdb_out = args.pdb_out or pdb_default

    structure = parse_pdb(args.input_pdb, keep_altloc=args.keep_altloc)
    target_idx = locate_target_index(
        structure=structure,
        chain=args.chain,
        resseq=args.resseq,
        model=args.model,
        icode=args.icode,
    )

    result = build_capped_fragment(structure, target_idx)
    fragment = result["fragment"]

    payload = {
        "input_pdb": args.input_pdb,
        "target_index": target_idx,
        "target": {
            "chain": _normalize_chain(args.chain),
            "resseq": args.resseq,
            "model": args.model,
            "icode": args.icode,
        },
        "ace_mode": result["ace_mode"],
        "nme_mode": result["nme_mode"],
        "fragment": fragment_to_dict(fragment),
    }
    Path(json_out).write_text(json.dumps(payload, indent=args.indent), encoding="utf-8")
    write_fragment_pdb(fragment, pdb_out)

    print(f"Input PDB: {args.input_pdb}")
    print(f"Target index: {target_idx}")
    print(f"Target chain/resseq: {_normalize_chain(args.chain)}/{args.resseq}")
    print(f"ACE/NME mode: {result['ace_mode']}/{result['nme_mode']}")
    print(f"JSON output: {json_out}")
    print(f"Capped PDB: {pdb_out}")


if __name__ == "__main__":
    main()

