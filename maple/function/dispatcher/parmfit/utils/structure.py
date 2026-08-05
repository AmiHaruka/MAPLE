"""Usage: parse PDB structures and provide shared residue/atom helpers."""

from __future__ import annotations

import re
from typing import Optional

import numpy as np

COVALENT_RADII = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "S": 1.05,
    "P": 1.07,
    "F": 0.57,
    "CL": 0.99,
    "BR": 1.14,
    "I": 1.33,
    "SE": 1.20,
    "ZN": 1.22,
    "MG": 1.41,
    "CA": 1.76,
    "NA": 1.66,
    "K": 2.03,
    "FE": 1.24,
    "CU": 1.32,
    "MN": 1.39,
    "CO": 1.26,
    "NI": 1.21,
}
WATER_NAMES = {"HOH", "WAT", "SOL"}

ION_ELEMENTS = {
    "LI", "NA", "K", "RB", "CS",
    "BE", "MG", "CA", "SR", "BA",
    "AL", "IN",
    "Y", "LA", "PR", "ND", "GD", "TB", "DY", "HO", "ER", "LU",
    "TI", "ZR", "HF", "TH", "U", "PU",
    "FE", "MN", "CO", "NI", "CU", "ZN",
    "CD", "HG", "PB",
    "PD", "PT",
    "NB", "TA", "TC", "MO",
}

METAL_SITE_DONOR_ELEMENTS = {"N", "O", "S", "P", "SE", "F", "CL", "BR", "I"}

ATOMIC_MASSES = {
    "H": 1.008,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998,
    "NA": 22.990,
    "MG": 24.305,
    "P": 30.974,
    "S": 32.060,
    "CL": 35.450,
    "K": 39.098,
    "CA": 40.078,
    "MN": 54.938,
    "FE": 55.845,
    "CO": 58.933,
    "NI": 58.693,
    "CU": 63.546,
    "ZN": 65.380,
    "SE": 78.971,
    "BR": 79.904,
    "I": 126.904,
    "TI": 47.867,
    "NB": 92.906,
    "MO": 95.950,
    "TC": 98.000,
    "TA": 180.948,
}

BOUNDARY_H_BOND_LENGTH = {
    "C": 1.090,
    "N": 1.010,
    "O": 0.960,
    "S": 1.340,
    "P": 1.420,
}
BOND_C_N_AMIDE = 1.335
BOND_C_O = 1.229
BOND_C_CH3_ACE = 1.522
BOND_N_CH3_NME = 1.458
BOND_C_H = 1.090
BOND_N_H = 1.010


def _norm(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else np.zeros(3)


def _bond_pair(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def covalent_cutoff(atom1: dict, atom2: dict, factor: float = 1.3) -> float:
    r1 = COVALENT_RADII.get(atom1["element"], 0.77)
    r2 = COVALENT_RADII.get(atom2["element"], 0.77)
    return factor * (r1 + r2)


def _measure_dihedral(atoms, quartet: tuple[int, int, int, int]) -> float:
    if hasattr(atoms, "get_dihedral"):
        return float(atoms.get_dihedral(*quartet))

    positions = np.asarray(atoms.get_positions(), dtype=float)
    p0, p1, p2, p3 = (positions[index] for index in quartet)
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1 /= np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return float(np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))))


def _pair_is_bonded(structure: dict, atom1: dict, atom2: dict, bond_policy: str = "auto") -> bool:
    pair = _bond_pair(atom1["serial"], atom2["serial"])
    if bond_policy == "record":
        return pair in structure.get("explicit_pairs", set())
    cutoff = covalent_cutoff(atom1, atom2)
    delta = get_atom_xyz(atom1) - get_atom_xyz(atom2)
    covalent = float(np.sqrt(np.dot(delta, delta))) <= cutoff
    if bond_policy == "covalent":
        return covalent
    if structure.get("bond_pairs"):
        return pair in structure["bond_pairs"]
    return pair in structure.get("explicit_pairs", set()) or covalent


def find_external_partners(structure: dict, atom: dict, selected_serials: set[int], bond_policy: str = "auto") -> list[dict]:
    pairs = structure.get("explicit_pairs", set()) if bond_policy == "record" else None
    if bond_policy == "auto":
        pairs = structure.get("bond_pairs") or None
    if pairs is None:
        pairs = {
            _bond_pair(atom["serial"], other["serial"])
            for other in structure["serial_to_atom"].values()
            if other["serial"] != atom["serial"] and _pair_is_bonded(structure, atom, other, bond_policy=bond_policy)
        }
    partner_serials = {
        right if left == atom["serial"] else left
        for left, right in pairs
        if left == atom["serial"] or right == atom["serial"]
    }
    partners = [
        structure["serial_to_atom"][serial]
        for serial in partner_serials
        if serial not in selected_serials
    ]
    partners.sort(key=lambda item: item["serial"])
    return partners


def peptide_link(prev_residue: dict, curr_residue: dict, structure: dict, bond_policy: str = "auto") -> bool:
    if not is_peptide_like(prev_residue) or not is_peptide_like(curr_residue):
        return False
    c_atom = search_atom(prev_residue, "C")
    n_atom = search_atom(curr_residue, "N")
    if c_atom is None or n_atom is None:
        return False
    return _pair_is_bonded(structure, c_atom, n_atom, bond_policy=bond_policy)


def _project_perp(vector: np.ndarray, axis: np.ndarray) -> np.ndarray:
    return vector - axis * float(np.dot(vector, axis))


def arbitrary_perp(axis: np.ndarray) -> np.ndarray:
    candidates = (
        np.array((1.0, 0.0, 0.0), dtype=float),
        np.array((0.0, 1.0, 0.0), dtype=float),
        np.array((0.0, 0.0, 1.0), dtype=float),
    )
    for candidate in candidates:
        projected = _project_perp(candidate, axis)
        if float(np.linalg.norm(projected)) > 1.0e-8:
            return _norm(projected)
    raise ValueError("Failed to find perpendicular direction.")


def trigonal_pair(primary_dir: np.ndarray, hint_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis = _norm(primary_dir)
    hint = _project_perp(hint_vec, axis)
    if float(np.linalg.norm(hint)) < 1.0e-8:
        hint = arbitrary_perp(axis)
    else:
        hint = _norm(hint)
    cos120 = -0.5
    sin120 = float(np.sqrt(3.0) * 0.5)
    direction1 = _norm(axis * cos120 + hint * sin120)
    direction2 = _norm(axis * cos120 - hint * sin120)
    return direction1, direction2


def tetrahedral_h_dirs(anchor_dir: np.ndarray, hint_vec: Optional[np.ndarray] = None) -> list[np.ndarray]:
    axis = _norm(anchor_dir)
    if hint_vec is None:
        basis1 = arbitrary_perp(axis)
    else:
        hint = _project_perp(hint_vec, axis)
        basis1 = arbitrary_perp(axis) if float(np.linalg.norm(hint)) < 1.0e-8 else _norm(hint)
    basis2 = _norm(np.cross(axis, basis1))

    coeff_axis = -1.0 / 3.0
    coeff_plane = float(np.sqrt(8.0) / 3.0)
    directions: list[np.ndarray] = []
    for phi in (0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0):
        plane = basis1 * float(np.cos(phi)) + basis2 * float(np.sin(phi))
        directions.append(_norm(axis * coeff_axis + plane * coeff_plane))
    return directions


def boundary_h_length(atom: dict) -> float:
    return BOUNDARY_H_BOND_LENGTH.get(atom["element"], BOUNDARY_H_BOND_LENGTH["C"])


def make_h_name(residue: dict, attached_atom: dict) -> str:
    prefix = {
        "C": "HC",
        "N": "HN",
        "O": "HO",
        "S": "HS",
        "P": "HP",
    }.get(attached_atom["element"], "H")
    used = {atom["name"] for atom in residue["atoms"]}
    for idx in range(1, 100):
        name = f"{prefix}{idx}"
        if len(name) <= 4 and name not in used:
            return name
    for idx in range(1, 1000):
        name = f"H{idx:03d}"
        if name not in used:
            return name
    raise ValueError(f"Failed to allocate hydrogen name for {get_resid_label(residue)}.")


def get_atom_xyz(atom: dict) -> np.ndarray:
    return np.asarray(atom["xyz"], dtype=float)


def search_atom(residue: dict, name: str) -> Optional[dict]:
    for atom in residue["atoms"]:
        if atom.get("role") == name:
            return atom
    for atom in residue["atoms"]:
        if atom["name"] == name:
            return atom
    return None


def get_atom_info(atom: dict) -> dict:
    info = {
        "serial": int(atom["serial"]),
        "name": atom["name"],
        "element": atom["element"],
        "xyz": [float(atom["xyz"][0]), float(atom["xyz"][1]), float(atom["xyz"][2])],
    }
    for key in ("source_name", "altloc", "role", "amber_type", "charge"):
        if key in atom:
            info[key] = atom[key]
    return info


def make_atom(serial: int, name: str, element: str, xyz: np.ndarray) -> dict:
    return {
        "serial": serial,
        "name": name,
        "element": element,
        "xyz": np.array(xyz, dtype=float),
    }


def copy_atom(
    atom: dict,
    *,
    serial: Optional[int] = None,
    name: Optional[str] = None,
    element: Optional[str] = None,
    xyz: Optional[np.ndarray] = None,
) -> dict:
    copied = {
        "serial": atom["serial"] if serial is None else serial,
        "name": atom["name"] if name is None else name,
        "element": atom["element"] if element is None else element,
        "xyz": np.array(atom["xyz"] if xyz is None else xyz, dtype=float),
    }
    for key, value in atom.items():
        if key not in copied and key != "xyz":
            copied[key] = value
    return copied


def get_resid_key(residue: dict) -> tuple[str, int, str]:
    return (residue["chain"], residue["resseq"], residue["icode"])


def get_resid_label(residue: dict) -> str:
    return f"{residue['chain']}{residue['resseq']}{residue['icode']}:{residue['resname']}"


def get_resid_mindist(left: dict, right: dict) -> float:
    diff = left["coords"][:, None, :] - right["coords"][None, :, :]
    return float(np.sqrt((diff * diff).sum(axis=2)).min())


def refresh_resid(residue: dict) -> None:
    residue["atoms"] = sorted(residue["atoms"], key=lambda atom: atom["serial"])
    residue["coords"] = np.asarray([atom["xyz"] for atom in residue["atoms"]], dtype=float)


def get_resid_info(residue: Optional[dict], resname: Optional[str] = None) -> Optional[dict]:
    if residue is None:
        return None
    info = {
        "chain": residue["chain"],
        "resseq": int(residue["resseq"]),
        "icode": residue["icode"],
        "resname": residue["resname"] if resname is None else resname,
        "kind": residue["kind"],
        "atoms": [get_atom_info(atom) for atom in sorted(residue["atoms"], key=lambda item: item["serial"])],
    }
    for key in ("source_resname", "selected_altloc", "template_id", "template_category", "net_charge"):
        if key in residue:
            info[key] = residue[key]
    return info


def make_residue(chain: str, resseq: int, icode: str, resname: str, atoms: list[dict], kind: str = "cap") -> dict:
    residue = {
        "chain": chain,
        "resseq": resseq,
        "icode": icode,
        "resname": resname,
        "kind": kind,
        "atoms": sorted(atoms, key=lambda atom: atom["serial"]),
    }
    residue["coords"] = np.asarray([atom["xyz"] for atom in residue["atoms"]], dtype=float)
    return residue


def match_resid(residue: dict, selector: dict) -> bool:
    if residue["resseq"] != selector["resseq"]:
        return False
    if "_icode" in selector and residue["icode"] != selector["_icode"]:
        return False
    if selector.get("chain") is not None and residue["chain"] != selector["chain"]:
        return False
    names = {str(residue.get("resname", "")).upper(), str(residue.get("source_resname", "")).upper()}
    if selector.get("resname") is not None and selector["resname"] not in names:
        return False
    if selector.get("altloc") and residue.get("selected_altloc", "") != selector["altloc"]:
        return False
    return True


def match_chain(left: dict, right: dict) -> bool:
    return left["chain"] == right["chain"]


def is_peptide_like(residue: dict) -> bool:
    if residue.get("kind") == "protein":
        return True
    names = {atom.get("role", atom["name"]) for atom in residue["atoms"]}
    return {"N", "CA", "C", "O"}.issubset(names)


def max_serial(structure_or_residues: dict | list[dict]) -> int:
    residues = structure_or_residues["residues"] if isinstance(structure_or_residues, dict) else structure_or_residues
    value = 0
    for residue in residues:
        for atom in residue["atoms"]:
            value = max(value, atom["serial"])
    return value


def copy_residue(residue: dict, *, resname: Optional[str] = None, kind: Optional[str] = None) -> dict:
    copied = {
        "chain": residue["chain"],
        "resseq": residue["resseq"],
        "icode": residue["icode"],
        "resname": residue["resname"] if resname is None else resname,
        "kind": residue["kind"] if kind is None else kind,
        "atoms": [copy_atom(atom) for atom in residue["atoms"]],
    }
    for key, value in residue.items():
        if key not in copied and key not in {"atoms", "coords", "_pair_cache"}:
            copied[key] = value
    copied["coords"] = (
        np.asarray([atom["xyz"] for atom in copied["atoms"]], dtype=float) if copied["atoms"] else np.zeros((0, 3))
    )
    return copied


def residue_sort_key(residue: dict) -> tuple[int, str, int, str]:
    return (int(residue.get("_index", 0)), residue["chain"], int(residue["resseq"]), residue["icode"])


def classify_kind(residue: dict) -> str:
    resname = residue["resname"].upper()
    if resname in WATER_NAMES:
        return "water"
    if len(residue["atoms"]) == 1 and residue["atoms"][0]["element"] in ION_ELEMENTS:
        return "ion"
    if is_peptide_like(residue):
        return "protein"
    return "ligand" #TODO: refine this classification


def parse_selector(selector: str) -> dict:
    text = str(selector).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1].strip()
    text = text.replace(":", "")
    chain_match = re.fullmatch(r"(?P<chain>[A-Za-z0-9_])(?P<resseq>[-+]?\d+)(?P<altloc>[A-Za-z]?)", text)
    if chain_match:
        return {
            "chain": chain_match.group("chain"),
            "resname": None,
            "resseq": int(chain_match.group("resseq")),
            "altloc": chain_match.group("altloc").upper(),
        }
    resname_match = re.fullmatch(r"(?P<resname>[A-Za-z]{2,4})(?P<resseq>[-+]?\d+)(?P<altloc>[A-Za-z]?)", text)
    if resname_match:
        return {
            "chain": None,
            "resname": resname_match.group("resname").upper(),
            "resseq": int(resname_match.group("resseq")),
            "altloc": resname_match.group("altloc").upper(),
        }
    raise ValueError(f"Invalid residue selector {selector!r}. Use forms like 'A11', 'A11A', or 'SER11'.")


def parse_pdb_coord(line: str) -> tuple[float, float, float]:
    from maple.function.read.filereader.pdb_reader import parse_pdb_coord as _parse_pdb_coord

    return _parse_pdb_coord(line)


def read_pdb(
    path: str,
    keep_altloc: str = "A",
    model: Optional[int] = None,
    *,
    prom: str = "ff14SB",
    altloc_selectors: list[str] | None = None,
) -> dict:
    from maple.function.read.filereader.pdb_reader import read_pdb as _read_pdb

    return _read_pdb(
        path,
        keep_altloc=keep_altloc,
        model=model,
        prom=prom,
        altloc_selectors=altloc_selectors,
    )
