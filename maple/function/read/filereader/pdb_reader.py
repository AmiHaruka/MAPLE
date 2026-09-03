"""PDB file block reader and parser."""

from __future__ import annotations

import os
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Optional

from ase import Atoms
import numpy as np

from maple.function.utility import Molecules

_PDB_ATOM_RECORDS = {"ATOM", "HETATM", "HEATOM"}


def _case_insensitive_lookup(path: str) -> str:
    directory, filename = os.path.split(path)
    if not directory or not os.path.isdir(directory):
        return path
    exact = os.path.join(directory, filename)
    if os.path.exists(exact):
        return exact
    target = filename.lower()
    for candidate in os.listdir(directory):
        if candidate.lower() == target:
            return os.path.join(directory, candidate)
    return path


def _parse_charge_mult(input_str: str) -> tuple[str, Optional[int], Optional[int]]:
    text = str(input_str).strip()
    parts = text.split()
    if parts and parts[0].upper() == "PDB":
        parts = parts[1:]
    if not parts:
        raise ValueError("PDB file reference requires a path.")

    if len(parts) == 1:
        return parts[0], None, None
    if len(parts) == 3:
        try:
            charge = int(parts[0])
            mult = int(parts[1])
        except ValueError as exc:
            raise ValueError("PDB charge/multiplicity syntax is: PDB <charge> <mult> <path>.") from exc
        return parts[2], charge, mult
    raise ValueError("PDB file reference syntax is: PDB <path> or PDB <charge> <mult> <path>.")


def _resolve_pdb_path(file_path: str, base_dir: Optional[str] = None) -> str:
    path, _charge, _mult = _parse_charge_mult(file_path)

    resolved = path
    if not os.path.isabs(resolved):
        resolved = os.path.join(base_dir if base_dir is not None else os.getcwd(), resolved)
    resolved = os.path.abspath(resolved)
    resolved = _case_insensitive_lookup(resolved)
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"PDB file not found: {file_path}")
    return resolved


def _infer_pdb_element(
    atom_name: str,
    element_field: str = "",
    *,
    record: str = "",
    resname: str = "",
) -> str:
    field = element_field.strip()
    if field:
        return field[:2].strip().capitalize()

    raw_name = atom_name[:4].ljust(4)
    stripped = raw_name.strip()
    letters = "".join(ch for ch in stripped if ch.isalpha())
    if not letters:
        return "X"
    two_letter = {"CL", "BR", "NA", "MG", "ZN", "FE", "MN", "CO", "NI", "CU", "CA", "SE"}
    token = letters.upper()
    is_hetatm = record.upper() in {"HETATM", "HEATOM"}
    if is_hetatm and token[:2] == resname.strip().upper()[:2] and token[:2] in two_letter:
        return token[:2].capitalize()
    # Two-letter elements (metals, halides) only appear on HETATM records; an ATOM
    # record carries a standard biopolymer atom (H/C/N/O/S/P), so names like "CA"
    # (C-alpha) must not be read as calcium regardless of column justification.
    if is_hetatm and raw_name[0].isalpha() and len(token) >= 2 and token[:2] in two_letter:
        return letters[:2].capitalize()
    return token[0]


def _pdb_atoms_to_ase(atoms: list[dict], template_lines: list[str]) -> Atoms:
    if not atoms:
        raise ValueError("PDB model contains no ATOM/HETATM coordinate records.")
    symbols = [atom["element"] for atom in atoms]
    positions = np.asarray([atom["xyz"] for atom in atoms], dtype=np.float64)
    ase_atoms = Atoms(symbols=symbols, positions=positions)
    ase_atoms.info["pdb_template"] = list(template_lines)
    return ase_atoms


def _apply_charge_mult(atoms: Atoms, charge: Optional[int], mult: Optional[int]) -> None:
    if charge is not None:
        atoms.info["charge"] = charge
    if mult is not None:
        atoms.info["mult"] = mult
        atoms.info["spin"] = (mult - 1) / 2


class PDBReader:
    """Read a PDB file reference as coordinate-only ASE Atoms or Molecules."""

    resolve_path = staticmethod(_resolve_pdb_path)

    def __new__(cls, file_path: str, base_dir: Optional[str] = None):
        _path, charge, mult = _parse_charge_mult(file_path)
        resolved = cls.resolve_path(file_path, base_dir=base_dir)
        frames = cls._read_coordinate_frames(resolved)
        if not frames:
            raise ValueError(f"No valid PDB coordinate records found in file: {resolved}")
        for atoms in frames:
            _apply_charge_mult(atoms, charge, mult)
        if len(frames) == 1:
            return frames[0]
        return Molecules(frames)

    @staticmethod
    def _read_coordinate_frames(path: str) -> list[Atoms]:
        frames: list[Atoms] = []
        current_atoms: list[dict] = []
        current_template: list[str] = []
        saw_model = False
        in_model = False

        def finish_frame() -> None:
            nonlocal current_atoms, current_template
            if current_atoms:
                frames.append(_pdb_atoms_to_ase(current_atoms, current_template))
                current_atoms = []
                current_template = []

        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            for raw in handle:
                line = raw.rstrip("\r\n").ljust(80)
                record = line[:6].strip().upper()

                if record == "MODEL":
                    saw_model = True
                    in_model = True
                    current_atoms = []
                    current_template = []
                    continue

                if record == "ENDMDL":
                    finish_frame()
                    in_model = False
                    continue

                if record == "END":
                    break

                if saw_model and not in_model:
                    continue

                if record == "TER":
                    if current_atoms:
                        current_template.append(line)
                    continue

                if record not in _PDB_ATOM_RECORDS:
                    continue

                altloc = line[16].strip()
                if altloc and altloc != "A":
                    continue

                atom_name = line[12:16].strip()
                current_template.append(line)
                current_atoms.append(
                    {
                        "element": _infer_pdb_element(
                            line[12:16],
                            line[76:78],
                            record=record,
                            resname=line[17:20],
                        ),
                        "xyz": parse_pdb_coord(line),
                    }
                )

        if not saw_model:
            finish_frame()
        elif in_model:
            finish_frame()

        return frames


def _format_pdb_atom_line(line: str, xyz: np.ndarray) -> str:
    padded = line.rstrip("\r\n").ljust(80)
    x, y, z = xyz
    return f"{padded[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{padded[54:]}"


def _pdb_frame_lines(atoms: Atoms, template_lines: list[str]) -> list[str]:
    positions = np.asarray(atoms.get_positions(), dtype=float)
    atom_line_count = sum(
        1 for line in template_lines if line[:6].strip().upper() in _PDB_ATOM_RECORDS
    )
    if atom_line_count != len(atoms):
        raise ValueError(
            f"PDB template atom count ({atom_line_count}) does not match atom count ({len(atoms)})."
        )

    lines: list[str] = []
    atom_index = 0
    for raw in template_lines:
        line = raw.rstrip("\r\n").ljust(80)
        record = line[:6].strip().upper()
        if record in _PDB_ATOM_RECORDS:
            lines.append(_format_pdb_atom_line(line, positions[atom_index]))
            atom_index += 1
        elif record == "TER":
            lines.append(line)
    return lines


def write_pdb(path: str, atoms: Atoms, template_lines: list[str]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for line in _pdb_frame_lines(atoms, template_lines):
            handle.write(line.rstrip("\r\n") + "\n")
        handle.write("END\n")


def write_pdb_model(
    handle,
    atoms: Atoms,
    template_lines: list[str],
    *,
    model_index: int = 1,
    remark: Optional[str] = None,
) -> None:
    handle.write(f"MODEL     {model_index:4d}\n")
    if remark:
        handle.write(f"REMARK   {remark}\n")
    for line in _pdb_frame_lines(atoms, template_lines):
        handle.write(line.rstrip("\r\n") + "\n")
    handle.write("ENDMDL\n")


def write_pdb_trajectory(
    path: str,
    atoms_list: list[Atoms],
    energies: Optional[list[float]] = None,
    mode: str = "w",
    start_index: int = 0,
) -> None:
    if not atoms_list:
        with open(path, mode, encoding="utf-8"):
            pass
        return
    base_template = atoms_list[0].info.get("pdb_template")
    with open(path, mode, encoding="utf-8") as handle:
        for i, atoms in enumerate(atoms_list):
            template = atoms.info.get("pdb_template") or base_template
            if not template:
                raise ValueError("PDB trajectory output requires atoms.info['pdb_template'].")
            remark = f"Energy = {energies[i]:.10f}" if energies is not None else None
            write_pdb_model(
                handle,
                atoms,
                template,
                model_index=start_index + i + 1,
                remark=remark,
            )


def _parse_conect_line(line: str) -> tuple[Optional[int], list[int]]:
    serials: list[int] = []
    for idx in range(6, len(line), 5):
        token = line[idx : idx + 5].strip()
        if not token:
            continue
        serials.append(int(token))
    if not serials:
        return None, []
    return serials[0], serials[1:]


def parse_pdb_coord(line: str) -> tuple[float, float, float]:
    padded = line.rstrip("\r\n").ljust(80)
    return (
        float(padded[30:38]),
        float(padded[38:46]),
        float(padded[46:54]),
    )


def _parse_link_line(line: str) -> Optional[dict]:
    atom1 = line[12:16].strip()
    atom2 = line[42:46].strip()
    if not atom1 or not atom2:
        return None
    return {
        "left": {
            "chain": (line[21].strip() or "_"),
            "resseq": int(line[22:26]),
            "icode": line[26].strip(),
            "resname": line[17:20].strip(),
            "atom": atom1,
        },
        "right": {
            "chain": (line[51].strip() or "_"),
            "resseq": int(line[52:56]),
            "icode": line[56].strip(),
            "resname": line[47:50].strip(),
            "atom": atom2,
        },
    }


@dataclass(frozen=True)
class PDBReadDiagnostics:
    matched: int = 0
    backbone_only: int = 0
    unmatched: int = 0
    warnings: tuple[str, ...] = field(default_factory=tuple)
    backbone_only_residues: tuple[str, ...] = field(default_factory=tuple)
    not_matched_residues: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class PDBReadResult:
    structure: dict
    diagnostics: PDBReadDiagnostics


def _read_pdb_records(
    path: str,
    keep_altloc: str = "A",
    model: Optional[int] = None,
    altloc_selectors: list[str] | None = None,
) -> dict:
    from maple.function.dispatcher.parmfit.utils import structure as structure_utils

    residues_by_key: OrderedDict[tuple[str, int, str], dict] = OrderedDict()
    conect: dict[int, set[int]] = defaultdict(set)
    raw_links: list[dict] = []
    raw_serial_to_atoms: dict[int, list[dict]] = defaultdict(list)

    current_model = 1
    target_model = model
    saw_model = False
    in_model = False
    requested_altlocs = [
        selector
        for text in altloc_selectors or ()
        if (selector := structure_utils.parse_selector(text))["altloc"]
    ]

    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.rstrip("\r\n").ljust(80)
            record = line[:6].strip().upper()

            if record == "MODEL":
                saw_model = True
                in_model = True
                model_field = line[10:14].strip()
                current_model = int(model_field) if model_field else current_model
                if target_model is None:
                    target_model = current_model
                continue

            if record == "ENDMDL":
                in_model = False
                continue

            if record == "LINK":
                if not in_model or current_model == target_model:
                    parsed = _parse_link_line(line)
                    if parsed is not None:
                        raw_links.append(parsed)
                continue

            if record == "CONECT":
                if not in_model or current_model == target_model:
                    root, neighbors = _parse_conect_line(line)
                    if root is not None:
                        for neighbor in neighbors:
                            conect[root].add(neighbor)
                            conect[neighbor].add(root)
                continue

            if target_model is None:
                target_model = 1

            if saw_model and current_model != target_model:
                continue

            if record in {"END", "TER"}:
                continue

            if record not in {"ATOM", "HETATM", "HEATOM"}:
                continue

            chain = line[21].strip() or "_"
            resseq = int(line[22:26])
            icode = line[26].strip()
            key = (chain, resseq, icode)
            residue = residues_by_key.get(key)
            if residue is None:
                residue = {
                    "chain": chain,
                    "resseq": resseq,
                    "icode": icode,
                    "resname": line[17:20].strip(),
                    "source_resname": line[17:20].strip(),
                    "atoms": [],
                    "_altloc_sites": defaultdict(list),
                    "_index": len(residues_by_key),
                }
                residues_by_key[key] = residue

            raw_atom_name = line[12:16]
            atom_name = raw_atom_name.strip()
            occ_field = line[54:60].strip()
            atom = {
                "serial": 0,
                "_pdb_serial": int(line[6:11]),
                "name": atom_name,
                "source_name": atom_name,
                "element": _infer_pdb_element(
                    raw_atom_name,
                    line[76:78],
                    record=record,
                    resname=line[17:20],
                ).upper(),
                "xyz": np.array(parse_pdb_coord(line), dtype=float),
                "record": "HETATM" if record == "HEATOM" else record,
                "altloc": line[16].strip().upper(),
                "occ": float(occ_field) if occ_field else 1.0,
            }
            residue["_altloc_sites"][raw_atom_name].append(atom)

    residues: list[dict] = []
    serial_to_residue: dict[int, dict] = {}
    serial_to_atom: dict[int, dict] = {}
    next_serial = 1
    for residue in residues_by_key.values():
        matching_selectors = [
            selector
            for selector in requested_altlocs
            if selector["resseq"] == residue["resseq"]
            and (selector.get("chain") is None or selector["chain"] == residue["chain"])
            and (
                selector.get("resname") is None
                or selector["resname"] == residue["source_resname"].upper()
            )
        ]
        requested = {selector["altloc"] for selector in matching_selectors}
        if len(requested) > 1:
            raise ValueError(f"Conflicting altloc selectors for {residue['chain']}{residue['resseq']}.")
        available = {
            atom["altloc"]
            for candidates in residue["_altloc_sites"].values()
            for atom in candidates
            if atom["altloc"]
        }
        if requested:
            selected_altloc = requested.pop()
            if selected_altloc not in available:
                raise ValueError(
                    f"Requested altloc {selected_altloc} was not found for "
                    f"{residue['chain']}{residue['resseq']}:{residue['source_resname']}."
                )
        elif keep_altloc.upper() in available:
            selected_altloc = keep_altloc.upper()
        elif available:
            occupancy = {
                label: sum(
                    atom["occ"]
                    for candidates in residue["_altloc_sites"].values()
                    for atom in candidates
                    if atom["altloc"] == label
                )
                for label in available
            }
            selected_altloc = max(sorted(available), key=occupancy.get)
        else:
            selected_altloc = ""

        selected_atoms: list[dict] = []
        for candidates in residue.pop("_altloc_sites").values():
            compatible = [atom for atom in candidates if atom["altloc"] == selected_altloc]
            if not compatible:
                compatible = [atom for atom in candidates if not atom["altloc"]]
            if compatible:
                selected_atoms.append(max(compatible, key=lambda atom: atom["occ"]))
        residue["atoms"] = selected_atoms
        residue["selected_altloc"] = selected_altloc
        for atom in residue["atoms"]:
            atom["serial"] = next_serial
            raw_serial_to_atoms[atom["_pdb_serial"]].append(atom)
            serial_to_residue[next_serial] = residue
            serial_to_atom[next_serial] = atom
            next_serial += 1
        residue["coords"] = np.asarray([atom["xyz"] for atom in residue["atoms"]], dtype=float)
        residues.append(residue)

    explicit_pairs: set[tuple[int, int]] = set()
    for root, neighbors in conect.items():
        for neighbor in neighbors:
            left_atoms = raw_serial_to_atoms.get(root, ())
            right_atoms = raw_serial_to_atoms.get(neighbor, ())
            if len(left_atoms) == 1 and len(right_atoms) == 1:
                explicit_pairs.add(structure_utils._bond_pair(left_atoms[0]["serial"], right_atoms[0]["serial"]))

    for link in raw_links:
        left_selector = {
            "chain": link["left"]["chain"],
            "resseq": link["left"]["resseq"],
            "_icode": link["left"]["icode"],
        }
        right_selector = {
            "chain": link["right"]["chain"],
            "resseq": link["right"]["resseq"],
            "_icode": link["right"]["icode"],
        }
        left_resid = next((res for res in residues if structure_utils.match_resid(res, left_selector)), None)
        right_resid = next((res for res in residues if structure_utils.match_resid(res, right_selector)), None)
        if left_resid is None or right_resid is None:
            continue
        left_atom = structure_utils.search_atom(left_resid, link["left"]["atom"])
        right_atom = structure_utils.search_atom(right_resid, link["right"]["atom"])
        if left_atom is None or right_atom is None:
            continue
        explicit_pairs.add(structure_utils._bond_pair(left_atom["serial"], right_atom["serial"]))

    structure = {
        "path": path,
        "residues": residues,
        "serial_to_residue": serial_to_residue,
        "serial_to_atom": serial_to_atom,
        "pdb_serial_to_serial": {
            pdb_serial: atoms[0]["serial"]
            for pdb_serial, atoms in raw_serial_to_atoms.items()
            if len(atoms) == 1
        },
        "explicit_pairs": explicit_pairs,
    }
    for atom in serial_to_atom.values():
        atom.pop("_pdb_serial", None)
    return structure


def _candidate_pairs(structure: dict) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    from scipy.spatial import cKDTree

    from maple.function.dispatcher.parmfit.utils.structure import (
        ION_ELEMENTS,
        METAL_SITE_DONOR_ELEMENTS,
        covalent_cutoff,
    )

    atoms = list(structure["serial_to_atom"].values())
    if len(atoms) < 2:
        return set(structure["explicit_pairs"]), set()
    positions = np.asarray([atom["xyz"] for atom in atoms], dtype=float)
    tree = cKDTree(positions)
    covalent: set[tuple[int, int]] = set()
    coordination: set[tuple[int, int]] = set()
    for left_index, right_index in tree.query_pairs(4.0):
        left = atoms[left_index]
        right = atoms[right_index]
        distance = float(np.linalg.norm(left["xyz"] - right["xyz"]))
        left_is_metal = left["element"] in ION_ELEMENTS
        right_is_metal = right["element"] in ION_ELEMENTS
        pair = tuple(sorted((left["serial"], right["serial"])))
        if left_is_metal != right_is_metal:
            donor = right if left_is_metal else left
            if donor["element"] in METAL_SITE_DONOR_ELEMENTS and distance <= 4.0:
                coordination.add(pair)
            continue
        if not left_is_metal and distance <= covalent_cutoff(left, right):
            covalent.add(pair)
    for pair in structure["explicit_pairs"]:
        left = structure["serial_to_atom"][pair[0]]
        right = structure["serial_to_atom"][pair[1]]
        if (left["element"] in ION_ELEMENTS) != (right["element"] in ION_ELEMENTS):
            coordination.add(pair)
        else:
            covalent.add(pair)
    return covalent, coordination


def _polymer_connection(residue: dict, serial: int) -> str | None:
    connect_atoms = residue.get("connect_atoms", ())
    if connect_atoms:
        if len(connect_atoms) >= 1 and serial == connect_atoms[0]:
            return "head"
        if len(connect_atoms) >= 2 and serial == connect_atoms[1]:
            return "tail"
        return None
    atom = next(atom for atom in residue["atoms"] if atom["serial"] == serial)
    if residue.get("kind") == "protein":
        role = atom.get("role", atom["name"])
        if role == "N":
            return "head"
        if role == "C":
            return "tail"
    return None


def _cross_residue_bond_allowed(structure: dict, pair: tuple[int, int]) -> bool:
    left_atom = structure["serial_to_atom"][pair[0]]
    right_atom = structure["serial_to_atom"][pair[1]]
    left_residue = structure["serial_to_residue"][pair[0]]
    right_residue = structure["serial_to_residue"][pair[1]]
    if left_atom["element"] == right_atom["element"] == "S":
        return (
            left_residue.get("kind") == right_residue.get("kind") == "protein"
            and left_atom.get("role", left_atom["name"]) == "SG"
            and right_atom.get("role", right_atom["name"]) == "SG"
            and float(np.linalg.norm(left_atom["xyz"] - right_atom["xyz"])) <= 2.3
        )
    if left_residue.get("kind") != right_residue.get("kind"):
        return False
    if left_residue.get("kind") not in {"protein", "nucleic"}:
        return False
    if left_residue["chain"] != right_residue["chain"]:
        return False
    left_index = int(left_residue["_index"])
    right_index = int(right_residue["_index"])
    lower, upper = sorted((left_index, right_index))
    if any(
        residue.get("kind") == left_residue.get("kind")
        and residue["chain"] == left_residue["chain"]
        and lower < int(residue["_index"]) < upper
        for residue in structure["residues"]
    ):
        return False
    left_connection = _polymer_connection(left_residue, pair[0])
    right_connection = _polymer_connection(right_residue, pair[1])
    if {left_connection, right_connection} != {"head", "tail"}:
        return False
    if left_residue.get("kind") == "protein":
        return float(np.linalg.norm(left_atom["xyz"] - right_atom["xyz"])) <= 1.65
    return True


def read_pdb_result(
    path: str,
    keep_altloc: str = "A",
    model: Optional[int] = None,
    *,
    prom: str = "ff14SB",
    altloc_selectors: list[str] | None = None,
) -> PDBReadResult:
    from maple.function.dispatcher.parmfit.utils.amber_templates import load_amber_template_registry
    from maple.function.dispatcher.parmfit.utils.residue_matcher import (
        apply_template_match,
        match_residue_template,
        match_peptide_backbone,
    )
    from maple.function.dispatcher.parmfit.utils.structure import classify_kind, get_resid_label

    structure = _read_pdb_records(
        path,
        keep_altloc=keep_altloc,
        model=model,
        altloc_selectors=altloc_selectors,
    )
    candidate_pairs, coordination_pairs = _candidate_pairs(structure)
    registry = load_amber_template_registry(prom)

    disulfide_serials: set[int] = set()
    for pair in candidate_pairs:
        left = structure["serial_to_atom"][pair[0]]
        right = structure["serial_to_atom"][pair[1]]
        if (
            left["element"] == "S"
            and right["element"] == "S"
            and structure["serial_to_residue"][pair[0]] is not structure["serial_to_residue"][pair[1]]
            and float(np.linalg.norm(left["xyz"] - right["xyz"])) <= 2.3
        ):
            disulfide_serials.update(pair)

    bond_pairs: set[tuple[int, int]] = set()
    matched = 0
    backbone_only = 0
    unmatched = 0
    warnings: list[str] = []
    backbone_only_labels: list[str] = []
    not_matched_labels: list[str] = []
    pending: list[dict] = []

    for residue in structure["residues"]:
        initial_kind = classify_kind(residue)
        if initial_kind in {"water", "ion"}:
            residue["kind"] = initial_kind
            if initial_kind == "water":
                residue["net_charge"] = 0
            serials = {atom["serial"] for atom in residue["atoms"]}
            bond_pairs.update(pair for pair in candidate_pairs if pair[0] in serials and pair[1] in serials)
            continue
        match = match_residue_template(
            residue,
            candidate_pairs,
            registry,
            disulfide_serials=disulfide_serials,
        )
        if match is not None:
            bond_pairs.update(apply_template_match(residue, match))
            matched += 1
            continue
        pending.append(residue)

    for residue in pending:
        before = [
            other
            for other in structure["residues"]
            if other.get("kind") not in {"water", "ion"}
            and other["chain"] == residue["chain"]
            and int(other["_index"]) < int(residue["_index"])
        ]
        after = [
            other
            for other in structure["residues"]
            if other.get("kind") not in {"water", "ion"}
            and other["chain"] == residue["chain"]
            and int(other["_index"]) > int(residue["_index"])
        ]
        previous = max(before, key=lambda other: int(other["_index"])) if before else None
        following = min(after, key=lambda other: int(other["_index"])) if after else None
        target_serials = {atom["serial"] for atom in residue["atoms"]}
        previous_carbons = {
            atom["serial"]
            for atom in previous["atoms"]
            if atom["element"] == "C"
            and any(tuple(sorted((atom["serial"], serial))) in candidate_pairs for serial in target_serials)
        } if previous else set()
        next_nitrogens = {
            atom["serial"]
            for atom in following["atoms"]
            if atom["element"] == "N"
            and any(tuple(sorted((atom["serial"], serial))) in candidate_pairs for serial in target_serials)
        } if following else set()
        if match_peptide_backbone(
            residue,
            candidate_pairs,
            previous_carbons=previous_carbons,
            next_nitrogens=next_nitrogens,
        ):
            residue["kind"] = "protein"
            residue["net_charge"] = 0
            backbone_only += 1
            backbone_only_labels.append(get_resid_label(residue))
            warnings.append(f"{residue['chain']}{residue['resseq']}:{residue['resname']} matched peptide backbone only; net charge defaults to 0.")
        else:
            residue["kind"] = classify_kind(residue)
            unmatched += 1
            not_matched_labels.append(get_resid_label(residue))
        serials = {atom["serial"] for atom in residue["atoms"]}
        bond_pairs.update(pair for pair in candidate_pairs if pair[0] in serials and pair[1] in serials)

    serial_to_residue = structure["serial_to_residue"]
    for pair in candidate_pairs:
        left_residue = serial_to_residue[pair[0]]
        right_residue = serial_to_residue[pair[1]]
        if left_residue is right_residue:
            continue
        if _cross_residue_bond_allowed(structure, pair):
            bond_pairs.add(pair)
    bond_pairs.update(
        pair
        for pair in structure["explicit_pairs"]
        if pair not in coordination_pairs
    )
    structure["bond_pairs"] = bond_pairs
    structure["coordination_pairs"] = coordination_pairs
    structure["coordination_cutoff"] = 4.0
    return PDBReadResult(
        structure=structure,
        diagnostics=PDBReadDiagnostics(
            matched=matched,
            backbone_only=backbone_only,
            unmatched=unmatched,
            warnings=tuple(warnings),
            backbone_only_residues=tuple(backbone_only_labels),
            not_matched_residues=tuple(not_matched_labels),
        ),
    )


def read_pdb(
    path: str,
    keep_altloc: str = "A",
    model: Optional[int] = None,
    *,
    prom: str = "ff14SB",
    altloc_selectors: list[str] | None = None,
) -> dict:
    return read_pdb_result(
        path,
        keep_altloc=keep_altloc,
        model=model,
        prom=prom,
        altloc_selectors=altloc_selectors,
    ).structure
