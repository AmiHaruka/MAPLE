"""PDB file block reader and parser."""

from __future__ import annotations

import os
from collections import OrderedDict, defaultdict
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


def _infer_pdb_element(atom_name: str, element_field: str = "") -> str:
    field = element_field.strip()
    if field:
        return field[:2].strip().capitalize()

    stripped = atom_name.strip()
    letters = "".join(ch for ch in stripped if ch.isalpha())
    if not letters:
        return "X"
    if len(letters) >= 2 and letters[:2].upper() in {"CL", "BR", "NA", "MG", "ZN", "FE", "MN", "CO", "NI", "CU"}:
        return letters[:2].capitalize()
    return letters[0].upper()


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
                line = raw.rstrip("\n").ljust(80)
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
                        "element": _infer_pdb_element(atom_name, line[76:78]),
                        "xyz": parse_pdb_coord(line),
                    }
                )

        if not saw_model:
            finish_frame()
        elif in_model:
            finish_frame()

        return frames


def _format_pdb_atom_line(line: str, xyz: np.ndarray) -> str:
    padded = line.rstrip("\n").ljust(80)
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
        line = raw.rstrip("\n").ljust(80)
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
            handle.write(line.rstrip("\n") + "\n")
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
        handle.write(line.rstrip("\n") + "\n")
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
    padded = line.rstrip("\n").ljust(80)
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


def read_pdb(path: str, keep_altloc: str = "A", model: Optional[int] = None) -> dict:
    from maple.function.dispatcher.parmfit.utils import structure as structure_utils

    residues_by_key: OrderedDict[tuple[str, int, str], dict] = OrderedDict()
    conect: dict[int, set[int]] = defaultdict(set)
    raw_links: list[dict] = []
    serial_to_residue: dict[int, dict] = {}
    serial_to_atom: dict[int, dict] = {}

    current_model = 1
    target_model = model
    saw_model = False

    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.rstrip("\n").ljust(80)
            record = line[:6].strip().upper()

            if record == "MODEL":
                saw_model = True
                model_field = line[10:14].strip()
                current_model = int(model_field) if model_field else current_model
                if target_model is None:
                    target_model = current_model
                continue

            if target_model is None:
                target_model = 1

            if saw_model and current_model != target_model:
                continue

            if record in {"ENDMDL", "END", "TER"}:
                continue

            if record == "LINK":
                parsed = _parse_link_line(line)
                if parsed is not None:
                    raw_links.append(parsed)
                continue

            if record == "CONECT":
                root, neighbors = _parse_conect_line(line)
                if root is not None:
                    for neighbor in neighbors:
                        conect[root].add(neighbor)
                        conect[neighbor].add(root)
                continue

            if record not in {"ATOM", "HETATM", "HEATOM"}:
                continue

            altloc = line[16].strip()
            if altloc and altloc not in {"A", keep_altloc}:
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
                    "atoms": {},
                    "_index": len(residues_by_key),
                }
                residues_by_key[key] = residue

            atom_name = line[12:16].strip()
            occ_field = line[54:60].strip()
            atom = {
                "serial": int(line[6:11]),
                "name": atom_name,
                "element": structure_utils._infer_element(atom_name, line[76:78]),
                "xyz": np.array(parse_pdb_coord(line), dtype=float),
                "record": "HETATM" if record == "HEATOM" else record,
                "altloc": altloc,
                "occ": float(occ_field) if occ_field else 1.0,
            }
            current = residue["atoms"].get(atom_name)
            if current is None or atom["occ"] >= current["occ"]:
                residue["atoms"][atom_name] = atom
                serial_to_residue[atom["serial"]] = residue
                serial_to_atom[atom["serial"]] = atom

    residues: list[dict] = []
    for residue in residues_by_key.values():
        residue["atoms"] = sorted(residue["atoms"].values(), key=lambda atom: atom["serial"])
        residue["kind"] = structure_utils.classify_kind(residue)
        residue["coords"] = np.asarray([atom["xyz"] for atom in residue["atoms"]], dtype=float)
        residues.append(residue)

    explicit_pairs: set[tuple[int, int]] = set()
    for root, neighbors in conect.items():
        for neighbor in neighbors:
            if root in serial_to_atom and neighbor in serial_to_atom:
                explicit_pairs.add(structure_utils._bond_pair(root, neighbor))

    for link in raw_links:
        left_selector = {
            "chain": link["left"]["chain"],
            "resseq": link["left"]["resseq"],
            "icode": link["left"]["icode"],
        }
        right_selector = {
            "chain": link["right"]["chain"],
            "resseq": link["right"]["resseq"],
            "icode": link["right"]["icode"],
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

    return {
        "path": path,
        "residues": residues,
        "serial_to_residue": serial_to_residue,
        "serial_to_atom": serial_to_atom,
        "explicit_pairs": explicit_pairs,
        "_pair_cache": {},
    }
