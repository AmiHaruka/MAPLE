"""Usage: write RESP inputs, read RESP charges, and export charged mol2 files."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re

from .context import find_unique_residue
from .model import flatten_model_atoms, infer_bond_pairs
from .structure import get_resid_key


RESP_ATOMIC_NUMBERS = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "NA": 11,
    "MG": 12,
    "P": 15,
    "S": 16,
    "CL": 17,
    "K": 19,
    "CA": 20,
    "MN": 25,
    "FE": 26,
    "CO": 27,
    "NI": 28,
    "CU": 29,
    "ZN": 30,
    "SE": 34,
    "BR": 35,
    "I": 53,
}

_BACKBONE_FIXED_ATOMS = {
    0: set(),
    1: {"CA", "N", "C", "O", "OXT"},
    2: {"CA", "H", "HA", "N", "C", "O", "OXT"},
    3: {"CA", "H", "HA", "N", "C", "O", "CB", "OXT"},
}

_CHARGE_LIBRARY_FILES = {
    "internal": "amino19.lib",
    "nterm": "aminont12.lib",
    "cterm": "aminoct12.lib",
}

_ATOM_NAME_ALIASES = {
    "HN": ("H",),
    "CMA": ("CH3",),
    "CAC": ("C",),
    "OAC": ("O",),
    "H1A": ("H1",),
    "H2A": ("H2",),
    "H3A": ("H3",),
    "NNM": ("N",),
    "CNM": ("C",),
    "HNM": ("H",),
    "H1M": ("H1",),
    "H2M": ("H2",),
    "H3M": ("H3",),
}


@dataclass(frozen=True)
class RespInputFiles:
    resp1_in: str
    resp2_in: str


def _params_lib_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "params" / "leap" / "lib"


def _flatten_model_atoms(model: dict) -> list[tuple[dict, dict]]:
    return flatten_model_atoms(model)


def _flattened_atoms_and_adjacency(model: dict) -> tuple[list[tuple[dict, dict]], list[dict], dict[int, set[int]]]:
    flattened = _flatten_model_atoms(model)
    atoms = [atom for _, atom in flattened]
    adjacency: dict[int, set[int]] = {index: set() for index in range(1, len(atoms) + 1)}
    for left, right in infer_bond_pairs(model):
        adjacency[left].add(right)
        adjacency[right].add(left)
    return flattened, atoms, adjacency


def _candidate_atom_names(atom_name: str) -> list[str]:
    names = [atom_name]
    for alias in _ATOM_NAME_ALIASES.get(atom_name, ()):
        if alias not in names:
            names.append(alias)
    return names


def _hydrogen_name_prefix(atom_name: str) -> str:
    match = re.match(r"([A-Za-z]+)", atom_name)
    return match.group(1) if match is not None else atom_name


def _parse_lib_atoms(path: Path) -> dict[str, dict[str, tuple[str, float]]]:
    residues: dict[str, dict[str, tuple[str, float]]] = {}
    current_residue: str | None = None
    in_atoms_table = False

    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if line.startswith("!entry.") and ".unit.atoms table" in line:
                match = re.match(r"!entry\.([^.]+)\.unit\.atoms table", line)
                if match is None:
                    current_residue = None
                    in_atoms_table = False
                    continue
                current_residue = match.group(1).upper()
                residues[current_residue] = {}
                in_atoms_table = True
                continue
            if line.startswith("!entry."):
                in_atoms_table = False
                current_residue = None
                continue
            if not in_atoms_table or current_residue is None:
                continue
            stripped = line.strip()
            if not stripped:
                continue
            tokens = re.findall(r'"[^"]*"|\S+', stripped)
            if len(tokens) < 8:
                continue
            atom_name = tokens[0].strip('"')
            atom_type = tokens[1].strip('"')
            charge = float(tokens[-1])
            residues[current_residue][atom_name] = (atom_type, charge)
    return residues


def _library_residue_names(resname: str, category: str) -> list[str]:
    upper = resname.upper()
    names: list[str] = []
    if category == "nterm":
        base = upper[1:] if upper.startswith("N") and len(upper) > 1 else upper
        names.extend([f"N{base}", upper, base])
    elif category == "cterm":
        base = upper[1:] if upper.startswith("C") and len(upper) > 1 else upper
        names.extend([f"C{base}", upper, base])
    else:
        names.append(upper)
        if upper.startswith(("N", "C")) and len(upper) > 1:
            names.append(upper[1:])

    ordered: list[str] = []
    for name in names:
        if name not in ordered:
            ordered.append(name)
    return ordered


@lru_cache(maxsize=1)
def load_reference_charge_library() -> dict[str, dict[str, dict[str, tuple[str, float]]]]:
    base = _params_lib_dir()
    return {
        category: _parse_lib_atoms(base / filename)
        for category, filename in _CHARGE_LIBRARY_FILES.items()
    }


def _reference_category(residue: dict, library: dict[str, dict[str, dict[str, tuple[str, float]]]]) -> str | None:
    if residue.get("kind") != "protein":
        return None
    resname = residue["resname"].upper()
    if residue.get("_prev_peptide_key") is None and residue.get("_next_peptide_key") is not None:
        return "nterm" if any(name in library["nterm"] for name in _library_residue_names(resname, "nterm")) else None
    if residue.get("_next_peptide_key") is None and residue.get("_prev_peptide_key") is not None:
        return "cterm" if any(name in library["cterm"] for name in _library_residue_names(resname, "cterm")) else None
    if residue.get("_prev_peptide_key") is not None and residue.get("_next_peptide_key") is not None:
        return "internal" if any(name in library["internal"] for name in _library_residue_names(resname, "internal")) else None
    return None


def _lookup_reference_entry(
    residue: dict,
    atom: dict,
    library: dict[str, dict[str, dict[str, tuple[str, float]]]],
) -> tuple[str, float] | None:
    category = _reference_category(residue, library)
    if category is None:
        return None
    for residue_name in _library_residue_names(residue["resname"], category):
        residue_entries = library[category].get(residue_name)
        if residue_entries is None:
            continue
        for atom_name in _candidate_atom_names(atom["name"]):
            entry = residue_entries.get(atom_name)
            if entry is not None:
                return entry
    return None


def lookup_standard_residue_entry(*, resname: str, atom_name: str, category: str) -> tuple[str, float] | None:
    library = load_reference_charge_library()
    for residue_name in _library_residue_names(resname, category):
        residue_entries = library.get(category, {}).get(residue_name)
        if residue_entries is None:
            continue
        for candidate_name in _candidate_atom_names(atom_name):
            entry = residue_entries.get(candidate_name)
            if entry is not None:
                return entry
    return None


def lookup_standard_atom_entry(residue: dict, atom: dict, *, category: str | None = None) -> tuple[str, float] | None:
    library = load_reference_charge_library()
    if category is not None:
        return lookup_standard_residue_entry(resname=residue["resname"], atom_name=atom["name"], category=category)
    return _lookup_reference_entry(residue, atom, library)


def _resolve_fixchg_residue_keys(model: dict, selectors: list[str] | None) -> set[tuple[str, int, str]]:
    keys: set[tuple[str, int, str]] = set()
    for selector in selectors or []:
        residue = find_unique_residue(model, selector, label="fixchg residue")
        keys.add(get_resid_key(residue))
    return keys


def _backbone_atom_matches(atom_name: str, allowed: set[str]) -> bool:
    if atom_name in allowed:
        return True
    if atom_name == "HN" and "H" in allowed:
        return True
    return False


def collect_fixed_charge_constraints(
    model: dict,
    *,
    chgmod: int,
    fixchg_resids: list[str] | None = None,
) -> dict[int, float]:
    library = load_reference_charge_library()
    fixed_keys = _resolve_fixchg_residue_keys(model, fixchg_resids)
    constraints: dict[int, float] = {}
    allowed_backbone_names = _BACKBONE_FIXED_ATOMS[int(chgmod)]

    for atom_index, (residue, atom) in enumerate(_flatten_model_atoms(model), start=1):
        residue_key = get_resid_key(residue)
        entry = _lookup_reference_entry(residue, atom, library)

        if residue_key in fixed_keys:
            if entry is None:
                raise ValueError(
                    f"fixchg residue {residue['chain']}{residue['resseq']}{residue['icode']}:{residue['resname']} "
                    "does not have a supported amber reference charge definition."
                )
            constraints[atom_index] = float(entry[1])
            continue

        if not allowed_backbone_names:
            continue
        if _reference_category(residue, library) != "internal":
            continue
        if not _backbone_atom_matches(atom["name"], allowed_backbone_names):
            continue
        if entry is None:
            continue
        constraints[atom_index] = float(entry[1])

    return constraints


def build_stage2_equivalence_map(model: dict, *, fixed_charge_indices: set[int] | None = None) -> dict[int, int]:
    fixed_charge_indices = fixed_charge_indices or set()
    _, atoms, adjacency = _flattened_atoms_and_adjacency(model)

    ivary: dict[int, int] = {index: 0 for index in range(1, len(atoms) + 1)}
    for heavy_index, atom in enumerate(atoms, start=1):
        if atom["element"] == "H":
            continue
        hydrogens = sorted(
            neighbor
            for neighbor in adjacency[heavy_index]
            if atoms[neighbor - 1]["element"] == "H" and neighbor not in fixed_charge_indices
        )
        groups: dict[str, list[int]] = {}
        for hydrogen in hydrogens:
            prefix = _hydrogen_name_prefix(atoms[hydrogen - 1]["name"])
            groups.setdefault(prefix, []).append(hydrogen)
        for grouped_hydrogens in groups.values():
            if len(grouped_hydrogens) not in {2, 3}:
                continue
            root = grouped_hydrogens[0]
            for hydrogen in grouped_hydrogens[1:]:
                ivary[hydrogen] = root
    return ivary


def _resp_header(total_charge: int, n_atoms: int, *, stage: int) -> str:
    qwt = "0.00050" if stage == 1 else "0.00100"
    lines = [
        "Resp charges for organic molecule",
        " ",
        " &cntrl",
        " ",
        " nmol = 1,",
        " ihfree = 1,",
        " ioutopt = 1,",
    ]
    if stage == 2:
        lines.append(" iqopt = 2,")
    lines.extend(
        [
            f" qwt = {qwt},",
            " ",
            " &end",
            "    1.0",
            "Resp charges for organic molecule",
            f"{int(total_charge):5d} {int(n_atoms):4d}",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_charge_constraints(
    handle,
    constraints: dict[int, float],
    charge_groups: list[tuple[list[int], float]] | None = None,
) -> None:
    for atom_indices, target_charge in charge_groups or []:
        _write_group_constraint(handle, [(1, atom_index) for atom_index in atom_indices], target_charge)
    for atom_index in sorted(constraints):
        handle.write(f"{1:5d}{constraints[atom_index]:10.5f}\n")
        handle.write(f"{1:5d}{atom_index:5d}\n")
    handle.write("\n\n")


def _write_group_constraint(handle, atom_pairs: list[tuple[int, int]], target_charge: float) -> None:
    handle.write(f"{len(atom_pairs):5d}{float(target_charge):10.5f}\n")
    values: list[int] = []
    for molecule_index, atom_index in atom_pairs:
        values.extend((molecule_index, atom_index))
    for start in range(0, len(values), 16):
        handle.write("".join(f"{value:5d}" for value in values[start : start + 16]) + "\n")


def _write_interstructure_equivalencing(handle, atom_groups: list[list[tuple[int, int]]]) -> None:
    for atom_group in atom_groups:
        handle.write(f"{len(atom_group):5d}\n")
        values: list[int] = []
        for molecule_index, atom_index in atom_group:
            values.extend((molecule_index, atom_index))
        for start in range(0, len(values), 16):
            handle.write("".join(f"{value:5d}" for value in values[start : start + 16]) + "\n")
    handle.write("\n")


def merge_esp_files(esp_files: list[str], output_path: str) -> str:
    with open(output_path, "wb") as out_handle:
        for esp_file in esp_files:
            with open(esp_file, "rb") as in_handle:
                out_handle.write(in_handle.read())
    return output_path


def _model_signature(model: dict) -> list[tuple[str, str, str]]:
    return [
        (residue["resname"], atom["name"], atom["element"])
        for residue, atom in _flatten_model_atoms(model)
    ]


def _group_equivalent_hydrogens(model: dict) -> dict[int, int]:
    _, atoms, adjacency = _flattened_atoms_and_adjacency(model)

    ivary: dict[int, int] = {index: 0 for index in range(1, len(atoms) + 1)}
    for heavy_index, atom in enumerate(atoms, start=1):
        if atom["element"] != "C":
            continue
        hydrogens = sorted(
            neighbor for neighbor in adjacency[heavy_index] if atoms[neighbor - 1]["element"] == "H"
        )
        if len(hydrogens) not in {2, 3}:
            continue
        root = hydrogens[0]
        for hydrogen in hydrogens[1:]:
            ivary[hydrogen] = root
    return ivary


def _free_stage2_atoms(model: dict) -> tuple[set[int], dict[int, int]]:
    _, atoms, adjacency = _flattened_atoms_and_adjacency(model)

    free_atoms: set[int] = set()
    hydrogen_roots: dict[int, int] = {}
    for carbon_index, atom in enumerate(atoms, start=1):
        if atom["element"] != "C":
            continue
        hydrogens = sorted(
            neighbor for neighbor in adjacency[carbon_index] if atoms[neighbor - 1]["element"] == "H"
        )
        if len(hydrogens) not in {2, 3}:
            continue
        free_atoms.add(carbon_index)
        root = hydrogens[0]
        free_atoms.update(hydrogens)
        hydrogen_roots[root] = 0
        for hydrogen in hydrogens[1:]:
            hydrogen_roots[hydrogen] = root
    return free_atoms, hydrogen_roots


def _write_multiconformer_resp_stage(
    handle,
    *,
    models: list[dict],
    ivary_blocks: list[dict[int, int]],
    residue_indices: list[int],
    total_charge: int,
    qwt: str,
    include_iqopt: bool,
) -> None:
    n_models = len(models)
    handle.write("Resp charges for organic molecule\n \n &cntrl\n \n")
    handle.write(f" nmol = {n_models},\n")
    handle.write(" ihfree = 1,\n")
    handle.write(" ioutopt = 1,\n")
    if include_iqopt:
        handle.write(" iqopt = 2,\n")
    handle.write(f" qwt = {qwt},\n \n &end\n")
    for model_index, model in enumerate(models, start=1):
        handle.write("    1.0\n")
        handle.write(f"{model.get('label', f'mol{model_index}')}\n")
        flattened = _flatten_model_atoms(model)
        handle.write(f"{int(model.get('charge', 0)):5d} {len(flattened):4d}\n")
        for atom_index, (_, atom) in enumerate(flattened, start=1):
            atomic_number = RESP_ATOMIC_NUMBERS.get(atom["element"].upper())
            if atomic_number is None:
                raise ValueError(f"Unsupported atomic element {atom['element']!r}.")
            handle.write(f"{atomic_number:5d}{ivary_blocks[model_index - 1][atom_index]:5d}\n")
        handle.write("\n")

    for model_index in range(1, n_models + 1):
        _write_group_constraint(
            handle,
            [(model_index, atom_index) for atom_index in residue_indices],
            float(total_charge),
        )
    handle.write("\n")

    atom_groups: list[list[tuple[int, int]]] = []
    for atom_index in range(1, len(_flatten_model_atoms(models[0])) + 1):
        if include_iqopt and ivary_blocks[0][atom_index] == -1:
            continue
        atom_groups.append([(model_index, atom_index) for model_index in range(1, n_models + 1)])
    _write_interstructure_equivalencing(handle, atom_groups)


def write_multiconformer_resp_input_files(
    workdir: str,
    models: list[dict],
    *,
    labels: list[str] | None = None,
    total_charge: int,
    residue_key: tuple[str, int, str],
) -> RespInputFiles:
    if not models:
        raise ValueError("Multiconformer RESP requires at least one model.")

    signature = _model_signature(models[0])
    for model in models[1:]:
        if _model_signature(model) != signature:
            raise ValueError("All multiconformer RESP models must have identical residue/atom ordering.")

    n_models = len(models)
    base_stage1 = _group_equivalent_hydrogens(models[0])
    stage1_blocks = [dict(base_stage1) for _ in range(n_models)]

    free_atoms, hydrogen_roots = _free_stage2_atoms(models[0])
    base_stage2 = {index: -1 for index in range(1, len(signature) + 1)}
    for atom_index in sorted(free_atoms):
        base_stage2[atom_index] = hydrogen_roots.get(atom_index, 0)
    stage2_blocks = [dict(base_stage2) for _ in range(n_models)]

    residue_indices = [
        atom_index
        for atom_index, (residue, _) in enumerate(_flatten_model_atoms(models[0]), start=1)
        if get_resid_key(residue) == residue_key
    ]
    if not residue_indices:
        raise ValueError("Residue charge constraint requires at least one target residue atom.")

    resolved_labels = labels or [f"conf{index}" for index in range(1, n_models + 1)]
    if len(resolved_labels) != n_models:
        raise ValueError("Multiconformer RESP labels must match the number of models.")
    labeled_models = []
    for model, label in zip(models, resolved_labels, strict=True):
        labeled_models.append(dict(model, label=label))

    resp1_in = Path(workdir) / "resp1.in"
    resp2_in = Path(workdir) / "resp2.in"

    with open(resp1_in, "w", encoding="utf-8") as handle:
        _write_multiconformer_resp_stage(
            handle,
            models=labeled_models,
            ivary_blocks=stage1_blocks,
            residue_indices=residue_indices,
            total_charge=total_charge,
            qwt="0.00050",
            include_iqopt=False,
        )

    with open(resp2_in, "w", encoding="utf-8") as handle:
        _write_multiconformer_resp_stage(
            handle,
            models=labeled_models,
            ivary_blocks=stage2_blocks,
            residue_indices=residue_indices,
            total_charge=total_charge,
            qwt="0.00100",
            include_iqopt=True,
        )

    return RespInputFiles(resp1_in=str(resp1_in), resp2_in=str(resp2_in))


def write_resp_input_files(
    workdir: str,
    model: dict,
    *,
    total_charge: int,
    chgmod: int,
    fixchg_resids: list[str] | None = None,
    charge_groups: list[tuple[list[int], float]] | None = None,
) -> RespInputFiles:
    flattened = _flatten_model_atoms(model)
    constraints = collect_fixed_charge_constraints(model, chgmod=chgmod, fixchg_resids=fixchg_resids)
    ivary_stage2 = build_stage2_equivalence_map(model, fixed_charge_indices=set(constraints))

    resp1_in = Path(workdir) / "resp1.in"
    resp2_in = Path(workdir) / "resp2.in"

    with open(resp1_in, "w", encoding="utf-8") as handle:
        handle.write(_resp_header(total_charge, len(flattened), stage=1))
        for _, atom in flattened:
            atomic_number = RESP_ATOMIC_NUMBERS.get(atom["element"].upper())
            if atomic_number is None:
                raise ValueError(f"Unsupported RESP atomic element {atom['element']!r}.")
            handle.write(f"{atomic_number:5d}{0:5d}\n")
        _write_charge_constraints(handle, constraints, charge_groups)

    with open(resp2_in, "w", encoding="utf-8") as handle:
        handle.write(_resp_header(total_charge, len(flattened), stage=2))
        for atom_index, (_, atom) in enumerate(flattened, start=1):
            atomic_number = RESP_ATOMIC_NUMBERS.get(atom["element"].upper())
            if atomic_number is None:
                raise ValueError(f"Unsupported RESP atomic element {atom['element']!r}.")
            handle.write(f"{atomic_number:5d}{ivary_stage2[atom_index]:5d}\n")
        _write_charge_constraints(handle, constraints, charge_groups)

    return RespInputFiles(resp1_in=str(resp1_in), resp2_in=str(resp2_in))


def read_resp_charges(path: str) -> list[float]:
    charges: list[float] = []
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            for token in line.split():
                charges.append(float(token))
    return charges


def apply_resp_charges(model: dict, charges: list[float]) -> dict:
    flattened = _flatten_model_atoms(model)
    if len(charges) != len(flattened):
        raise ValueError(f"RESP returned {len(charges)} charges for a model with {len(flattened)} atoms.")

    updated = deepcopy(model)
    charge_index = 0
    for residue in updated["residues"]:
        for atom in sorted(residue["atoms"], key=lambda item: item["serial"]):
            atom["charge"] = float(charges[charge_index])
            charge_index += 1
    return updated


def _default_atom_type(atom: dict) -> str:
    return atom["element"].lower()


def _mol2_atom_type(
    residue: dict,
    atom: dict,
    library: dict[str, dict[str, dict[str, tuple[str, float]]]],
) -> str:
    explicit = atom.get("atom_type")
    if explicit:
        return str(explicit)
    entry = _lookup_reference_entry(residue, atom, library)
    if entry is not None:
        return entry[0].strip() or _default_atom_type(atom)
    return _default_atom_type(atom)


def write_resp_mol2(
    path: str,
    model: dict,
    bond_pairs: list[tuple[int, int]],
    *,
    atom_type_overrides: dict[int, str] | None = None,
) -> None:
    library = load_reference_charge_library()
    flattened = _flatten_model_atoms(model)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("@<TRIPOS>MOLECULE\n")
        handle.write(f"{Path(path).stem}\n")
        handle.write(f"{len(flattened):5d}{len(bond_pairs):6d}{1:6d}{0:6d}{0:6d}\n")
        handle.write("SMALL\n")
        handle.write("USER_CHARGES\n\n\n")
        handle.write("@<TRIPOS>ATOM\n")
        for atom_index, (residue, atom) in enumerate(flattened, start=1):
            x, y, z = atom["xyz"]
            charge = float(atom.get("charge", 0.0))
            atom_type = (
                atom_type_overrides[atom_index]
                if atom_type_overrides is not None and atom_index in atom_type_overrides
                else _mol2_atom_type(residue, atom, library)
            )
            handle.write(
                f"{atom_index:7d} {atom['name']:<4s} {x:10.4f}{y:10.4f}{z:10.4f} "
                f"{atom_type:<4s} {1:6d} {residue['resname']:<4s} {charge:12.6f}\n"
            )
        handle.write("@<TRIPOS>BOND\n")
        for bond_id, (left, right) in enumerate(bond_pairs, start=1):
            handle.write(f"{bond_id:6d}{left:5d}{right:5d}{1:2d}\n")
        handle.write("@<TRIPOS>SUBSTRUCTURE\n")
        handle.write(f"{1:6d} {'MTS':<4s} {1:8d} TEMP              0 ****  ****    0 ROOT\n")
