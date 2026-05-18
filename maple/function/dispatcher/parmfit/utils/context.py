"""Usage: locate target residues and neighboring context for parmfit workflows."""

from __future__ import annotations

from .structure import (
    get_resid_info,
    get_resid_key,
    get_resid_label,
    get_resid_mindist,
    is_peptide_like,
    match_chain,
    match_resid,
    parse_selector,
    peptide_link,
    read_pdb,
)


def find_unique_residue(structure: dict, selector: str, *, label: str = "Residue") -> dict:
    parsed = parse_selector(selector)
    selector_label = f"{parsed.get('resname') or parsed['chain']}{parsed['resseq']}{parsed['icode']}"
    matches = [residue for residue in structure["residues"] if match_resid(residue, parsed)]
    if not matches:
        raise ValueError(f"{label} {selector_label} was not found.")
    if len(matches) > 1:
        labels = ", ".join(get_resid_label(residue) for residue in matches[:5])
        raise ValueError(
            f"Selector {selector_label} matched multiple residues: {labels}."
        )
    return matches[0]


def find_residue_by_key(structure: dict, residue_key: tuple[str, int, str], *, label: str = "Residue") -> dict:
    for residue in structure["residues"]:
        if get_resid_key(residue) == residue_key:
            return residue
    chain, resseq, icode = residue_key
    suffix = icode or ""
    raise ValueError(f"{label} {chain}{resseq}{suffix} was not found.")


def find_keep_residues(structure: dict, keep: list[str], target_residue: dict) -> list[dict]:
    keep_residues: list[dict] = []
    seen_keys: set[tuple[str, int, str]] = {get_resid_key(target_residue)}
    for selector in keep or []:
        residue = find_unique_residue(structure, selector, label="Keep residue")
        key = get_resid_key(residue)
        if key in seen_keys:
            continue
        keep_residues.append(residue)
        seen_keys.add(key)
    return keep_residues


def _resolve_residue_index(structure: dict, residue: dict) -> int:
    residues = structure["residues"]
    target_idx = int(residue.get("_index", -1))
    if 0 <= target_idx < len(residues):
        candidate = residues[target_idx]
        if candidate is residue or get_resid_key(candidate) == get_resid_key(residue):
            return target_idx

    residue_key = get_resid_key(residue)
    for idx, candidate in enumerate(residues):
        if candidate is residue or get_resid_key(candidate) == residue_key:
            return idx

    raise ValueError(f"Residue {get_resid_label(residue)} is not present in the supplied structure.")


def find_prev_next_peptide_residues(
    structure: dict,
    residue: dict,
    bond_policy: str = "auto",
) -> tuple[dict | None, dict | None]:
    if not is_peptide_like(residue):
        return None, None

    residues = structure["residues"]
    target_idx = _resolve_residue_index(structure, residue)
    prev_residue: dict | None = None
    next_residue: dict | None = None

    for idx in range(target_idx - 1, -1, -1):
        candidate = residues[idx]
        if not match_chain(candidate, residue):
            continue
        if not is_peptide_like(candidate):
            continue
        if peptide_link(candidate, residue, structure, bond_policy=bond_policy):
            prev_residue = candidate
        break

    for idx in range(target_idx + 1, len(residues)):
        candidate = residues[idx]
        if not match_chain(candidate, residue):
            continue
        if not is_peptide_like(candidate):
            continue
        if peptide_link(residue, candidate, structure, bond_policy=bond_policy):
            next_residue = candidate
        break

    return prev_residue, next_residue


def locate_context(
    structure: dict,
    target: str,
    keep: list[str],
    bond_policy: str = "auto",
) -> dict:
    target_residue = find_unique_residue(structure, target, label="Target residue")
    keep_residues = find_keep_residues(structure, keep, target_residue)
    prev_residue, next_residue = find_prev_next_peptide_residues(structure, target_residue, bond_policy=bond_policy)
    return {
        "target": target_residue,
        "prev_residue": prev_residue,
        "next_residue": next_residue,
        "keep_residues": keep_residues,
    }


def collect_environment_residues(
    structure: dict,
    target_residue: dict,
    *,
    cutoff: float,
    excluded_keys: set[tuple[str, int, str]] | None = None,
    include_water: bool = True,
) -> list[dict]:
    if cutoff < 0.0:
        raise ValueError(f"cutoff must be >= 0.0, got {cutoff}.")

    blocked = set(excluded_keys or ())
    blocked.add(get_resid_key(target_residue))
    environment: list[dict] = []
    for residue in structure["residues"]:
        if get_resid_key(residue) in blocked:
            continue
        if not include_water and residue["kind"] == "water":
            continue
        if get_resid_mindist(target_residue, residue) <= cutoff:
            environment.append(residue)
    return environment


def extract_cluster(
    pdb_path: str,
    target: str,
    cutoff: float = 4.0,
    keep: str = "",
    keep_altloc: str = "A",
    bond_policy: str = "auto",
) -> dict:
    structure = read_pdb(pdb_path, keep_altloc=keep_altloc)
    context = locate_context(structure, target=target, keep=keep.split(), bond_policy=bond_policy)
    target_residue = context["target"]
    keep_residues = context["keep_residues"]
    environment = collect_environment_residues(
        structure,
        target_residue,
        cutoff=cutoff,
        excluded_keys={get_resid_key(residue) for residue in keep_residues},
    )

    return {
        "target": get_resid_info(target_residue),
        "environment_residues": [get_resid_info(residue) for residue in environment],
        "keep_residues": [get_resid_info(residue) for residue in keep_residues],
        "cutoff": float(cutoff),
    }
