from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from recognize import Atom, Residue, ResidueKey, Structure, find_context, peptide_link
from addcap import (
    _add_bond, _add_methyl_hydrogens,
    _build_ace_from_prev, _build_nme_from_next,
    _clone_atom, _ensure_nme_hydrogen, _make_atom,
    _max_structure_serial, _require_backbone_atoms,
)


BOUNDARY_H_BOND_LENGTH = {
    "C": 1.090,
    "N": 1.010,
    "O": 0.960,
    "S": 1.340,
}


@dataclass
class ClusterFragment:
    atoms: list[Atom]
    bonds: list[tuple[int, int]]
    residue_groups: dict[str, list[int]]
    ace_mode: str
    nme_mode: str
    cutoff: float
    context: dict


def _safe_pair(a: int, b: int) -> tuple[int, int]:
    if a < b:
        return (a, b)
    return (b, a)


def _atom_xyz(atom: Atom) -> np.ndarray:
    return np.array((atom.x, atom.y, atom.z), dtype=float)


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1.0e-12:
        raise ValueError("Zero-length vector encountered during cluster construction.")
    return v / n


def _atom_element(atom: Atom) -> str:
    e = (atom.element or "").strip().upper()
    if e:
        return e
    for ch in atom.name:
        if ch.isalpha():
            return ch.upper()
    return "C"


def _boundary_h_length(atom: Atom) -> float:
    return BOUNDARY_H_BOND_LENGTH.get(_atom_element(atom), BOUNDARY_H_BOND_LENGTH["C"])


def _clone_residue(residue: Residue) -> Residue:
    cloned = Residue(
        key=residue.key,
        atoms={},
        file_order=residue.file_order,
        ter_after=residue.ter_after,
        parent_std=residue.parent_std,
        prev_parent_std=residue.prev_parent_std,
        next_parent_std=residue.next_parent_std,
        ccd_type=residue.ccd_type,
    )
    for atom in sorted(residue.atoms.values(), key=lambda a: a.serial):
        copied = _clone_atom(atom, residue.key)
        cloned.atoms[copied.name] = copied
    return cloned


def _residue_has_peptide_backbone(residue: Residue) -> bool:
    return residue.has("N", "CA", "C", "O")


def _residue_min_distance(left: Residue, right: Residue) -> float:
    left_xyz = np.array([_atom_xyz(atom) for atom in left.atoms.values()], dtype=float)
    right_xyz = np.array([_atom_xyz(atom) for atom in right.atoms.values()], dtype=float)
    if left_xyz.size == 0 or right_xyz.size == 0:
        return float("inf")
    diff = left_xyz[:, None, :] - right_xyz[None, :, :]
    return float(np.sqrt((diff * diff).sum(axis=2)).min())


def _build_residue_lookup(structure: Structure) -> tuple[dict[ResidueKey, Residue], dict[ResidueKey, int]]:
    by_key: dict[ResidueKey, Residue] = {}
    idx_by_key: dict[ResidueKey, int] = {}
    for idx, residue in enumerate(structure.residues):
        by_key[residue.key] = residue
        idx_by_key[residue.key] = idx
    return by_key, idx_by_key


def _build_env_exclusions(
    structure: Structure,
    idx_by_key: dict[ResidueKey, int],
    target: Residue,
    prev: Optional[Residue],
    nxt: Optional[Residue],
) -> set[ResidueKey]:
    # base exclusions: core residues
    excluded: set[ResidueKey] = {target.key}
    if prev is not None:
        excluded.add(prev.key)
    if nxt is not None:
        excluded.add(nxt.key)

    # one-hop peptide neighbors of prev/next
    if prev is not None:
        prev_ctx = find_context(structure, idx_by_key[prev.key])
        prev_prev = prev_ctx["prev_peptide"]
        if prev_prev is not None and prev_prev.key not in {target.key, prev.key, nxt.key if nxt else None}:
            if peptide_link(prev_prev, prev, structure.links, structure.conect):
                excluded.add(prev_prev.key)

    if nxt is not None:
        next_ctx = find_context(structure, idx_by_key[nxt.key])
        next_next = next_ctx["next_peptide"]
        if next_next is not None and next_next.key not in {target.key, prev.key if prev else None, nxt.key}:
            if peptide_link(nxt, next_next, structure.links, structure.conect):
                excluded.add(next_next.key)

    return excluded


def _collect_environment_residues(
    structure: Structure,
    target: Residue,
    excluded_keys: set[ResidueKey],
    cutoff: float,
) -> list[Residue]:
    env: list[Residue] = []
    for residue in structure.residues:
        if residue.key in excluded_keys:
            continue
        if residue.key.model != target.key.model:
            continue
        if not _residue_has_peptide_backbone(residue):
            continue
        if _residue_min_distance(residue, target) <= cutoff:
            env.append(residue)
    return env


def _collect_candidate_edges(structure: Structure) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()

    for root, neighbors in structure.conect.items():
        for nbr in neighbors:
            if root == nbr:
                continue
            edges.add(_safe_pair(root, nbr))

    residue_by_key, _ = _build_residue_lookup(structure)
    for rk1, a1, rk2, a2 in structure.links:
        r1 = residue_by_key.get(rk1)
        r2 = residue_by_key.get(rk2)
        if r1 is None or r2 is None:
            continue
        if a1 not in r1.atoms or a2 not in r2.atoms:
            continue
        edges.add(_safe_pair(r1[a1].serial, r2[a2].serial))

    for idx, residue in enumerate(structure.residues):
        ctx = find_context(structure, idx)
        nxt = ctx["next_peptide"]
        if nxt and residue.has("C") and nxt.has("N"):
            edges.add(_safe_pair(residue["C"].serial, nxt["N"].serial))

    return edges


def _collect_structure_atoms(structure: Structure) -> dict[int, Atom]:
    atoms: dict[int, Atom] = {}
    for residue in structure.residues:
        for atom in residue.atoms.values():
            atoms.setdefault(atom.serial, atom)
    return atoms


def _add_backbone_fallback_bonds(residues: list[Residue], bonds: set[tuple[int, int]]) -> None:
    for residue in residues:
        if residue.has("N", "CA"):
            _add_bond(bonds, residue["N"].serial, residue["CA"].serial)
        if residue.has("CA", "C"):
            _add_bond(bonds, residue["CA"].serial, residue["C"].serial)
        if residue.has("C", "O"):
            _add_bond(bonds, residue["C"].serial, residue["O"].serial)


def _make_h_name(residue: Residue, attached_atom: Atom) -> str:
    prefix = {
        "C": "HC",
        "N": "HN",
        "O": "HO",
        "S": "HS",
    }.get(_atom_element(attached_atom), "H")

    for idx in range(1, 100):
        name = f"{prefix}{idx}"
        if len(name) <= 4 and name not in residue.atoms:
            return name
    for idx in range(1, 1000):
        name = f"H{idx:03d}"
        if name not in residue.atoms:
            return name
    raise ValueError(f"Failed to allocate unique hydrogen name in residue {residue.key}.")


def _trim_env_residue_to_rgroup(
    structure: Structure,
    residue: Residue,
) -> tuple[list[np.ndarray], set[tuple[int, int]]]:
    """
    Keep side chain + CA for environment residue:
    - Remove backbone N/C/O/OXT
    - Remove N-attached hydrogens
    - Return CA saturation directions for broken CA-N / CA-C bonds
    """
    ca_atom = residue.atoms.get("CA")
    if ca_atom is None:
        return [], set()

    remove_names: set[str] = set()
    for name in ("N", "C", "O", "OXT"):
        if name in residue.atoms:
            remove_names.add(name)

    n_atom = residue.atoms.get("N")
    n_serial = n_atom.serial if n_atom is not None else None
    if n_serial is not None:
        for name, atom in residue.atoms.items():
            if _atom_element(atom) != "H":
                continue
            if n_serial in structure.conect.get(atom.serial, set()) or atom.serial in structure.conect.get(n_serial, set()):
                remove_names.add(name)
        if not any(name in remove_names for name in ("H", "H1", "H2", "H3", "HN", "HT1", "HT2", "HT3")):
            for name in ("H", "H1", "H2", "H3", "HN", "HT1", "HT2", "HT3"):
                if name in residue.atoms:
                    remove_names.add(name)

    ca_xyz = _atom_xyz(ca_atom)
    ca_h_dirs: list[np.ndarray] = []
    replaced_pairs: set[tuple[int, int]] = set()
    for backbone_name in ("N", "C"):
        if backbone_name not in remove_names:
            continue
        atom = residue.atoms.get(backbone_name)
        if atom is None:
            continue
        ca_h_dirs.append(_atom_xyz(atom) - ca_xyz)
        replaced_pairs.add(_safe_pair(ca_atom.serial, atom.serial))

    for name in remove_names:
        if name in residue.atoms:
            del residue.atoms[name]

    return ca_h_dirs, replaced_pairs


def build_cluster_fragment(
    structure: Structure,
    target_idx: int,
    cutoff: float = 4.5,
    env_mode: str = "rgroup",
):
    if cutoff < 0.0:
        raise ValueError(f"cutoff must be >= 0.0, got {cutoff}.")
    if env_mode not in {"full", "rgroup"}:
        raise ValueError(f"env_mode must be one of ['full', 'rgroup'], got {env_mode!r}.")

    ctx_target = find_context(structure, target_idx)
    target = ctx_target["target"]
    prev = ctx_target["prev_peptide"]
    nxt = ctx_target["next_peptide"]

    _require_backbone_atoms(target)
    if prev is not None:
        _require_backbone_atoms(prev)
    if nxt is not None:
        _require_backbone_atoms(nxt)

    core_residues: list[Residue] = []
    if prev is not None:
        core_residues.append(prev)
    core_residues.append(target)
    if nxt is not None:
        core_residues.append(nxt)

    _, idx_by_key = _build_residue_lookup(structure)
    excluded_env_keys = _build_env_exclusions(structure, idx_by_key, target, prev, nxt)

    core_key_set = {r.key for r in core_residues}
    env_residues = _collect_environment_residues(structure, target, excluded_env_keys, cutoff=cutoff)
    env_key_set = {r.key for r in env_residues}
    selected_key_set = core_key_set | env_key_set

    selected_residue_clones: dict[ResidueKey, Residue] = {}
    cloned_atom_by_serial: dict[int, Atom] = {}
    copied_core_serials: list[int] = []
    copied_env_serials: list[int] = []
    rgroup_ca_dirs: list[tuple[ResidueKey, np.ndarray]] = []
    rgroup_replaced_pairs: set[tuple[int, int]] = set()
    rgroup_ca_h_bonds: list[tuple[int, int]] = []

    for residue in structure.residues:
        if residue.key not in selected_key_set:
            continue
        clone = _clone_residue(residue)
        selected_residue_clones[clone.key] = clone
        in_core = clone.key in core_key_set

        if (not in_core) and env_mode == "rgroup":
            ca_dirs, replaced_pairs = _trim_env_residue_to_rgroup(structure, clone)
            for v in ca_dirs:
                rgroup_ca_dirs.append((clone.key, v))
            rgroup_replaced_pairs.update(replaced_pairs)

        for atom in clone.atoms.values():
            if atom.serial in cloned_atom_by_serial:
                raise ValueError(
                    f"Duplicated atom serial {atom.serial} in selected cluster; cannot preserve original serials."
                )
            cloned_atom_by_serial[atom.serial] = atom
            if in_core:
                copied_core_serials.append(atom.serial)
            else:
                copied_env_serials.append(atom.serial)

    selected_original_serials = set(cloned_atom_by_serial.keys())

    left_anchor = prev if prev is not None else target
    right_anchor = nxt if nxt is not None else target

    left_idx = idx_by_key[left_anchor.key]
    right_idx = idx_by_key[right_anchor.key]
    left_outer: Optional[Residue] = find_context(structure, left_idx)["prev_peptide"]
    right_outer: Optional[Residue] = find_context(structure, right_idx)["next_peptide"]

    next_serial = _max_structure_serial(structure) + 1
    ace_key = ResidueKey(
        model=left_anchor.key.model,
        chain=left_anchor.key.chain,
        resseq=left_anchor.key.resseq,
        icode=left_anchor.key.icode,
        resname="ACE",
    )
    nme_key = ResidueKey(
        model=right_anchor.key.model,
        chain=right_anchor.key.chain,
        resseq=right_anchor.key.resseq,
        icode=right_anchor.key.icode,
        resname="NME",
    )

    ace, ace_mode, next_serial = _build_ace_from_prev(left_outer, left_anchor, next_serial, ace_key)
    nme, nme_mode, next_serial = _build_nme_from_next(right_anchor, right_outer, next_serial, nme_key)
    next_serial = _add_methyl_hydrogens(ace, "CH3", "C", ("H11", "H12", "H13"), next_serial, ace_key)
    next_serial = _add_methyl_hydrogens(nme, "CH3", "N", ("H21", "H22", "H23"), next_serial, nme_key)
    next_serial = _ensure_nme_hydrogen(nme, right_anchor, next_serial, nme_key)

    # In rgroup mode, saturate removed CA-N / CA-C bonds with CA-attached hydrogens.
    if env_mode == "rgroup":
        for residue_key, ca_dir in rgroup_ca_dirs:
            residue_clone = selected_residue_clones.get(residue_key)
            if residue_clone is None:
                continue
            ca_atom = residue_clone.atoms.get("CA")
            if ca_atom is None:
                continue
            try:
                h_dir = _normalize(ca_dir)
            except ValueError:
                continue
            h_name = _make_h_name(residue_clone, ca_atom)
            h_xyz = _atom_xyz(ca_atom) + h_dir * _boundary_h_length(ca_atom)
            h_atom = _make_atom(next_serial, h_name, h_xyz, "H", residue_key)
            residue_clone.atoms[h_name] = h_atom
            cloned_atom_by_serial[h_atom.serial] = h_atom
            copied_env_serials.append(h_atom.serial)
            rgroup_ca_h_bonds.append((ca_atom.serial, h_atom.serial))
            next_serial += 1

    bonds: set[tuple[int, int]] = set()
    candidate_edges = _collect_candidate_edges(structure)
    for a, b in candidate_edges:
        if a in selected_original_serials and b in selected_original_serials:
            _add_bond(bonds, a, b)

    _add_backbone_fallback_bonds(list(selected_residue_clones.values()), bonds)

    left_anchor_clone = selected_residue_clones[left_anchor.key]
    right_anchor_clone = selected_residue_clones[right_anchor.key]

    _add_bond(bonds, ace["CH3"].serial, ace["C"].serial)
    _add_bond(bonds, ace["C"].serial, ace["O"].serial)
    _add_bond(bonds, ace["C"].serial, left_anchor_clone["N"].serial)
    for h_name in ("H11", "H12", "H13"):
        _add_bond(bonds, ace["CH3"].serial, ace[h_name].serial)

    _add_bond(bonds, right_anchor_clone["C"].serial, nme["N"].serial)
    _add_bond(bonds, nme["N"].serial, nme["CH3"].serial)
    _add_bond(bonds, nme["N"].serial, nme["HN"].serial)
    for h_name in ("H21", "H22", "H23"):
        _add_bond(bonds, nme["CH3"].serial, nme[h_name].serial)

    replaced_pairs: set[tuple[int, int]] = set()
    if left_outer is not None and left_anchor.has("N") and left_outer.has("C"):
        replaced_pairs.add(_safe_pair(left_anchor["N"].serial, left_outer["C"].serial))
    if right_outer is not None and right_anchor.has("C") and right_outer.has("N"):
        replaced_pairs.add(_safe_pair(right_anchor["C"].serial, right_outer["N"].serial))

    atom_by_serial = _collect_structure_atoms(structure)
    boundary_h_serials: list[int] = []

    for a, b in sorted(candidate_edges):
        if _safe_pair(a, b) in replaced_pairs:
            continue
        if _safe_pair(a, b) in rgroup_replaced_pairs:
            continue
        a_in = a in selected_original_serials
        b_in = b in selected_original_serials
        if a_in == b_in:
            continue

        in_serial, out_serial = (a, b) if a_in else (b, a)
        in_atom = cloned_atom_by_serial.get(in_serial)
        out_atom = atom_by_serial.get(out_serial)
        if in_atom is None or out_atom is None:
            continue

        # Place H toward the detached partner direction to preserve broken-bond orientation.
        direction = _atom_xyz(out_atom) - _atom_xyz(in_atom)
        try:
            h_dir = _normalize(direction)
        except ValueError:
            continue

        residue_key = in_atom.residue_key
        if residue_key is None:
            continue
        residue_clone = selected_residue_clones.get(residue_key)
        if residue_clone is None:
            continue

        h_name = _make_h_name(residue_clone, in_atom)
        h_xyz = _atom_xyz(in_atom) + h_dir * _boundary_h_length(in_atom)
        h_atom = _make_atom(next_serial, h_name, h_xyz, "H", residue_key)
        residue_clone.atoms[h_name] = h_atom
        cloned_atom_by_serial[h_atom.serial] = h_atom
        boundary_h_serials.append(h_atom.serial)
        _add_bond(bonds, in_serial, h_atom.serial)
        next_serial += 1

    # Ensure explicit bonds for CA hydrogens added by rgroup trimming.
    if env_mode == "rgroup":
        for a, b in rgroup_ca_h_bonds:
            _add_bond(bonds, a, b)

    all_atoms = sorted(
        list(cloned_atom_by_serial.values()) + list(ace.values()) + list(nme.values()),
        key=lambda atom: atom.serial,
    )

    target_clone = selected_residue_clones[target.key]
    prev_clone = selected_residue_clones[prev.key] if prev is not None else None
    next_clone = selected_residue_clones[nxt.key] if nxt is not None else None

    residue_groups = {
        "ACE": sorted(atom.serial for atom in ace.values()),
        "CORE": sorted(set(copied_core_serials)),
        "ENV": sorted(set(copied_env_serials)),
        "NME": sorted(atom.serial for atom in nme.values()),
        "BOUNDARY_H": sorted(boundary_h_serials),
    }

    context = {
        "target_idx": target_idx,
        "target_key": target.key,
        "env_mode": env_mode,
        "env_keys": [res.key for res in env_residues],
        "n_open": ctx_target["n_open"],
        "c_open": ctx_target["c_open"],
    }

    fragment = ClusterFragment(
        atoms=all_atoms,
        bonds=sorted(bonds),
        residue_groups=residue_groups,
        ace_mode=ace_mode,
        nme_mode=nme_mode,
        cutoff=float(cutoff),
        context=context,
    )

    return {
        "fragment": fragment,
        "core": {
            "prev": prev_clone,
            "target": target_clone,
            "next": next_clone,
            "residues": [selected_residue_clones[r.key] for r in core_residues],
        },
        "environment": [selected_residue_clones[r.key] for r in env_residues],
        "ace": ace,
        "nme": nme,
        "ace_mode": ace_mode,
        "nme_mode": nme_mode,
    }
