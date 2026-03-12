from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from .recognize import Atom, Residue, ResidueKey, Structure, find_context
except ImportError:
    from recognize import Atom, Residue, ResidueKey, Structure, find_context


# Bond lengths (Angstrom)
BOND_C_N_AMIDE = 1.335
BOND_C_O = 1.229
BOND_C_CH3_ACE = 1.522
BOND_N_CH3_NME = 1.458
BOND_C_H = 1.090
BOND_N_H = 1.010


Vec3 = tuple[float, float, float]
Array3 = np.ndarray


@dataclass
class CappedFragment:
    atoms: list[Atom]
    bonds: list[tuple[int, int]]
    residue_groups: dict[str, list[int]]
    ace_mode: str
    nme_mode: str
    context: dict


def _as_np(xyz: Vec3 | Array3) -> Array3:
    arr = np.asarray(xyz, dtype=float)
    if arr.shape != (3,):
        if arr.size != 3:
            raise ValueError(f"Expected a 3D vector, got shape {arr.shape}.")
        arr = arr.reshape(3)
    return arr


def _to_vec(a: Array3) -> Vec3:
    v = _as_np(a)
    return (float(v[0]), float(v[1]), float(v[2]))


def _normalize(v: Vec3 | Array3) -> Array3:
    a = _as_np(v)
    n = float(np.linalg.norm(a))
    if n < 1.0e-12:
        raise ValueError("Zero-length vector encountered during cap construction.")
    return a / n


def _project_perp(v: Vec3 | Array3, axis: Vec3 | Array3) -> Array3:
    vv = _as_np(v)
    aa = _as_np(axis)
    return vv - aa * float(np.dot(vv, aa))


def _arbitrary_perp(axis: Vec3 | Array3) -> Array3:
    aa = _as_np(axis)
    candidates: tuple[Array3, ...] = (
        np.array((1.0, 0.0, 0.0), dtype=float),
        np.array((0.0, 1.0, 0.0), dtype=float),
        np.array((0.0, 0.0, 1.0), dtype=float),
    )
    for c in candidates:
        p = _project_perp(c, aa)
        if float(np.linalg.norm(p)) > 1.0e-8:
            return _normalize(p)
    raise ValueError("Failed to find perpendicular direction.")


def _trigonal_pair(primary_dir: Vec3 | Array3, hint_vec: Vec3 | Array3) -> tuple[Array3, Array3]:
    u = _normalize(primary_dir)
    v = _project_perp(hint_vec, u)
    if float(np.linalg.norm(v)) < 1.0e-8:
        v = _arbitrary_perp(u)
    else:
        v = _normalize(v)

    cos120 = -0.5
    sin120 = float(np.sqrt(3.0) * 0.5)
    d1 = _normalize(u * cos120 + v * sin120)
    d2 = _normalize(u * cos120 - v * sin120)
    return d1, d2


def _tetrahedral_h_dirs(anchor_dir: Vec3 | Array3, hint_vec: Optional[Vec3 | Array3] = None) -> list[Array3]:
    u = _normalize(anchor_dir)
    if hint_vec is None:
        e1 = _arbitrary_perp(u)
    else:
        h = _project_perp(hint_vec, u)
        if float(np.linalg.norm(h)) < 1.0e-8:
            e1 = _arbitrary_perp(u)
        else:
            e1 = _normalize(h)
    e2 = _normalize(np.cross(u, e1))

    coeff_u = -1.0 / 3.0
    coeff_plane = float(np.sqrt(8.0) / 3.0)
    phis = (0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0)
    dirs: list[Array3] = []
    for phi in phis:
        plane = e1 * float(np.cos(phi)) + e2 * float(np.sin(phi))
        v = u * coeff_u + plane * coeff_plane
        dirs.append(_normalize(v))
    return dirs


def _atom_xyz(atom: Atom) -> Array3:
    return np.array((atom.x, atom.y, atom.z), dtype=float)


def _make_atom(
    serial: int,
    name: str,
    xyz: Vec3 | Array3,
    element: str,
    residue_key: ResidueKey,
    record: str = "ATOM",
    occ: float = 1.0,
    bfac: float = 0.0,
    altloc: str = "",
    charge: str = "",
) -> Atom:
    p = _as_np(xyz)
    return Atom(
        serial=serial,
        record=record,
        name=name,
        altloc=altloc,
        x=float(p[0]),
        y=float(p[1]),
        z=float(p[2]),
        occ=occ,
        bfac=bfac,
        element=element,
        charge=charge,
        residue_key=residue_key,
    )


def _clone_atom(
    atom: Atom,
    residue_key: ResidueKey,
    serial: Optional[int] = None,
    name: Optional[str] = None,
    xyz: Optional[Vec3 | Array3] = None,
    element: Optional[str] = None,
    record: Optional[str] = None,
) -> Atom:
    p = _atom_xyz(atom) if xyz is None else xyz
    q = _as_np(p)
    return Atom(
        serial=atom.serial if serial is None else serial,
        record=atom.record if record is None else record,
        name=atom.name if name is None else name,
        altloc=atom.altloc,
        x=float(q[0]),
        y=float(q[1]),
        z=float(q[2]),
        occ=atom.occ,
        bfac=atom.bfac,
        element=atom.element if element is None else element,
        charge=atom.charge,
        residue_key=residue_key,
    )


def _max_structure_serial(structure: Structure) -> int:
    max_serial = 0
    for residue in structure.residues:
        for atom in residue.atoms.values():
            if atom.serial > max_serial:
                max_serial = atom.serial
    return max_serial


def _require_backbone_atoms(target: Residue) -> None:
    required = ("N", "CA", "C", "O")
    missing = [name for name in required if name not in target.atoms]
    if missing:
        raise ValueError(
            f"Target residue {target.key.resname} {target.key.chain}:{target.key.resseq}{target.key.icode or ''} "
            f"missing backbone atoms: {', '.join(missing)}"
        )


def _build_ace_ideal(target: Residue, next_serial: int, ace_key: ResidueKey) -> tuple[dict[str, Atom], int]:
    n = target["N"]
    ca = target["CA"]
    c_target = target["C"]

    n_xyz = _atom_xyz(n)
    ca_xyz = _atom_xyz(ca)
    c_target_xyz = _atom_xyz(c_target)

    c_ace = n_xyz + _normalize(n_xyz - ca_xyz) * BOND_C_N_AMIDE
    primary = n_xyz - c_ace
    hint = c_target_xyz - c_ace
    o_dir, ch3_dir = _trigonal_pair(primary, hint)

    o_xyz = c_ace + o_dir * BOND_C_O
    ch3_xyz = c_ace + ch3_dir * BOND_C_CH3_ACE

    ace = {
        "CH3": _make_atom(next_serial, "CH3", ch3_xyz, "C", ace_key),
        "C": _make_atom(next_serial + 1, "C", c_ace, "C", ace_key),
        "O": _make_atom(next_serial + 2, "O", o_xyz, "O", ace_key),
    }
    return ace, next_serial + 3


def _build_nme_ideal(target: Residue, next_serial: int, nme_key: ResidueKey) -> tuple[dict[str, Atom], int]:
    c_target = target["C"]
    ca = target["CA"]
    o_target = target["O"]

    c_target_xyz = _atom_xyz(c_target)
    ca_xyz = _atom_xyz(ca)
    o_target_xyz = _atom_xyz(o_target)

    n_xyz = c_target_xyz + _normalize(c_target_xyz - ca_xyz) * BOND_C_N_AMIDE
    primary = c_target_xyz - n_xyz
    hint = o_target_xyz - n_xyz
    ch3_dir, hn_dir = _trigonal_pair(primary, hint)

    ch3_xyz = n_xyz + ch3_dir * BOND_N_CH3_NME
    hn_xyz = n_xyz + hn_dir * BOND_N_H

    nme = {
        "N": _make_atom(next_serial, "N", n_xyz, "N", nme_key),
        "CH3": _make_atom(next_serial + 1, "CH3", ch3_xyz, "C", nme_key),
        "HN": _make_atom(next_serial + 2, "HN", hn_xyz, "H", nme_key),
    }
    return nme, next_serial + 3


def _build_ace_from_prev(
    prev: Optional[Residue],
    target: Residue,
    next_serial: int,
    ace_key: ResidueKey,
) -> tuple[dict[str, Atom], str, int]:
    if prev and prev.has("CA", "C", "O"):
        ace = {
            "CH3": _clone_atom(prev["CA"], ace_key, serial=next_serial, name="CH3", element="C"),
            "C": _clone_atom(prev["C"], ace_key, serial=next_serial + 1, name="C", element="C"),
            "O": _clone_atom(prev["O"], ace_key, serial=next_serial + 2, name="O", element="O"),
        }
        return ace, "neighbor", next_serial + 3

    ace, next_serial = _build_ace_ideal(target, next_serial, ace_key)
    return ace, "ideal", next_serial


def _build_nme_from_next(
    target: Residue,
    next_res: Optional[Residue],
    next_serial: int,
    nme_key: ResidueKey,
) -> tuple[dict[str, Atom], str, int]:
    if next_res and next_res.has("N", "CA"):
        nme = {
            "N": _clone_atom(next_res["N"], nme_key, serial=next_serial, name="N", element="N"),
            "CH3": _clone_atom(next_res["CA"], nme_key, serial=next_serial + 1, name="CH3", element="C"),
        }
        return nme, "neighbor", next_serial + 2

    nme, next_serial = _build_nme_ideal(target, next_serial, nme_key)
    return nme, "ideal", next_serial


def _add_methyl_hydrogens(
    residue_atoms: dict[str, Atom],
    carbon_name: str,
    anchor_name: str,
    h_names: tuple[str, str, str],
    next_serial: int,
    residue_key: ResidueKey,
) -> int:
    carbon = residue_atoms[carbon_name]
    anchor = residue_atoms[anchor_name]
    center = _atom_xyz(carbon)
    anchor_dir = _atom_xyz(anchor) - center
    h_dirs = _tetrahedral_h_dirs(anchor_dir)
    for idx, h_name in enumerate(h_names):
        h_xyz = center + h_dirs[idx] * BOND_C_H
        residue_atoms[h_name] = _make_atom(next_serial, h_name, h_xyz, "H", residue_key)
        next_serial += 1
    return next_serial


def _ensure_nme_hydrogen(
    nme_atoms: dict[str, Atom],
    target: Residue,
    next_serial: int,
    nme_key: ResidueKey,
) -> int:
    if "HN" in nme_atoms:
        return next_serial

    n_atom = nme_atoms["N"]
    ch3 = nme_atoms["CH3"]
    c_target = target["C"]

    u1 = _normalize(_atom_xyz(c_target) - _atom_xyz(n_atom))
    u2 = _normalize(_atom_xyz(ch3) - _atom_xyz(n_atom))
    hn_dir_raw = -(u1 + u2)
    if float(np.linalg.norm(hn_dir_raw)) < 1.0e-8:
        hn_dir = _arbitrary_perp(u1)
    else:
        hn_dir = _normalize(hn_dir_raw)

    hn_xyz = _atom_xyz(n_atom) + hn_dir * BOND_N_H
    nme_atoms["HN"] = _make_atom(next_serial, "HN", hn_xyz, "H", nme_key)
    return next_serial + 1


def _clone_target_residue(target: Residue) -> Residue:
    cloned = Residue(
        key=target.key,
        atoms={},
        file_order=target.file_order,
        ter_after=target.ter_after,
        parent_std=target.parent_std,
        prev_parent_std=target.prev_parent_std,
        next_parent_std=target.next_parent_std,
        ccd_type=target.ccd_type,
    )
    for atom in sorted(target.atoms.values(), key=lambda a: a.serial):
        cloned_atom = _clone_atom(atom, target.key)
        cloned.atoms[cloned_atom.name] = cloned_atom
    return cloned


def _add_bond(bonds: set[tuple[int, int]], a: int, b: int) -> None:
    if a == b:
        return
    bonds.add((a, b) if a < b else (b, a))


def _populate_target_bonds(
    structure: Structure,
    target_clone: Residue,
    bonds: set[tuple[int, int]],
) -> None:
    target_serials = {atom.serial for atom in target_clone.atoms.values()}
    for serial in target_serials:
        for nbr in structure.conect.get(serial, set()):
            if nbr in target_serials:
                _add_bond(bonds, serial, nbr)

    # Always keep minimal backbone connectivity for downstream term enumeration.
    backbone_pairs = (("N", "CA"), ("CA", "C"), ("C", "O"))
    for a_name, b_name in backbone_pairs:
        if target_clone.has(a_name, b_name):
            _add_bond(bonds, target_clone[a_name].serial, target_clone[b_name].serial)


def _context_payload(ctx: dict, target_idx: int) -> dict:
    prev_res = ctx["prev_peptide"]
    next_res = ctx["next_peptide"]
    target = ctx["target"]
    return {
        "target_idx": target_idx,
        "target_key": target.key,
        "prev_key": prev_res.key if prev_res else None,
        "next_key": next_res.key if next_res else None,
        "n_open": ctx["n_open"],
        "c_open": ctx["c_open"],
        "parent_std": target.parent_std,
    }


def build_capped_fragment(structure: Structure, target_idx: int):
    ctx = find_context(structure, target_idx)
    target = ctx["target"]
    _require_backbone_atoms(target)

    next_serial = _max_structure_serial(structure) + 1
    ace_key = ResidueKey(
        model=target.key.model,
        chain=target.key.chain,
        resseq=target.key.resseq,
        icode=target.key.icode,
        resname="ACE",
    )
    nme_key = ResidueKey(
        model=target.key.model,
        chain=target.key.chain,
        resseq=target.key.resseq,
        icode=target.key.icode,
        resname="NME",
    )

    ace, ace_mode, next_serial = _build_ace_from_prev(ctx["prev_peptide"], target, next_serial, ace_key)
    nme, nme_mode, next_serial = _build_nme_from_next(target, ctx["next_peptide"], next_serial, nme_key)

    next_serial = _add_methyl_hydrogens(ace, "CH3", "C", ("H11", "H12", "H13"), next_serial, ace_key)
    next_serial = _add_methyl_hydrogens(nme, "CH3", "N", ("H21", "H22", "H23"), next_serial, nme_key)
    _ = _ensure_nme_hydrogen(nme, target, next_serial, nme_key)

    target_clone = _clone_target_residue(target)

    bonds: set[tuple[int, int]] = set()
    _populate_target_bonds(structure, target_clone, bonds)

    _add_bond(bonds, ace["CH3"].serial, ace["C"].serial)
    _add_bond(bonds, ace["C"].serial, ace["O"].serial)
    _add_bond(bonds, ace["C"].serial, target_clone["N"].serial)
    _add_bond(bonds, target_clone["C"].serial, nme["N"].serial)
    _add_bond(bonds, nme["N"].serial, nme["CH3"].serial)

    for h_name in ("H11", "H12", "H13"):
        _add_bond(bonds, ace["CH3"].serial, ace[h_name].serial)
    _add_bond(bonds, nme["N"].serial, nme["HN"].serial)
    for h_name in ("H21", "H22", "H23"):
        _add_bond(bonds, nme["CH3"].serial, nme[h_name].serial)

    all_atoms = []
    all_atoms.extend(sorted(ace.values(), key=lambda a: a.serial))
    all_atoms.extend(sorted(target_clone.atoms.values(), key=lambda a: a.serial))
    all_atoms.extend(sorted(nme.values(), key=lambda a: a.serial))

    residue_groups = {
        "ACE": sorted(atom.serial for atom in ace.values()),
        "TARGET": sorted(atom.serial for atom in target_clone.atoms.values()),
        "NME": sorted(atom.serial for atom in nme.values()),
    }
    fragment = CappedFragment(
        atoms=all_atoms,
        bonds=sorted(bonds),
        residue_groups=residue_groups,
        ace_mode=ace_mode,
        nme_mode=nme_mode,
        context=_context_payload(ctx, target_idx),
    )

    return {
        "fragment": fragment,
        "ace": ace,
        "target": target_clone,
        "nme": nme,
        "ace_mode": ace_mode,
        "nme_mode": nme_mode,
    }
