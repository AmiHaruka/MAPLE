"""Usage: build ACE/NME caps and glycine bridge residues for parmfit models."""

from __future__ import annotations

import numpy as np

from .structure import (
    BOND_C_CH3_ACE,
    BOND_C_H,
    BOND_C_N_AMIDE,
    BOND_C_O,
    BOND_N_CH3_NME,
    BOND_N_H,
    _norm,
    arbitrary_perp,
    copy_atom,
    get_atom_xyz,
    make_atom,
    make_residue,
    refresh_resid,
    search_atom,
    tetrahedral_h_dirs,
    trigonal_pair,
)

_ACE_METHYL_TEMPLATE_NAMES = ("HA", "CB", "N", "HA2", "HA3")
_NME_METHYL_TEMPLATE_NAMES = ("HA", "CB", "C", "HA2", "HA3")
_NME_HYDROGEN_NAMES = ("H", "HN", "H1", "HN1", "H2", "HN2", "H3", "HN3")
_GLY_ALPHA_HYDROGEN_NAMES = ("HA2", "HA3", "HA", "CB")


def _copy_named_atom(
    source_residue: dict,
    source_name: str,
    serial: int,
    target_name: str,
    element: str,
) -> dict | None:
    atom = search_atom(source_residue, source_name)
    if atom is None:
        return None
    return copy_atom(atom, serial=serial, name=target_name, element=element)


def _append_hydrogen(residue: dict, name: str, anchor_xyz: np.ndarray, direction: np.ndarray, bond_length: float, serial: int) -> int:
    residue["atoms"].append(make_atom(serial, name, "H", anchor_xyz + _norm(direction) * bond_length))
    return serial + 1


def _build_ace_cap_ideal(target: dict, next_serial: int) -> tuple[dict, int]:
    n_atom = search_atom(target, "N")
    ca_atom = search_atom(target, "CA")
    c_atom = search_atom(target, "C")
    if n_atom is None or ca_atom is None or c_atom is None:
        raise ValueError("ACE ideal cap requires target backbone atoms N, CA, and C.")

    n_xyz = get_atom_xyz(n_atom)
    ca_xyz = get_atom_xyz(ca_atom)
    c_xyz = get_atom_xyz(c_atom)
    c_ace_xyz = n_xyz + _norm(n_xyz - ca_xyz) * BOND_C_N_AMIDE
    o_dir, ch3_dir = trigonal_pair(n_xyz - c_ace_xyz, c_xyz - c_ace_xyz)

    atoms = [
        make_atom(next_serial, "CMA", "C", c_ace_xyz + ch3_dir * BOND_C_CH3_ACE),
        make_atom(next_serial + 1, "CAC", "C", c_ace_xyz),
        make_atom(next_serial + 2, "OAC", "O", c_ace_xyz + o_dir * BOND_C_O),
    ]
    residue = make_residue(target["chain"], target["resseq"] - 1, target["icode"], "ACE", atoms)
    return residue, next_serial + 3


def _build_nme_cap_ideal(target: dict, next_serial: int) -> tuple[dict, int]:
    c_atom = search_atom(target, "C")
    ca_atom = search_atom(target, "CA")
    o_atom = search_atom(target, "O")
    if c_atom is None or ca_atom is None or o_atom is None:
        raise ValueError("NME ideal cap requires target backbone atoms C, CA, and O.")

    c_xyz = get_atom_xyz(c_atom)
    ca_xyz = get_atom_xyz(ca_atom)
    o_xyz = get_atom_xyz(o_atom)
    n_nme_xyz = c_xyz + _norm(c_xyz - ca_xyz) * BOND_C_N_AMIDE
    ch3_dir, hn_dir = trigonal_pair(c_xyz - n_nme_xyz, o_xyz - n_nme_xyz)

    atoms = [
        make_atom(next_serial, "NNM", "N", n_nme_xyz),
        make_atom(next_serial + 1, "CNM", "C", n_nme_xyz + ch3_dir * BOND_N_CH3_NME),
        make_atom(next_serial + 2, "HNM", "H", n_nme_xyz + hn_dir * BOND_N_H),
    ]
    residue = make_residue(target["chain"], target["resseq"] + 1, target["icode"], "NME", atoms)
    return residue, next_serial + 3


def build_ace_cap(target: dict, next_serial: int, prev_residue: dict | None = None) -> tuple[dict, int]:
    if prev_residue is not None:
        cma_atom = _copy_named_atom(prev_residue, "CA", next_serial, "CMA", "C")
        cac_atom = _copy_named_atom(prev_residue, "C", next_serial + 1, "CAC", "C")
        oac_atom = _copy_named_atom(prev_residue, "O", next_serial + 2, "OAC", "O")
        if cma_atom is not None and cac_atom is not None and oac_atom is not None:
            residue = make_residue(target["chain"], target["resseq"] - 1, target["icode"], "ACE", [cma_atom, cac_atom, oac_atom])
            next_serial += 3
            carbon_xyz = get_atom_xyz(cma_atom)
            anchor_xyz = get_atom_xyz(cac_atom)
            directions: list[np.ndarray] = []
            for atom_name in _ACE_METHYL_TEMPLATE_NAMES:
                atom = search_atom(prev_residue, atom_name)
                if atom is None:
                    continue
                direction = get_atom_xyz(atom) - carbon_xyz
                if float(np.linalg.norm(direction)) < 1.0e-8:
                    continue
                normalized = _norm(direction)
                if any(abs(float(np.dot(normalized, existing))) >= 0.95 for existing in directions):
                    continue
                directions.append(normalized)
                if len(directions) == 3:
                    break

            axis = anchor_xyz - carbon_xyz
            hint = directions[0] if directions else None
            for fallback in tetrahedral_h_dirs(axis, hint_vec=hint):
                if any(abs(float(np.dot(fallback, existing))) >= 0.95 for existing in directions):
                    continue
                directions.append(fallback)
                if len(directions) == 3:
                    break
            if len(directions) < 3:
                basis = arbitrary_perp(_norm(axis))
                if not any(abs(float(np.dot(basis, existing))) >= 0.95 for existing in directions):
                    directions.append(basis)
            if len(directions) < 3:
                directions.append(-directions[0] if directions else arbitrary_perp(_norm(axis)))

            for index, direction in enumerate(directions[:3], start=1):
                next_serial = _append_hydrogen(residue, f"H{index}A", carbon_xyz, direction, BOND_C_H, next_serial)
            refresh_resid(residue)
            return residue, next_serial

    residue, next_serial = _build_ace_cap_ideal(target, next_serial)
    carbon_atom = search_atom(residue, "CMA")
    anchor_atom = search_atom(residue, "CAC")
    if carbon_atom is not None and anchor_atom is not None:
        carbon_xyz = get_atom_xyz(carbon_atom)
        anchor_xyz = get_atom_xyz(anchor_atom)
        directions: list[np.ndarray] = []
        axis = anchor_xyz - carbon_xyz
        for fallback in tetrahedral_h_dirs(axis):
            if any(abs(float(np.dot(fallback, existing))) >= 0.95 for existing in directions):
                continue
            directions.append(fallback)
            if len(directions) == 3:
                break
        if len(directions) < 3:
            basis = arbitrary_perp(_norm(axis))
            if not any(abs(float(np.dot(basis, existing))) >= 0.95 for existing in directions):
                directions.append(basis)
        if len(directions) < 3:
            directions.append(-directions[0] if directions else arbitrary_perp(_norm(axis)))
        for index, direction in enumerate(directions[:3], start=1):
            next_serial = _append_hydrogen(residue, f"H{index}A", carbon_xyz, direction, BOND_C_H, next_serial)
        refresh_resid(residue)
    return residue, next_serial


def build_nme_cap(target: dict, next_serial: int, next_residue: dict | None = None) -> tuple[dict, int]:
    if next_residue is not None:
        nnm_atom = _copy_named_atom(next_residue, "N", next_serial, "NNM", "N")
        cnm_atom = _copy_named_atom(next_residue, "CA", next_serial + 1, "CNM", "C")
        if nnm_atom is not None and cnm_atom is not None:
            residue = make_residue(target["chain"], target["resseq"] + 1, target["icode"], "NME", [nnm_atom, cnm_atom])
            next_serial += 2
            nnm_xyz = get_atom_xyz(nnm_atom)
            for atom_name in _NME_HYDROGEN_NAMES:
                atom = search_atom(next_residue, atom_name)
                if atom is None:
                    continue
                residue["atoms"].append(copy_atom(atom, serial=next_serial, name="HNM", element="H"))
                next_serial += 1
                break
            if search_atom(residue, "HNM") is None and next_residue["resname"].upper() == "PRO":
                atom = search_atom(next_residue, "CD")
                if atom is not None:
                    direction = get_atom_xyz(atom) - nnm_xyz
                    if float(np.linalg.norm(direction)) >= 1.0e-8:
                        next_serial = _append_hydrogen(residue, "HNM", nnm_xyz, direction, BOND_N_H, next_serial)

            if search_atom(residue, "HNM") is None:
                n_atom = search_atom(residue, "NNM")
                ch3_atom = search_atom(residue, "CNM")
                c_atom = search_atom(target, "C")
                if n_atom is not None and ch3_atom is not None and c_atom is not None:
                    direction = -(
                        _norm(get_atom_xyz(c_atom) - get_atom_xyz(n_atom))
                        + _norm(get_atom_xyz(ch3_atom) - get_atom_xyz(n_atom))
                    )
                    if float(np.linalg.norm(direction)) < 1.0e-8:
                        direction = arbitrary_perp(_norm(get_atom_xyz(ch3_atom) - get_atom_xyz(n_atom)))
                    next_serial = _append_hydrogen(residue, "HNM", get_atom_xyz(n_atom), direction, BOND_N_H, next_serial)

            cnm_xyz = get_atom_xyz(cnm_atom)
            directions: list[np.ndarray] = []
            for atom_name in _NME_METHYL_TEMPLATE_NAMES:
                atom = search_atom(next_residue, atom_name)
                if atom is None:
                    continue
                direction = get_atom_xyz(atom) - cnm_xyz
                if float(np.linalg.norm(direction)) < 1.0e-8:
                    continue
                normalized = _norm(direction)
                if any(abs(float(np.dot(normalized, existing))) >= 0.95 for existing in directions):
                    continue
                directions.append(normalized)
                if len(directions) == 3:
                    break

            axis = nnm_xyz - cnm_xyz
            hint = directions[0] if directions else None
            for fallback in tetrahedral_h_dirs(axis, hint_vec=hint):
                if any(abs(float(np.dot(fallback, existing))) >= 0.95 for existing in directions):
                    continue
                directions.append(fallback)
                if len(directions) == 3:
                    break
            if len(directions) < 3:
                basis = arbitrary_perp(_norm(axis))
                if not any(abs(float(np.dot(basis, existing))) >= 0.95 for existing in directions):
                    directions.append(basis)
            if len(directions) < 3:
                directions.append(-directions[0] if directions else arbitrary_perp(_norm(axis)))

            for index, direction in enumerate(directions[:3], start=1):
                next_serial = _append_hydrogen(residue, f"H{index}M", cnm_xyz, direction, BOND_C_H, next_serial)
            refresh_resid(residue)
            return residue, next_serial

    residue, next_serial = _build_nme_cap_ideal(target, next_serial)
    carbon_atom = search_atom(residue, "CNM")
    anchor_atom = search_atom(residue, "NNM")
    if carbon_atom is not None and anchor_atom is not None:
        carbon_xyz = get_atom_xyz(carbon_atom)
        anchor_xyz = get_atom_xyz(anchor_atom)
        directions: list[np.ndarray] = []
        axis = anchor_xyz - carbon_xyz
        for fallback in tetrahedral_h_dirs(axis):
            if any(abs(float(np.dot(fallback, existing))) >= 0.95 for existing in directions):
                continue
            directions.append(fallback)
            if len(directions) == 3:
                break
        if len(directions) < 3:
            basis = arbitrary_perp(_norm(axis))
            if not any(abs(float(np.dot(basis, existing))) >= 0.95 for existing in directions):
                directions.append(basis)
        if len(directions) < 3:
            directions.append(-directions[0] if directions else arbitrary_perp(_norm(axis)))
        for index, direction in enumerate(directions[:3], start=1):
            next_serial = _append_hydrogen(residue, f"H{index}M", carbon_xyz, direction, BOND_C_H, next_serial)
        refresh_resid(residue)
    return residue, next_serial


def build_gly_bridge(residue: dict, next_serial: int, prev_residue: dict | None = None) -> tuple[dict, int]:
    n_atom = search_atom(residue, "N")
    ca_atom = search_atom(residue, "CA")
    c_atom = search_atom(residue, "C")
    o_atom = search_atom(residue, "O")
    if n_atom is None or ca_atom is None or c_atom is None or o_atom is None:
        raise ValueError("GLY bridge requires residue backbone atoms N, CA, C, and O.")

    atoms = [
        copy_atom(n_atom, serial=next_serial, name="N", element="N"),
        copy_atom(ca_atom, serial=next_serial + 1, name="CA", element="C"),
        copy_atom(c_atom, serial=next_serial + 2, name="C", element="C"),
        copy_atom(o_atom, serial=next_serial + 3, name="O", element="O"),
    ]
    next_serial += 4
    bridge = make_residue(residue["chain"], residue["resseq"], residue["icode"], "GLY", atoms, kind="protein")

    if residue["resname"].upper() != "PRO":
        n_xyz = get_atom_xyz(search_atom(bridge, "N"))
        for atom_name in _NME_HYDROGEN_NAMES:
            atom = search_atom(residue, atom_name)
            if atom is None:
                continue
            bridge["atoms"].append(copy_atom(atom, serial=next_serial, name="H", element="H"))
            next_serial += 1
            break
        if search_atom(bridge, "H") is None:
            ca_xyz = get_atom_xyz(search_atom(bridge, "CA"))
            ca_dir = _norm(ca_xyz - n_xyz)
            prev_c_atom = search_atom(prev_residue, "C") if prev_residue is not None else None
            if prev_c_atom is not None:
                direction = -(ca_dir + _norm(get_atom_xyz(prev_c_atom) - n_xyz))
            else:
                direction = -ca_dir
            if float(np.linalg.norm(direction)) < 1.0e-8:
                direction = arbitrary_perp(ca_dir)
            bridge["atoms"].append(make_atom(next_serial, "H", "H", n_xyz + _norm(direction) * BOND_N_H))
            next_serial += 1

    ca_xyz = get_atom_xyz(search_atom(bridge, "CA"))
    alpha_h_directions: list[np.ndarray] = []
    for atom_name in _GLY_ALPHA_HYDROGEN_NAMES:
        atom = search_atom(residue, atom_name)
        if atom is None:
            continue
        direction = get_atom_xyz(atom) - ca_xyz
        if float(np.linalg.norm(direction)) < 1.0e-8:
            continue
        normalized = _norm(direction)
        if any(abs(float(np.dot(normalized, existing))) >= 0.95 for existing in alpha_h_directions):
            continue
        alpha_h_directions.append(normalized)
        if len(alpha_h_directions) == 2:
            break

    if len(alpha_h_directions) < 2:
        n_dir = get_atom_xyz(n_atom) - ca_xyz
        c_dir = get_atom_xyz(c_atom) - ca_xyz
        fallback = -(n_dir + c_dir + sum(alpha_h_directions, np.zeros(3)))
        if float(np.linalg.norm(fallback)) < 1.0e-8:
            fallback = arbitrary_perp(_norm(n_dir + c_dir))
        if not any(abs(float(np.dot(_norm(fallback), direction))) >= 0.95 for direction in alpha_h_directions):
            alpha_h_directions.append(_norm(fallback))
    if len(alpha_h_directions) < 2:
        secondary = arbitrary_perp(_norm(get_atom_xyz(c_atom) - ca_xyz))
        if not any(abs(float(np.dot(secondary, direction))) >= 0.95 for direction in alpha_h_directions):
            alpha_h_directions.append(secondary)

    for name, direction in zip(("HA2", "HA3"), alpha_h_directions[:2], strict=False):
        next_serial = _append_hydrogen(bridge, name, ca_xyz, direction, BOND_C_H, next_serial)

    refresh_resid(bridge)
    return bridge, next_serial


__all__ = [
    "build_ace_cap",
    "build_gly_bridge",
    "build_nme_cap",
]
