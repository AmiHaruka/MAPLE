from __future__ import annotations

import os
from math import degrees

import numpy as np
from ase import Atoms

from .mmcalc import build_mm_topology_cache
from .readparm import CorrectionParameterSet


_KCAL_TO_KJ = 4.184
_ANG_TO_NM = 0.1
_BOX_PADDING_NM = 0.5
_MIN_BOX_LENGTH_NM = 0.5
_SIGMA_DENOM = 2.0 ** (1.0 / 6.0)
_ATOMIC_FALLBACK = {
    "H": (1, 1.008),
    "B": (5, 10.81),
    "C": (6, 12.01),
    "N": (7, 14.01),
    "O": (8, 16.00),
    "F": (9, 19.00),
    "P": (15, 30.97),
    "S": (16, 32.06),
    "CL": (17, 35.45),
    "BR": (35, 79.904),
    "I": (53, 126.90),
}


def _section(title: str, body: list[str]) -> list[str]:
    return [f"[ {title} ]\n"] + body + ["\n"]


def _comment(line: str) -> str:
    return f"; {line}\n"


def _normalize_symbol(symbol: str) -> str:
    if len(symbol) <= 1:
        return symbol.upper()
    return symbol[0].upper() + symbol[1:].lower()


def _atomic_number_and_mass(symbol: str) -> tuple[int, float]:
    normalized = _normalize_symbol(symbol)
    try:
        from ase.data import atomic_masses, atomic_numbers  # type: ignore

        atomic_number = int(atomic_numbers[normalized])
        return atomic_number, float(atomic_masses[atomic_number])
    except Exception:
        key = normalized.upper()
        if key not in _ATOMIC_FALLBACK:
            raise ValueError(f"GROMACS export does not know atomic number/mass for element '{symbol}'.")
        return _ATOMIC_FALLBACK[key]


def _box_lengths_nm(atoms: Atoms) -> tuple[float, float, float]:
    positions_nm = np.asarray(atoms.get_positions(), dtype=float) * _ANG_TO_NM
    mins = np.min(positions_nm, axis=0)
    maxs = np.max(positions_nm, axis=0)
    lengths = (maxs - mins) + _BOX_PADDING_NM
    lengths = np.maximum(lengths, _MIN_BOX_LENGTH_NM)
    return float(lengths[0]), float(lengths[1]), float(lengths[2])


def _title_from_base(output_base: str, title: str | None) -> str:
    if title:
        return str(title)
    return os.path.basename(os.path.splitext(output_base)[0]) or "MAPLE parmfit export"


def _format_defaults_lines() -> list[str]:
    return [
        _comment("nbfunc        comb-rule       gen-pairs       fudgeLJ fudgeQQ"),
        "1               2               yes             0.5     0.83333333\n",
    ]


def _format_atomtypes_lines(parameter_set: CorrectionParameterSet, atoms: Atoms, meta: dict) -> list[str]:
    lines = [_comment("name    at.num    mass    charge ptype  sigma      epsilon")]
    symbols = atoms.get_chemical_symbols()
    seen: set[str] = set()
    for nonbond in parameter_set.nonbonds:
        if nonbond.atom_type in seen:
            continue
        seen.add(nonbond.atom_type)
        atom_index = nonbond.atom - 1
        atomic_number, mass = _atomic_number_and_mass(symbols[atom_index])
        if nonbond.rmin_half is None or nonbond.epsilon is None:
            meta["omitted_counts"]["atomtypes"] += 1
            warning = f"atom type {nonbond.atom_type} omitted from [ atomtypes ] because LJ parameters are missing"
            meta["warnings"].append(warning)
            lines.append(_comment(warning))
            continue
        sigma_nm = (2.0 * float(nonbond.rmin_half) / _SIGMA_DENOM) * _ANG_TO_NM
        epsilon_kj = float(nonbond.epsilon) * _KCAL_TO_KJ
        lines.append(
            f"{nonbond.atom_type:<12}{atomic_number:>5d}  {mass:>10.6f}  0.00000000  A  "
            f"{sigma_nm:>12.8f}  {epsilon_kj:>12.7f}\n"
        )
    return lines


def _format_atoms_lines(parameter_set: CorrectionParameterSet, atoms: Atoms) -> list[str]:
    lines = [
        _comment("nr       type  resnr residue  atom   cgnr    charge       mass"),
        _comment("residue    1 MOL rtp MOL q 0.0"),
    ]
    symbols = atoms.get_chemical_symbols()
    for index, (mol2_atom, nonbond) in enumerate(zip(parameter_set.mol2.atoms, parameter_set.nonbonds), start=1):
        _, mass = _atomic_number_and_mass(symbols[index - 1])
        lines.append(
            f"{index:>5d}  {nonbond.atom_type:>9s}  {1:>5d}  {'MOL':>6s}  {mol2_atom.name:>5s}  {index:>5d}"
            f"  {nonbond.charge:>10.8f}  {mass:>10.6f}\n"
        )
    return lines


def _format_bonds_lines(parameter_set: CorrectionParameterSet, meta: dict) -> list[str]:
    lines = [_comment("ai     aj funct         c0         c1")]
    for bond in parameter_set.bonds:
        if bond.kBond is None or bond.rEq is None:
            meta["omitted_counts"]["bonds"] += 1
            warning = f"bond {bond.atoms} omitted from [ bonds ] because kBond/rEq is missing"
            meta["warnings"].append(warning)
            lines.append(_comment(warning))
            continue
        r0_nm = float(bond.rEq) * _ANG_TO_NM
        k_gmx = 2.0 * float(bond.kBond) * _KCAL_TO_KJ * 100.0
        lines.append(f"{bond.atoms[0]:>6d}{bond.atoms[1]:>7d}{1:>6d}{r0_nm:>12.5f}{k_gmx:>14.6f}\n")
    return lines


def _format_pairs_lines(parameter_set: CorrectionParameterSet) -> list[str]:
    cache = build_mm_topology_cache(parameter_set)
    lines = [_comment("ai     aj funct")]
    for atom_i, atom_j in sorted(cache.scaled_14):
        lines.append(f"{atom_i:>6d}{atom_j:>7d}{1:>6d}\n")
    return lines


def _format_angles_lines(parameter_set: CorrectionParameterSet, meta: dict) -> list[str]:
    lines = [_comment("ai     aj     ak funct         c0         c1")]
    for angle in parameter_set.angles:
        if angle.kTheta is None or angle.thetaEq is None:
            meta["omitted_counts"]["angles"] += 1
            warning = f"angle {angle.atoms} omitted from [ angles ] because kTheta/thetaEq is missing"
            meta["warnings"].append(warning)
            lines.append(_comment(warning))
            continue
        theta_deg = degrees(float(angle.thetaEq))
        k_gmx = 2.0 * float(angle.kTheta) * _KCAL_TO_KJ
        lines.append(
            f"{angle.atoms[0]:>6d}{angle.atoms[1]:>7d}{angle.atoms[2]:>7d}{1:>6d}"
            f"{theta_deg:>12.6f}{k_gmx:>14.6f}\n"
        )
    return lines


def _period_to_mult(period: float, label: str, meta: dict) -> int:
    rounded = int(round(float(period)))
    if abs(float(period) - rounded) > 1.0e-8:
        warning = f"{label} period {period} was rounded to integer multiplicity {rounded} for GROMACS export"
        meta["warnings"].append(warning)
    return rounded


def _format_dihedrals_lines(parameter_set: CorrectionParameterSet, meta: dict) -> list[str]:
    lines = [_comment("ai     aj     ak     al funct         c0         c1    mult")]
    for dihedral in parameter_set.dihedrals:
        if not dihedral.terms:
            meta["omitted_counts"]["dihedrals"] += 1
            warning = f"proper dihedral {dihedral.atoms} omitted from [ dihedrals ] because no torsion terms are assigned"
            meta["warnings"].append(warning)
            lines.append(_comment(warning))
            continue
        for term in dihedral.terms:
            lines.append(
                f"{dihedral.atoms[0]:>6d}{dihedral.atoms[1]:>7d}{dihedral.atoms[2]:>7d}{dihedral.atoms[3]:>7d}{1:>6d}"
                f"{degrees(float(term.phase)):>12.7f}{(float(term.kPhi) * _KCAL_TO_KJ):>12.7f}"
                f"{_period_to_mult(term.period, f'proper dihedral {dihedral.atoms}', meta):>5d}\n"
            )
    for improper in parameter_set.impropers:
        if not improper.terms:
            meta["omitted_counts"]["impropers"] += 1
            continue
        for term in improper.terms:
            lines.append(
                f"{improper.atoms[0]:>6d}{improper.atoms[1]:>7d}{improper.atoms[2]:>7d}{improper.atoms[3]:>7d}{4:>6d}"
                f"{degrees(float(term.phase)):>12.7f}{(float(term.kPhi) * _KCAL_TO_KJ):>12.7f}"
                f"{_period_to_mult(term.period, f'improper {improper.atoms}', meta):>5d}\n"
            )
    return lines


def _format_system_lines(title: str) -> list[str]:
    return [_comment("Name"), f"{title}\n"]


def _format_molecules_lines() -> list[str]:
    return [_comment("Compound       #mols"), f"{'MOL':<16}{1:>5d}\n"]


def write_top(
    parameter_set: CorrectionParameterSet,
    atoms: Atoms,
    top_path: str,
    title: str | None = None,
) -> dict:
    meta = {
        "warnings": [],
        "omitted_counts": {
            "atomtypes": 0,
            "bonds": 0,
            "angles": 0,
            "dihedrals": 0,
            "impropers": 0,
            "nonbonds": 0,
        },
        "written_sections": [
            "defaults",
            "atomtypes",
            "moleculetype",
            "atoms",
            "bonds",
            "pairs",
            "angles",
            "dihedrals",
            "system",
            "molecules",
        ],
    }
    title_str = _title_from_base(top_path, title)
    lines = [
        _comment(f"File {os.path.basename(top_path)} was generated by MAPLE parmfit"),
        _comment("This is a standalone topology file"),
    ]
    lines.extend(_section("defaults", _format_defaults_lines()))
    lines.extend(_section("atomtypes", _format_atomtypes_lines(parameter_set, atoms, meta)))
    lines.extend(_section("moleculetype", [_comment("Name            nrexcl"), f"{'MOL':<12}{3:>5d}\n"]))
    lines.extend(_section("atoms", _format_atoms_lines(parameter_set, atoms)))
    lines.extend(_section("bonds", _format_bonds_lines(parameter_set, meta)))
    lines.extend(_section("pairs", _format_pairs_lines(parameter_set)))
    lines.extend(_section("angles", _format_angles_lines(parameter_set, meta)))
    lines.extend(_section("dihedrals", _format_dihedrals_lines(parameter_set, meta)))
    lines.extend(_section("system", _format_system_lines(title_str)))
    lines.extend(_section("molecules", _format_molecules_lines()))

    if meta["warnings"]:
        warning_header = [_comment("Warnings during GROMACS export:")]
        warning_header.extend(_comment(warning) for warning in meta["warnings"])
        lines = warning_header + ["\n"] + lines

    with open(top_path, "w", encoding="utf-8") as handle:
        handle.writelines(lines)
    return meta


def write_gro(
    parameter_set: CorrectionParameterSet,
    atoms: Atoms,
    gro_path: str,
    title: str | None = None,
) -> str:
    title_str = _title_from_base(gro_path, title)
    positions_nm = np.asarray(atoms.get_positions(), dtype=float) * _ANG_TO_NM
    box_x, box_y, box_z = _box_lengths_nm(atoms)

    with open(gro_path, "w", encoding="utf-8") as handle:
        handle.write(f"{title_str}\n")
        handle.write(f"{len(atoms):5d}\n")
        for index, (mol2_atom, xyz_nm) in enumerate(zip(parameter_set.mol2.atoms, positions_nm), start=1):
            atom_name = mol2_atom.name[-5:]
            handle.write(
                f"{1:5d}{'MOL':<5}{atom_name:>5}{index:5d}{xyz_nm[0]:8.3f}{xyz_nm[1]:8.3f}{xyz_nm[2]:8.3f}\n"
            )
        handle.write(f"{box_x:10.5f}{box_y:10.5f}{box_z:10.5f}\n")
    return gro_path


def write_gromacs_files(
    parameter_set: CorrectionParameterSet,
    atoms: Atoms,
    output_base: str,
    title: str | None = None,
) -> tuple[str, str, dict]:
    base = os.path.splitext(output_base)[0]
    top_path = base + ".top"
    gro_path = base + ".gro"
    meta = write_top(parameter_set, atoms, top_path, title=title)
    write_gro(parameter_set, atoms, gro_path, title=title)
    return top_path, gro_path, meta
