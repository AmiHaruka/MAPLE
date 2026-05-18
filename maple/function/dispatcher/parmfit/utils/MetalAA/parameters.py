"""Usage: read and match Amber parameters for MetalAA exports."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from math import pi
from pathlib import Path
import re

from ..ionparams import infer_ion_frcmod_name, infer_ion_identity

_NUM = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


@dataclass(frozen=True)
class TorsionParameter:
    amplitude: float
    phase_deg: float
    periodicity: float


@dataclass
class AmberParameterDB:
    mass: dict[str, float] = field(default_factory=dict)
    bond: dict[tuple[str, str], tuple[float, float]] = field(default_factory=dict)
    angle: dict[tuple[str, str, str], tuple[float, float]] = field(default_factory=dict)
    dihedral: dict[tuple[str, str, str, str], list[TorsionParameter]] = field(default_factory=dict)
    improper: dict[tuple[str, str, str, str], list[TorsionParameter]] = field(default_factory=dict)
    nonbond: dict[str, tuple[float, float]] = field(default_factory=dict)


def parm_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "params" / "leap" / "parm"


def canonical_pair(atom_types: tuple[str, str]) -> tuple[str, str]:
    reverse = (atom_types[1], atom_types[0])
    return atom_types if atom_types <= reverse else reverse


def canonical_angle(atom_types: tuple[str, str, str]) -> tuple[str, str, str]:
    reverse = (atom_types[2], atom_types[1], atom_types[0])
    return atom_types if atom_types <= reverse else reverse


def _strip_comment(line: str) -> str:
    return line.split("!", 1)[0].split("#", 1)[0]


def _merge(into: AmberParameterDB, other: AmberParameterDB) -> AmberParameterDB:
    into.mass.update(other.mass)
    into.bond.update(other.bond)
    into.angle.update(other.angle)
    for key, terms in other.dihedral.items():
        into.dihedral[key] = list(terms)
    for key, terms in other.improper.items():
        into.improper[key] = list(terms)
    into.nonbond.update(other.nonbond)
    return into


def _parse_mass_lines(lines: list[str], db: AmberParameterDB) -> None:
    for raw in lines:
        line = _strip_comment(raw)
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            db.mass[parts[0]] = float(parts[1])
        except ValueError:
            continue


def _parse_bond_lines(lines: list[str], db: AmberParameterDB) -> None:
    bond_re = re.compile(rf"^\s*(\S+)\s*-\s*(\S+)\s+({_NUM})\s+({_NUM})")
    for raw in lines:
        match = bond_re.match(_strip_comment(raw))
        if match is None:
            continue
        key = canonical_pair((match.group(1), match.group(2)))
        db.bond[key] = (float(match.group(3)), float(match.group(4)))


def _parse_angle_lines(lines: list[str], db: AmberParameterDB) -> None:
    angle_re = re.compile(rf"^\s*(\S+)\s*-\s*(\S+)\s*-\s*(\S+)\s+({_NUM})\s+({_NUM})")
    for raw in lines:
        match = angle_re.match(_strip_comment(raw))
        if match is None:
            continue
        key = canonical_angle((match.group(1), match.group(2), match.group(3)))
        db.angle[key] = (float(match.group(4)), float(match.group(5)))


def _parse_dihedral_lines(lines: list[str], db: AmberParameterDB) -> None:
    dihe_re = re.compile(
        rf"^\s*(\S+)\s*-\s*(\S+)\s*-\s*(\S+)\s*-\s*(\S+)\s+({_NUM})\s+({_NUM})\s+({_NUM})\s+({_NUM})"
    )
    for raw in lines:
        match = dihe_re.match(_strip_comment(raw))
        if match is None:
            continue
        idivf = float(match.group(5))
        amplitude = float(match.group(6))
        if idivf != 0.0:
            amplitude /= idivf
        key = (match.group(1), match.group(2), match.group(3), match.group(4))
        db.dihedral.setdefault(key, []).append(
            TorsionParameter(
                amplitude=amplitude,
                phase_deg=float(match.group(7)),
                periodicity=float(match.group(8)),
            )
        )


def _parse_improper_lines(lines: list[str], db: AmberParameterDB) -> None:
    improper_re = re.compile(
        rf"^\s*(\S+)\s*-\s*(\S+)\s*-\s*(\S+)\s*-\s*(\S+)\s+({_NUM})\s+({_NUM})\s+({_NUM})"
    )
    for raw in lines:
        match = improper_re.match(_strip_comment(raw))
        if match is None:
            continue
        key = (match.group(1), match.group(2), match.group(3), match.group(4))
        db.improper.setdefault(key, []).append(
            TorsionParameter(
                amplitude=float(match.group(5)),
                phase_deg=float(match.group(6)),
                periodicity=float(match.group(7)),
            )
        )


def _parse_nonbond_lines(lines: list[str], db: AmberParameterDB) -> None:
    for raw in lines:
        line = _strip_comment(raw)
        parts = line.split()
        if len(parts) < 3 or parts[0].upper() in {"MOD4", "END"}:
            continue
        try:
            db.nonbond[parts[0]] = (float(parts[1]), float(parts[2]))
        except ValueError:
            continue


def _parse_nonbond_equivalence_lines(lines: list[str]) -> dict[str, list[str]]:
    equivalents: dict[str, list[str]] = {}
    for raw in lines:
        parts = _strip_comment(raw).split()
        if len(parts) < 2:
            continue
        try:
            [float(part) for part in parts[1:]]
        except ValueError:
            equivalents[parts[0]] = parts[1:]
    return equivalents


def _expand_nonbond_equivalences(lines: list[str], db: AmberParameterDB) -> None:
    for source, aliases in _parse_nonbond_equivalence_lines(lines).items():
        if source not in db.nonbond:
            continue
        for alias in aliases:
            db.nonbond.setdefault(alias, db.nonbond[source])


def _split_nonempty_sections(path: Path, *, skip_title: bool = False) -> list[list[str]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if skip_title and lines:
        lines = lines[1:]
    sections: list[list[str]] = []
    current: list[str] = []
    for raw in lines:
        if raw.strip():
            current.append(raw)
            continue
        if current:
            sections.append(current)
            current = []
    if current:
        sections.append(current)
    return sections


def parse_amber_dat(path: Path) -> AmberParameterDB:
    db = AmberParameterDB()
    sections = _split_nonempty_sections(path, skip_title=True)
    if len(sections) > 0:
        _parse_mass_lines(sections[0], db)
    if len(sections) > 1:
        _parse_bond_lines(sections[1], db)
    if len(sections) > 2:
        _parse_angle_lines(sections[2], db)
    if len(sections) > 3:
        _parse_dihedral_lines(sections[3], db)
    if len(sections) > 4:
        _parse_improper_lines(sections[4], db)
    if len(sections) > 6:
        _parse_nonbond_lines(sections[6], db)
    if len(sections) > 5:
        _expand_nonbond_equivalences(sections[5], db)
    return db


def parse_amber_frcmod(path: Path) -> AmberParameterDB:
    db = AmberParameterDB()
    section: str | None = None
    section_lines: list[str] = []

    def flush() -> None:
        nonlocal section_lines
        if section == "MASS":
            _parse_mass_lines(section_lines, db)
        elif section == "BOND":
            _parse_bond_lines(section_lines, db)
        elif section == "ANGLE":
            _parse_angle_lines(section_lines, db)
        elif section == "DIHE":
            _parse_dihedral_lines(section_lines, db)
        elif section == "IMPROPER":
            _parse_improper_lines(section_lines, db)
        elif section == "NONBON":
            _parse_nonbond_lines(section_lines, db)
        section_lines = []

    section_aliases = {
        "MASS": "MASS",
        "BOND": "BOND",
        "ANGL": "ANGLE",
        "ANGLE": "ANGLE",
        "DIHE": "DIHE",
        "IMPR": "IMPROPER",
        "IMPROPER": "IMPROPER",
        "NONB": "NONBON",
        "NONBON": "NONBON",
    }
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            stripped = raw.strip()
            upper = stripped.upper()
            if upper in section_aliases:
                flush()
                section = section_aliases[upper]
                continue
            if upper == "CMAP":
                flush()
                section = None
                continue
            if section is None:
                continue
            section_lines.append(raw.rstrip("\n"))
    flush()
    return db


@lru_cache(maxsize=8)
def load_default_parameters(watm: str = "opc") -> AmberParameterDB:
    base = parm_dir()
    db = AmberParameterDB()
    for filename, parser in (
        ("parm19.dat", parse_amber_dat),
        ("frcmod.ff19SB", parse_amber_frcmod),
        ("gaff2.dat", parse_amber_dat),
        (f"frcmod.{watm}", parse_amber_frcmod),
    ):
        path = base / filename
        if path.exists():
            _merge(db, parser(path))
    return db


def match_dihedral(
    atom_types: tuple[str, str, str, str],
    templates: dict[tuple[str, str, str, str], list[TorsionParameter]],
) -> list[TorsionParameter]:
    match = match_dihedral_template(atom_types, templates)
    return [] if match is None else match[1]


def match_dihedral_template(
    atom_types: tuple[str, str, str, str],
    templates: dict[tuple[str, str, str, str], list[TorsionParameter]],
) -> tuple[tuple[str, str, str, str], list[TorsionParameter], bool] | None:
    best_match: tuple[tuple[str, str, str, str], list[TorsionParameter], bool] | None = None
    best_score = -1
    for template, terms in templates.items():
        for reversed_match, candidate in ((False, atom_types), (True, tuple(reversed(atom_types)))):
            if all(template_type == "X" or template_type == atom_type for template_type, atom_type in zip(template, candidate)):
                score = sum(template_type != "X" for template_type in template)
                if score > best_score:
                    best_score = score
                    best_match = (template, list(terms), reversed_match)
                break
    return best_match


def match_improper(
    atom_types: tuple[str, str, str, str],
    templates: dict[tuple[str, str, str, str], list[TorsionParameter]],
) -> list[TorsionParameter]:
    from itertools import permutations

    outer = (atom_types[0], atom_types[1], atom_types[3])
    center = atom_types[2]
    best_terms: list[TorsionParameter] = []
    best_score = -1
    for template, terms in templates.items():
        if template[2] != "X" and template[2] != center:
            continue
        for perm in permutations(outer):
            candidate = (perm[0], perm[1], center, perm[2])
            if all(template_type == "X" or template_type == atom_type for template_type, atom_type in zip(template, candidate)):
                score = sum(template_type != "X" for template_type in template)
                if score > best_score:
                    best_score = score
                    best_terms = list(terms)
                break
    return best_terms


def amber_ion_atom_type(element: str, formal_charge: int) -> str:
    normalized = element[0].upper() + element[1:].lower()
    charge = int(formal_charge)
    if charge == 0:
        return normalized
    if charge == 1:
        return f"{normalized}+"
    if charge == -1:
        return f"{normalized}-"
    suffix = "+" if charge > 0 else "-"
    return f"{normalized}{abs(charge)}{suffix}"


def lookup_ion_lj_from_frcmod(*, watm: str, ionm: str, residue: dict | str) -> tuple[str, str, float, tuple[float, float]]:
    element, formal_charge, _ion_key = infer_ion_identity(residue)
    frcmod_name = infer_ion_frcmod_name(watm=watm, ionm=ionm, residue=residue)
    path = parm_dir() / frcmod_name
    if not path.exists():
        raise ValueError(f"Could not find ion frcmod file {frcmod_name!r} under {parm_dir()}.")
    db = parse_amber_frcmod(path)
    amber_type = amber_ion_atom_type(element, formal_charge)
    nonbond = db.nonbond.get(amber_type)
    if nonbond is None:
        raise ValueError(
            f"Ion frcmod {frcmod_name!r} does not define NONBON parameters for {amber_type!r}."
        )
    mass = db.mass.get(amber_type)
    if mass is None:
        raise ValueError(f"Ion frcmod {frcmod_name!r} does not define MASS for {amber_type!r}.")
    return frcmod_name, amber_type, mass, nonbond
