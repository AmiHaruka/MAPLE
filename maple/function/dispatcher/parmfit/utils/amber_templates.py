"""Amber residue-template loading for PDB graph matching."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path
import re
import shlex

from ase.data import atomic_numbers, chemical_symbols

from .amber_data import amber_lib_dir


_SECTION_RE = re.compile(r"^!entry\.([^.]+)\.unit\.([a-z]+)")


def _resolve_element(atomic_number: int, atom_name: str) -> str | None:
    """Resolve an upper-case element symbol for one Amber lib atom row.

    The lib record carries an atomic number, so a direct periodic-table lookup is
    exact. A few legacy modified-nucleotide entries mark atoms with atomic number
    -1; for those the element is taken from the leading letters of the atom name
    and validated against the real periodic table rather than a hand-kept subset.
    """
    if 0 < atomic_number < len(chemical_symbols):
        return chemical_symbols[atomic_number].upper()
    letters = "".join(character for character in atom_name if character.isalpha())
    for width in (2, 1):
        candidate = letters[:width].capitalize()
        if candidate and candidate in atomic_numbers:
            return candidate.upper()
    return None


@dataclass(frozen=True)
class AmberTemplateAtom:
    name: str
    amber_type: str
    element: str
    charge: float
    xyz: tuple[float, float, float]


@dataclass(frozen=True)
class AmberResidueTemplate:
    template_id: str
    name: str
    output_name: str
    source: str
    category: str
    family: str
    leaprc: str | None
    atoms: tuple[AmberTemplateAtom, ...]
    bonds: frozenset[tuple[int, int]]
    connect: tuple[int, ...]
    net_charge: int | float

    @property
    def element_counts(self) -> tuple[tuple[str, int], ...]:
        return tuple(sorted(Counter(atom.element for atom in self.atoms).items()))


class AmberTemplateRegistry:
    def __init__(self, templates: list[AmberResidueTemplate]):
        self.templates = tuple(templates)
        by_name: dict[str, list[AmberResidueTemplate]] = defaultdict(list)
        by_composition: dict[tuple[int, tuple[tuple[str, int], ...]], list[AmberResidueTemplate]] = defaultdict(list)
        for template in templates:
            by_name[template.name].append(template)
            by_name[template.output_name].append(template)
            key = (len(template.atoms), template.element_counts)
            by_composition[key].append(template)
        self._by_name = {name: tuple(dict.fromkeys(items)) for name, items in by_name.items()}
        self._by_composition = {key: tuple(items) for key, items in by_composition.items()}
        self._by_id = {template.template_id: template for template in templates}

    def templates_for_name(self, name: str) -> tuple[AmberResidueTemplate, ...]:
        upper = str(name).strip().upper()
        aliases = {
            "HIS": ("HID", "HIE", "HIP"),
            "ASP": ("ASP", "ASH"),
            "GLU": ("GLU", "GLH"),
            "LYS": ("LYS", "LYN"),
        }
        names = aliases.get(upper, (upper,))
        result: list[AmberResidueTemplate] = []
        for candidate in names:
            result.extend(self._by_name.get(candidate, ()))
        return tuple(dict.fromkeys(result))

    def templates_for_elements(self, elements: list[str]) -> tuple[AmberResidueTemplate, ...]:
        key = (len(elements), tuple(sorted(Counter(elements).items())))
        return self._by_composition.get(key, ())

    def template_by_id(self, template_id: str) -> AmberResidueTemplate | None:
        return self._by_id.get(template_id)


def _lib_dir() -> Path:
    return amber_lib_dir()


def _read_sections(path: Path) -> dict[str, dict[str, list[str]]]:
    entries: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    current: tuple[str, str] | None = None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            match = _SECTION_RE.match(line)
            if match:
                current = (match.group(1).upper(), match.group(2))
                continue
            if current is not None and line and not line.startswith("!"):
                entries[current[0]][current[1]].append(line)
    return entries


def _output_name(name: str, category: str) -> str:
    if category == "nterm" and name.startswith("N") and name not in {"NHE", "NME"}:
        return name[1:]
    if category == "cterm" and name.startswith("C"):
        return name[1:]
    return name


def _parse_template_file(
    filename: str,
    *,
    category: str,
    family: str,
    leaprc: str | None,
) -> list[AmberResidueTemplate]:
    path = _lib_dir() / filename
    templates: list[AmberResidueTemplate] = []
    for name, sections in _read_sections(path).items():
        atom_rows = sections.get("atoms", ())
        if not atom_rows:
            continue
        positions = [tuple(float(value) for value in row.split()[:3]) for row in sections.get("positions", ())]
        if len(positions) != len(atom_rows):
            raise ValueError(f"Amber template {filename}:{name} has inconsistent atom and coordinate counts.")
        atoms: list[AmberTemplateAtom] = []
        for row, xyz in zip(atom_rows, positions):
            fields = shlex.split(row)
            atomic_number = int(fields[6])
            element = _resolve_element(atomic_number, fields[0])
            if element is None:
                raise ValueError(
                    f"Cannot resolve element for {filename}:{name}:{fields[0]} "
                    f"(atomic number {atomic_number}, name {fields[0]!r})."
                )
            atoms.append(
                AmberTemplateAtom(
                    name=fields[0],
                    amber_type=fields[1],
                    element=element,
                    charge=float(fields[7]),
                    xyz=xyz,
                )
            )
        bonds = {
            tuple(sorted((int(fields[0]) - 1, int(fields[1]) - 1)))
            for row in sections.get("connectivity", ())
            if len(fields := row.split()) >= 2
        }
        connect = tuple(int(row.split()[0]) - 1 for row in sections.get("connect", ()) if row.split())
        charge_sum = sum(atom.charge for atom in atoms)
        rounded_charge = int(round(charge_sum))
        net_charge = rounded_charge if abs(charge_sum - rounded_charge) < 0.05 else charge_sum
        template = AmberResidueTemplate(
            template_id=f"{filename}:{category}:{name}",
            name=name,
            output_name=_output_name(name, category),
            source=filename,
            category=category,
            family=family,
            leaprc=leaprc,
            atoms=tuple(atoms),
            bonds=frozenset(bonds),
            connect=connect,
            net_charge=net_charge,
        )
        _validate_template(template)
        templates.append(template)
    return templates


def _validate_template(template: AmberResidueTemplate) -> None:
    atom_count = len(template.atoms)
    if len({atom.name for atom in template.atoms}) != atom_count:
        raise ValueError(f"Amber template {template.template_id} contains duplicate atom names.")
    if any(index < 0 or index >= atom_count for bond in template.bonds for index in bond):
        raise ValueError(f"Amber template {template.template_id} contains an invalid bond index.")
    if any(left == right for left, right in template.bonds):
        raise ValueError(f"Amber template {template.template_id} contains a self bond.")
    if any(index < -1 or index >= atom_count for index in template.connect):
        raise ValueError(f"Amber template {template.template_id} contains an invalid connection index.")
    if any(
        not math.isfinite(value)
        for atom in template.atoms
        for value in (*atom.xyz, atom.charge)
    ):
        raise ValueError(f"Amber template {template.template_id} contains non-finite parameters.")
    if template.family == "protein" and not isinstance(template.net_charge, int):
        raise ValueError(f"Amber protein template {template.template_id} has a non-integral net charge.")


@lru_cache(maxsize=2)
def load_amber_template_registry(prom: str = "ff14SB") -> AmberTemplateRegistry:
    model = str(prom).strip().lower()
    if model not in {"ff14sb", "ff19sb"}:
        raise ValueError(f"Unsupported protein model {prom!r}; expected ff14SB or ff19SB.")
    model = "ff19SB" if model == "ff19sb" else "ff14SB"
    suffix = "19" if model == "ff19SB" else "12"
    specifications = (
        (f"amino{suffix}.lib", "internal", "protein", f"leaprc.protein.{model}"),
        ("aminont12.lib", "nterm", "protein", f"leaprc.protein.{model}"),
        ("aminoct12.lib", "cterm", "protein", f"leaprc.protein.{model}"),
        # Modified protein residues intentionally remain on the NCAA route.
        ("nucleic12.lib", "nucleic", "nucleic", None),
        ("all_modrna08.lib", "modified", "nucleic", None),
    )
    templates: list[AmberResidueTemplate] = []
    for filename, category, family, leaprc in specifications:
        templates.extend(_parse_template_file(filename, category=category, family=family, leaprc=leaprc))
    return AmberTemplateRegistry(templates)


def required_template_leaprcs(residues: list[dict], prom: str = "ff14SB") -> list[str]:
    registry = load_amber_template_registry(prom)
    base = f"leaprc.protein.{prom}"
    leaprcs: list[str] = []
    for residue in residues:
        template = registry.template_by_id(str(residue.get("template_id", "")))
        if (
            template is None
            or template.family != "protein"
            or template.leaprc in {None, base}
            or template.leaprc in leaprcs
        ):
            continue
        leaprcs.append(template.leaprc)
    return leaprcs
