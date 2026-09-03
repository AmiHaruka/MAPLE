"""Usage: parse mol2/frcmod files into parmfit parameter objects."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import combinations, permutations
from math import pi
from typing import Iterable, Optional

from ase import Atoms


@dataclass(frozen=True)
class FourierTerm:
    """Single Fourier term for proper or improper torsions."""

    kPhi: float
    period: float
    phase: float

    def __str__(self) -> str:
        return f"<k={self.kPhi:.6f}, n={self.period:.3f}, phase={self.phase * 180.0 / pi:.2f}>"

    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class Bond:
    atoms: tuple[int, int]
    atom_types: tuple[str, str]
    kBond: Optional[float] = None
    rEq: Optional[float] = None

    def __str__(self) -> str:
        return f"<{self.atoms}, r={self.rEq}>"

    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class Angle:
    atoms: tuple[int, int, int]
    atom_types: tuple[str, str, str]
    kTheta: Optional[float] = None
    thetaEq: Optional[float] = None

    def __str__(self) -> str:
        angle_deg = None if self.thetaEq is None else self.thetaEq * 180.0 / pi
        return f"<{self.atoms}, ang={angle_deg}>"

    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class Dihedral:
    atoms: tuple[int, int, int, int]
    atom_types: tuple[str, str, str, str]
    terms: list[FourierTerm] = field(default_factory=list)

    def __str__(self) -> str:
        return f"<{self.atoms}, n_terms={len(self.terms)}>"

    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class Improper:
    atoms: tuple[int, int, int, int]
    atom_types: tuple[str, str, str, str]
    terms: list[FourierTerm] = field(default_factory=list)

    def __str__(self) -> str:
        return f"<{self.atoms}, n_terms={len(self.terms)}>"

    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class Nonbond:
    atom: int
    atom_type: str
    charge: float
    rmin_half: Optional[float] = None
    epsilon: Optional[float] = None

    def __str__(self) -> str:
        return f"<{self.atom}:{self.atom_type}, q={self.charge}>"

    def __repr__(self) -> str:
        return self.__str__()


_NUM = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


@dataclass(frozen=True)
class Mol2Atom:
    atom_id: int
    name: str
    atom_type: str
    charge: float


@dataclass(frozen=True)
class Mol2Bond:
    bond_id: int
    atom1: int
    atom2: int
    bond_type: str


@dataclass
class Mol2Topology:
    atoms: list[Mol2Atom]
    bonds: list[Mol2Bond]
    id_to_index: dict[int, int]
    adjacency: dict[int, set[int]] = field(default_factory=dict)


@dataclass
class FrcmodDB:
    mass_params: dict[str, float] = field(default_factory=dict)
    bond_params: dict[tuple[str, str], tuple[float, float]] = field(default_factory=dict)
    angle_params: dict[tuple[str, str, str], tuple[float, float]] = field(default_factory=dict)
    dihedral_params: dict[tuple[str, str, str, str], list[FourierTerm]] = field(default_factory=dict)
    improper_params: dict[tuple[str, str, str, str], list[FourierTerm]] = field(default_factory=dict)
    nonbond_params: dict[str, tuple[float, float]] = field(default_factory=dict)


@dataclass
class CorrectionParameterSet:
    mol2: Mol2Topology
    frcmod: FrcmodDB
    bonds: list[Bond]
    angles: list[Angle]
    dihedrals: list[Dihedral]
    impropers: list[Improper]
    nonbonds: list[Nonbond]
    unmatched_bonds: list[tuple[int, int]]
    unmatched_angles: list[tuple[int, int, int]]
    unmatched_dihedrals: list[tuple[int, int, int, int]]
    unmatched_impropers: list[tuple[int, int, int, int]]
    unmatched_nonbonds: list[int]


def _canonical_pair(atom_types: tuple[str, str]) -> tuple[str, str]:
    reverse = (atom_types[1], atom_types[0])
    return atom_types if atom_types <= reverse else reverse


def _canonical_angle(atom_types: tuple[str, str, str]) -> tuple[str, str, str]:
    reverse = (atom_types[2], atom_types[1], atom_types[0])
    return atom_types if atom_types <= reverse else reverse


def parse_mol2(path: str) -> Mol2Topology:
    """Parse the minimal mol2 fields needed by correction parmfit."""
    atoms: list[Mol2Atom] = []
    bonds: list[Mol2Bond] = []
    section: str | None = None

    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            stripped = raw.strip()
            if not stripped:
                continue
            if stripped.startswith("@<TRIPOS>"):
                section = stripped.upper()
                continue

            if section == "@<TRIPOS>ATOM":
                parts = raw.split()
                if len(parts) < 6:
                    raise ValueError(f"Invalid mol2 atom line in {path}: {raw.rstrip()}")
                atom_id = int(parts[0])
                name = parts[1]
                atom_type = parts[5]
                charge = float(parts[8]) if len(parts) >= 9 else 0.0
                atoms.append(Mol2Atom(atom_id=atom_id, name=name, atom_type=atom_type, charge=charge))
            elif section == "@<TRIPOS>BOND":
                parts = raw.split()
                if len(parts) < 4:
                    raise ValueError(f"Invalid mol2 bond line in {path}: {raw.rstrip()}")
                bonds.append(
                    Mol2Bond(
                        bond_id=int(parts[0]),
                        atom1=int(parts[1]),
                        atom2=int(parts[2]),
                        bond_type=parts[3],
                    )
                )

    if not atoms:
        raise ValueError(f"mol2 file has no @<TRIPOS>ATOM section: {path}")
    if not bonds:
        raise ValueError(f"mol2 file has no @<TRIPOS>BOND section: {path}")

    id_to_index: dict[int, int] = {}
    for index, atom in enumerate(atoms, start=1):
        if atom.atom_id in id_to_index:
            raise ValueError(f"Duplicate mol2 atom id {atom.atom_id} in {path}")
        id_to_index[atom.atom_id] = index

    adjacency: dict[int, set[int]] = {index: set() for index in range(1, len(atoms) + 1)}
    for bond in bonds:
        if bond.atom1 not in id_to_index or bond.atom2 not in id_to_index:
            raise ValueError(f"mol2 bond references unknown atom id in {path}: {bond}")
        i = id_to_index[bond.atom1]
        j = id_to_index[bond.atom2]
        adjacency[i].add(j)
        adjacency[j].add(i)

    return Mol2Topology(atoms=atoms, bonds=bonds, id_to_index=id_to_index, adjacency=adjacency)


def parse_frcmod(path: str) -> FrcmodDB:
    """Parse frcmod bonded and nonbonded parameter sections."""
    db = FrcmodDB()
    section: str | None = None

    bond_re = re.compile(rf"^\s*(\S+)\s*-\s*(\S+)\s+({_NUM})\s+({_NUM})")
    angle_re = re.compile(rf"^\s*(\S+)\s*-\s*(\S+)\s*-\s*(\S+)\s+({_NUM})\s+({_NUM})")
    dihe_re = re.compile(
        rf"^\s*(\S+)\s*-\s*(\S+)\s*-\s*(\S+)\s*-\s*(\S+)\s+({_NUM})\s+({_NUM})\s+({_NUM})\s+({_NUM})"
    )
    impr_re = re.compile(rf"^\s*(\S+)\s*-\s*(\S+)\s*-\s*(\S+)\s*-\s*(\S+)\s+({_NUM})\s+({_NUM})\s+({_NUM})")

    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            stripped = raw.strip()
            if not stripped:
                continue
            upper = stripped.upper()
            if upper.startswith("REMARK"):
                continue
            if upper in {"MASS", "BOND", "ANGLE", "DIHE", "IMPROPER", "NONBON"}:
                section = upper
                continue

            if section == "MASS":
                parts = raw.split()
                db.mass_params[parts[0]] = float(parts[1])
                continue

            if section == "BOND":
                match = bond_re.match(raw)
                if not match:
                    continue
                key = _canonical_pair((match.group(1), match.group(2)))
                db.bond_params[key] = (float(match.group(3)), float(match.group(4)))
                continue

            if section == "ANGLE":
                match = angle_re.match(raw)
                if not match:
                    continue
                key = _canonical_angle((match.group(1), match.group(2), match.group(3)))
                db.angle_params[key] = (float(match.group(4)), float(match.group(5)) * pi / 180.0)
                continue

            if section == "DIHE":
                match = dihe_re.match(raw)
                if not match:
                    continue
                idivf = float(match.group(5))
                pk = float(match.group(6))
                phase = float(match.group(7)) * pi / 180.0
                period = abs(float(match.group(8)))
                key = (match.group(1), match.group(2), match.group(3), match.group(4))
                db.dihedral_params.setdefault(key, []).append(
                    FourierTerm(kPhi=pk / idivf if idivf != 0.0 else pk, period=period, phase=phase)
                )
                continue

            if section == "IMPROPER":
                match = impr_re.match(raw)
                if not match:
                    continue
                key = (match.group(1), match.group(2), match.group(3), match.group(4))
                db.improper_params.setdefault(key, []).append(
                    FourierTerm(
                        kPhi=float(match.group(5)),
                        phase=float(match.group(6)) * pi / 180.0,
                        period=abs(float(match.group(7))),
                    )
                )
                continue

            if section == "NONBON":
                parts = raw.split()
                if len(parts) < 3:
                    continue
                db.nonbond_params[parts[0]] = (float(parts[1]), float(parts[2]))

    return db


def _iter_mol2_bonds(topology: Mol2Topology) -> Iterable[tuple[int, int]]:
    for bond in topology.bonds:
        i = topology.id_to_index[bond.atom1]
        j = topology.id_to_index[bond.atom2]
        yield (i, j) if i < j else (j, i)


def _enumerate_angles(topology: Mol2Topology) -> list[tuple[int, int, int]]:
    angles: list[tuple[int, int, int]] = []
    for center, neighbors in topology.adjacency.items():
        for left, right in combinations(sorted(neighbors), 2):
            angles.append((left, center, right))
    return angles


def _enumerate_dihedrals(topology: Mol2Topology) -> list[tuple[int, int, int, int]]:
    seen: set[tuple[int, int, int, int]] = set()
    dihedrals: list[tuple[int, int, int, int]] = []

    for j in sorted(topology.adjacency):
        for k in sorted(topology.adjacency[j]):
            if j >= k:
                continue
            for i in sorted(topology.adjacency[j] - {k}):
                for l in sorted(topology.adjacency[k] - {j}):
                    if len({i, j, k, l}) != 4:
                        continue
                    path = (i, j, k, l)
                    reverse = tuple(reversed(path))
                    canonical = path if path <= reverse else reverse
                    if canonical in seen:
                        continue
                    seen.add(canonical)
                    dihedrals.append(canonical)
    return dihedrals


def _enumerate_improper_candidates(topology: Mol2Topology) -> list[tuple[int, int, int, int]]:
    candidates: list[tuple[int, int, int, int]] = []
    for center, neighbors in topology.adjacency.items():
        if len(neighbors) < 3:
            continue
        for trio in combinations(sorted(neighbors), 3):
            candidates.append((trio[0], trio[1], center, trio[2]))
    return candidates


def _instance_atom_types(indices: tuple[int, ...], topology: Mol2Topology) -> tuple[str, ...]:
    return tuple(topology.atoms[index - 1].atom_type for index in indices)


def _match_dihedral(
    atom_types: tuple[str, str, str, str],
    templates: dict[tuple[str, str, str, str], list[FourierTerm]],
) -> list[FourierTerm]:
    best_terms: list[FourierTerm] = []
    best_score = -1

    for template, terms in templates.items():
        for candidate in (atom_types, tuple(reversed(atom_types))):
            if all(t == "X" or t == a for t, a in zip(template, candidate)):
                score = sum(t != "X" for t in template)
                if score > best_score:
                    best_score = score
                    best_terms = list(terms)
                break
    return best_terms


def _match_improper(
    atom_types: tuple[str, str, str, str],
    templates: dict[tuple[str, str, str, str], list[FourierTerm]],
) -> list[FourierTerm]:
    outer = (atom_types[0], atom_types[1], atom_types[3])
    center = atom_types[2]
    best_terms: list[FourierTerm] = []
    best_score = -1

    for template, terms in templates.items():
        if template[2] != "X" and template[2] != center:
            continue
        for perm in permutations(outer):
            candidate = (perm[0], perm[1], center, perm[2])
            if all(t == "X" or t == a for t, a in zip(template, candidate)):
                score = sum(t != "X" for t in template)
                if score > best_score:
                    best_score = score
                    best_terms = list(terms)
                break
    return best_terms


def build_correction_parameter_set(atoms: Atoms, mol2_path: str, frcmod_path: str) -> CorrectionParameterSet:
    """Build correction-mode parmfit instances from inp atoms, mol2 topology, and frcmod templates."""
    mol2 = parse_mol2(mol2_path)
    frcmod = parse_frcmod(frcmod_path)

    if len(atoms) != len(mol2.atoms):
        raise ValueError(
            f"inp coordinate atom count ({len(atoms)}) does not match mol2 atom count ({len(mol2.atoms)})."
        )

    bonds: list[Bond] = []
    unmatched_bonds: list[tuple[int, int]] = []
    for bond_atoms in sorted(set(_iter_mol2_bonds(mol2))):
        atom_types = _instance_atom_types(bond_atoms, mol2)
        params = frcmod.bond_params.get(_canonical_pair(atom_types))
        if params is None:
            unmatched_bonds.append(bond_atoms)
        bonds.append(
            Bond(
                atoms=bond_atoms,
                atom_types=atom_types,
                kBond=None if params is None else params[0],
                rEq=None if params is None else params[1],
            )
        )

    angles: list[Angle] = []
    unmatched_angles: list[tuple[int, int, int]] = []
    for angle_atoms in _enumerate_angles(mol2):
        atom_types = _instance_atom_types(angle_atoms, mol2)
        params = frcmod.angle_params.get(_canonical_angle(atom_types))
        if params is None:
            unmatched_angles.append(angle_atoms)
        angles.append(
            Angle(
                atoms=angle_atoms,
                atom_types=atom_types,
                kTheta=None if params is None else params[0],
                thetaEq=None if params is None else params[1],
            )
        )

    dihedrals: list[Dihedral] = []
    unmatched_dihedrals: list[tuple[int, int, int, int]] = []
    for dihedral_atoms in _enumerate_dihedrals(mol2):
        atom_types = _instance_atom_types(dihedral_atoms, mol2)
        terms = _match_dihedral(atom_types, frcmod.dihedral_params)
        if not terms:
            unmatched_dihedrals.append(dihedral_atoms)
        dihedrals.append(Dihedral(atoms=dihedral_atoms, atom_types=atom_types, terms=list(terms)))

    impropers: list[Improper] = []
    unmatched_impropers: list[tuple[int, int, int, int]] = []
    for improper_atoms in _enumerate_improper_candidates(mol2):
        atom_types = _instance_atom_types(improper_atoms, mol2)
        terms = _match_improper(atom_types, frcmod.improper_params)
        if not terms:
            unmatched_impropers.append(improper_atoms)
            continue
        impropers.append(Improper(atoms=improper_atoms, atom_types=atom_types, terms=list(terms)))

    nonbonds: list[Nonbond] = []
    unmatched_nonbonds: list[int] = []
    for index, mol2_atom in enumerate(mol2.atoms, start=1):
        params = frcmod.nonbond_params.get(mol2_atom.atom_type)
        if params is None:
            unmatched_nonbonds.append(index)
        nonbonds.append(
            Nonbond(
                atom=index,
                atom_type=mol2_atom.atom_type,
                charge=mol2_atom.charge,
                rmin_half=None if params is None else params[0],
                epsilon=None if params is None else params[1],
            )
        )

    return CorrectionParameterSet(
        mol2=mol2,
        frcmod=frcmod,
        bonds=bonds,
        angles=angles,
        dihedrals=dihedrals,
        impropers=impropers,
        nonbonds=nonbonds,
        unmatched_bonds=unmatched_bonds,
        unmatched_angles=unmatched_angles,
        unmatched_dihedrals=unmatched_dihedrals,
        unmatched_impropers=unmatched_impropers,
        unmatched_nonbonds=unmatched_nonbonds,
    )
