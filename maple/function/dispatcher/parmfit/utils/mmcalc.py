import numpy as np
from collections import deque
from dataclasses import dataclass, field
from math import cos, sqrt

from ase import Atoms

from .mmfunc import angle_radians, dihedral_radians, distance_angstrom
from .parm import Dihedral
from .readparm import CorrectionParameterSet


COULOMB_KCAL_ANG_E2 = 332.05221729
SCEE = 1.2
SCNB = 2.0
_TOL = 1.0e-12

@dataclass(frozen=True)
class MMEnergy:
    bond: float = 0.0
    angle: float = 0.0
    proper: float = 0.0
    improper: float = 0.0
    vdw: float = 0.0
    elec: float = 0.0
    total: float = 0.0


@dataclass
class MMTopologyCache:
    excluded_12: set[tuple[int, int]] = field(default_factory=set)
    excluded_13: set[tuple[int, int]] = field(default_factory=set)
    scaled_14: set[tuple[int, int]] = field(default_factory=set)
    proper_by_center_bond: dict[tuple[int, int], list[Dihedral]] = field(default_factory=dict)


def _pair(i: int, j: int) -> tuple[int, int]:
    return (i, j) if i < j else (j, i)


def _center_bond(dihedral: Dihedral) -> tuple[int, int]:
    return _pair(dihedral.atoms[1], dihedral.atoms[2])


def _total_from_terms(bond: float, angle: float, proper: float, improper: float, vdw: float, elec: float) -> float:
    return bond + angle + proper + improper + vdw + elec


def build_mm_topology_cache(parameter_set: CorrectionParameterSet) -> MMTopologyCache:
    adjacency = parameter_set.mol2.adjacency
    excluded_12: set[tuple[int, int]] = set()
    excluded_13: set[tuple[int, int]] = set()
    scaled_14: set[tuple[int, int]] = set()

    for start in sorted(adjacency):
        distances: dict[int, int] = {start: 0}
        queue: deque[int] = deque([start])

        while queue:
            node = queue.popleft()
            depth = distances[node]
            if depth >= 3:
                continue
            for neighbor in sorted(adjacency[node]):
                if neighbor in distances:
                    continue
                distances[neighbor] = depth + 1
                queue.append(neighbor)

        for atom, distance in distances.items():
            if atom <= start:
                continue
            pair = _pair(start, atom)
            if distance == 1:
                excluded_12.add(pair)
            elif distance == 2:
                excluded_13.add(pair)
            elif distance == 3:
                scaled_14.add(pair)

    proper_by_center_bond: dict[tuple[int, int], list[Dihedral]] = {}
    for dihedral in sorted(parameter_set.dihedrals, key=lambda item: item.atoms):
        proper_by_center_bond.setdefault(_center_bond(dihedral), []).append(dihedral)

    return MMTopologyCache(
        excluded_12=excluded_12,
        excluded_13=excluded_13,
        scaled_14=scaled_14,
        proper_by_center_bond=proper_by_center_bond,
    )


def evaluate_mm_energy(
    atoms: Atoms,
    parameter_set: CorrectionParameterSet,
    topology_cache: MMTopologyCache | None = None,
    zero_proper_center_bond: tuple[int, int] | None = None,
) -> MMEnergy:
    if len(atoms) != len(parameter_set.mol2.atoms):
        raise ValueError(
            f"Atoms length ({len(atoms)}) does not match parameter set size ({len(parameter_set.mol2.atoms)})."
        )

    positions = atoms.get_positions()
    cache = topology_cache if topology_cache is not None else build_mm_topology_cache(parameter_set)
    zero_center = None if zero_proper_center_bond is None else _pair(*zero_proper_center_bond)

    bond_energy = 0.0
    for bond in parameter_set.bonds:
        if bond.kBond is None or bond.rEq is None:
            continue
        r = distance_angstrom(positions, *bond.atoms)
        bond_energy += bond.kBond * (r - bond.rEq) ** 2

    angle_energy = 0.0
    for angle in parameter_set.angles:
        if angle.kTheta is None or angle.thetaEq is None:
            continue
        theta = angle_radians(positions, *angle.atoms)
        angle_energy += angle.kTheta * (theta - angle.thetaEq) ** 2

    proper_energy = 0.0
    for dihedral in parameter_set.dihedrals:
        if zero_center is not None and _center_bond(dihedral) == zero_center:
            continue
        if not dihedral.terms:
            continue
        phi = dihedral_radians(positions, *dihedral.atoms)
        for term in dihedral.terms:
            proper_energy += term.kPhi * (1.0 + cos(term.period * phi - term.phase))

    improper_energy = 0.0
    for improper in parameter_set.impropers:
        if not improper.terms:
            continue
        phi = dihedral_radians(positions, *improper.atoms)
        for term in improper.terms:
            improper_energy += term.kPhi * (1.0 + cos(term.period * phi - term.phase))

    vdw_energy = 0.0
    elec_energy = 0.0
    nonbonds = parameter_set.nonbonds
    for i in range(1, len(nonbonds) + 1):
        atom_i = nonbonds[i - 1]
        for j in range(i + 1, len(nonbonds) + 1):
            pair = _pair(i, j)
            if pair in cache.excluded_12 or pair in cache.excluded_13:
                continue

            atom_j = nonbonds[j - 1]
            r = distance_angstrom(positions, i, j)
            if r <= _TOL:
                raise ValueError(f"Nonbonded distance is too small for pair {pair}: {r}")

            if atom_i.rmin_half is not None and atom_i.epsilon is not None and atom_j.rmin_half is not None and atom_j.epsilon is not None:
                rij = atom_i.rmin_half + atom_j.rmin_half
                epsilon_ij = sqrt(atom_i.epsilon * atom_j.epsilon)
                ratio = rij / r
                vdw = epsilon_ij * (ratio ** 12 - 2.0 * ratio ** 6)
                if pair in cache.scaled_14:
                    vdw /= SCNB
                vdw_energy += vdw

            elec = COULOMB_KCAL_ANG_E2 * atom_i.charge * atom_j.charge / r
            if pair in cache.scaled_14:
                elec /= SCEE
            elec_energy += elec

    total = _total_from_terms(
        bond=bond_energy,
        angle=angle_energy,
        proper=proper_energy,
        improper=improper_energy,
        vdw=vdw_energy,
        elec=elec_energy,
    )
    return MMEnergy(
        bond=bond_energy,
        angle=angle_energy,
        proper=proper_energy,
        improper=improper_energy,
        vdw=vdw_energy,
        elec=elec_energy,
        total=total,
    )
