"""Usage: select torsion centers and apply fitted torsion parameters."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from ..mechanics import build_mm_topology_cache
from ..readparm import CorrectionParameterSet, Dihedral, FourierTerm
from .config import TorsionFitParams, _pair, normalize_center_bond

if TYPE_CHECKING:
    from .records import TorsionFitReport


_ROTATABLE_MOL2_BOND_TYPES = {"1", "1.0", "s", "single"}


def _mol2_bond_type_by_center_bond(parameter_set: CorrectionParameterSet) -> dict[tuple[int, int], str]:
    id_to_index = parameter_set.mol2.id_to_index
    bond_types: dict[tuple[int, int], str] = {}
    for bond in parameter_set.mol2.bonds:
        bond_types[_pair(id_to_index[bond.atom1], id_to_index[bond.atom2])] = str(bond.bond_type)
    return bond_types


def _is_rotatable_mol2_bond_type(bond_type: str | None) -> bool:
    if bond_type is None:
        return False
    return bond_type.strip().lower() in _ROTATABLE_MOL2_BOND_TYPES


def _center_bond_dihedral_indices(
    parameter_set: CorrectionParameterSet,
    center_bond: tuple[int, int],
) -> list[int]:
    center = normalize_center_bond(center_bond)
    return sorted(
        [
            index
            for index, dihedral in enumerate(parameter_set.dihedrals)
            if _pair(dihedral.atoms[1], dihedral.atoms[2]) == center
        ],
        key=lambda index: parameter_set.dihedrals[index].atoms,
    )


def center_bond_dihedrals(
    parameter_set: CorrectionParameterSet,
    center_bond: tuple[int, int],
    topology_cache=None,
) -> list[Dihedral]:
    return [parameter_set.dihedrals[index] for index in _center_bond_dihedral_indices(parameter_set, center_bond)]


def representative_dihedral_for_center_bond(
    parameter_set: CorrectionParameterSet,
    center_bond: tuple[int, int],
    topology_cache=None,
) -> Dihedral:
    dihedrals = center_bond_dihedrals(parameter_set, center_bond, topology_cache=topology_cache)
    if not dihedrals:
        raise ValueError(f"No proper dihedrals found for center bond {normalize_center_bond(center_bond)}.")
    return dihedrals[0]


def center_bond_group_atoms(
    parameter_set: CorrectionParameterSet,
    center_bond: tuple[int, int],
    topology_cache=None,
) -> tuple[int, ...]:
    atoms = {
        atom
        for dihedral in center_bond_dihedrals(parameter_set, center_bond, topology_cache=topology_cache)
        for atom in dihedral.atoms
    }
    return tuple(sorted(atoms))


def is_ring_center_bond(parameter_set: CorrectionParameterSet, center_bond: tuple[int, int]) -> bool:
    start, end = normalize_center_bond(center_bond)
    adjacency = parameter_set.mol2.adjacency
    if end not in adjacency.get(start, set()):
        return False

    seen = {start}
    queue: deque[int] = deque([start])
    while queue:
        node = queue.popleft()
        for neighbor in adjacency.get(node, set()):
            if (node == start and neighbor == end) or (node == end and neighbor == start):
                continue
            if neighbor in seen:
                continue
            if neighbor == end:
                return True
            seen.add(neighbor)
            queue.append(neighbor)
    return False


def enumerate_fittable_center_bonds(
    parameter_set: CorrectionParameterSet,
    topology_cache=None,
    include_ring: bool = False,
) -> list[tuple[int, int]]:
    cache = topology_cache if topology_cache is not None else build_mm_topology_cache(parameter_set)
    bond_types = _mol2_bond_type_by_center_bond(parameter_set)
    center_bonds: list[tuple[int, int]] = []
    for center_bond in sorted(cache.proper_by_center_bond):
        if not cache.proper_by_center_bond[center_bond]:
            continue
        if not _is_rotatable_mol2_bond_type(bond_types.get(center_bond)):
            continue
        if not include_ring and is_ring_center_bond(parameter_set, center_bond):
            continue
        center_bonds.append(center_bond)
    return center_bonds


def resolve_torsion_center_bonds(
    parameter_set: CorrectionParameterSet,
    params: TorsionFitParams,
    topology_cache=None,
) -> tuple[list[tuple[int, int]], list[str]]:
    warnings: list[str] = []
    if params.center_bonds is None:
        cache = topology_cache if topology_cache is not None else build_mm_topology_cache(parameter_set)
        return enumerate_fittable_center_bonds(parameter_set, topology_cache=cache, include_ring=False), warnings

    bond_types = _mol2_bond_type_by_center_bond(parameter_set)
    center_bonds = [normalize_center_bond(bond) for bond in params.center_bonds]
    for center_bond in center_bonds:
        bond_type = bond_types.get(center_bond)
        if not _is_rotatable_mol2_bond_type(bond_type):
            display_type = "missing" if bond_type is None else str(bond_type)
            raise ValueError(f"Explicit torsion center bond {center_bond} is not rotatable: mol2 bond_type={display_type}.")
        if is_ring_center_bond(parameter_set, center_bond):
            warnings.append(
                f"Explicit torsion center bond {center_bond} is ring-internal; running anyway."
            )
    return center_bonds, warnings


def _clone_terms(dihedrals: list[Dihedral]) -> list[list[FourierTerm]]:
    return [[FourierTerm(term.kPhi, term.period, term.phase) for term in dihedral.terms] for dihedral in dihedrals]


def canonical_torsion_atom_types(atom_types: tuple[str, str, str, str]) -> tuple[str, str, str, str]:
    reverse = tuple(reversed(atom_types))
    return atom_types if atom_types <= reverse else reverse


def shared_group_label(atom_types: tuple[str, str, str, str]) -> str:
    return "-".join(atom_types)


def _validate_fit_targets(dihedrals: list[Dihedral]) -> None:
    if not dihedrals:
        raise ValueError("No target dihedrals were selected for torsion fitting.")
    for dihedral in dihedrals:
        if not dihedral.terms:
            raise ValueError(f"Target dihedral {dihedral.atoms} has no torsion terms to fit.")


def apply_center_bond_terms(
    parameter_set: CorrectionParameterSet,
    center_bond: tuple[int, int],
    fitted_terms: list[list[FourierTerm]],
) -> CorrectionParameterSet:
    center = normalize_center_bond(center_bond)
    target_indices = _center_bond_dihedral_indices(parameter_set, center)
    if len(target_indices) != len(fitted_terms):
        raise ValueError(
            "The supplied parameter set does not match the fitted torsion terms for the requested center bond."
        )

    dihedrals = list(parameter_set.dihedrals)
    for fit_index, dihedral_index in enumerate(target_indices):
        dihedral = parameter_set.dihedrals[dihedral_index]
        dihedrals[dihedral_index] = Dihedral(
            atoms=dihedral.atoms,
            atom_types=dihedral.atom_types,
            terms=[FourierTerm(term.kPhi, term.period, term.phase) for term in fitted_terms[fit_index]],
        )

    return CorrectionParameterSet(
        mol2=parameter_set.mol2,
        frcmod=parameter_set.frcmod,
        bonds=parameter_set.bonds,
        angles=parameter_set.angles,
        dihedrals=dihedrals,
        impropers=parameter_set.impropers,
        nonbonds=parameter_set.nonbonds,
        unmatched_bonds=parameter_set.unmatched_bonds,
        unmatched_angles=parameter_set.unmatched_angles,
        unmatched_dihedrals=parameter_set.unmatched_dihedrals,
        unmatched_impropers=parameter_set.unmatched_impropers,
        unmatched_nonbonds=parameter_set.unmatched_nonbonds,
    )
def apply_fitted_torsion(result: "TorsionFitReport", parameter_set: CorrectionParameterSet) -> CorrectionParameterSet:
    return apply_center_bond_terms(parameter_set, result.center_bond, result.terms.fitted_terms)
