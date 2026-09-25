"""Usage: select torsion centers and apply fitted torsion parameters."""

from __future__ import annotations

from collections import deque
from dataclasses import replace
from typing import TYPE_CHECKING

from ..mechanics import build_mm_topology_cache
from ..readparm import CorrectionParameterSet, Dihedral, FourierTerm, Improper

if TYPE_CHECKING:
    from .config import TorsionFitParams
    from .records import TorsionFitReport




def normalize_torsion_bond(torsion_bond: tuple[int, int]) -> tuple[int, int]:
    i, j = int(torsion_bond[0]), int(torsion_bond[1])
    return (i, j) if i < j else (j, i)


def _bond_type_map(paramset: CorrectionParameterSet) -> dict[tuple[int, int], str]:
    # Antechamber emits canonical types ("1"/"2"/"3"/"ar"/"am"), but the
    # correction route takes the mol2 from the user (OpenBabel / RDKit /
    # hand edits), so values are normalized before any comparison; canonical
    # spellings pass through unchanged.
    id_to_index = paramset.mol2.id_to_index
    return {
        normalize_torsion_bond((id_to_index[bond.atom1], id_to_index[bond.atom2])): str(bond.bond_type).strip().lower()
        for bond in paramset.mol2.bonds
    }


_ROTATABLE_MOL2_BOND_TYPES = {"1", "1.0", "s", "single"}


def _is_rotatable_mol2_bond_type(bond_type: str | None) -> bool:
    if bond_type is None:
        return False
    return bond_type.strip().lower() in _ROTATABLE_MOL2_BOND_TYPES


def _torsion_bond_dihedral_indices(
    paramset: CorrectionParameterSet,
    torsion_bond: tuple[int, int],
) -> list[int]:
    center = normalize_torsion_bond(torsion_bond)
    return sorted(
        [
            index
            for index, dihedral in enumerate(paramset.dihedrals)
            if normalize_torsion_bond((dihedral.atoms[1], dihedral.atoms[2])) == center
        ],
        key=lambda index: paramset.dihedrals[index].atoms,
    )


def torsion_bond_dihedrals(
    paramset: CorrectionParameterSet,
    torsion_bond: tuple[int, int],
    topology_cache=None,
) -> list[Dihedral]:
    return [paramset.dihedrals[index] for index in _torsion_bond_dihedral_indices(paramset, torsion_bond)]


def representative_dihedral_for_torsion_bond(
    paramset: CorrectionParameterSet,
    torsion_bond: tuple[int, int],
    topology_cache=None,
) -> Dihedral:
    dihedrals = torsion_bond_dihedrals(paramset, torsion_bond, topology_cache=topology_cache)
    if not dihedrals:
        raise ValueError(f"No proper dihedrals found for torsion bond {normalize_torsion_bond(torsion_bond)}.")
    return dihedrals[0]


def torsion_bond_group_atoms(
    paramset: CorrectionParameterSet,
    torsion_bond: tuple[int, int],
    topology_cache=None,
) -> tuple[int, ...]:
    atoms = {
        atom
        for dihedral in torsion_bond_dihedrals(paramset, torsion_bond, topology_cache=topology_cache)
        for atom in dihedral.atoms
    }
    return tuple(sorted(atoms))


def is_ring_bond(paramset: CorrectionParameterSet, torsion_bond: tuple[int, int]) -> bool:
    start, end = normalize_torsion_bond(torsion_bond)
    adjacency = paramset.mol2.adjacency
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


def enumerate_fittable_torsion_bonds(
    paramset: CorrectionParameterSet,
    topology_cache=None,
    include_ring: bool = False,
) -> list[tuple[int, int]]:
    cache = topology_cache if topology_cache is not None else build_mm_topology_cache(paramset)
    bond_types = _bond_type_map(paramset)
    torsion_bonds: list[tuple[int, int]] = []
    for torsion_bond in sorted(cache.proper_by_torsion_bond):
        if not cache.proper_by_torsion_bond[torsion_bond]:
            continue
        if not _is_rotatable_mol2_bond_type(bond_types.get(torsion_bond)):
            continue
        if not include_ring and is_ring_bond(paramset, torsion_bond):
            continue
        torsion_bonds.append(torsion_bond)
    return torsion_bonds


def resolve_torsion_bonds(
    paramset: CorrectionParameterSet,
    params: TorsionFitParams,
    topology_cache=None,
) -> tuple[list[tuple[int, int]], list[str]]:
    warnings: list[str] = []
    if params.torsion_bonds is None:
        cache = topology_cache if topology_cache is not None else build_mm_topology_cache(paramset)
        return enumerate_fittable_torsion_bonds(paramset, topology_cache=cache, include_ring=False), warnings

    bond_types = _bond_type_map(paramset)
    torsion_bonds = [normalize_torsion_bond(bond) for bond in params.torsion_bonds]
    for torsion_bond in torsion_bonds:
        bond_type = bond_types.get(torsion_bond)
        if not _is_rotatable_mol2_bond_type(bond_type):
            display_type = "missing" if bond_type is None else str(bond_type)
            raise ValueError(f"Explicit torsion bond {torsion_bond} is not rotatable: mol2 bond_type={display_type}.")
        if is_ring_bond(paramset, torsion_bond):
            warnings.append(
                f"Explicit torsion bond {torsion_bond} is ring-internal; running anyway."
            )
    return torsion_bonds, warnings


def _clone_terms(dihedrals: list[Dihedral]) -> list[list[FourierTerm]]:
    return [[FourierTerm(term.kPhi, term.period, term.phase) for term in dihedral.terms] for dihedral in dihedrals]


def canonical_torsion_atom_types(atom_types: tuple[str, str, str, str]) -> tuple[str, str, str, str]:
    reverse = tuple(reversed(atom_types))
    return atom_types if atom_types <= reverse else reverse


def _validate_fit_targets(dihedrals: list[Dihedral]) -> None:
    if not dihedrals:
        raise ValueError("No target dihedrals were selected for torsion fitting.")
    for dihedral in dihedrals:
        if not dihedral.terms:
            raise ValueError(f"Target dihedral {dihedral.atoms} has no torsion terms to fit.")


def apply_torsion_bond_terms(
    paramset: CorrectionParameterSet,
    torsion_bond: tuple[int, int],
    fitted_terms: list[list[FourierTerm]],
) -> CorrectionParameterSet:
    center = normalize_torsion_bond(torsion_bond)
    target_indices = _torsion_bond_dihedral_indices(paramset, center)
    if len(target_indices) != len(fitted_terms):
        raise ValueError(
            "The supplied parameter set does not match the fitted torsion terms for the requested torsion bond."
        )

    dihedrals = list(paramset.dihedrals)
    for fit_index, dihedral_index in enumerate(target_indices):
        dihedral = paramset.dihedrals[dihedral_index]
        dihedrals[dihedral_index] = Dihedral(
            atoms=dihedral.atoms,
            atom_types=dihedral.atom_types,
            terms=[FourierTerm(term.kPhi, term.period, term.phase) for term in fitted_terms[fit_index]],
        )

    return replace(paramset, dihedrals=dihedrals)


def apply_fitted_torsion(result: "TorsionFitReport", paramset: CorrectionParameterSet) -> CorrectionParameterSet:
    return apply_torsion_bond_terms(paramset, result.torsion_bond, result.terms.fitted_terms)

def apply_fitted_improper(result: "TorsionFitReport", paramset: CorrectionParameterSet) -> CorrectionParameterSet:
    """Replace terms on the center's matched instances; append the transient target when none exist."""
    target = result.target_instances[0]
    center = target.atoms[2]
    fitted_terms = [FourierTerm(term.kPhi, term.period, term.phase) for term in result.terms.fitted_terms[0]]
    impropers = list(paramset.impropers)
    replaced = False
    for index, improper in enumerate(impropers):
        if improper.atoms[2] != center:
            continue
        impropers[index] = Improper(
            atoms=improper.atoms,
            atom_types=improper.atom_types,
            terms=[FourierTerm(term.kPhi, term.period, term.phase) for term in fitted_terms],
            refit=True,
        )
        replaced = True
    if not replaced:
        impropers.append(
            Improper(atoms=target.atoms, atom_types=target.atom_types, terms=fitted_terms, refit=True)
        )
    return replace(paramset, impropers=impropers)
