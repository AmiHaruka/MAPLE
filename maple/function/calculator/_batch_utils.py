"""Shared helpers for calculator-native batch evaluation.

The public contract stays in :mod:`calculator_base` / ``BatchResult``.  This
module only holds small mechanics that are useful across backend-specific
``calculate_many`` implementations:

* normalize an energy/forces property request;
* keep the sequential fallback result-driven and cache-safe;
* split atom-wise tensors back into ASE-ordered per-structure arrays.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import copy
import operator
from typing import Iterable, Sequence

import numpy as np
from ase.calculators.calculator import all_changes

from ._batch_types import BatchResult


def atoms_has_pbc(atoms) -> bool:
    """Return True when an ASE Atoms object has any periodic axis enabled."""
    if atoms is None:
        return False
    return bool(np.any(getattr(atoms, "pbc", False)))


def atoms_list_has_pbc(atoms_list: Sequence) -> bool:
    """Return True when any structure in a candidate batch is periodic."""
    return any(atoms_has_pbc(at) for at in atoms_list)


def normalize_energy_forces_request(properties) -> tuple[tuple[str, ...], bool, bool, list[str]]:
    """Return ``(props, want_energy, want_forces, request)``.

    ``calculate_many`` is intentionally limited to energy/forces.  Numerical
    Hessian assembly belongs to ``FDHessianEvaluator``; analytic batched
    Hessians should be an explicit backend override rather than an accidental
    property side effect.
    """
    if properties is None:
        props = ("energy",)
    elif isinstance(properties, str):
        props = (properties,)
    else:
        props = tuple(properties)
    if "hessian" in props:
        raise NotImplementedError(
            "calculate_many does not assemble Hessians; use FDHessianEvaluator "
            "for numerical Hessians or a backend-specific analytic Hessian path."
        )
    unknown = [repr(prop) for prop in props if prop not in {"energy", "forces"}]
    if unknown:
        raise ValueError(
            "Unsupported calculate_many properties: " + ", ".join(unknown)
        )

    want_energy = "energy" in props
    want_forces = "forces" in props
    request = list(dict.fromkeys(props))
    return props, want_energy, want_forces, request


def empty_batch_result(want_energy: bool, want_forces: bool) -> BatchResult:
    return BatchResult(
        energies=np.zeros(0, dtype=np.float64) if want_energy else None,
        forces=[] if want_forces else None,
    )


@contextmanager
def preserve_calculator_state(calc):
    """Restore ASE's single-structure cache after temporary evaluations.

    The batch API is return-value driven, so neither successful sequential
    fallback nor an exception may leave ``calc.atoms``/``calc.results`` pointing
    at one of the temporary structures.
    """
    missing = object()
    original_atoms = getattr(calc, "atoms", missing)
    original_results = getattr(calc, "results", missing)
    if original_results is not missing and hasattr(original_results, "items"):
        try:
            saved_results = copy.deepcopy(dict(original_results))
        except Exception:
            saved_results = dict(original_results)
    else:
        saved_results = original_results
    try:
        yield
    finally:
        if original_atoms is missing:
            try:
                delattr(calc, "atoms")
            except AttributeError:
                pass
        else:
            calc.atoms = original_atoms

        if original_results is missing:
            try:
                delattr(calc, "results")
            except AttributeError:
                pass
        else:
            if hasattr(original_results, "clear") and hasattr(
                original_results, "update"
            ):
                original_results.clear()
                original_results.update(saved_results)
            calc.results = original_results


def sequential_calculate_many(
    calc,
    atoms_list: Sequence,
    request: Sequence[str],
    want_energy: bool,
    want_forces: bool,
) -> BatchResult:
    """Sequential fallback matching ``CalcABC.calculate_many`` semantics."""
    atoms_list = list(atoms_list)
    energies = [] if want_energy else None
    forces_list = [] if want_forces else None

    with preserve_calculator_state(calc):
        for at in atoms_list:
            calc.calculate(at, properties=list(request), system_changes=all_changes)
            if want_energy:
                if "free_energy" in calc.results:
                    energies.append(float(calc.results["free_energy"]))
                else:
                    energies.append(float(calc.results["energy"]))
            if want_forces:
                forces_list.append(
                    np.array(calc.results["forces"], dtype=np.float64, copy=True)
                )

    return BatchResult(
        energies=np.asarray(energies, dtype=np.float64) if energies is not None else None,
        forces=forces_list,
    ).validate_against(atoms_list, request)


def atom_counts(atoms_list: Sequence) -> list[int]:
    return [len(at) for at in atoms_list]


def split_atomwise_array(values: np.ndarray, counts: Sequence[int]) -> list[np.ndarray]:
    """Split a flat ``(sum(N_i), 3)`` array into per-Atoms force arrays."""
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(
            "Atom-wise values must have shape (sum(counts), 3), "
            f"got {values.shape}"
        )

    normalized_counts = []
    for count in counts:
        if isinstance(count, bool):
            raise ValueError("Atom counts must be non-negative integers")
        try:
            normalized = operator.index(count)
        except TypeError as exc:
            raise ValueError(
                "Atom counts must be non-negative integers"
            ) from exc
        if normalized < 0:
            raise ValueError("Atom counts must be non-negative integers")
        normalized_counts.append(normalized)

    expected_rows = sum(normalized_counts)
    if values.shape[0] != expected_rows:
        raise ValueError(
            "Atom-wise array row count must equal sum(counts): "
            f"got {values.shape[0]} rows, sum(counts)={expected_rows}"
        )

    out = []
    start = 0
    for n_atoms in normalized_counts:
        stop = start + n_atoms
        out.append(np.array(values[start:stop], dtype=np.float64, copy=True))
        start = stop
    return out


def grouped_indices_by_numbers(atoms_list: Sequence) -> Iterable[tuple[tuple[int, ...], list[int]]]:
    """Yield original indices grouped by identical atomic-number sequence."""
    groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for i, at in enumerate(atoms_list):
        groups[tuple(int(z) for z in at.get_atomic_numbers())].append(i)
    return groups.items()
