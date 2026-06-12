"""Usage: build local and global torsion fitting problems."""

from __future__ import annotations

from collections import OrderedDict, defaultdict

import numpy as np

from ..mechanics import build_mm_topology_cache, dihedral_radians, evaluate_mm_energy
from ..readparm import CorrectionParameterSet
from .topology import (
    _center_bond_dihedral_indices,
    _validate_fit_targets,
    canonical_torsion_atom_types,
    center_bond_dihedrals,
    normalize_center_bond,
)
from .records import TorsionGlobalProblem, TorsionLocalProblem, TorsionScanData, TorsionSharedGroupSpec
from .config import TorsionFitParams

_SLOT_PERIODS = (1, 2, 3, 4)
_SLOT_PHASES = {
    1: 0.0,
    2: np.pi,
    3: 0.0,
    4: np.pi,
    5: 0.0,
    6: np.pi,
}
_LOCAL_EXISTING_TERM_PRIOR_WEIGHT = 6.0
_LOCAL_NEW_TERM_PRIOR_SCALE = 4.0
_LOCAL_NEW_TERM_PRIOR_WEIGHT = _LOCAL_EXISTING_TERM_PRIOR_WEIGHT * _LOCAL_NEW_TERM_PRIOR_SCALE
_GLOBAL_EXISTING_TERM_PRIOR_WEIGHT = _LOCAL_EXISTING_TERM_PRIOR_WEIGHT * 0.25
_GLOBAL_NEW_TERM_PRIOR_WEIGHT = _LOCAL_NEW_TERM_PRIOR_WEIGHT
_GLOBAL_INACTIVE_TERM_PRIOR_WEIGHT = _GLOBAL_NEW_TERM_PRIOR_WEIGHT * 2.0
_PROFILE_WEIGHT_FLOOR = 0.10
_PROFILE_WEIGHT_E0 = 2.0
_PROFILE_SCALE_FLOOR = 0.50


def _robust_profile_range(values: np.ndarray) -> float:
    data = np.asarray(values, dtype=float).reshape(-1)
    if data.size == 0:
        return 0.0
    if data.size < 8:
        return float(np.max(data) - np.min(data))
    return float(np.percentile(data, 95.0) - np.percentile(data, 5.0))


def _profile_fit_scale(qm_rel: np.ndarray, target_like: np.ndarray | None = None) -> float:
    qm_scale = _robust_profile_range(np.asarray(qm_rel, dtype=float))
    target_scale = _robust_profile_range(np.asarray(target_like, dtype=float)) if target_like is not None else 0.0
    return max(qm_scale, target_scale, _PROFILE_SCALE_FLOOR)


def _profile_loss_metrics(
    qm_rel: np.ndarray,
    mm_rel: np.ndarray,
    *,
    profile_scale: float | None = None,
    weights: np.ndarray | None = None,
) -> dict[str, float]:
    qm_values = np.asarray(qm_rel, dtype=float)
    mm_values = np.asarray(mm_rel, dtype=float)
    residual = mm_values - qm_values
    scale = float(profile_scale) if profile_scale is not None else _profile_fit_scale(qm_values)
    weight_values = np.asarray(weights, dtype=float) if weights is not None else _scan_energy_weights(qm_values)
    if weight_values.shape != qm_values.shape:
        weight_values = _scan_energy_weights(qm_values)
    weight_sum = float(np.sum(weight_values))
    normalized_residual = residual / max(scale, 1.0e-12)
    data_loss = float(np.sum(weight_values * (normalized_residual**2)) / weight_sum) if weight_sum > 0.0 else 0.0
    weighted_rmse = float(np.sqrt(data_loss)) if data_loss > 0.0 else 0.0
    return {
        "data_loss": float(data_loss),
        "weighted_rmse": float(weighted_rmse),
        "scale": float(scale),
    }


def _scan_energy_weights(qm_rel: np.ndarray) -> np.ndarray:
    qm_values = np.maximum(np.asarray(qm_rel, dtype=float), 0.0)
    return _PROFILE_WEIGHT_FLOOR + ((1.0 - _PROFILE_WEIGHT_FLOOR) / (1.0 + qm_values / _PROFILE_WEIGHT_E0))


def _normalize_phase_signed(phase: float) -> float:
    value = ((float(phase) + np.pi) % (2.0 * np.pi)) - np.pi
    return float(np.pi) if np.isclose(abs(value), np.pi, atol=1.0e-12) else float(value)


def _default_slot_phase(slot_period: int) -> float:
    return float(_SLOT_PHASES.get(int(slot_period), 0.0 if int(slot_period) % 2 else np.pi))


def _integer_period(period: float) -> int | None:
    value = float(period)
    rounded = int(round(value))
    if rounded <= 0 or abs(value - float(rounded)) > 1.0e-8:
        return None
    return rounded


def _terms_for_period(dihedral, slot_period: int) -> list:
    return [
        term
        for term in dihedral.terms
        if abs(float(term.period) - float(slot_period)) <= 1.0e-8
    ]


def _aggregate_period_coefficients(dihedrals, slot_period: int) -> tuple[float, float] | None:
    coefficients: list[tuple[float, float]] = []
    for dihedral in dihedrals:
        terms = _terms_for_period(dihedral, slot_period)
        if not terms:
            continue
        cos_coeff = sum(float(term.kPhi) * np.cos(float(term.phase)) for term in terms)
        sin_coeff = sum(float(term.kPhi) * np.sin(float(term.phase)) for term in terms)
        coefficients.append((float(cos_coeff), float(sin_coeff)))
    if not coefficients:
        return None
    values = np.asarray(coefficients, dtype=float)
    return float(np.mean(values[:, 0])), float(np.mean(values[:, 1]))


def _coefficients_to_k_phase(coefficients: tuple[float, float] | None, slot_period: int) -> tuple[float, float]:
    if coefficients is None:
        return 0.0, _default_slot_phase(int(slot_period))
    cos_coeff, sin_coeff = coefficients
    k_value = float(np.hypot(cos_coeff, sin_coeff))
    if k_value <= 1.0e-12:
        return 0.0, float(_SLOT_PHASES[int(slot_period)])
    return k_value, _normalize_phase_signed(float(np.arctan2(sin_coeff, cos_coeff)))


def _template_slot_signature(dihedral) -> tuple[float | None, ...]:
    signature: list[float | None] = []
    periods = sorted(set(_SLOT_PERIODS) | {period for period in (_integer_period(term.period) for term in dihedral.terms) if period is not None})
    for slot_period in periods:
        coefficients = _aggregate_period_coefficients((dihedral,), slot_period)
        if coefficients is None:
            signature.append(None)
            continue
        signature.extend((round(coefficients[0], 12), round(coefficients[1], 12)))
    return tuple(signature)


def _member_template_periods(members: list[tuple[int, object]]) -> tuple[int, ...]:
    periods: set[int] = set()
    for _local_index, dihedral in members:
        for term in dihedral.terms:
            period = _integer_period(term.period)
            if period in _SLOT_PERIODS:
                periods.add(period)
    return tuple(sorted(periods))


def _build_group_spec(
    atom_types: tuple[str, str, str, str],
    members: list[tuple[int, object]],
    slot_offset: int,
    *,
    label: str | None = None,
    spectral_slots=(),
) -> tuple[TorsionSharedGroupSpec, int]:
    slot_indices: list[int] = []
    slot_periods: list[int] = []
    slot_phases: list[float] = []
    existing_slot_mask: list[bool] = []
    slot_sources: list[str] = []
    slot_coherences: list[float] = []
    spectral_by_period = {
        int(slot.period): slot
        for slot in spectral_slots
        if int(slot.period) in _SLOT_PERIODS
    }
    candidate_periods = sorted(set(_SLOT_PERIODS) | set(_member_template_periods(members)) | set(spectral_by_period))
    for slot_period in candidate_periods:
        member_dihedrals = [dihedral for _, dihedral in members]
        coefficients = _aggregate_period_coefficients(member_dihedrals, slot_period)
        slot_k, slot_phase = _coefficients_to_k_phase(coefficients, slot_period)
        spectral_slot = spectral_by_period.get(int(slot_period))
        source_parts: list[str] = []
        if coefficients is not None:
            source_parts.append("existing")
        if spectral_slot is not None:
            source_parts.append("spectral")
            if coefficients is None or abs(float(slot_k)) <= 1.0e-10:
                slot_phase = float(spectral_slot.phase_seed)
        slot_indices.append(slot_offset)
        slot_periods.append(slot_period)
        slot_phases.append(slot_phase)
        existing_slot_mask.append(coefficients is not None and abs(float(slot_k)) > 1.0e-10)
        slot_sources.append("+".join(source_parts) if source_parts else "candidate")
        slot_coherences.append(float(getattr(spectral_slot, "coherence", 1.0)) if spectral_slot is not None else 1.0)
        slot_offset += 1

    return (
        TorsionSharedGroupSpec(
            label="-".join(atom_types) if label is None else label,
            atom_types=atom_types,
            improper=False,
            dihedral_indices=tuple(local_index for local_index, _ in members),
            instances=tuple(dihedral.atoms for _, dihedral in members),
            slot_indices=tuple(slot_indices),
            slot_periods=tuple(slot_periods),
            slot_phases=tuple(slot_phases),
            existing_slot_mask=tuple(existing_slot_mask),
            slot_sources=tuple(slot_sources),
            slot_coherences=tuple(slot_coherences),
        ),
        slot_offset,
    )


def _environment_atom_type(atom_type: str) -> str:
    value = str(atom_type).strip().lower()
    return "H" if value.startswith("h") else value


def _bond_type_map(parameter_set: CorrectionParameterSet) -> dict[tuple[int, int], str]:
    return {
        normalize_center_bond((int(bond.atom1), int(bond.atom2))): str(bond.bond_type).strip().lower()
        for bond in parameter_set.mol2.bonds
    }


def _torsion_atom_environment(
    parameter_set: CorrectionParameterSet,
    bond_types: dict[tuple[int, int], str],
    atom_types: dict[int, str],
    atom: int,
    excluded_atoms: set[int],
) -> tuple[tuple[str, str], ...]:
    neighbors = parameter_set.mol2.adjacency.get(int(atom), set())
    environment: list[tuple[str, str]] = []
    for neighbor in neighbors:
        neighbor = int(neighbor)
        if neighbor in excluded_atoms:
            continue
        environment.append(
            (
                _environment_atom_type(atom_types[neighbor]),
                bond_types.get(normalize_center_bond((int(atom), neighbor)), ""),
            )
        )
    return tuple(sorted(environment))


def _directed_torsion_environment_key(
    parameter_set: CorrectionParameterSet,
    bond_types: dict[tuple[int, int], str],
    atom_types: dict[int, str],
    atoms: tuple[int, int, int, int],
) -> tuple[tuple[tuple[str, str], ...], ...]:
    atom_a, atom_b, atom_c, atom_d = (int(atom) for atom in atoms)
    return (
        _torsion_atom_environment(parameter_set, bond_types, atom_types, atom_a, {atom_b}),
        _torsion_atom_environment(parameter_set, bond_types, atom_types, atom_b, {atom_a, atom_c}),
        _torsion_atom_environment(parameter_set, bond_types, atom_types, atom_c, {atom_b, atom_d}),
        _torsion_atom_environment(parameter_set, bond_types, atom_types, atom_d, {atom_c}),
    )


def _torsion_environment_key(
    parameter_set: CorrectionParameterSet,
    bond_types: dict[tuple[int, int], str],
    atom_types: dict[int, str],
    atoms: tuple[int, int, int, int],
) -> tuple[tuple[tuple[str, str], ...], ...]:
    forward = _directed_torsion_environment_key(parameter_set, bond_types, atom_types, atoms)
    reverse = _directed_torsion_environment_key(parameter_set, bond_types, atom_types, tuple(reversed(atoms)))
    return forward if forward <= reverse else reverse


def _group_center_bond_dihedrals(
    dihedrals,
    *,
    parameter_set: CorrectionParameterSet,
    preserve_distinct_template_k: bool = False,
) -> tuple[TorsionSharedGroupSpec, ...]:
    bond_types = _bond_type_map(parameter_set)
    environment_atom_types = {int(mol2_atom.atom_id): str(mol2_atom.atom_type) for mol2_atom in parameter_set.mol2.atoms}
    grouped: "OrderedDict[object, list[tuple[int, object]]]" = OrderedDict()
    for local_index, dihedral in enumerate(dihedrals):
        atom_types = canonical_torsion_atom_types(dihedral.atom_types)
        environment_key = _torsion_environment_key(parameter_set, bond_types, environment_atom_types, dihedral.atoms)
        key = (atom_types, environment_key)
        if preserve_distinct_template_k:
            key = (atom_types, environment_key, _template_slot_signature(dihedral))
        grouped.setdefault(key, []).append((local_index, dihedral))

    base_label_counts: dict[str, int] = defaultdict(int)
    for key in grouped:
        base_label_counts["-".join(key[0])] += 1

    base_label_seen: dict[str, int] = defaultdict(int)
    slot_offset = 0
    group_specs: list[TorsionSharedGroupSpec] = []
    for key, members in grouped.items():
        atom_types = key[0]
        base_label = "-".join(atom_types)
        base_label_seen[base_label] += 1
        label = base_label
        if base_label_counts[base_label] > 1:
            label = f"{base_label} #{base_label_seen[base_label]}"
        group_spec, slot_offset = _build_group_spec(atom_types, members, slot_offset, label=label)
        group_specs.append(group_spec)

    return tuple(group_specs)


def _group_slot_coefficient_basis(
    scan_data: TorsionScanData,
    dihedrals,
    group_specs: tuple[TorsionSharedGroupSpec, ...],
) -> tuple[np.ndarray, np.ndarray]:
    n_points = len(scan_data.frames)
    n_slots = sum(len(group.slot_indices) for group in group_specs)
    cos_basis = np.zeros((n_points, n_slots), dtype=float)
    sin_basis = np.zeros((n_points, n_slots), dtype=float)

    for point_index, atoms in enumerate(scan_data.frames):
        positions = atoms.get_positions()
        phi_cache: dict[int, float] = {}
        for group in group_specs:
            for local_index in group.dihedral_indices:
                if local_index not in phi_cache:
                    phi_cache[local_index] = dihedral_radians(positions, *dihedrals[local_index].atoms)
            for slot_index, slot_period in zip(group.slot_indices, group.slot_periods):
                cos_basis[point_index, slot_index] = sum(
                    np.cos(slot_period * phi_cache[local_index])
                    for local_index in group.dihedral_indices
                )
                sin_basis[point_index, slot_index] = sum(
                    np.sin(slot_period * phi_cache[local_index])
                    for local_index in group.dihedral_indices
                )

    ref_idx = int(scan_data.ref_idx)
    return cos_basis - cos_basis[ref_idx], sin_basis - sin_basis[ref_idx]


def _group_slot_initial_values(dihedrals, group_specs: tuple[TorsionSharedGroupSpec, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_slots = sum(len(group.slot_indices) for group in group_specs)
    k_orig = np.zeros(n_slots, dtype=float)
    scales = np.ones(n_slots, dtype=float)
    prior_weights = np.zeros(n_slots, dtype=float)

    for group in group_specs:
        for slot_index, slot_period, slot_phase, existing in zip(
            group.slot_indices,
            group.slot_periods,
            group.slot_phases,
            group.existing_slot_mask,
        ):
            coefficients = _aggregate_period_coefficients(
                [dihedrals[local_index] for local_index in group.dihedral_indices],
                slot_period,
            )
            slot_k, _slot_phase = _coefficients_to_k_phase(coefficients, slot_period)
            k_orig[slot_index] = float(slot_k)
            scales[slot_index] = max(1.0, abs(k_orig[slot_index]))
            slot_source = str(group.slot_sources[group.slot_indices.index(slot_index)])
            prior_weights[slot_index] = (
                _LOCAL_EXISTING_TERM_PRIOR_WEIGHT
                if "existing" in slot_source
                else _LOCAL_NEW_TERM_PRIOR_WEIGHT
            )

    return k_orig, scales, prior_weights


def _global_basis_matrix(
    scan_data: TorsionScanData,
    parameter_set: CorrectionParameterSet,
    term_paths: tuple[tuple[int, int], ...],
) -> np.ndarray:
    n_points = len(scan_data.frames)
    n_terms = len(term_paths)
    basis = np.zeros((n_points, n_terms), dtype=float)
    dihedrals = parameter_set.dihedrals

    for point_index, atoms in enumerate(scan_data.frames):
        positions = atoms.get_positions()
        phi_cache: dict[int, float] = {}
        for global_index, (dihedral_index, term_index) in enumerate(term_paths):
            dihedral = dihedrals[dihedral_index]
            if dihedral_index not in phi_cache:
                phi_cache[dihedral_index] = dihedral_radians(positions, *dihedral.atoms)
            term = dihedral.terms[term_index]
            basis[point_index, global_index] = 1.0 + np.cos(term.period * phi_cache[dihedral_index] - term.phase)

    return basis


def _center_torsion_relative_profile(scan_data: TorsionScanData, dihedrals) -> np.ndarray:
    torsion_total = np.zeros(len(scan_data.frames), dtype=float)
    for frame_index, atoms in enumerate(scan_data.frames):
        positions = atoms.get_positions()
        for dihedral in dihedrals:
            if not dihedral.terms:
                continue
            phi = dihedral_radians(positions, *dihedral.atoms)
            for term in dihedral.terms:
                torsion_total[frame_index] += float(term.kPhi) * (
                    1.0 + np.cos(float(term.period) * phi - float(term.phase))
                )
    return torsion_total - torsion_total[int(scan_data.ref_idx)]


class _MMProfileCache:
    """Cache fixed-geometry MM profiles for fast torsion refit cycles."""

    def __init__(
        self,
        reference_parameter_set: CorrectionParameterSet,
        center_bonds,
        scan_map: dict[tuple[int, int], TorsionScanData],
        topology_cache=None,
    ) -> None:
        self.center_bonds = tuple(normalize_center_bond(center_bond) for center_bond in center_bonds)
        self.scan_map = {
            normalize_center_bond(center_bond): scan_data
            for center_bond, scan_data in scan_map.items()
        }
        self.dihedral_indices_by_center: dict[tuple[int, int], tuple[int, ...]] = {}
        for center_bond in self.center_bonds:
            indices = tuple(_center_bond_dihedral_indices(reference_parameter_set, center_bond))
            if not indices:
                raise ValueError(f"No proper torsions were found for center bond {center_bond}.")
            self.dihedral_indices_by_center[center_bond] = indices

        tracked_dihedral_indices = tuple(
            sorted({index for indices in self.dihedral_indices_by_center.values() for index in indices})
        )
        cache = topology_cache if topology_cache is not None else build_mm_topology_cache(reference_parameter_set)
        self._ref_idx_by_scan: dict[tuple[int, int], int] = {}
        self._base_total_by_scan: dict[tuple[int, int], np.ndarray] = {}
        self._phi_by_scan: dict[tuple[int, int], dict[int, np.ndarray]] = {}

        for scan_center in self.center_bonds:
            if scan_center not in self.scan_map:
                raise ValueError(f"No fixed scan data was supplied for center bond {scan_center}.")
            scan_data = self.scan_map[scan_center]
            self._ref_idx_by_scan[scan_center] = int(scan_data.ref_idx)
            full_total = np.asarray(
                [evaluate_mm_energy(atoms, reference_parameter_set, topology_cache=cache).total for atoms in scan_data.frames],
                dtype=float,
            )
            phi_by_dihedral: dict[int, np.ndarray] = {}
            for dihedral_index in tracked_dihedral_indices:
                dihedral = reference_parameter_set.dihedrals[int(dihedral_index)]
                phi_by_dihedral[int(dihedral_index)] = np.asarray(
                    [
                        dihedral_radians(atoms.get_positions(), *dihedral.atoms)
                        for atoms in scan_data.frames
                    ],
                    dtype=float,
                )
            self._phi_by_scan[scan_center] = phi_by_dihedral
            fitted_torsion_total = np.zeros(len(scan_data.frames), dtype=float)
            for target_center in self.center_bonds:
                fitted_torsion_total += self._center_torsion_total(reference_parameter_set, scan_center, target_center)
            self._base_total_by_scan[scan_center] = full_total - fitted_torsion_total

    def _relative(self, scan_center: tuple[int, int], total_values: np.ndarray) -> np.ndarray:
        totals = np.asarray(total_values, dtype=float)
        return totals - totals[self._ref_idx_by_scan[scan_center]]

    def _center_torsion_total(
        self,
        parameter_set: CorrectionParameterSet,
        scan_center: tuple[int, int],
        target_center: tuple[int, int],
    ) -> np.ndarray:
        scan_center = normalize_center_bond(scan_center)
        target_center = normalize_center_bond(target_center)
        if scan_center not in self._phi_by_scan:
            raise ValueError(f"No cached scan profile for center bond {scan_center}.")
        if target_center not in self.dihedral_indices_by_center:
            raise ValueError(f"No cached torsion profile for center bond {target_center}.")
        n_points = len(self.scan_map[scan_center].frames)
        torsion_total = np.zeros(n_points, dtype=float)
        phi_by_dihedral = self._phi_by_scan[scan_center]
        for dihedral_index in self.dihedral_indices_by_center[target_center]:
            dihedral = parameter_set.dihedrals[int(dihedral_index)]
            phi_values = phi_by_dihedral[int(dihedral_index)]
            for term in dihedral.terms:
                torsion_total += float(term.kPhi) * (
                    1.0 + np.cos(float(term.period) * phi_values - float(term.phase))
                )
        return torsion_total

    def center_torsion_rel(
        self,
        parameter_set: CorrectionParameterSet,
        scan_center: tuple[int, int],
        target_center: tuple[int, int],
    ) -> np.ndarray:
        scan_center = normalize_center_bond(scan_center)
        return self._relative(scan_center, self._center_torsion_total(parameter_set, scan_center, target_center))

    def full_total(self, parameter_set: CorrectionParameterSet, scan_center: tuple[int, int]) -> np.ndarray:
        scan_center = normalize_center_bond(scan_center)
        if scan_center not in self._base_total_by_scan:
            raise ValueError(f"No cached scan profile for center bond {scan_center}.")
        total = np.asarray(self._base_total_by_scan[scan_center], dtype=float).copy()
        for target_center in self.center_bonds:
            total += self._center_torsion_total(parameter_set, scan_center, target_center)
        return total

    def full_rel(self, parameter_set: CorrectionParameterSet, scan_center: tuple[int, int]) -> np.ndarray:
        scan_center = normalize_center_bond(scan_center)
        return self._relative(scan_center, self.full_total(parameter_set, scan_center))

    def center_zeroed_rel(
        self,
        parameter_set: CorrectionParameterSet,
        scan_center: tuple[int, int],
        target_center: tuple[int, int],
    ) -> np.ndarray:
        scan_center = normalize_center_bond(scan_center)
        zeroed_total = self.full_total(parameter_set, scan_center) - self._center_torsion_total(
            parameter_set,
            scan_center,
            target_center,
        )
        return self._relative(scan_center, zeroed_total)


def build_local_torsion_problem(
    scan_data: TorsionScanData,
    parameter_set: CorrectionParameterSet,
    center_bond: tuple[int, int],
    topology_cache=None,
    *,
    mm_base_rel_override: np.ndarray | None = None,
    stage0_mm_rel_override: np.ndarray | None = None,
) -> TorsionLocalProblem:
    cache = topology_cache if topology_cache is not None else build_mm_topology_cache(parameter_set)
    center = normalize_center_bond(center_bond)
    target_dihedrals = center_bond_dihedrals(parameter_set, center, topology_cache=cache)
    if not target_dihedrals:
        raise ValueError(f"No proper dihedrals found for center bond {center}.")

    _validate_fit_targets(target_dihedrals)
    shared_groups = _group_center_bond_dihedrals(target_dihedrals, parameter_set=parameter_set)
    cos_basis, sin_basis = _group_slot_coefficient_basis(scan_data, target_dihedrals, shared_groups)
    phase_orig = np.asarray([phase for group in shared_groups for phase in group.slot_phases], dtype=float)
    basis = (cos_basis * np.cos(phase_orig)[np.newaxis, :]) + (sin_basis * np.sin(phase_orig)[np.newaxis, :])
    representative_dihedral = target_dihedrals[0].atoms
    qm_rel = np.asarray(scan_data.qm_rel, dtype=float).copy()

    if mm_base_rel_override is None:
        mm_base_total = np.asarray(
            [
                evaluate_mm_energy(
                    atoms,
                    parameter_set,
                    topology_cache=cache,
                    zero_proper_center_bond=center,
                ).total
                for atoms in scan_data.frames
            ],
            dtype=float,
        )
        mm_base_rel = mm_base_total - mm_base_total[int(scan_data.ref_idx)]
    else:
        mm_base_rel = np.asarray(mm_base_rel_override, dtype=float).copy()
        if mm_base_rel.shape != qm_rel.shape:
            raise ValueError(f"MM base relative profile for center bond {center} must match qm_rel shape.")
    if stage0_mm_rel_override is None:
        orig_mm_rel = mm_base_rel + _center_torsion_relative_profile(scan_data, target_dihedrals)
    else:
        orig_mm_rel = np.asarray(stage0_mm_rel_override, dtype=float).copy()
        if orig_mm_rel.shape != qm_rel.shape:
            raise ValueError(f"stage-0 MM relative profile for center bond {center} must match qm_rel shape.")
    fit_target_rel = qm_rel - mm_base_rel

    k_orig, scales, prior_weights = _group_slot_initial_values(target_dihedrals, shared_groups)
    active_mask = np.ones(sum(len(group.slot_indices) for group in shared_groups), dtype=bool)
    return TorsionLocalProblem(
        center_bond=center,
        scan_data=scan_data,
        target_dihedrals=target_dihedrals,
        representative_dihedral=representative_dihedral,
        basis=basis,
        qm_rel=qm_rel,
        orig_mm_rel=orig_mm_rel,
        mm_zeroed_rel=mm_base_rel,
        residual=fit_target_rel,
        k_orig=k_orig,
        scales=scales,
        active_mask=active_mask,
        prior_weights=prior_weights,
        shared_groups=shared_groups,
        phase_orig=phase_orig,
        cos_basis=cos_basis,
        sin_basis=sin_basis,
    )


def build_global_torsion_problem(
    stage0_parameter_set: CorrectionParameterSet,
    center_bonds: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    scan_map: dict[tuple[int, int], TorsionScanData],
    topology_cache=None,
    *,
    typed_shared: bool = False,
    original_parameter_set: CorrectionParameterSet | None = None,
    params: TorsionFitParams | None = None,
    stage_mm_rel_map: dict[tuple[int, int], np.ndarray] | None = None,
) -> TorsionGlobalProblem:
    if typed_shared:
        return _build_grouped_global_torsion_problem(
            stage0_parameter_set,
            center_bonds,
            scan_map,
            topology_cache=topology_cache,
            original_parameter_set=original_parameter_set,
            params=params,
            stage_mm_rel_map=stage_mm_rel_map,
        )
    del topology_cache
    normalized_scan_map = {
        normalize_center_bond(center_bond): scan_data
        for center_bond, scan_data in scan_map.items()
    }
    normalized_center_bonds = tuple(normalize_center_bond(center_bond) for center_bond in center_bonds)
    normalized_stage_mm_rel_map = (
        {
            normalize_center_bond(center_bond): np.asarray(values, dtype=float)
            for center_bond, values in stage_mm_rel_map.items()
        }
        if stage_mm_rel_map is not None
        else {}
    )
    term_paths: list[tuple[int, int]] = []
    block_slices: dict[tuple[int, int], tuple[int, int]] = {}
    offset = 0

    for center_bond in normalized_center_bonds:
        if center_bond not in normalized_scan_map:
            raise ValueError(f"No fixed scan data was supplied for center bond {center_bond}.")
        dihedral_indices = _center_bond_dihedral_indices(stage0_parameter_set, center_bond)
        if not dihedral_indices:
            raise ValueError(f"No proper torsions were found for center bond {center_bond} in the stage-0 parameter set.")

        start = offset
        for dihedral_index in dihedral_indices:
            dihedral = stage0_parameter_set.dihedrals[dihedral_index]
            if not dihedral.terms:
                raise ValueError(f"Target dihedral {dihedral.atoms} has no torsion terms to fit.")
            for term_index in range(len(dihedral.terms)):
                term_paths.append((dihedral_index, term_index))
                offset += 1
        block_slices[center_bond] = (start, offset)

    k_orig = np.asarray(
        [stage0_parameter_set.dihedrals[dihedral_index].terms[term_index].kPhi for dihedral_index, term_index in term_paths],
        dtype=float,
    )
    phase_orig = np.asarray(
        [stage0_parameter_set.dihedrals[dihedral_index].terms[term_index].phase for dihedral_index, term_index in term_paths],
        dtype=float,
    )
    period_orig = np.asarray(
        [stage0_parameter_set.dihedrals[dihedral_index].terms[term_index].period for dihedral_index, term_index in term_paths],
        dtype=float,
    )
    scales = np.maximum(1.0, np.abs(k_orig))
    cache = build_mm_topology_cache(stage0_parameter_set)
    qm_rel_map = {
        center_bond: np.asarray(normalized_scan_map[center_bond].qm_rel, dtype=float).copy()
        for center_bond in normalized_center_bonds
    }
    centered_basis_map: dict[tuple[int, int], np.ndarray] = {}
    centered_cos_basis_map: dict[tuple[int, int], np.ndarray] = {}
    centered_sin_basis_map: dict[tuple[int, int], np.ndarray] = {}
    constant_rel_map: dict[tuple[int, int], np.ndarray] = {}

    for center_bond in normalized_center_bonds:
        scan_data = normalized_scan_map[center_bond]
        base_mm_total = None
        stage_mm_rel = normalized_stage_mm_rel_map.get(center_bond)
        if stage_mm_rel is not None and stage_mm_rel.shape != qm_rel_map[center_bond].shape:
            raise ValueError(f"stage MM relative profile for center bond {center_bond} must match qm_rel shape.")
        if stage_mm_rel is None:
            base_mm_total = np.asarray(
                [evaluate_mm_energy(atoms, stage0_parameter_set, topology_cache=cache).total for atoms in scan_data.frames],
                dtype=float,
            )
        basis = _global_basis_matrix(scan_data, stage0_parameter_set, tuple(term_paths))
        abs_const = np.zeros((len(scan_data.frames), len(term_paths)), dtype=float)
        abs_cos = np.zeros((len(scan_data.frames), len(term_paths)), dtype=float)
        abs_sin = np.zeros((len(scan_data.frames), len(term_paths)), dtype=float)
        start, end = block_slices[center_bond]
        for point_index, atoms in enumerate(scan_data.frames):
            positions = atoms.get_positions()
            phi_cache: dict[int, float] = {}
            for global_index in range(start, end):
                dihedral_index, term_index = term_paths[global_index]
                dihedral = stage0_parameter_set.dihedrals[dihedral_index]
                if dihedral_index not in phi_cache:
                    phi_cache[dihedral_index] = dihedral_radians(positions, *dihedral.atoms)
                phi = phi_cache[dihedral_index]
                period = float(dihedral.terms[term_index].period)
                abs_const[point_index, global_index] = 1.0
                abs_cos[point_index, global_index] = np.cos(period * phi)
                abs_sin[point_index, global_index] = np.sin(period * phi)
        ref_idx = int(scan_data.ref_idx)
        stage1_torsion_total = np.sum(
            k_orig[np.newaxis, :]
            * (
                abs_const
                + (abs_cos * np.cos(phase_orig)[np.newaxis, :])
                + (abs_sin * np.sin(phase_orig)[np.newaxis, :])
            ),
            axis=1,
        )
        centered_basis_map[center_bond] = basis - basis[ref_idx]
        centered_cos_basis_map[center_bond] = abs_cos - abs_cos[ref_idx]
        centered_sin_basis_map[center_bond] = abs_sin - abs_sin[ref_idx]
        stage1_torsion_rel = stage1_torsion_total - stage1_torsion_total[ref_idx]
        if stage_mm_rel is not None:
            constant_rel_map[center_bond] = np.asarray(stage_mm_rel, dtype=float) - stage1_torsion_rel
        else:
            constant_total = np.asarray(base_mm_total, dtype=float) - stage1_torsion_total
            constant_rel_map[center_bond] = constant_total - constant_total[ref_idx]

    return TorsionGlobalProblem(
        stage0_parameter_set=stage0_parameter_set,
        center_bonds=normalized_center_bonds,
        scan_map={center_bond: normalized_scan_map[center_bond] for center_bond in normalized_center_bonds},
        term_paths=tuple(term_paths),
        block_slices=block_slices,
        k_orig=k_orig,
        phase_orig=phase_orig,
        period_orig=period_orig,
        scales=scales,
        qm_rel_map=qm_rel_map,
        centered_basis_map=centered_basis_map,
        centered_cos_basis_map=centered_cos_basis_map,
        centered_sin_basis_map=centered_sin_basis_map,
        constant_rel_map=constant_rel_map,
        reference_parameter_set=stage0_parameter_set,
        prior_weights=np.full_like(scales, 6.0, dtype=float),
        prior_weight=1.0,
        global_max_iter=50,
    )


def _build_grouped_global_torsion_problem(
    reference_parameter_set: CorrectionParameterSet,
    center_bonds: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    scan_map: dict[tuple[int, int], TorsionScanData],
    topology_cache=None,
    *,
    original_parameter_set: CorrectionParameterSet | None = None,
    params: TorsionFitParams | None = None,
    stage_mm_rel_map: dict[tuple[int, int], np.ndarray] | None = None,
) -> TorsionGlobalProblem:
    del topology_cache
    original_parameter_set = original_parameter_set if original_parameter_set is not None else reference_parameter_set
    normalized_scan_map = {
        normalize_center_bond(center_bond): scan_data
        for center_bond, scan_data in scan_map.items()
    }
    normalized_center_bonds = tuple(normalize_center_bond(center_bond) for center_bond in center_bonds)
    normalized_stage_mm_rel_map = (
        {
            normalize_center_bond(center_bond): np.asarray(values, dtype=float)
            for center_bond, values in stage_mm_rel_map.items()
        }
        if stage_mm_rel_map is not None
        else {}
    )
    reference_cache = build_mm_topology_cache(reference_parameter_set)
    group_paths: list[tuple[int, int]] = []
    block_slices: dict[tuple[int, int], tuple[int, int]] = {}
    shared_groups_map: dict[tuple[int, int], tuple[TorsionSharedGroupSpec, ...]] = {}
    offset = 0

    for center_bond in normalized_center_bonds:
        if center_bond not in normalized_scan_map:
            raise ValueError(f"No fixed scan data was supplied for center bond {center_bond}.")
        dihedral_indices = _center_bond_dihedral_indices(reference_parameter_set, center_bond)
        if not dihedral_indices:
            raise ValueError(f"No proper torsions were found for center bond {center_bond} in the stage-1 parameter set.")
        dihedrals = [reference_parameter_set.dihedrals[index] for index in dihedral_indices]
        local_groups = _group_center_bond_dihedrals(
            dihedrals,
            parameter_set=reference_parameter_set,
            preserve_distinct_template_k=True,
        )
        shared_groups: list[TorsionSharedGroupSpec] = []
        start = offset
        for group in local_groups:
            active_slot_indices: list[int] = []
            active_slot_periods: list[int] = []
            active_slot_phases: list[float] = []
            existing_slot_mask: list[bool] = []
            for local_slot_index, slot_period, slot_phase, _ in zip(
                group.slot_indices,
                group.slot_periods,
                group.slot_phases,
                group.existing_slot_mask,
            ):
                slot_present = bool(group.existing_slot_mask[group.slot_indices.index(local_slot_index)])
                active_slot_indices.append(offset)
                active_slot_periods.append(slot_period)
                active_slot_phases.append(slot_phase)
                existing_slot_mask.append(bool(slot_present))
                group_paths.append((dihedral_indices[group.dihedral_indices[0]], local_slot_index))
                offset += 1
            if not active_slot_indices:
                continue
            shared_groups.append(
                TorsionSharedGroupSpec(
                    label=group.label,
                    atom_types=group.atom_types,
                    improper=False,
                    dihedral_indices=tuple(dihedral_indices[index] for index in group.dihedral_indices),
                    instances=group.instances,
                    slot_indices=tuple(active_slot_indices),
                    slot_periods=tuple(active_slot_periods),
                    slot_phases=tuple(active_slot_phases),
                    existing_slot_mask=tuple(existing_slot_mask),
                )
            )
        shared_groups_map[center_bond] = tuple(shared_groups)
        block_slices[center_bond] = (start, offset)

    n_slots = len(group_paths)
    k_orig = np.zeros(n_slots, dtype=float)
    phase_orig = np.zeros(n_slots, dtype=float)
    period_orig = np.zeros(n_slots, dtype=float)
    scales = np.ones(n_slots, dtype=float)
    prior_weights = np.zeros(n_slots, dtype=float)
    qm_rel_map = {
        center_bond: np.asarray(normalized_scan_map[center_bond].qm_rel, dtype=float).copy()
        for center_bond in normalized_center_bonds
    }
    centered_basis_map: dict[tuple[int, int], np.ndarray] = {}
    centered_cos_basis_map: dict[tuple[int, int], np.ndarray] = {}
    centered_sin_basis_map: dict[tuple[int, int], np.ndarray] = {}
    constant_rel_map: dict[tuple[int, int], np.ndarray] = {}

    for center_bond in normalized_center_bonds:
        scan_data = normalized_scan_map[center_bond]
        base_mm_total = None
        stage_mm_rel = normalized_stage_mm_rel_map.get(center_bond)
        if stage_mm_rel is not None and stage_mm_rel.shape != qm_rel_map[center_bond].shape:
            raise ValueError(f"stage MM relative profile for center bond {center_bond} must match qm_rel shape.")
        if stage_mm_rel is None:
            base_mm_total = np.asarray(
                [evaluate_mm_energy(atoms, reference_parameter_set, topology_cache=reference_cache).total for atoms in scan_data.frames],
                dtype=float,
            )

        basis = np.zeros((len(scan_data.frames), n_slots), dtype=float)
        abs_const = np.zeros((len(scan_data.frames), n_slots), dtype=float)
        abs_cos = np.zeros((len(scan_data.frames), n_slots), dtype=float)
        abs_sin = np.zeros((len(scan_data.frames), n_slots), dtype=float)
        center_groups = shared_groups_map[center_bond]
        ref_idx = int(scan_data.ref_idx)
        for point_index, atoms in enumerate(scan_data.frames):
            positions = atoms.get_positions()
            phi_cache: dict[int, float] = {}
            for group in center_groups:
                for dihedral_index in group.dihedral_indices:
                    if dihedral_index not in phi_cache:
                        phi_cache[dihedral_index] = dihedral_radians(positions, *reference_parameter_set.dihedrals[dihedral_index].atoms)
                for slot_index, slot_period, slot_phase, slot_original in zip(
                    group.slot_indices,
                    group.slot_periods,
                    group.slot_phases,
                    group.existing_slot_mask,
                ):
                    abs_const[point_index, slot_index] = float(len(group.dihedral_indices))
                    abs_cos[point_index, slot_index] = sum(
                        np.cos(slot_period * phi_cache[dihedral_index])
                        for dihedral_index in group.dihedral_indices
                    )
                    abs_sin[point_index, slot_index] = sum(
                        np.sin(slot_period * phi_cache[dihedral_index])
                        for dihedral_index in group.dihedral_indices
                    )
                    basis[point_index, slot_index] = (
                        abs_const[point_index, slot_index]
                        + (abs_cos[point_index, slot_index] * np.cos(slot_phase))
                        + (abs_sin[point_index, slot_index] * np.sin(slot_phase))
                    )
                    if point_index == 0:
                        reference_coefficients = _aggregate_period_coefficients(
                            [reference_parameter_set.dihedrals[dihedral_index] for dihedral_index in group.dihedral_indices],
                            slot_period,
                        )
                        original_coefficients = _aggregate_period_coefficients(
                            [original_parameter_set.dihedrals[dihedral_index] for dihedral_index in group.dihedral_indices],
                            slot_period,
                        )
                        slot_k, resolved_phase = _coefficients_to_k_phase(reference_coefficients, slot_period)
                        k_orig[slot_index] = float(slot_k)
                        phase_orig[slot_index] = float(resolved_phase)
                        period_orig[slot_index] = slot_period
                        scales[slot_index] = max(1.0, abs(float(slot_k)))
                        if original_coefficients is not None:
                            prior_weights[slot_index] = _GLOBAL_EXISTING_TERM_PRIOR_WEIGHT
                        elif slot_original:
                            prior_weights[slot_index] = _GLOBAL_NEW_TERM_PRIOR_WEIGHT
                        else:
                            prior_weights[slot_index] = _GLOBAL_INACTIVE_TERM_PRIOR_WEIGHT
        centered_basis_map[center_bond] = basis - basis[ref_idx]
        centered_cos_basis_map[center_bond] = abs_cos - abs_cos[ref_idx]
        centered_sin_basis_map[center_bond] = abs_sin - abs_sin[ref_idx]
        stage1_torsion_total = np.sum(
            k_orig[np.newaxis, :]
            * (
                abs_const
                + (abs_cos * np.cos(phase_orig)[np.newaxis, :])
                + (abs_sin * np.sin(phase_orig)[np.newaxis, :])
            ),
            axis=1,
        )
        stage1_torsion_rel = stage1_torsion_total - stage1_torsion_total[ref_idx]
        if stage_mm_rel is not None:
            constant_rel_map[center_bond] = np.asarray(stage_mm_rel, dtype=float) - stage1_torsion_rel
        else:
            constant_total = np.asarray(base_mm_total, dtype=float) - stage1_torsion_total
            constant_rel_map[center_bond] = constant_total - constant_total[ref_idx]

    return TorsionGlobalProblem(
        stage0_parameter_set=reference_parameter_set,
        center_bonds=normalized_center_bonds,
        scan_map={center_bond: normalized_scan_map[center_bond] for center_bond in normalized_center_bonds},
        term_paths=tuple(group_paths),
        block_slices=block_slices,
        k_orig=k_orig,
        phase_orig=phase_orig,
        period_orig=period_orig,
        scales=scales,
        qm_rel_map=qm_rel_map,
        centered_basis_map=centered_basis_map,
        centered_cos_basis_map=centered_cos_basis_map,
        centered_sin_basis_map=centered_sin_basis_map,
        constant_rel_map=constant_rel_map,
        grouped=True,
        reference_parameter_set=reference_parameter_set,
        prior_weights=prior_weights,
        shared_groups_map=shared_groups_map,
        prior_weight=1.0,
        global_max_iter=50,
    )
