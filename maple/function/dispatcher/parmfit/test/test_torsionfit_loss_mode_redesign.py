from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

try:
    from ase import Atoms
except ModuleNotFoundError:
    ase_stub = types.ModuleType("ase")
    constraints_stub = types.ModuleType("ase.constraints")
    neighborlist_stub = types.ModuleType("ase.neighborlist")

    class Atoms:  # type: ignore[override]
        def __init__(self, symbols, positions):
            self.symbols = list(symbols)
            self.positions = np.asarray(positions, dtype=float)

        def __len__(self):
            return len(self.symbols)

        def copy(self):
            return Atoms(self.symbols[:], self.positions.copy())

        def get_positions(self):
            return np.asarray(self.positions, dtype=float)

        def get_chemical_symbols(self):
            return self.symbols[:]

    class FixInternals:  # pragma: no cover - import stub only
        def __init__(self, *args, **kwargs):
            pass

    class NeighborList:  # pragma: no cover - import stub only
        def __init__(self, *args, **kwargs):
            pass

        def update(self, atoms):
            return None

        def get_neighbors(self, index):
            return np.asarray([], dtype=int), np.asarray([], dtype=int)

    def natural_cutoffs(atoms):
        return [1.0] * len(atoms)

    ase_stub.Atoms = Atoms
    constraints_stub.FixInternals = FixInternals
    neighborlist_stub.NeighborList = NeighborList
    neighborlist_stub.natural_cutoffs = natural_cutoffs
    sys.modules["ase"] = ase_stub
    sys.modules["ase.constraints"] = constraints_stub
    sys.modules["ase.neighborlist"] = neighborlist_stub
else:
    ase_module = sys.modules.get("ase")
    if ase_module is not None and not hasattr(ase_module, "__path__"):
        ase_module.__path__ = []  # type: ignore[attr-defined]
    if "ase.constraints" not in sys.modules:
        constraints_stub = types.ModuleType("ase.constraints")

        class FixInternals:  # pragma: no cover - import stub only
            def __init__(self, *args, **kwargs):
                pass

        constraints_stub.FixInternals = FixInternals
        sys.modules["ase.constraints"] = constraints_stub
        if ase_module is not None:
            ase_module.constraints = constraints_stub
    if "ase.neighborlist" not in sys.modules:
        neighborlist_stub = types.ModuleType("ase.neighborlist")

        class NeighborList:  # pragma: no cover - import stub only
            def __init__(self, *args, **kwargs):
                pass

            def update(self, atoms):
                return None

            def get_neighbors(self, index):
                return np.asarray([], dtype=int), np.asarray([], dtype=int)

        def natural_cutoffs(atoms):
            return [1.0] * len(atoms)

        neighborlist_stub.NeighborList = NeighborList
        neighborlist_stub.natural_cutoffs = natural_cutoffs
        sys.modules["ase.neighborlist"] = neighborlist_stub
        if ase_module is not None:
            ase_module.neighborlist = neighborlist_stub

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maple.function.dispatcher.parmfit.utils.TorsionFit import format_torsion_fit_report
from maple.function.dispatcher.parmfit.utils.TorsionFit.topology import apply_fitted_torsion
from maple.function.dispatcher.parmfit.utils.TorsionFit import stage1 as torsion_stage1_module
from maple.function.dispatcher.parmfit.utils.TorsionFit import stage2 as torsion_stage2_module
from maple.function.dispatcher.parmfit.utils.TorsionFit.fit import fit_torsion_scan
from maple.function.dispatcher.parmfit.utils.TorsionFit.basis import build_global_torsion_problem
from maple.function.dispatcher.parmfit.utils.TorsionFit.quality import _scan_energy_weights
from maple.function.dispatcher.parmfit.utils.TorsionFit.stage1 import (
    _build_local_stage1_solve_cache,
    _solve_local_problem_stage1,
    _solve_local_stage1_active_set,
    _solve_local_stage1_active_set_with_phases,
    _stage1_retained_rows,
)
from maple.function.dispatcher.parmfit.utils.TorsionFit.spectral import (
    DEFAULT_SPECTRAL_PERIODS,
    dominant_spectral_peaks,
    shared_group_response,
)
from maple.function.dispatcher.parmfit.utils.TorsionFit import ensemble as torsion_ensemble_module
from maple.function.dispatcher.parmfit.utils.TorsionFit.stage2 import (
    _build_stage2_objective_cache,
    _delta_from_stage2_k_phase,
    _evaluate_stage2_k_phase_objective_with_gradient,
    _global_mm_rel_map,
    _optimize_stage2_k_phase,
    _pack_stage2_k_phase,
    _stage2_k_phase_bounds,
    _split_global_vector,
    apply_global_delta,
    evaluate_global_refit_objective,
    refine_torsion_scans_global,
)
from maple.function.dispatcher.parmfit.utils.TorsionFit.fit import _stage2_fit_report_from_stage1
from maple.function.dispatcher.parmfit.utils.TorsionFit.fit import run_loss_mode
from maple.function.dispatcher.parmfit.utils.TorsionFit.ensemble import build_stage2_extra_targets
from maple.function.dispatcher.parmfit.utils.TorsionFit.records import (
    TorsionEnsembleResult,
    TorsionFitCurves,
    TorsionFitMetrics,
    TorsionFitReport,
    TorsionFitTerms,
    TorsionGlobalProblem,
    TorsionLocalProblem,
    TorsionObjectiveTarget,
    TorsionRefineCycle,
    TorsionScanData,
    TorsionSharedGroupSpec,
    TorsionWorkflowResult,
)
from maple.function.dispatcher.parmfit.utils.TorsionFit.config import TorsionFitParams, normalize_center_bond
from maple.function.dispatcher.parmfit.utils.TorsionFit import basis as torsion_problem_module
from maple.function.dispatcher.parmfit.utils.TorsionFit.scanio import HARTREE_TO_KCAL_MOL, read_scan_xyz
from maple.function.dispatcher.parmfit.utils.TorsionFit.workflow import run_torsion_workflow
from maple.function.dispatcher.parmfit.utils.mechanics import MMEnergy, dihedral_radians
from maple.function.dispatcher.parmfit.utils.readparm import (
    CorrectionParameterSet,
    Dihedral,
    FourierTerm,
    FrcmodDB,
    Mol2Atom,
    Mol2Bond,
    Mol2Topology,
    Nonbond,
)
from maple.function.dispatcher.parmfit.utils.TorsionFit.records import TorsionScanRuntime


def _four_atom_frame(phi_deg: float) -> Atoms:
    rad = math.radians(phi_deg)
    positions = [
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        (math.cos(rad), math.sin(rad), 1.0),
    ]
    return Atoms(symbols=["C", "C", "C", "C"], positions=positions)


def _two_fragment_frame(phi_a_deg: float, phi_b_deg: float) -> Atoms:
    frag_a = _four_atom_frame(phi_a_deg)
    frag_b = _four_atom_frame(phi_b_deg)
    shifted_b = np.asarray(frag_b.get_positions(), dtype=float).copy()
    shifted_b[:, 0] += 8.0
    return Atoms(
        symbols=["C"] * 8,
        positions=np.vstack((frag_a.get_positions(), shifted_b)),
    )


def _make_parameter_set(k_values: tuple[float, ...]) -> CorrectionParameterSet:
    mol2 = Mol2Topology(
        atoms=[Mol2Atom(atom_id=i, name=f"A{i}", atom_type="c", charge=0.0) for i in range(1, 5)],
        bonds=[
            Mol2Bond(bond_id=1, atom1=1, atom2=2, bond_type="1"),
            Mol2Bond(bond_id=2, atom1=2, atom2=3, bond_type="1"),
            Mol2Bond(bond_id=3, atom1=3, atom2=4, bond_type="1"),
        ],
        id_to_index={i: i for i in range(1, 5)},
        adjacency={1: {2}, 2: {1, 3}, 3: {2, 4}, 4: {3}},
    )
    dihedral = Dihedral(
        atoms=(1, 2, 3, 4),
        atom_types=("c", "c", "c", "c"),
        terms=[
            FourierTerm(kPhi=float(k_value), period=float(index), phase=0.0 if index % 2 == 1 else np.pi)
            for index, k_value in enumerate(k_values, start=1)
        ],
    )
    return CorrectionParameterSet(
        mol2=mol2,
        frcmod=FrcmodDB(),
        bonds=[],
        angles=[],
        dihedrals=[dihedral],
        impropers=[],
        nonbonds=[Nonbond(atom=i, atom_type="c", charge=0.0, rmin_half=1.5, epsilon=0.0) for i in range(1, 5)],
        unmatched_bonds=[],
        unmatched_angles=[],
        unmatched_dihedrals=[],
        unmatched_impropers=[],
        unmatched_nonbonds=[],
    )


def _make_two_center_parameter_set() -> CorrectionParameterSet:
    mol2 = Mol2Topology(
        atoms=[Mol2Atom(atom_id=i, name=f"A{i}", atom_type="c", charge=0.0) for i in range(1, 9)],
        bonds=[
            Mol2Bond(bond_id=1, atom1=1, atom2=2, bond_type="1"),
            Mol2Bond(bond_id=2, atom1=2, atom2=3, bond_type="1"),
            Mol2Bond(bond_id=3, atom1=3, atom2=4, bond_type="1"),
            Mol2Bond(bond_id=4, atom1=5, atom2=6, bond_type="1"),
            Mol2Bond(bond_id=5, atom1=6, atom2=7, bond_type="1"),
            Mol2Bond(bond_id=6, atom1=7, atom2=8, bond_type="1"),
        ],
        id_to_index={i: i for i in range(1, 9)},
        adjacency={
            1: {2},
            2: {1, 3},
            3: {2, 4},
            4: {3},
            5: {6},
            6: {5, 7},
            7: {6, 8},
            8: {7},
        },
    )
    dihedrals = [
        Dihedral(
            atoms=(1, 2, 3, 4),
            atom_types=("c", "c", "c", "c"),
            terms=[FourierTerm(kPhi=0.40, period=1.0, phase=0.0)],
        ),
        Dihedral(
            atoms=(5, 6, 7, 8),
            atom_types=("c", "c", "c", "c"),
            terms=[FourierTerm(kPhi=0.60, period=2.0, phase=np.pi)],
        ),
    ]
    return CorrectionParameterSet(
        mol2=mol2,
        frcmod=FrcmodDB(),
        bonds=[],
        angles=[],
        dihedrals=dihedrals,
        impropers=[],
        nonbonds=[Nonbond(atom=i, atom_type="c", charge=0.0, rmin_half=1.5, epsilon=0.0) for i in range(1, 9)],
        unmatched_bonds=[],
        unmatched_angles=[],
        unmatched_dihedrals=[],
        unmatched_impropers=[],
        unmatched_nonbonds=[],
    )


def _make_terminal_h_first_rotor_parameter_set() -> CorrectionParameterSet:
    mol2 = Mol2Topology(
        atoms=[
            Mol2Atom(atom_id=1, name="C1", atom_type="c", charge=0.0),
            Mol2Atom(atom_id=2, name="H1", atom_type="h1", charge=0.0),
            Mol2Atom(atom_id=3, name="H2", atom_type="h1", charge=0.0),
            Mol2Atom(atom_id=4, name="H3", atom_type="h1", charge=0.0),
            Mol2Atom(atom_id=5, name="C2", atom_type="c", charge=0.0),
            Mol2Atom(atom_id=6, name="H4", atom_type="h1", charge=0.0),
            Mol2Atom(atom_id=7, name="H5", atom_type="h1", charge=0.0),
            Mol2Atom(atom_id=8, name="O1", atom_type="oh", charge=0.0),
        ],
        bonds=[
            Mol2Bond(bond_id=1, atom1=1, atom2=2, bond_type="1"),
            Mol2Bond(bond_id=2, atom1=1, atom2=3, bond_type="1"),
            Mol2Bond(bond_id=3, atom1=1, atom2=4, bond_type="1"),
            Mol2Bond(bond_id=4, atom1=1, atom2=5, bond_type="1"),
            Mol2Bond(bond_id=5, atom1=5, atom2=6, bond_type="1"),
            Mol2Bond(bond_id=6, atom1=5, atom2=7, bond_type="1"),
            Mol2Bond(bond_id=7, atom1=5, atom2=8, bond_type="1"),
        ],
        id_to_index={i: i for i in range(1, 9)},
        adjacency={
            1: {2, 3, 4, 5},
            2: {1},
            3: {1},
            4: {1},
            5: {1, 6, 7, 8},
            6: {5},
            7: {5},
            8: {5},
        },
    )
    dihedrals = [
        Dihedral(
            atoms=(2, 1, 5, 6),
            atom_types=("h1", "c", "c", "h1"),
            terms=[FourierTerm(kPhi=0.25, period=1.0, phase=0.0)],
        ),
        Dihedral(
            atoms=(2, 1, 5, 8),
            atom_types=("h1", "c", "c", "oh"),
            terms=[FourierTerm(kPhi=0.25, period=1.0, phase=0.0)],
        ),
    ]
    return CorrectionParameterSet(
        mol2=mol2,
        frcmod=FrcmodDB(),
        bonds=[],
        angles=[],
        dihedrals=dihedrals,
        impropers=[],
        nonbonds=[Nonbond(atom=i, atom_type="c", charge=0.0, rmin_half=1.5, epsilon=0.0) for i in range(1, 9)],
        unmatched_bonds=[],
        unmatched_angles=[],
        unmatched_dihedrals=[],
        unmatched_impropers=[],
        unmatched_nonbonds=[],
    )


def _make_terminal_cl_rotor_parameter_set() -> CorrectionParameterSet:
    mol2 = Mol2Topology(
        atoms=[
            Mol2Atom(atom_id=1, name="C1", atom_type="c", charge=0.0),
            Mol2Atom(atom_id=2, name="C2", atom_type="c", charge=0.0),
            Mol2Atom(atom_id=3, name="Cl1", atom_type="cl", charge=0.0),
            Mol2Atom(atom_id=4, name="C3", atom_type="c", charge=0.0),
        ],
        bonds=[
            Mol2Bond(bond_id=1, atom1=1, atom2=2, bond_type="1"),
            Mol2Bond(bond_id=2, atom1=2, atom2=3, bond_type="1"),
            Mol2Bond(bond_id=3, atom1=1, atom2=4, bond_type="1"),
        ],
        id_to_index={i: i for i in range(1, 5)},
        adjacency={1: {2, 4}, 2: {1, 3}, 3: {2}, 4: {1}},
    )
    return CorrectionParameterSet(
        mol2=mol2,
        frcmod=FrcmodDB(),
        bonds=[],
        angles=[],
        dihedrals=[
            Dihedral(
                atoms=(1, 2, 3, 1),
                atom_types=("c", "c", "c", "cl"),
                terms=[FourierTerm(kPhi=0.30, period=1.0, phase=0.0)],
            )
        ],
        impropers=[],
        nonbonds=[Nonbond(atom=i, atom_type="c", charge=0.0, rmin_half=1.5, epsilon=0.0) for i in range(1, 5)],
        unmatched_bonds=[],
        unmatched_angles=[],
        unmatched_dihedrals=[],
        unmatched_impropers=[],
        unmatched_nonbonds=[],
    )


def _make_ring_center_parameter_set() -> CorrectionParameterSet:
    mol2 = Mol2Topology(
        atoms=[Mol2Atom(atom_id=i, name=f"C{i}", atom_type="c", charge=0.0) for i in range(1, 5)],
        bonds=[
            Mol2Bond(bond_id=1, atom1=1, atom2=2, bond_type="1"),
            Mol2Bond(bond_id=2, atom1=2, atom2=3, bond_type="1"),
            Mol2Bond(bond_id=3, atom1=3, atom2=4, bond_type="1"),
            Mol2Bond(bond_id=4, atom1=4, atom2=1, bond_type="1"),
        ],
        id_to_index={i: i for i in range(1, 5)},
        adjacency={1: {2, 4}, 2: {1, 3}, 3: {2, 4}, 4: {1, 3}},
    )
    return CorrectionParameterSet(
        mol2=mol2,
        frcmod=FrcmodDB(),
        bonds=[],
        angles=[],
        dihedrals=[
            Dihedral(
                atoms=(1, 2, 3, 4),
                atom_types=("c", "c", "c", "c"),
                terms=[FourierTerm(kPhi=0.30, period=1.0, phase=0.0)],
            )
        ],
        impropers=[],
        nonbonds=[Nonbond(atom=i, atom_type="c", charge=0.0, rmin_half=1.5, epsilon=0.0) for i in range(1, 5)],
        unmatched_bonds=[],
        unmatched_angles=[],
        unmatched_dihedrals=[],
        unmatched_impropers=[],
        unmatched_nonbonds=[],
    )


def _two_center_scan(center: tuple[int, int]) -> TorsionScanData:
    angles = np.asarray([0.0, 45.0, 90.0, 135.0], dtype=float)
    if center == (2, 3):
        frames = [_two_fragment_frame(float(angle), 20.0) for angle in angles]
    else:
        frames = [_two_fragment_frame(float(angle) * 0.5, float(angle)) for angle in angles]
    qm_kcal = np.zeros(len(frames), dtype=float)
    return TorsionScanData(
        angles_deg=angles,
        qm_hartree=qm_kcal,
        qm_kcal=qm_kcal,
        frames=frames,
        source_path=f"two_center_{center[0]}-{center[1]}.xyz",
        ref_idx=0,
        qm_rel=qm_kcal.copy(),
    )


def _make_sparse_parameter_set(term_specs: list[tuple[float, float, float]]) -> CorrectionParameterSet:
    mol2 = Mol2Topology(
        atoms=[Mol2Atom(atom_id=i, name=f"A{i}", atom_type="c", charge=0.0) for i in range(1, 5)],
        bonds=[
            Mol2Bond(bond_id=1, atom1=1, atom2=2, bond_type="1"),
            Mol2Bond(bond_id=2, atom1=2, atom2=3, bond_type="1"),
            Mol2Bond(bond_id=3, atom1=3, atom2=4, bond_type="1"),
        ],
        id_to_index={i: i for i in range(1, 5)},
        adjacency={1: {2}, 2: {1, 3}, 3: {2, 4}, 4: {3}},
    )
    dihedral = Dihedral(
        atoms=(1, 2, 3, 4),
        atom_types=("c", "c", "c", "c"),
        terms=[
            FourierTerm(kPhi=float(k_value), period=float(period), phase=float(phase))
            for k_value, period, phase in term_specs
        ],
    )
    return CorrectionParameterSet(
        mol2=mol2,
        frcmod=FrcmodDB(),
        bonds=[],
        angles=[],
        dihedrals=[dihedral],
        impropers=[],
        nonbonds=[Nonbond(atom=i, atom_type="c", charge=0.0, rmin_half=1.5, epsilon=0.0) for i in range(1, 5)],
        unmatched_bonds=[],
        unmatched_angles=[],
        unmatched_dihedrals=[],
        unmatched_impropers=[],
        unmatched_nonbonds=[],
    )


def test_stage1_preserves_inactive_zero_amplitude_template_terms():
    original_terms = [FourierTerm(kPhi=0.0, period=2.0, phase=np.pi)]

    merged = torsion_stage1_module._merge_template_and_frozen_terms(original_terms, {})

    assert len(merged) == 1
    assert merged[0].kPhi == pytest.approx(0.0)
    assert merged[0].period == pytest.approx(2.0)
    assert merged[0].phase == pytest.approx(np.pi)


def test_spectral_selector_uses_amber_compatible_periods_and_keeps_phase():
    phi_values = np.linspace(0.0, 2.0 * np.pi, 72, endpoint=False)
    phase = 0.73
    target = 1.4 * (np.cos(6.0 * phi_values - phase) - np.cos(6.0 * phi_values[0] - phase))

    peaks = dominant_spectral_peaks(phi_values, target, ref_idx=0, allowed_periods=DEFAULT_SPECTRAL_PERIODS)

    periods = [peak.period for peak in peaks]
    assert 6 in periods
    assert 5 not in periods
    period6 = next(peak for peak in peaks if peak.period == 6)
    assert period6.phase == pytest.approx(phase, abs=1.0e-10)


def test_shared_group_response_migrates_path_phase_and_detects_cancellation():
    phi_rep = np.linspace(-np.pi, np.pi, 24, endpoint=False)
    shifted_paths = np.column_stack((phi_rep + 0.40, phi_rep + 0.40))

    response = shared_group_response(shifted_paths, phi_rep, period=2)

    assert response.coherence == pytest.approx(1.0, abs=1.0e-12)
    assert response.phase_shift == pytest.approx(0.80, abs=1.0e-12)

    cancelling_paths = np.column_stack((phi_rep, phi_rep + np.pi / 2.0))
    cancelled = shared_group_response(cancelling_paths, phi_rep, period=2)

    assert cancelled.coherence < 1.0e-10


def test_mm_profile_cache_matches_direct_zeroed_mm_for_multiple_centers(monkeypatch):
    monkeypatch.setattr(torsion_problem_module, "evaluate_mm_energy", _fake_mm_energy)
    parameter_set = _make_two_center_parameter_set()
    scan_map = {
        (2, 3): _two_center_scan((2, 3)),
        (6, 7): _two_center_scan((6, 7)),
    }

    cache = torsion_problem_module._MMProfileCache(parameter_set, [(2, 3), (6, 7)], scan_map)

    updated = deepcopy(parameter_set)
    updated.dihedrals[0].terms = [FourierTerm(kPhi=1.15, period=1.0, phase=0.35)]
    updated.dihedrals[1].terms = [FourierTerm(kPhi=0.25, period=2.0, phase=np.pi)]

    direct = np.asarray(
        [
            _fake_mm_energy(atoms, updated, zero_proper_center_bond=(6, 7)).total
            for atoms in scan_map[(6, 7)].frames
        ],
        dtype=float,
    )
    direct_rel = direct - direct[int(scan_map[(6, 7)].ref_idx)]

    assert cache.center_zeroed_rel(updated, (6, 7), (6, 7)).tolist() == pytest.approx(direct_rel.tolist(), abs=1.0e-8)


def test_mm_profile_cache_uses_cached_phi_for_new_period_terms(monkeypatch):
    monkeypatch.setattr(torsion_problem_module, "evaluate_mm_energy", _fake_mm_energy)
    parameter_set = _make_two_center_parameter_set()
    scan_map = {(6, 7): _two_center_scan((6, 7))}
    cache = torsion_problem_module._MMProfileCache(parameter_set, [(6, 7)], scan_map)

    updated = deepcopy(parameter_set)
    updated.dihedrals[1].terms = [
        FourierTerm(kPhi=0.60, period=2.0, phase=np.pi),
        FourierTerm(kPhi=0.45, period=6.0, phase=0.42),
    ]

    direct_full = np.asarray([_fake_mm_energy(atoms, updated).total for atoms in scan_map[(6, 7)].frames], dtype=float)
    direct_zeroed = np.asarray(
        [
            _fake_mm_energy(atoms, updated, zero_proper_center_bond=(6, 7)).total
            for atoms in scan_map[(6, 7)].frames
        ],
        dtype=float,
    )
    direct_rel = (direct_full - direct_zeroed) - (direct_full - direct_zeroed)[int(scan_map[(6, 7)].ref_idx)]

    assert cache.center_torsion_rel(updated, (6, 7), (6, 7)).tolist() == pytest.approx(direct_rel.tolist(), abs=1.0e-8)


def test_fast_mm_cycle_reuses_profile_cache_instead_of_full_mm_refresh(monkeypatch):
    call_count = {"count": 0}

    def counted_mm_energy(*args, **kwargs):
        call_count["count"] += 1
        return _fake_mm_energy(*args, **kwargs)

    monkeypatch.setattr(torsion_problem_module, "evaluate_mm_energy", counted_mm_energy)
    monkeypatch.setattr(torsion_stage1_module, "evaluate_mm_energy", counted_mm_energy)

    scan_data = _period_phase_shifted_scan(1, 1.20, 0.30)
    parameter_set = _make_sparse_parameter_set([(0.20, 1.0, 0.0)])

    run_loss_mode(
        base_parameter_set=parameter_set,
        original_parameter_set=parameter_set,
        center_bonds=[(2, 3)],
        scan_data_map={(2, 3): scan_data},
        scan_xyz_map={(2, 3): "synthetic.xyz"},
        scan_mm_orig_rel_map={(2, 3): np.zeros_like(scan_data.qm_rel)},
        params=TorsionFitParams(enabled=True, refine_rounds=2, refine_max_iter=3),
    )

    assert call_count["count"] == len(scan_data.frames)


def _write_scan_xyz(path: Path, qm_k_values: tuple[float, ...]) -> None:
    frames = []
    for phi_deg in (0.0, 60.0, 120.0, 180.0, -120.0, -60.0):
        phi = math.radians(phi_deg)
        energy_kcal = 0.0
        for period, k_value in enumerate(qm_k_values, start=1):
            phase = 0.0 if period % 2 == 1 else np.pi
            energy_kcal += float(k_value) * (1.0 + math.cos(period * phi - phase))
        frames.append((phi_deg, energy_kcal / HARTREE_TO_KCAL_MOL, _four_atom_frame(phi_deg)))
    with path.open("w", encoding="utf-8") as handle:
        for index, (angle_deg, energy_hartree, atoms) in enumerate(frames, start=1):
            handle.write(f"{len(atoms)}\n")
            handle.write(
                f"Scanning combination {index}/{len(frames)}: [{angle_deg:.4f}]  Energy = {energy_hartree:.10f}\n"
            )
            for symbol, (x, y, z) in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
                handle.write(f"{symbol:2s} {x: .10f} {y: .10f} {z: .10f}\n")


def _phase_shifted_scan(k_value: float, phase: float) -> TorsionScanData:
    angles_deg = np.asarray((0.0, 60.0, 120.0, 180.0, 240.0, 300.0), dtype=float)
    frames = [_four_atom_frame(float(angle_deg)) for angle_deg in angles_deg]
    qm_kcal = np.zeros(len(frames), dtype=float)
    for index, atoms in enumerate(frames):
        phi = dihedral_radians(atoms.get_positions(), 1, 2, 3, 4)
        qm_kcal[index] = float(k_value) * (1.0 + math.cos(phi - float(phase)))
    ref_idx = int(np.argmin(qm_kcal))
    return TorsionScanData(
        angles_deg=angles_deg,
        qm_hartree=qm_kcal / HARTREE_TO_KCAL_MOL,
        qm_kcal=qm_kcal,
        frames=frames,
        source_path="synthetic_phase_shifted.xyz",
        ref_idx=ref_idx,
        qm_rel=qm_kcal - qm_kcal[ref_idx],
    )


def _period_phase_shifted_scan(period: int, k_value: float, phase: float) -> TorsionScanData:
    angles_deg = np.asarray((0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 210.0, 240.0, 270.0, 300.0, 330.0), dtype=float)
    frames = [_four_atom_frame(float(angle_deg)) for angle_deg in angles_deg]
    qm_kcal = np.zeros(len(frames), dtype=float)
    for index, atoms in enumerate(frames):
        phi = dihedral_radians(atoms.get_positions(), 1, 2, 3, 4)
        qm_kcal[index] = float(k_value) * (1.0 + math.cos(float(period) * phi - float(phase)))
    ref_idx = int(np.argmin(qm_kcal))
    return TorsionScanData(
        angles_deg=angles_deg,
        qm_hartree=qm_kcal / HARTREE_TO_KCAL_MOL,
        qm_kcal=qm_kcal,
        frames=frames,
        source_path="synthetic_period_phase_shifted.xyz",
        ref_idx=ref_idx,
        qm_rel=qm_kcal - qm_kcal[ref_idx],
    )


def _workflow_filter_scan_data() -> tuple[list[Atoms], TorsionScanData, dict[int, float]]:
    frames = [_four_atom_frame(0.0), _four_atom_frame(60.0), _four_atom_frame(120.0)]
    scan_data = TorsionScanData(
        angles_deg=np.asarray([0.0, 60.0, 120.0], dtype=float),
        qm_hartree=np.asarray([5.0, 0.0, 2.0], dtype=float) / HARTREE_TO_KCAL_MOL,
        qm_kcal=np.asarray([5.0, 0.0, 2.0], dtype=float),
        frames=frames,
        source_path="fake.xyz",
        ref_idx=1,
        qm_rel=np.asarray([5.0, 0.0, 2.0], dtype=float),
    )
    mm_total_by_frame = {
        id(frames[0]): 100.0,
        id(frames[1]): 0.0,
        id(frames[2]): 10.0,
    }
    return frames, scan_data, mm_total_by_frame


def _fake_mm_energy(atoms, parameter_set, topology_cache=None, zero_proper_center_bond=None):
    del topology_cache
    positions = atoms.get_positions()
    center = None if zero_proper_center_bond is None else tuple(sorted(zero_proper_center_bond))
    proper = 0.0
    for dihedral in parameter_set.dihedrals:
        dihedral_center = tuple(sorted((dihedral.atoms[1], dihedral.atoms[2])))
        if center is not None and dihedral_center == center:
            continue
        phi = dihedral_radians(positions, *dihedral.atoms)
        for term in dihedral.terms:
            proper += float(term.kPhi) * (1.0 + math.cos(float(term.period) * phi - float(term.phase)))
    return MMEnergy(proper=proper, total=proper)


def _rotate_about_x(vector: tuple[float, float, float], angle_deg: float) -> tuple[float, float, float]:
    angle = math.radians(angle_deg)
    x, y, z = vector
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    return (x, (cos_angle * y) - (sin_angle * z), (sin_angle * y) + (cos_angle * z))


def _methyl_like_frame(angle_deg: float) -> Atoms:
    b = np.asarray((0.0, 0.0, 0.0), dtype=float)
    c = np.asarray((1.5, 0.0, 0.0), dtype=float)
    a1 = np.asarray((-0.55, 1.10, 0.0), dtype=float)
    a2 = np.asarray((-0.55, -1.10, 0.0), dtype=float)
    hydrogen_vectors = [
        (0.36, 0.94, 0.00),
        (0.36, -0.47, 0.81),
        (0.36, -0.47, -0.81),
    ]
    hydrogens = [c + np.asarray(_rotate_about_x(vector, angle_deg), dtype=float) for vector in hydrogen_vectors]
    positions = np.vstack((a1, a2, b, c, *hydrogens))
    return Atoms(symbols=["C", "C", "C", "C", "H", "H", "H"], positions=positions)


def _make_methyl_like_parameter_set(per_dihedral_terms: list[list[FourierTerm]] | None = None) -> CorrectionParameterSet:
    mol2 = Mol2Topology(
        atoms=[
            Mol2Atom(atom_id=1, name="A1", atom_type="ca", charge=0.0),
            Mol2Atom(atom_id=2, name="A2", atom_type="ca", charge=0.0),
            Mol2Atom(atom_id=3, name="B", atom_type="ca", charge=0.0),
            Mol2Atom(atom_id=4, name="C", atom_type="c3", charge=0.0),
            Mol2Atom(atom_id=5, name="H1", atom_type="hc", charge=0.0),
            Mol2Atom(atom_id=6, name="H2", atom_type="hc", charge=0.0),
            Mol2Atom(atom_id=7, name="H3", atom_type="hc", charge=0.0),
        ],
        bonds=[
            Mol2Bond(bond_id=1, atom1=1, atom2=3, bond_type="ar"),
            Mol2Bond(bond_id=2, atom1=2, atom2=3, bond_type="ar"),
            Mol2Bond(bond_id=3, atom1=3, atom2=4, bond_type="1"),
            Mol2Bond(bond_id=4, atom1=4, atom2=5, bond_type="1"),
            Mol2Bond(bond_id=5, atom1=4, atom2=6, bond_type="1"),
            Mol2Bond(bond_id=6, atom1=4, atom2=7, bond_type="1"),
        ],
        id_to_index={index: index for index in range(1, 8)},
        adjacency={
            1: {3},
            2: {3},
            3: {1, 2, 4},
            4: {3, 5, 6, 7},
            5: {4},
            6: {4},
            7: {4},
        },
    )
    dihedral_atoms = [
        (1, 3, 4, 5),
        (1, 3, 4, 6),
        (1, 3, 4, 7),
        (2, 3, 4, 5),
        (2, 3, 4, 6),
        (2, 3, 4, 7),
    ]
    if per_dihedral_terms is None:
        per_dihedral_terms = [[FourierTerm(0.0, 1.0, 0.0)] for _ in dihedral_atoms]
    dihedrals = [
        Dihedral(atoms=atoms, atom_types=("ca", "ca", "c3", "hc"), terms=[FourierTerm(term.kPhi, term.period, term.phase) for term in terms])
        for atoms, terms in zip(dihedral_atoms, per_dihedral_terms)
    ]
    nonbonds = [
        Nonbond(atom=index, atom_type=atom_type, charge=0.0, rmin_half=1.5, epsilon=0.0)
        for index, atom_type in ((1, "ca"), (2, "ca"), (3, "ca"), (4, "c3"), (5, "hc"), (6, "hc"), (7, "hc"))
    ]
    return CorrectionParameterSet(
        mol2=mol2,
        frcmod=FrcmodDB(),
        bonds=[],
        angles=[],
        dihedrals=dihedrals,
        impropers=[],
        nonbonds=nonbonds,
        unmatched_bonds=[],
        unmatched_angles=[],
        unmatched_dihedrals=[],
        unmatched_impropers=[],
        unmatched_nonbonds=[],
    )


def _methyl_like_scan(left_k3: float, right_k3: float) -> TorsionScanData:
    angles_deg = np.arange(0.0, 360.0, 30.0, dtype=float)
    frames = [_methyl_like_frame(float(angle_deg)) for angle_deg in angles_deg]
    qm_kcal = np.zeros(len(frames), dtype=float)
    left_dihedrals = ((1, 3, 4, 5), (1, 3, 4, 6), (1, 3, 4, 7))
    right_dihedrals = ((2, 3, 4, 5), (2, 3, 4, 6), (2, 3, 4, 7))
    for index, atoms in enumerate(frames):
        positions = atoms.get_positions()
        for dihedral_atoms in left_dihedrals:
            phi = dihedral_radians(positions, *dihedral_atoms)
            qm_kcal[index] += float(left_k3) * (1.0 + math.cos(3.0 * phi))
        for dihedral_atoms in right_dihedrals:
            phi = dihedral_radians(positions, *dihedral_atoms)
            qm_kcal[index] += float(right_k3) * (1.0 + math.cos(3.0 * phi))
    ref_idx = int(np.argmin(qm_kcal))
    qm_rel = qm_kcal - qm_kcal[ref_idx]
    return TorsionScanData(
        angles_deg=angles_deg,
        qm_hartree=qm_kcal / HARTREE_TO_KCAL_MOL,
        qm_kcal=qm_kcal,
        frames=frames,
        source_path="synthetic_methyl_like_scan.xyz",
        ref_idx=ref_idx,
        qm_rel=qm_rel,
    )


def _single_frame_scan_map(atoms: Atoms, centers: tuple[tuple[int, int], ...]) -> dict[tuple[int, int], TorsionScanData]:
    return {
        normalize_center_bond(center): TorsionScanData(
            angles_deg=np.asarray([0.0], dtype=float),
            qm_hartree=np.asarray([0.0], dtype=float),
            qm_kcal=np.asarray([0.0], dtype=float),
            frames=[atoms.copy()],
            source_path=f"{center[0]}-{center[1]}_scan.xyz",
            ref_idx=0,
            qm_rel=np.asarray([0.0], dtype=float),
        )
        for center in centers
    }


def _guard_problem(
    qm_rel: list[float],
    constant_rel: list[float],
    basis: list[list[float]],
    *,
    scales: list[float] | None = None,
) -> TorsionGlobalProblem:
    center = (2, 3)
    qm = np.asarray(qm_rel, dtype=float)
    constant = np.asarray(constant_rel, dtype=float)
    matrix = np.asarray(basis, dtype=float)
    parameter_set = _make_parameter_set(tuple(0.0 for _ in range(matrix.shape[1])))
    scan_data = TorsionScanData(
        angles_deg=np.linspace(0.0, 360.0, len(qm), endpoint=False),
        qm_hartree=qm / HARTREE_TO_KCAL_MOL,
        qm_kcal=qm,
        frames=[_four_atom_frame(float(angle)) for angle in np.linspace(0.0, 360.0, len(qm), endpoint=False)],
        source_path="guard_synthetic.xyz",
        ref_idx=int(np.argmin(qm)),
        qm_rel=qm,
    )
    return TorsionGlobalProblem(
        stage0_parameter_set=parameter_set,
        center_bonds=(center,),
        scan_map={center: scan_data},
        term_paths=tuple((0, index) for index in range(matrix.shape[1])),
        block_slices={center: (0, matrix.shape[1])},
        k_orig=np.zeros(matrix.shape[1], dtype=float),
        phase_orig=np.zeros(matrix.shape[1], dtype=float),
        period_orig=np.arange(1, matrix.shape[1] + 1, dtype=float),
        scales=np.asarray(scales, dtype=float) if scales is not None else np.ones(matrix.shape[1], dtype=float),
        qm_rel_map={center: qm},
        centered_basis_map={center: matrix},
        centered_cos_basis_map={center: matrix},
        centered_sin_basis_map={center: np.zeros_like(matrix)},
        constant_rel_map={center: constant.copy()},
        global_max_iter=5,
    )


class _FakeMinimizeResult:
    def __init__(self, x):
        self.x = np.asarray(x, dtype=float)


def _patch_stage2_minimize(monkeypatch, fake_minimize):
    monkeypatch.setattr(torsion_stage2_module, "minimize", fake_minimize)


def test_stage2_optimizer_uses_k_phase_vector(monkeypatch):
    problem = replace(
        _guard_problem(
            qm_rel=[0.0, 0.0],
            constant_rel=[0.0, 0.0],
            basis=[[0.0], [1.0]],
            scales=[100.0],
        ),
        k_orig=np.asarray([1.0], dtype=float),
        phase_orig=np.asarray([0.0], dtype=float),
    )
    seen: dict[str, np.ndarray] = {}

    def fake_minimize(fun, x0, jac, method, bounds, options):
        del jac, method, options
        seen["x0"] = np.asarray(x0, dtype=float).copy()
        assert seen["x0"].tolist() == pytest.approx([1.0, 0.0])
        assert bounds[0][0] == pytest.approx(0.0)
        assert bounds[0][1] > 0.0
        loss, gradient = fun(np.asarray([0.0, 0.0], dtype=float))
        assert np.isfinite(loss)
        assert gradient.shape == (2,)
        return _FakeMinimizeResult([0.0, 0.0])

    _patch_stage2_minimize(monkeypatch, fake_minimize)

    vector_final, cycles = refine_torsion_scans_global(
        problem,
        delta_init=np.zeros(2 * len(problem.k_orig), dtype=float),
        enabled=True,
        max_block_iter=5,
        tol=1.0e-8,
    )

    assert seen["x0"].tolist() == pytest.approx([1.0, 0.0])
    assert vector_final.tolist() == pytest.approx([-1.0, 0.0])
    assert cycles[-1].accepted_blocks == 1
    assert cycles[-1].diagnostics["status"] == "accepted"


def test_stage2_uses_optimizer_final_without_intermediate_selection(monkeypatch):
    problem = _guard_problem(
        qm_rel=[0.0, 1.0, 0.0, 1.0],
        constant_rel=[0.0, 0.0, 0.0, 0.0],
        basis=[[0.0], [1.0], [0.0], [1.0]],
        scales=[100.0],
    )
    better_intermediate = np.asarray([1.0, 0.0], dtype=float)
    worse_final = np.asarray([0.5, 0.0], dtype=float)

    def fake_minimize(fun, x0, jac, method, bounds, options):
        del x0, jac, method, bounds, options
        fun(better_intermediate)
        fun(worse_final)
        return _FakeMinimizeResult(worse_final)

    _patch_stage2_minimize(monkeypatch, fake_minimize)

    vector_final, cycles = refine_torsion_scans_global(
        problem,
        delta_init=np.zeros(2 * len(problem.k_orig), dtype=float),
        enabled=True,
        max_block_iter=5,
        tol=1.0e-8,
    )

    assert vector_final.tolist() == pytest.approx(worse_final.tolist())
    assert cycles[-1].diagnostics["status"] == "accepted"


def test_stage2_direct_optimizer_keeps_stage1_when_total_loss_not_improved(monkeypatch):
    problem = _guard_problem(
        qm_rel=[0.0, 1.0, 2.0, 20.0, 3.0, 2.0],
        constant_rel=[0.0, 0.975, 1.975, 19.975, 2.975, 1.975],
        basis=[[0.0], [0.01], [0.01], [0.01], [0.01], [0.01]],
        scales=[100.0],
    )
    candidate = np.asarray([2.5, 0.0], dtype=float)
    calls = {"count": 0}

    def fake_minimize(fun, x0, jac, method, bounds, options):
        del x0, jac, method, bounds, options
        calls["count"] += 1
        fun(candidate)
        return _FakeMinimizeResult(candidate)

    _patch_stage2_minimize(monkeypatch, fake_minimize)

    vector_final, cycles = refine_torsion_scans_global(
        problem,
        delta_init=np.zeros(2 * len(problem.k_orig), dtype=float),
        enabled=True,
        max_block_iter=5,
        tol=1.0e-8,
    )

    assert calls["count"] == 1
    assert vector_final.tolist() == pytest.approx([0.0, 0.0])
    assert cycles[-1].accepted_blocks == 0
    assert cycles[-1].rejected_blocks == 1
    assert cycles[-1].diagnostics["status"] == "kept_stage1"
    assert "per_scan" not in cycles[-1].diagnostics


def test_stage2_accepts_only_when_final_total_loss_improves(monkeypatch):
    problem = _guard_problem(
        qm_rel=[0.0, 2.0],
        constant_rel=[0.0, -1.0],
        basis=[[0.0], [1.0]],
        scales=[1.0],
    )
    low_total = np.asarray([1.0, 0.0], dtype=float)
    low_data_high_prior = np.asarray([3.0, 0.0], dtype=float)
    cache = _build_stage2_objective_cache(problem)
    low_total_eval = evaluate_global_refit_objective(problem, low_total, cache=cache)
    low_data_eval = evaluate_global_refit_objective(problem, low_data_high_prior, cache=cache)
    assert low_data_eval.data_loss < low_total_eval.data_loss
    assert low_total_eval.total_loss < low_data_eval.total_loss

    def fake_minimize(fun, x0, jac, method, bounds, options):
        del x0, jac, method, bounds, options
        fun(low_total)
        fun(low_data_high_prior)
        return _FakeMinimizeResult(low_data_high_prior)

    _patch_stage2_minimize(monkeypatch, fake_minimize)

    vector_final, cycles = refine_torsion_scans_global(
        problem,
        delta_init=np.zeros(2 * len(problem.k_orig), dtype=float),
        enabled=True,
        max_block_iter=5,
        tol=1.0e-8,
    )

    assert vector_final.tolist() == pytest.approx([0.0, 0.0])
    assert cycles[-1].diagnostics["status"] == "kept_stage1"


def test_stage2_extra_target_contributes_to_hybrid_objective():
    scan_only = _guard_problem(
        qm_rel=[0.0, 0.0],
        constant_rel=[0.0, 0.0],
        basis=[[0.0], [0.0]],
        scales=[1.0],
    )
    extra = TorsionObjectiveTarget(
        label="ensemble",
        center_bond=(2, 3),
        qm_rel=np.asarray([0.0, 2.0], dtype=float),
        constant_rel=np.asarray([0.0, 0.0], dtype=float),
        cos_basis=np.asarray([[0.0], [0.0]], dtype=float),
        sin_basis=np.asarray([[0.0], [0.0]], dtype=float),
        weight=0.5,
        source_path="ensemble.xyz",
    )
    hybrid = replace(scan_only, extra_targets=(extra,))
    zero_weight = replace(scan_only, extra_targets=(replace(extra, weight=0.0),))
    vector = np.zeros(2 * len(scan_only.k_orig), dtype=float)

    scan_eval = evaluate_global_refit_objective(scan_only, vector)
    hybrid_eval = evaluate_global_refit_objective(hybrid, vector)
    zero_eval = evaluate_global_refit_objective(zero_weight, vector)

    assert hybrid_eval.data_loss > scan_eval.data_loss
    assert hybrid_eval.total_loss > scan_eval.total_loss
    assert zero_eval.data_loss == pytest.approx(scan_eval.data_loss)
    assert zero_eval.total_loss == pytest.approx(scan_eval.total_loss)


def test_stage2_extra_target_builder_converts_ensemble_frames_to_centered_basis():
    problem = _guard_problem(
        qm_rel=[0.0, 0.0],
        constant_rel=[0.0, 0.0],
        basis=[[0.0], [1.0]],
        scales=[1.0],
    )
    frames = (_four_atom_frame(0.0), _four_atom_frame(60.0))
    ensemble = TorsionEnsembleResult(
        frames_by_center={(2, 3): frames},
        energies_by_center={(2, 3): np.asarray([10.0, 13.0], dtype=float)},
        xyz_paths={(2, 3): "ensemble.xyz"},
    )

    targets = build_stage2_extra_targets(
        problem,
        ensemble,
        TorsionFitParams(enabled=True, torsion_ensemble=True, torsion_ensemble_weight=0.25),
    )

    assert len(targets) == 1
    assert targets[0].qm_rel.tolist() == pytest.approx([10.0, 13.0])
    assert targets[0].cos_basis.shape == (2, len(problem.k_orig))
    assert targets[0].sin_basis.shape == (2, len(problem.k_orig))
    assert targets[0].weight == pytest.approx(0.25)
    assert targets[0].source_path == "ensemble.xyz"


def test_stage2_extra_target_builder_aligns_ensemble_to_scan_reference():
    problem = _guard_problem(
        qm_rel=[4.0, 1.0],
        constant_rel=[0.0, 0.0],
        basis=[[0.0], [1.0]],
        scales=[1.0],
    )
    frames = (_four_atom_frame(0.0), _four_atom_frame(60.0))
    ensemble = TorsionEnsembleResult(
        frames_by_center={(2, 3): frames},
        energies_by_center={(2, 3): np.asarray([10.0, 13.0], dtype=float)},
        xyz_paths={(2, 3): "ensemble.xyz"},
    )

    targets = build_stage2_extra_targets(
        problem,
        ensemble,
        TorsionFitParams(enabled=True, torsion_ensemble=True, torsion_ensemble_weight=0.25),
    )

    scan_data = problem.scan_map[(2, 3)]
    scan_ref_frame = scan_data.frames[int(scan_data.ref_idx)]
    ensemble_const, ensemble_cos, ensemble_sin = torsion_ensemble_module._absolute_basis_for_frames(problem, frames)
    ref_const, ref_cos, ref_sin = torsion_ensemble_module._absolute_basis_for_frames(problem, (scan_ref_frame,))

    assert len(targets) == 1
    assert targets[0].qm_rel.tolist() == pytest.approx([9.0, 12.0])
    assert targets[0].constant_rel.tolist() == pytest.approx([0.0, 0.0])
    np.testing.assert_allclose(targets[0].cos_basis, ensemble_cos - ref_cos[0])
    np.testing.assert_allclose(targets[0].sin_basis, ensemble_sin - ref_sin[0])


def test_stage2_accepts_optimizer_final_when_total_loss_improves(monkeypatch):
    scan_only = _guard_problem(
        qm_rel=[0.0, 1.0],
        constant_rel=[0.0, 0.0],
        basis=[[0.0], [1.0]],
        scales=[1.0],
    )
    extra = TorsionObjectiveTarget(
        label="ensemble",
        center_bond=(2, 3),
        qm_rel=np.asarray([0.0, -0.08], dtype=float),
        constant_rel=np.asarray([0.0, 0.0], dtype=float),
        cos_basis=np.asarray([[0.0], [1.0]], dtype=float),
        sin_basis=np.asarray([[0.0], [0.0]], dtype=float),
        weight=10.0,
        source_path="ensemble.xyz",
    )
    problem = replace(scan_only, extra_targets=(extra,))
    candidate = np.asarray([-0.02, 0.0], dtype=float)
    before = evaluate_global_refit_objective(problem, np.zeros(2 * len(problem.k_orig), dtype=float))
    after = evaluate_global_refit_objective(problem, candidate)

    assert after.total_loss < before.total_loss

    monkeypatch.setattr(torsion_stage2_module, "minimize", lambda *args, **kwargs: _FakeMinimizeResult(candidate))

    vector_final, cycles = refine_torsion_scans_global(
        problem,
        delta_init=np.zeros(2 * len(problem.k_orig), dtype=float),
        enabled=True,
        max_block_iter=5,
        tol=1.0e-8,
    )

    assert vector_final.tolist() == pytest.approx(candidate.tolist())
    assert cycles[-1].accepted_blocks == 1
    assert cycles[-1].diagnostics["status"] == "accepted"


def test_stage2_accepts_total_loss_gain_even_when_scan_loss_worsens(monkeypatch):
    scan_only = _guard_problem(
        qm_rel=[0.0, 1.0],
        constant_rel=[0.0, 0.0],
        basis=[[0.0], [1.0]],
        scales=[1.0],
    )
    extra = TorsionObjectiveTarget(
        label="ensemble",
        center_bond=(2, 3),
        qm_rel=np.asarray([0.0, -0.20], dtype=float),
        constant_rel=np.asarray([0.0, 0.0],
        dtype=float),
        cos_basis=np.asarray([[0.0], [1.0]], dtype=float),
        sin_basis=np.asarray([[0.0], [0.0]], dtype=float),
        weight=20.0,
        source_path="ensemble.xyz",
    )
    problem = replace(scan_only, extra_targets=(extra,))
    candidate = np.asarray([-0.20, 0.0], dtype=float)
    before = evaluate_global_refit_objective(problem, np.zeros(2 * len(problem.k_orig), dtype=float))
    after = evaluate_global_refit_objective(problem, candidate)

    assert after.total_loss < before.total_loss

    monkeypatch.setattr(torsion_stage2_module, "minimize", lambda *args, **kwargs: _FakeMinimizeResult(candidate))

    vector_final, cycles = refine_torsion_scans_global(
        problem,
        delta_init=np.zeros(2 * len(problem.k_orig), dtype=float),
        enabled=True,
        max_block_iter=5,
        tol=1.0e-8,
    )

    assert vector_final.tolist() == pytest.approx(candidate.tolist())
    assert cycles[-1].accepted_blocks == 1
    assert cycles[-1].diagnostics["status"] == "accepted"


def test_ensemble_effective_size_uses_ratio_of_scan_conf_count():
    params = TorsionFitParams(enabled=True, torsion_ensemble=True, torsion_ensemble_ratio=0.25, torsion_steps=36)

    assert torsion_ensemble_module._effective_ensemble_size(params, ((2, 3), (6, 7))) == 19
    assert torsion_ensemble_module._effective_ensemble_size(
        params,
        ((1, 2), (2, 3), (3, 4), (4, 5), (5, 6)),
    ) == 47
    assert torsion_ensemble_module._effective_ensemble_size(
        params,
        tuple((index, index + 1) for index in range(1, 21)),
    ) == 185

    small_ratio = TorsionFitParams(enabled=True, torsion_ensemble=True, torsion_ensemble_ratio=0.01, torsion_steps=36)
    assert torsion_ensemble_module._effective_ensemble_size(small_ratio, ((2, 3), (6, 7))) == 1


def test_ensemble_trial_sampling_can_activate_more_than_four_rotors():
    rotors = tuple((idx, idx + 1, idx + 2, idx + 3) for idx in range(1, 8))
    center_bonds = tuple((idx + 1, idx + 2) for idx in range(1, 8))

    trials = torsion_ensemble_module._trial_offset_maps(rotors, center_bonds, size=50)

    assert any(len(trial) > 4 for _center, trial in trials)


def test_ensemble_trial_sampling_can_include_noncenter_eligible_rotors():
    rotors = (
        (1, 2, 3, 4),
        (4, 5, 6, 7),
        (7, 8, 9, 10),
    )

    trials = torsion_ensemble_module._trial_offset_maps(rotors, ((2, 3), (5, 6)), size=80)

    sampled_centers = {
        tuple(sorted((rotor[1], rotor[2])))
        for _center, trial in trials
        for rotor in trial
    }
    assert (8, 9) in sampled_centers


def test_ensemble_rotor_uses_graph_fragment_not_terminal_dihedral_atom():
    parameter_set = _make_terminal_h_first_rotor_parameter_set()

    rotor = next(
        rotor
        for rotor in torsion_ensemble_module._eligible_rotors(parameter_set, None, 8)
        if tuple(sorted((rotor[1], rotor[2]))) == (1, 5)
    )
    moved_atoms = [
        index + 1
        for index, flag in enumerate(torsion_ensemble_module._rotation_mask(parameter_set, rotor, None, 8))
        if flag
    ]

    assert moved_atoms != [6]
    assert len(moved_atoms) > 1


def test_ensemble_rotor_keeps_terminal_heavy_atom_sides():
    parameter_set = _make_terminal_cl_rotor_parameter_set()

    rotors = torsion_ensemble_module._eligible_rotors(parameter_set, None, 4)

    assert rotors


def test_ensemble_rotor_skips_ring_center_bonds():
    parameter_set = _make_ring_center_parameter_set()

    rotors = torsion_ensemble_module._eligible_rotors(parameter_set, None, 4)

    assert rotors == ()


def test_ensemble_sampling_uses_global_budget_and_mlip_sp_only_after_mm_screen(tmp_path, monkeypatch):
    parameter_set = _make_two_center_parameter_set()
    atoms = _two_fragment_frame(0.0, 20.0)
    params = TorsionFitParams(
        enabled=True,
        torsion_ensemble=True,
        torsion_ensemble_ratio=0.05,
    )
    mlip_calls = {"count": 0}

    def fake_mlip_single_point(candidate):
        mlip_calls["count"] += 1
        return float(0.01 * mlip_calls["count"])

    monkeypatch.setattr(torsion_ensemble_module, "_ensemble_worker_count", lambda: 1, raising=False)
    monkeypatch.setattr(torsion_ensemble_module, "evaluate_mm_energy", _fake_mm_energy)
    monkeypatch.setattr(torsion_ensemble_module, "get_potential_energy", fake_mlip_single_point)

    result = torsion_ensemble_module.build_torsion_local_ensemble(
        atoms=atoms,
        parameter_set=parameter_set,
        center_bonds=((2, 3), (6, 7)),
        scan_data_map=_single_frame_scan_map(atoms, ((2, 3), (6, 7))),
        params=params,
        output=str(tmp_path / "toy.out"),
    )

    selected = sum(len(frames) for frames in result.frames_by_center.values())
    assert 0 < selected <= 4
    assert mlip_calls["count"] == selected
    assert result.xyz_paths
    assert len(set(result.xyz_paths.values())) == 1
    xyz_path = next(iter(result.xyz_paths.values()))
    assert xyz_path.endswith("_torsionfit_ensemble.xyz")
    xyz_text = Path(xyz_path).read_text()
    assert "mm_screen_rel=" in xyz_text
    assert "mlip_stage_rel=" in xyz_text
    assert "mm_rel=" not in xyz_text
    assert "mlip_rel=" not in xyz_text
    assert "sampled=" in xyz_text


def test_ensemble_sampling_filters_high_mlip_relative_frames_after_sp(tmp_path, monkeypatch):
    parameter_set = _make_two_center_parameter_set()
    atoms = _two_fragment_frame(0.0, 20.0)
    params = TorsionFitParams(enabled=True, torsion_ensemble=True, torsion_ensemble_ratio=0.25)
    base_positions = np.asarray(atoms.get_positions(), dtype=float)
    selected = []
    for index, dy in enumerate((0.10, 0.25, 0.40), start=1):
        shifted = base_positions.copy()
        shifted[3, 1] += dy
        selected.append(((2, 3), shifted, float(index), {(1, 2, 3, 4): float(15 * index)}))

    mlip_kcal = iter((0.0, 10.0, 80.0))

    monkeypatch.setattr(torsion_ensemble_module, "_ensemble_worker_count", lambda: 1, raising=False)
    monkeypatch.setattr(torsion_ensemble_module, "_screen_trials_with_mm", lambda **kwargs: list(selected))
    monkeypatch.setattr(torsion_ensemble_module, "_select_low_mm_energy_diverse", lambda candidates, **kwargs: list(candidates))
    monkeypatch.setattr(torsion_ensemble_module, "get_potential_energy", lambda atoms: next(mlip_kcal) / HARTREE_TO_KCAL_MOL)

    result = torsion_ensemble_module.build_torsion_local_ensemble(
        atoms=atoms,
        parameter_set=parameter_set,
        center_bonds=((2, 3), (6, 7)),
        scan_data_map=_single_frame_scan_map(atoms, ((2, 3), (6, 7))),
        params=params,
        output=str(tmp_path / "toy.out"),
    )

    assert len(result.frames_by_center[(2, 3)]) == 2
    assert any("high-MLIP" in warning for warning in result.warnings)
    xyz_text = Path(result.xyz_paths[(2, 3)]).read_text()
    assert "mlip_stage_rel=80" not in xyz_text


def test_ensemble_sampling_skips_single_center_bond(tmp_path, monkeypatch):
    params = TorsionFitParams(
        enabled=True,
        torsion_ensemble=True,
        torsion_ensemble_ratio=0.25,
    )
    mlip_calls = {"count": 0}

    def fake_mlip_single_point(candidate):
        del candidate
        mlip_calls["count"] += 1
        return 0.0

    monkeypatch.setattr(torsion_ensemble_module, "_ensemble_worker_count", lambda: 1, raising=False)
    monkeypatch.setattr(torsion_ensemble_module, "evaluate_mm_energy", _fake_mm_energy)
    monkeypatch.setattr(torsion_ensemble_module, "get_potential_energy", fake_mlip_single_point)

    result = torsion_ensemble_module.build_torsion_local_ensemble(
        atoms=_four_atom_frame(0.0),
        parameter_set=_make_sparse_parameter_set([(0.2, 1.0, 0.0)]),
        center_bonds=((2, 3),),
        scan_data_map={},
        params=params,
        output=str(tmp_path / "single.out"),
    )

    assert result.frames_by_center == {}
    assert result.xyz_paths == {}
    assert mlip_calls["count"] == 0
    assert any("only one fitted center bond" in warning for warning in result.warnings)


def test_stage2_direct_optimizer_updates_global_vector(monkeypatch):
    center_a = (2, 3)
    center_b = (3, 4)
    parameter_set = _make_parameter_set((0.0, 0.0))
    scan_a = TorsionScanData(
        angles_deg=np.asarray([0.0, 180.0], dtype=float),
        qm_hartree=np.asarray([0.0, 2.0], dtype=float) / HARTREE_TO_KCAL_MOL,
        qm_kcal=np.asarray([0.0, 2.0], dtype=float),
        frames=[_four_atom_frame(0.0), _four_atom_frame(180.0)],
        source_path="stage2_center_a.xyz",
        ref_idx=0,
        qm_rel=np.asarray([0.0, 2.0], dtype=float),
    )
    scan_b = TorsionScanData(
        angles_deg=np.asarray([0.0, 180.0], dtype=float),
        qm_hartree=np.asarray([0.0, 0.2], dtype=float) / HARTREE_TO_KCAL_MOL,
        qm_kcal=np.asarray([0.0, 0.2], dtype=float),
        frames=[_four_atom_frame(0.0), _four_atom_frame(180.0)],
        source_path="stage2_center_b.xyz",
        ref_idx=0,
        qm_rel=np.asarray([0.0, 0.2], dtype=float),
    )
    problem = TorsionGlobalProblem(
        stage0_parameter_set=parameter_set,
        center_bonds=(center_a, center_b),
        scan_map={center_a: scan_a, center_b: scan_b},
        term_paths=((0, 0), (0, 1)),
        block_slices={center_a: (0, 1), center_b: (1, 2)},
        k_orig=np.zeros(2, dtype=float),
        phase_orig=np.zeros(2, dtype=float),
        period_orig=np.asarray([1.0, 2.0], dtype=float),
        scales=np.asarray([100.0, 100.0], dtype=float),
        qm_rel_map={center_a: scan_a.qm_rel, center_b: scan_b.qm_rel},
        centered_basis_map={
            center_a: np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=float),
            center_b: np.asarray([[0.0, 0.0], [0.0, 1.0]], dtype=float),
        },
        centered_cos_basis_map={
            center_a: np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=float),
            center_b: np.asarray([[0.0, 0.0], [0.0, 1.0]], dtype=float),
        },
        centered_sin_basis_map={
            center_a: np.zeros((2, 2), dtype=float),
            center_b: np.zeros((2, 2), dtype=float),
        },
        constant_rel_map={center_a: np.zeros(2, dtype=float), center_b: np.zeros(2, dtype=float)},
        global_max_iter=5,
    )
    candidate = np.asarray([2.0, 0.5, 0.0, 0.0], dtype=float)

    monkeypatch.setattr(
        "maple.function.dispatcher.parmfit.utils.TorsionFit.stage2.minimize",
        lambda *args, **kwargs: _FakeMinimizeResult(candidate),
    )

    vector_final, cycles = refine_torsion_scans_global(
        problem,
        delta_init=np.zeros(4, dtype=float),
        enabled=True,
        max_block_iter=5,
        tol=1.0e-8,
    )

    assert vector_final.tolist() == pytest.approx([2.0, 0.5, 0.0, 0.0])
    assert cycles[-1].accepted_blocks == 2
    assert cycles[-1].rejected_blocks == 0
    assert cycles[-1].diagnostics["status"] == "accepted"


def test_run_loss_mode_exposes_public_five_curve_outputs(tmp_path: Path, monkeypatch):
    import maple.function.dispatcher.parmfit.utils.TorsionFit.fit as fit_module
    import maple.function.dispatcher.parmfit.utils.TorsionFit.basis as problem_module

    monkeypatch.setattr(problem_module, "evaluate_mm_energy", _fake_mm_energy)
    monkeypatch.setattr(torsion_stage1_module, "evaluate_mm_energy", _fake_mm_energy)

    scan_path = tmp_path / "scan.xyz"
    _write_scan_xyz(scan_path, (1.20, 0.25, 0.0, 0.0))
    scan_data = read_scan_xyz(str(scan_path))
    original_parameter_set = _make_parameter_set((0.20, 0.0, 0.0, 0.0))
    stage0_parameter_set = _make_parameter_set((0.45, 0.0, 0.0, 0.0))

    log_lines: list[str] = []
    result = run_loss_mode(
        base_parameter_set=stage0_parameter_set,
        original_parameter_set=original_parameter_set,
        center_bonds=[(2, 3)],
        scan_data_map={(2, 3): scan_data},
        scan_xyz_map={(2, 3): str(scan_path)},
        params=TorsionFitParams(enabled=True),
        log_info=log_lines.extend,
    )

    text = "".join(log_lines)
    assert "MM_orig" in text
    assert "MM_stage0" in text
    assert "MM_stage1" in text
    assert "MM_stage2" in text
    assert f"{float(scan_data.qm_rel[0]):10.6f}" in text
    assert result.stage1_diagnostics["solver"] == "local_restrained_lls"
    assert result.stage2_diagnostics["solver"] == "continuous_phase_k_refine"
    assert result.stage2_diagnostics["requested_cycles"] == 2
    assert result.stage2_diagnostics["cycles"] == len(result.refine_cycles)
    assert "final_cycle" in result.stage2_diagnostics


def test_run_loss_mode_does_not_accept_rejected_stage2_first_cycle(monkeypatch):
    import maple.function.dispatcher.parmfit.utils.TorsionFit.fit as fit_module

    parameter_set = _make_parameter_set((0.0,))
    report = types.SimpleNamespace(
        center_bond=(2, 3),
        curves=types.SimpleNamespace(mm_stage1_rel=np.asarray([0.0, 1.0], dtype=float)),
    )
    before_eval = types.SimpleNamespace(
        total_loss=1.0,
        data_loss=1.0,
        prior_loss=0.0,
        global_rmse=1.0,
        objective_kind="direct_k_phase",
    )
    after_eval = types.SimpleNamespace(
        total_loss=1.0,
        data_loss=1.0,
        prior_loss=0.0,
        global_rmse=1.0,
        objective_kind="direct_k_phase",
    )

    def fake_fit_stage1_cycle(**_kwargs):
        return parameter_set, [report], {"solver": "fake_stage1"}

    def fake_run_stage2_cycle(**_kwargs):
        cycle = TorsionRefineCycle(
            cycle=1,
            total_loss_before=1.0,
            total_loss_after=1.0,
            global_rmse_before=1.0,
            global_rmse_after=1.0,
            accepted_blocks=0,
            rejected_blocks=1,
            per_scan_rmse_before={(2, 3): 1.0},
            per_scan_rmse_after={(2, 3): 1.0},
            diagnostics={"status": "kept_stage1", "reject_reason": "no_total_loss_gain"},
        )
        return parameter_set, [report], [cycle], before_eval, after_eval, dict(cycle.diagnostics)

    monkeypatch.setattr(fit_module, "_MMProfileCache", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(fit_module, "_fit_stage1_cycle", fake_fit_stage1_cycle)
    monkeypatch.setattr(fit_module, "_run_stage2_cycle", fake_run_stage2_cycle)

    result = fit_module.run_loss_mode(
        base_parameter_set=parameter_set,
        center_bonds=[(2, 3)],
        scan_data_map={},
        scan_xyz_map={},
        params=TorsionFitParams(enabled=True, refine_rounds=1),
    )

    assert result.stage2_diagnostics["accepted_cycles"] == 0
    assert result.stage2_diagnostics["rejected_cycles"] == 1
    assert result.refine_cycles[0].diagnostics["outer_cycle_rejected"] is True


def test_stage2_rejects_large_cancelling_terms_even_when_profile_improves(monkeypatch):
    import maple.function.dispatcher.parmfit.utils.TorsionFit.fit as fit_module

    problem = _guard_problem(
        qm_rel=[0.0, 2.0, 0.0, 2.0],
        constant_rel=[0.0, 20.0, 0.0, 20.0],
        basis=[[0.0, 0.0], [-9.0, -9.0], [0.0, 0.0], [-9.0, -9.0]],
        scales=[100.0, 100.0],
    )
    candidate = np.asarray([20.0, -18.0, 0.0, 0.0], dtype=float)
    before = evaluate_global_refit_objective(problem, np.zeros(2 * len(problem.k_orig), dtype=float))
    after = evaluate_global_refit_objective(problem, candidate)
    assert after.total_loss < before.total_loss
    assert after.per_scan_rmse[(2, 3)] < before.per_scan_rmse[(2, 3)]

    monkeypatch.setattr(torsion_stage2_module, "minimize", lambda *args, **kwargs: _FakeMinimizeResult(candidate))

    vector_final, cycles = refine_torsion_scans_global(
        problem,
        delta_init=np.zeros(2 * len(problem.k_orig), dtype=float),
        enabled=True,
        max_block_iter=5,
        tol=1.0e-8,
    )

    assert vector_final.tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0])
    assert cycles[-1].accepted_blocks == 0


def test_stage2_records_data_loss_regression_without_hard_reject(monkeypatch):
    import maple.function.dispatcher.parmfit.utils.TorsionFit.fit as fit_module

    problem = _guard_problem(
        qm_rel=[0.0, 1.0, 0.0, 1.0],
        constant_rel=[0.0, 0.2, 0.0, 0.2],
        basis=[[0.0], [3.0], [0.0], [3.0]],
    )
    candidate = np.asarray([1.0, 0.0], dtype=float)
    before = evaluate_global_refit_objective(problem, np.zeros(2 * len(problem.k_orig), dtype=float))
    after = evaluate_global_refit_objective(problem, candidate)
    assert after.data_loss > before.data_loss

    monkeypatch.setattr(torsion_stage2_module, "minimize", lambda *args, **kwargs: _FakeMinimizeResult(candidate))

    vector_final, cycles = refine_torsion_scans_global(
        problem,
        delta_init=np.zeros(2 * len(problem.k_orig), dtype=float),
        enabled=True,
        max_block_iter=5,
        tol=1.0e-8,
    )

    assert vector_final.tolist() == pytest.approx([0.0, 0.0])
    assert cycles[-1].accepted_blocks == 0
    assert cycles[-1].diagnostics["reject_reason"] == "no_total_loss_gain"
    assert cycles[-1].diagnostics["status"] == "kept_stage1"


def test_stage2_keeps_stage1_when_optimizer_returns_initial_vector(monkeypatch):
    import maple.function.dispatcher.parmfit.utils.TorsionFit.fit as fit_module

    problem = _guard_problem(
        qm_rel=[0.0, 1.0, 2.0, 20.0, 3.0, 2.0],
        constant_rel=[0.0, 1.0, 2.0, 20.0, 3.0, 2.0],
        basis=[[0.0], [0.2], [0.3], [1.0], [0.3], [0.2]],
        scales=[100.0],
    )
    initial = np.zeros(2 * len(problem.k_orig), dtype=float)
    calls = {"count": 0}

    def fake_minimize(fun, x0, jac, method, bounds, options):
        del fun, x0, jac, method, bounds, options
        calls["count"] += 1
        return _FakeMinimizeResult(initial)

    monkeypatch.setattr(torsion_stage2_module, "minimize", fake_minimize)

    vector_final, cycles = refine_torsion_scans_global(
        problem,
        delta_init=initial,
        enabled=True,
        max_block_iter=5,
        tol=1.0e-8,
    )

    assert calls["count"] == 1
    assert vector_final.tolist() == pytest.approx(initial.tolist())
    assert cycles[-1].accepted_blocks == 0
    assert cycles[-1].rejected_blocks == 1
    assert cycles[-1].diagnostics["status"] == "kept_stage1"
    assert "per_scan" not in cycles[-1].diagnostics


def test_stage2_keeps_smooth_high_barrier_refinable(monkeypatch):
    import maple.function.dispatcher.parmfit.utils.TorsionFit.fit as fit_module

    qm_rel = np.asarray([0.0, 5.0, 20.0, 5.0, 0.0, 5.0, 20.0, 5.0], dtype=float)
    basis = np.asarray([[0.0], [1.0], [1.0], [1.0], [0.0], [1.0], [1.0], [1.0]], dtype=float)
    candidate = np.asarray([0.2, 0.0], dtype=float)
    problem = _guard_problem(
        qm_rel=qm_rel.tolist(),
        constant_rel=(qm_rel - 0.2 * basis[:, 0]).tolist(),
        basis=basis.tolist(),
        scales=[100.0],
    )

    monkeypatch.setattr(torsion_stage2_module, "minimize", lambda *args, **kwargs: _FakeMinimizeResult(candidate))

    vector_final, cycles = refine_torsion_scans_global(
        problem,
        delta_init=np.zeros(2 * len(problem.k_orig), dtype=float),
        enabled=True,
        max_block_iter=5,
        tol=1.0e-8,
    )

    assert vector_final.tolist() == pytest.approx(candidate.tolist())
    assert cycles[-1].accepted_blocks == 1
    assert cycles[-1].diagnostics["status"] == "accepted"


def test_stage2_records_low_efficiency_large_k_update(monkeypatch):
    import maple.function.dispatcher.parmfit.utils.TorsionFit.fit as fit_module

    problem = _guard_problem(
        qm_rel=[0.0, 10.0, 0.0, 10.0],
        constant_rel=[0.0, 11.0, 0.0, 11.0],
        basis=[[0.0], [-0.1], [0.0], [-0.1]],
        scales=[100.0],
    )
    candidate = np.asarray([3.2, 0.0], dtype=float)
    before = evaluate_global_refit_objective(problem, np.zeros(2 * len(problem.k_orig), dtype=float))
    after = evaluate_global_refit_objective(problem, candidate)
    assert after.total_loss < before.total_loss

    monkeypatch.setattr(torsion_stage2_module, "minimize", lambda *args, **kwargs: _FakeMinimizeResult(candidate))

    vector_final, cycles = refine_torsion_scans_global(
        problem,
        delta_init=np.zeros(2 * len(problem.k_orig), dtype=float),
        enabled=True,
        max_block_iter=5,
        tol=1.0e-8,
    )

    assert vector_final.tolist() == pytest.approx([3.0, 0.0])
    assert cycles[-1].accepted_blocks == 1
    assert cycles[-1].diagnostics["reject_reason"] is None
    assert cycles[-1].diagnostics["status"] == "accepted"


def test_stage_weights_keep_high_energy_points_visible():
    weights = _scan_energy_weights(np.asarray([0.0, 2.0, 10.0, 50.0], dtype=float))

    assert weights[0] == pytest.approx(1.0)
    assert np.all(weights >= 0.20)
    assert weights[-1] > 0.20


def test_stage1_retains_high_energy_rows_with_weight_floor():
    qm_rel = np.asarray([0.0, 5.0, 20.0], dtype=float)
    scan_data = TorsionScanData(
        angles_deg=np.asarray([0.0, 120.0, 240.0], dtype=float),
        qm_hartree=qm_rel / HARTREE_TO_KCAL_MOL,
        qm_kcal=qm_rel,
        frames=[_four_atom_frame(float(angle)) for angle in (0.0, 120.0, 240.0)],
        source_path="stage1_weights.xyz",
        ref_idx=0,
        qm_rel=qm_rel,
    )
    problem = TorsionLocalProblem(
        center_bond=(2, 3),
        scan_data=scan_data,
        target_dihedrals=[],
        representative_dihedral=(1, 2, 3, 4),
        basis=np.ones((3, 1), dtype=float),
        qm_rel=qm_rel,
        orig_mm_rel=np.zeros(3, dtype=float),
        mm_zeroed_rel=np.zeros(3, dtype=float),
        residual=qm_rel,
        k_orig=np.zeros(1, dtype=float),
        scales=np.ones(1, dtype=float),
        active_mask=np.ones(1, dtype=bool),
        prior_weights=np.ones(1, dtype=float),
        shared_groups=(),
    )

    retained, weights = _stage1_retained_rows(problem, TorsionFitParams(enabled=True))

    assert retained.tolist() == [0, 1, 2]
    assert np.all(weights >= 0.20)


def test_stage2_normalized_loss_penalizes_low_scale_reversed_profile():
    problem = _guard_problem(
        qm_rel=[0.0, 0.10, 0.0, 0.10],
        constant_rel=[0.0, -0.10, 0.0, -0.10],
        basis=[[0.0], [0.0], [0.0], [0.0]],
    )

    evaluation = evaluate_global_refit_objective(problem, np.zeros(2 * len(problem.k_orig), dtype=float))

    assert evaluation.data_loss > 0.05


def test_stage_slots_use_one_phase_seed_per_period():
    parameter_set = _make_sparse_parameter_set([(0.80, 1.0, math.radians(45.0))])
    groups = torsion_problem_module._group_center_bond_dihedrals(
        parameter_set.dihedrals,
        parameter_set=parameter_set,
    )

    assert len(groups) == 1
    assert groups[0].slot_periods == (1,)
    assert len(groups[0].slot_indices) == 1
    assert math.degrees(groups[0].slot_phases[0]) == pytest.approx(45.0)
    assert groups[0].slot_sources == ("existing",)


def test_existing_noncanonical_phase_initialization_preserves_relative_profile(monkeypatch):
    monkeypatch.setattr(torsion_problem_module, "evaluate_mm_energy", _fake_mm_energy)

    phase = math.radians(45.0)
    k_value = 0.80
    parameter_set = _make_sparse_parameter_set([(k_value, 1.0, phase)])
    scan_data = _phase_shifted_scan(k_value, phase)

    problem = torsion_problem_module.build_local_torsion_problem(
        scan_data,
        parameter_set,
        center_bond=(2, 3),
    )
    direct_total = []
    dihedral = parameter_set.dihedrals[0]
    for atoms in scan_data.frames:
        phi = dihedral_radians(atoms.get_positions(), *dihedral.atoms)
        direct_total.append(k_value * (1.0 + math.cos(phi - phase)))
    direct_rel = np.asarray(direct_total, dtype=float) - float(direct_total[int(scan_data.ref_idx)])

    assert problem.shared_groups[0].slot_phases[0] == pytest.approx(phase)
    assert (problem.basis @ problem.k_orig).tolist() == pytest.approx(direct_rel.tolist())


def test_stage1_existing_term_uses_variable_phase_when_spectral_signal_is_clear(monkeypatch):
    import maple.function.dispatcher.parmfit.utils.TorsionFit.fit as fit_module

    monkeypatch.setattr(torsion_problem_module, "evaluate_mm_energy", _fake_mm_energy)
    monkeypatch.setattr(torsion_stage1_module, "evaluate_mm_energy", _fake_mm_energy)

    phase = 0.85
    scan_data = _phase_shifted_scan(1.20, phase)
    parameter_set = _make_sparse_parameter_set([(0.30, 1.0, 0.0)])

    result = fit_torsion_scan(
        scan_data,
        parameter_set,
        center_bond=(2, 3),
        params=TorsionFitParams(enabled=True, refine_rounds=0),
    )

    fitted = result.terms.fitted_terms[0]
    period1 = [term for term in fitted if int(term.period) == 1]
    assert len(period1) == 1
    assert period1[0].phase == pytest.approx(phase, abs=5.0e-2)
    assert "k1" in result.terms.shared_groups[0].active_slots


def test_stage1_candidate_expansion_uses_variable_phase(monkeypatch):
    import maple.function.dispatcher.parmfit.utils.TorsionFit.fit as fit_module

    monkeypatch.setattr(torsion_problem_module, "evaluate_mm_energy", _fake_mm_energy)
    monkeypatch.setattr(torsion_stage1_module, "evaluate_mm_energy", _fake_mm_energy)

    phase = 0.70
    scan_data = _period_phase_shifted_scan(2, 1.10, phase)
    parameter_set = _make_sparse_parameter_set([(0.05, 1.0, 0.0)])

    result = fit_torsion_scan(
        scan_data,
        parameter_set,
        center_bond=(2, 3),
        params=TorsionFitParams(enabled=True, refine_rounds=0),
    )

    fitted_period2 = [term for term in result.terms.fitted_terms[0] if int(term.period) == 2]
    assert len(fitted_period2) == 1
    assert fitted_period2[0].phase == pytest.approx(phase, abs=8.0e-2)
    assert "k2" in result.terms.shared_groups[0].active_slots


def test_stage1_geometry_jump_diagnostic_blocks_new_slot_expansion(monkeypatch):
    import maple.function.dispatcher.parmfit.utils.TorsionFit.fit as fit_module

    monkeypatch.setattr(torsion_problem_module, "evaluate_mm_energy", _fake_mm_energy)
    monkeypatch.setattr(torsion_stage1_module, "evaluate_mm_energy", _fake_mm_energy)

    phase = 0.70
    base_scan_data = _period_phase_shifted_scan(2, 1.10, phase)
    qm_rel = np.asarray(base_scan_data.qm_rel, dtype=float).copy()
    qm_rel[5] += 8.0
    scan_data = TorsionScanData(
        angles_deg=base_scan_data.angles_deg,
        qm_hartree=qm_rel / HARTREE_TO_KCAL_MOL,
        qm_kcal=qm_rel,
        frames=base_scan_data.frames,
        source_path=base_scan_data.source_path,
        ref_idx=base_scan_data.ref_idx,
        qm_rel=qm_rel,
    )
    parameter_set = _make_sparse_parameter_set([(0.05, 1.0, 0.0)])

    result = fit_torsion_scan(
        scan_data,
        parameter_set,
        center_bond=(2, 3),
        params=TorsionFitParams(enabled=True, refine_rounds=0),
    )

    assert "k2" not in result.terms.shared_groups[0].active_slots
    assert "geometry_jump" in result.terms.shared_groups[0].diagnostic_flags
    assert "stage1_fallback_original" not in result.terms.shared_groups[0].diagnostic_flags
    assert any(
        result.terms.fitted_terms[0][0].phase == pytest.approx(value, abs=1.0e-12)
        for value in (0.0, np.pi)
    )


def test_stage1_report_hides_trial_details_unless_debug_enabled(monkeypatch):
    import maple.function.dispatcher.parmfit.utils.TorsionFit.fit as fit_module

    monkeypatch.setattr(torsion_problem_module, "evaluate_mm_energy", _fake_mm_energy)
    monkeypatch.setattr(torsion_stage1_module, "evaluate_mm_energy", _fake_mm_energy)

    scan_data = _period_phase_shifted_scan(2, 1.10, 0.70)
    parameter_set = _make_sparse_parameter_set([(0.05, 1.0, 0.0)])
    base_report = fit_torsion_scan(
        scan_data,
        parameter_set,
        center_bond=(2, 3),
        params=TorsionFitParams(enabled=True, refine_rounds=0),
    )
    debug_report = fit_torsion_scan(
        scan_data,
        parameter_set,
        center_bond=(2, 3),
        params=TorsionFitParams(enabled=True, refine_rounds=0, report_debug=True),
    )

    base_text = "".join(format_torsion_fit_report(base_report))
    debug_text = "".join(format_torsion_fit_report(debug_report))
    assert "candidate_trials" not in base_text
    assert "rejected_reason" not in base_text
    assert "candidate_trials" in debug_text


def test_refine_torsion_scans_global_does_not_call_mm_engine_in_inner_loop(monkeypatch, tmp_path: Path):
    import maple.function.dispatcher.parmfit.utils.TorsionFit.fit as fit_module
    import maple.function.dispatcher.parmfit.utils.TorsionFit.basis as problem_module

    call_count = {"value": 0}

    def counted_mm_energy(*args, **kwargs):
        call_count["value"] += 1
        return _fake_mm_energy(*args, **kwargs)

    monkeypatch.setattr(problem_module, "evaluate_mm_energy", counted_mm_energy)
    monkeypatch.setattr(torsion_stage1_module, "evaluate_mm_energy", counted_mm_energy)

    scan_path = tmp_path / "scan.xyz"
    _write_scan_xyz(scan_path, (1.20, 0.20, 0.0, 0.0))
    scan_data = read_scan_xyz(str(scan_path))
    stage0_parameter_set = _make_parameter_set((0.30, 0.10, 0.0, 0.0))

    problem = build_global_torsion_problem(
        stage0_parameter_set,
        [(2, 3)],
        {(2, 3): scan_data},
        typed_shared=True,
        original_parameter_set=stage0_parameter_set,
        params=TorsionFitParams(enabled=True),
    )
    build_calls = call_count["value"]

    def fail_mm_energy(*args, **kwargs):
        raise AssertionError("Stage2 refine should stay on the precomputed basis and not call evaluate_mm_energy.")

    monkeypatch.setattr(torsion_stage1_module, "evaluate_mm_energy", fail_mm_energy)
    refine_torsion_scans_global(
        problem,
        delta_init=np.zeros(2 * len(problem.k_orig), dtype=float),
        enabled=True,
        max_block_iter=5,
        tol=1.0e-6,
    )

    assert build_calls > 0


def test_stage2_continuous_phase_profile_matches_direct_mm(monkeypatch, tmp_path: Path):
    import maple.function.dispatcher.parmfit.utils.TorsionFit.fit as fit_module
    import maple.function.dispatcher.parmfit.utils.TorsionFit.basis as problem_module

    monkeypatch.setattr(problem_module, "evaluate_mm_energy", _fake_mm_energy)
    monkeypatch.setattr(torsion_stage1_module, "evaluate_mm_energy", _fake_mm_energy)

    scan_path = tmp_path / "scan.xyz"
    _write_scan_xyz(scan_path, (1.20, 0.15, 0.0, 0.0))
    scan_data = read_scan_xyz(str(scan_path))
    parameter_set = _make_parameter_set((0.30, 0.10, 0.05, 0.02))

    problem = build_global_torsion_problem(
        parameter_set,
        [(2, 3)],
        {(2, 3): scan_data},
        typed_shared=True,
        original_parameter_set=parameter_set,
        params=TorsionFitParams(enabled=True),
    )
    trial = np.zeros(2 * len(problem.k_orig), dtype=float)
    n_terms = len(problem.k_orig)
    for global_index, period in enumerate(problem.period_orig):
        if problem.phase_orig[global_index] in (0.0, np.pi):
            target_k = {1: 0.75, 2: 0.25, 3: 0.05, 4: 0.02}[int(round(float(period)))]
            target_phase = float(problem.phase_orig[global_index]) + 0.37
            trial[global_index] = target_k * math.cos(target_phase) - problem.k_orig[global_index] * math.cos(problem.phase_orig[global_index])
            trial[n_terms + global_index] = target_k * math.sin(target_phase) - problem.k_orig[global_index] * math.sin(problem.phase_orig[global_index])

    fast_rel = _global_mm_rel_map(problem, trial)[(2, 3)]
    updated = apply_global_delta(problem, trial)
    direct_total = np.asarray([_fake_mm_energy(atoms, updated).total for atoms in scan_data.frames], dtype=float)
    direct_rel = direct_total - direct_total[int(scan_data.ref_idx)]

    assert fast_rel.tolist() == pytest.approx(direct_rel.tolist())


def test_stage2_objective_cache_matches_public_mm_map():
    problem = _guard_problem(
        qm_rel=[0.0, 0.7, 0.2, 1.1],
        constant_rel=[0.0, 0.1, -0.2, 0.3],
        basis=[[0.0, 0.0], [0.8, -0.3], [0.2, 0.5], [-0.4, 0.9]],
        scales=[1.2, 0.8],
    )
    vector = np.asarray([0.13, -0.27, 0.31, 0.19], dtype=float)
    cache = _build_stage2_objective_cache(problem)

    cached_map = _global_mm_rel_map(problem, vector, cache=cache)
    direct_map = _global_mm_rel_map(problem, vector)

    assert cached_map.keys() == direct_map.keys()
    assert cached_map[(2, 3)].tolist() == pytest.approx(direct_map[(2, 3)].tolist())
    assert evaluate_global_refit_objective(problem, vector).total_loss == pytest.approx(
        evaluate_global_refit_objective(problem, vector, cache=cache).total_loss
    )


def test_stage2_k_phase_objective_gradient_matches_finite_difference():
    problem = _guard_problem(
        qm_rel=[0.0, 0.7, 0.2, 1.1, 0.4],
        constant_rel=[0.0, 0.1, -0.2, 0.3, -0.1],
        basis=[[0.0, 0.0], [0.8, -0.3], [0.2, 0.5], [-0.4, 0.9], [0.6, 0.1]],
        scales=[1.2, 0.8],
    )
    vector = np.asarray([0.6, 0.4, 0.7, -0.3], dtype=float)
    cache = _build_stage2_objective_cache(problem)

    evaluation, gradient = _evaluate_stage2_k_phase_objective_with_gradient(problem, vector, cache=cache)

    epsilon = 1.0e-6
    finite_difference = np.zeros_like(vector)
    for index in range(vector.size):
        step = np.zeros_like(vector)
        step[index] = epsilon
        plus = _evaluate_stage2_k_phase_objective_with_gradient(problem, vector + step, cache=cache)[0].total_loss
        minus = _evaluate_stage2_k_phase_objective_with_gradient(problem, vector - step, cache=cache)[0].total_loss
        finite_difference[index] = (plus - minus) / (2.0 * epsilon)

    assert np.isfinite(evaluation.total_loss)
    assert gradient.tolist() == pytest.approx(finite_difference.tolist(), rel=1.0e-5, abs=1.0e-6)


def test_stage2_k_phase_pack_round_trips_coefficient_delta():
    problem = replace(
        _guard_problem(
            qm_rel=[0.0, 0.7],
            constant_rel=[0.0, 0.1],
            basis=[[0.0, 0.0], [0.8, -0.3]],
            scales=[1.2, 0.8],
        ),
        k_orig=np.asarray([0.3, 0.4], dtype=float),
        phase_orig=np.asarray([0.2, -0.5], dtype=float),
    )
    target = np.asarray([0.9, 0.2, 1.1, -2.0], dtype=float)

    delta = _delta_from_stage2_k_phase(problem, target)
    packed = _pack_stage2_k_phase(problem, delta)

    assert packed[:2].tolist() == pytest.approx(target[:2].tolist())
    assert packed[2:].tolist() == pytest.approx(target[2:].tolist())


def test_stage1_active_set_cache_matches_uncached_solver():
    qm_rel = np.asarray([0.0, 0.8, 0.2, 1.1, 0.3], dtype=float)
    basis = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.8, -0.2, 0.1],
            [0.2, 0.5, -0.3],
            [-0.4, 0.9, 0.2],
            [0.6, 0.1, 0.7],
        ],
        dtype=float,
    )
    scan_data = TorsionScanData(
        angles_deg=np.arange(qm_rel.size, dtype=float),
        qm_hartree=qm_rel / HARTREE_TO_KCAL_MOL,
        qm_kcal=qm_rel,
        frames=[_four_atom_frame(float(angle)) for angle in range(qm_rel.size)],
        source_path="stage1_cache.xyz",
        ref_idx=0,
        qm_rel=qm_rel,
    )
    problem = TorsionLocalProblem(
        center_bond=(2, 3),
        scan_data=scan_data,
        target_dihedrals=[],
        representative_dihedral=(1, 2, 3, 4),
        basis=basis,
        qm_rel=qm_rel,
        orig_mm_rel=np.zeros_like(qm_rel),
        mm_zeroed_rel=np.zeros_like(qm_rel),
        residual=qm_rel,
        k_orig=np.asarray([0.1, 0.2, 0.3], dtype=float),
        scales=np.asarray([1.0, 1.2, 0.9], dtype=float),
        active_mask=np.asarray([True, False, True], dtype=bool),
        prior_weights=np.asarray([6.0, 24.0, 6.0], dtype=float),
        shared_groups=(),
    )
    params = TorsionFitParams(enabled=True)
    active_mask = np.asarray([True, False, True], dtype=bool)
    cache = _build_local_stage1_solve_cache(problem, params)

    uncached = _solve_local_stage1_active_set(problem, active_mask, params)
    cached = _solve_local_stage1_active_set(problem, active_mask, params, solve_cache=cache)
    cached_again = _solve_local_stage1_active_set(problem, active_mask, params, solve_cache=cache)

    assert cached[0].tolist() == pytest.approx(uncached[0].tolist())
    assert cached[1] == uncached[1]
    assert cached[2] == uncached[2]
    assert cached[3] == pytest.approx(uncached[3])
    assert cached[4].tolist() == uncached[4].tolist()
    assert cached_again[0].tolist() == pytest.approx(cached[0].tolist())
    assert len(cache.solutions) == 1


def test_stage1_fixed_phase_negative_solution_flips_phase():
    qm_rel = np.asarray([0.0, -1.0], dtype=float)
    basis = np.asarray([[0.0], [1.0]], dtype=float)
    scan_data = TorsionScanData(
        angles_deg=np.asarray([0.0, 1.0], dtype=float),
        qm_hartree=qm_rel / HARTREE_TO_KCAL_MOL,
        qm_kcal=qm_rel,
        frames=[_four_atom_frame(0.0), _four_atom_frame(1.0)],
        source_path="stage1_negative_fixed_phase.xyz",
        ref_idx=0,
        qm_rel=qm_rel,
    )
    problem = TorsionLocalProblem(
        center_bond=(2, 3),
        scan_data=scan_data,
        target_dihedrals=[],
        representative_dihedral=(1, 2, 3, 4),
        basis=basis,
        qm_rel=qm_rel,
        orig_mm_rel=np.zeros_like(qm_rel),
        mm_zeroed_rel=np.zeros_like(qm_rel),
        residual=qm_rel,
        k_orig=np.asarray([0.0], dtype=float),
        scales=np.asarray([1.0], dtype=float),
        active_mask=np.asarray([True], dtype=bool),
        prior_weights=np.asarray([0.0], dtype=float),
        shared_groups=(),
        phase_orig=np.asarray([0.0], dtype=float),
        cos_basis=basis,
        sin_basis=np.zeros_like(basis),
    )

    solution = _solve_local_stage1_active_set_with_phases(
        problem,
        np.asarray([True], dtype=bool),
        TorsionFitParams(enabled=True),
    )

    assert solution.k_values[0] == pytest.approx(1.0)
    assert solution.phase_values[0] == pytest.approx(np.pi)
    profile = problem.mm_zeroed_rel + basis[:, 0] * solution.k_values[0] * math.cos(solution.phase_values[0])
    assert profile.tolist() == pytest.approx(qm_rel.tolist())


def test_stage1_does_not_fallback_to_original_when_fixed_lls_is_worse(monkeypatch):
    import maple.function.dispatcher.parmfit.utils.TorsionFit.fit as fit_module

    qm_rel = np.asarray([0.0, 0.5], dtype=float)
    basis = np.asarray([[0.0], [1.0]], dtype=float)
    scan_data = TorsionScanData(
        angles_deg=np.asarray([0.0, 1.0], dtype=float),
        qm_hartree=qm_rel / HARTREE_TO_KCAL_MOL,
        qm_kcal=qm_rel,
        frames=[_four_atom_frame(0.0), _four_atom_frame(1.0)],
        source_path="stage1_fallback.xyz",
        ref_idx=0,
        qm_rel=qm_rel,
    )
    problem = TorsionLocalProblem(
        center_bond=(2, 3),
        scan_data=scan_data,
        target_dihedrals=[],
        representative_dihedral=(1, 2, 3, 4),
        basis=basis,
        qm_rel=qm_rel,
        orig_mm_rel=qm_rel.copy(),
        mm_zeroed_rel=np.zeros_like(qm_rel),
        residual=qm_rel,
        k_orig=np.asarray([0.5], dtype=float),
        scales=np.asarray([1.0], dtype=float),
        active_mask=np.asarray([True], dtype=bool),
        prior_weights=np.asarray([6.0], dtype=float),
        shared_groups=(),
        phase_orig=np.asarray([0.0], dtype=float),
        cos_basis=basis,
        sin_basis=np.zeros_like(basis),
    )

    def fake_solve(local_problem, active_mask, params, *, solve_cache=None, variable_phase_mask=None):
        del local_problem, active_mask, params, solve_cache, variable_phase_mask
        return torsion_stage1_module._LocalStage1Solution(
            k_values=np.asarray([3.0], dtype=float),
            phase_values=np.asarray([0.0], dtype=float),
            cos_coeff=np.asarray([3.0], dtype=float),
            sin_coeff=np.asarray([0.0], dtype=float),
            rank=1,
            dropped=0,
            min_relative_sv=1.0,
            retained_rows=np.asarray([0, 1], dtype=int),
        )

    monkeypatch.setattr(torsion_stage1_module, "_solve_local_stage1_active_set_with_phases", fake_solve)

    delta, active_mask, diagnostics = _solve_local_problem_stage1(problem, TorsionFitParams(enabled=True))

    assert delta.tolist() == pytest.approx([2.5])
    assert active_mask.tolist() == [True]
    assert "stage1_fallback_original" not in diagnostics["_stage1"]["diagnostic_flags"]


def test_stage1_report_reuses_matrix_profile_without_refit_mm_calls(monkeypatch, tmp_path: Path):
    import maple.function.dispatcher.parmfit.utils.TorsionFit.fit as fit_module
    import maple.function.dispatcher.parmfit.utils.TorsionFit.basis as problem_module

    scan_path = tmp_path / "scan.xyz"
    _write_scan_xyz(scan_path, (1.20, 0.0, 0.0, 0.0))
    scan_data = read_scan_xyz(str(scan_path))
    parameter_set = _make_parameter_set((0.30, 0.0, 0.0, 0.0))
    call_count = {"count": 0}

    def counted_mm_energy(*args, **kwargs):
        call_count["count"] += 1
        return _fake_mm_energy(*args, **kwargs)

    monkeypatch.setattr(problem_module, "evaluate_mm_energy", counted_mm_energy)
    monkeypatch.setattr(torsion_stage1_module, "evaluate_mm_energy", counted_mm_energy)

    report = fit_torsion_scan(scan_data, parameter_set, center_bond=(2, 3), params=TorsionFitParams(enabled=True))

    assert call_count["count"] == len(scan_data.frames)
    assert report.curves.mm_stage1_rel.shape == report.curves.qm_rel.shape
    assert (report.curves.mm_stage1_rel - report.curves.mm_stage0_rel).shape == report.curves.qm_rel.shape


def test_stage1_can_activate_two_new_harmonics_for_high_signal_profile(monkeypatch, tmp_path: Path):
    import maple.function.dispatcher.parmfit.utils.TorsionFit.fit as fit_module
    import maple.function.dispatcher.parmfit.utils.TorsionFit.basis as problem_module

    monkeypatch.setattr(problem_module, "evaluate_mm_energy", _fake_mm_energy)
    monkeypatch.setattr(torsion_stage1_module, "evaluate_mm_energy", _fake_mm_energy)

    scan_path = tmp_path / "scan.xyz"
    _write_scan_xyz(scan_path, (0.95, 0.25, 0.85, 0.05))
    scan_data = read_scan_xyz(str(scan_path))
    parameter_set = _make_sparse_parameter_set([(0.25, 2.0, np.pi)])

    report = fit_torsion_scan(scan_data, parameter_set, center_bond=(2, 3))

    assert len(report.terms.shared_groups) == 1
    group = report.terms.shared_groups[0]
    original_periods = {int(term.period) for term in group.original_terms}
    fitted_periods = {int(term.period) for term in group.fitted_terms}
    assert 1 <= len(fitted_periods - original_periods) <= 3
    assert all(1 <= int(round(float(term.period))) <= 6 for term in group.fitted_terms)


def test_stage1_refits_existing_phase_when_spectral_signal_is_asymmetric(monkeypatch):
    import maple.function.dispatcher.parmfit.utils.TorsionFit.fit as fit_module
    import maple.function.dispatcher.parmfit.utils.TorsionFit.basis as problem_module

    monkeypatch.setattr(problem_module, "evaluate_mm_energy", _fake_mm_energy)
    monkeypatch.setattr(torsion_stage1_module, "evaluate_mm_energy", _fake_mm_energy)

    scan_data = _phase_shifted_scan(1.10, -np.pi / 6.0)
    parameter_set = _make_sparse_parameter_set([(0.60, 1.0, 0.0)])

    report = fit_torsion_scan(scan_data, parameter_set, center_bond=(2, 3))

    period1_terms = [
        term
        for group in report.terms.shared_groups
        for term in group.fitted_terms
        if int(round(float(term.period))) == 1 and abs(float(term.kPhi)) > 1.0e-6
    ]
    assert len(period1_terms) == 1
    assert period1_terms[0].phase == pytest.approx(-np.pi / 6.0, abs=5.0e-2)


def test_stage1_can_add_period6_spectral_candidate(monkeypatch):
    import maple.function.dispatcher.parmfit.utils.TorsionFit.fit as fit_module
    import maple.function.dispatcher.parmfit.utils.TorsionFit.basis as problem_module

    monkeypatch.setattr(problem_module, "evaluate_mm_energy", _fake_mm_energy)
    monkeypatch.setattr(torsion_stage1_module, "evaluate_mm_energy", _fake_mm_energy)

    phase = 0.65
    angles_deg = np.arange(0.0, 360.0, 15.0, dtype=float)
    frames = [_four_atom_frame(float(angle_deg)) for angle_deg in angles_deg]
    qm_kcal = np.zeros(len(frames), dtype=float)
    for index, atoms in enumerate(frames):
        phi = dihedral_radians(atoms.get_positions(), 1, 2, 3, 4)
        qm_kcal[index] = 1.1 * (1.0 + math.cos(6.0 * phi - phase))
    scan_data = TorsionScanData(
        angles_deg=angles_deg,
        qm_hartree=qm_kcal / HARTREE_TO_KCAL_MOL,
        qm_kcal=qm_kcal,
        frames=frames,
        source_path="synthetic_period6_phase_shifted.xyz",
        ref_idx=int(np.argmin(qm_kcal)),
        qm_rel=qm_kcal - qm_kcal[int(np.argmin(qm_kcal))],
    )
    parameter_set = _make_sparse_parameter_set([(0.0, 1.0, 0.0)])

    report = fit_torsion_scan(scan_data, parameter_set, center_bond=(2, 3), params=TorsionFitParams(enabled=True))

    period6_terms = [
        term
        for group in report.terms.shared_groups
        for term in group.fitted_terms
        if int(round(float(term.period))) == 6 and abs(float(term.kPhi)) > 1.0e-6
    ]
    assert len(period6_terms) == 1
    assert period6_terms[0].phase == pytest.approx(phase, abs=8.0e-2)


def test_stage1_never_keeps_two_phase_variants_for_same_period(monkeypatch):
    import maple.function.dispatcher.parmfit.utils.TorsionFit.fit as fit_module
    import maple.function.dispatcher.parmfit.utils.TorsionFit.basis as problem_module

    monkeypatch.setattr(problem_module, "evaluate_mm_energy", _fake_mm_energy)
    monkeypatch.setattr(torsion_stage1_module, "evaluate_mm_energy", _fake_mm_energy)

    scan_data = _phase_shifted_scan(1.10, np.pi / 6.0)
    parameter_set = _make_sparse_parameter_set([(0.60, 1.0, 0.0)])

    report = fit_torsion_scan(scan_data, parameter_set, center_bond=(2, 3))

    for group in report.terms.shared_groups:
        active_by_period: dict[int, int] = {}
        for term in group.fitted_terms:
            if abs(float(term.kPhi)) <= 1.0e-6:
                continue
            period = int(round(float(term.period)))
            active_by_period[period] = active_by_period.get(period, 0) + 1
        assert all(count == 1 for count in active_by_period.values())


def test_stage1_low_signal_center_can_activate_more_than_one_group_but_caps_each_group(monkeypatch):
    import maple.function.dispatcher.parmfit.utils.TorsionFit.fit as fit_module
    import maple.function.dispatcher.parmfit.utils.TorsionFit.basis as problem_module

    monkeypatch.setattr(problem_module, "evaluate_mm_energy", _fake_mm_energy)
    monkeypatch.setattr(torsion_stage1_module, "evaluate_mm_energy", _fake_mm_energy)

    scan_data = _methyl_like_scan(0.02, -0.02)
    parameter_set = _make_methyl_like_parameter_set()

    report = fit_torsion_scan(scan_data, parameter_set, center_bond=(3, 4))

    for group in report.terms.shared_groups:
        original_periods = {int(term.period) for term in group.original_terms}
        fitted_periods = {int(term.period) for term in group.fitted_terms}
        assert len(fitted_periods - original_periods) <= 3


def test_stage1_new_harmonic_limit_is_shared_across_whole_center():
    angles = np.arange(8.0, dtype=float)
    basis = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
            [0.0, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.0],
        ],
        dtype=float,
    )
    true_k = np.asarray([0.0, 1.2, 0.8, 0.0, 0.0, 1.1, 0.7, 0.0], dtype=float)
    qm_rel = basis @ true_k
    scan_data = TorsionScanData(
        angles_deg=angles,
        qm_hartree=qm_rel / HARTREE_TO_KCAL_MOL,
        qm_kcal=qm_rel,
        frames=[_four_atom_frame(float(angle)) for angle in angles],
        source_path="synthetic_two_group_stage1.xyz",
        ref_idx=0,
        qm_rel=qm_rel,
    )
    groups = (
        TorsionSharedGroupSpec(
            label="left",
            atom_types=("c", "c", "c", "o"),
            improper=False,
            dihedral_indices=(0,),
            instances=((1, 2, 3, 4),),
            slot_indices=(0, 1, 2, 3),
            slot_periods=(1, 2, 3, 4),
            slot_phases=(0.0, np.pi, 0.0, np.pi),
            existing_slot_mask=(True, False, False, False),
        ),
        TorsionSharedGroupSpec(
            label="right",
            atom_types=("n", "c", "c", "h"),
            improper=False,
            dihedral_indices=(1,),
            instances=((5, 2, 3, 4),),
            slot_indices=(4, 5, 6, 7),
            slot_periods=(1, 2, 3, 4),
            slot_phases=(0.0, np.pi, 0.0, np.pi),
            existing_slot_mask=(True, False, False, False),
        ),
    )
    problem = TorsionLocalProblem(
        center_bond=(2, 3),
        scan_data=scan_data,
        target_dihedrals=[],
        representative_dihedral=(1, 2, 3, 4),
        basis=basis,
        qm_rel=qm_rel,
        orig_mm_rel=np.zeros_like(qm_rel),
        mm_zeroed_rel=np.zeros_like(qm_rel),
        residual=qm_rel,
        k_orig=np.zeros(8, dtype=float),
        scales=np.ones(8, dtype=float),
        active_mask=np.asarray([True, False, False, False, True, False, False, False], dtype=bool),
        prior_weights=np.asarray([6.0, 24.0, 24.0, 24.0, 6.0, 24.0, 24.0, 24.0], dtype=float),
        shared_groups=groups,
    )

    _delta, active_mask, _diagnostics = _solve_local_problem_stage1(problem, TorsionFitParams(enabled=True))

    active_new_slots = [
        slot_index
        for group in groups
        for slot_index, existing in zip(group.slot_indices, group.existing_slot_mask)
        if active_mask[slot_index] and not existing
    ]
    assert len(active_new_slots) <= 3


def test_stage2_rejects_legacy_k_delta_vector(monkeypatch, tmp_path: Path):
    import maple.function.dispatcher.parmfit.utils.TorsionFit.fit as fit_module
    import maple.function.dispatcher.parmfit.utils.TorsionFit.basis as problem_module

    monkeypatch.setattr(problem_module, "evaluate_mm_energy", _fake_mm_energy)
    monkeypatch.setattr(torsion_stage1_module, "evaluate_mm_energy", _fake_mm_energy)

    scan_path = tmp_path / "scan.xyz"
    _write_scan_xyz(scan_path, (1.20, 0.15, 0.0, 0.0))
    scan_data = read_scan_xyz(str(scan_path))
    parameter_set = _make_sparse_parameter_set([(0.30, 1.0, 0.0)])
    problem = build_global_torsion_problem(
        parameter_set,
        [(2, 3)],
        {(2, 3): scan_data},
        typed_shared=True,
        original_parameter_set=parameter_set,
        params=TorsionFitParams(enabled=True),
    )

    with pytest.raises(ValueError, match="coefficient deltas"):
        _split_global_vector(problem, np.zeros(len(problem.k_orig), dtype=float))

    k_values, phase_values = _split_global_vector(problem, np.zeros(2 * len(problem.k_orig), dtype=float))

    assert k_values.shape == problem.k_orig.shape
    assert phase_values.shape == problem.phase_orig.shape


def test_stage2_optimizer_projects_k_phase_result_to_k_cap(monkeypatch):
    import maple.function.dispatcher.parmfit.utils.TorsionFit.fit as fit_module

    problem = _guard_problem(
        qm_rel=[0.0, 1.0],
        constant_rel=[0.0, 0.0],
        basis=[[0.0], [1.0]],
        scales=[100.0],
    )
    vector_init = np.zeros(2 * len(problem.k_orig), dtype=float)
    cache = _build_stage2_objective_cache(problem)
    bounds = _stage2_k_phase_bounds(problem)
    k_cap = float(bounds[0][1])
    illegal_result = np.asarray([2.0 * k_cap, 0.0], dtype=float)

    def fake_minimize(fun, x0, jac, method, bounds, options):
        del fun, x0, jac, method, bounds, options
        return _FakeMinimizeResult(illegal_result)

    monkeypatch.setattr(torsion_stage2_module, "minimize", fake_minimize)

    vector_final = _optimize_stage2_k_phase(
        problem,
        vector_init,
        max_iter=5,
        tol=1.0e-8,
        cache=cache,
    )
    k_values, _phase_values = _split_global_vector(problem, vector_final)

    assert k_values[0] <= k_cap + 1.0e-10
    assert k_values[0] > 0.0


def test_stage1_zero_placeholder_term_is_not_existing_active_slot(monkeypatch):
    import maple.function.dispatcher.parmfit.utils.TorsionFit.fit as fit_module
    import maple.function.dispatcher.parmfit.utils.TorsionFit.basis as problem_module

    monkeypatch.setattr(problem_module, "evaluate_mm_energy", _fake_mm_energy)
    monkeypatch.setattr(torsion_stage1_module, "evaluate_mm_energy", _fake_mm_energy)

    scan_data = _period_phase_shifted_scan(2, 6.0, 0.5)
    parameter_set = _make_sparse_parameter_set([(0.0, 2.0, np.pi)])
    problem = torsion_problem_module.build_local_torsion_problem(scan_data, parameter_set, (2, 3))

    period2_slots = [
        slot_index
        for group in problem.shared_groups
        for slot_index, period in zip(group.slot_indices, group.slot_periods)
        if int(period) == 2
    ]
    assert period2_slots
    assert not any(bool(problem.active_mask[slot_index]) for slot_index in period2_slots)


def test_stage1_caps_existing_fixed_phase_fit(monkeypatch):
    import maple.function.dispatcher.parmfit.utils.TorsionFit.fit as fit_module
    import maple.function.dispatcher.parmfit.utils.TorsionFit.basis as problem_module

    monkeypatch.setattr(problem_module, "evaluate_mm_energy", _fake_mm_energy)
    monkeypatch.setattr(torsion_stage1_module, "evaluate_mm_energy", _fake_mm_energy)

    scan_data = _phase_shifted_scan(6.0, 0.7)
    parameter_set = _make_sparse_parameter_set([(0.5, 1.0, 0.0)])
    report = fit_torsion_scan(scan_data, parameter_set, (2, 3), params=TorsionFitParams(enabled=True))

    fitted_k = [term.kPhi for terms in report.terms.fitted_terms for term in terms]
    assert fitted_k
    assert max(fitted_k) <= 3.0 + 1.0e-10


def test_stage1_report_and_writeback_cap_replaceable_solver_output(monkeypatch):
    import maple.function.dispatcher.parmfit.utils.TorsionFit.fit as fit_module

    scan_data = _phase_shifted_scan(1.0, 0.0)
    parameter_set = _make_sparse_parameter_set([(0.5, 1.0, 0.0)])

    def fake_local_solver(problem, params=None, *, return_problem=False):
        del params
        active_mask = np.asarray(problem.active_mask, dtype=bool).copy()
        delta = np.zeros_like(problem.k_orig)
        delta[np.flatnonzero(active_mask)[0]] = 8.5
        diagnostics = {
            "_stage1": {
                "phase_values": [0.0 for _ in problem.k_orig],
            }
        }
        if return_problem:
            return problem, delta, active_mask, diagnostics
        return delta, active_mask, diagnostics

    monkeypatch.setattr(fit_module, "_default_local_fit_solver", fake_local_solver)

    report = fit_torsion_scan(scan_data, parameter_set, center_bond=(2, 3), params=TorsionFitParams(enabled=True))
    updated = apply_fitted_torsion(report, parameter_set)

    assert report.terms.fitted_terms[0][0].kPhi == pytest.approx(3.0)
    assert updated.dihedrals[0].terms[0].kPhi == pytest.approx(3.0)


def test_stage1_caps_selected_frozen_non_template_terms(monkeypatch):
    import maple.function.dispatcher.parmfit.utils.TorsionFit.fit as fit_module

    monkeypatch.setattr(torsion_problem_module, "evaluate_mm_energy", _fake_mm_energy)
    monkeypatch.setattr(torsion_stage1_module, "evaluate_mm_energy", _fake_mm_energy)

    scan_data = _phase_shifted_scan(0.2, 0.0)
    parameter_set = _make_sparse_parameter_set([(0.2, 1.0, 0.0), (6.5, 6.0, 0.0)])

    report = fit_torsion_scan(scan_data, parameter_set, (2, 3), params=TorsionFitParams(enabled=True))
    updated = apply_fitted_torsion(report, parameter_set)

    report_period6 = next(term for term in report.terms.fitted_terms[0] if int(term.period) == 6)
    updated_period6 = next(term for term in updated.dihedrals[0].terms if int(term.period) == 6)
    assert report_period6.kPhi == pytest.approx(3.0)
    assert updated_period6.kPhi == pytest.approx(3.0)


def test_apply_global_delta_caps_target_terms_only():
    parameter_set = _make_parameter_set((4.5,))
    parameter_set.dihedrals.append(
        Dihedral(
            atoms=(4, 3, 2, 1),
            atom_types=("c", "c", "c", "c"),
            terms=[FourierTerm(kPhi=6.5, period=2.0, phase=np.pi)],
        )
    )
    base_problem = _guard_problem(
        qm_rel=[0.0, 1.0],
        constant_rel=[0.0, 0.0],
        basis=[[0.0], [1.0]],
    )
    problem = replace(
        base_problem,
        stage0_parameter_set=parameter_set,
        k_orig=np.asarray([4.5], dtype=float),
        phase_orig=np.asarray([0.0], dtype=float),
        period_orig=np.asarray([1.0], dtype=float),
        term_paths=((0, 0),),
    )

    final_parameter_set = apply_global_delta(problem, np.zeros(2, dtype=float))

    assert final_parameter_set.dihedrals[0].terms[0].kPhi <= 3.0 + 1.0e-10
    assert final_parameter_set.dihedrals[1].terms[0].kPhi == pytest.approx(6.5)


def test_stage2_continuously_refines_existing_phase_for_shifted_profile(monkeypatch):
    import maple.function.dispatcher.parmfit.utils.TorsionFit.fit as fit_module
    import maple.function.dispatcher.parmfit.utils.TorsionFit.basis as problem_module

    monkeypatch.setattr(problem_module, "evaluate_mm_energy", _fake_mm_energy)
    monkeypatch.setattr(torsion_stage1_module, "evaluate_mm_energy", _fake_mm_energy)

    scan_data = _phase_shifted_scan(1.10, 0.85)
    parameter_set = _make_sparse_parameter_set([(0.60, 1.0, 0.0)])
    problem = build_global_torsion_problem(
        parameter_set,
        [(2, 3)],
        {(2, 3): scan_data},
        typed_shared=True,
        original_parameter_set=parameter_set,
        params=TorsionFitParams(enabled=True),
    )
    vector_init = np.zeros(2 * len(problem.k_orig), dtype=float)
    before_eval = evaluate_global_refit_objective(problem, vector_init)

    vector_final, cycles = refine_torsion_scans_global(
        problem,
        vector_init,
        enabled=True,
        max_block_iter=30,
        tol=1.0e-8,
    )

    assert cycles
    final_parameter_set = apply_global_delta(problem, vector_final)
    after_eval = evaluate_global_refit_objective(problem, vector_final)
    refined_terms = [
        term
        for term in final_parameter_set.dihedrals[0].terms
        if int(round(float(term.period))) == 1 and abs(float(term.kPhi)) > 1.0e-6
    ]
    assert len(refined_terms) == 1
    assert after_eval.data_loss < before_eval.data_loss
    assert refined_terms[0].phase == pytest.approx(0.85, abs=0.20)


def test_format_torsion_fit_report_only_emits_public_curve_columns(tmp_path: Path, monkeypatch):
    import maple.function.dispatcher.parmfit.utils.TorsionFit.fit as fit_module
    import maple.function.dispatcher.parmfit.utils.TorsionFit.basis as problem_module

    monkeypatch.setattr(problem_module, "evaluate_mm_energy", _fake_mm_energy)
    monkeypatch.setattr(torsion_stage1_module, "evaluate_mm_energy", _fake_mm_energy)

    scan_path = tmp_path / "scan.xyz"
    _write_scan_xyz(scan_path, (1.20, 0.25, 0.0, 0.0))
    scan_data = read_scan_xyz(str(scan_path))
    original_parameter_set = _make_parameter_set((0.20, 0.0, 0.0, 0.0))
    stage0_parameter_set = _make_parameter_set((0.45, 0.0, 0.0, 0.0))

    log_lines: list[str] = []
    run_loss_mode(
        base_parameter_set=stage0_parameter_set,
        original_parameter_set=original_parameter_set,
        center_bonds=[(2, 3)],
        scan_data_map={(2, 3): scan_data},
        scan_xyz_map={(2, 3): str(scan_path)},
        params=TorsionFitParams(enabled=True),
        log_info=log_lines.extend,
    )

    text = "".join(log_lines)
    assert "MM_zeroed_rel" not in text
    assert " torsion_fit " not in text
    assert " residual_after" not in text
    assert "QM_ref" in text
    assert "MM_orig" in text
    assert "MM_stage0" in text
    assert "MM_stage1" in text
    assert "MM_stage2" in text


def test_stage2_fit_report_allows_new_template_slots_without_shape_mismatch():
    stage1_report = TorsionFitReport(
        center_bond=(2, 3),
        target_dihedrals=[Dihedral(atoms=(1, 2, 3, 4), atom_types=("c", "c", "c", "c"), terms=[FourierTerm(0.4, 1.0, 0.0)])],
        representative_dihedral=(1, 2, 3, 4),
        scan_source_path="scan.xyz",
        terms=TorsionFitTerms(
            original_terms=[[FourierTerm(0.4, 1.0, 0.0)]],
            fitted_terms=[[FourierTerm(0.8, 1.0, 0.0)]],
            delta_kphi=np.asarray([0.4], dtype=float),
        ),
        curves=TorsionFitCurves(
            angles_deg=np.asarray([0.0, 180.0], dtype=float),
            qm_rel=np.asarray([0.0, 1.0], dtype=float),
            mm_orig_rel=np.asarray([0.0, 0.2], dtype=float),
            mm_stage0_rel=np.asarray([0.0, 0.4], dtype=float),
            mm_stage1_rel=np.asarray([0.0, 0.8], dtype=float),
        ),
        metrics=TorsionFitMetrics(
            residual_before=np.asarray([0.0, 0.1], dtype=float),
            residual_after=np.asarray([0.0, 0.2], dtype=float),
            rmse=0.1,
            mae=0.1,
            max_abs_error=0.1,
        ),
    )
    final_parameter_set = _make_parameter_set((0.8, 0.2, 0.0, 0.0))

    report = _stage2_fit_report_from_stage1(
        stage1_report,
        final_parameter_set,
        mm_stage2_rel=np.asarray([0.0, 0.95], dtype=float),
    )

    assert report.terms.delta_kphi.shape == (4,)
    assert report.terms.delta_kphi.tolist() == pytest.approx([0.4, 0.2, 0.0, 0.0])


def test_fit_torsion_scan_splits_methyl_like_shared_group(monkeypatch):
    import maple.function.dispatcher.parmfit.utils.TorsionFit.fit as fit_module
    import maple.function.dispatcher.parmfit.utils.TorsionFit.basis as problem_module

    monkeypatch.setattr(problem_module, "evaluate_mm_energy", _fake_mm_energy)
    monkeypatch.setattr(torsion_stage1_module, "evaluate_mm_energy", _fake_mm_energy)

    scan_data = _methyl_like_scan(0.20, -0.20)
    parameter_set = _make_methyl_like_parameter_set()

    report = fit_torsion_scan(scan_data, parameter_set, center_bond=(3, 4))

    assert len(report.terms.shared_groups) == 2
    assert all("[methyl-split outer=" in group.label for group in report.terms.shared_groups)
    assert all(len(group.instances) == 3 for group in report.terms.shared_groups)
    assert any(any(int(term.period) == 3 for term in group.fitted_terms) for group in report.terms.shared_groups)


def test_grouped_global_problem_preserves_stage1_split_subgroups(monkeypatch):
    import maple.function.dispatcher.parmfit.utils.TorsionFit.fit as fit_module
    import maple.function.dispatcher.parmfit.utils.TorsionFit.basis as problem_module

    monkeypatch.setattr(problem_module, "evaluate_mm_energy", _fake_mm_energy)
    monkeypatch.setattr(torsion_stage1_module, "evaluate_mm_energy", _fake_mm_energy)

    zero_parameter_set = _make_methyl_like_parameter_set()
    split_parameter_set = _make_methyl_like_parameter_set(
        per_dihedral_terms=[
            [FourierTerm(0.20, 3.0, 0.0)],
            [FourierTerm(0.20, 3.0, 0.0)],
            [FourierTerm(0.20, 3.0, 0.0)],
            [FourierTerm(-0.20, 3.0, 0.0)],
            [FourierTerm(-0.20, 3.0, 0.0)],
            [FourierTerm(-0.20, 3.0, 0.0)],
        ]
    )

    problem = build_global_torsion_problem(
        split_parameter_set,
        [(3, 4)],
        {(3, 4): _methyl_like_scan(0.0, 0.0)},
        typed_shared=True,
        original_parameter_set=zero_parameter_set,
        params=TorsionFitParams(enabled=True),
    )

    assert len(problem.shared_groups_map[(3, 4)]) == 2


def test_run_torsion_workflow_filters_loss_mode_scan_points(tmp_path: Path, monkeypatch):
    parameter_set = _make_parameter_set((0.5, 0.0, 0.0, 0.0))
    atoms = _four_atom_frame(0.0)
    _, scan_data, mm_total_by_frame = _workflow_filter_scan_data()
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "maple.function.dispatcher.parmfit.utils.TorsionFit.workflow.resolve_torsion_center_bonds",
        lambda parameter_set, params, topology_cache=None: ([(2, 3)], []),
    )
    monkeypatch.setattr(
        "maple.function.dispatcher.parmfit.utils.TorsionFit.workflow.representative_dihedral_for_center_bond",
        lambda parameter_set, center_bond, topology_cache=None: types.SimpleNamespace(atoms=(1, 2, 3, 4)),
    )
    monkeypatch.setattr(
        "maple.function.dispatcher.parmfit.utils.TorsionFit.workflow._run_center_bond_scan",
        lambda *args, **kwargs: str(tmp_path / "fake.xyz"),
    )
    monkeypatch.setattr("maple.function.dispatcher.parmfit.utils.TorsionFit.workflow.read_scan_xyz", lambda path: scan_data)
    monkeypatch.setattr(
        "maple.function.dispatcher.parmfit.utils.TorsionFit.workflow.evaluate_mm_energy",
        lambda atoms, parameter_set, topology_cache=None, zero_proper_center_bond=None: types.SimpleNamespace(
            total=mm_total_by_frame[id(atoms)]
        ),
    )

    def fake_run_loss_mode(
        *,
        base_parameter_set,
        center_bonds,
        scan_data_map,
        scan_xyz_map,
        scan_mm_orig_rel_map=None,
        params,
        topology_cache=None,
        original_parameter_set=None,
        log_info=None,
        ensemble_result=None,
    ):
        del base_parameter_set, scan_xyz_map, scan_mm_orig_rel_map, params, topology_cache
        del original_parameter_set, log_info, ensemble_result
        filtered = scan_data_map[(2, 3)]
        captured["angles_deg"] = filtered.angles_deg.copy()
        captured["qm_rel"] = filtered.qm_rel.copy()
        captured["ref_idx"] = filtered.ref_idx
        return TorsionWorkflowResult(
            stage1_parameter_set=parameter_set,
            final_parameter_set=parameter_set,
            refine_cycles=[],
            scan_xyz={},
            center_bonds=list(center_bonds),
            warnings=[],
        )

    monkeypatch.setattr("maple.function.dispatcher.parmfit.utils.TorsionFit.workflow.run_loss_mode", fake_run_loss_mode)

    result = run_torsion_workflow(
        atoms=atoms,
        output=str(tmp_path / "loss_filter.out"),
        parameter_set=parameter_set,
        params=TorsionFitParams(enabled=True),
        runtime=TorsionScanRuntime(max_iter=5, memory=5, curvature=70.0, max_step=0.2),
    )

    assert tuple(captured["angles_deg"]) == pytest.approx((60.0, 120.0))
    assert tuple(captured["qm_rel"]) == pytest.approx((0.0, 2.0))
    assert captured["ref_idx"] == 0
    assert any("center bond (2, 3)" in warning for warning in result.warnings)
    assert any("0.0000" in warning for warning in result.warnings)


def test_run_torsion_workflow_writes_filtered_point_table(tmp_path: Path, monkeypatch):
    parameter_set = _make_parameter_set((0.5, 0.0, 0.0, 0.0))
    atoms = _four_atom_frame(0.0)
    _, scan_data, mm_total_by_frame = _workflow_filter_scan_data()

    monkeypatch.setattr(
        "maple.function.dispatcher.parmfit.utils.TorsionFit.workflow.resolve_torsion_center_bonds",
        lambda parameter_set, params, topology_cache=None: ([(2, 3)], []),
    )
    monkeypatch.setattr(
        "maple.function.dispatcher.parmfit.utils.TorsionFit.workflow.representative_dihedral_for_center_bond",
        lambda parameter_set, center_bond, topology_cache=None: types.SimpleNamespace(atoms=(1, 2, 3, 4)),
    )
    monkeypatch.setattr(
        "maple.function.dispatcher.parmfit.utils.TorsionFit.workflow._run_center_bond_scan",
        lambda *args, **kwargs: str(tmp_path / "fake.xyz"),
    )
    monkeypatch.setattr("maple.function.dispatcher.parmfit.utils.TorsionFit.workflow.read_scan_xyz", lambda path: scan_data)
    monkeypatch.setattr(
        "maple.function.dispatcher.parmfit.utils.TorsionFit.workflow.evaluate_mm_energy",
        lambda atoms, parameter_set, topology_cache=None, zero_proper_center_bond=None: types.SimpleNamespace(
            total=mm_total_by_frame[id(atoms)]
        ),
    )

    def fake_run_loss_mode(
        *,
        base_parameter_set,
        center_bonds,
        scan_data_map,
        scan_xyz_map,
        scan_mm_orig_rel_map=None,
        params,
        topology_cache=None,
        original_parameter_set=None,
        log_info=None,
        ensemble_result=None,
    ):
        del scan_xyz_map, scan_mm_orig_rel_map, params, topology_cache, original_parameter_set, ensemble_result
        filtered = scan_data_map[(2, 3)]
        report = TorsionFitReport(
            center_bond=(2, 3),
            target_dihedrals=deepcopy(base_parameter_set.dihedrals),
            representative_dihedral=base_parameter_set.dihedrals[0].atoms,
            scan_source_path=filtered.source_path,
            terms=TorsionFitTerms(
                original_terms=[deepcopy(dihedral.terms) for dihedral in base_parameter_set.dihedrals],
                fitted_terms=[deepcopy(dihedral.terms) for dihedral in base_parameter_set.dihedrals],
                delta_kphi=np.zeros(1, dtype=float),
            ),
            curves=TorsionFitCurves(
                angles_deg=np.asarray(filtered.angles_deg, dtype=float).copy(),
                qm_rel=np.asarray(filtered.qm_rel, dtype=float).copy(),
                mm_orig_rel=np.zeros(len(filtered.angles_deg), dtype=float),
                mm_stage0_rel=np.zeros(len(filtered.angles_deg), dtype=float),
                mm_stage1_rel=np.zeros(len(filtered.angles_deg), dtype=float),
                mm_stage2_rel=np.zeros(len(filtered.angles_deg), dtype=float),
            ),
            metrics=TorsionFitMetrics(
                residual_before=np.zeros(len(filtered.angles_deg), dtype=float),
                residual_after=np.zeros(len(filtered.angles_deg), dtype=float),
                rmse=0.0,
                mae=0.0,
                max_abs_error=0.0,
            ),
        )
        if log_info is not None:
            log_info(format_torsion_fit_report(report))
        return TorsionWorkflowResult(
            stage1_parameter_set=base_parameter_set,
            final_parameter_set=base_parameter_set,
            refine_cycles=[],
            scan_xyz={},
            center_bonds=list(center_bonds),
            warnings=[],
        )

    monkeypatch.setattr("maple.function.dispatcher.parmfit.utils.TorsionFit.workflow.run_loss_mode", fake_run_loss_mode)

    log_lines: list[str] = []
    run_torsion_workflow(
        atoms=atoms,
        output=str(tmp_path / "loss_filtered_output.out"),
        parameter_set=parameter_set,
        params=TorsionFitParams(enabled=True, refine_rounds=0),
        runtime=TorsionScanRuntime(max_iter=5, memory=5, curvature=70.0, max_step=0.2),
        log_info=log_lines.extend,
    )

    text = "".join(log_lines)
    marker = " angle_deg      QM_ref      MM_orig      MM_stage0    MM_stage1    MM_stage2\n"
    tail = text.split(marker, 1)[1]
    data_lines = [line for line in tail.splitlines() if line.lstrip().startswith(("60.0000", "120.0000"))]
    assert len(data_lines) == 2
    assert data_lines[0].lstrip().startswith("60.0000")
    assert data_lines[1].lstrip().startswith("120.0000")
