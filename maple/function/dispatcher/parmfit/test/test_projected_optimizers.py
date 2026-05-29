from __future__ import annotations

import importlib
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

    class Atoms:  # type: ignore[override]
        def __init__(self, symbols, positions):
            self.symbols = list(symbols)
            self.positions = np.asarray(positions, dtype=float)
            self.info = {}
            self.calc = None

        def __len__(self):
            return len(self.symbols)

        def copy(self):
            atoms = Atoms(self.symbols[:], self.positions.copy())
            atoms.info = dict(self.info)
            atoms.calc = self.calc
            return atoms

        def get_positions(self):
            return np.asarray(self.positions, dtype=float)

        def set_positions(self, positions):
            self.positions = np.asarray(positions, dtype=float)

        def get_chemical_symbols(self):
            return self.symbols[:]

    ase_stub.Atoms = Atoms
    sys.modules["ase"] = ase_stub


class _AtomProxy:
    __slots__ = ("symbol", "position")

    def __init__(self, symbol, position):
        self.symbol = symbol
        self.position = np.asarray(position, dtype=float)


if not hasattr(Atoms, "get_forces"):
    def _get_forces(self):
        if getattr(self, "calc", None) is None:
            raise AttributeError("atoms.calc is required")
        return self.calc.get_forces(self)

    Atoms.get_forces = _get_forces  # type: ignore[attr-defined]

if not hasattr(Atoms, "get_potential_energy"):
    def _get_potential_energy(self, force_consistent=True):
        if getattr(self, "calc", None) is None:
            raise AttributeError("atoms.calc is required")
        return self.calc.get_potential_energy(self, force_consistent=force_consistent)

    Atoms.get_potential_energy = _get_potential_energy  # type: ignore[attr-defined]

if not hasattr(Atoms, "__iter__"):
    def _iter_atoms(self):
        for symbol, position in zip(self.get_chemical_symbols(), self.get_positions()):
            yield _AtomProxy(symbol, position)

    Atoms.__iter__ = _iter_atoms  # type: ignore[attr-defined]


ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maple.function.dispatcher.parmfit.correction import config as correction_config_module
from maple.function.dispatcher.parmfit.utils import runtime as runtime_module
from maple.function.dispatcher.parmfit.utils.Scan import optimizer as projected_opt_module
from maple.function.dispatcher.parmfit.utils.Scan import optimizer as cgws_module
from maple.function.dispatcher.parmfit.utils.Scan import engine as scan_engine_module
from maple.function.dispatcher.parmfit.utils.Scan import optimizer as projected_lbfgs_module
from maple.function.dispatcher.parmfit.utils.Scan.optimizer import CGWS
from maple.function.dispatcher.parmfit.utils.Scan.optimizer import LBFGS as ProjectedLBFGS


def _make_atoms(positions, calc=None):
    atoms = Atoms(symbols=["C"] * len(positions), positions=np.asarray(positions, dtype=float))
    atoms.calc = QuadraticCalculator() if calc is None else calc
    atoms.f_max_th = 1.0e-8
    atoms.f_rms_th = 1.0e-8
    atoms.dp_max_th = 1.0e-8
    atoms.dp_rms_th = 1.0e-8
    return atoms


class QuadraticCalculator:
    def __init__(self, stiffness: float = 1.0):
        self.stiffness = float(stiffness)

    def get_forces(self, atoms):
        positions = np.asarray(atoms.get_positions(), dtype=float)
        return -self.stiffness * positions

    def get_potential_energy(self, atoms, force_consistent=True):
        positions = np.asarray(atoms.get_positions(), dtype=float)
        del force_consistent
        return 0.5 * self.stiffness * float(np.sum(positions ** 2))


class ShiftedQuadraticCalculator:
    def __init__(self, target, stiffness: float = 1.0):
        self.target = np.asarray(target, dtype=float)
        self.stiffness = float(stiffness)

    def get_forces(self, atoms):
        positions = np.asarray(atoms.get_positions(), dtype=float)
        return -self.stiffness * (positions - self.target)

    def get_potential_energy(self, atoms, force_consistent=True):
        positions = np.asarray(atoms.get_positions(), dtype=float)
        del force_consistent
        delta = positions - self.target
        return 0.5 * self.stiffness * float(np.sum(delta ** 2))


class LinearXCalculator:
    def get_forces(self, atoms):
        forces = np.zeros_like(np.asarray(atoms.get_positions(), dtype=float))
        forces[0, 0] = -1.0
        return forces

    def get_potential_energy(self, atoms, force_consistent=True):
        del force_consistent
        return float(np.asarray(atoms.get_positions(), dtype=float)[0, 0])


class StepBarrierXCalculator:
    def __init__(self, barrier_x: float, stiffness: float = 100.0, penalty: float = 1.0):
        self.barrier_x = float(barrier_x)
        self.stiffness = float(stiffness)
        self.penalty = float(penalty)

    def get_forces(self, atoms):
        positions = np.asarray(atoms.get_positions(), dtype=float)
        forces = np.zeros_like(positions)
        forces[0, 0] = -self.stiffness * (positions[0, 0] - self.barrier_x)
        return forces

    def get_potential_energy(self, atoms, force_consistent=True):
        del force_consistent
        x_value = float(np.asarray(atoms.get_positions(), dtype=float)[0, 0])
        dx = x_value - self.barrier_x
        return 0.5 * self.stiffness * dx * dx + (self.penalty if x_value < self.barrier_x else 0.0)


def test_parmfit_runtime_and_config_use_scan_lbfgs_types():
    assert runtime_module.LBFGS is ProjectedLBFGS
    assert runtime_module.LBFGSParams is projected_lbfgs_module.LBFGSParams
    assert correction_config_module.LBFGSParams is projected_lbfgs_module.LBFGSParams


def test_optimize_atoms_geometry_calls_scan_lbfgs(monkeypatch, tmp_path: Path):
    calls = {}

    class FakeLBFGS:
        def __init__(self, atoms, output, params=None, paras=None):
            calls["atoms"] = atoms
            calls["output"] = output
            calls["params"] = params
            calls["paras"] = paras
            self.atoms = atoms
            self.params = params
            self.converged = True

        def run(self):
            calls["ran"] = True
            return self.atoms

    monkeypatch.setattr(runtime_module, "LBFGS", FakeLBFGS)
    atoms = _make_atoms([(1.0, 0.0, 0.0)])

    result = runtime_module.optimize_atoms_geometry(
        atoms,
        output=str(tmp_path / "geom.out"),
        max_iter=7,
        max_step=0.12,
    )

    assert result is atoms
    assert calls["ran"] is True
    assert calls["atoms"] is atoms
    assert calls["output"] == str(tmp_path / "geom.out")
    assert calls["paras"] is None
    assert calls["params"].max_iter == 7
    assert calls["params"].max_step == pytest.approx(0.12)
    assert calls["params"].verbose == 0
    assert calls["params"].write_traj is False
    assert calls["params"].use_projection is False
    assert calls["params"].use_line_search is False


def test_fixinternals_scan_lbfgs_uses_scan_optimizer(monkeypatch, tmp_path: Path):
    calls = {}

    class FakeLBFGS:
        def __init__(self, atoms, output, paras=None):
            calls["atoms"] = atoms
            calls["output"] = output
            calls["paras"] = paras
            self.atoms = atoms

        def run(self):
            calls["ran"] = True
            return self.atoms

    monkeypatch.setattr(projected_lbfgs_module, "LBFGS", FakeLBFGS)
    atoms = _make_atoms([(1.0, 0.0, 0.0)])
    engine = scan_engine_module.SilentScanEngine(
        output=str(tmp_path / "scan.out"),
        atoms=atoms,
        constraints=[],
        params={"backend": "lbfgs", "opt": {"max_iter": 3}},
        method="lbfgs",
        constraint_mode="fixinternals",
    )

    result = engine._run_fixinternals_optimizer(atoms)

    assert result is atoms
    assert calls["ran"] is True
    assert calls["atoms"] is atoms
    assert calls["output"] == str(tmp_path / "scan.out")
    assert calls["paras"]["opt"]["max_iter"] == 3
    assert calls["paras"]["opt"]["verbose"] == 0
    assert calls["paras"]["opt"]["write_traj"] is False
    assert calls["paras"]["opt"]["use_projection"] is False


def _as_positions(coords):
    coords = np.asarray(coords, dtype=float)
    return coords.squeeze()


def _unit(vector):
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if norm == 0.0:
        raise AssertionError("degenerate frame")
    return vector / norm


def _angle_local_frame(coords, i, j, k):
    positions = _as_positions(coords)
    x_axis = _unit(positions[k] - positions[j])
    y_raw = positions[i] - positions[j]
    y_raw = y_raw - np.dot(y_raw, x_axis) * x_axis
    y_axis = _unit(y_raw)
    z_axis = _unit(np.cross(x_axis, y_axis))
    y_axis = _unit(np.cross(z_axis, x_axis))
    return np.vstack([x_axis, y_axis, z_axis])


def _torsion_local_frame(coords, i, j, k, l):
    positions = _as_positions(coords)
    x_axis = _unit(positions[k] - positions[j])
    plane_normal = np.cross(positions[j] - positions[i], positions[k] - positions[j])
    if np.linalg.norm(plane_normal) < 1.0e-12:
        plane_normal = np.cross(x_axis, positions[l] - positions[k])
    z_axis = _unit(plane_normal)
    y_axis = _unit(np.cross(z_axis, x_axis))
    z_axis = _unit(np.cross(x_axis, y_axis))
    return np.vstack([x_axis, y_axis, z_axis])


def _angle_deg(coords, i, j, k):
    positions = _as_positions(coords)
    v1 = positions[i] - positions[j]
    v2 = positions[k] - positions[j]
    cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cosang = np.clip(cosang, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def _dihedral_deg(coords, i, j, k, l):
    positions = _as_positions(coords)
    p0, p1, p2, p3 = positions[[i, j, k, l]]
    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    b1 /= np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x_val = np.dot(v, w)
    y_val = np.dot(np.cross(b1, v), w)
    return float(np.degrees(np.arctan2(y_val, x_val)))


def _distance_ang(coords, i, j):
    positions = _as_positions(coords)
    return float(np.linalg.norm(positions[i] - positions[j]))


def _resolve_projector(module):
    for name in ("project_force", "force_projection", "projected_force", "apply_force_projection"):
        fn = getattr(module, name, None)
        if callable(fn):
            return fn
    raise AssertionError("projection helper not found")


def _call_projector(project_fn, coordinates, force, bond_constraints=(), angle_constraints=(), torsion_constraints=()):
    coord_variants = [
        np.asarray(coordinates, dtype=float).copy(),
        np.asarray(coordinates, dtype=float).squeeze().copy(),
    ]
    force_variants = [
        np.asarray(force, dtype=float).copy(),
        np.asarray(force, dtype=float).squeeze().copy(),
    ]
    constraint_variants = [
        (
            list(bond_constraints),
            list(angle_constraints),
            list(torsion_constraints),
        ),
        (
            np.asarray(bond_constraints, dtype=int).reshape(-1) if bond_constraints else np.asarray([], dtype=int),
            np.asarray(angle_constraints, dtype=int).reshape(-1) if angle_constraints else np.asarray([], dtype=int),
            np.asarray(torsion_constraints, dtype=int).reshape(-1) if torsion_constraints else np.asarray([], dtype=int),
        ),
    ]

    attempts = []
    for coords in coord_variants:
        for force_arr in force_variants:
            for bonds, angles, torsions in constraint_variants:
                attempts.extend(
                    [
                        lambda c=coords, f=force_arr, b=bonds, a=angles, t=torsions: project_fn(
                            force=f.copy(),
                            coordinates=c.copy(),
                            bond_constraints=b,
                            angle_constraints=a,
                            torsion_constraints=t,
                        ),
                        lambda c=coords, f=force_arr, b=bonds, a=angles, t=torsions: project_fn(
                            f.copy(),
                            c.copy(),
                            bond_constraints=b,
                            angle_constraints=a,
                            torsion_constraints=t,
                        ),
                        lambda c=coords, f=force_arr, b=bonds, a=angles, t=torsions: project_fn(
                            b,
                            a,
                            t,
                            f.copy(),
                            c.copy(),
                        ),
                    ]
                )

    last_error = None
    for attempt in attempts:
        try:
            projected = attempt()
        except (TypeError, IndexError, AttributeError) as exc:
            last_error = exc
            continue
        if projected is None:
            projected = force
        projected = np.asarray(projected, dtype=float).squeeze()
        return projected

    raise AssertionError("projection helper did not accept any expected call pattern") from last_error


PROJECT_FORCE = _resolve_projector(projected_opt_module)


def _require_cgbs_class():
    try:
        module = importlib.import_module("maple.function.dispatcher.parmfit.utils.Scan.optimizer")
    except ModuleNotFoundError:
        pytest.fail("Scan optimizer module is missing: maple.function.dispatcher.parmfit.utils.Scan.optimizer")
    optimizer_cls = getattr(module, "CGBS", None)
    if optimizer_cls is None:
        pytest.fail("CGBS optimizer class is missing from maple.function.dispatcher.parmfit.utils.Scan.optimizer")
    return optimizer_cls


def _load_xyz_frames(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    frames = []
    index = 0
    while index < len(lines):
        natoms = int(lines[index].strip())
        coords = []
        for line in lines[index + 2:index + 2 + natoms]:
            _symbol, x_val, y_val, z_val = line.split()
            coords.append((float(x_val), float(y_val), float(z_val)))
        frames.append(np.asarray(coords, dtype=float))
        index += natoms + 2
    return frames


def _canonical_angle_frame():
    coords = np.array(
        [
            [[0.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0]],
        ],
        dtype=float,
    ).reshape(1, 3, 3)
    force = np.array(
        [
            [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0], [8.0, 9.0, 10.0]],
        ],
        dtype=float,
    )
    return coords, force


def _canonical_torsion_frame():
    coords = np.array(
        [
            [[0.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0]],
            [[1.0, 1.0, 0.0]],
        ],
        dtype=float,
    ).reshape(1, 4, 3)
    force = np.array(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        ],
        dtype=float,
    )
    return coords, force


def test_project_force_zeroes_angle_forbidden_local_components():
    coords, force = _canonical_angle_frame()
    projected = _call_projector(PROJECT_FORCE, coords, force, angle_constraints=[(1, 2, 3)])
    local_i, local_j, local_k = projected_opt_module.angle_local_components(
        _as_positions(coords),
        projected,
        (0, 1, 2),
    )

    np.testing.assert_allclose(local_i[1:], 0.0, atol=1.0e-10)
    np.testing.assert_allclose(local_j, 0.0, atol=1.0e-10)
    np.testing.assert_allclose(local_k[1:], 0.0, atol=1.0e-10)


def test_project_force_zeroes_torsion_forbidden_local_components():
    coords, force = _canonical_torsion_frame()
    projected = _call_projector(PROJECT_FORCE, coords, force, torsion_constraints=[(1, 2, 3, 4)])
    local_a, local_b, local_c, local_d = projected_opt_module.torsion_local_components(
        _as_positions(coords),
        projected,
        (0, 1, 2, 3),
    )

    np.testing.assert_allclose(local_a[2], 0.0, atol=1.0e-10)
    np.testing.assert_allclose(local_b[1:], 0.0, atol=1.0e-10)
    np.testing.assert_allclose(local_c[1:], 0.0, atol=1.0e-10)
    np.testing.assert_allclose(local_d[2], 0.0, atol=1.0e-10)


def test_project_force_angle_projection_keeps_only_ref_allowed_components():
    coords, force = _canonical_angle_frame()
    projected = _call_projector(PROJECT_FORCE, coords, force, angle_constraints=[(1, 2, 3)])
    original_i, original_j, original_k = projected_opt_module.angle_local_components(
        _as_positions(coords),
        force.squeeze(),
        (0, 1, 2),
    )
    local_i, local_j, local_k = projected_opt_module.angle_local_components(
        _as_positions(coords),
        projected,
        (0, 1, 2),
    )

    np.testing.assert_allclose(local_i, np.array([original_i[0], 0.0, 0.0]), atol=1.0e-10)
    np.testing.assert_allclose(local_j, np.zeros(3), atol=1.0e-10)
    np.testing.assert_allclose(local_k, np.array([original_k[0], 0.0, 0.0]), atol=1.0e-10)
    np.testing.assert_allclose(original_j, force.squeeze()[1], atol=1.0e-10)


def test_project_force_torsion_projection_keeps_only_ref_allowed_components():
    coords, force = _canonical_torsion_frame()
    projected = _call_projector(PROJECT_FORCE, coords, force, torsion_constraints=[(1, 2, 3, 4)])
    original_a, original_b, original_c, original_d = projected_opt_module.torsion_local_components(
        _as_positions(coords),
        force.squeeze(),
        (0, 1, 2, 3),
    )
    local_a, local_b, local_c, local_d = projected_opt_module.torsion_local_components(
        _as_positions(coords),
        projected,
        (0, 1, 2, 3),
    )

    np.testing.assert_allclose(local_a, np.array([original_a[0], original_a[1], 0.0]), atol=1.0e-10)
    np.testing.assert_allclose(local_b, np.array([original_b[0], 0.0, 0.0]), atol=1.0e-10)
    np.testing.assert_allclose(local_c, np.array([original_c[0], 0.0, 0.0]), atol=1.0e-10)
    np.testing.assert_allclose(local_d, np.array([original_d[0], original_d[1], 0.0]), atol=1.0e-10)


def test_project_force_multiple_torsions_zeroes_each_legacy_center_atom():
    coords_a, force_a = _canonical_torsion_frame()
    coords_b, force_b = _canonical_torsion_frame()
    positions = np.vstack(
        [
            _as_positions(coords_a),
            _as_positions(coords_b) + np.asarray((5.0, 0.0, 0.0), dtype=float),
        ]
    )
    coords = positions.reshape(1, 8, 3)
    force = np.vstack([force_a.squeeze(), 1.5 * force_b.squeeze()]).reshape(1, 8, 3)

    projected = _call_projector(
        PROJECT_FORCE,
        coords,
        force,
        torsion_constraints=[(1, 2, 3, 4), (5, 6, 7, 8)],
    )

    np.testing.assert_allclose(projected[2], np.zeros(3), atol=1.0e-12)
    np.testing.assert_allclose(projected[6], np.zeros(3), atol=1.0e-12)


def test_clip_step_by_atom_norm_matches_ref_longest_atom_scaling():
    step = np.asarray(
        [
            [3.0, 4.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    clipped = projected_opt_module.clip_step_by_atom_norm(step, max_step=2.0)

    np.testing.assert_allclose(
        clipped,
        np.asarray(
            [
                [1.2, 1.6, 0.0],
                [0.0, 0.0, 0.4],
            ],
            dtype=float,
        ),
        atol=1.0e-12,
    )
    assert np.linalg.norm(clipped, axis=1).max() == pytest.approx(2.0)


@pytest.mark.parametrize(
    "coords, force, geometry_fn, geometry_args, constraints",
    [
        (
            *_canonical_angle_frame(),
            _angle_deg,
            (0, 1, 2),
            {"angle_constraints": [(1, 2, 3)]},
        ),
        (
            *_canonical_torsion_frame(),
            _dihedral_deg,
            (0, 1, 2, 3),
            {"torsion_constraints": [(1, 2, 3, 4)]},
        ),
    ],
)
def test_small_step_projection_keeps_geometry_almost_fixed(coords, force, geometry_fn, geometry_args, constraints):
    projected = _call_projector(PROJECT_FORCE, coords, force, **constraints)
    before = geometry_fn(coords, *geometry_args)
    stepped = _as_positions(coords) + 1.0e-6 * projected
    after = geometry_fn(stepped, *geometry_args)

    assert after == pytest.approx(before, abs=1.0e-4)


def test_small_step_projection_keeps_all_active_torsions_almost_fixed():
    coords_a, force_a = _canonical_torsion_frame()
    coords_b, force_b = _canonical_torsion_frame()
    positions = np.vstack(
        [
            _as_positions(coords_a),
            _as_positions(coords_b) + np.asarray((5.0, 0.0, 0.0), dtype=float),
        ]
    )
    coords = positions.reshape(1, 8, 3)
    force = np.vstack([force_a.squeeze(), 1.5 * force_b.squeeze()]).reshape(1, 8, 3)

    projected = _call_projector(
        PROJECT_FORCE,
        coords,
        force,
        torsion_constraints=[(1, 2, 3, 4), (5, 6, 7, 8)],
    )

    stepped = _as_positions(coords) + 1.0e-6 * projected
    assert _dihedral_deg(stepped, 0, 1, 2, 3) == pytest.approx(_dihedral_deg(coords, 0, 1, 2, 3), abs=1.0e-4)
    assert _dihedral_deg(stepped, 4, 5, 6, 7) == pytest.approx(_dihedral_deg(coords, 4, 5, 6, 7), abs=1.0e-4)


def test_small_step_projection_keeps_mixed_constraints_almost_fixed():
    bond_coords = np.asarray(
        [
            [[0.0, 0.0, 0.0]],
            [[1.2, 0.0, 0.0]],
        ],
        dtype=float,
    ).reshape(1, 2, 3)
    bond_force = np.asarray(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        ],
        dtype=float,
    )
    angle_coords, angle_force = _canonical_angle_frame()
    torsion_coords, torsion_force = _canonical_torsion_frame()
    positions = np.vstack(
        [
            _as_positions(bond_coords),
            _as_positions(angle_coords) + np.asarray((5.0, 0.0, 0.0), dtype=float),
            _as_positions(torsion_coords) + np.asarray((10.0, 0.0, 0.0), dtype=float),
        ]
    )
    coords = positions.reshape(1, 9, 3)
    force = np.vstack([bond_force.squeeze(), angle_force.squeeze(), torsion_force.squeeze()]).reshape(1, 9, 3)

    projected = _call_projector(
        PROJECT_FORCE,
        coords,
        force,
        bond_constraints=[(1, 2)],
        angle_constraints=[(3, 4, 5)],
        torsion_constraints=[(6, 7, 8, 9)],
    )

    stepped = _as_positions(coords) + 1.0e-6 * projected
    assert _distance_ang(stepped, 0, 1) == pytest.approx(_distance_ang(coords, 0, 1), abs=1.0e-6)
    assert _angle_deg(stepped, 2, 3, 4) == pytest.approx(_angle_deg(coords, 2, 3, 4), abs=1.0e-4)
    assert _dihedral_deg(stepped, 5, 6, 7, 8) == pytest.approx(_dihedral_deg(coords, 5, 6, 7, 8), abs=1.0e-4)


def test_cgws_smoke_test_reduces_energy(tmp_path: Path):
    atoms = _make_atoms(
        [
            (1.2, 0.4, 0.1),
            (0.2, -0.8, 0.3),
            (-0.9, 0.5, -0.4),
            (0.6, 0.9, 1.1),
        ]
    )
    output = tmp_path / "cgws.out"
    initial_energy = float(atoms.get_potential_energy(force_consistent=True))

    optimizer = CGWS(
        atoms=atoms,
        output=str(output),
        paras={
            "opt": {
                "max_iter": 4,
                "max_step": 0.15,
                "verbose": 0,
                "write_traj": False,
            }
        },
    )
    final_atoms = optimizer.run()
    if final_atoms is None:
        final_atoms = atoms

    final_energy = float(final_atoms.get_potential_energy(force_consistent=True))
    assert final_energy < initial_energy
    assert not np.allclose(final_atoms.get_positions(), np.asarray(
        [
            (1.2, 0.4, 0.1),
            (0.2, -0.8, 0.3),
            (-0.9, 0.5, -0.4),
            (0.6, 0.9, 1.1),
        ],
        dtype=float,
    ))


def test_cgws_uses_ref_per_atom_norm_step_clip(tmp_path: Path, monkeypatch):
    start = np.asarray([(3.0, 4.0, 0.0)], dtype=float)
    atoms = _make_atoms(start)

    def _unit_line_search(*args, **kwargs):
        return 1.0, 0.0, np.asarray(args[3], dtype=float)

    monkeypatch.setattr(cgws_module, "strong_wolfe_line_search", _unit_line_search)
    optimizer = CGWS(
        atoms=atoms,
        output=str(tmp_path / "cgws_atom_norm_clip.out"),
        paras={"opt": {"max_iter": 1, "max_step": 0.2, "verbose": 0, "write_traj": False}},
    )

    result = optimizer.run()
    if result is None:
        result = atoms

    displacement = result.get_positions() - start
    np.testing.assert_allclose(displacement[0], np.asarray([-0.12, -0.16, 0.0]), atol=1.0e-12)
    assert np.linalg.norm(displacement[0]) == pytest.approx(0.2)


def test_cgws_uses_component_rms_formula(tmp_path: Path):
    atoms = _make_atoms([(0.0, 0.0, 0.0)])
    optimizer = CGWS(
        atoms=atoms,
        output=str(tmp_path / "cgws_rms.out"),
        paras={"opt": {"verbose": 0, "write_traj": False}},
    )
    step = np.asarray([[3.0, 4.0, 0.0]], dtype=float)
    forces = np.asarray([[6.0, 8.0, 0.0]], dtype=float)

    optimizer._build_iter_message(1, 0.0, step, forces)

    expected_step_rms = math.sqrt(np.mean(step ** 2))
    expected_force_rms = math.sqrt(np.mean(forces ** 2))
    assert atoms.rms_dp == pytest.approx(expected_step_rms)
    assert atoms.rms_f == pytest.approx(expected_force_rms)


def test_projected_lbfgs_uses_component_rms_formula(tmp_path: Path):
    atoms = _make_atoms([(0.0, 0.0, 0.0)])
    optimizer = ProjectedLBFGS(
        atoms=atoms,
        output=str(tmp_path / "lbfgs_rms.out"),
        paras={
            "opt": {
                "memory": 3,
                "curvature": 70.0,
                "verbose": 0,
                "write_traj": False,
            }
        },
    )
    step = np.asarray([[3.0, 4.0, 0.0]], dtype=float)
    forces = np.asarray([[6.0, 8.0, 0.0]], dtype=float)

    optimizer._build_iter_message(1, 0.0, step, forces)

    expected_step_rms = math.sqrt(np.mean(step ** 2))
    expected_force_rms = math.sqrt(np.mean(forces ** 2))
    assert atoms.rms_dp == pytest.approx(expected_step_rms)
    assert atoms.rms_f == pytest.approx(expected_force_rms)


def test_cgws_line_search_failure_does_not_fallback_to_unit_step(tmp_path: Path, monkeypatch):
    atoms = _make_atoms([(1.2, 0.4, 0.1), (0.2, -0.8, 0.3)])
    start = atoms.get_positions().copy()

    def _fail_line_search(*args, **kwargs):
        direction = np.asarray(args[1], dtype=float)
        return None, 123.0, np.zeros_like(direction)

    monkeypatch.setattr(cgws_module, "strong_wolfe_line_search", _fail_line_search)
    optimizer = CGWS(
        atoms=atoms,
        output=str(tmp_path / "cgws_ls_fail.out"),
        paras={"opt": {"max_iter": 3, "verbose": 0, "write_traj": False}},
    )

    result = optimizer.run()
    if result is None:
        result = atoms

    np.testing.assert_allclose(result.get_positions(), start, atol=1.0e-12)


def test_projected_lbfgs_default_takes_plain_lbfgs_step(tmp_path: Path):
    initial_positions = np.array(
        [
            (1.1, -0.4, 0.3),
            (-0.7, 0.9, -0.2),
            (0.5, 0.2, 1.0),
        ],
        dtype=float,
    )
    paras = {
        "opt": {
            "max_iter": 1,
            "max_step": 0.2,
            "memory": 3,
            "curvature": 70.0,
            "verbose": 0,
            "write_traj": False,
            "traj_every": 1,
        }
    }

    test_atoms = _make_atoms(initial_positions)
    test_optimizer = ProjectedLBFGS(test_atoms, output=str(tmp_path / "projected.out"), paras=paras)
    test_result = test_optimizer.run()

    if test_result is None:
        test_result = test_atoms

    expected_step = projected_lbfgs_module.clip_step(-initial_positions / 70.0, 0.2)
    expected_positions = initial_positions + expected_step
    np.testing.assert_allclose(test_result.get_positions(), expected_positions, atol=1.0e-10)


def test_projected_lbfgs_projection_mode_matches_default_without_constraints(tmp_path: Path):
    initial_positions = np.array(
        [
            (1.4, -0.2, 0.7),
            (-0.5, 1.0, -0.1),
            (0.3, 0.6, 0.8),
            (-1.0, -0.3, 0.4),
        ],
        dtype=float,
    )
    paras_default = {
        "opt": {
            "max_iter": 1,
            "max_step": 0.2,
            "memory": 3,
            "curvature": 70.0,
            "verbose": 0,
            "write_traj": False,
            "use_line_search": False,
        }
    }
    paras_projected = {
        "opt": {
            "max_iter": 1,
            "max_step": 0.2,
            "memory": 3,
            "curvature": 70.0,
            "verbose": 0,
            "write_traj": False,
            "use_line_search": False,
            "use_projection": True,
        }
    }

    atoms_default = _make_atoms(initial_positions)
    default_optimizer = ProjectedLBFGS(atoms_default, output=str(tmp_path / "default.out"), paras=paras_default)
    default_result = default_optimizer.run()

    atoms_projected = _make_atoms(initial_positions)
    projected_optimizer = ProjectedLBFGS(
        atoms_projected,
        output=str(tmp_path / "projected_no_constraints.out"),
        paras=paras_projected,
    )
    projected_result = projected_optimizer.run()

    if default_result is None:
        default_result = atoms_default
    if projected_result is None:
        projected_result = atoms_projected

    np.testing.assert_allclose(projected_result.get_positions(), default_result.get_positions(), atol=1.0e-10)


def test_projected_lbfgs_projection_mode_changes_result_with_torsion_constraints(tmp_path: Path):
    initial_positions = np.array(
        [
            (0.1, 1.1, 0.9),
            (0.0, 0.0, 0.2),
            (1.1, -0.1, -0.4),
            (1.4, 0.8, 1.2),
        ],
        dtype=float,
    )
    paras_default = {
        "opt": {
            "max_iter": 1,
            "max_step": 0.2,
            "memory": 3,
            "curvature": 70.0,
            "verbose": 0,
            "write_traj": False,
            "use_line_search": False,
        }
    }
    paras_projected = {
        "opt": {
            "max_iter": 1,
            "max_step": 0.2,
            "memory": 3,
            "curvature": 70.0,
            "verbose": 0,
            "write_traj": False,
            "use_line_search": False,
            "use_projection": True,
            "torsion_constraints": [(1, 2, 3, 4)],
        }
    }

    atoms_default = _make_atoms(initial_positions)
    default_optimizer = ProjectedLBFGS(atoms_default, output=str(tmp_path / "default_torsion.out"), paras=paras_default)
    default_result = default_optimizer.run()

    atoms_projected = _make_atoms(initial_positions)
    projected_optimizer = ProjectedLBFGS(
        atoms_projected,
        output=str(tmp_path / "projected_torsion.out"),
        paras=paras_projected,
    )
    projected_result = projected_optimizer.run()

    if default_result is None:
        default_result = atoms_default
    if projected_result is None:
        projected_result = atoms_projected

    assert not np.allclose(projected_result.get_positions(), default_result.get_positions(), atol=1.0e-10)


def test_projected_lbfgs_uses_ref_per_atom_norm_clip_when_projection_enabled(tmp_path: Path):
    start = np.asarray([(3.0, 4.0, 0.0)], dtype=float)
    atoms = _make_atoms(start)
    optimizer = ProjectedLBFGS(
        atoms,
        output=str(tmp_path / "lbfgs_projected_atom_norm_clip.out"),
        paras={
            "opt": {
                "max_iter": 1,
                "max_step": 0.2,
                "memory": 3,
                "curvature": 1.0,
                "verbose": 0,
                "write_traj": False,
                "use_projection": True,
                "use_line_search": False,
            }
        },
    )

    result = optimizer.run()
    if result is None:
        result = atoms

    displacement = result.get_positions() - start
    np.testing.assert_allclose(displacement[0], np.asarray([-0.12, -0.16, 0.0]), atol=1.0e-12)
    assert np.linalg.norm(displacement[0]) == pytest.approx(0.2)


def test_projected_lbfgs_line_search_failure_does_not_fallback_to_unit_step(tmp_path: Path, monkeypatch):
    initial_positions = np.array(
        [
            (0.1, 1.1, 0.9),
            (0.0, 0.0, 0.2),
            (1.1, -0.1, -0.4),
            (1.4, 0.8, 1.2),
        ],
        dtype=float,
    )
    atoms = _make_atoms(initial_positions)

    def _fail_line_search(*args, **kwargs):
        direction = np.asarray(args[1], dtype=float)
        return None, 456.0, np.zeros_like(direction)

    monkeypatch.setattr(projected_lbfgs_module, "strong_wolfe_line_search", _fail_line_search)
    optimizer = ProjectedLBFGS(
        atoms,
        output=str(tmp_path / "lbfgs_ls_fail.out"),
        paras={
            "opt": {
                "max_iter": 3,
                "memory": 3,
                "curvature": 70.0,
                "verbose": 0,
                "write_traj": False,
                "use_projection": True,
                "use_line_search": True,
                "torsion_constraints": [(1, 2, 3, 4)],
            }
        },
    )
    result = optimizer.run()
    if result is None:
        result = atoms

    np.testing.assert_allclose(result.get_positions(), initial_positions, atol=1.0e-12)


def test_lbfgs_ws_line_search_failure_does_not_fallback_to_unit_step(tmp_path: Path, monkeypatch):
    initial_positions = np.array(
        [
            (0.8, -0.2, 0.1),
            (-0.1, 0.4, -0.3),
            (0.2, 0.3, 0.7),
        ],
        dtype=float,
    )
    atoms = _make_atoms(initial_positions)

    def _fail_line_search(*args, **kwargs):
        direction = np.asarray(args[1], dtype=float)
        return None, 789.0, np.zeros_like(direction)

    monkeypatch.setattr(projected_lbfgs_module, "strong_wolfe_line_search", _fail_line_search)
    optimizer = ProjectedLBFGS(
        atoms,
        output=str(tmp_path / "lbfgs_ws_ls_fail.out"),
        paras={
            "opt": {
                "max_iter": 3,
                "memory": 3,
                "curvature": 70.0,
                "verbose": 0,
                "write_traj": False,
                "use_projection": False,
                "use_line_search": True,
            }
        },
    )
    result = optimizer.run()
    if result is None:
        result = atoms

    np.testing.assert_allclose(result.get_positions(), initial_positions, atol=1.0e-12)
    assert optimizer.S == []
    assert optimizer.Y == []
    assert optimizer.rhos == []


def test_cgbs_smoke_test_reduces_energy(tmp_path: Path):
    CGBS = _require_cgbs_class()
    atoms = _make_atoms(
        [
            (1.2, 0.4, 0.1),
            (0.2, -0.8, 0.3),
            (-0.9, 0.5, -0.4),
            (0.6, 0.9, 1.1),
        ]
    )
    output = tmp_path / "cgbs.out"
    initial_energy = float(atoms.get_potential_energy(force_consistent=True))

    optimizer = CGBS(
        atoms=atoms,
        output=str(output),
        paras={
            "opt": {
                "max_iter": 4,
                "max_step": 0.15,
                "verbose": 0,
                "write_traj": False,
            }
        },
    )
    final_atoms = optimizer.run()
    if final_atoms is None:
        final_atoms = atoms

    final_energy = float(final_atoms.get_potential_energy(force_consistent=True))
    assert final_energy < initial_energy


def test_cgbs_backtracking_reduces_initial_step_when_energy_increases(tmp_path: Path):
    CGBS = _require_cgbs_class()
    atoms = _make_atoms(
        [(0.0, 0.0, 0.0)],
        calc=ShiftedQuadraticCalculator(target=[1.0, 0.0, 0.0], stiffness=20.0),
    )
    optimizer = CGBS(
        atoms=atoms,
        output=str(tmp_path / "cgbs_backtrack.out"),
        paras={
            "opt": {
                "max_iter": 1,
                "max_step": 10.0,
                "alpha0": 0.2,
                "alpha_max": 0.8,
                "verbose": 0,
                "write_traj": False,
            }
        },
    )
    final_atoms = optimizer.run()
    if final_atoms is None:
        final_atoms = atoms

    final_x = float(final_atoms.get_positions()[0, 0])
    assert final_x == pytest.approx(1.0, abs=1.0e-8)
    assert final_x < 4.0


def test_cgbs_resets_conjugate_direction_after_backtracking_shrink(tmp_path: Path, monkeypatch):
    CGBS = _require_cgbs_class()
    atoms = _make_atoms([(-1.0, 0.0, 0.0)])
    optimizer = CGBS(
        atoms=atoms,
        output=str(tmp_path / "cgbs_gamma_reset.out"),
        paras={
            "opt": {
                "max_iter": 2,
                "max_step": 1.0,
                "alpha0": 0.2,
                "verbose": 0,
                "write_traj": False,
            }
        },
    )
    directions = []

    def _fake_backtracking_step(self, positions, energy, forces, direction, alpha):
        del energy, alpha
        directions.append(np.asarray(direction, dtype=float).copy())
        if len(directions) == 1:
            step = np.asarray([[0.01, 0.0, 0.0]], dtype=float)
            self.atoms.set_positions(positions + step)
            return step, -1.0, np.asarray([[1.0, 1.0, 0.0]], dtype=float), 0.5, True
        return None, -1.0, np.asarray(forces, dtype=float), 1.0, False

    monkeypatch.setattr(optimizer, "_backtracking_step", types.MethodType(_fake_backtracking_step, optimizer))

    optimizer.run()

    np.testing.assert_allclose(directions[0], np.asarray([[1.0, 0.0, 0.0]]), atol=1.0e-12)
    np.testing.assert_allclose(directions[1], np.asarray([[1.0, 1.0, 0.0]]), atol=1.0e-12)


def test_cgbs_caps_alpha_growth(tmp_path: Path):
    CGBS = _require_cgbs_class()
    atoms = _make_atoms([(0.0, 0.0, 0.0)], calc=LinearXCalculator())
    optimizer = CGBS(
        atoms=atoms,
        output=str(tmp_path / "cgbs_alpha.out"),
        paras={
            "opt": {
                "max_iter": 80,
                "max_step": 10.0,
                "alpha0": 0.2,
                "alpha_max": 0.8,
                "verbose": 0,
                "write_traj": False,
            }
        },
    )
    result = optimizer.run()
    if result is None:
        result = atoms

    frames = _load_xyz_frames(tmp_path / "cgbs_alpha_traj.xyz")
    max_dx = max(abs(float(curr[0, 0] - prev[0, 0])) for prev, curr in zip(frames, frames[1:]))
    assert max_dx <= 0.8 + 1.0e-12


def test_cgbs_returns_best_so_far_geometry_after_nonconvergence(tmp_path: Path):
    CGBS = _require_cgbs_class()
    calc = StepBarrierXCalculator(barrier_x=0.1, stiffness=100.0, penalty=1.0)
    atoms = _make_atoms([(0.2, 0.0, 0.0)], calc=calc)
    optimizer = CGBS(
        atoms=atoms,
        output=str(tmp_path / "cgbs_best.out"),
        paras={
            "opt": {
                "max_iter": 2,
                "max_step": 10.0,
                "alpha0": 0.2,
                "alpha_max": 0.8,
                "verbose": 0,
                "write_traj": False,
            }
        },
    )
    result = optimizer.run()
    if result is None:
        result = atoms

    frames = _load_xyz_frames(tmp_path / "cgbs_best_traj.xyz")
    frame_energies = []
    for frame in frames:
        probe = _make_atoms(frame, calc=StepBarrierXCalculator(barrier_x=0.1, stiffness=100.0, penalty=1.0))
        frame_energies.append(float(probe.get_potential_energy(force_consistent=True)))

    result_energy = float(result.get_potential_energy(force_consistent=True))
    assert result_energy <= min(frame_energies) + 1.0e-12
