import io

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from maple.function.dispatcher.md.ensemble.nvt import NVT, _apply_projection_with_work
from maple.function.dispatcher.md.logger import MDLogger
from maple.function.dispatcher.md.utils import calculate_kinetic_energy


def test_apply_projection_with_work_tracks_com_projection_energy_change():
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    atoms.pbc = [True, True, True]

    velocities = np.array([
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])

    kinetic_before = calculate_kinetic_energy(atoms, velocities)
    projected, projection, delta_w_proj = _apply_projection_with_work(
        atoms,
        velocities,
        step=1,
        remove_com_every=1,
        remove_angular_every=0,
    )
    kinetic_after = calculate_kinetic_energy(atoms, projected)

    assert projection == "com"
    assert np.allclose(projected, 0.0)
    assert np.isclose(delta_w_proj, kinetic_after - kinetic_before)
    assert delta_w_proj < 0.0


def test_apply_projection_with_work_tracks_angular_projection_energy_change():
    atoms = Atoms("H2", positions=[[-0.37, 0.0, 0.0], [0.37, 0.0, 0.0]])
    atoms.pbc = [False, False, False]

    velocities = np.array([
        [0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0],
    ])

    kinetic_before = calculate_kinetic_energy(atoms, velocities)
    projected, projection, delta_w_proj = _apply_projection_with_work(
        atoms,
        velocities,
        step=1,
        remove_com_every=0,
        remove_angular_every=1,
    )
    kinetic_after = calculate_kinetic_energy(atoms, projected)

    assert projection == "angular"
    assert np.allclose(projected, 0.0, atol=1e-12)
    assert np.isclose(delta_w_proj, kinetic_after - kinetic_before)
    assert delta_w_proj < 0.0


class ZeroCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        n_atoms = len(atoms)
        self.results["energy"] = 0.0
        self.results["forces"] = np.zeros((n_atoms, 3))


def test_nvt_vrescale_conserved_energy_includes_projection_work(monkeypatch, tmp_path):
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    atoms.pbc = [True, True, True]
    atoms.calc = ZeroCalculator()

    md = NVT(
        output=str(tmp_path / "nvt.out"),
        atoms=atoms,
        paras={
            "thermostat": "v-rescale",
            "steps": 1,
            "timestep": 0.5,
            "temperature": 300.0,
            "tau_t": 100.0,
            "log_every": 1,
            "traj_every": 1,
            "remove_com": False,
            "remove_com_every": 1,
            "random_seed": 1,
            "verbose": 0,
        },
    )

    initial_v = np.array([
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])

    class StubThermostat:
        def __init__(self, velocities):
            self.timestep = md.params.timestep
            self._velocities = velocities

        def apply(self, velocities):
            return self._velocities.copy(), 0.0

    class StubIntegrator:
        def __init__(self, atoms, timestep):
            self.timestep = timestep

        def step(self, velocities, forces):
            return velocities.copy(), np.zeros_like(forces)

    captured = {}

    def capture_log_step(**kwargs):
        captured["conserved_energy"] = kwargs["conserved_energy"]
        captured["kinetic_energy"] = kwargs["kinetic_energy"]
        captured["potential_energy"] = kwargs["potential_energy"]

    monkeypatch.setattr("maple.function.dispatcher.md.ensemble.nvt.initialize_velocities", lambda *args, **kwargs: initial_v.copy())
    monkeypatch.setattr("maple.function.dispatcher.md.ensemble.nvt.VelocityVerlet", StubIntegrator)
    monkeypatch.setattr(md, "thermostat", StubThermostat(initial_v))
    monkeypatch.setattr(md.logger, "log_step", capture_log_step)
    monkeypatch.setattr(md.logger, "start_simulation", lambda **kwargs: None)
    monkeypatch.setattr(md.logger, "end_simulation", lambda **kwargs: None)
    monkeypatch.setattr(md.logger, "log_main", lambda lines: None)

    md.run()

    expected_projection_work = -calculate_kinetic_energy(atoms, initial_v)
    expected_conserved = captured["kinetic_energy"] + captured["potential_energy"] - expected_projection_work

    assert np.isclose(captured["kinetic_energy"], 0.0)
    assert np.isclose(captured["conserved_energy"], expected_conserved)
    assert np.isclose(captured["conserved_energy"], -expected_projection_work)


def test_mdlogger_progress_and_thermo_include_conserved_energy_for_vrescale(tmp_path, monkeypatch):
    output_path = tmp_path / "nvt.out"
    output_path.write_text("", encoding="utf-8")
    logger = MDLogger(output_path=str(output_path), log_every=100, traj_every=100, verbose=1)

    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    atoms.pbc = [False, False, False]
    velocities = np.zeros((2, 3))

    printed = []
    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: printed.append("".join(str(a) for a in args)))

    logger.start_simulation(
        ensemble="nvt",
        timestep=0.5,
        n_steps=100,
        temperature=300.0,
        atoms=atoms,
        step_offset=0,
        write_sync_thermo=False,
        write_conserved_energy=True,
    )
    logger.log_step(
        step=100,
        time=50.0,
        temperature=300.0,
        kinetic_energy=1.0,
        potential_energy=2.0,
        total_energy=3.0,
        atoms=atoms,
        velocities=velocities,
        conserved_energy=2.5,
    )

    thermo_text = logger.thermo_path.read_text(encoding="utf-8")

    assert "H_cons_ext(Ha)" in thermo_text
    assert any("H_cons_ext(Ha)" in line for line in printed)


def test_mdlogger_start_simulation_backs_up_preexisting_rst_files(tmp_path, monkeypatch):
    output_path = tmp_path / "nvt.out"
    output_path.write_text("", encoding="utf-8")
    logger = MDLogger(output_path=str(output_path), log_every=100, traj_every=100, verbose=0)

    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    atoms.pbc = [False, False, False]

    logger.rst_path.write_text("rst-current", encoding="utf-8")
    logger.rst_prev_path.write_text("rst-prev", encoding="utf-8")

    monkeypatch.setattr(logger, "log_main", lambda *args, **kwargs: None)

    logger.start_simulation(
        ensemble="nvt",
        timestep=0.5,
        n_steps=10,
        temperature=300.0,
        atoms=atoms,
        step_offset=0,
        write_sync_thermo=False,
    )

    backup_current = logger.rst_path.parent / f"#{logger.rst_path.name}.1#"
    backup_prev = logger.rst_prev_path.parent / f"#{logger.rst_prev_path.name}.1#"

    assert backup_current.exists()
    assert backup_prev.exists()
