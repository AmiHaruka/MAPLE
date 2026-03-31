import sys
import os
import numpy as np
import pytest
from ase import Atoms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from maple.function.dispatcher.md.logger import MDLogger
from maple.function.dispatcher.md.rst_io import write_rst, get_rng_state_hex


def build_atoms():
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    atoms.info["charge"] = 0
    atoms.info["mult"] = 1
    return atoms


def test_logger_restart_simulation_reads_rst(tmp_path):
    atoms = build_atoms()
    velocities = np.array([[0.001, 0.002, 0.003], [0.004, 0.005, 0.006]])
    out = tmp_path / "job.out"
    logger = MDLogger(str(out), log_every=10, traj_every=10, verbose=0)

    write_rst(
        tmp_path / "job_md.rst",
        atoms=atoms,
        velocities=velocities,
        step=20,
        timestep=0.5,
        ensemble="nvt",
        energy=-1.23,
    )

    resumed_atoms, resumed_velocities, step_offset = logger.restart_simulation(
        ensemble="nvt",
        timestep=0.5,
        n_steps=100,
        temperature=300.0,
        atoms=atoms,
    )

    assert step_offset == 20
    assert np.allclose(resumed_atoms.get_positions(), atoms.get_positions())
    assert np.allclose(resumed_velocities, velocities)


def test_logger_restart_uses_prev_when_current_is_corrupt(tmp_path):
    atoms = build_atoms()
    velocities = np.array([[0.001, 0.002, 0.003], [0.004, 0.005, 0.006]])
    out = tmp_path / "job.out"
    logger = MDLogger(str(out), log_every=10, traj_every=10, verbose=0)

    (tmp_path / "job_md.rst").write_text("broken\n")
    write_rst(
        tmp_path / "job_md_prev.rst",
        atoms=atoms,
        velocities=velocities,
        step=15,
        timestep=0.5,
        ensemble="nvt",
        energy=-1.23,
    )

    resumed_atoms, resumed_velocities, step_offset = logger.restart_simulation(
        ensemble="nvt",
        timestep=0.5,
        n_steps=100,
        temperature=300.0,
        atoms=atoms,
    )

    assert step_offset == 15
    assert np.allclose(resumed_velocities, velocities)


def test_logger_restart_rejects_ensemble_mismatch(tmp_path):
    atoms = build_atoms()
    velocities = np.array([[0.001, 0.002, 0.003], [0.004, 0.005, 0.006]])
    out = tmp_path / "job.out"
    logger = MDLogger(str(out), log_every=10, traj_every=10, verbose=0)

    write_rst(
        tmp_path / "job_md.rst",
        atoms=atoms,
        velocities=velocities,
        step=20,
        timestep=0.5,
        ensemble="nve",
        energy=-1.23,
    )

    with pytest.raises(RuntimeError, match="Ensemble mismatch"):
        logger.restart_simulation(
            ensemble="nvt",
            timestep=0.5,
            n_steps=100,
            temperature=300.0,
            atoms=atoms,
        )


from maple.function.dispatcher.md.ensemble.nve import NVEParams
from maple.function.dispatcher.md.ensemble.nvt import NVTParams
from maple.function.dispatcher.md.ensemble.npt import NPTParams
from maple.function.dispatcher.md.rst_io import restore_rng_from_hex, get_rng_state_hex
from numpy.random import default_rng


def test_nve_params_expose_restart_fields_only():
    params = NVEParams()

    assert params.restart is False
    assert params.rst_every == 1000
    assert not hasattr(params, "resume")
    assert not hasattr(params, "init_from")


def test_nvt_and_npt_params_expose_restart_fields_only():
    nvt = NVTParams()
    npt = NPTParams()

    for params in (nvt, npt):
        assert params.restart is False
        assert params.rst_every == 1000
        assert not hasattr(params, "resume")
        assert not hasattr(params, "init_from")


def test_rng_state_restores_deterministically():
    rng1 = default_rng(123)
    rng2 = default_rng(999)
    state_hex = get_rng_state_hex(rng1)

    restore_rng_from_hex(rng2, state_hex)

    assert np.allclose(rng1.standard_normal(4), rng2.standard_normal(4))
