import numpy as np
import pytest
import sys
import os
from ase import Atoms
from numpy.random import default_rng

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from maple.function.dispatcher.md.rst_io import read_rst, write_rst, get_rng_state_hex, restore_rng_from_hex



def build_atoms():
    atoms = Atoms(
        "H2O",
        positions=[[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2390, 0.9270, 0.0]],
    )
    atoms.set_cell([10.0, 11.0, 12.0])
    atoms.set_pbc([True, True, True])
    atoms.info["charge"] = 0
    atoms.info["mult"] = 1
    return atoms



def test_rst_round_trip(tmp_path):
    atoms = build_atoms()
    velocities = np.array([
        [0.001, 0.002, 0.003],
        [0.004, 0.005, 0.006],
        [0.007, 0.008, 0.009],
    ])
    rng = default_rng(123)
    rst_path = tmp_path / "state.rst"

    write_rst(
        rst_path,
        atoms=atoms,
        velocities=velocities,
        step=15000,
        timestep=0.5,
        ensemble="nvt",
        energy=-76.123456789,
        rng_state=get_rng_state_hex(rng),
    )

    state = read_rst(rst_path)

    assert state["natoms"] == 3
    assert state["step"] == 15000
    assert state["time"] == pytest.approx(7500.0)
    assert state["ensemble"] == "nvt"
    assert state["timestep"] == pytest.approx(0.5)
    assert state["energy"] == pytest.approx(-76.123456789)
    assert state["symbols"] == ["H", "H", "O"]
    assert np.allclose(state["positions"], atoms.get_positions())
    assert np.allclose(state["velocities"], velocities)
    assert np.allclose(state["cell"], [10.0, 11.0, 12.0, 90.0, 90.0, 90.0])
    assert state["pbc"] == [True, True, True]
    assert isinstance(state["rng_state"], str)



def test_rst_invalid_header(tmp_path):
    path = tmp_path / "bad.rst"
    path.write_text("NOT_MAPLE_RST\nEND_RST\n")

    with pytest.raises(ValueError, match="Not a valid MAPLE RST file"):
        read_rst(path)



def test_rst_missing_end_marker(tmp_path):
    path = tmp_path / "bad.rst"
    path.write_text("MAPLE_RST_V1\nnatoms = 1\nstep = 0\ntime = 0.0\nensemble = nve\ntimestep = 1.0\nenergy = 0.0\nH 0.0 0.0 0.0 0.0 0.0 0.0\n")

    with pytest.raises(ValueError, match="Missing END_RST"):
        read_rst(path)



def test_rst_missing_required_fields(tmp_path):
    path = tmp_path / "missing.rst"
    path.write_text("MAPLE_RST_V1\nnatoms = 1\nstep = 0\nensemble = nve\nenergy = 0.0\nH 0.0 0.0 0.0 0.0 0.0 0.0\nEND_RST\n")

    with pytest.raises(ValueError, match="Missing required RST header fields"):
        read_rst(path)



def test_rst_velocity_shape_mismatch(tmp_path):
    atoms = build_atoms()
    velocities = np.array([[0.001, 0.002, 0.003]])
    rst_path = tmp_path / "state.rst"

    with pytest.raises(ValueError, match="Velocities must have shape"):
        write_rst(
            rst_path,
            atoms=atoms,
            velocities=velocities,
            step=1,
            timestep=0.5,
            ensemble="nvt",
            energy=-1.0,
        )



def test_rng_state_round_trip():
    rng1 = default_rng(123)
    rng2 = default_rng(999)

    state_hex = get_rng_state_hex(rng1)
    restore_rng_from_hex(rng2, state_hex)

    assert np.allclose(rng1.standard_normal(5), rng2.standard_normal(5))



def test_rng_helpers_live_in_rst_io_not_utils():
    import maple.function.dispatcher.md.rst_io as rst_io
    import maple.function.dispatcher.md.utils as utils

    assert hasattr(rst_io, "get_rng_state_hex")
    assert hasattr(rst_io, "restore_rng_from_hex")
    assert not hasattr(utils, "read_last_xyz_frame")
