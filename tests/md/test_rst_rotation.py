import numpy as np
import sys
import os
from ase import Atoms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from maple.function.dispatcher.md.rst_io import rotate_rst_checkpoint, read_rst



def build_atoms():
    atoms = Atoms("He", positions=[[0.0, 0.0, 0.0]])
    velocities = np.array([[0.001, 0.002, 0.003]])
    return atoms, velocities



def test_first_checkpoint_writes_current_only(tmp_path):
    atoms, velocities = build_atoms()
    current = tmp_path / "job_md.rst"
    previous = tmp_path / "job_md_prev.rst"

    rotate_rst_checkpoint(
        rst_path=current,
        rst_prev_path=previous,
        atoms=atoms,
        velocities=velocities,
        step=100,
        timestep=0.5,
        ensemble="nve",
        energy=-1.0,
    )

    assert current.exists()
    assert not previous.exists()
    assert read_rst(current)["step"] == 100



def test_second_checkpoint_rotates_old_current_to_previous(tmp_path):
    atoms, velocities = build_atoms()
    current = tmp_path / "job_md.rst"
    previous = tmp_path / "job_md_prev.rst"

    rotate_rst_checkpoint(current, previous, atoms, velocities, 100, 0.5, "nve", -1.0)
    rotate_rst_checkpoint(current, previous, atoms, velocities, 200, 0.5, "nve", -2.0)

    assert read_rst(current)["step"] == 200
    assert read_rst(previous)["step"] == 100
