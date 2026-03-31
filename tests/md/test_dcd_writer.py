import pytest
import sys
import os
import struct
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ase import Atoms
from maple.function.dispatcher.md.dcd_writer import DCDWriter


def build_atoms():
    """Create a simple H2O test system."""
    atoms = Atoms(
        "H2O",
        positions=[[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2390, 0.9270, 0.0]],
    )
    atoms.set_cell([10.0, 11.0, 12.0])
    atoms.set_pbc([True, True, True])
    return atoms


def test_dcd_header_format(tmp_path):
    """Test DCD header is written correctly."""
    dcd_path = tmp_path / "test.dcd"
    writer = DCDWriter(str(dcd_path), natoms=3, timestep=0.5)
    writer.close()  # Must close before reading

    # Read header directly
    with open(dcd_path, "rb") as f:
        # First record marker (84 bytes of header data)
        marker = struct.unpack('<i', f.read(4))[0]
        assert marker == 84, f"Expected header marker 84, got {marker}"

        # Header data (21 int32 = 84 bytes)
        header_data = f.read(84)
        hdr = np.frombuffer(header_data, dtype=np.int32)

        # CORD magic number
        assert hdr[0] == 84, f"Expected CORD magic 84 at position 0, got {hdr[0]}"

        # CHARMM version
        assert hdr[20] == 24, f"Expected CHARMM version 24, got {hdr[20]}"

        # NATOMS
        assert hdr[6] == 3, f"Expected 3 atoms, got {hdr[6]}"

        # DELTA (timestep in picoseconds)
        # 0.5 fs = 0.0005 ps, rounded to 0 as int
        assert hdr[9] == 0, f"Expected DELTA 0 (0.5 fs rounds to 0 ps), got {hdr[9]}"

        # Closing marker
        marker = struct.unpack('<i', f.read(4))[0]
        assert marker == 84


def test_dcd_header_with_larger_timestep(tmp_path):
    """Test DELTA field with a timestep that rounds to non-zero ps."""
    dcd_path = tmp_path / "test.dcd"
    writer = DCDWriter(str(dcd_path), natoms=3, timestep=2500.0)  # 2.5 ps
    writer.close()

    with open(dcd_path, "rb") as f:
        f.read(4)  # marker
        header_data = f.read(84)
        hdr = np.frombuffer(header_data, dtype=np.int32)
        # 2.5 ps should round to 3 ps, then 2 (int)
        # Actually 2500 fs = 2.5 ps, int(2.5) = 2
        assert hdr[9] == 2, f"Expected DELTA 2 (2.5 ps rounds to 2), got {hdr[9]}"


def test_dcd_write_frame(tmp_path):
    """Test writing a single frame."""
    atoms = build_atoms()
    dcd_path = tmp_path / "test.dcd"

    with DCDWriter(str(dcd_path), natoms=3, timestep=1.0, is_periodic=True) as w:
        w.write_frame(atoms, step=0)
    # File closed by context manager

    # Read back and verify
    with open(dcd_path, "rb") as f:
        # Skip header (92 bytes total with markers)
        f.read(92)

        # Skip title block (168 bytes)
        f.read(168)

        # Skip NATOM block (12 bytes)
        f.read(12)

        # Unit cell record
        marker = struct.unpack('<i', f.read(4))[0]
        assert marker == 48  # 6 doubles = 48 bytes
        cell_data = np.frombuffer(f.read(48), dtype=np.float64)
        marker = struct.unpack('<i', f.read(4))[0]
        assert marker == 48

        # DCD cell order: A, gamma, B, beta, alpha, C
        a, gamma, b, beta, alpha, c = cell_data
        assert a == pytest.approx(10.0)
        assert b == pytest.approx(11.0)
        assert c == pytest.approx(12.0)
        # alpha, beta, gamma should be 90 degrees
        assert alpha == pytest.approx(90.0)
        assert beta == pytest.approx(90.0)
        assert gamma == pytest.approx(90.0)

        # X coordinates
        marker = struct.unpack('<i', f.read(4))[0]
        assert marker == 12  # 3 floats * 4 bytes = 12
        x_data = np.frombuffer(f.read(12), dtype=np.float32)
        marker = struct.unpack('<i', f.read(4))[0]
        assert marker == 12

        # Y coordinates
        marker = struct.unpack('<i', f.read(4))[0]
        y_data = np.frombuffer(f.read(12), dtype=np.float32)
        marker = struct.unpack('<i', f.read(4))[0]

        # Z coordinates
        marker = struct.unpack('<i', f.read(4))[0]
        z_data = np.frombuffer(f.read(12), dtype=np.float32)
        marker = struct.unpack('<i', f.read(4))[0]

        # Verify coordinates (original positions)
        assert x_data[0] == pytest.approx(0.0, abs=1e-5)
        assert y_data[0] == pytest.approx(0.0, abs=1e-5)
        assert z_data[0] == pytest.approx(0.0, abs=1e-5)

        assert x_data[1] == pytest.approx(0.9572, abs=1e-5)
        assert y_data[1] == pytest.approx(0.0, abs=1e-5)
        assert z_data[1] == pytest.approx(0.0, abs=1e-5)

        assert x_data[2] == pytest.approx(-0.2390, abs=1e-5)
        assert y_data[2] == pytest.approx(0.9270, abs=1e-5)
        assert z_data[2] == pytest.approx(0.0, abs=1e-5)


def test_dcd_multiple_frames(tmp_path):
    """Test writing multiple frames."""
    atoms = build_atoms()
    dcd_path = tmp_path / "test.dcd"

    with DCDWriter(str(dcd_path), natoms=3, timestep=1.0, is_periodic=True) as w:
        w.write_frame(atoms, step=0)
        atoms[0].position = [1.0, 1.0, 1.0]  # Move first atom
        w.write_frame(atoms, step=1)
        w.write_frame(atoms, step=2)

    # Verify NSET was updated in header
    with open(dcd_path, "rb") as f:
        f.read(4)  # marker
        f.read(4)  # HDR[0] = 84 (CORD)
        f.read(4)  # HDR[1] = NPRIV
        nset_bytes = f.read(4)  # HDR[2] = NSET
        nset = struct.unpack('<i', nset_bytes)[0]
        assert nset == 3, f"Expected NSET=3, got {nset}"


def test_dcd_non_periodic(tmp_path):
    """Test DCD for non-periodic system."""
    atoms = build_atoms()
    atoms.set_pbc([False, False, False])
    dcd_path = tmp_path / "test.dcd"

    with DCDWriter(str(dcd_path), natoms=3, timestep=1.0, is_periodic=False) as w:
        w.write_frame(atoms, step=0)
    # File closed by context manager

    # Read and verify zero cell is written
    with open(dcd_path, "rb") as f:
        # Skip to first frame (header + title + natom)
        f.read(92 + 168 + 12)

        # Unit cell should be all zeros
        f.read(4)  # marker
        cell_data = np.frombuffer(f.read(48), dtype=np.float64)
        assert np.allclose(cell_data, 0.0), "Expected zero cell for non-periodic system"


def test_dcd_atom_count_mismatch(tmp_path):
    """Test that atom count mismatch raises error."""
    atoms = build_atoms()  # 3 atoms
    dcd_path = tmp_path / "test.dcd"

    with DCDWriter(str(dcd_path), natoms=5, timestep=1.0) as w:
        with pytest.raises(ValueError, match="Atom count mismatch"):
            w.write_frame(atoms, step=0)


def test_dcd_open_for_append(tmp_path):
    """Test appending to an existing DCD file."""
    atoms = build_atoms()
    dcd_path = tmp_path / "test.dcd"

    # Write initial frames with timestep that produces non-zero DELTA
    # 2.5 ps = 2500 fs -> DELTA = 2 (stored as int ps)
    with DCDWriter(str(dcd_path), natoms=3, timestep=2500.0, is_periodic=True) as w:
        w.write_frame(atoms, step=0)
        w.write_frame(atoms, step=1)

    # Append more frames
    writer = DCDWriter.open_for_append(str(dcd_path))
    assert writer._nframes == 2
    assert writer.natoms == 3
    # timestep is read back from DELTA field (int ps * 1000)
    # DELTA = 2 -> timestep = 2000 fs (stored as int, lost precision)
    assert writer.timestep == 2000.0
    writer.write_frame(atoms, step=2)
    writer.write_frame(atoms, step=3)
    writer.close()

    # Verify total frames
    with open(dcd_path, "rb") as f:
        f.read(4)  # marker
        f.read(4)  # HDR[0] = 84 (CORD)
        f.read(4)  # HDR[1] = NPRIV
        nset = struct.unpack('<i', f.read(4))[0]  # HDR[2] = NSET
        assert nset == 4


def test_dcd_open_for_append_nonexistent(tmp_path):
    """Test that appending to non-existent file raises error."""
    dcd_path = tmp_path / "nonexistent.dcd"

    with pytest.raises(ValueError, match="does not exist"):
        DCDWriter.open_for_append(str(dcd_path))


def test_dcd_context_manager(tmp_path):
    """Test using DCDWriter as context manager."""
    atoms = build_atoms()
    dcd_path = tmp_path / "test.dcd"

    with DCDWriter(str(dcd_path), natoms=3, timestep=1.0) as w:
        w.write_frame(atoms, step=0)
        # File should be properly closed after context

    # Verify NSET was updated (file was properly closed)
    with open(dcd_path, "rb") as f:
        f.read(4)  # marker
        f.read(4)  # HDR[0] = 84 (CORD)
        f.read(4)  # HDR[1] = NPRIV
        nset = struct.unpack('<i', f.read(4))[0]  # HDR[2] = NSET
        assert nset == 1


def test_dcd_invalid_header_on_append(tmp_path):
    """Test that invalid DCD file is detected on append."""
    dcd_path = tmp_path / "invalid.dcd"
    dcd_path.write_text("NOT A DCD FILE")

    with pytest.raises(ValueError, match="Invalid DCD file"):
        DCDWriter.open_for_append(str(dcd_path))
