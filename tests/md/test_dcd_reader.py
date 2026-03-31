"""Tests for DCD binary trajectory reader."""

import pytest
import sys
import os
import struct
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ase import Atoms
from maple.function.dispatcher.md.dcd_writer import DCDWriter
from maple.function.read.filereader.dcd_reader import DCDReader, read_dcd


def build_atoms():
    """Create a simple H2O test system."""
    atoms = Atoms(
        "H2O",
        positions=[[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2390, 0.9270, 0.0]],
    )
    atoms.set_cell([10.0, 11.0, 12.0])
    atoms.set_pbc([True, True, True])
    return atoms


class TestDCDReaderBasic:
    """Basic DCD reader tests."""

    def test_read_single_frame(self, tmp_path):
        """Test reading a single frame from DCD file."""
        atoms = build_atoms()
        dcd_path = tmp_path / "test.dcd"

        with DCDWriter(str(dcd_path), natoms=3, timestep=1.0, is_periodic=True) as w:
            w.write_frame(atoms, step=0)

        with DCDReader(str(dcd_path)) as reader:
            assert reader.natoms == 3
            assert reader.nframes == 1

            frame = reader.read_frame(0)
            assert len(frame) == 3

            # Verify positions
            positions = frame.get_positions()
            assert positions[0] == pytest.approx([0.0, 0.0, 0.0], abs=1e-5)
            assert positions[1] == pytest.approx([0.9572, 0.0, 0.0], abs=1e-5)
            assert positions[2] == pytest.approx([-0.2390, 0.9270, 0.0], abs=1e-5)

            # Verify cell
            cell = frame.get_cell()
            assert cell[0, 0] == pytest.approx(10.0, abs=1e-5)
            assert cell[1, 1] == pytest.approx(11.0, abs=1e-5)
            assert cell[2, 2] == pytest.approx(12.0, abs=1e-5)

    def test_read_multiple_frames(self, tmp_path):
        """Test reading multiple frames."""
        atoms = build_atoms()
        dcd_path = tmp_path / "test.dcd"

        with DCDWriter(str(dcd_path), natoms=3, timestep=1.0, is_periodic=True) as w:
            w.write_frame(atoms, step=0)
            atoms[0].position = [1.0, 1.0, 1.0]
            w.write_frame(atoms, step=1)
            atoms[1].position = [2.0, 2.0, 2.0]
            w.write_frame(atoms, step=2)

        with DCDReader(str(dcd_path)) as reader:
            assert reader.nframes == 3

            frames = reader.read_all()
            assert len(frames) == 3

            # Check first frame
            assert frames[0].get_positions()[0] == pytest.approx([0.0, 0.0, 0.0], abs=1e-5)

            # Check second frame
            assert frames[1].get_positions()[0] == pytest.approx([1.0, 1.0, 1.0], abs=1e-5)

            # Check third frame
            assert frames[2].get_positions()[1] == pytest.approx([2.0, 2.0, 2.0], abs=1e-5)

    def test_read_slice(self, tmp_path):
        """Test reading a slice of frames."""
        atoms = build_atoms()
        dcd_path = tmp_path / "test.dcd"

        with DCDWriter(str(dcd_path), natoms=3, timestep=1.0, is_periodic=True) as w:
            for i in range(10):
                atoms[0].position = [float(i), 0.0, 0.0]
                w.write_frame(atoms, step=i)

        with DCDReader(str(dcd_path)) as reader:
            # Read frames 2, 4, 6, 8 (step=2)
            frames = reader.read_slice(2, 10, step=2)
            assert len(frames) == 4

            # Verify positions
            assert frames[0].get_positions()[0] == pytest.approx([2.0, 0.0, 0.0], abs=1e-5)
            assert frames[1].get_positions()[0] == pytest.approx([4.0, 0.0, 0.0], abs=1e-5)
            assert frames[2].get_positions()[0] == pytest.approx([6.0, 0.0, 0.0], abs=1e-5)
            assert frames[3].get_positions()[0] == pytest.approx([8.0, 0.0, 0.0], abs=1e-5)

    def test_read_non_periodic(self, tmp_path):
        """Test reading DCD file with non-periodic system."""
        atoms = build_atoms()
        atoms.set_pbc([False, False, False])
        dcd_path = tmp_path / "test.dcd"

        with DCDWriter(str(dcd_path), natoms=3, timestep=1.0, is_periodic=False) as w:
            w.write_frame(atoms, step=0)

        with DCDReader(str(dcd_path)) as reader:
            frame = reader.read_frame(0)
            # Non-periodic should have zero cell
            assert not any(frame.get_pbc())

    def test_read_with_natoms_validation(self, tmp_path):
        """Test that natoms mismatch raises error."""
        atoms = build_atoms()
        dcd_path = tmp_path / "test.dcd"

        with DCDWriter(str(dcd_path), natoms=3, timestep=1.0) as w:
            w.write_frame(atoms, step=0)

        # Should work with correct natoms
        with DCDReader(str(dcd_path), natoms=3) as reader:
            assert reader.natoms == 3

        # Should fail with wrong natoms
        with pytest.raises(ValueError, match="Atom count mismatch"):
            DCDReader(str(dcd_path), natoms=5)


class TestDCDReaderErrors:
    """Error handling tests."""

    def test_file_not_found(self, tmp_path):
        """Test that missing file raises FileNotFoundError."""
        dcd_path = tmp_path / "nonexistent.dcd"
        with pytest.raises(FileNotFoundError):
            DCDReader(str(dcd_path))

    def test_invalid_frame_index(self, tmp_path):
        """Test that invalid frame index raises IndexError."""
        atoms = build_atoms()
        dcd_path = tmp_path / "test.dcd"

        with DCDWriter(str(dcd_path), natoms=3, timestep=1.0) as w:
            w.write_frame(atoms, step=0)

        with DCDReader(str(dcd_path)) as reader:
            with pytest.raises(IndexError):
                reader.read_frame(1)  # Only frame 0 exists

            with pytest.raises(IndexError):
                reader.read_frame(-1)  # Negative index not supported

    def test_invalid_dcd_file(self, tmp_path):
        """Test that invalid DCD file raises ValueError."""
        dcd_path = tmp_path / "invalid.dcd"
        dcd_path.write_text("NOT A DCD FILE")

        with pytest.raises(ValueError, match="Invalid DCD file"):
            DCDReader(str(dcd_path))


class TestDCDReaderProperties:
    """Property access tests."""

    def test_timestep_property(self, tmp_path):
        """Test timestep property returns correct value."""
        atoms = build_atoms()
        dcd_path = tmp_path / "test.dcd"

        # Use larger timestep that produces non-zero DELTA
        # 2.5 ps = 2500 fs -> DELTA = 2
        with DCDWriter(str(dcd_path), natoms=3, timestep=2500.0) as w:
            w.write_frame(atoms, step=0)

        with DCDReader(str(dcd_path)) as reader:
            # DELTA = 2 -> timestep = 2000 fs (due to integer rounding)
            assert reader.timestep == pytest.approx(2000.0)

    def test_len_method(self, tmp_path):
        """Test __len__ returns number of frames."""
        atoms = build_atoms()
        dcd_path = tmp_path / "test.dcd"

        with DCDWriter(str(dcd_path), natoms=3, timestep=1.0) as w:
            for i in range(5):
                w.write_frame(atoms, step=i)

        with DCDReader(str(dcd_path)) as reader:
            assert len(reader) == 5


class TestReadDcdConvenience:
    """Tests for read_dcd convenience function."""

    def test_read_all_frames(self, tmp_path):
        """Test read_dcd without frame_index reads all frames."""
        atoms = build_atoms()
        dcd_path = tmp_path / "test.dcd"

        with DCDWriter(str(dcd_path), natoms=3, timestep=1.0) as w:
            for i in range(3):
                w.write_frame(atoms, step=i)

        frames = read_dcd(str(dcd_path))
        assert len(frames) == 3

    def test_read_single_frame(self, tmp_path):
        """Test read_dcd with frame_index reads single frame."""
        atoms = build_atoms()
        dcd_path = tmp_path / "test.dcd"

        with DCDWriter(str(dcd_path), natoms=3, timestep=1.0) as w:
            atoms[0].position = [0.0, 0.0, 0.0]
            w.write_frame(atoms, step=0)
            atoms[0].position = [1.0, 1.0, 1.0]
            w.write_frame(atoms, step=1)

        # Read frame 1
        frame = read_dcd(str(dcd_path), frame_index=1)
        assert len(frame) == 3
        assert frame.get_positions()[0] == pytest.approx([1.0, 1.0, 1.0], abs=1e-5)


class TestDCDRoundTrip:
    """Round-trip tests: write then read back."""

    def test_write_read_round_trip(self, tmp_path):
        """Test that positions survive write->read round trip."""
        atoms = build_atoms()
        original_positions = atoms.get_positions().copy()
        dcd_path = tmp_path / "test.dcd"

        with DCDWriter(str(dcd_path), natoms=3, timestep=1.0, is_periodic=True) as w:
            w.write_frame(atoms, step=0)

        with DCDReader(str(dcd_path)) as reader:
            frame = reader.read_frame(0)
            read_positions = frame.get_positions()

        # Positions should match within float32 precision
        assert np.allclose(original_positions, read_positions, atol=1e-5)

    def test_cell_round_trip(self, tmp_path):
        """Test that unit cell survives write->read round trip."""
        atoms = build_atoms()
        original_cell = atoms.get_cell().copy()
        dcd_path = tmp_path / "test.dcd"

        with DCDWriter(str(dcd_path), natoms=3, timestep=1.0, is_periodic=True) as w:
            w.write_frame(atoms, step=0)

        with DCDReader(str(dcd_path)) as reader:
            frame = reader.read_frame(0)
            read_cell = frame.get_cell()

        # Cell should match
        assert np.allclose(original_cell, read_cell, atol=1e-5)