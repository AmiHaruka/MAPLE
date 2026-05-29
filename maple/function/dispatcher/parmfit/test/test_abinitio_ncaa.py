from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest
from ase import Atoms

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maple.function.dispatcher.parmfit.utils import interface as interface_module
from maple.function.dispatcher.parmfit.utils.context import collect_environment_residues
from maple.function.dispatcher.parmfit.utils.model import model_to_atoms
from maple.function.dispatcher.parmfit.utils.NCAA import artifacts as ncaa_amber_module
from maple.function.dispatcher.parmfit.utils.NCAA import artifacts as ncaa_export_module
from maple.function.dispatcher.parmfit.utils.NCAA import artifacts as ncaa_mapping_module
from maple.function.dispatcher.parmfit.utils.NCAA import models as ncaa_build_module
from maple.function.dispatcher.parmfit.utils.NCAA import models as ncaa_graph_module
from maple.function.dispatcher.parmfit.utils.NCAA.models import NCAAConformer, NCAAIdentity, build_capped_ncaa_model, build_ncaa_sidechain_relax_indices, conformer_targets, detect_ncaa_chirality, infer_terminal_omit_names, minimize_conformer, warn_capped_proton_transfer
from maple.function.dispatcher.parmfit.utils.NCAA.config import (
    parse_ncaa_abinitio_config,
)
from maple.function.dispatcher.parmfit.utils.NCAA import report as ncaa_report_module
from maple.function.dispatcher.parmfit.utils.NCAA import workflow as ncaa_workflow_module
from maple.function.dispatcher.parmfit.utils.TorsionFit import TorsionWorkflowResult
from maple.function.dispatcher.parmfit.utils.outputparm import format_tleap_add_atom_types_lines
from maple.function.dispatcher.parmfit.utils.readparm import (
    Bond,
    CorrectionParameterSet,
    Dihedral,
    FourierTerm,
    FrcmodDB,
    Mol2Atom,
    Mol2Bond,
    Mol2Topology,
    Nonbond,
)
from maple.function.dispatcher.parmfit.utils.runtime import MultiRespPipelineResult
from maple.function.dispatcher.parmfit.utils.structure import get_resid_key, make_atom, make_residue, search_atom


class ZeroCalculator:
    def get_potential_energy(self, atoms, force_consistent=False):
        del atoms, force_consistent
        return 0.0

    def get_forces(self, atoms):
        return np.zeros((len(atoms), 3), dtype=float)

    def get_hessian(self, atoms):
        size = 3 * len(atoms)
        return np.eye(size, dtype=float)


def _protein_residue(*, chirality: str = "L", resname: str = "NAA") -> dict:
    cb_z = 1.0 if chirality == "L" else -1.0
    return make_residue(
        "A",
        1,
        "",
        resname,
        [
            make_atom(1, "N", "N", np.array((0.0, 1.0, 0.0))),
            make_atom(2, "CA", "C", np.array((0.0, 0.0, 0.0))),
            make_atom(3, "C", "C", np.array((-1.5, -0.2, 0.0))),
            make_atom(4, "O", "O", np.array((-2.5, -0.9, 0.0))),
            make_atom(5, "CB", "C", np.array((0.0, 0.0, cb_z))),
            make_atom(6, "H", "H", np.array((0.8, 0.5, 0.0))),
            make_atom(7, "HA", "H", np.array((0.8, -0.5, 0.0))),
        ],
        kind="protein",
    )


def _translated_residue(
    residue: dict,
    *,
    delta: tuple[float, float, float],
    resseq: int,
    resname: str | None = None,
    kind: str | None = None,
    serial_offset: int = 0,
) -> dict:
    shift = np.asarray(delta, dtype=float)
    copied = deepcopy(residue)
    copied["resseq"] = resseq
    if resname is not None:
        copied["resname"] = resname
    if kind is not None:
        copied["kind"] = kind
    for atom in copied["atoms"]:
        atom["serial"] += serial_offset
        atom["xyz"] = np.asarray(atom["xyz"], dtype=float) + shift
    copied["coords"] = np.asarray([atom["xyz"] for atom in copied["atoms"]], dtype=float)
    return copied


def _previous_residue_for_boundary(target: dict) -> dict:
    target_n = next(atom for atom in target["atoms"] if atom["name"] == "N")
    n_xyz = np.asarray(target_n["xyz"], dtype=float)
    return make_residue(
        target["chain"],
        target["resseq"] - 1,
        "",
        "GLY",
        [
            make_atom(101, "N", "N", n_xyz + np.array((0.0, 2.6, 0.0))),
            make_atom(102, "CA", "C", n_xyz + np.array((0.0, 1.3, 0.0))),
            make_atom(103, "C", "C", n_xyz + np.array((0.0, 0.0, 0.0))),
            make_atom(104, "O", "O", n_xyz + np.array((0.0, -0.8, 0.9))),
        ],
        kind="protein",
    )


def _leading_residue_for_boundary(previous: dict) -> dict:
    previous_n = next(atom for atom in previous["atoms"] if atom["name"] == "N")
    n_xyz = np.asarray(previous_n["xyz"], dtype=float)
    return make_residue(
        previous["chain"],
        previous["resseq"] - 1,
        "",
        "GLY",
        [
            make_atom(1, "N", "N", n_xyz + np.array((-2.4, 0.0, 0.0))),
            make_atom(2, "CA", "C", n_xyz + np.array((-1.2, 0.0, 0.0))),
            make_atom(3, "C", "C", n_xyz + np.array((0.0, 0.0, 0.0))),
            make_atom(4, "O", "O", n_xyz + np.array((0.0, -0.8, 0.9))),
        ],
        kind="protein",
    )


def _next_residue_for_boundary(target: dict, resname: str) -> dict:
    target_c = next(atom for atom in target["atoms"] if atom["name"] == "C")
    c_xyz = np.asarray(target_c["xyz"], dtype=float)
    if resname == "PRO":
        atoms = [
            make_atom(201, "N", "N", c_xyz + np.array((0.0, 0.0, 0.0))),
            make_atom(202, "CD", "C", c_xyz + np.array((0.0, 1.2, 0.0))),
            make_atom(203, "CA", "C", c_xyz + np.array((1.2, 0.0, 0.0))),
            make_atom(204, "C", "C", c_xyz + np.array((2.4, 0.0, 0.0))),
            make_atom(205, "O", "O", c_xyz + np.array((3.1, 0.6, 0.0))),
        ]
    else:
        atoms = [
            make_atom(201, "N", "N", c_xyz + np.array((0.0, 0.0, 0.0))),
            make_atom(202, "H", "H", c_xyz + np.array((0.0, 0.9, 0.0))),
            make_atom(203, "CA", "C", c_xyz + np.array((1.2, 0.0, 0.0))),
            make_atom(204, "C", "C", c_xyz + np.array((2.4, 0.0, 0.0))),
            make_atom(205, "O", "O", c_xyz + np.array((3.1, 0.6, 0.0))),
        ]
    return make_residue(target["chain"], target["resseq"] + 1, "", resname, atoms, kind="protein")


def _following_residue_for_boundary(previous: dict) -> dict:
    previous_c = next(atom for atom in previous["atoms"] if atom["name"] == "C")
    c_xyz = np.asarray(previous_c["xyz"], dtype=float)
    return make_residue(
        previous["chain"],
        previous["resseq"] + 1,
        "",
        "GLY",
        [
            make_atom(301, "N", "N", c_xyz + np.array((0.0, 0.0, 0.0))),
            make_atom(302, "H", "H", c_xyz + np.array((0.0, 0.9, 0.0))),
            make_atom(303, "CA", "C", c_xyz + np.array((1.2, 0.0, 0.0))),
            make_atom(304, "C", "C", c_xyz + np.array((2.4, 0.0, 0.0))),
            make_atom(305, "O", "O", c_xyz + np.array((3.1, 0.6, 0.0))),
        ],
        kind="protein",
    )


def _boundary_structure(
    target: dict,
    *,
    next_resname: str = "ALA",
    include_leading: bool = False,
    include_following: bool = True,
) -> dict:
    previous = _previous_residue_for_boundary(target)
    next_residue = _next_residue_for_boundary(target, next_resname)
    residues = [previous, deepcopy(target), next_residue]
    if include_leading:
        residues.insert(0, _leading_residue_for_boundary(previous))
    if include_following:
        residues.append(_following_residue_for_boundary(next_residue))
    return {"residues": residues}


def _maple_type_context_for_residue(residue: dict) -> tuple[dict[str, int], dict[int, str]]:
    names = [atom["name"] for atom in sorted(residue["atoms"], key=lambda atom: atom["serial"])]
    preferred = {
        "N": "Z0",
        "H": "Z1",
        "CA": "Z2",
        "HA": "Z3",
        "CB": "Z4",
        "C": "ZE",
        "O": "ZF",
    }
    name_to_global_index = {name: index for index, name in enumerate(names, start=1)}
    global_to_maple_type = {
        index: preferred.get(name, f"Z{index}")
        for name, index in name_to_global_index.items()
    }
    return name_to_global_index, global_to_maple_type


def _minimal_parameter_set() -> CorrectionParameterSet:
    mol2 = Mol2Topology(
        atoms=[
            Mol2Atom(atom_id=1, name="A1", atom_type="c", charge=0.0),
            Mol2Atom(atom_id=2, name="A2", atom_type="c", charge=0.0),
            Mol2Atom(atom_id=3, name="A3", atom_type="c", charge=0.0),
            Mol2Atom(atom_id=4, name="A4", atom_type="c", charge=0.0),
        ],
        bonds=[
            Mol2Bond(bond_id=1, atom1=1, atom2=2, bond_type="1"),
            Mol2Bond(bond_id=2, atom1=2, atom2=3, bond_type="1"),
            Mol2Bond(bond_id=3, atom1=3, atom2=4, bond_type="1"),
        ],
        id_to_index={1: 1, 2: 2, 3: 3, 4: 4},
        adjacency={1: {2}, 2: {1, 3}, 3: {2, 4}, 4: {3}},
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
                terms=[FourierTerm(kPhi=0.5, period=1.0, phase=0.0)],
            )
        ],
        impropers=[],
        nonbonds=[Nonbond(atom=index, atom_type="c", charge=0.0) for index in range(1, 5)],
        unmatched_bonds=[],
        unmatched_angles=[],
        unmatched_dihedrals=[],
        unmatched_impropers=[],
        unmatched_nonbonds=[],
    )


def _fixinternals_dihedrals_deg(constraint) -> list:
    if hasattr(constraint, "dihedrals_deg"):
        return constraint.dihedrals_deg
    return constraint.dihedrals


def test_parse_ncaa_abinitio_config_uses_current_defaults() -> None:
    config = parse_ncaa_abinitio_config(
        {"pdb": "demo.pdb", "target": "A1"},
        pdb_path="demo.pdb",
        target="A1",
        charge=0,
        mult=1,
    )

    assert config.rn == "MOL"
    assert not hasattr(config, "qm")
    assert config.resp.qm.theory == "HF"
    assert config.resp.qm.basis == "6-31G(d)"
    assert config.resp.qm.nproc == 8
    assert config.resp.qm.mem == 16
    assert config.vib_scale == pytest.approx(1.0)
    assert not hasattr(config, "max_iter")
    assert not hasattr(config, "max_step")
    assert config.watm == "tip3p"
    assert config.ionm == "12_6"
    assert config.resp.watm == "tip3p"
    assert config.prom == "ff14SB"
    assert config.resp.prom == "ff14SB"
    assert config.torsion.enabled
    assert config.torsion.torsion_steps == 36
    assert config.torsion.torsion_step_deg == pytest.approx(10.0)


@pytest.mark.parametrize(
    ("raw_prom", "expected"),
    [
        ("ff14SB", "ff14SB"),
        ("FF14SB", "ff14SB"),
        ("ff14sb", "ff14SB"),
        ("ff19SB", "ff19SB"),
        ("FF19SB", "ff19SB"),
    ],
)
def test_parse_ncaa_abinitio_config_accepts_prom(raw_prom: str, expected: str) -> None:
    config = parse_ncaa_abinitio_config(
        {"pdb": "demo.pdb", "target": "A1", "prom": raw_prom},
        pdb_path="demo.pdb",
        target="A1",
        charge=0,
        mult=1,
    )

    assert config.prom == expected
    assert config.resp.prom == expected

    with pytest.raises(ValueError, match="protein model"):
        parse_ncaa_abinitio_config(
            {"pdb": "demo.pdb", "target": "A1", "prom": "bad"},
            pdb_path="demo.pdb",
            target="A1",
            charge=0,
            mult=1,
        )


def test_parse_ncaa_abinitio_config_accepts_watm_and_ionm() -> None:
    config = parse_ncaa_abinitio_config(
        {"pdb": "demo.pdb", "target": "A1", "watm": "opc", "ionm": "hfe"},
        pdb_path="demo.pdb",
        target="A1",
        charge=0,
        mult=1,
    )

    assert config.watm == "opc"
    assert config.ionm == "hfe"
    assert config.resp.watm == "opc"

    with pytest.raises(ValueError, match="water model"):
        parse_ncaa_abinitio_config(
            {"pdb": "demo.pdb", "target": "A1", "watm": "bad"},
            pdb_path="demo.pdb",
            target="A1",
            charge=0,
            mult=1,
        )
    custom_ion_config = parse_ncaa_abinitio_config(
        {"pdb": "demo.pdb", "target": "A1", "ionm": "custom"},
        pdb_path="demo.pdb",
        target="A1",
        charge=0,
        mult=1,
    )
    assert custom_ion_config.ionm == "custom"


def test_parse_ncaa_abinitio_config_reads_shared_torsion_steps() -> None:
    config = parse_ncaa_abinitio_config(
        {"pdb": "demo.pdb", "target": "A1", "torsion_steps": 36},
        pdb_path="demo.pdb",
        target="A1",
        charge=0,
        mult=1,
    )

    assert config.torsion.torsion_steps == 36
    assert config.torsion.torsion_step_deg == pytest.approx(10.0)


def test_parse_ncaa_abinitio_config_reads_torsion_backend_without_breaking_qm_backend() -> None:
    config = parse_ncaa_abinitio_config(
        {"pdb": "demo.pdb", "target": "A1", "backend": "cgws", "constraint_mode": "projected"},
        pdb_path="demo.pdb",
        target="A1",
        charge=0,
        mult=1,
    )

    assert config.resp.qm.backend == "gaussian"
    assert config.torsion.backend == "cgws"
    assert config.torsion.constraint_mode == "projected"


def test_ncaa_report_start_lines_cover_current_summary_fields() -> None:
    identity = NCAAIdentity(residue_key=("A", 1, ""), resname="NAA", chirality="L", sidechain_anchor="CB")

    config = parse_ncaa_abinitio_config(
        {"pdb": "demo.pdb", "target": "A1"},
        pdb_path="demo.pdb",
        target="A1",
        charge=0,
        mult=1,
    )

    lines = "".join(
        ncaa_report_module.format_ncaa_start_lines(
            config=config,
            identity=identity,
        )
    )
    assert "NCAA target selector: A1" in lines
    assert "residue name: MOL" in lines
    assert "chirality: L" in lines
    assert "charge/mult: 0 1" in lines
    assert "QM ESP method: HF/6-31G(d)" in lines


def test_ncaa_identity_helpers_use_current_public_interface() -> None:
    structure_l = {"residues": [_protein_residue(chirality="L")]}
    structure_d = {"residues": [_protein_residue(chirality="D")]}

    assert detect_ncaa_chirality(structure_l["residues"][0]) == "L"
    assert detect_ncaa_chirality(structure_d["residues"][0]) == "D"
    assert conformer_targets("L") == [("alpha", -60.0, -40.0), ("beta", -120.0, -140.0)]
    assert conformer_targets("D") == [("alpha", 60.0, 40.0), ("beta", 120.0, 140.0)]


def test_build_capped_ncaa_model_uses_non_conflicting_cap_names() -> None:
    model = build_capped_ncaa_model(_protein_residue(chirality="L"), "NSR")
    ace_residue, target_residue, nme_residue = model["residues"]

    ace_names = {atom["name"] for atom in ace_residue["atoms"]}
    target_names = {atom["name"] for atom in target_residue["atoms"]}
    nme_names = {atom["name"] for atom in nme_residue["atoms"]}

    assert {"CMA", "CAC", "OAC", "H1A", "H2A", "H3A"} <= ace_names
    assert {"N", "CA", "C", "O"} <= target_names
    assert {"NNM", "CNM", "HNM", "H1M", "H2M", "H3M"} <= nme_names
    assert "C" not in ace_names
    assert "O" not in ace_names
    assert "N" not in nme_names


def test_build_ncaa_sidechain_relax_indices_are_one_based_r_group_atoms() -> None:
    target = _protein_residue(chirality="L")
    target["atoms"].append(make_atom(8, "HB", "H", np.array((0.0, 0.0, 2.0))))
    model = build_capped_ncaa_model(target, "NSR")
    residue_start = model["segment_sizes"]["ace"] + 1
    residue_atoms = sorted(model["residues"][1]["atoms"], key=lambda atom: atom["serial"])
    index_by_name = {atom["name"]: residue_start + offset for offset, atom in enumerate(residue_atoms)}

    assert build_ncaa_sidechain_relax_indices(model) == (
        index_by_name["CB"],
        index_by_name["HB"],
    )


def test_prepare_ncaa_models_uses_peptide_neighbors_for_caps(monkeypatch, tmp_path: Path) -> None:
    target = _protein_residue(chirality="L")
    previous = _previous_residue_for_boundary(target)
    next_residue = _next_residue_for_boundary(target, "ALA")
    structure = {"residues": [previous, target, next_residue], "explicit_pairs": set(), "_pair_cache": {}}
    captured: dict[str, dict] = {}

    def fake_optimize_capped_reference(model, **kwargs):
        captured["model"] = model
        captured["frozen_indices"] = kwargs["frozen_indices"]
        return NCAAConformer(label="ref", phi_deg=0.0, psi_deg=0.0, energy=0.0, model=model)

    monkeypatch.setattr(ncaa_workflow_module, "optimize_capped_reference", fake_optimize_capped_reference)
    monkeypatch.setattr(ncaa_workflow_module, "build_resp_conformers_from_reference", lambda *args, **kwargs: [])

    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    config = parse_ncaa_abinitio_config(
        {"target": "A1", "rn": "NSR"},
        pdb_path=str(tmp_path / "protein.pdb"),
        target="A1",
        charge=0,
        mult=1,
    )

    ncaa_workflow_module._prepare_ncaa_models(
        output=str(tmp_path / "ncaa.out"),
        source_atoms=atoms,
        structure=structure,
        target_residue=target,
        config=config,
    )

    ace_residue, _target_residue, nme_residue = captured["model"]["residues"]
    np.testing.assert_allclose(search_atom(ace_residue, "CAC")["xyz"], search_atom(previous, "C")["xyz"])
    np.testing.assert_allclose(search_atom(ace_residue, "CMA")["xyz"], search_atom(previous, "CA")["xyz"])
    np.testing.assert_allclose(search_atom(ace_residue, "OAC")["xyz"], search_atom(previous, "O")["xyz"])
    np.testing.assert_allclose(search_atom(nme_residue, "NNM")["xyz"], search_atom(next_residue, "N")["xyz"])
    np.testing.assert_allclose(search_atom(nme_residue, "CNM")["xyz"], search_atom(next_residue, "CA")["xyz"])
    sidechain_relax_indices = set(build_ncaa_sidechain_relax_indices(captured["model"]))
    atom_count = sum(len(residue["atoms"]) for residue in captured["model"]["residues"])
    assert set(captured["frozen_indices"]) == {
        index - 1 for index in range(1, atom_count + 1) if index not in sidechain_relax_indices
    }


def test_warn_capped_proton_transfer_flags_nme_hnm_shift(capsys) -> None:
    model = build_capped_ncaa_model(_protein_residue(chirality="L"), "NSR")
    nme = model["residues"][2]
    target = model["residues"][1]
    hnm = search_atom(nme, "HNM")
    o_atom = search_atom(target, "O")
    hnm["xyz"] = np.asarray(o_atom["xyz"], dtype=float) + np.array((0.0, 0.0, 0.95))

    conformer = NCAAConformer(label="alpha", phi_deg=-60.0, psi_deg=-40.0, energy=0.0, model=model)
    warn_capped_proton_transfer([conformer])
    captured = capsys.readouterr()
    assert "[WARNING] NCAA capped model alpha: possible cap proton transfer" in captured.out


def test_minimize_conformer_uses_cap_only_prescan_then_optimizes_last_frame(monkeypatch, tmp_path) -> None:
    model = build_capped_ncaa_model(_protein_residue(chirality="L"), "NSR")
    source_atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    source_atoms.calc = ZeroCalculator()
    start_atoms = model_to_atoms(model)
    captured: dict[str, object] = {}

    def fake_optimize_atoms_geometry(atoms, *, output, max_iter, max_step, failure_message=None):
        del max_iter, max_step, failure_message
        captured["opt_output"] = output
        captured["constraints_after_scan"] = getattr(atoms, "constraints", None)
        captured["positions_before_opt"] = atoms.positions.copy()
        atoms.positions += np.array([-0.05, 0.05, 0.02])
        return atoms

    monkeypatch.setattr(ncaa_build_module, "optimize_atoms_geometry", fake_optimize_atoms_geometry)

    conformer = minimize_conformer(
        model=model,
        label="alpha",
        phi_deg=-60.0,
        psi_deg=-40.0,
        source_atoms=source_atoms,
        output=str(tmp_path / "alpha.out"),
        max_iter=24,
        max_step=0.08,
    )

    index_map = ncaa_build_module._build_backbone_rotation_map(model)
    current_phi = ncaa_build_module._measure_dihedral(start_atoms, index_map["phi"])
    current_psi = ncaa_build_module._measure_dihedral(start_atoms, index_map["psi"])
    expected_phi_delta = ((-60.0 - current_phi + 180.0) % 360.0) - 180.0
    expected_psi_delta = ((-40.0 - current_psi + 180.0) % 360.0) - 180.0
    guess_xyz = tmp_path / "alpha_guess.xyz"
    assert guess_xyz.exists()
    assert sum(1 for line in guess_xyz.read_text(encoding="utf-8").splitlines() if line.isdigit()) == 1

    ace_count = model["segment_sizes"]["ace"]
    target_count = model["segment_sizes"]["residue"]
    target_slice = slice(ace_count, ace_count + target_count)
    assert np.allclose(captured["positions_before_opt"][target_slice], start_atoms.positions[target_slice])
    assert str(captured["opt_output"]).endswith("alpha_opt.out")
    constraint = captured["constraints_after_scan"][0]
    assert constraint.__class__.__name__ == "FixInternals"
    nme_residue = model["residues"][2]
    index_by_serial = {atom["serial"]: index for index, (_residue, atom) in enumerate(ncaa_build_module.flatten_model_atoms(model))}
    nnm_index = index_by_serial[search_atom(nme_residue, "NNM")["serial"]]
    hnm_index = index_by_serial[search_atom(nme_residue, "HNM")["serial"]]
    nme_nh_distance = np.linalg.norm(start_atoms.positions[nnm_index] - start_atoms.positions[hnm_index])
    assert constraint.bonds == [[pytest.approx(nme_nh_distance), [nnm_index, hnm_index]]]
    dihedrals = _fixinternals_dihedrals_deg(constraint)
    assert dihedrals[0][0] == pytest.approx(current_phi + expected_phi_delta)
    assert dihedrals[0][1] == list(index_map["phi"])
    assert dihedrals[1][0] == pytest.approx(current_psi + expected_psi_delta)
    assert dihedrals[1][1] == list(index_map["psi"])

    minimized_atoms = model_to_atoms(conformer.model)
    assert np.allclose(minimized_atoms.positions[target_slice], start_atoms.positions[target_slice] + np.array([-0.05, 0.05, 0.02]))
    assert conformer.label == "alpha"
    assert conformer.phi_deg == -60.0
    assert conformer.psi_deg == -40.0
    assert conformer.energy == pytest.approx(0.0)


def test_minimize_conformer_resolves_scan_target_to_nearest_360_equivalent(monkeypatch, tmp_path) -> None:
    model = build_capped_ncaa_model(_protein_residue(chirality="L"), "NSR")
    source_atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    source_atoms.calc = ZeroCalculator()
    captured: dict[str, object] = {}
    measured_angles = [300.0, 320.0]

    def fake_optimize_atoms_geometry(atoms, *, output, max_iter, max_step, failure_message=None):
        del output, max_iter, max_step, failure_message
        captured["constraints_after_scan"] = getattr(atoms, "constraints", None)
        return atoms

    monkeypatch.setattr(ncaa_build_module, "optimize_atoms_geometry", fake_optimize_atoms_geometry)
    monkeypatch.setattr(ncaa_build_module, "_measure_dihedral", lambda atoms, indices: measured_angles.pop(0) if measured_angles else 300.0)

    minimize_conformer(
        model=model,
        label="alpha",
        phi_deg=-60.0,
        psi_deg=-40.0,
        source_atoms=source_atoms,
        output=str(tmp_path / "alpha_equiv.out"),
        max_iter=24,
        max_step=0.08,
    )

    constraint = captured["constraints_after_scan"][0]
    assert constraint.__class__.__name__ == "FixInternals"
    dihedrals = _fixinternals_dihedrals_deg(constraint)
    assert dihedrals[0][0] == pytest.approx(300.0)
    assert dihedrals[1][0] == pytest.approx(320.0)


def test_infer_terminal_omit_names_keeps_backbone_h_and_o_but_drops_terminal_extras() -> None:
    residue = make_residue(
        "A",
        1,
        "",
        "NAA",
        [
            make_atom(1, "N", "N", np.array((0.0, 0.0, 0.0))),
            make_atom(2, "H", "H", np.array((0.0, 1.0, 0.0))),
            make_atom(3, "H2", "H", np.array((-0.8, -0.8, 0.0))),
            make_atom(4, "H3", "H", np.array((0.8, -0.8, 0.0))),
            make_atom(5, "CA", "C", np.array((1.4, 0.0, 0.0))),
            make_atom(6, "CB", "C", np.array((1.4, 0.0, 1.2))),
            make_atom(7, "C", "C", np.array((2.8, 0.0, 0.0))),
            make_atom(8, "O", "O", np.array((3.9, 0.0, 0.0))),
            make_atom(9, "OXT", "O", np.array((2.8, 1.2, 0.0))),
            make_atom(10, "HXT", "H", np.array((2.8, 2.1, 0.0))),
        ],
        kind="protein",
    )

    assert infer_terminal_omit_names(residue) == ["H2", "H3", "HXT", "OXT"]


def test_collect_environment_residues_filters_target_and_excluded_keys() -> None:
    target = _protein_residue()
    near_protein = _translated_residue(_protein_residue(resname="GLY"), delta=(3.0, 0.0, 0.0), resseq=2, resname="GLY")
    near_ligand = make_residue(
        "B",
        1,
        "",
        "LIG",
        [
            make_atom(101, "C1", "C", np.array((0.0, 2.8, 0.0))),
            make_atom(102, "O1", "O", np.array((0.8, 3.5, 0.0))),
        ],
        kind="ligand",
    )
    near_water = make_residue(
        "A",
        201,
        "",
        "HOH",
        [make_atom(103, "O", "O", np.array((0.0, 0.0, 3.2)))],
        kind="water",
    )
    far_residue = _translated_residue(_protein_residue(resname="SER"), delta=(10.0, 0.0, 0.0), resseq=9, resname="SER")
    structure = {"residues": [target, near_protein, near_ligand, near_water, far_residue]}

    environment = collect_environment_residues(
        structure,
        target,
        cutoff=4.0,
        excluded_keys={get_resid_key(near_protein)},
    )

    assert [get_resid_key(residue) for residue in environment] == [
        get_resid_key(near_ligand),
        get_resid_key(near_water),
    ]


def test_collect_environment_residues_can_exclude_water() -> None:
    target = _protein_residue()
    near_protein = _translated_residue(_protein_residue(resname="GLY"), delta=(3.0, 0.0, 0.0), resseq=2, resname="GLY")
    near_ligand = make_residue(
        "B",
        1,
        "",
        "LIG",
        [
            make_atom(101, "C1", "C", np.array((0.0, 2.8, 0.0))),
            make_atom(102, "O1", "O", np.array((0.8, 3.5, 0.0))),
        ],
        kind="ligand",
    )
    near_water = make_residue(
        "A",
        201,
        "",
        "HOH",
        [make_atom(103, "O", "O", np.array((0.0, 0.0, 3.2)))],
        kind="water",
    )
    near_ion = make_residue(
        "A",
        202,
        "",
        "ZN",
        [make_atom(104, "ZN", "ZN", np.array((0.0, 3.1, 0.0)))],
        kind="ion",
    )
    structure = {"residues": [target, near_protein, near_ligand, near_water, near_ion]}

    environment = collect_environment_residues(
        structure,
        target,
        cutoff=4.0,
        include_water=False,
    )

    assert [get_resid_key(residue) for residue in environment] == [
        get_resid_key(near_protein),
        get_resid_key(near_ligand),
        get_resid_key(near_ion),
    ]


def test_build_ncaa_amber_artifacts_writes_expected_outputs(monkeypatch, tmp_path: Path) -> None:
    representative_model = build_capped_ncaa_model(_protein_residue(chirality="L"), "NAA")
    representative_model["charge"] = 0
    representative_model["mult"] = 1
    charged_residue = representative_model["residues"][1]

    config = parse_ncaa_abinitio_config(
        {"pdb": str(tmp_path / "ncaa.pdb"), "target": "A1", "rn": "NAA"},
        pdb_path=str(tmp_path / "ncaa.pdb"),
        target="A1",
        charge=0,
        mult=1,
    )
    mol2_path = tmp_path / "capped.mol2"
    mol2_path.write_text("@<TRIPOS>MOLECULE\nNAA\n", encoding="utf-8")
    charge_path = tmp_path / "target.chg"
    charge_path.write_text("", encoding="utf-8")
    resp_result = MultiRespPipelineResult(
        model=deepcopy(representative_model),
        files={"mol2": str(mol2_path)},
        resp_files={"target_chg": str(charge_path)},
        conformers={},
        decision=interface_module.set_method({"theory": "HF", "basis": "6-31G(d)", "nproc": 8, "mem": 16, "route": ""}),
    )

    def fake_run_antechamber(input_file, cfg, workdir, *, input_format="gout", output_format="ac", charge_mode=None, charge_file=None):
        del input_file, input_format, charge_mode, charge_file
        suffix = "mol2" if output_format == "mol2" else "ac"
        ac_path = Path(workdir) / f"{cfg['residue_name']}.{suffix}"
        atom_names = [
            atom["name"]
            for capped_residue in representative_model["residues"]
            for atom in sorted(capped_residue["atoms"], key=lambda item: item["serial"])
        ]
        if output_format == "mol2":
            ac_path.write_text("@<TRIPOS>MOLECULE\nNAA\n", encoding="utf-8")
        else:
            ac_path.write_text(
                "".join(f"ATOM  {index:5d} {name:<4s} c3\n" for index, name in enumerate(atom_names, start=1)),
                encoding="utf-8",
            )
        return interface_module.AntechamberResult(
            ac_path=str(ac_path),
            input_path=str(ac_path),
            input_format=output_format,
            residue_name=cfg["residue_name"],
        )

    def fake_run_prepgen(ac_file, mc_file, cfg, workdir):
        del ac_file, mc_file
        prepin = Path(workdir) / f"{cfg['residue_name']}.prepin"
        res = Path(workdir) / f"{cfg['residue_name']}.res"
        newpdb = Path(workdir) / "NEWPDB.PDB"
        for path in (prepin, res, newpdb):
            path.write_text("", encoding="utf-8")
        return interface_module.PrepgenResult(
            prepin_path=str(prepin),
            res_path=str(res),
            newpdb_path=str(newpdb),
            mainchain_path=str(Path(workdir) / f"{cfg['residue_name']}.mc"),
            residue_name=cfg["residue_name"],
        )

    def fake_run_parmchk2(input_file, cfg, ifmol2, workdir):
        del input_file, ifmol2
        frcmod = Path(workdir) / f"{cfg['residue_name']}.frcmod"
        frcmod.write_text("", encoding="utf-8")
        return interface_module.Parmchk2Result(
            frcmod_path=str(frcmod),
            input_path=str(frcmod),
            residue_name=cfg["residue_name"],
        )

    monkeypatch.setattr(ncaa_amber_module.amber_interface, "run_antechamber", fake_run_antechamber)
    monkeypatch.setattr(ncaa_amber_module.amber_interface, "run_prepgen", fake_run_prepgen)
    monkeypatch.setattr(ncaa_amber_module.amber_interface, "run_parmchk2", fake_run_parmchk2)

    artifacts = ncaa_amber_module.build_ncaa_amber_artifacts(
        output=str(tmp_path / "ncaa.out"),
        representative_model=representative_model,
        charged_residue=charged_residue,
        resp_result=resp_result,
        config=config,
    )

    assert Path(artifacts.prepin).is_file()
    assert Path(artifacts.frcmod).is_file()
    assert Path(artifacts.gaff2_mol2).is_file()
    assert Path(artifacts.prepin).parent == tmp_path / "ncaa_work"
    assert Path(artifacts.frcmod).parent == tmp_path / "ncaa_work"
    assert Path(artifacts.mc).parent.name == "ncaa"
    mainchain_lines = Path(artifacts.mc).read_text(encoding="utf-8").splitlines()
    assert "HEAD_NAME N" in mainchain_lines
    assert "TAIL_NAME C" in mainchain_lines
    assert "MAIN_CHAIN CA" in mainchain_lines
    assert "MAIN_CHAIN N" not in mainchain_lines
    assert "MAIN_CHAIN C" not in mainchain_lines
    assert "OMIT_NAME N" not in mainchain_lines
    assert "OMIT_NAME CA" not in mainchain_lines
    assert "OMIT_NAME C" not in mainchain_lines


def test_export_ncaa_artifacts_writes_processed_tleap_pdb_and_loads_it(tmp_path: Path) -> None:
    target = _translated_residue(_protein_residue(resname="VAL"), delta=(0.0, 0.0, 0.0), resseq=89)
    decoy = _translated_residue(target, delta=(5.0, 0.0, 0.0), resseq=90, resname="GLY", serial_offset=20)
    structure = {"residues": [target, decoy]}
    representative_model = build_capped_ncaa_model(target, "NSR")
    amber = ncaa_amber_module.NCAAAmberArtifacts(
        capped_mol2=str(tmp_path / "NSR.mol2"),
        gaff2_mol2=str(tmp_path / "NSR_gaff2.mol2"),
        ac=str(tmp_path / "NSR.ac"),
        mc=str(tmp_path / "NSR.mc"),
        prepin=str(tmp_path / "NSR.prepin"),
        refined_prepin=str(tmp_path / "NSR_maple.prepin"),
        res=str(tmp_path / "NSR.res"),
        newpdb=str(tmp_path / "NEWPDB.PDB"),
        frcmod=str(tmp_path / "NSR.frcmod"),
        refined_frcmod=str(tmp_path / "NSR_maple.frcmod"),
    )
    config = parse_ncaa_abinitio_config(
        {"pdb": str(tmp_path / "protein.pdb"), "target": "A1", "rn": "NSR", "watm": "opc", "ionm": "hfe"},
        pdb_path=str(tmp_path / "protein.pdb"),
        target="A1",
        charge=0,
        mult=1,
    )

    artifacts = ncaa_amber_module.export_ncaa_artifacts(
        str(tmp_path / "nacc.out"),
        amber=amber,
        representative_model=representative_model,
        conformers=[],
        config=config,
        atom_type_rows=[],
        structure=structure,
        target_residue=target,
    )

    tleap_pdb = Path(artifacts.files["tleap_pdb"])
    assert tleap_pdb.is_file()
    pdb_text = tleap_pdb.read_text(encoding="utf-8")
    assert " NSR A   1" in pdb_text
    assert " GLY A   2" in pdb_text
    assert " VAL A   1" not in pdb_text
    assert "TER" in pdb_text
    assert pdb_text.rstrip().endswith("END")

    tleap_text = Path(artifacts.files["tleap_input"]).read_text(encoding="utf-8")
    assert "source leaprc.water.opc" in tleap_text
    assert "loadamberparams frcmod.ionslm_hfe_opc" in tleap_text
    assert "solvatebox mol OPCBOX 10.0" in tleap_text
    assert "mol = loadpdb nacc_ncaa_tleap.pdb" in tleap_text
    assert "mol = loadpdb protein.pdb" not in tleap_text


def test_ncaa_workflow_runs_current_stage_pipeline_and_delegates_torsion(monkeypatch, tmp_path: Path) -> None:
    residue = _protein_residue()
    decoy_residue = make_residue(
        "A",
        2,
        "",
        "GLY",
        [
            make_atom(11, "N", "N", np.array((3.0, 1.0, 0.0))),
            make_atom(12, "CA", "C", np.array((4.0, 1.0, 0.0))),
            make_atom(13, "C", "C", np.array((5.0, 1.0, 0.0))),
            make_atom(14, "O", "O", np.array((6.0, 1.0, 0.0))),
        ],
        kind="protein",
    )
    structure = {"residues": [decoy_residue, residue], "explicit_pairs": set(), "_pair_cache": {}}

    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    atoms.info["charge"] = 0
    atoms.info["mult"] = 1
    atoms.calc = ZeroCalculator()
    atoms.f_max_th = 0.00285
    atoms.f_rms_th = 0.00190
    atoms.dp_max_th = 0.00315
    atoms.dp_rms_th = 0.00210

    config = parse_ncaa_abinitio_config(
        {"pdb": str(tmp_path / "ncaa.pdb"), "target": "A1", "rn": "NAA"},
        pdb_path=str(tmp_path / "ncaa.pdb"),
        target="A1",
        charge=0,
        mult=1,
    )
    config.torsion.torsion_ensemble = False

    minimized_model = {
        "name": "ncaa_capped_model",
        "target_key": ("A", 1, ""),
        "residues": [
            make_residue("A", 0, "", "ACE", [make_atom(11, "CMA", "C", np.array((1.3, 2.2, 0.0))), make_atom(12, "CAC", "C", np.array((0.6, 1.4, 0.0))), make_atom(13, "OAC", "O", np.array((0.7, 0.2, 0.0)))], kind="cap"),
            deepcopy(residue),
            make_residue("A", 2, "", "NME", [make_atom(14, "NNM", "N", np.array((-2.0, 0.1, 0.0))), make_atom(15, "CNM", "C", np.array((-3.0, 0.5, 0.0))), make_atom(16, "HNM", "H", np.array((-1.9, -0.9, 0.0)))], kind="cap"),
        ],
        "segment_sizes": {"ace": 3, "residue": len(residue["atoms"]), "nme": 3},
        "charge": 0,
        "mult": 1,
    }

    def fake_optimize_capped_reference(*args, **kwargs):
        del args, kwargs
        return NCAAConformer(label="reference", phi_deg=0.0, psi_deg=0.0, energy=-2.0, model=deepcopy(minimized_model))

    def fake_build_resp_conformers_from_reference(*args, **kwargs):
        del args, kwargs
        return [
            NCAAConformer(label="alpha", phi_deg=-60.0, psi_deg=-40.0, energy=0.0, model=deepcopy(minimized_model)),
            NCAAConformer(label="beta", phi_deg=-120.0, psi_deg=-140.0, energy=1.0, model=deepcopy(minimized_model)),
        ]

    def fake_run_multiconformer_resp(**kwargs):
        assert kwargs["qm"].theory == "HF"
        mol2_path = tmp_path / "capped.mol2"
        mol2_path.write_text("@<TRIPOS>MOLECULE\nNAA\n", encoding="utf-8")
        resp_files = {
            "alpha_gaussian_input": str(tmp_path / "alpha.com"),
            "alpha_gaussian_log": str(tmp_path / "alpha.log"),
            "alpha_esp": str(tmp_path / "alpha.esp"),
            "beta_gaussian_input": str(tmp_path / "beta.com"),
            "beta_gaussian_log": str(tmp_path / "beta.log"),
            "beta_esp": str(tmp_path / "beta.esp"),
            "all_esp": str(tmp_path / "all.esp"),
            "resp1_in": str(tmp_path / "resp1.in"),
            "resp1_out": str(tmp_path / "resp1.out"),
            "resp1_pch": str(tmp_path / "resp1.pch"),
            "resp1_chg": str(tmp_path / "resp1.chg"),
            "resp1_calc_esp": str(tmp_path / "resp1_calc.esp"),
            "resp2_in": str(tmp_path / "resp2.in"),
            "resp2_out": str(tmp_path / "resp2.out"),
                "resp2_pch": str(tmp_path / "resp2.pch"),
                "resp2_chg": str(tmp_path / "resp2.chg"),
                "resp2_calc_esp": str(tmp_path / "resp2_calc.esp"),
                "representative_chg": str(tmp_path / "reference.chg"),
                "target_chg": str(tmp_path / "reference.chg"),
            }
        for path in resp_files.values():
            Path(path).write_text("", encoding="utf-8")
        return MultiRespPipelineResult(
            model=deepcopy(minimized_model),
            files={"mol2": str(mol2_path)},
            resp_files=resp_files,
            conformers={"alpha": {}, "beta": {}},
            decision=interface_module.set_method({"theory": "HF", "basis": "6-31G(d)", "nproc": 8, "mem": 16, "route": ""}),
        )

    def fake_run_antechamber(input_file, cfg, workdir, *, input_format="gout", output_format="ac", charge_mode=None, charge_file=None):
        del input_file, input_format, charge_mode, charge_file
        suffix = "mol2" if output_format == "mol2" else "ac"
        ac_path = Path(workdir) / f"{cfg['residue_name']}.{suffix}"
        atom_names = [
            atom["name"]
            for capped_residue in minimized_model["residues"]
            for atom in sorted(capped_residue["atoms"], key=lambda item: item["serial"])
        ]
        if output_format == "mol2":
            ac_path.write_text("@<TRIPOS>MOLECULE\nNAA\n", encoding="utf-8")
        else:
            ac_path.write_text(
                "".join(f"ATOM  {index:5d} {name:<4s} c3\n" for index, name in enumerate(atom_names, start=1)),
                encoding="utf-8",
            )
        return interface_module.AntechamberResult(
            ac_path=str(ac_path),
            input_path=str(ac_path),
            input_format=output_format,
            residue_name=cfg["residue_name"],
        )

    def fake_run_prepgen(ac_file, mc_file, cfg, workdir):
        del ac_file, mc_file
        prepin = Path(workdir) / f"{cfg['residue_name']}.prepin"
        res = Path(workdir) / f"{cfg['residue_name']}.res"
        newpdb = Path(workdir) / "NEWPDB.PDB"
        for path in (prepin, res, newpdb):
            path.write_text("", encoding="utf-8")
        return interface_module.PrepgenResult(
            prepin_path=str(prepin),
            res_path=str(res),
            newpdb_path=str(newpdb),
            mainchain_path=str(Path(workdir) / "mainchain.mc"),
            residue_name=cfg["residue_name"],
        )

    def fake_run_parmchk2(input_file, cfg, ifmol2, workdir):
        del ifmol2
        frcmod = Path(workdir) / f"{cfg['residue_name']}.frcmod"
        frcmod.write_text("", encoding="utf-8")
        return interface_module.Parmchk2Result(
            frcmod_path=str(frcmod),
            input_path=str(input_file),
            residue_name=cfg["residue_name"],
        )

    torsion_calls = {"count": 0}
    fake_parameter_set = _minimal_parameter_set()

    def fake_run_torsion_workflow(*, atoms, output, parameter_set, params, runtime, center_bond_filter=None, mobile_atoms=None):
        del atoms, output
        torsion_calls["count"] += 1
        assert parameter_set is fake_parameter_set
        assert params is config.torsion
        assert runtime.backend == config.torsion.backend
        assert runtime.constraint_mode == config.torsion.constraint_mode
        assert mobile_atoms == (11,)
        assert center_bond_filter is not None
        assert center_bond_filter((5, 8))
        assert not center_bond_filter((4, 5))
        assert not center_bond_filter((5, 6))
        assert not center_bond_filter((1, 2))
        return TorsionWorkflowResult(
            stage1_parameter_set=fake_parameter_set,
            final_parameter_set=fake_parameter_set,
            refine_cycles=[],
            scan_xyz={},
            center_bonds=[(4, 5)],
            warnings=[],
        )

    monkeypatch.setattr(ncaa_workflow_module, "optimize_capped_reference", fake_optimize_capped_reference)
    monkeypatch.setattr(ncaa_workflow_module, "build_resp_conformers_from_reference", fake_build_resp_conformers_from_reference)
    monkeypatch.setattr(ncaa_workflow_module, "run_multiconformer_resp", fake_run_multiconformer_resp)
    monkeypatch.setattr(ncaa_workflow_module, "run_torsion_workflow", fake_run_torsion_workflow)
    monkeypatch.setattr(ncaa_workflow_module, "build_correction_parameter_set", lambda *args, **kwargs: fake_parameter_set)
    monkeypatch.setattr(ncaa_workflow_module, "apply_mseminario", lambda *args, **kwargs: None)
    monkeypatch.setattr(ncaa_amber_module.amber_interface, "run_antechamber", fake_run_antechamber)
    monkeypatch.setattr(ncaa_amber_module.amber_interface, "run_prepgen", fake_run_prepgen)
    monkeypatch.setattr(ncaa_amber_module.amber_interface, "run_parmchk2", fake_run_parmchk2)
    monkeypatch.setattr(ncaa_workflow_module.amber_interface, "patch_frcmod_crossterms", lambda *args, **kwargs: None)
    logged_blocks: list[str] = []

    def fake_build_ncaa_export_bundle(
        output,
        *,
        amber,
        representative_model,
        conformers,
        final_parameter_set,
        structure,
        target_residue,
        **kwargs,
    ):
        del final_parameter_set, kwargs
        Path(amber.refined_prepin).write_text("", encoding="utf-8")
        Path(amber.refined_frcmod).write_text("", encoding="utf-8")
        atom_type_rows = [
            ncaa_export_module.AtomTypeRow("N", "N", "ns", "Z0", -0.123456),
            ncaa_export_module.AtomTypeRow("CA", "C", "c3", "Z1", 0.234567),
        ]
        artifacts = ncaa_amber_module.export_ncaa_artifacts(
            output,
            amber=amber,
            representative_model=representative_model,
            conformers=conformers,
            config=config,
            atom_type_rows=atom_type_rows,
            structure=structure,
            target_residue=target_residue,
        )
        return ncaa_mapping_module.NCAAExportBundle(
            amber=amber,
            atom_type_rows=atom_type_rows,
            artifacts=artifacts,
        )

    monkeypatch.setattr(ncaa_workflow_module, "build_ncaa_export_bundle", fake_build_ncaa_export_bundle)
    tleap_calls: list[tuple[str, str | None]] = []

    def fake_run_tleap(input_file, workdir=None, executable="tleap"):
        del executable
        tleap_calls.append((input_file, workdir))
        output_path = Path(workdir or Path(input_file).parent) / (Path(input_file).stem + ".out")
        output_path.write_text("Errors = 0; Warnings = 1; Notes = 2\n", encoding="utf-8")
        return interface_module.TleapResult(
            input_path=str(Path(input_file).resolve()),
            output_path=str(output_path),
            returncode=0,
            command=f"tleap -s -f {Path(input_file).name}",
        )

    monkeypatch.setattr(ncaa_workflow_module.amber_interface, "run_tleap", fake_run_tleap)

    result = ncaa_workflow_module.run_ncaa_abinitio(
        output=str(tmp_path / "ncaa.out"),
        source_atoms=atoms,
        structure=structure,
        target_residue=residue,
        config=config,
        log_info=lambda lines: logged_blocks.append("".join(lines)),
    )

    assert torsion_calls["count"] == 1
    assert result.identity.residue_key == ("A", 1, "")
    assert result.chirality == "L"
    assert result.representative_conformer == "reference"
    assert result.parameter_set is fake_parameter_set
    assert result.torsion.center_bonds == [(4, 5)]
    assert Path(result.files["prepin"]).is_file()
    assert Path(result.files["frcmod"]).is_file()
    assert Path(result.files["refined_prepin"]).is_file()
    assert Path(result.files["refined_frcmod"]).is_file()
    assert Path(result.files["res"]).is_file()
    assert Path(result.files["newpdb"]).is_file()
    assert Path(result.files["target_capped_pdb"]).parent.name == "ncaa"
    assert Path(result.files["alpha_capped_pdb"]).parent.name == "ncaa"
    assert Path(result.files["beta_capped_pdb"]).parent.name == "ncaa"
    assert Path(result.files["prepin"]).parent == tmp_path / "ncaa_work"
    assert Path(result.files["frcmod"]).parent == tmp_path / "ncaa_work"
    assert Path(result.files["refined_prepin"]).parent == tmp_path / "ncaa_work"
    assert Path(result.files["refined_frcmod"]).parent == tmp_path / "ncaa_work"
    assert Path(result.files["mc"]).parent.name == "ncaa"
    assert Path(result.files["res"]).parent.name == "ncaa"
    assert Path(result.files["newpdb"]).parent.name == "ncaa"
    assert Path(result.files["gaff2_mol2"]).parent.name == "ncaa"
    mainchain_lines = Path(result.files["mc"]).read_text(encoding="utf-8").splitlines()
    assert "HEAD_NAME N" in mainchain_lines
    assert "TAIL_NAME C" in mainchain_lines
    assert "MAIN_CHAIN CA" in mainchain_lines
    assert "MAIN_CHAIN N" not in mainchain_lines
    assert "MAIN_CHAIN C" not in mainchain_lines
    assert "OMIT_NAME N" not in mainchain_lines
    assert "OMIT_NAME CA" not in mainchain_lines
    assert "OMIT_NAME C" not in mainchain_lines
    assert Path(result.files["gaff2_mol2"]).is_file()
    assert Path(result.files["target_capped_pdb"]).is_file()
    assert Path(result.files["tleap_input"]).is_file()
    assert Path(result.files["tleap_pdb"]).is_file()
    tleap_text = Path(result.files["tleap_input"]).read_text(encoding="utf-8")
    assert "source leaprc.protein.ff14SB" in tleap_text
    assert "source leaprc.water.tip3p" in tleap_text
    assert "loadamberprep NAA_maple.prepin" in tleap_text
    assert "loadamberparams NAA_maple.frcmod" in tleap_text
    assert "loadamberparams frcmod.ions1lm_126_tip3p" in tleap_text
    assert "mol = loadpdb ncaa_ncaa_tleap.pdb" in tleap_text
    assert "mol = loadpdb ncaa.pdb" not in tleap_text
    assert "solvatebox mol TIP3PBOX 10.0" in tleap_text
    assert tleap_calls == [(result.files["tleap_input"], str(Path(result.files["tleap_input"]).parent))]
    joined_logs = "\n".join(logged_blocks)
    assert "[NCAA] model preparation + reference optimization ..." in joined_logs
    assert "[NCAA] multiconformer RESP ..." in joined_logs
    assert "[NCAA] tleap validation ..." in joined_logs
    assert "[NCAA] route completed; final summary follows." in joined_logs


def test_build_ncaa_center_bond_filter_selects_sidechain_rotors() -> None:
    representative_model = build_capped_ncaa_model(_protein_residue(chirality="L"), "NSR")
    residue_atoms = sorted(representative_model["residues"][1]["atoms"], key=lambda atom: atom["serial"])
    residue_start = representative_model["segment_sizes"]["ace"] + 1
    index_by_name = {
        atom["name"]: residue_start + offset
        for offset, atom in enumerate(residue_atoms)
    }

    keep = ncaa_graph_module.build_ncaa_center_bond_filter(representative_model)

    assert keep((index_by_name["CA"], index_by_name["CB"]))
    assert not keep((index_by_name["N"], index_by_name["CA"]))
    assert not keep((index_by_name["CA"], index_by_name["C"]))
    assert not keep((1, 2))


def test_infer_mainchain_names_uses_internal_shortest_path() -> None:
    residue = make_residue(
        "A",
        1,
        "",
        "CRX",
        [
            make_atom(1, "N", "N", np.array((0.0, 0.0, 0.0))),
            make_atom(2, "CA1", "C", np.array((1.4, 0.0, 0.0))),
            make_atom(3, "C1", "C", np.array((2.8, 0.0, 0.0))),
            make_atom(4, "N3", "N", np.array((4.2, 0.0, 0.0))),
            make_atom(5, "CA3", "C", np.array((5.6, 0.0, 0.0))),
            make_atom(6, "C", "C", np.array((7.0, 0.0, 0.0))),
            make_atom(7, "CB", "C", np.array((1.4, 1.2, 0.0))),
            make_atom(8, "O1", "O", np.array((2.8, -1.2, 0.0))),
        ],
        kind="protein",
    )

    assert ncaa_graph_module.infer_mainchain_names(residue) == ["CA1", "C1", "N3", "CA3"]


def test_write_ncaa_amber_files_generates_refined_prepin_and_residue_only_frcmod(tmp_path: Path) -> None:
    target_residue = _protein_residue(chirality="L")
    representative_model = build_capped_ncaa_model(target_residue, "NSR")
    charged_residue = representative_model["residues"][1]
    residue_atoms = sorted(charged_residue["atoms"], key=lambda atom: atom["serial"])
    residue_start = representative_model["segment_sizes"]["ace"] + 1

    atom_names = [
        atom["name"]
        for residue in representative_model["residues"]
        for atom in sorted(residue["atoms"], key=lambda item: item["serial"])
    ]
    atom_types = [f"t{index}" for index in range(1, len(atom_names) + 1)]
    nonbonds = [
        Nonbond(atom=index, atom_type=atom_types[index - 1], charge=0.1 * index, rmin_half=1.5, epsilon=0.2)
        for index in range(1, len(atom_names) + 1)
    ]
    parameter_set = CorrectionParameterSet(
        mol2=Mol2Topology(
            atoms=[
                Mol2Atom(atom_id=index, name=name, atom_type=atom_type, charge=0.0)
                for index, (name, atom_type) in enumerate(zip(atom_names, atom_types), start=1)
            ],
            bonds=[],
            id_to_index={index: index for index in range(1, len(atom_names) + 1)},
            adjacency={},
        ),
        frcmod=FrcmodDB(mass_params={atom_type: 12.01 + index for index, atom_type in enumerate(atom_types)}),
        bonds=[
            Bond(
                atoms=(residue_start, residue_start + 1),
                atom_types=(atom_types[residue_start - 1], atom_types[residue_start]),
                kBond=300.0,
                rEq=1.45,
            ),
            Bond(
                atoms=(1, 2),
                atom_types=(atom_types[0], atom_types[1]),
                kBond=300.0,
                rEq=1.45,
            ),
        ],
        angles=[],
        dihedrals=[
            Dihedral(
                atoms=(residue_start, residue_start + 1, residue_start + 2, residue_start + 3),
                atom_types=tuple(atom_types[residue_start - 1: residue_start + 3]),
                terms=[FourierTerm(kPhi=0.5, period=1.0, phase=0.0)],
            ),
            Dihedral(
                atoms=(1, 2, 3, 4),
                atom_types=tuple(atom_types[:4]),
                terms=[FourierTerm(kPhi=0.5, period=1.0, phase=0.0)],
            ),
        ],
        impropers=[],
        nonbonds=nonbonds,
        unmatched_bonds=[],
        unmatched_angles=[],
        unmatched_dihedrals=[],
        unmatched_impropers=[],
        unmatched_nonbonds=[],
    )

    raw_prepin = tmp_path / "NSR.prepin"
    raw_prepin.write_text(
        "\n".join(
            [
                "    0    0    2",
                "",
                "This is a remark line",
                "NSR.res",
                "NSR   INT  0",
                "CORRECT     OMIT DU   BEG",
                "  0.0000",
            ]
            + [
                f"{local_index:4d} {atom['name']:<4s} {atom_types[residue_start + local_index - 2]:<4s} M 0 0 0 0.000 0.0 0.0 0.00000"
                for local_index, atom in enumerate(residue_atoms, start=1)
            ]
            + ["", "LOOP", "", "IMPROPER", "DONE", "STOP", ""]
        ),
        encoding="utf-8",
    )

    amber = ncaa_export_module.NCAAAmberArtifacts(
        capped_mol2=str(tmp_path / "capped.mol2"),
        gaff2_mol2=str(tmp_path / "typed.mol2"),
        ac=str(tmp_path / "NSR.ac"),
        mc=str(tmp_path / "NSR.mc"),
        prepin=str(raw_prepin),
        refined_prepin=str(tmp_path / "NSR_maple.prepin"),
        res=str(tmp_path / "NSR.res"),
        newpdb=str(tmp_path / "NEWPDB.PDB"),
        frcmod=str(tmp_path / "NSR.frcmod"),
        refined_frcmod=str(tmp_path / "NSR_maple.frcmod"),
    )

    atom_type_rows = ncaa_export_module.write_ncaa_amber_files(
        amber=amber,
        representative_model=representative_model,
        charged_residue=charged_residue,
        final_parameter_set=parameter_set,
        structure=_boundary_structure(target_residue),
        target_residue=target_residue,
    )

    prepin_text = Path(amber.refined_prepin).read_text(encoding="utf-8")
    frcmod_text = Path(amber.refined_frcmod).read_text(encoding="utf-8")
    prepin_types = {
        parts[1]: parts[2]
        for parts in (line.split() for line in prepin_text.splitlines())
        if len(parts) >= 11 and parts[1] != "DUMM"
    }

    assert Path(amber.refined_prepin).is_file()
    assert Path(amber.refined_frcmod).is_file()
    assert prepin_types["N"] == "Z0"
    assert prepin_types["CA"] == "Z1"
    assert prepin_types["C"] == "Z2"
    assert "   1 N    Z0   M 0 0 0 0.000 0.0 0.0 0.00000" in prepin_text
    assert "   4 O    Z3   M 0 0 0 0.000 0.0 0.0 0.00000" in prepin_text
    assert "MASS\nZ0" in frcmod_text
    assert "Z0-Z1" in frcmod_text
    assert "C -Z0" in frcmod_text
    assert "ff14SB/gaff2 peptide boundary" in frcmod_text
    assert "Z2-N" in frcmod_text
    assert "O -C -Z0" in frcmod_text
    assert "Z3-Z2-N" in frcmod_text
    assert "C -Z0-Z1" in frcmod_text
    assert "C -Z0-Z1-Z4" in frcmod_text
    assert "C -Z0-Z1-Z6" in frcmod_text
    assert "t1-t2" not in frcmod_text
    assert "t1" not in frcmod_text
    assert atom_type_rows[:3] == [
        ncaa_export_module.AtomTypeRow(
            "N", "N", atom_types[residue_start - 1], "Z0", pytest.approx(nonbonds[residue_start - 1].charge)
        ),
        ncaa_export_module.AtomTypeRow(
            "CA", "C", atom_types[residue_start], "Z1", pytest.approx(nonbonds[residue_start].charge)
        ),
        ncaa_export_module.AtomTypeRow(
            "C", "C", atom_types[residue_start + 1], "Z2", pytest.approx(nonbonds[residue_start + 1].charge)
        ),
    ]


def test_ncaa_boundary_crossterms_follow_actual_ff19sb_non_pro_neighbor() -> None:
    target_residue = _protein_residue(chirality="L")
    charged_residue = build_capped_ncaa_model(target_residue, "NSR")["residues"][1]
    name_to_global_index, global_to_maple_type = _maple_type_context_for_residue(charged_residue)

    sections = ncaa_export_module._generate_maple_crossterms(
        residue=charged_residue,
        name_to_global_index=name_to_global_index,
        global_to_maple_type=global_to_maple_type,
        structure=_boundary_structure(target_residue, next_resname="ALA"),
        target_residue=target_residue,
        prom="ff19SB",
    )
    angle_text = "".join(sections["ANGLE"])
    dihe_text = "".join(sections["DIHE"])

    assert "ZE-N -XC" in angle_text
    assert "ZE-N -H " in angle_text
    assert "ZE-N -CT" not in angle_text
    assert "ZF-ZE-N -XC" in dihe_text
    assert "Z2-ZE-N -XC" in dihe_text
    assert "ZF-ZE-N -CT" not in dihe_text


def test_ncaa_boundary_crossterms_use_cx_for_n_terminal_previous_residue() -> None:
    target_residue = _translated_residue(_protein_residue(chirality="L"), delta=(0.0, 0.0, 0.0), resseq=2)
    charged_residue = build_capped_ncaa_model(target_residue, "NSR")["residues"][1]
    name_to_global_index, global_to_maple_type = _maple_type_context_for_residue(charged_residue)

    sections = ncaa_export_module._generate_maple_crossterms(
        residue=charged_residue,
        name_to_global_index=name_to_global_index,
        global_to_maple_type=global_to_maple_type,
        structure=_boundary_structure(target_residue, next_resname="PRO"),
        target_residue=target_residue,
    )
    angle_text = "".join(sections["ANGLE"])
    dihe_text = "".join(sections["DIHE"])

    assert "CX-C -Z0" in angle_text
    assert "XC-C -Z0" not in angle_text
    assert "CX-C -Z0-Z1" in dihe_text
    assert "CX-C -Z0-Z2" in dihe_text


def test_ncaa_boundary_crossterms_keep_xc_for_internal_previous_residue() -> None:
    target_residue = _translated_residue(_protein_residue(chirality="L"), delta=(0.0, 0.0, 0.0), resseq=3)
    charged_residue = build_capped_ncaa_model(target_residue, "NSR")["residues"][1]
    name_to_global_index, global_to_maple_type = _maple_type_context_for_residue(charged_residue)

    sections = ncaa_export_module._generate_maple_crossterms(
        residue=charged_residue,
        name_to_global_index=name_to_global_index,
        global_to_maple_type=global_to_maple_type,
        structure=_boundary_structure(target_residue, next_resname="PRO", include_leading=True),
        target_residue=target_residue,
        prom="ff19SB",
    )
    angle_text = "".join(sections["ANGLE"])
    dihe_text = "".join(sections["DIHE"])

    assert "XC-C -Z0" in angle_text
    assert "CX-C -Z0" not in angle_text
    assert "XC-C -Z0-Z1" in dihe_text
    assert "XC-C -Z0-Z2" in dihe_text


def test_ncaa_boundary_crossterms_default_ff14sb_uses_cx_for_internal_previous_residue() -> None:
    target_residue = _translated_residue(_protein_residue(chirality="L"), delta=(0.0, 0.0, 0.0), resseq=3)
    charged_residue = build_capped_ncaa_model(target_residue, "NSR")["residues"][1]
    name_to_global_index, global_to_maple_type = _maple_type_context_for_residue(charged_residue)

    sections = ncaa_export_module._generate_maple_crossterms(
        residue=charged_residue,
        name_to_global_index=name_to_global_index,
        global_to_maple_type=global_to_maple_type,
        structure=_boundary_structure(target_residue, next_resname="PRO", include_leading=True),
        target_residue=target_residue,
    )
    angle_text = "".join(sections["ANGLE"])
    dihe_text = "".join(sections["DIHE"])

    assert "CX-C -Z0" in angle_text
    assert "XC-C -Z0" not in angle_text
    assert "CX-C -Z0-Z1" in dihe_text
    assert "CX-C -Z0-Z2" in dihe_text


def test_ncaa_boundary_crossterms_use_cx_for_c_terminal_next_residue() -> None:
    target_residue = _protein_residue(chirality="L")
    charged_residue = build_capped_ncaa_model(target_residue, "NSR")["residues"][1]
    name_to_global_index, global_to_maple_type = _maple_type_context_for_residue(charged_residue)

    sections = ncaa_export_module._generate_maple_crossterms(
        residue=charged_residue,
        name_to_global_index=name_to_global_index,
        global_to_maple_type=global_to_maple_type,
        structure=_boundary_structure(target_residue, next_resname="ALA", include_following=False),
        target_residue=target_residue,
    )
    angle_text = "".join(sections["ANGLE"])
    dihe_text = "".join(sections["DIHE"])

    assert "ZE-N -CX" in angle_text
    assert "ZE-N -XC" not in angle_text
    assert "ZF-ZE-N -CX" in dihe_text
    assert "Z2-ZE-N -CX" in dihe_text


def test_ncaa_boundary_crossterms_follow_actual_ff19sb_pro_neighbor() -> None:
    target_residue = _protein_residue(chirality="L")
    charged_residue = build_capped_ncaa_model(target_residue, "NSR")["residues"][1]
    name_to_global_index, global_to_maple_type = _maple_type_context_for_residue(charged_residue)

    sections = ncaa_export_module._generate_maple_crossterms(
        residue=charged_residue,
        name_to_global_index=name_to_global_index,
        global_to_maple_type=global_to_maple_type,
        structure=_boundary_structure(target_residue, next_resname="PRO"),
        target_residue=target_residue,
        prom="ff19SB",
    )
    angle_text = "".join(sections["ANGLE"])
    dihe_text = "".join(sections["DIHE"])

    assert "ZE-N -XC" in angle_text
    assert "ZE-N -CT" in angle_text
    assert "ZE-N -H " not in angle_text
    for term in (
        "ZF-ZE-N -XC",
        "ZF-ZE-N -CT",
        "Z2-ZE-N -XC",
        "Z2-ZE-N -CT",
        "C -Z0-Z2-ZE",
        "Z0-Z2-ZE-N",
        "Z3-Z2-ZE-N",
        "Z4-Z2-ZE-N",
    ):
        assert term in dihe_text
    assert "C -Z0-Z2-ZE     6      0.0000" in dihe_text
    assert "C -Z0-Z2-Z3     6      0.0000" in dihe_text
    assert "C -Z0-Z2-Z4     6      0.0000" in dihe_text


def test_write_ncaa_amber_files_rejects_invalid_prepgen_backbone(tmp_path: Path) -> None:
    target_residue = _protein_residue(chirality="L")
    representative_model = build_capped_ncaa_model(target_residue, "NSR")
    charged_residue = representative_model["residues"][1]
    atom_names = [
        atom["name"]
        for residue in representative_model["residues"]
        for atom in sorted(residue["atoms"], key=lambda item: item["serial"])
    ]
    atom_types = [f"t{index}" for index in range(1, len(atom_names) + 1)]
    parameter_set = CorrectionParameterSet(
        mol2=Mol2Topology(
            atoms=[
                Mol2Atom(atom_id=index, name=name, atom_type=atom_type, charge=0.0)
                for index, (name, atom_type) in enumerate(zip(atom_names, atom_types), start=1)
            ],
            bonds=[],
            id_to_index={index: index for index in range(1, len(atom_names) + 1)},
            adjacency={},
        ),
        frcmod=FrcmodDB(mass_params={atom_type: 12.01 for atom_type in atom_types}),
        bonds=[],
        angles=[],
        dihedrals=[],
        impropers=[],
        nonbonds=[Nonbond(atom=index, atom_type=atom_type, charge=0.0) for index, atom_type in enumerate(atom_types, start=1)],
        unmatched_bonds=[],
        unmatched_angles=[],
        unmatched_dihedrals=[],
        unmatched_impropers=[],
        unmatched_nonbonds=[],
    )

    raw_prepin = tmp_path / "NSR.prepin"
    raw_prepin.write_text(
        "\n".join(
            [
                "    0    0    2",
                "",
                "This is a remark line",
                "NSR.res",
                "NSR   INT  0",
                "CORRECT     OMIT DU   BEG",
                "  0.0000",
                "   1 N    t1   M 0 0 0 0.000 0.0 0.0 0.00000",
                "   2 CA   t2   M 0 0 0 0.000 0.0 0.0 0.00000",
                "   3 C    t3   M 0 0 0 0.000 0.0 0.0 0.00000",
                "   4 N    t4   M 0 0 0 0.000 0.0 0.0 0.00000",
                "",
                "LOOP",
                "",
                "IMPROPER",
                "DONE",
                "STOP",
                "",
            ]
        ),
        encoding="utf-8",
    )

    amber = ncaa_export_module.NCAAAmberArtifacts(
        capped_mol2=str(tmp_path / "capped.mol2"),
        gaff2_mol2=str(tmp_path / "typed.mol2"),
        ac=str(tmp_path / "NSR.ac"),
        mc=str(tmp_path / "NSR.mc"),
        prepin=str(raw_prepin),
        refined_prepin=str(tmp_path / "NSR_maple.prepin"),
        res=str(tmp_path / "NSR.res"),
        newpdb=str(tmp_path / "NEWPDB.PDB"),
        frcmod=str(tmp_path / "NSR.frcmod"),
        refined_frcmod=str(tmp_path / "NSR_maple.frcmod"),
    )

    with pytest.raises(ValueError, match="prepgen produced an invalid NCAA template"):
        ncaa_export_module.write_ncaa_amber_files(
            amber=amber,
            representative_model=representative_model,
            charged_residue=charged_residue,
            final_parameter_set=parameter_set,
            structure=_boundary_structure(target_residue),
            target_residue=target_residue,
        )


def test_format_tleap_add_atom_types_lines_raises_on_missing_hybridization() -> None:
    with pytest.raises(ValueError, match="Could not determine tleap hybridization"):
        format_tleap_add_atom_types_lines([("X1", "C", "zz", "Z0")])
