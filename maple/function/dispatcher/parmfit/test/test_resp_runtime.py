from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maple.function.dispatcher.parmfit.utils import interface as interface_module
from maple.function.dispatcher.parmfit.utils import ionparams as ionparams_module
from maple.function.dispatcher.parmfit.utils import resp as resp_module
from maple.function.dispatcher.parmfit.utils import runtime as runtime_module
from maple.function.dispatcher.parmfit.utils.NCAA.models import build_capped_ncaa_model
from maple.function.dispatcher.parmfit.utils.model import flatten_model_atoms, infer_bond_pairs
from maple.function.dispatcher.parmfit.utils.structure import make_atom, make_residue


def _internal_ser_site_model() -> dict:
    zn = make_residue(
        "A",
        301,
        "",
        "ZN",
        [make_atom(1, "ZN", "ZN", np.array((0.0, 0.0, 0.0)))],
        kind="ion",
    )
    ser = make_residue(
        "A",
        10,
        "",
        "SER",
        [
            make_atom(2, "N", "N", np.array((-1.8, 1.5, 0.0))),
            make_atom(3, "HN", "H", np.array((-2.4, 1.9, 0.0))),
            make_atom(4, "CA", "C", np.array((-1.1, 0.4, 0.0))),
            make_atom(5, "HA", "H", np.array((-1.7, -0.4, 0.0))),
            make_atom(6, "CB", "C", np.array((0.3, -0.1, 0.0))),
            make_atom(7, "HB2", "H", np.array((0.9, 0.7, 0.0))),
            make_atom(8, "HB3", "H", np.array((0.6, -1.1, 0.0))),
            make_atom(9, "OG", "O", np.array((0.9, 0.1, 1.1))),
            make_atom(10, "HG", "H", np.array((1.7, -0.3, 1.0))),
            make_atom(11, "C", "C", np.array((-1.6, -0.7, -1.1))),
            make_atom(12, "O", "O", np.array((-0.9, -1.7, -1.1))),
        ],
        kind="protein",
    )
    ser["_prev_peptide_key"] = ("A", 9, "")
    ser["_next_peptide_key"] = ("A", 11, "")
    return {
        "name": "site_model",
        "target_key": ("A", 301, ""),
        "residues": [zn, ser],
        "donor_atoms": {("A", 10, ""): ["OG"]},
    }


def test_reference_charge_library_reads_default_ff14sb_libs() -> None:
    library = resp_module.load_reference_charge_library()

    assert "ALA" in library["internal"]
    assert "NALA" in library["nterm"]
    assert "CALA" in library["cterm"]
    assert library["internal"]["ALA"]["N"][1] == pytest.approx(-0.4157)
    assert library["nterm"]["NALA"]["N"][1] == pytest.approx(0.1414)
    assert library["cterm"]["CALA"]["OXT"][1] == pytest.approx(-0.8055)


def test_prom_reference_lookup_switches_internal_ca_type_without_changing_charge() -> None:
    ca_ff14 = resp_module.lookup_standard_residue_entry(
        resname="SER",
        atom_name="CA",
        category="internal",
        prom="ff14SB",
    )
    ca_ff19 = resp_module.lookup_standard_residue_entry(
        resname="SER",
        atom_name="CA",
        category="internal",
        prom="ff19SB",
    )
    n_ff14 = resp_module.lookup_standard_residue_entry(
        resname="SER",
        atom_name="N",
        category="internal",
        prom="ff14SB",
    )
    n_ff19 = resp_module.lookup_standard_residue_entry(
        resname="SER",
        atom_name="N",
        category="internal",
        prom="ff19SB",
    )
    nterm_ca_ff19 = resp_module.lookup_standard_residue_entry(
        resname="SER",
        atom_name="CA",
        category="nterm",
        prom="ff19SB",
    )

    assert ca_ff14 is not None
    assert ca_ff19 is not None
    assert n_ff14 is not None
    assert n_ff19 is not None
    assert nterm_ca_ff19 is not None
    assert ca_ff14[0] == "CX"
    assert ca_ff19[0] == "XC"
    assert ca_ff14[1] == pytest.approx(ca_ff19[1])
    assert n_ff14[0] == n_ff19[0]
    assert n_ff14[1] == pytest.approx(n_ff19[1])
    assert nterm_ca_ff19[0] == "CX"


@pytest.mark.parametrize(
    ("chgmod", "expected_names"),
    [
        (0, set()),
        (1, {"N", "CA", "C", "O"}),
        (2, {"N", "HN", "CA", "HA", "C", "O"}),
        (3, {"N", "HN", "CA", "HA", "CB", "C", "O"}),
    ],
)
def test_collect_fixed_charge_constraints_matches_chgmod_policy(chgmod: int, expected_names: set[str]) -> None:
    model = _internal_ser_site_model()
    constraints = resp_module.collect_fixed_charge_constraints(model, chgmod=chgmod)
    flattened = flatten_model_atoms(model)
    actual_names = {flattened[index - 1][1]["name"] for index in constraints}
    assert actual_names == expected_names


def test_fixchg_resids_requires_supported_standard_reference() -> None:
    ligand = make_residue(
        "A",
        15,
        "",
        "LIG",
        [make_atom(20, "C1", "C", np.array((1.0, 1.0, 1.0)))],
        kind="ligand",
    )
    model = _internal_ser_site_model()
    model["residues"].append(ligand)

    with pytest.raises(ValueError, match="fixchg residue"):
        resp_module.collect_fixed_charge_constraints(model, chgmod=0, fixchg_resids=["A15"])


def test_stage2_equivalence_groups_detect_methylene_hydrogens() -> None:
    model = _internal_ser_site_model()
    ivary = resp_module.build_stage2_equivalence_map(model)
    flattened = flatten_model_atoms(model)
    hydrogen_map = {flattened[index - 1][1]["name"]: value for index, value in ivary.items()}

    assert hydrogen_map["HB2"] == 0 or hydrogen_map["HB3"] == 0
    assert {hydrogen_map["HB2"], hydrogen_map["HB3"]} != {0}


def test_write_resp_input_files_generates_two_stage_inputs(tmp_path: Path) -> None:
    model = _internal_ser_site_model()
    files = resp_module.write_resp_input_files(str(tmp_path), model, total_charge=2, chgmod=3)

    stage1 = Path(files.resp1_in).read_text(encoding="utf-8")
    stage2 = Path(files.resp2_in).read_text(encoding="utf-8")

    assert "qwt = 0.00050" in stage1
    assert "iqopt = 2" in stage2
    assert "    2" in stage1
    assert "    2" in stage2


def test_merge_esp_files_matches_plain_cat(tmp_path: Path) -> None:
    first = tmp_path / "alpha.esp"
    second = tmp_path / "beta.esp"
    merged = tmp_path / "all.esp"
    first.write_bytes(b"alpha\nline2")
    second.write_bytes(b"\nbeta\n")

    resp_module.merge_esp_files([str(first), str(second)], str(merged))

    assert merged.read_bytes() == first.read_bytes() + second.read_bytes()


def test_interface_set_method_and_gaussian_input(tmp_path: Path) -> None:
    decision = interface_module.set_method({"theory": "HF", "basis": "6-31G*", "nproc": "8", "mem": 32, "route": ""})
    model = _internal_ser_site_model()
    com_file = tmp_path / "resp.com"

    interface_module.prepare_gaussian_esp_input(
        str(com_file),
        model,
        total_charge=2,
        multiplicity=1,
        decision=decision,
        watm="opc",
    )
    content = com_file.read_text(encoding="utf-8")

    assert decision.nproc == 8
    assert decision.mem == 32
    assert "%nproc=8" in content
    assert "%mem=32GB" in content
    route_line = next(line for line in content.splitlines() if line.startswith("#"))
    assert "HF/6-31G*" in route_line
    assert "Pop(MK,ReadRadii)" in route_line
    assert "IOp(6/33=2)" in route_line
    assert "SCF=Tight" in route_line
    assert "Zn 1.373" in content


def test_interface_gaussian_input_omits_radii_without_ion(tmp_path: Path) -> None:
    decision = interface_module.set_method({"theory": "HF", "basis": "6-31G*", "nproc": "4", "mem": 8, "route": ""})
    model = {"residues": [_internal_ser_site_model()["residues"][1]]}
    com_file = tmp_path / "resp_no_ion.com"

    interface_module.prepare_gaussian_esp_input(
        str(com_file),
        model,
        total_charge=0,
        multiplicity=1,
        decision=decision,
    )
    content = com_file.read_text(encoding="utf-8")

    route_line = next(line for line in content.splitlines() if line.startswith("#"))
    assert "Pop=MK" in route_line
    assert "ReadRadii" not in route_line
    assert "Zn " not in content


def test_ionparams_resolve_common_mcpb_names_and_reject_ambiguous_ones() -> None:
    assert ionparams_module.infer_ion_identity("FE") == ("Fe", 3, "Fe3")
    assert ionparams_module.infer_ion_identity("FE2") == ("Fe", 2, "Fe2")
    assert ionparams_module.infer_ion_identity("CU") == ("Cu", 2, "Cu2")
    assert ionparams_module.infer_ion_identity("CU1") == ("Cu", 1, "Cu1")
    assert ionparams_module.infer_ion_identity("ZN") == ("Zn", 2, "Zn2")

    with pytest.raises(ValueError, match="ambiguous ion residue name"):
        ionparams_module.infer_ion_identity("CE")


def test_interface_errors_when_gaussian_is_missing(monkeypatch) -> None:
    monkeypatch.setattr(interface_module.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="Gaussian executable"):
        interface_module._resolve_gaussian_command()


def test_write_residue_pdb_uses_internal_pdb_coordinate_parser(tmp_path: Path) -> None:
    ac_path = tmp_path / "demo.ac"
    ac_path.write_text(
        "\n".join(
            [
                "ATOM      1  ACE1 c3",
                "ATOM      2  ACE2 c3",
                "ATOM      3  N    n ",
                "ATOM      4  CA   c3",
                "ATOM      5  C    c ",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    residue_pdb = tmp_path / "residue.pdb"
    residue_pdb.write_text(
        "".join(
            [
                "ATOM      1  N   SER A   1       1.111   2.222   3.333  1.00  0.00           N  \n",
                "ATOM      2  CA  SER A   1       4.444   5.555   6.666  1.00  0.00           C  \n",
                "ATOM      3  C   SER A   1       7.777   8.888   9.999  1.00  0.00           C  \n",
                "END\n",
            ]
        ),
        encoding="utf-8",
    )
    out_pdb = tmp_path / "res_out.pdb"

    interface_module.write_residue_pdb(
        str(ac_path),
        str(out_pdb),
        n_ace=2,
        n_res=3,
        rn="SER",
        cfg={"residue_file": str(residue_pdb)},
    )

    text = out_pdb.read_text(encoding="utf-8")
    assert "  N   SER A   1       1.111   2.222   3.333" in text
    assert "  CA  SER A   1       4.444   5.555   6.666" in text
    assert "  C   SER A   1       7.777   8.888   9.999" in text


def test_run_resp_pipeline_orchestrates_gaussian_and_resp(monkeypatch, tmp_path: Path) -> None:
    model = _internal_ser_site_model()
    bond_pairs = infer_bond_pairs(model)
    n_atoms = len(flatten_model_atoms(model))
    decision = interface_module.set_method({"theory": "HF", "basis": "6-31G*", "nproc": "4", "mem": 8, "route": ""})

    def fake_run_gaussian(com_file, decision):
        del decision
        log_file = Path(com_file).with_suffix(".log")
        log_file.write_text("Normal termination\n", encoding="utf-8")
        return str(log_file)

    def fake_run_espgen(log_file, esp_file):
        del log_file
        Path(esp_file).write_text("ESP\n", encoding="utf-8")

    def fake_run_resp_stage(*, output_path, punch_path, charge_path, **kwargs):
        del kwargs
        Path(output_path).write_text("RESP OUT\n", encoding="utf-8")
        Path(punch_path).write_text("RESP PCH\n", encoding="utf-8")
        charges = " ".join(f"{0.1 * (index + 1):.6f}" for index in range(n_atoms))
        Path(charge_path).write_text(charges + "\n", encoding="utf-8")

    monkeypatch.setattr(runtime_module.interface, "run_gaussian", fake_run_gaussian)
    monkeypatch.setattr(runtime_module, "run_espgen", fake_run_espgen)
    monkeypatch.setattr(runtime_module, "run_resp_stage", fake_run_resp_stage)
    monkeypatch.setattr(
        runtime_module.interface,
        "set_method",
        lambda *args, **kwargs: pytest.fail("run_resp_pipeline should consume the supplied QMMethod directly."),
    )

    result = runtime_module.run_resp_pipeline(
        output=str(tmp_path / "metal.out"),
        model=model,
        bond_pairs=bond_pairs,
        total_charge=2,
        multiplicity=1,
        chgmod=2,
        qm=decision,
        watm="opc",
    )

    assert Path(result.files["gaussian_input"]).is_file()
    assert Path(result.files["mol2"]).is_file()
    assert Path(result.resp_files["esp"]).is_file()
    assert Path(result.resp_files["resp2_chg"]).is_file()
    assert Path(result.files["gaussian_input"]).parent.name == "metalaa"
    assert Path(result.resp_files["esp"]).parent.name == "metalaa"
    assert Path(result.files["mol2"]).parent.name == "metalaa"
    assert result.decision.nproc == 4
    assert result.decision.mem == 8
    assigned_charges = [atom["charge"] for residue in result.model["residues"] for atom in residue["atoms"]]
    assert len(assigned_charges) == n_atoms
    assert assigned_charges[0] == pytest.approx(0.1)


def _ncaa_target_residue() -> dict:
    return make_residue(
        "A",
        1,
        "",
        "NAA",
        [
            make_atom(1, "N", "N", np.array((0.0, 1.0, 0.0))),
            make_atom(2, "CA", "C", np.array((0.0, 0.0, 0.0))),
            make_atom(3, "C", "C", np.array((-1.0, 0.0, 0.0))),
            make_atom(4, "O", "O", np.array((-1.7, -0.8, 0.0))),
            make_atom(5, "CB", "C", np.array((0.0, 0.0, 1.0))),
            make_atom(6, "H", "H", np.array((0.8, 0.5, 0.0))),
            make_atom(7, "HA", "H", np.array((0.8, -0.5, 0.0))),
        ],
        kind="protein",
    )


def test_write_multiconformer_resp_input_files_builds_group_charge_constraints(tmp_path: Path) -> None:
    model = build_capped_ncaa_model(_ncaa_target_residue(), "NAA")
    files = resp_module.write_multiconformer_resp_input_files(
        str(tmp_path),
        [model, model],
        labels=["alpha", "beta"],
        total_charge=0,
        residue_key=("A", 1, ""),
    )

    stage1 = Path(files.resp1_in).read_text(encoding="utf-8")
    stage2 = Path(files.resp2_in).read_text(encoding="utf-8")

    assert "nmol = 2" in stage1
    assert "nmol = 2" in stage2
    assert "iqopt = 2" in stage2
    assert "\n    1.0\nalpha\n" in stage1
    assert "\n    1.0\nbeta\n" in stage1
    assert "    7   0.00000" in stage1
    assert "    1    7    1    8    1    9    1   10" in stage1
    assert "    2    7    2    8    2    9    2   10" in stage1
    assert "    2\n    1    1    2    1\n" in stage1
    assert "   30" not in stage1
    assert "-1" in stage2


def test_run_multiconformer_resp_orchestrates_all_esp_and_two_stage_resp(monkeypatch, tmp_path: Path) -> None:
    model = build_capped_ncaa_model(_ncaa_target_residue(), "NAA")
    model["charge"] = 0
    model["mult"] = 1
    bond_pairs = infer_bond_pairs(model)
    n_atoms = len(flatten_model_atoms(model))
    decision = interface_module.set_method({"theory": "HF", "basis": "6-31G(d)", "nproc": "4", "mem": 8, "route": ""})

    def fake_run_gaussian(com_file, decision):
        del decision
        log_file = Path(com_file).with_suffix(".log")
        log_file.write_text("Normal termination\n", encoding="utf-8")
        return str(log_file)

    def fake_run_espgen(log_file, esp_file):
        Path(esp_file).write_text(f"ESP from {Path(log_file).name}\n", encoding="utf-8")

    def fake_run_resp_stage(*, output_path, punch_path, charge_path, **kwargs):
        del kwargs
        Path(output_path).write_text("RESP OUT\n", encoding="utf-8")
        Path(punch_path).write_text("RESP PCH\n", encoding="utf-8")
        Path(charge_path).write_text(" ".join(["0.0"] * n_atoms) + "\n", encoding="utf-8")

    monkeypatch.setattr(runtime_module.interface, "run_gaussian", fake_run_gaussian)
    monkeypatch.setattr(runtime_module, "run_espgen", fake_run_espgen)
    monkeypatch.setattr(runtime_module, "run_resp_stage", fake_run_resp_stage)
    monkeypatch.setattr(
        runtime_module.interface,
        "set_method",
        lambda *args, **kwargs: pytest.fail("run_multiconformer_resp should consume the supplied QMMethod directly."),
    )

    result = runtime_module.run_multiconformer_resp(
        output=str(tmp_path / "ncaa.out"),
        conformers=[("alpha", model), ("beta", model)],
        representative_model=model,
        residue_key=("A", 1, ""),
        bond_pairs=bond_pairs,
        total_charge=0,
        multiplicity=1,
        qm=decision,
    )

    all_esp = Path(result.resp_files["all_esp"]).read_text(encoding="utf-8")
    stage1 = Path(result.resp_files["resp1_in"]).read_text(encoding="utf-8")
    assert all_esp == "ESP from ncaa_alpha_resp.log\nESP from ncaa_beta_resp.log\n"
    assert "nmol = 2" in stage1
    assert "\n    1.0\nalpha\n" in stage1
    assert "\n    1.0\nbeta\n" in stage1
    assert Path(result.files["mol2"]).is_file()
    assert Path(result.files["mol2"]).parent.name == "ncaa"
    assert Path(result.resp_files["all_esp"]).parent.name == "ncaa"


def test_run_prepgen_preserves_supplied_file_arguments(monkeypatch, tmp_path: Path) -> None:
    commands: list[str] = []

    def fake_run_cmd(cmd, cwd=None):
        del cwd
        commands.append(cmd)
        return 0, "", ""

    monkeypatch.setattr(interface_module, "_run_cmd", fake_run_cmd)
    result = interface_module.run_prepgen(
        "NAA.ac",
        "mainchain.mc",
        {"residue_name": "NAA"},
        str(tmp_path),
    )

    assert "-i NAA.ac" in commands[0]
    assert "-m mainchain.mc" in commands[0]
    assert "-rf NAA.res" in commands[0]
    assert result.prepin_path.endswith("NAA.prepin")
    assert result.res_path.endswith("NAA.res")
    assert result.newpdb_path.endswith("NEWPDB.PDB")


def test_ambertools_commands_preserve_supplied_arguments(monkeypatch, tmp_path: Path) -> None:
    commands: list[str] = []

    def fake_run_cmd(cmd, cwd=None):
        commands.append((cmd, cwd))
        return 0, "", ""

    monkeypatch.setattr(interface_module, "_run_cmd", fake_run_cmd)

    interface_module.run_antechamber(
        "input.mol2",
        {"residue_name": "NSR", "net_charge": 0},
        str(tmp_path),
        input_format="mol2",
        charge_mode="rc",
        charge_file="target.chg",
    )
    interface_module.run_parmchk2(
        "NSR.prepin",
        {"residue_name": "NSR"},
        False,
        str(tmp_path),
    )

    assert commands[0][1] == str(tmp_path)
    assert "-i input.mol2" in commands[0][0]
    assert "-cf target.chg" in commands[0][0]
    assert commands[1][1] == str(tmp_path)
    assert "-i NSR.prepin" in commands[1][0]
