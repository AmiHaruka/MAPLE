from __future__ import annotations

from copy import deepcopy
import inspect
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maple.function.dispatcher.parmfit import Parmfit
from maple.function.dispatcher.parmfit.abinitio.abinitio import Abinitio
from maple.function.dispatcher.parmfit.abinitio import abinitio as abinitio_module
from maple.function.dispatcher.parmfit.abinitio import report as abinitio_report_module
from maple.function.dispatcher.parmfit.correction.correction import Correction
from maple.function.dispatcher.parmfit.utils import MetalAA
from maple.function.dispatcher.parmfit.utils.capping import build_ace_cap, build_gly_bridge, build_nme_cap
from maple.function.dispatcher.parmfit.utils import context as context_module
from maple.function.dispatcher.parmfit.utils import model as model_module
from maple.function.dispatcher.parmfit.utils import resp as resp_module
from maple.function.dispatcher.parmfit.utils import runtime as runtime_module
from maple.function.dispatcher.parmfit.utils.MetalAA.artifacts import MetalArtifacts, MetalAtomTypeRow
from maple.function.dispatcher.parmfit.utils.MetalAA.charges import project_resp_charges_onto_site_model
from maple.function.dispatcher.parmfit.utils.MetalAA.config import parse_metal_abinitio_config
from maple.function.dispatcher.parmfit.utils.MetalAA import report as metal_report_module
from maple.function.dispatcher.parmfit.utils.MetalAA import artifacts as metal_export_module
from maple.function.dispatcher.parmfit.utils.MetalAA import parameters as metal_parameters_module
from maple.function.dispatcher.parmfit.utils.MetalAA import workflow as metal_workflow_module
from maple.function.dispatcher.parmfit.utils.MetalAA import (
    MetalWorkflowResult,
    build_metal_large_model,
    build_metal_site_model,
    extract_metal_cluster,
    identify_metal_site_core,
)
from maple.function.dispatcher.parmfit.utils.structure import (
    get_atom_xyz,
    get_resid_key,
    make_atom,
    make_residue,
    refresh_resid,
    read_pdb,
    search_atom,
)


class ZeroHessianCalculator:
    def get_potential_energy(self, atoms, force_consistent=False):
        return 0.0

    def get_forces(self, atoms):
        return np.zeros((len(atoms), 3), dtype=float)

    def get_hessian(self, atoms):
        n_atoms = len(atoms)
        return np.ones((3 * n_atoms, 3 * n_atoms), dtype=float) * 0.01 + np.eye(3 * n_atoms, dtype=float) * 0.1


def _pdb_atom(
    record: str,
    serial: int,
    name: str,
    resname: str,
    chain: str,
    resseq: int,
    x: float,
    y: float,
    z: float,
    element: str,
    icode: str = "",
) -> str:
    return (
        f"{record:<6}{serial:5d} {name:>4s} {resname:>3s} {chain:1s}{resseq:4d}{icode:1s}   "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{0.0:6.2f}          {element:>2s}\n"
    )


def _write_text(path: Path, text: str) -> str:
    path.write_text(text, encoding="utf-8")
    return str(path)


def _make_zn_residue(chain: str, resseq: int) -> dict:
    return make_residue(
        chain,
        resseq,
        "",
        "ZN",
        [make_atom(1, "ZN", "ZN", np.array((0.0, 0.0, 0.0)))],
        kind="ion",
    )


def test_write_model_pdb_renders_blank_chain_as_a(tmp_path: Path) -> None:
    residue = make_residue(
        "_",
        2,
        "",
        "SER",
        [
            make_atom(1, "N", "N", np.array((0.0, 0.0, 0.0))),
        ],
        kind="protein",
    )
    pdb_path = tmp_path / "blank_chain.pdb"

    model_module.write_model_pdb(str(pdb_path), {"residues": [residue]})

    line = pdb_path.read_text(encoding="utf-8").splitlines()[0]
    assert line[21] == "A"
    assert residue["chain"] == "_"


def _make_metal_route_inputs(tmp_path: Path) -> tuple[str, Atoms, Path, Path]:
    pdb = "".join(
        [
            _pdb_atom("HETATM", 1, "ZN", "ZN", "A", 301, 0.0, 0.0, 0.0, "ZN"),
            _pdb_atom("ATOM", 2, "N", "CYS", "A", 10, -3.0, 2.8, 0.0, "N"),
            _pdb_atom("ATOM", 3, "CA", "CYS", "A", 10, -2.0, 1.8, 0.0, "C"),
            _pdb_atom("ATOM", 4, "C", "CYS", "A", 10, -1.0, 2.3, 0.0, "C"),
            _pdb_atom("ATOM", 5, "O", "CYS", "A", 10, -0.2, 3.1, 0.0, "O"),
            _pdb_atom("ATOM", 6, "CB", "CYS", "A", 10, -2.0, 0.4, 0.0, "C"),
            _pdb_atom("ATOM", 7, "SG", "CYS", "A", 10, -0.7, 0.2, 0.0, "S"),
            _pdb_atom("HETATM", 8, "O", "HOH", "A", 401, 2.4, 0.0, 0.0, "O"),
            _pdb_atom("ATOM", 9, "N", "GLU", "A", 20, 3.7, 0.0, 0.0, "N"),
            _pdb_atom("ATOM", 10, "CA", "GLU", "A", 20, 4.7, 0.0, 0.0, "C"),
            _pdb_atom("ATOM", 11, "C", "GLU", "A", 20, 5.7, 0.0, 0.0, "C"),
            _pdb_atom("ATOM", 12, "O", "GLU", "A", 20, 6.7, 0.0, 0.0, "O"),
            "END\n",
        ]
    )
    pdb_path = _write_text(tmp_path / "metal_run.pdb", pdb)
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    atoms.info["charge"] = 2
    atoms.info["mult"] = 1
    atoms.calc = ZeroHessianCalculator()
    atoms.f_max_th = 0.00285
    atoms.f_rms_th = 0.00190
    atoms.dp_max_th = 0.00315
    atoms.dp_rms_th = 0.00210
    output = tmp_path / "metal_abinitio.out"
    output_root = tmp_path / f"{output.with_suffix('').name}_work"
    return pdb_path, atoms, output, output_root


def _make_carboxylate_route_inputs(tmp_path: Path) -> tuple[str, Atoms, Path, Path]:
    pdb = "".join(
        [
            _pdb_atom("HETATM", 1, "ZN", "ZN", "A", 301, 0.0, 0.0, 0.0, "ZN"),
            _pdb_atom("ATOM", 2, "N", "GLU", "A", 10, -8.0, 0.0, 0.0, "N"),
            _pdb_atom("ATOM", 3, "CA", "GLU", "A", 10, -7.0, 0.0, 0.0, "C"),
            _pdb_atom("ATOM", 4, "C", "GLU", "A", 10, -6.0, 0.0, 0.0, "C"),
            _pdb_atom("ATOM", 5, "O", "GLU", "A", 10, -5.0, 0.0, 0.0, "O"),
            _pdb_atom("ATOM", 6, "CB", "GLU", "A", 10, -7.0, 1.2, 0.0, "C"),
            _pdb_atom("ATOM", 7, "CG", "GLU", "A", 10, -6.0, 1.8, 0.0, "C"),
            _pdb_atom("ATOM", 8, "CD", "GLU", "A", 10, -5.0, 1.4, 0.0, "C"),
            _pdb_atom("ATOM", 9, "OE1", "GLU", "A", 10, -4.4, 2.5, 0.0, "O"),
            _pdb_atom("ATOM", 10, "OE2", "GLU", "A", 10, -4.4, 0.3, 0.0, "O"),
            "END\n",
        ]
    )
    pdb_path = _write_text(tmp_path / "metal_carboxylate.pdb", pdb)
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    atoms.info["charge"] = 2
    atoms.info["mult"] = 1
    atoms.calc = ZeroHessianCalculator()
    atoms.f_max_th = 0.00285
    atoms.f_rms_th = 0.00190
    atoms.dp_max_th = 0.00315
    atoms.dp_rms_th = 0.00210
    output = tmp_path / "metal_carboxylate.out"
    output_root = tmp_path / f"{output.with_suffix('').name}_work"
    return pdb_path, atoms, output, output_root


def _fake_metal_large_resp_pipeline(output: str):
    def fake_run_resp_pipeline(
        *,
        output,
        model,
        bond_pairs,
        total_charge,
        multiplicity,
        chgmod,
        fixchg_resids,
        qm,
        label,
        watm,
        prom="ff14SB",
        charge_groups=None,
    ):
        del bond_pairs, multiplicity, chgmod, fixchg_resids, qm
        output_root = Path(output).with_suffix("").with_name(f"{Path(output).with_suffix('').name}_work")
        metalaa_dir = output_root / "metalaa"
        metalaa_dir.mkdir(parents=True, exist_ok=True)
        base = Path(output).with_suffix("")
        gaussian_input = metalaa_dir / f"{base.name}_{label}.gjf"
        gaussian_log = metalaa_dir / f"{base.name}_{label}.log"
        esp = metalaa_dir / f"{label}.esp"
        mol2 = metalaa_dir / f"{label}.mol2"
        sidecars = {
            "gaussian_log": gaussian_log,
            "esp": esp,
            "resp1_in": metalaa_dir / "resp1.in",
            "resp1_out": metalaa_dir / "resp1.out",
            "resp1_pch": metalaa_dir / "resp1.pch",
            "resp1_chg": metalaa_dir / "resp1.chg",
            "resp1_calc_esp": metalaa_dir / "resp1_calc.esp",
            "resp2_in": metalaa_dir / "resp2.in",
            "resp2_out": metalaa_dir / "resp2.out",
            "resp2_pch": metalaa_dir / "resp2.pch",
            "resp2_chg": metalaa_dir / "resp2.chg",
            "resp2_calc_esp": metalaa_dir / "resp2_calc.esp",
        }
        assert model["name"] == "large_model"
        assert label == "metal_large_resp"
        assert total_charge == 1
        assert watm == "tip3p"
        assert prom in {"ff14SB", "ff19SB"}

        gaussian_input.write_text("# HF/6-31G* Pop=MK IOp(6/33=2) SCF=Tight\n", encoding="utf-8")
        mol2.write_text("@<TRIPOS>MOLECULE\nFAKE\n", encoding="utf-8")
        for path in sidecars.values():
            path.write_text("", encoding="utf-8")

        deployment_charge = float(total_charge) - sum(float(target) for _indices, target in (charge_groups or []))
        charged_model = deepcopy(model)
        first_atom = True
        for residue in charged_model["residues"]:
            for atom in residue["atoms"]:
                atom["charge"] = deployment_charge if first_atom and residue["kind"] == "ion" else 0.0
                first_atom = False if residue["kind"] == "ion" else first_atom

        return runtime_module.RespPipelineResult(
            model=charged_model,
            files={"gaussian_input": str(gaussian_input), "mol2": str(mol2)},
            resp_files={key: str(value) for key, value in sidecars.items()},
            decision=runtime_module.QMMethod(
                backend="gaussian",
                theory="HF",
                basis="6-31G*",
                nproc=1,
                mem=1,
                route="HF/6-31G* Pop=MK IOp(6/33=2) SCF=Tight",
            ),
        )

    return fake_run_resp_pipeline


def test_parse_metal_abinitio_config_splits_fixchg_resids_string() -> None:
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    atoms.info["charge"] = 2
    atoms.info["mult"] = 1
    atoms.calc = ZeroHessianCalculator()

    config = parse_metal_abinitio_config(
        {
            "pdb": "demo.pdb",
            "target": "A301",
            "fixchg_resids": "A10 A11",
        },
        pdb_path="demo.pdb",
        target="A301",
        charge=2,
        mult=1,
        target_residue=_make_zn_residue("A", 301),
    )
    empty_config = parse_metal_abinitio_config(
        {
            "pdb": "demo.pdb",
            "target": "A301",
            "fixchg_resids": "",
        },
        pdb_path="demo.pdb",
        target="A301",
        charge=2,
        mult=1,
        target_residue=_make_zn_residue("A", 301),
    )

    assert not hasattr(config, "qm")
    assert not hasattr(config, "chgmod")
    assert not hasattr(config, "fixchg_resids")
    assert config.resp.fixchg_resids == ["A10", "A11"]
    assert empty_config.resp.fixchg_resids == []
    assert config.watm == "tip3p"
    assert config.resp.watm == "tip3p"
    assert config.ionm == "12_6"
    assert config.cluster_cutoff == pytest.approx(3.0)
    assert config.donor_cutoff == pytest.approx(2.7)
    assert config.resp.chgmod == 1
    assert config.opt_max_iter == 256
    assert config.opt_max_step == pytest.approx(0.2)
    assert config.resp.qm.theory == "PBE1PBE"
    assert config.resp.qm.basis == "def2SVP"
    assert config.prom == "ff14SB"
    assert config.resp.prom == "ff14SB"


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
def test_parse_metal_abinitio_config_accepts_prom(raw_prom: str, expected: str) -> None:
    config = parse_metal_abinitio_config(
        {"prom": raw_prom},
        pdb_path="demo.pdb",
        target="A301",
        charge=2,
        mult=1,
        target_residue=_make_zn_residue("A", 301),
    )

    assert config.prom == expected
    assert config.resp.prom == expected


def test_parse_metal_abinitio_config_splits_add_resid_string() -> None:
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    atoms.info["charge"] = 2
    atoms.info["mult"] = 1
    atoms.calc = ZeroHessianCalculator()

    config = parse_metal_abinitio_config(
        {
            "pdb": "demo.pdb",
            "target": "A301",
            "add_resid": "A10 B11",
        },
        pdb_path="demo.pdb",
        target="A301",
        charge=2,
        mult=1,
        target_residue=_make_zn_residue("A", 301),
    )

    assert config.target == "A301"
    assert config.add_resid == ["A10", "B11"]


def test_parse_metal_abinitio_config_parses_set_bonded_pairs() -> None:
    config = parse_metal_abinitio_config(
        {
            "pdb": "demo.pdb",
            "target": "A301",
            "set_bonded": "1-144,2-144",
        },
        pdb_path="demo.pdb",
        target="A301",
        charge=2,
        mult=1,
        target_residue=_make_zn_residue("A", 301),
    )
    spaced = parse_metal_abinitio_config(
        {
            "pdb": "demo.pdb",
            "target": "A301",
            "set_bonded": "1-144 2-144",
        },
        pdb_path="demo.pdb",
        target="A301",
        charge=2,
        mult=1,
        target_residue=_make_zn_residue("A", 301),
    )

    assert config.set_bonded == [(1, 144), (2, 144)]
    assert spaced.set_bonded == [(1, 144), (2, 144)]
    with pytest.raises(ValueError, match="set_bonded"):
        parse_metal_abinitio_config(
            {
                "pdb": "demo.pdb",
                "target": "A301",
                "set_bonded": "1:144",
            },
            pdb_path="demo.pdb",
            target="A301",
            charge=2,
            mult=1,
            target_residue=_make_zn_residue("A", 301),
        )


def test_parse_metal_abinitio_config_records_optional_oxy_and_cfmol2() -> None:
    config = parse_metal_abinitio_config(
        {
            "pdb": "demo.pdb",
            "target": "A301",
            "cfmol2": "heme.mol2 flavin.mol2",
        },
        pdb_path="demo.pdb",
        target="A301",
        charge=-1,
        mult=6,
        oxy=3,
        target_residue=_make_zn_residue("A", 301),
    )
    fallback = parse_metal_abinitio_config(
        {
            "pdb": "demo.pdb",
            "target": "A301",
        },
        pdb_path="demo.pdb",
        target="A301",
        charge=2,
        mult=1,
        target_residue=_make_zn_residue("A", 301),
    )

    assert config.charge == -1
    assert config.mult == 6
    assert config.oxy == 3
    assert config.cfmol2 == ["heme.mol2", "flavin.mol2"]
    assert fallback.oxy is None
    assert fallback.cfmol2 == []
    comma_config = parse_metal_abinitio_config(
        {
            "pdb": "demo.pdb",
            "target": "A301",
            "cfmol2": "heme.mol2,flavin.mol2",
        },
        pdb_path="demo.pdb",
        target="A301",
        charge=-1,
        mult=6,
        target_residue=_make_zn_residue("A", 301),
    )
    assert comma_config.cfmol2 == ["heme.mol2", "flavin.mol2"]


def test_parse_metal_abinitio_config_validates_watm_and_ionm() -> None:
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    atoms.info["charge"] = 2
    atoms.info["mult"] = 1
    atoms.calc = ZeroHessianCalculator()

    config = parse_metal_abinitio_config(
        {
            "pdb": "demo.pdb",
            "target": "A301",
            "watm": "opc3",
            "ionm": "hfe",
        },
        pdb_path="demo.pdb",
        target="A301",
        charge=2,
        mult=1,
        target_residue=_make_zn_residue("A", 301),
    )

    assert config.watm == "opc3"
    assert config.resp.watm == "opc3"
    assert config.ionm == "hfe"

    with pytest.raises(ValueError, match="water model"):
        parse_metal_abinitio_config(
            {"pdb": "demo.pdb", "target": "A301", "watm": "bad"},
            pdb_path="demo.pdb",
            target="A301",
            charge=2,
            mult=1,
            target_residue=_make_zn_residue("A", 301),
        )
    with pytest.raises(ValueError, match="ion parameter set"):
        parse_metal_abinitio_config(
            {"pdb": "demo.pdb", "target": "A301", "ionm": "bad"},
            pdb_path="demo.pdb",
            target="A301",
            charge=2,
            mult=1,
            target_residue=_make_zn_residue("A", 301),
        )


def test_parse_metal_abinitio_config_validates_prom_as_protein_model() -> None:
    config = parse_metal_abinitio_config(
        {"pdb": "demo.pdb", "target": "A301", "prom": "FF19SB"},
        pdb_path="demo.pdb",
        target="A301",
        charge=2,
        mult=1,
        target_residue=_make_zn_residue("A", 301),
    )

    assert config.prom == "ff19SB"
    assert config.resp.prom == "ff19SB"

    with pytest.raises(ValueError, match="protein model"):
        parse_metal_abinitio_config(
            {"pdb": "demo.pdb", "target": "A301", "prom": "bad"},
            pdb_path="demo.pdb",
            target="A301",
            charge=2,
            mult=1,
            target_residue=_make_zn_residue("A", 301),
        )


def test_parse_metal_abinitio_config_parses_bonded_method() -> None:
    default_config = parse_metal_abinitio_config(
        {"pdb": "demo.pdb", "target": "A301"},
        pdb_path="demo.pdb",
        target="A301",
        charge=2,
        mult=1,
        target_residue=_make_zn_residue("A", 301),
    )
    seminario_config = parse_metal_abinitio_config(
        {"pdb": "demo.pdb", "target": "A301", "bonded": "Seminario"},
        pdb_path="demo.pdb",
        target="A301",
        charge=2,
        mult=1,
        target_residue=_make_zn_residue("A", 301),
    )

    assert default_config.bonded == "mseminario"
    assert seminario_config.bonded == "seminario"
    with pytest.raises(ValueError, match="bonded method"):
        parse_metal_abinitio_config(
            {"pdb": "demo.pdb", "target": "A301", "bonded": "bad"},
            pdb_path="demo.pdb",
            target="A301",
            charge=2,
            mult=1,
            target_residue=_make_zn_residue("A", 301),
        )


def test_metal_workflow_uses_oxy_for_ion_identity_but_charge_for_large_model(tmp_path: Path) -> None:
    pdb = "".join(
        [
            _pdb_atom("HETATM", 1, "FE", "FE", "A", 301, 0.0, 0.0, 0.0, "FE"),
            _pdb_atom("ATOM", 2, "N", "CYM", "A", 10, -3.0, 2.8, 0.0, "N"),
            _pdb_atom("ATOM", 3, "CA", "CYM", "A", 10, -2.0, 1.8, 0.0, "C"),
            _pdb_atom("ATOM", 4, "C", "CYM", "A", 10, -1.0, 2.3, 0.0, "C"),
            _pdb_atom("ATOM", 5, "O", "CYM", "A", 10, -0.2, 3.1, 0.0, "O"),
            _pdb_atom("ATOM", 6, "CB", "CYM", "A", 10, -2.0, 0.4, 0.0, "C"),
            _pdb_atom("ATOM", 7, "SG", "CYM", "A", 10, 1.9, 0.0, 0.0, "S"),
            "END\n",
        ]
    )
    pdb_path = _write_text(tmp_path / "fe_cym.pdb", pdb)
    structure = read_pdb(pdb_path)
    config = parse_metal_abinitio_config(
        {"pdb": pdb_path, "target": "A301"},
        pdb_path=pdb_path,
        target="A301",
        charge=-1,
        mult=6,
        oxy=3,
        target_residue=structure["residues"][0],
    )

    _selection, bundle = metal_workflow_module._prepare_metal_large_model(structure, config)

    assert bundle.large_charge == -2
    assert bundle.large_mult == 6
    target_residue = next(residue for residue in bundle.large_model["residues"] if residue["kind"] == "ion")
    assert target_residue["formal_charge"] == 3


def test_metal_defaults_match_config_and_helper_defaults(tmp_path: Path) -> None:
    pdb = "".join(
        [
            _pdb_atom("HETATM", 1, "ZN", "ZN", "A", 301, 0.0, 0.0, 0.0, "ZN"),
            _pdb_atom("ATOM", 2, "N", "CYS", "A", 10, -3.0, 2.8, 0.0, "N"),
            _pdb_atom("ATOM", 3, "CA", "CYS", "A", 10, -2.0, 1.8, 0.0, "C"),
            _pdb_atom("ATOM", 4, "C", "CYS", "A", 10, -1.0, 2.3, 0.0, "C"),
            _pdb_atom("ATOM", 5, "O", "CYS", "A", 10, -0.2, 3.1, 0.0, "O"),
            _pdb_atom("ATOM", 6, "CB", "CYS", "A", 10, -2.0, 0.4, 0.0, "C"),
            _pdb_atom("ATOM", 7, "SG", "CYS", "A", 10, -0.7, 0.2, 0.0, "S"),
            "END\n",
        ]
    )
    pdb_path = _write_text(tmp_path / "metal_defaults.pdb", pdb)
    structure = read_pdb(pdb_path)
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    atoms.info["charge"] = 2
    atoms.info["mult"] = 1
    atoms.calc = ZeroHessianCalculator()

    config = parse_metal_abinitio_config(
        {"pdb": pdb_path, "target": "A301"},
        pdb_path=pdb_path,
        target="A301",
        charge=2,
        mult=1,
        target_residue=structure["residues"][0],
    )
    cluster_info = extract_metal_cluster(structure, target="A301")
    large_model = build_metal_large_model(structure, target="A301")

    assert config.cluster_cutoff == pytest.approx(3.0)
    assert cluster_info["cutoff"] == pytest.approx(3.0)
    assert large_model["cluster_cutoff"] == pytest.approx(3.0)


def test_parmfit_abinitio_bootstraps_charge_mult_from_cmo_without_oxy(monkeypatch, tmp_path: Path) -> None:
    observed = {}
    pdb_path = _write_text(tmp_path / "stub.pdb", "END\n")
    expected_result = object()

    def fake_run(self):
        observed["charge"] = self.atoms.info["charge"]
        observed["mult"] = self.atoms.info["mult"]
        observed["spin"] = self.atoms.info["spin"]
        observed["oxy"] = self.atoms.info.get("oxy")
        observed["pdb"] = self.params["pdb"]
        return expected_result

    monkeypatch.setattr(Abinitio, "run", fake_run)

    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    atoms.calc = ZeroHessianCalculator()

    result = Parmfit(
        output=str(tmp_path / "bootstrap.out"),
        atoms=atoms,
        params={
            "method": "abinitio",
            "pdb": str(pdb_path),
            "target": "A301",
            "cmo": "2 3",
        },
    ).run()

    assert observed == {"charge": 2, "mult": 3, "spin": 1.0, "oxy": None, "pdb": str(pdb_path)}
    assert result is expected_result


def test_parmfit_correction_returns_subdispatcher_result(monkeypatch, tmp_path: Path) -> None:
    expected_result = object()
    mol2_path = _write_text(tmp_path / "demo.mol2", "")

    def fake_run(self):
        assert self.params.mol2 == str(mol2_path)
        return expected_result

    monkeypatch.setattr(Correction, "run", fake_run)

    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    result = Parmfit(
        output=str(tmp_path / "correction.out"),
        atoms=atoms,
        params={"method": "correction", "mol2": str(mol2_path)},
    ).run()

    assert result is expected_result


def test_parmfit_abinitio_bootstraps_optional_oxy_from_cmo(monkeypatch, tmp_path: Path) -> None:
    observed = {}
    pdb_path = _write_text(tmp_path / "stub.pdb", "END\n")

    def fake_run(self):
        observed["charge"] = self.atoms.info["charge"]
        observed["mult"] = self.atoms.info["mult"]
        observed["spin"] = self.atoms.info["spin"]
        observed["oxy"] = self.atoms.info["oxy"]

    monkeypatch.setattr(Abinitio, "run", fake_run)

    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    atoms.calc = ZeroHessianCalculator()

    Parmfit(
        output=str(tmp_path / "bootstrap_oxy.out"),
        atoms=atoms,
        params={
            "method": "abinitio",
            "pdb": str(pdb_path),
            "target": "A301",
            "cmo": "-1 6 3",
        },
    ).run()

    assert observed == {"charge": -1, "mult": 6, "spin": 2.5, "oxy": 3}


def test_parmfit_abinitio_defaults_cmo_to_zero_one_without_oxy(monkeypatch, tmp_path: Path) -> None:
    observed = {}
    pdb_path = _write_text(tmp_path / "stub.pdb", "END\n")

    def fake_run(self):
        observed["charge"] = self.atoms.info["charge"]
        observed["mult"] = self.atoms.info["mult"]
        observed["oxy"] = self.atoms.info.get("oxy")

    monkeypatch.setattr(Abinitio, "run", fake_run)

    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    atoms.info["charge"] = 5
    atoms.info["mult"] = 7
    atoms.info["oxy"] = 9
    atoms.calc = ZeroHessianCalculator()

    Parmfit(
        output=str(tmp_path / "default_cmo.out"),
        atoms=atoms,
        params={"method": "abinitio", "pdb": str(pdb_path), "target": "A301"},
    ).run()

    assert observed == {"charge": 0, "mult": 1, "oxy": None}


def test_parmfit_abinitio_rejects_malformed_cmo(tmp_path: Path) -> None:
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    atoms.calc = ZeroHessianCalculator()
    pdb_path = _write_text(tmp_path / "stub.pdb", "END\n")

    with pytest.raises(ValueError, match="cmo"):
        Parmfit(
            output=str(tmp_path / "bad_cmo.out"),
            atoms=atoms,
            params={
                "method": "abinitio",
                "pdb": str(pdb_path),
                "target": "A301",
                "cmo": "2",
            },
        ).run()


def test_metal_report_lines_cover_summary_and_warning() -> None:
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    atoms.info["charge"] = 2
    atoms.info["mult"] = 1
    atoms.calc = ZeroHessianCalculator()
    config = parse_metal_abinitio_config(
        {
            "pdb": "demo.pdb",
            "target": "A301",
            "add_resid": "A10",
        },
        pdb_path="demo.pdb",
        target="A301",
        charge=2,
        mult=1,
        target_residue=_make_zn_residue("A", 301),
    )
    artifacts = MetalArtifacts(
        files={
            "large_raw_pdb": "raw.pdb",
            "large_opt_pdb": "opt.pdb",
            "site_pdb": "site.pdb",
            "gaussian_input": "resp.com",
            "mol2": "site.mol2",
            "site_mol2": "site.mol2",
            "frcmod": "demo_metal.frcmod",
            "tleap_pdb": "demo_metal_tleap.pdb",
            "tleap_input": "demo_metal_tleap.in",
        },
        resp_files={},
        mol2_files={"ZN1": "ZN1.mol2"},
        tleap_lines=[
            "source leaprc.protein.ff14SB\n",
            'addAtomTypes {\n',
            '    { "M1" "Zn" "sp3" }\n',
            '}\n',
            "ZN1 = loadmol2 ZN1.mol2\n",
            "loadamberparams frcmod.ionslm_126_opc\n",
            "loadamberparams demo_metal.frcmod\n",
            "mol = loadpdb demo_metal_tleap.pdb\n",
            "quit\n",
        ],
    )

    start_lines = "".join(metal_report_module.format_metal_start_lines(config, large_charge=1, large_mult=1))
    final_lines = "".join(
        metal_report_module.format_metal_final_lines(
            artifacts=artifacts,
            mseminario_warning="seminario warning",
            external_residues=["B501:HBI"],
            stage_timings=[
                ("large optimization", 2.5),
                ("Hessian/mSeminario/frcmod export", 10.25),
            ],
        )
    )

    assert "Target metal selector: A301" in start_lines
    assert "add_resid: ['A10']" in start_lines
    assert "water model: tip3p" in start_lines
    assert "ion parameter set: 12_6" in start_lines
    assert "metal site charge/mult: 2 1" in start_lines
    assert "metal oxidation: 2" in start_lines
    assert "large model charge/mult: 1 1" in start_lines
    assert "[MetalAA] route completed; final summary follows." in final_lines

    final_lines_with_tleap = "".join(
        metal_report_module.format_metal_final_lines(
            artifacts=artifacts,
            atom_type_rows=[
                MetalAtomTypeRow(
                    residue="A301:ZN",
                    atom_name="ZN",
                    element="ZN",
                    old_type="zn",
                    new_type="M1",
                    charge=2.0,
                ),
                MetalAtomTypeRow(
                    residue="A10:HIS",
                    atom_name="ND1",
                    element="N",
                    old_type="na",
                    new_type="Y1",
                    charge=-0.4,
                ),
            ],
            ion_frcmods=["frcmod.ionslm_126_opc"],
        )
    )

    assert "[MetalAA] route completed; final summary follows." in final_lines_with_tleap


def test_abinitio_report_helpers_format_timing_paths_warnings_and_tleap(tmp_path: Path) -> None:
    timing_text = "".join(
        abinitio_report_module._timing_lines(
            SimpleNamespace(
                stage_timings=[
                    ("large RESP", 120.0),
                    ("large RESP/Gaussian ESP", 120.0),
                    ("Hessian evaluation", 30.0),
                    ("Hessian/mSeminario/frcmod export", 45.0),
                    ("site deployment export", 0.4),
                ]
            )
        )
    )
    assert "large RESP/Gaussian ESP" in timing_text
    assert "Hessian + mSeminario" in timing_text
    assert "site export" in timing_text
    assert timing_text.count("large RESP") == 1
    assert "slowest" in timing_text

    warning_text = "".join(abinitio_report_module._warning_lines(["repeat", "repeat", "unique"]))
    assert warning_text.count("repeat") == 1
    assert "unique" in warning_text

    nested = tmp_path / "work" / "demo.in"
    nested.parent.mkdir()
    nested.write_text("", encoding="utf-8")
    assert abinitio_report_module._relative_path(str(nested), base_dir=str(tmp_path)) == os.path.join("work", "demo.in")

    tleap_out = tmp_path / "metal_tleap.out"
    tleap_out.write_text(
        """
/amber/bin/teLeap: Warning!
There is a bond of 3.436 angstroms between OE1 and FE atoms:
Exiting LEaP: Errors = 0; Warnings = 14; Notes = 6.
""",
        encoding="utf-8",
    )
    summary = abinitio_report_module._parse_tleap_summary(str(tleap_out))
    assert summary.errors == 0
    assert summary.warnings == 14
    assert summary.notes == 6
    assert summary.checks == ("long bond OE1-FE = 3.436 A",)


def test_metal_parameter_parser_reads_bundled_dat_and_frcmod() -> None:
    parm_dir = metal_parameters_module.parm_dir()

    parm19 = metal_parameters_module.parse_amber_dat(parm_dir / "parm19.dat")
    ff19sb = metal_parameters_module.parse_amber_frcmod(parm_dir / "frcmod.ff19SB")
    gaff2 = metal_parameters_module.parse_amber_dat(parm_dir / "gaff2.dat")

    assert parm19.mass["C"] == pytest.approx(12.01)
    assert parm19.bond[metal_parameters_module.canonical_pair(("C", "N"))] == pytest.approx((490.0, 1.335))
    assert parm19.angle[metal_parameters_module.canonical_angle(("C", "N", "H"))] == pytest.approx((50.0, 120.0))
    assert parm19.dihedral[("X", "C", "N", "X")][0].amplitude == pytest.approx(2.5)
    assert parm19.improper[("X", "X", "C", "O")][0].amplitude == pytest.approx(10.5)
    assert parm19.nonbond["O2"] == pytest.approx((1.6612, 0.2100))
    assert parm19.nonbond["NB"] == pytest.approx(parm19.nonbond["N"])
    assert parm19.nonbond["CA"] == pytest.approx(parm19.nonbond["C*"])
    assert parm19.nonbond["CB"] == pytest.approx(parm19.nonbond["C*"])

    assert ff19sb.mass["2C"] == pytest.approx(12.01)
    assert ff19sb.dihedral[("X", "SH", "2C", "X")][0].amplitude == pytest.approx(0.25)
    assert ff19sb.improper[("CA", "CA", "CA", "2C")][0].amplitude == pytest.approx(1.1)
    assert ff19sb.nonbond["SH"] == pytest.approx((1.9825, 0.2824))

    assert gaff2.mass["c3"] == pytest.approx(12.01)
    assert gaff2.bond[metal_parameters_module.canonical_pair(("c3", "c3"))] == pytest.approx((228.89, 1.5354))
    assert gaff2.angle[metal_parameters_module.canonical_angle(("c3", "c3", "c3"))] == pytest.approx((59.87, 112.63))
    assert gaff2.nonbond["c3"] == pytest.approx((1.9069, 0.1078))


def test_load_parameters_uses_prom_and_optional_system_pool() -> None:
    protein_params = metal_parameters_module.load_parameters("ff14SB")
    system_params = metal_parameters_module.load_parameters("ff14SB", "opc")
    ff19_params = metal_parameters_module.load_parameters("ff19SB", "opc")

    assert protein_params.mass["CX"] == pytest.approx(12.01)
    assert "c3" not in protein_params.mass
    assert system_params.mass["c3"] == pytest.approx(12.01)
    assert ff19_params.mass["XC"] == pytest.approx(12.01)


def test_metal_parameter_parser_accepts_frcmod_section_aliases(tmp_path: Path) -> None:
    frcmod = tmp_path / "alias.frcmod"
    frcmod.write_text(
        "\n".join(
            [
                "MASS",
                "Y1  14.01",
                "",
                "BOND",
                "Y1-M1  100.0  2.000",
                "",
                "ANGL",
                "Y1-M1-Y2  45.0  109.5",
                "",
                "DIHE",
                "X -Y1-M1-X    2    0.50  180.0  2.",
                "",
                "IMPR",
                "X -X -Y1-M1    1.1  180.0  2.",
                "",
                "NONB",
                "Y1  1.500  0.200",
                "",
            ]
        ),
        encoding="utf-8",
    )

    parsed = metal_parameters_module.parse_amber_frcmod(frcmod)

    assert parsed.mass["Y1"] == pytest.approx(14.01)
    assert parsed.bond[metal_parameters_module.canonical_pair(("Y1", "M1"))] == pytest.approx((100.0, 2.0))
    assert parsed.angle[metal_parameters_module.canonical_angle(("Y1", "M1", "Y2"))] == pytest.approx((45.0, 109.5))
    assert parsed.dihedral[("X", "Y1", "M1", "X")][0].amplitude == pytest.approx(0.25)
    assert parsed.improper[("X", "X", "Y1", "M1")][0].amplitude == pytest.approx(1.1)
    assert parsed.nonbond["Y1"] == pytest.approx((1.5, 0.2))


def test_metal_ion_lj_lookup_uses_selected_water_and_ion_model() -> None:
    zn = _make_zn_residue("A", 301)
    fe = make_residue(
        "A",
        302,
        "",
        "FE",
        [make_atom(1, "FE", "FE", np.array((0.0, 0.0, 0.0)))],
        kind="ion",
    )
    fe["formal_charge"] = 3

    zn_frcmod, zn_type, zn_mass, zn_nonbond = metal_parameters_module.lookup_ion_lj_from_frcmod(
        watm="opc",
        ionm="12_6",
        residue=zn,
    )
    fe_frcmod, fe_type, fe_mass, fe_nonbond = metal_parameters_module.lookup_ion_lj_from_frcmod(
        watm="opc",
        ionm="12_6",
        residue=fe,
    )

    assert (zn_frcmod, zn_type) == ("frcmod.ionslm_126_opc", "Zn2+")
    assert zn_mass == pytest.approx(65.4)
    assert zn_nonbond == pytest.approx((1.2190, 0.00150903))
    assert (fe_frcmod, fe_type) == ("frcmod.ionslm_126_opc", "Fe3+")
    assert fe_mass == pytest.approx(55.85)
    assert fe_nonbond == pytest.approx((1.4000, 0.01570749))


def test_resp_input_files_write_charge_groups_in_both_stages(tmp_path: Path) -> None:
    model = model_module.rebuild_model_index(
        {
            "residues": [
                make_residue(
                    "A",
                    1,
                    "",
                    "GLY",
                    [
                        make_atom(1, "N", "N", np.array((0.0, 0.0, 0.0))),
                        make_atom(2, "CA", "C", np.array((1.0, 0.0, 0.0))),
                        make_atom(3, "C", "C", np.array((2.0, 0.0, 0.0))),
                    ],
                    kind="protein",
                )
            ],
            "explicit_pairs": set(),
            "_pair_cache": {},
        }
    )

    files = resp_module.write_resp_input_files(
        str(tmp_path),
        model,
        total_charge=0,
        chgmod=0,
        charge_groups=[([1, 2], 0.0), ([3], -1.0)],
    )

    for path in (files.resp1_in, files.resp2_in):
        text = Path(path).read_text(encoding="utf-8")
        assert f"{2:5d}{0.0:10.5f}\n{1:5d}{1:5d}{1:5d}{2:5d}\n" in text
        assert f"{1:5d}{-1.0:10.5f}\n{1:5d}{3:5d}\n" in text


def test_write_site_model_files_builds_its_own_donor_only_bond_graph() -> None:
    signature = inspect.signature(metal_export_module.write_site_model_files)

    assert "site_bond_pairs" not in signature.parameters


def _make_simple_site_model_nh_case() -> dict:
    zn = make_residue("A", 301, "", "ZN", [make_atom(1, "ZN", "ZN", np.array((0.0, 0.0, 0.0)))], kind="ion")
    donor_o = make_residue(
        "A",
        10,
        "",
        "GLY",
        [
            make_atom(2, "N", "N", np.array((-3.0, 2.5, 0.0))),
            make_atom(3, "CA", "C", np.array((-2.0, 1.5, 0.0))),
            make_atom(4, "C", "C", np.array((-1.0, 1.2, 0.0))),
            make_atom(5, "O", "O", np.array((-0.2, 0.2, 0.0))),
        ],
        kind="protein",
    )
    donor_o["_prev_peptide_key"] = None
    donor_o["_next_peptide_key"] = ("A", 11, "")
    sidechain = make_residue(
        "A",
        11,
        "",
        "SER",
        [
            make_atom(6, "N", "N", np.array((-1.0, 2.5, 0.0))),
            make_atom(7, "CA", "C", np.array((0.0, 2.5, 0.0))),
            make_atom(8, "C", "C", np.array((1.1, 2.5, 0.0))),
            make_atom(9, "O", "O", np.array((2.1, 2.5, 0.0))),
            make_atom(10, "CB", "C", np.array((0.0, 1.1, 0.0))),
            make_atom(11, "OG", "O", np.array((0.3, 0.2, 0.0))),
            make_atom(12, "HN", "H", np.array((-1.2, 3.3, 0.0))),
        ],
        kind="protein",
    )
    sidechain["_prev_peptide_key"] = ("A", 10, "")
    sidechain["_next_peptide_key"] = None
    return {
        "name": "site_model",
        "target_key": ("A", 301, ""),
        "residues": [zn, donor_o, sidechain],
        "donor_atoms": {
            ("A", 10, ""): ["O"],
            ("A", 11, ""): ["OG"],
        },
    }


def _make_cys_zn_site_model_for_export() -> dict:
    zn = make_residue("A", 301, "", "ZN", [make_atom(1, "ZN", "ZN", np.array((0.0, 0.0, 0.0)))], kind="ion")
    zn["formal_charge"] = 2
    cys = make_residue(
        "A",
        10,
        "",
        "CYS",
        [
            make_atom(2, "N", "N", np.array((4.8, 2.3, 0.0))),
            make_atom(3, "H", "H", np.array((4.1, 2.9, 0.0))),
            make_atom(4, "CA", "C", np.array((4.95, 1.0, 0.0))),
            make_atom(5, "HA", "H", np.array((5.2, 1.0, 1.0))),
            make_atom(6, "C", "C", np.array((6.25, 0.7, 0.0))),
            make_atom(7, "O", "O", np.array((7.1, 1.5, 0.0))),
            make_atom(8, "CB", "C", np.array((3.95, 0.0, 0.0))),
            make_atom(9, "HB2", "H", np.array((4.2, -0.7, 0.8))),
            make_atom(10, "HB3", "H", np.array((4.2, -0.7, -0.8))),
            make_atom(11, "SG", "S", np.array((2.25, 0.0, 0.0))),
        ],
        kind="protein",
    )
    cys["_prev_peptide_key"] = ("A", 9, "")
    cys["_next_peptide_key"] = ("A", 11, "")
    return model_module.rebuild_model_index(
        {
            "name": "site_model",
            "target_key": ("A", 301, ""),
            "residues": [zn, cys],
            "donor_atoms": {("A", 10, ""): ["SG"]},
            "explicit_pairs": set(),
            "_pair_cache": {},
        }
    )


def _make_metal_export_artifacts(tmp_path: Path) -> MetalArtifacts:
    return MetalArtifacts(
        files={
            "large_raw_pdb": str(tmp_path / "raw.pdb"),
            "large_opt_pdb": str(tmp_path / "opt.pdb"),
            "site_pdb": str(tmp_path / "site.pdb"),
            "gaussian_input": str(tmp_path / "resp.gjf"),
            "mol2": str(tmp_path / "site.mol2"),
            "site_mol2": str(tmp_path / "site.mol2"),
            "frcmod": str(tmp_path / "metal.frcmod"),
            "tleap_pdb": str(tmp_path / "metal_tleap.pdb"),
            "tleap_input": str(tmp_path / "metal_tleap.in"),
        },
        resp_files={},
    )


def _metal_fit_terms(site_model: dict):
    bond_terms, angle_terms = model_module.build_bond_angle_terms(site_model, source_structure=site_model)
    metal_indices = {
        atom_index
        for atom_index, (residue, _atom) in enumerate(model_module.flatten_model_atoms(site_model), start=1)
        if residue.get("kind") == "ion"
    }
    for bond in bond_terms:
        if metal_indices.intersection(bond.atoms):
            bond.kBond = 123.4
            bond.rEq = 2.25
    for angle in angle_terms:
        if metal_indices.intersection(angle.atoms):
            angle.kTheta = 45.6
            angle.thetaEq = np.deg2rad(109.5)
    return bond_terms, angle_terms


def test_write_site_frcmod_inherits_standard_terms_and_writes_nonbon(tmp_path: Path) -> None:
    site_model = _make_cys_zn_site_model_for_export()
    typing = metal_export_module._build_site_typing(site_model, watm="opc", ionm="12_6")
    bond_terms, angle_terms = _metal_fit_terms(site_model)
    artifacts = _make_metal_export_artifacts(tmp_path)

    metal_export_module.write_site_frcmod(
        artifacts,
        site_model=site_model,
        bond_terms=bond_terms,
        angle_terms=angle_terms,
        typing=typing,
    )

    text = Path(artifacts.files["frcmod"]).read_text(encoding="utf-8")
    lines = text.splitlines()
    assert lines[0] == "REMARK MAPLE MetalAA generated metal-site frcmod"
    assert lines[1] == ""
    assert lines[2] == "MASS"
    assert re.search(r"^Y1\s+2\.0000\s+0\.25000000", text, re.MULTILINE)
    assert re.search(r"^M1\s+1\.2190\s+0\.00150903", text, re.MULTILINE)
    assert re.search(r"^(?:2C-Y1|Y1-2C)\s+237\.0+\s+1\.8100", text, re.MULTILINE)
    assert re.search(r"^(?:M1-Y1|Y1-M1)\s+123\.4000\s+2\.2500", text, re.MULTILINE)
    assert "DIHE\n\nIMPROPER\n\nNONBON\n" not in text
    dihe_lines = text.split("\nDIHE\n", 1)[1].split("\nIMPROPER\n", 1)[0].splitlines()
    improper_lines = text.split("\nIMPROPER\n", 1)[1].split("\nNONBON\n", 1)[0].splitlines()
    assert all(len(line.split()) == 5 for line in dihe_lines if line.strip())
    assert all(len(line.split()) == 4 for line in improper_lines if line.strip())
    assert not any(re.match(r"^\S+\s+1\s+", line) for line in improper_lines if line.strip())
    assert all("Y1" not in line for line in improper_lines if line.strip())


def test_metal_site_typing_uses_prom_for_internal_protein_atom_types() -> None:
    site_model = _make_cys_zn_site_model_for_export()
    ca_index = next(
        index
        for index, (_residue, atom) in enumerate(model_module.flatten_model_atoms(site_model), start=1)
        if atom["name"] == "CA"
    )

    ff14_typing = metal_export_module._build_site_typing(site_model, watm="opc", ionm="12_6", prom="ff14SB")
    ff19_typing = metal_export_module._build_site_typing(site_model, watm="opc", ionm="12_6", prom="ff19SB")

    assert ff14_typing.old_type_by_index[ca_index] == "CX"
    assert ff19_typing.old_type_by_index[ca_index] == "XC"


def test_metal_site_typing_keeps_terminal_ca_as_cx_under_ff19sb() -> None:
    zn = make_residue("A", 301, "", "ZN", [make_atom(1, "ZN", "ZN", np.array((0.0, 0.0, 0.0)))], kind="ion")
    gly = make_residue(
        "A",
        10,
        "",
        "GLY",
        [
            make_atom(2, "N", "N", np.array((1.0, 0.0, 0.0))),
            make_atom(3, "H", "H", np.array((1.0, 0.9, 0.0))),
            make_atom(4, "CA", "C", np.array((2.2, 0.0, 0.0))),
            make_atom(5, "C", "C", np.array((3.0, 1.1, 0.0))),
            make_atom(6, "O", "O", np.array((4.0, 1.1, 0.0))),
        ],
        kind="protein",
    )
    gly["_next_peptide_key"] = ("A", 11, "")
    site_model = model_module.rebuild_model_index(
        {
            "name": "site_model",
            "target_key": ("A", 301, ""),
            "residues": [zn, gly],
            "donor_atoms": {},
            "explicit_pairs": set(),
            "_pair_cache": {},
        }
    )
    ca_index = next(
        index
        for index, (_residue, atom) in enumerate(model_module.flatten_model_atoms(site_model), start=1)
        if atom["name"] == "CA"
    )

    typing = metal_export_module._build_site_typing(site_model, watm="opc", ionm="12_6", prom="ff19SB")

    assert typing.old_type_by_index[ca_index] == "CX"


def test_frcmod_formats_proper_and_improper_torsions_with_amber_fields() -> None:
    term = metal_parameters_module.TorsionParameter(amplitude=1.2, phase_deg=180.0, periodicity=2.0)

    dihedral = metal_export_module._format_dihedral_line(("N", "XC", "2C", "Y1"), term)
    improper = metal_export_module._format_improper_line(("N", "XC", "2C", "Y1"), term)

    assert dihedral.split() == ["N-XC-2C-Y1", "1", "1.2000", "180.0000", "2.0000"]
    assert improper.split() == ["N-XC-2C-Y1", "1.2000", "180.0000", "2.0000"]
    assert "N -" not in dihedral
    assert "N -" not in improper


def test_match_dihedral_template_reports_wildcard_template_and_orientation() -> None:
    term = metal_parameters_module.TorsionParameter(amplitude=4.75, phase_deg=180.0, periodicity=2.0)

    match = metal_parameters_module.match_dihedral_template(
        ("cd", "nc", "cd", "ce"),
        {("X", "cd", "nc", "X"): [term]},
    )

    assert match is not None
    template, terms, reversed_match = match
    assert template == ("X", "cd", "nc", "X")
    assert terms == [term]
    assert reversed_match is True


def test_write_site_frcmod_filters_non_donor_metal_bonds(tmp_path: Path) -> None:
    site_model = _make_cys_zn_site_model_for_export()
    cys = next(residue for residue in site_model["residues"] if residue["resname"] == "CYS")
    search_atom(cys, "C")["xyz"] = np.array((0.0, 1.45, 0.0))
    refresh_resid(cys)
    model_module.rebuild_model_index(site_model)
    typing = metal_export_module._build_site_typing(site_model, watm="opc", ionm="12_6")
    bond_terms, angle_terms = _metal_fit_terms(site_model)
    artifacts = _make_metal_export_artifacts(tmp_path)

    metal_export_module.write_site_frcmod(
        artifacts,
        site_model=site_model,
        bond_terms=bond_terms,
        angle_terms=angle_terms,
        typing=typing,
    )

    text = Path(artifacts.files["frcmod"]).read_text(encoding="utf-8")
    assert re.search(r"^(?:M1-Y1|Y1-M1)\s+123\.4000\s+2\.2500", text, re.MULTILINE)
    assert not re.search(r"^(?:C\s*-M1|M1-C\s*)\s+", text, re.MULTILINE)


def test_write_site_frcmod_fails_when_selected_metal_bond_is_unfitted(tmp_path: Path) -> None:
    site_model = _make_cys_zn_site_model_for_export()
    typing = metal_export_module._build_site_typing(site_model, watm="opc", ionm="12_6")
    artifacts = _make_metal_export_artifacts(tmp_path)

    with pytest.raises(ValueError, match="fitted MetalAA BOND"):
        metal_export_module.write_site_frcmod(
            artifacts,
            site_model=site_model,
            bond_terms=[],
            angle_terms=[],
            typing=typing,
        )


def test_write_site_frcmod_fails_when_selected_metal_angle_is_unfitted(tmp_path: Path) -> None:
    site_model = _make_cys_zn_site_model_for_export()
    typing = metal_export_module._build_site_typing(site_model, watm="opc", ionm="12_6")
    bond_terms, _angle_terms = _metal_fit_terms(site_model)
    artifacts = _make_metal_export_artifacts(tmp_path)

    with pytest.raises(ValueError, match="fitted MetalAA ANGLE"):
        metal_export_module.write_site_frcmod(
            artifacts,
            site_model=site_model,
            bond_terms=bond_terms,
            angle_terms=[],
            typing=typing,
        )


def test_write_site_model_files_preserves_input_atom_names_in_tleap_pdb(tmp_path: Path) -> None:
    site_model = model_module.rebuild_model_index(
        {
            **_make_simple_site_model_nh_case(),
            "explicit_pairs": set(),
            "_pair_cache": {},
        }
    )
    site_model["residues"][0]["formal_charge"] = 2
    structure = model_module.rebuild_model_index(
        {
            "path": "source.pdb",
            "residues": [
                *[deepcopy(residue) for residue in site_model["residues"]],
                make_residue(
                    "A",
                    900,
                    "",
                    "HBI",
                    [make_atom(100, "N1", "N", np.array((5.0, 5.0, 5.0)))],
                    kind="ligand",
                ),
            ],
            "explicit_pairs": set(),
            "_pair_cache": {},
        }
    )
    artifacts = _make_metal_export_artifacts(tmp_path)

    metal_export_module.write_site_model_files(
        artifacts,
        structure=structure,
        site_model=site_model,
        watm="opc",
        ionm="12_6",
    )

    tleap_pdb = Path(artifacts.files["tleap_pdb"]).read_text(encoding="utf-8").splitlines()
    tleap_input = Path(artifacts.files["tleap_input"]).read_text(encoding="utf-8")
    atom_name_by_serial = {int(line[6:11]): line[12:16].strip() for line in tleap_pdb if line.startswith("ATOM")}
    assert atom_name_by_serial[12] == "HN"
    assert '{ "Y1" "O" "sp3" }' in tleap_input
    assert '{ "M1" "Zn" "sp3" }' in tleap_input
    assert "# External residues not parameterized by MetalAA: A900:HBI" in tleap_input
    assert "# Load matching ligand/NCAA templates before loadpdb for these residues." in tleap_input
    assert "loadmol2 HBI_ff.mol2" not in tleap_input
    assert tleap_input.index("# External residues not parameterized") < tleap_input.index("mol = loadpdb")


def test_write_site_model_files_preserves_aliased_cofactor_frcmods(tmp_path: Path) -> None:
    site_model = model_module.rebuild_model_index(
        {
            **_make_simple_site_model_nh_case(),
            "explicit_pairs": set(),
            "_pair_cache": {},
        }
    )
    structure = model_module.rebuild_model_index(deepcopy(site_model))
    artifacts = _make_metal_export_artifacts(tmp_path)
    cofactor_frcmod = str(tmp_path / "hem_orig.frcmod")
    artifacts.cofactor_frcmods.append(cofactor_frcmod)

    _site_pdb, _site_mol2, typing = metal_export_module.write_site_model_files(
        artifacts,
        structure=structure,
        site_model=site_model,
        watm="opc",
        ionm="12_6",
        cofactor_frcmods=artifacts.cofactor_frcmods,
    )

    assert artifacts.cofactor_frcmods == [cofactor_frcmod]
    assert typing.cofactor_frcmods == [cofactor_frcmod]


def test_cofactor_orig_frcmod_is_folded_into_final_frcmod_with_renamed_terms(tmp_path: Path) -> None:
    fe = make_residue("A", 301, "", "FE", [make_atom(1, "FE", "FE", np.array((0.0, 0.0, 0.0)))], kind="ion")
    fe["formal_charge"] = 3
    hem = make_residue(
        "A",
        500,
        "",
        "HEM",
        [
            make_atom(2, "N1", "N", np.array((1.9, 0.0, 0.0))),
            make_atom(3, "C1", "C", np.array((3.1, 0.0, 0.0))),
            make_atom(4, "C2", "C", np.array((4.2, 0.7, 0.0))),
            make_atom(5, "C3", "C", np.array((4.2, -0.7, 0.0))),
        ],
        kind="cofactor",
    )
    hem["atoms"][0]["atom_type"] = "nx"
    hem["atoms"][1]["atom_type"] = "cx"
    hem["atoms"][2]["atom_type"] = "cc"
    hem["atoms"][3]["atom_type"] = "ce"
    site_model = model_module.rebuild_model_index(
        {
            "name": "site_model",
            "target_key": ("A", 301, ""),
            "residues": [fe, hem],
            "donor_atoms": {("A", 500, ""): ["N1"]},
            "explicit_pairs": {
                tuple(sorted((1, 2))),
                tuple(sorted((2, 3))),
                tuple(sorted((3, 4))),
                tuple(sorted((3, 5))),
                tuple(sorted((4, 5))),
            },
            "_pair_cache": {},
        }
    )
    cofactor_frcmod = tmp_path / "hem_orig.frcmod"
    cofactor_frcmod.write_text(
        """MASS
nx 14.0100
cx 12.0100
cc 12.0200
ce 12.0300

BOND
nx-cx   321.0000  1.2300
cx-cc   222.0000  1.4100
cx-ce   223.0000  1.4200
cc-ce   224.0000  1.4300

ANGLE
nx-cx-cc  44.0000  120.0000
nx-cx-ce  45.0000  121.0000

DIHE
nx-cx-cc-ce  1  0.5000  180.0000  2.0000
nx-cx-ce-cc  1  0.5000  180.0000  2.0000

IMPROPER
nx-cc-cx-ce  1.1000  180.0000  2.0000

NONBON
nx 1.7000 0.12000000
cx 1.9000 0.09000000
cc 1.8000 0.08000000
ce 1.8100 0.07000000
""",
        encoding="utf-8",
    )
    artifacts = _make_metal_export_artifacts(tmp_path)

    _site_pdb, _site_mol2, typing = metal_export_module.write_site_model_files(
        artifacts,
        structure=site_model,
        site_model=site_model,
        watm="opc",
        ionm="12_6",
        cofactor_frcmods=[str(cofactor_frcmod)],
    )
    bond_terms, angle_terms = model_module.build_bond_angle_terms(
        site_model,
        source_structure=site_model,
        bond_pairs=[(1, 2), (2, 3), (3, 4), (3, 5), (4, 5)],
    )
    for bond in bond_terms:
        if 1 in bond.atoms:
            bond.kBond = 111.0
            bond.rEq = 1.90
    for angle in angle_terms:
        if 1 in angle.atoms:
            angle.kTheta = 22.0
            angle.thetaEq = np.deg2rad(120.0)

    metal_export_module.write_site_frcmod(
        artifacts,
        site_model=site_model,
        bond_terms=bond_terms,
        angle_terms=angle_terms,
        typing=typing,
    )

    tleap_input = Path(artifacts.files["tleap_input"]).read_text(encoding="utf-8")
    frcmod_text = Path(artifacts.files["frcmod"]).read_text(encoding="utf-8")
    assert "loadamberparams hem_orig.frcmod" not in tleap_input
    assert "loadamberparams metal.frcmod" in tleap_input
    assert re.search(r"^nx\s+14\.0100", frcmod_text, re.MULTILINE)
    assert re.search(r"^cx\s+12\.0100", frcmod_text, re.MULTILINE)
    assert re.search(r"^Y1\s+14\.0100", frcmod_text, re.MULTILINE)
    assert re.search(r"^(?:nx-cx|cx-nx)\s+321\.0000\s+1\.2300", frcmod_text, re.MULTILINE)
    assert re.search(r"^Y1-cx\s+321\.0000\s+1\.2300", frcmod_text, re.MULTILINE)
    assert re.search(r"^(?:nx-cx-cc|cc-cx-nx)\s+44\.0000\s+120\.0000", frcmod_text, re.MULTILINE)
    assert re.search(r"^Y1-cx-cc\s+44\.0000\s+120\.0000", frcmod_text, re.MULTILINE)
    assert re.search(r"^nx-cx-cc-ce\s+1\s+0\.5000\s+180\.0000\s+2\.0000", frcmod_text, re.MULTILINE)
    assert re.search(r"^Y1-cx-cc-ce\s+1\s+0\.5000\s+180\.0000\s+2\.0000", frcmod_text, re.MULTILINE)
    assert re.search(r"^Y1-cc-cx-ce\s+1\.1000\s+180\.0000\s+2\.0000", frcmod_text, re.MULTILINE)
    assert re.search(r"^nx\s+1\.7000\s+0\.12000000", frcmod_text, re.MULTILINE)
    assert re.search(r"^Y1\s+1\.7000\s+0\.12000000", frcmod_text, re.MULTILINE)
    assert re.search(r"^(?:M1-Y1|Y1-M1)\s+111\.0000\s+1\.9000", frcmod_text, re.MULTILINE)
    assert re.search(r"^(?:M1-Y1-cx|cx-Y1-M1)\s+22\.0000\s+120\.0000", frcmod_text, re.MULTILINE)


def test_multiple_cofactor_frcmods_remap_terms_from_their_own_residue_source(tmp_path: Path) -> None:
    fe = make_residue("A", 301, "", "FE", [make_atom(1, "FE", "FE", np.array((0.0, 0.0, 0.0)))], kind="ion")
    fe["formal_charge"] = 3
    lig1 = make_residue(
        "A",
        500,
        "",
        "LGA",
        [
            make_atom(2, "N1", "N", np.array((1.9, 0.0, 0.0))),
            make_atom(3, "C1", "C", np.array((3.1, 0.0, 0.0))),
        ],
        kind="cofactor",
    )
    lig2 = make_residue(
        "A",
        501,
        "",
        "LGB",
        [
            make_atom(4, "N1", "N", np.array((-1.9, 0.0, 0.0))),
            make_atom(5, "C1", "C", np.array((-3.1, 0.0, 0.0))),
        ],
        kind="cofactor",
    )
    for residue in (lig1, lig2):
        residue["atoms"][0]["atom_type"] = "nx"
        residue["atoms"][1]["atom_type"] = "cx"
    site_model = model_module.rebuild_model_index(
        {
            "name": "site_model",
            "target_key": ("A", 301, ""),
            "residues": [fe, lig1, lig2],
            "donor_atoms": {("A", 500, ""): ["N1"], ("A", 501, ""): ["N1"]},
            "explicit_pairs": {
                tuple(sorted((2, 3))),
                tuple(sorted((4, 5))),
            },
            "_pair_cache": {},
        }
    )
    first_frcmod = tmp_path / "first_orig.frcmod"
    first_frcmod.write_text(
        """MASS
nx 14.0100
cx 12.0100

BOND
nx-cx   111.0000  1.1100

ANGLE

DIHE

IMPROPER

NONBON
nx 1.7000 0.12000000
cx 1.9000 0.09000000
""",
        encoding="utf-8",
    )
    second_frcmod = tmp_path / "second_orig.frcmod"
    second_frcmod.write_text(
        """MASS
nx 14.0100
cx 12.0100

BOND
nx-cx   222.0000  1.2200

ANGLE

DIHE

IMPROPER

NONBON
nx 1.7000 0.12000000
cx 1.9000 0.09000000
""",
        encoding="utf-8",
    )
    artifacts = _make_metal_export_artifacts(tmp_path)
    _site_pdb, _site_mol2, typing = metal_export_module.write_site_model_files(
        artifacts,
        structure=site_model,
        site_model=site_model,
        watm="opc",
        ionm="12_6",
        cofactor_frcmods=[str(first_frcmod), str(second_frcmod)],
        cofactor_frcmod_by_residue={
            ("A", 500, ""): str(first_frcmod),
            ("A", 501, ""): str(second_frcmod),
        },
    )
    bond_terms, angle_terms = model_module.build_bond_angle_terms(
        site_model,
        source_structure=site_model,
        bond_pairs=metal_export_module._export_bond_pairs(site_model),
    )
    for bond in bond_terms:
        if 1 in bond.atoms:
            bond.kBond = 333.0
            bond.rEq = 1.90
    for angle in angle_terms:
        if 1 in angle.atoms:
            angle.kTheta = 44.0
            angle.thetaEq = np.deg2rad(120.0)

    metal_export_module.write_site_frcmod(
        artifacts,
        site_model=site_model,
        bond_terms=bond_terms,
        angle_terms=angle_terms,
        typing=typing,
    )

    frcmod_text = Path(artifacts.files["frcmod"]).read_text(encoding="utf-8")
    assert re.search(r"^Y1-cx\s+111\.0000\s+1\.1100", frcmod_text, re.MULTILINE)
    assert re.search(r"^Y2-cx\s+222\.0000\s+1\.2200", frcmod_text, re.MULTILINE)
    assert len(re.findall(r"^(?:M1-Y[12]|Y[12]-M1)\s+333\.0000\s+1\.9000", frcmod_text, re.MULTILINE)) == 2


def test_cofactor_remap_preserves_wildcard_dihedral_for_heme_donor(tmp_path: Path) -> None:
    fe = make_residue("A", 301, "", "FE", [make_atom(1, "FE", "FE", np.array((0.0, 0.0, 0.0)))], kind="ion")
    fe["formal_charge"] = 3
    cym = make_residue(
        "A",
        400,
        "",
        "CYM",
        [make_atom(2, "SG", "S", np.array((0.0, 0.0, 2.2)))],
        kind="ligand",
    )
    cym["atoms"][0]["atom_type"] = "ss"
    cym["atoms"][0]["charge"] = -0.2
    hem = make_residue(
        "A",
        500,
        "",
        "HEM",
        [
            make_atom(3, "NA", "N", np.array((1.9, 0.0, 0.0))),
            make_atom(4, "NB", "N", np.array((0.0, 1.9, 0.0))),
            make_atom(5, "ND", "N", np.array((-1.9, 0.0, 0.0))),
            make_atom(6, "C4C", "C", np.array((0.0, -3.0, 0.0))),
            make_atom(7, "CHD", "C", np.array((0.9, -4.0, 0.0))),
            make_atom(8, "NC", "N", np.array((0.0, -1.9, 0.0))),
            make_atom(9, "C1C", "C", np.array((-0.9, -1.0, 0.0))),
        ],
        kind="cofactor",
    )
    for atom, atom_type in zip(hem["atoms"], ("nd", "nc", "nd", "cd", "ce", "nc", "cd")):
        atom["atom_type"] = atom_type
    site_model = model_module.rebuild_model_index(
        {
            "name": "site_model",
            "target_key": ("A", 301, ""),
            "residues": [fe, cym, hem],
            "donor_atoms": {("A", 400, ""): ["SG"], ("A", 500, ""): ["NA", "NB", "ND", "NC"]},
            "explicit_pairs": {
                tuple(sorted((6, 7))),
                tuple(sorted((6, 8))),
                tuple(sorted((8, 9))),
            },
            "_pair_cache": {},
        }
    )
    cofactor_frcmod = tmp_path / "hem_orig.frcmod"
    cofactor_frcmod.write_text(
        """MASS
ss 32.0600
nd 14.0100
nc 14.0100
cd 12.0100
ce 12.0100

BOND
ce-cd  300.0000  1.4000
cd-nc  310.0000  1.3500
nc-cd  310.0000  1.3500

ANGLE
ce-cd-nc  67.6000  120.7600
cd-nc-cd  71.0000  110.0000

DIHE
X -cd-nc-X  2  9.5000  180.0000  2.0000
ce-cd-nc-cd  2  9.5000  180.0000  2.0000

IMPROPER

NONBON
ss 1.9825 0.28240000
nd 1.8240 0.17000000
nc 1.8240 0.17000000
cd 1.9080 0.08600000
ce 1.9080 0.08600000
""",
        encoding="utf-8",
    )
    artifacts = _make_metal_export_artifacts(tmp_path)
    _site_pdb, _site_mol2, typing = metal_export_module.write_site_model_files(
        artifacts,
        structure=site_model,
        site_model=site_model,
        watm="opc",
        ionm="12_6",
        cofactor_frcmods=[str(cofactor_frcmod)],
    )
    bond_terms, angle_terms = model_module.build_bond_angle_terms(
        site_model,
        source_structure=site_model,
        bond_pairs=metal_export_module._export_bond_pairs(site_model),
    )
    for bond in bond_terms:
        if 1 in bond.atoms:
            bond.kBond = 100.0
            bond.rEq = 2.0
    for angle in angle_terms:
        if 1 in angle.atoms:
            angle.kTheta = 30.0
            angle.thetaEq = np.deg2rad(120.0)

    metal_export_module.write_site_frcmod(
        artifacts,
        site_model=site_model,
        bond_terms=bond_terms,
        angle_terms=angle_terms,
        typing=typing,
    )

    frcmod_text = Path(artifacts.files["frcmod"]).read_text(encoding="utf-8")
    tleap_input = Path(artifacts.files["tleap_input"]).read_text(encoding="utf-8")
    assert "loadamberparams hem_orig.frcmod" not in tleap_input
    assert re.search(r"^(?:ce-cd-Y5|Y5-cd-ce)\s+67\.6000\s+120\.7600", frcmod_text, re.MULTILINE)
    assert re.search(r"^X-cd-Y5-X\s+1\s+4\.7500\s+180\.0000\s+2\.0000", frcmod_text, re.MULTILINE)
    assert "ce-cd-Y5-cd" not in frcmod_text
    assert re.search(r"^(?:M1-Y5-cd|cd-Y5-M1)\s+30\.0000\s+120\.0000", frcmod_text, re.MULTILINE)


def test_cofactor_remap_inherits_reverse_exact_dihedral_for_heme_donor(tmp_path: Path) -> None:
    fe = make_residue("A", 301, "", "FE", [make_atom(1, "FE", "FE", np.array((0.0, 0.0, 0.0)))], kind="ion")
    fe["formal_charge"] = 3
    cym = make_residue(
        "A",
        400,
        "",
        "CYM",
        [make_atom(2, "SG", "S", np.array((0.0, 0.0, 2.2)))],
        kind="ligand",
    )
    cym["atoms"][0]["atom_type"] = "ss"
    hem = make_residue(
        "A",
        500,
        "",
        "HEM",
        [
            make_atom(3, "C4D", "C", np.array((3.0, 0.0, 0.0))),
            make_atom(4, "CHD", "C", np.array((2.0, 0.0, 0.0))),
            make_atom(5, "C1A", "C", np.array((1.0, 0.0, 0.0))),
            make_atom(6, "NA", "N", np.array((0.0, 0.0, 0.0))),
        ],
        kind="cofactor",
    )
    for atom, atom_type in zip(hem["atoms"], ("cd", "ce", "cc", "nd")):
        atom["atom_type"] = atom_type
    site_model = model_module.rebuild_model_index(
        {
            "name": "site_model",
            "target_key": ("A", 301, ""),
            "residues": [fe, cym, hem],
            "donor_atoms": {("A", 400, ""): ["SG"], ("A", 500, ""): ["NA"]},
            "explicit_pairs": {
                tuple(sorted((3, 4))),
                tuple(sorted((4, 5))),
                tuple(sorted((5, 6))),
            },
            "_pair_cache": {},
        }
    )
    cofactor_frcmod = tmp_path / "hem_orig.frcmod"
    cofactor_frcmod.write_text(
        """MASS
ss 32.0600
nd 14.0100
cc 12.0100
ce 12.0100
cd 12.0100

BOND
cc-nd  310.0000  1.3500
ce-cc  300.0000  1.4000
cd-ce  300.0000  1.4000

ANGLE
ce-cc-nd  67.0000  120.0000
cd-ce-cc  68.0000  121.0000

DIHE
nd-cc-ce-cd  1  1.0000  180.0000  2.0000

IMPROPER

NONBON
ss 1.9825 0.28240000
nd 1.8240 0.17000000
cc 1.9080 0.08600000
ce 1.9080 0.08600000
cd 1.9080 0.08600000
""",
        encoding="utf-8",
    )
    artifacts = _make_metal_export_artifacts(tmp_path)
    _site_pdb, _site_mol2, typing = metal_export_module.write_site_model_files(
        artifacts,
        structure=site_model,
        site_model=site_model,
        watm="opc",
        ionm="12_6",
        cofactor_frcmods=[str(cofactor_frcmod)],
    )
    bond_terms, angle_terms = model_module.build_bond_angle_terms(
        site_model,
        source_structure=site_model,
        bond_pairs=metal_export_module._export_bond_pairs(site_model),
    )
    for bond in bond_terms:
        if 1 in bond.atoms:
            bond.kBond = 100.0
            bond.rEq = 2.0
    for angle in angle_terms:
        if 1 in angle.atoms:
            angle.kTheta = 30.0
            angle.thetaEq = np.deg2rad(120.0)

    metal_export_module.write_site_frcmod(
        artifacts,
        site_model=site_model,
        bond_terms=bond_terms,
        angle_terms=angle_terms,
        typing=typing,
    )

    frcmod_text = Path(artifacts.files["frcmod"]).read_text(encoding="utf-8")
    assert re.search(r"^(?:Y2-cc-ce-cd|cd-ce-cc-Y2)\s+1\s+1\.0000\s+180\.0000\s+2\.0000", frcmod_text, re.MULTILINE)
    assert "Y2-cc-ce-cd    1     0.0000" not in frcmod_text


def test_write_site_frcmod_fails_when_renamed_nonmetal_dihedral_has_no_template(tmp_path: Path) -> None:
    cof = make_residue(
        "A",
        500,
        "",
        "COF",
        [
            make_atom(1, "A1", "C", np.array((0.0, 0.0, 0.0))),
            make_atom(2, "B1", "C", np.array((1.4, 0.0, 0.0))),
            make_atom(3, "D1", "N", np.array((2.8, 0.0, 0.0))),
            make_atom(4, "C1", "C", np.array((4.2, 0.0, 0.0))),
        ],
        kind="cofactor",
    )
    for atom, atom_type in zip(cof["atoms"], ("qa", "qb", "qc", "qd")):
        atom["atom_type"] = atom_type
    site_model = model_module.rebuild_model_index(
        {
            "name": "site_model",
            "residues": [cof],
            "donor_atoms": {("A", 500, ""): ["D1"]},
            "explicit_pairs": {tuple(sorted((1, 2))), tuple(sorted((2, 3))), tuple(sorted((3, 4)))},
            "_pair_cache": {},
        }
    )
    cofactor_frcmod = tmp_path / "cof_orig.frcmod"
    cofactor_frcmod.write_text(
        """MASS
qa 12.0100
qb 12.0100
qc 14.0100
qd 12.0100

BOND
qa-qb  100.0000  1.4000
qb-qc  100.0000  1.4000
qc-qd  100.0000  1.4000

ANGLE
qa-qb-qc  50.0000  120.0000
qb-qc-qd  50.0000  120.0000

DIHE

IMPROPER

NONBON
qa 1.5000 0.01000000
qb 1.5000 0.01000000
qc 1.5000 0.01000000
qd 1.5000 0.01000000
""",
        encoding="utf-8",
    )
    typing = metal_export_module._build_site_typing(
        site_model,
        watm="opc",
        ionm="12_6",
        cofactor_frcmods=[str(cofactor_frcmod)],
    )
    bond_terms, angle_terms = model_module.build_bond_angle_terms(site_model, source_structure=site_model)
    artifacts = _make_metal_export_artifacts(tmp_path)

    with pytest.raises(ValueError, match="DIHE"):
        metal_export_module.write_site_frcmod(
            artifacts,
            site_model=site_model,
            bond_terms=bond_terms,
            angle_terms=angle_terms,
            typing=typing,
        )


def test_identify_metal_site_core_auto_detects_and_merges_manual_residues(tmp_path: Path) -> None:
    pdb = "".join(
        [
            _pdb_atom("HETATM", 1, "ZN", "ZN", "A", 301, 0.0, 0.0, 0.0, "ZN"),
            _pdb_atom("ATOM", 2, "N", "CYS", "A", 10, -3.0, 2.8, 0.0, "N"),
            _pdb_atom("ATOM", 3, "CA", "CYS", "A", 10, -2.0, 1.8, 0.0, "C"),
            _pdb_atom("ATOM", 4, "C", "CYS", "A", 10, -1.0, 2.3, 0.0, "C"),
            _pdb_atom("ATOM", 5, "O", "CYS", "A", 10, -0.2, 3.1, 0.0, "O"),
            _pdb_atom("ATOM", 6, "CB", "CYS", "A", 10, -2.0, 0.4, 0.0, "C"),
            _pdb_atom("ATOM", 7, "SG", "CYS", "A", 10, -0.7, 0.2, 0.0, "S"),
            _pdb_atom("HETATM", 8, "O", "HOH", "A", 401, 2.4, 0.0, 0.0, "O"),
            _pdb_atom("HETATM", 9, "O", "HOH", "A", 402, -2.4, 0.0, 0.0, "O"),
            _pdb_atom("ATOM", 10, "N", "GLU", "A", 20, 4.0, 0.0, 0.0, "N"),
            _pdb_atom("ATOM", 11, "CA", "GLU", "A", 20, 5.0, 0.0, 0.0, "C"),
            _pdb_atom("ATOM", 12, "C", "GLU", "A", 20, 6.0, 0.0, 0.0, "C"),
            _pdb_atom("ATOM", 13, "O", "GLU", "A", 20, 7.0, 0.0, 0.0, "O"),
            "END\n",
        ]
    )
    path = _write_text(tmp_path / "metal_core.pdb", pdb)

    info = identify_metal_site_core(read_pdb(path), target="A301", add_resid="A401 A402")

    assert [(res["chain"], res["resseq"]) for res in info["auto_core_residues"]] == [("A", 10)]
    assert [(res["chain"], res["resseq"]) for res in info["manual_core_residues"]] == [("A", 401), ("A", 402)]
    assert info["donor_atoms"][("A", 10, "")] == ["SG"]
    assert any("potentially charged standard residues" in warning for warning in info["warnings"])


def test_identify_metal_site_core_set_bonded_overrides_auto_donor_detection(tmp_path: Path) -> None:
    pdb = "".join(
        [
            _pdb_atom("HETATM", 144, "ZN", "ZN", "A", 301, 0.0, 0.0, 0.0, "ZN"),
            _pdb_atom("ATOM", 1, "SG", "CYS", "A", 10, 8.0, 0.0, 0.0, "S"),
            _pdb_atom("ATOM", 3, "N", "CYS", "A", 10, 7.0, 0.0, 0.0, "N"),
            _pdb_atom("ATOM", 4, "CA", "CYS", "A", 10, 7.0, 1.0, 0.0, "C"),
            _pdb_atom("ATOM", 5, "C", "CYS", "A", 10, 7.0, 2.0, 0.0, "C"),
            _pdb_atom("ATOM", 6, "O", "CYS", "A", 10, 7.0, 3.0, 0.0, "O"),
            _pdb_atom("ATOM", 7, "CB", "CYS", "A", 10, 8.0, 1.0, 0.0, "C"),
            _pdb_atom("ATOM", 2, "NE2", "HIS", "A", 11, 1.8, 0.0, 0.0, "N"),
            _pdb_atom("ATOM", 8, "N", "HIS", "A", 11, 2.8, 0.0, 0.0, "N"),
            _pdb_atom("ATOM", 9, "CA", "HIS", "A", 11, 2.8, 1.0, 0.0, "C"),
            _pdb_atom("ATOM", 10, "C", "HIS", "A", 11, 2.8, 2.0, 0.0, "C"),
            _pdb_atom("ATOM", 11, "O", "HIS", "A", 11, 2.8, 3.0, 0.0, "O"),
            "END\n",
        ]
    )
    path = _write_text(tmp_path / "explicit_metal_core.pdb", pdb)

    info = identify_metal_site_core(read_pdb(path), target="A301", set_bonded="1-144")

    assert [(res["chain"], res["resseq"]) for res in info["auto_core_residues"]] == [("A", 10)]
    assert info["donor_atoms"] == {("A", 10, ""): ["SG"]}


def test_cfmol2_injects_cofactor_atom_types_bonds_and_enables_auto_core(tmp_path: Path) -> None:
    from maple.function.dispatcher.parmfit.utils.MetalAA.recognize import apply_cfmol2_templates

    pdb = "".join(
        [
            _pdb_atom("HETATM", 1, "FE", "FE", "A", 301, 0.0, 0.0, 0.0, "FE"),
            _pdb_atom("HETATM", 2, "N1", "HEM", "A", 500, 1.9, 0.0, 0.0, "N"),
            _pdb_atom("HETATM", 3, "C1", "HEM", "A", 500, 3.1, 0.0, 0.0, "C"),
            "END\n",
        ]
    )
    mol2 = """@<TRIPOS>MOLECULE
HEM
 2 1 0 0 0
SMALL
USER_CHARGES
@<TRIPOS>ATOM
      1 N1          1.9000    0.0000    0.0000 nx        1 HEM       -0.300000
      2 C1          3.1000    0.0000    0.0000 cx        1 HEM        0.300000
@<TRIPOS>BOND
     1    1    2 ar
"""
    pdb_path = _write_text(tmp_path / "fe_hem.pdb", pdb)
    mol2_path = _write_text(tmp_path / "hem.mol2", mol2)
    structure = read_pdb(pdb_path)

    templates = apply_cfmol2_templates(structure, [mol2_path])
    selection = identify_metal_site_core(structure, target="A301")

    hem = next(residue for residue in structure["residues"] if residue["resname"] == "HEM")
    assert templates[0].residue_key == ("A", 500, "")
    assert {atom["name"]: atom["atom_type"] for atom in hem["atoms"]} == {"N1": "nx", "C1": "cx"}
    assert tuple(sorted((2, 3))) not in structure["explicit_pairs"]
    assert hem["_cfmol2_bond_name_pairs"] == {("C1", "N1")}
    assert selection["auto_core_residues"][0]["resname"] == "HEM"
    assert selection["donor_atoms"][("A", 500, "")] == ["N1"]


def test_cfmol2_preserves_name_bonds_when_pdb_serials_are_duplicated(tmp_path: Path) -> None:
    from maple.function.dispatcher.parmfit.utils.MetalAA.recognize import apply_cfmol2_templates

    pdb = "".join(
        [
            _pdb_atom("HETATM", 0, "HN31", "MNS", "A", 862, 0.0, 0.0, 0.0, "H"),
            _pdb_atom("HETATM", 0, "HM23", "MNS", "A", 862, 4.0, 0.0, 0.0, "H"),
            _pdb_atom("HETATM", 2049, "N3S", "MNS", "A", 862, 0.9, 0.0, 0.0, "N"),
            "END\n",
        ]
    )
    mol2 = """@<TRIPOS>MOLECULE
MNS
 3 1 0 0 0
SMALL
USER_CHARGES
@<TRIPOS>ATOM
      1 HN31        0.0000    0.0000    0.0000 hn        1 MNS        0.300000
      2 HM23        4.0000    0.0000    0.0000 h1        1 MNS        0.100000
      3 N3S         0.9000    0.0000    0.0000 n2        1 MNS       -0.400000
@<TRIPOS>BOND
     1    1    3 1
"""
    structure = read_pdb(_write_text(tmp_path / "mns.pdb", pdb))
    apply_cfmol2_templates(structure, [_write_text(tmp_path / "mns.mol2", mol2)])

    residue = structure["residues"][0]
    assert residue["_cfmol2_bond_name_pairs"] == {("HN31", "N3S")}


def test_cfmol2_templates_bind_ambiguous_residues_in_structure_order(tmp_path: Path) -> None:
    from maple.function.dispatcher.parmfit.utils.MetalAA.recognize import apply_cfmol2_templates

    pdb = "".join(
        [
            _pdb_atom("HETATM", 1, "N1", "LIG", "A", 114, 0.0, 0.0, 0.0, "N"),
            _pdb_atom("HETATM", 2, "C1", "LIG", "A", 114, 1.2, 0.0, 0.0, "C"),
            _pdb_atom("HETATM", 3, "N1", "LIG", "A", 115, 3.0, 0.0, 0.0, "N"),
            _pdb_atom("HETATM", 4, "C1", "LIG", "A", 115, 4.2, 0.0, 0.0, "C"),
            "END\n",
        ]
    )
    mol2 = """@<TRIPOS>MOLECULE
LIG
 2 1 0 0 0
SMALL
USER_CHARGES
@<TRIPOS>ATOM
      1 N1          0.0000    0.0000    0.0000 n1        1 LIG       -0.300000
      2 C1          1.2000    0.0000    0.0000 c1        1 LIG        0.300000
@<TRIPOS>BOND
     1    1    2 1
"""
    structure = read_pdb(_write_text(tmp_path / "two_lig.pdb", pdb))
    first = _write_text(tmp_path / "first.mol2", mol2)
    second = _write_text(tmp_path / "second.mol2", mol2.replace("n1", "n2").replace("c1", "c2"))

    templates = apply_cfmol2_templates(structure, [first, second])

    assert [template.residue_key for template in templates] == [("A", 114, ""), ("A", 115, "")]
    first_residue = next(residue for residue in structure["residues"] if residue["resseq"] == 114)
    second_residue = next(residue for residue in structure["residues"] if residue["resseq"] == 115)
    assert {atom["name"]: atom["atom_type"] for atom in first_residue["atoms"]} == {"N1": "n1", "C1": "c1"}
    assert {atom["name"]: atom["atom_type"] for atom in second_residue["atoms"]} == {"N1": "n2", "C1": "c2"}


def test_extract_metal_cluster_excludes_nearby_waters(tmp_path: Path) -> None:
    pdb = "".join(
        [
            _pdb_atom("HETATM", 1, "ZN", "ZN", "A", 301, 0.0, 0.0, 0.0, "ZN"),
            _pdb_atom("ATOM", 2, "N", "CYS", "A", 10, -3.0, 2.8, 0.0, "N"),
            _pdb_atom("ATOM", 3, "CA", "CYS", "A", 10, -2.0, 1.8, 0.0, "C"),
            _pdb_atom("ATOM", 4, "C", "CYS", "A", 10, -1.0, 2.3, 0.0, "C"),
            _pdb_atom("ATOM", 5, "O", "CYS", "A", 10, -0.2, 3.1, 0.0, "O"),
            _pdb_atom("ATOM", 6, "CB", "CYS", "A", 10, -2.0, 0.4, 0.0, "C"),
            _pdb_atom("ATOM", 7, "SG", "CYS", "A", 10, -0.7, 0.2, 0.0, "S"),
            _pdb_atom("HETATM", 8, "O", "HOH", "A", 401, 2.4, 0.0, 0.0, "O"),
            _pdb_atom("HETATM", 9, "C1", "LIG", "B", 501, 0.0, 0.0, 3.2, "C"),
            _pdb_atom("HETATM", 10, "C2", "LIG", "B", 501, 1.2, 0.0, 3.2, "C"),
            "END\n",
        ]
    )
    structure = read_pdb(_write_text(tmp_path / "metal_cluster_nowater.pdb", pdb))

    cluster = extract_metal_cluster(
        structure,
        target="A301",
        cluster_cutoff=4.0,
        donor_cutoff=3.0,
    )

    environment = {(residue["chain"], residue["resseq"], residue["resname"]) for residue in cluster["environment_residues"]}
    assert ("A", 401, "HOH") not in environment
    assert ("B", 501, "LIG") in environment


def test_build_bond_angle_terms_accepts_site_model_directly() -> None:
    bond_terms, angle_terms = model_module.build_bond_angle_terms(_make_simple_site_model_nh_case())

    assert bond_terms
    assert angle_terms


def test_build_bond_angle_terms_uses_authoritative_bond_pairs() -> None:
    model = _make_simple_site_model_nh_case()

    bond_terms, angle_terms = model_module.build_bond_angle_terms(model, bond_pairs=[(1, 2), (2, 3)])

    assert {term.atoms for term in bond_terms} == {(1, 2), (2, 3)}
    assert [term.atoms for term in angle_terms] == [(1, 2, 3)]


def test_canonical_structure_and_metal_helpers_cover_previous_geom_uses(tmp_path: Path) -> None:
    pdb = "".join(
        [
            _pdb_atom("HETATM", 1, "ZN", "ZN", "A", 301, 0.0, 0.0, 0.0, "ZN"),
            _pdb_atom("ATOM", 2, "N", "CYS", "A", 10, -3.0, 2.8, 0.0, "N"),
            _pdb_atom("ATOM", 3, "CA", "CYS", "A", 10, -2.0, 1.8, 0.0, "C"),
            _pdb_atom("ATOM", 4, "C", "CYS", "A", 10, -1.0, 2.3, 0.0, "C"),
            _pdb_atom("ATOM", 5, "O", "CYS", "A", 10, -0.2, 3.1, 0.0, "O"),
            _pdb_atom("ATOM", 6, "CB", "CYS", "A", 10, -2.0, 0.4, 0.0, "C"),
            _pdb_atom("ATOM", 7, "SG", "CYS", "A", 10, -0.7, 0.2, 0.0, "S"),
            "END\n",
        ]
    )
    pdb_path = _write_text(tmp_path / "geom_facade.pdb", pdb)

    structure = read_pdb(pdb_path)
    info = identify_metal_site_core(structure, target="A301")
    site_model = build_metal_site_model(structure, info["auto_core_residues"], donor_atoms=info["donor_atoms"])

    assert callable(context_module.find_unique_residue)
    assert callable(model_module.copy_structure_subset)
    assert info["auto_core_residues"][0]["resname"] == "CYS"
    assert site_model["name"] == "site_model"


def test_metalaa_package_is_the_canonical_metal_helper_source() -> None:
    assert identify_metal_site_core is MetalAA.identify_metal_site_core
    assert extract_metal_cluster is MetalAA.extract_metal_cluster
    assert build_metal_large_model is MetalAA.build_metal_large_model
    assert build_metal_site_model is MetalAA.build_metal_site_model
    assert not hasattr(MetalAA, "build_metal_fc_model")


def test_build_metal_site_model_is_deployment_model_without_caps(tmp_path: Path) -> None:
    pdb = "".join(
        [
            _pdb_atom("ATOM", 1, "N", "HIS", "A", 10, 0.0, 0.0, 0.0, "N"),
            _pdb_atom("ATOM", 2, "CA", "HIS", "A", 10, 1.2, 0.0, 0.0, "C"),
            _pdb_atom("ATOM", 3, "C", "HIS", "A", 10, 2.4, 0.0, 0.0, "C"),
            _pdb_atom("ATOM", 4, "O", "HIS", "A", 10, 3.4, 0.2, 0.0, "O"),
            _pdb_atom("ATOM", 5, "CB", "HIS", "A", 10, 1.2, -1.2, 0.0, "C"),
            _pdb_atom("ATOM", 6, "CG", "HIS", "A", 10, 0.4, -2.2, 0.0, "C"),
            _pdb_atom("HETATM", 7, "ZN", "ZN", "A", 301, 1.2, -3.0, 0.0, "ZN"),
            _pdb_atom("ATOM", 8, "N", "SER", "A", 11, 3.7, 0.0, 0.0, "N"),
            _pdb_atom("ATOM", 9, "CA", "SER", "A", 11, 4.9, 0.0, 0.0, "C"),
            _pdb_atom("ATOM", 10, "C", "SER", "A", 11, 6.1, 0.0, 0.0, "C"),
            _pdb_atom("ATOM", 11, "O", "SER", "A", 11, 7.1, 0.0, 0.0, "O"),
            "END\n",
        ]
    )
    pdb_path = _write_text(tmp_path / "shared_basis.pdb", pdb)
    structure = read_pdb(pdb_path)
    selected = [residue for residue in structure["residues"] if residue["resseq"] in {10, 301}]

    site_model = build_metal_site_model(structure, selected)

    assert [residue["resname"] for residue in site_model["residues"]] == ["HIS", "ZN"]
    assert site_model["name"] == "site_model"


def test_optimize_model_geometry_is_the_shared_model_optimizer(monkeypatch) -> None:
    source_atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    source_atoms.calc = ZeroHessianCalculator()
    source_atoms.f_max_th = 0.11
    source_atoms.f_rms_th = 0.12
    source_atoms.dp_max_th = 0.13
    source_atoms.dp_rms_th = 0.14
    observed: dict[str, object] = {}

    def fake_optimize_atoms_geometry(atoms, *, output, max_iter, max_step, failure_message=None):
        del failure_message
        observed["output"] = output
        observed["max_iter"] = max_iter
        observed["max_step"] = max_step
        observed["thresholds"] = (
            atoms.f_max_th,
            atoms.f_rms_th,
            atoms.dp_max_th,
            atoms.dp_rms_th,
        )
        atoms.positions += np.array([0.25, 0.0, -0.25])
        return atoms

    monkeypatch.setattr(runtime_module, "optimize_atoms_geometry", fake_optimize_atoms_geometry)

    optimized = runtime_module.optimize_model_geometry(
        _make_simple_site_model_nh_case(),
        output="shared_opt.out",
        source_atoms=source_atoms,
        max_iter=12,
        max_step=0.08,
    )

    assert observed["output"] == "shared_opt.out"
    assert observed["max_iter"] == 12
    assert observed["max_step"] == pytest.approx(0.08)
    assert observed["thresholds"] == pytest.approx((0.11, 0.12, 0.13, 0.14))
    first_atom = optimized["residues"][0]["atoms"][0]
    assert np.allclose(first_atom["xyz"], np.array((0.25, 0.0, -0.25)))


def test_project_resp_charges_maps_deployment_atoms_by_model_order() -> None:
    site_model = _make_simple_site_model_nh_case()
    charged_large_model = deepcopy(site_model)
    for residue in charged_large_model["residues"]:
        for atom in residue["atoms"]:
            atom["charge"] = float(atom["serial"]) / 100.0

    projected, warnings = project_resp_charges_onto_site_model(site_model, charged_large_model)

    assert warnings == []
    for residue in projected["residues"]:
        for atom in residue["atoms"]:
            assert atom["charge"] == pytest.approx(float(atom["serial"]) / 100.0)


def test_project_resp_charges_preserves_order_when_serials_are_duplicated() -> None:
    site_model = {
        "name": "site_model",
        "target_key": ("A", 301, ""),
        "core_keys": [("A", 301, ""), ("A", 862, "")],
        "residues": [
            make_residue("A", 301, "", "ZN", [make_atom(1, "ZN", "ZN", np.zeros(3))], kind="ion"),
            make_residue(
                "A",
                862,
                "",
                "MNS",
                [
                    make_atom(0, "H1", "H", np.array((0.0, 0.0, 0.0))),
                    make_atom(0, "H2", "H", np.array((1.0, 0.0, 0.0))),
                    make_atom(0, "H3", "H", np.array((2.0, 0.0, 0.0))),
                ],
                kind="cofactor",
            ),
        ],
    }
    charged_large_model = deepcopy(site_model)
    charges = {
        ("ZN",): [0.2],
        ("H1", "H2", "H3"): [0.11, 0.22, 0.33],
    }
    for residue in charged_large_model["residues"]:
        atom_names = tuple(atom["name"] for atom in residue["atoms"])
        for atom, charge in zip(residue["atoms"], charges[atom_names], strict=True):
            atom["charge"] = charge

    projected, warnings = project_resp_charges_onto_site_model(site_model, charged_large_model)

    assert warnings == []
    ligand_charges = [atom["charge"] for atom in projected["residues"][1]["atoms"]]
    assert ligand_charges == pytest.approx([0.11, 0.22, 0.33])
    assert sum(atom["charge"] for residue in projected["residues"] for atom in residue["atoms"]) == pytest.approx(0.86)


def test_project_resp_charges_raises_for_missing_deployment_atom() -> None:
    site_model = _make_simple_site_model_nh_case()
    charged_large_model = deepcopy(site_model)
    charged_large_model["residues"][1]["atoms"] = charged_large_model["residues"][1]["atoms"][:-1]
    for residue in charged_large_model["residues"]:
        for atom in residue["atoms"]:
            atom["charge"] = 0.0

    with pytest.raises(ValueError, match="atom count mismatch"):
        project_resp_charges_onto_site_model(site_model, charged_large_model)


def test_mseminario_terms_remap_by_model_order_when_serials_are_duplicated() -> None:
    from maple.function.dispatcher.parmfit.utils.readparm import Angle

    zn = make_residue("A", 301, "", "ZN", [make_atom(1, "ZN", "ZN", np.zeros(3))], kind="ion")
    ligand = make_residue(
        "A",
        862,
        "",
        "MNS",
        [
            make_atom(0, "HN31", "H", np.array((1.0, 0.0, 0.0))),
            make_atom(0, "HM23", "H", np.array((2.0, 0.0, 0.0))),
            make_atom(10, "N3S", "N", np.array((0.0, 1.0, 0.0))),
        ],
        kind="cofactor",
    )
    large_model = model_module.rebuild_model_index(
        {
            "name": "large_model",
            "target_key": ("A", 301, ""),
            "residues": [zn, ligand],
            "explicit_pairs": set(),
            "_pair_cache": {},
        }
    )
    site_model = deepcopy(large_model)
    angle = Angle(
        atoms=(1, 4, 2),
        atom_types=("ZN", "n2", "hn"),
        kTheta=12.0,
        thetaEq=np.deg2rad(109.5),
    )

    _mapped_bonds, mapped_angles = metal_workflow_module._remap_terms_to_site_model(large_model, site_model, [], [angle])

    assert mapped_angles[0].atoms == (1, 4, 2)


def test_metal_bonded_export_selects_seminario_method(monkeypatch, tmp_path: Path) -> None:
    metal = make_residue(
        "A",
        301,
        "",
        "ZN",
        [make_atom(1, "ZN", "ZN", np.array((0.0, 0.0, 0.0)))],
        kind="ion",
    )
    donor = make_residue(
        "A",
        10,
        "",
        "CYS",
        [make_atom(2, "SG", "S", np.array((2.0, 0.0, 0.0)))],
        kind="protein",
    )
    model = {
        "residues": [metal, donor],
        "target_key": get_resid_key(metal),
        "donor_atoms": {get_resid_key(donor): ["SG"]},
    }
    bundle = SimpleNamespace(
        large_model=deepcopy(model),
        site_model=deepcopy(model),
        large_charge=2,
        large_mult=1,
    )
    source_atoms = Atoms("ZnS", positions=[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    source_atoms.calc = ZeroHessianCalculator()
    calls: list[str] = []

    monkeypatch.setattr(
        metal_workflow_module,
        "apply_seminario",
        lambda atoms, hessian, bonds, angles: calls.append("seminario"),
    )
    monkeypatch.setattr(
        metal_workflow_module,
        "apply_mseminario",
        lambda atoms, hessian, bonds, angles: calls.append("mseminario"),
    )
    monkeypatch.setattr(metal_workflow_module, "write_site_frcmod", lambda *args, **kwargs: str(tmp_path / "metal.frcmod"))

    for bonded_method in ("seminario", "mseminario"):
        metal_workflow_module._export_metal_bonded_frcmod(
            source_atoms=source_atoms,
            bundle=bundle,
            resp_problem=metal_workflow_module.MetalRespProblem(bond_pairs=[(1, 2)], charge_groups=[]),
            artifacts=SimpleNamespace(files={"frcmod": str(tmp_path / "metal.frcmod")}),
            site_typing=SimpleNamespace(),
            stage_timings=[],
            bonded_method=bonded_method,
        )

    assert calls == ["seminario", "mseminario"]


def test_large_resp_charge_groups_do_not_depend_on_duplicate_serials() -> None:
    zn = make_residue("A", 301, "", "ZN", [make_atom(1, "ZN", "ZN", np.zeros(3))], kind="ion")
    ligand = make_residue(
        "A",
        862,
        "",
        "MNS",
        [
            make_atom(0, "HN31", "H", np.array((1.0, 0.0, 0.0))),
            make_atom(0, "HM23", "H", np.array((2.0, 0.0, 0.0))),
            make_atom(10, "N3S", "N", np.array((0.0, 1.0, 0.0))),
        ],
        kind="cofactor",
    )
    ligand["formal_charge"] = 0
    large_model = model_module.rebuild_model_index(
        {
            "name": "large_model",
            "target_key": ("A", 301, ""),
            "residues": [zn, ligand],
            "explicit_pairs": set(),
            "_pair_cache": {},
        }
    )
    core = SimpleNamespace(
        final_core_residues=[zn],
        target_key=("A", 301, ""),
        metal_atom=zn["atoms"][0],
        optimized_donor_atoms={},
    )

    problem = metal_workflow_module._build_large_resp_problem(SimpleNamespace(large_model=large_model), core)

    assert problem.charge_groups == [([2, 3, 4], 0.0)]


def test_large_resp_bond_graph_uses_cfmol2_name_pairs_with_duplicate_serials() -> None:
    zn = make_residue("A", 301, "", "ZN", [make_atom(1, "ZN", "ZN", np.zeros(3))], kind="ion")
    ligand = make_residue(
        "A",
        862,
        "",
        "MNS",
        [
            make_atom(0, "HN31", "H", np.array((0.0, 0.0, 0.0))),
            make_atom(0, "HM23", "H", np.array((4.0, 0.0, 0.0))),
            make_atom(2049, "N3S", "N", np.array((0.9, 0.0, 0.0))),
        ],
        kind="cofactor",
    )
    ligand["_cfmol2_path"] = "mns.mol2"
    ligand["_cfmol2_bond_name_pairs"] = {("HN31", "N3S")}
    large_model = model_module.rebuild_model_index(
        {
            "name": "large_model",
            "target_key": ("A", 301, ""),
            "residues": [zn, ligand],
            "explicit_pairs": {tuple(sorted((0, 2049)))},
            "_pair_cache": {},
        }
    )
    core = SimpleNamespace(
        final_core_residues=[zn, ligand],
        target_key=("A", 301, ""),
        metal_atom=zn["atoms"][0],
        optimized_donor_atoms={("A", 862, ""): ["N3S"]},
    )

    problem = metal_workflow_module._build_large_resp_problem(SimpleNamespace(large_model=large_model), core)

    assert (2, 4) in problem.bond_pairs
    assert (1, 4) in problem.bond_pairs
    assert (3, 4) not in problem.bond_pairs


def test_residue_mol2_atom_type_overrides_do_not_depend_on_duplicate_serials(tmp_path: Path) -> None:
    ligand = make_residue(
        "A",
        862,
        "",
        "MNS",
        [
            make_atom(0, "HN31", "H", np.array((1.0, 0.0, 0.0))),
            make_atom(0, "HM23", "H", np.array((2.0, 0.0, 0.0))),
            make_atom(10, "N3S", "N", np.array((0.0, 1.0, 0.0))),
        ],
        kind="cofactor",
    )
    for atom, charge in zip(ligand["atoms"], [0.1, 0.2, -0.3], strict=True):
        atom["charge"] = charge
    site_model = model_module.rebuild_model_index(
        {
            "name": "site_model",
            "target_key": ("A", 301, ""),
            "residues": [ligand],
            "explicit_pairs": set(),
            "_pair_cache": {},
        }
    )
    typing = metal_export_module.MetalSiteTyping(
        atom_type_rows=[],
        mol2_atom_types={1: "hA", 2: "hB", 3: "nX"},
        atom_type_overrides={},
        old_type_by_index={},
        renamed_atom_indices=set(),
        ion_frcmods=[],
        residue_names={("A", 862, ""): "MNS1"},
    )
    artifacts = _make_metal_export_artifacts(tmp_path)

    mol2_files = metal_export_module._write_residue_mol2_files(artifacts, site_model=site_model, typing=typing)

    mol2_text = Path(mol2_files["MNS1"]).read_text(encoding="utf-8")
    atom_types = {}
    for line in mol2_text.splitlines():
        parts = line.split()
        if len(parts) >= 9 and parts[0].isdigit():
            atom_types[parts[1]] = parts[5]
    assert atom_types["HN31"] == "hA"
    assert atom_types["HM23"] == "hB"
    assert atom_types["N3S"] == "nX"


def test_typed_cofactor_bond_graph_uses_cfmol2_name_pairs_with_duplicate_serials(tmp_path: Path) -> None:
    ligand = make_residue(
        "A",
        862,
        "",
        "MNS",
        [
            make_atom(0, "HN31", "H", np.array((0.0, 0.0, 0.0))),
            make_atom(0, "HM23", "H", np.array((4.0, 0.0, 0.0))),
            make_atom(2049, "N3S", "N", np.array((0.9, 0.0, 0.0))),
        ],
        kind="cofactor",
    )
    ligand["_cfmol2_path"] = "mns.mol2"
    ligand["_cfmol2_bond_name_pairs"] = {("HN31", "N3S")}
    for atom, atom_type, charge in zip(ligand["atoms"], ["hn", "h1", "Y6"], [0.3, 0.1, -0.4], strict=True):
        atom["atom_type"] = atom_type
        atom["charge"] = charge
    protein = make_residue(
        "B",
        10,
        "",
        "CYS",
        [make_atom(500, "CB", "C", np.array((4.7, 0.0, 0.0)))],
        kind="protein",
    )
    site_model = model_module.rebuild_model_index(
        {
            "name": "site_model",
            "target_key": ("A", 301, ""),
            "residues": [ligand, protein],
            "explicit_pairs": {tuple(sorted((0, 2049)))},
            "_pair_cache": {},
        }
    )
    typing = metal_export_module.MetalSiteTyping(
        atom_type_rows=[],
        mol2_atom_types={1: "hn", 2: "h1", 3: "Y6"},
        atom_type_overrides={3: "Y6"},
        old_type_by_index={1: "hn", 2: "h1", 3: "n2"},
        renamed_atom_indices={3},
        ion_frcmods=[],
        residue_names={("A", 862, ""): "MS1"},
    )
    artifacts = _make_metal_export_artifacts(tmp_path)

    export_pairs = metal_export_module._export_bond_pairs(site_model)
    mol2_files = metal_export_module._write_residue_mol2_files(artifacts, site_model=site_model, typing=typing)

    assert export_pairs == [(1, 3)]
    assert not any(set(pair) == {1, 2} for pair in export_pairs)
    assert not any(set(pair) == {2, 3} for pair in export_pairs)
    assert all("Y6" not in metal_export_module._resolved_types(dihedral, typing) for dihedral in metal_export_module._enumerate_dihedrals_from_pairs(export_pairs))
    mol2_text = Path(mol2_files["MS1"]).read_text(encoding="utf-8")
    assert "\n     1    1    3 1\n" in mol2_text
    assert "    1    1    2 " not in mol2_text


def test_metal_site_typing_renames_only_metal_and_direct_donor_atoms() -> None:
    site_model = _make_simple_site_model_nh_case()
    site_model["residues"][0]["formal_charge"] = 2
    typing = metal_export_module._build_site_typing(site_model, watm="opc", ionm="12_6")
    flattened = model_module.flatten_model_atoms(site_model)

    renamed_by_residue: dict[tuple[str, int, str], set[str]] = {}
    for atom_index in typing.renamed_atom_indices:
        residue, atom = flattened[atom_index - 1]
        renamed_by_residue.setdefault(get_resid_key(residue), set()).add(atom["name"])

    assert renamed_by_residue[("A", 301, "")] == {"ZN"}
    assert renamed_by_residue[("A", 10, "")] == {"O"}
    assert renamed_by_residue[("A", 11, "")] == {"OG"}
    assert "CA" not in renamed_by_residue[("A", 10, "")]
    assert "CB" not in renamed_by_residue[("A", 11, "")]


def test_metal_site_typing_resolves_numbered_histidine_hydrogens_without_element_fallback() -> None:
    zn = make_residue("A", 301, "", "ZN", [make_atom(1, "ZN", "ZN", np.array((0.0, 0.0, 0.0)))], kind="ion")
    his = make_residue(
        "A",
        272,
        "",
        "HIS",
        [
            make_atom(2, "N", "N", np.array((0.0, 0.0, 0.0))),
            make_atom(3, "CA", "C", np.array((1.2, 0.0, 0.0))),
            make_atom(4, "C", "C", np.array((2.4, 0.0, 0.0))),
            make_atom(5, "O", "O", np.array((3.4, 0.2, 0.0))),
            make_atom(6, "CB", "C", np.array((1.2, -1.2, 0.0))),
            make_atom(7, "CG", "C", np.array((0.4, -2.2, 0.0))),
            make_atom(8, "CD2", "C", np.array((1.0, -3.4, 0.0))),
            make_atom(9, "ND1", "N", np.array((-0.9, -2.2, 0.0))),
            make_atom(10, "CE1", "C", np.array((-1.3, -3.4, 0.0))),
            make_atom(11, "NE2", "N", np.array((-0.1, -4.0, 0.0))),
            make_atom(12, "H01", "H", np.array((1.8, 0.8, 0.0))),
            make_atom(13, "H02", "H", np.array((2.0, -1.5, 0.8))),
            make_atom(14, "H03", "H", np.array((1.9, -1.5, -0.8))),
            make_atom(15, "H04", "H", np.array((1.9, -3.9, 0.0))),
            make_atom(16, "H05", "H", np.array((-0.5, 0.6, 0.0))),
            make_atom(17, "H06", "H", np.array((-1.7, -1.7, 0.0))),
            make_atom(18, "H09", "H", np.array((-2.2, -3.9, 0.0))),
        ],
        kind="protein",
    )
    his["_prev_peptide_key"] = ("A", 271, "")
    his["_next_peptide_key"] = ("A", 273, "")
    site_model = {
        "name": "site_model",
        "target_key": ("A", 301, ""),
        "residues": [zn, his],
        "donor_atoms": {("A", 272, ""): ["ND1"]},
    }

    typing = metal_export_module._build_site_typing(site_model, watm="opc", ionm="12_6")
    flattened = model_module.flatten_model_atoms(site_model)
    type_by_name = {
        atom["name"]: typing.old_type_by_index[index]
        for index, (residue, atom) in enumerate(flattened, start=1)
        if get_resid_key(residue) == ("A", 272, "")
    }

    assert type_by_name["H01"] != "h"
    assert type_by_name["H05"] != "h"
    assert type_by_name["H06"] != "h"
    assert all(row.old_type != "h" for row in typing.atom_type_rows)


def test_build_ace_cap_prefers_real_previous_residue_geometry() -> None:
    prev_residue = make_residue(
        "A",
        9,
        "",
        "ALA",
        [
            make_atom(1, "N", "N", np.array((-0.8, 1.1, 0.0))),
            make_atom(2, "CA", "C", np.array((0.2, 0.0, 0.0))),
            make_atom(3, "C", "C", np.array((1.4, 0.0, 0.0))),
            make_atom(4, "O", "O", np.array((2.4, 0.2, 0.0))),
            make_atom(5, "CB", "C", np.array((0.1, -1.1, 0.7))),
            make_atom(6, "HA", "H", np.array((0.0, -0.4, -1.0))),
        ],
        kind="protein",
    )
    target = make_residue(
        "A",
        10,
        "",
        "CYS",
        [
            make_atom(7, "N", "N", np.array((2.0, 0.0, 0.0))),
            make_atom(8, "CA", "C", np.array((3.2, 0.0, 0.0))),
            make_atom(9, "C", "C", np.array((4.4, 0.0, 0.0))),
            make_atom(10, "O", "O", np.array((5.4, 0.2, 0.0))),
        ],
        kind="protein",
    )

    ace_residue, _ = build_ace_cap(target, 100, prev_residue=prev_residue)

    assert np.allclose(get_atom_xyz(search_atom(ace_residue, "CMA")), get_atom_xyz(search_atom(prev_residue, "CA")))
    assert np.allclose(get_atom_xyz(search_atom(ace_residue, "CAC")), get_atom_xyz(search_atom(prev_residue, "C")))
    assert np.allclose(get_atom_xyz(search_atom(ace_residue, "OAC")), get_atom_xyz(search_atom(prev_residue, "O")))
    assert {atom["name"] for atom in ace_residue["atoms"]} >= {"H1A", "H2A", "H3A"}


def test_build_nme_cap_prefers_real_next_residue_geometry() -> None:
    target = make_residue(
        "A",
        10,
        "",
        "CYS",
        [
            make_atom(1, "N", "N", np.array((0.0, 0.0, 0.0))),
            make_atom(2, "CA", "C", np.array((1.2, 0.0, 0.0))),
            make_atom(3, "C", "C", np.array((2.4, 0.0, 0.0))),
            make_atom(4, "O", "O", np.array((3.4, 0.2, 0.0))),
        ],
        kind="protein",
    )
    next_residue = make_residue(
        "A",
        11,
        "",
        "SER",
        [
            make_atom(5, "N", "N", np.array((3.8, 0.1, 0.0))),
            make_atom(6, "H", "H", np.array((3.7, 1.0, 0.0))),
            make_atom(7, "CA", "C", np.array((5.1, 0.0, 0.0))),
            make_atom(8, "C", "C", np.array((6.2, 0.1, 0.0))),
            make_atom(9, "CB", "C", np.array((5.3, -1.2, 0.4))),
            make_atom(10, "HA", "H", np.array((5.1, 0.2, -1.0))),
        ],
        kind="protein",
    )

    nme_residue, _ = build_nme_cap(target, 200, next_residue=next_residue)

    assert np.allclose(get_atom_xyz(search_atom(nme_residue, "NNM")), get_atom_xyz(search_atom(next_residue, "N")))
    assert np.allclose(get_atom_xyz(search_atom(nme_residue, "CNM")), get_atom_xyz(search_atom(next_residue, "CA")))
    assert np.allclose(get_atom_xyz(search_atom(nme_residue, "HNM")), get_atom_xyz(search_atom(next_residue, "H")))
    assert {atom["name"] for atom in nme_residue["atoms"]} >= {"H1M", "H2M", "H3M"}


def test_build_gly_bridge_rewrites_sidechain_to_backbone_only_gly() -> None:
    residue = make_residue(
        "A",
        20,
        "",
        "ALA",
        [
            make_atom(1, "N", "N", np.array((0.0, 0.0, 0.0))),
            make_atom(2, "H", "H", np.array((-0.4, 0.7, 0.0))),
            make_atom(3, "CA", "C", np.array((1.3, 0.0, 0.0))),
            make_atom(4, "HA", "H", np.array((1.5, 0.5, -0.9))),
            make_atom(5, "CB", "C", np.array((1.7, -1.3, 0.6))),
            make_atom(6, "C", "C", np.array((2.5, 1.0, 0.0))),
            make_atom(7, "O", "O", np.array((3.5, 1.0, 0.0))),
        ],
        kind="protein",
    )

    bridge_residue, _ = build_gly_bridge(residue, 300)

    assert bridge_residue["resname"] == "GLY"
    assert bridge_residue["kind"] == "protein"
    assert {atom["name"] for atom in bridge_residue["atoms"]} >= {"N", "H", "CA", "HA2", "HA3", "C", "O"}
    assert "CB" not in {atom["name"] for atom in bridge_residue["atoms"]}


def test_build_gly_bridge_adds_backbone_h_when_source_lacks_explicit_h() -> None:
    prev_residue = make_residue(
        "A",
        19,
        "",
        "GLY",
        [
            make_atom(1, "N", "N", np.array((-2.2, -0.8, 0.0))),
            make_atom(2, "CA", "C", np.array((-1.2, 0.0, 0.0))),
            make_atom(3, "C", "C", np.array((0.0, 0.0, 0.0))),
            make_atom(4, "O", "O", np.array((1.0, 0.2, 0.0))),
        ],
        kind="protein",
    )
    residue = make_residue(
        "A",
        20,
        "",
        "ALA",
        [
            make_atom(5, "N", "N", np.array((1.3, 0.0, 0.0))),
            make_atom(6, "CA", "C", np.array((2.5, 0.0, 0.0))),
            make_atom(7, "CB", "C", np.array((2.9, -1.2, 0.6))),
            make_atom(8, "C", "C", np.array((3.6, 1.0, 0.0))),
            make_atom(9, "O", "O", np.array((4.6, 1.0, 0.0))),
        ],
        kind="protein",
    )

    bridge_residue, _ = build_gly_bridge(residue, 400, prev_residue=prev_residue)

    assert "H" in {atom["name"] for atom in bridge_residue["atoms"]}
    h_atom = search_atom(bridge_residue, "H")
    n_atom = search_atom(bridge_residue, "N")
    assert h_atom is not None and n_atom is not None
    assert np.isclose(np.linalg.norm(get_atom_xyz(h_atom) - get_atom_xyz(n_atom)), 1.01, atol=1.0e-3)


def test_build_metal_large_model_caps_outer_peptide_boundaries(tmp_path: Path) -> None:
    pdb = "".join(
        [
            _pdb_atom("ATOM", 1, "N", "GLY", "A", 9, -3.6, 0.0, 0.0, "N"),
            _pdb_atom("ATOM", 2, "CA", "GLY", "A", 9, -2.6, 0.0, 0.0, "C"),
            _pdb_atom("ATOM", 3, "C", "GLY", "A", 9, -1.3, 0.0, 0.0, "C"),
            _pdb_atom("ATOM", 4, "O", "GLY", "A", 9, -0.3, 0.0, 0.0, "O"),
            _pdb_atom("ATOM", 5, "N", "CYS", "A", 10, 0.0, 0.0, 0.0, "N"),
            _pdb_atom("ATOM", 6, "CA", "CYS", "A", 10, 1.2, 0.0, 0.0, "C"),
            _pdb_atom("ATOM", 7, "C", "CYS", "A", 10, 2.4, 0.0, 0.0, "C"),
            _pdb_atom("ATOM", 8, "O", "CYS", "A", 10, 3.4, 0.0, 0.0, "O"),
            _pdb_atom("ATOM", 9, "CB", "CYS", "A", 10, 1.2, -1.2, 0.0, "C"),
            _pdb_atom("ATOM", 10, "SG", "CYS", "A", 10, 0.8, -2.4, 0.0, "S"),
            _pdb_atom("ATOM", 11, "N", "SER", "A", 11, 3.7, 0.0, 0.0, "N"),
            _pdb_atom("ATOM", 12, "CA", "SER", "A", 11, 4.9, 0.0, 0.0, "C"),
            _pdb_atom("ATOM", 13, "C", "SER", "A", 11, 6.1, 0.0, 0.0, "C"),
            _pdb_atom("ATOM", 14, "O", "SER", "A", 11, 7.1, 0.0, 0.0, "O"),
            _pdb_atom("ATOM", 15, "CB", "SER", "A", 11, 4.9, 1.2, 0.0, "C"),
            _pdb_atom("ATOM", 16, "OG", "SER", "A", 11, 4.7, 2.2, 0.0, "O"),
            _pdb_atom("HETATM", 17, "ZN", "ZN", "A", 301, 2.4, -2.0, 0.0, "ZN"),
            _pdb_atom("ATOM", 18, "N", "GLU", "A", 12, 7.4, 0.0, 0.0, "N"),
            _pdb_atom("ATOM", 19, "CA", "GLU", "A", 12, 8.6, 0.0, 0.0, "C"),
            _pdb_atom("ATOM", 20, "C", "GLU", "A", 12, 9.8, 0.0, 0.0, "C"),
            _pdb_atom("ATOM", 21, "O", "GLU", "A", 12, 10.8, 0.0, 0.0, "O"),
            _pdb_atom("ATOM", 22, "N", "ALA", "A", 13, 13.5, 0.0, 0.0, "N"),
            _pdb_atom("ATOM", 23, "CA", "ALA", "A", 13, 14.7, 0.0, 0.0, "C"),
            _pdb_atom("ATOM", 24, "C", "ALA", "A", 13, 15.9, 0.0, 0.0, "C"),
            _pdb_atom("ATOM", 25, "O", "ALA", "A", 13, 16.9, 0.0, 0.0, "O"),
            "END\n",
        ]
    )
    pdb_path = _write_text(tmp_path / "metal_large_caps.pdb", pdb)
    structure = read_pdb(pdb_path)

    model = build_metal_large_model(structure, target="A301", cluster_cutoff=2.6, donor_cutoff=3.0)

    resnames = [residue["resname"] for residue in model["residues"]]
    assert resnames.count("ACE") == 1
    assert resnames.count("NME") == 1
    assert ("A", 9, "") not in set(model["environment_keys"])
    assert ("A", 12, "") not in set(model["environment_keys"])


def test_build_metal_large_model_excludes_cluster_waters_but_keeps_manual_core_water(tmp_path: Path) -> None:
    pdb = "".join(
        [
            _pdb_atom("HETATM", 1, "ZN", "ZN", "A", 301, 0.0, 0.0, 0.0, "ZN"),
            _pdb_atom("ATOM", 2, "N", "CYS", "A", 10, -3.0, 2.8, 0.0, "N"),
            _pdb_atom("ATOM", 3, "CA", "CYS", "A", 10, -2.0, 1.8, 0.0, "C"),
            _pdb_atom("ATOM", 4, "C", "CYS", "A", 10, -1.0, 2.3, 0.0, "C"),
            _pdb_atom("ATOM", 5, "O", "CYS", "A", 10, -0.2, 3.1, 0.0, "O"),
            _pdb_atom("ATOM", 6, "CB", "CYS", "A", 10, -2.0, 0.4, 0.0, "C"),
            _pdb_atom("ATOM", 7, "SG", "CYS", "A", 10, -0.7, 0.2, 0.0, "S"),
            _pdb_atom("HETATM", 8, "O", "HOH", "A", 401, 2.4, 0.0, 0.0, "O"),
            _pdb_atom("HETATM", 9, "O", "HOH", "A", 402, -2.4, 0.0, 0.0, "O"),
            _pdb_atom("HETATM", 10, "C1", "LIG", "B", 501, 0.0, 0.0, 3.2, "C"),
            _pdb_atom("HETATM", 11, "C2", "LIG", "B", 501, 1.2, 0.0, 3.2, "C"),
            "END\n",
        ]
    )
    structure = read_pdb(_write_text(tmp_path / "metal_large_nowater.pdb", pdb))

    model = build_metal_large_model(
        structure,
        target="A301",
        add_resid=["A402"],
        cluster_cutoff=4.0,
        donor_cutoff=3.0,
    )

    residue_names = {(residue["chain"], residue["resseq"], residue["resname"]) for residue in model["residues"]}
    assert ("A", 402, "HOH") in residue_names
    assert ("A", 401, "HOH") not in residue_names
    assert ("B", 501, "LIG") in residue_names
    assert ("A", 402, "") in set(model["core_keys"])
    assert ("A", 401, "") not in set(model["environment_keys"])
    assert ("B", 501, "") in set(model["environment_keys"])


def test_build_metal_large_model_bridges_nearby_peptide_fragments_with_gly(tmp_path: Path) -> None:
    pdb = "".join(
        [
            _pdb_atom("ATOM", 1, "N", "GLY", "A", 9, -4.0, 0.0, 0.0, "N"),
            _pdb_atom("ATOM", 2, "CA", "GLY", "A", 9, -2.8, 0.0, 0.0, "C"),
            _pdb_atom("ATOM", 3, "C", "GLY", "A", 9, -1.6, 0.0, 0.0, "C"),
            _pdb_atom("ATOM", 4, "O", "GLY", "A", 9, -0.6, 0.2, 0.0, "O"),
            _pdb_atom("ATOM", 5, "N", "HIS", "A", 10, 0.0, 0.0, 0.0, "N"),
            _pdb_atom("ATOM", 6, "CA", "HIS", "A", 10, 1.2, 0.0, 0.0, "C"),
            _pdb_atom("ATOM", 7, "C", "HIS", "A", 10, 2.4, 0.0, 0.0, "C"),
            _pdb_atom("ATOM", 8, "O", "HIS", "A", 10, 3.4, 0.2, 0.0, "O"),
            _pdb_atom("ATOM", 9, "CB", "HIS", "A", 10, 1.2, -1.2, 0.0, "C"),
            _pdb_atom("ATOM", 10, "CG", "HIS", "A", 10, 0.4, -2.2, 0.0, "C"),
            _pdb_atom("ATOM", 11, "N", "ALA", "A", 11, 3.7, 0.0, 0.0, "N"),
            _pdb_atom("ATOM", 12, "CA", "ALA", "A", 11, 4.9, 0.0, 0.0, "C"),
            _pdb_atom("ATOM", 13, "C", "ALA", "A", 11, 6.1, 0.0, 0.0, "C"),
            _pdb_atom("ATOM", 14, "O", "ALA", "A", 11, 7.1, 0.2, 0.0, "O"),
            _pdb_atom("ATOM", 15, "CB", "ALA", "A", 11, 4.9, -1.3, 0.2, "C"),
            _pdb_atom("ATOM", 16, "N", "SER", "A", 12, 7.4, 0.0, 0.0, "N"),
            _pdb_atom("ATOM", 17, "CA", "SER", "A", 12, 8.6, 0.0, 0.0, "C"),
            _pdb_atom("ATOM", 18, "C", "SER", "A", 12, 9.8, 0.0, 0.0, "C"),
            _pdb_atom("ATOM", 19, "O", "SER", "A", 12, 10.8, 0.2, 0.0, "O"),
            _pdb_atom("ATOM", 20, "CB", "SER", "A", 12, 8.7, -1.2, 0.5, "C"),
            _pdb_atom("ATOM", 21, "N", "ASN", "A", 13, 11.1, 0.0, 0.0, "N"),
            _pdb_atom("ATOM", 22, "CA", "ASN", "A", 13, 12.3, 0.0, 0.0, "C"),
            _pdb_atom("ATOM", 23, "C", "ASN", "A", 13, 13.5, 0.0, 0.0, "C"),
            _pdb_atom("ATOM", 24, "O", "ASN", "A", 13, 14.5, 0.2, 0.0, "O"),
            _pdb_atom("ATOM", 25, "CB", "ASN", "A", 13, 12.2, -1.1, 0.7, "C"),
            _pdb_atom("ATOM", 26, "N", "HIS", "A", 14, 14.8, 0.0, 0.0, "N"),
            _pdb_atom("ATOM", 27, "CA", "HIS", "A", 14, 16.0, 0.0, 0.0, "C"),
            _pdb_atom("ATOM", 28, "C", "HIS", "A", 14, 17.2, 0.0, 0.0, "C"),
            _pdb_atom("ATOM", 29, "O", "HIS", "A", 14, 18.2, 0.2, 0.0, "O"),
            _pdb_atom("ATOM", 30, "CB", "HIS", "A", 14, 16.1, -1.2, 0.0, "C"),
            _pdb_atom("ATOM", 31, "CG", "HIS", "A", 14, 15.3, -2.2, 0.0, "C"),
            _pdb_atom("ATOM", 32, "N", "GLY", "A", 15, 18.5, 0.0, 0.0, "N"),
            _pdb_atom("ATOM", 33, "CA", "GLY", "A", 15, 19.7, 0.0, 0.0, "C"),
            _pdb_atom("ATOM", 34, "C", "GLY", "A", 15, 20.9, 0.0, 0.0, "C"),
            _pdb_atom("ATOM", 35, "O", "GLY", "A", 15, 21.9, 0.2, 0.0, "O"),
            _pdb_atom("HETATM", 36, "ZN", "ZN", "A", 301, 8.0, -6.0, 0.0, "ZN"),
            "END\n",
        ]
    )
    pdb_path = _write_text(tmp_path / "metal_large_bridge.pdb", pdb)
    structure = read_pdb(pdb_path)

    model = build_metal_large_model(
        structure,
        target="A301",
        add_resid=["A10", "A14"],
        cluster_cutoff=0.5,
        donor_cutoff=3.0,
    )

    residue_names = {(residue["resseq"], residue["resname"]) for residue in model["residues"]}
    assert (11, "GLY") in residue_names
    assert (12, "GLY") in residue_names
    assert (13, "GLY") in residue_names
    assert sum(1 for residue in model["residues"] if residue["resname"] == "ACE") == 1
    assert sum(1 for residue in model["residues"] if residue["resname"] == "NME") == 1


def test_build_metal_site_model_drops_large_model_bridges_from_deployment_model(tmp_path: Path) -> None:
    pdb_lines: list[str] = []
    serial = 1
    for resseq in range(1, 10):
        pdb_lines.append(_pdb_atom("HETATM", serial, "O", "HOH", "A", resseq, -20.0 - resseq, 0.0, 0.0, "O"))
        serial += 1

    pdb_lines.extend(
        [
            _pdb_atom("ATOM", serial, "N", "HIS", "A", 10, 0.0, 0.0, 0.0, "N"),
            _pdb_atom("ATOM", serial + 1, "CA", "HIS", "A", 10, 1.2, 0.0, 0.0, "C"),
            _pdb_atom("ATOM", serial + 2, "C", "HIS", "A", 10, 2.4, 0.0, 0.0, "C"),
            _pdb_atom("ATOM", serial + 3, "O", "HIS", "A", 10, 3.4, 0.2, 0.0, "O"),
            _pdb_atom("ATOM", serial + 4, "CB", "HIS", "A", 10, 1.2, -1.2, 0.0, "C"),
            _pdb_atom("ATOM", serial + 5, "CG", "HIS", "A", 10, 0.4, -2.2, 0.0, "C"),
            _pdb_atom("ATOM", serial + 6, "N", "ALA", "A", 11, 3.7, 0.0, 0.0, "N"),
            _pdb_atom("ATOM", serial + 7, "CA", "ALA", "A", 11, 4.9, 0.0, 0.0, "C"),
            _pdb_atom("ATOM", serial + 8, "C", "ALA", "A", 11, 6.1, 0.0, 0.0, "C"),
            _pdb_atom("ATOM", serial + 9, "O", "ALA", "A", 11, 7.1, 0.2, 0.0, "O"),
            _pdb_atom("ATOM", serial + 10, "CB", "ALA", "A", 11, 4.9, -1.3, 0.2, "C"),
            _pdb_atom("ATOM", serial + 11, "N", "SER", "A", 12, 7.4, 0.0, 0.0, "N"),
            _pdb_atom("ATOM", serial + 12, "CA", "SER", "A", 12, 8.6, 0.0, 0.0, "C"),
            _pdb_atom("ATOM", serial + 13, "C", "SER", "A", 12, 9.8, 0.0, 0.0, "C"),
            _pdb_atom("ATOM", serial + 14, "O", "SER", "A", 12, 10.8, 0.2, 0.0, "O"),
            _pdb_atom("ATOM", serial + 15, "CB", "SER", "A", 12, 8.7, -1.2, 0.5, "C"),
            _pdb_atom("ATOM", serial + 16, "N", "ASN", "A", 13, 11.1, 0.0, 0.0, "N"),
            _pdb_atom("ATOM", serial + 17, "CA", "ASN", "A", 13, 12.3, 0.0, 0.0, "C"),
            _pdb_atom("ATOM", serial + 18, "C", "ASN", "A", 13, 13.5, 0.0, 0.0, "C"),
            _pdb_atom("ATOM", serial + 19, "O", "ASN", "A", 13, 14.5, 0.2, 0.0, "O"),
            _pdb_atom("ATOM", serial + 20, "CB", "ASN", "A", 13, 12.2, -1.1, 0.7, "C"),
            _pdb_atom("ATOM", serial + 21, "N", "HIS", "A", 14, 14.8, 0.0, 0.0, "N"),
            _pdb_atom("ATOM", serial + 22, "CA", "HIS", "A", 14, 16.0, 0.0, 0.0, "C"),
            _pdb_atom("ATOM", serial + 23, "C", "HIS", "A", 14, 17.2, 0.0, 0.0, "C"),
            _pdb_atom("ATOM", serial + 24, "O", "HIS", "A", 14, 18.2, 0.2, 0.0, "O"),
            _pdb_atom("ATOM", serial + 25, "CB", "HIS", "A", 14, 16.1, -1.2, 0.0, "C"),
            _pdb_atom("ATOM", serial + 26, "CG", "HIS", "A", 14, 15.3, -2.2, 0.0, "C"),
            _pdb_atom("HETATM", serial + 27, "ZN", "ZN", "A", 301, 8.0, -6.0, 0.0, "ZN"),
            "END\n",
        ]
    )
    pdb_path = _write_text(tmp_path / "metal_site_large_model_indices.pdb", "".join(pdb_lines))
    structure = read_pdb(pdb_path)

    large_model = build_metal_large_model(
        structure,
        target="A301",
        add_resid=["A10", "A14"],
        cluster_cutoff=0.5,
        donor_cutoff=3.0,
    )
    core_keys = {("A", 301, ""), ("A", 10, ""), ("A", 14, "")}
    optimized_core = [
        residue
        for residue in large_model["residues"]
        if (residue["chain"], residue["resseq"], residue["icode"]) in core_keys
    ]

    site_model = build_metal_site_model(
        large_model,
        optimized_core,
        donor_atoms=large_model["donor_atoms"],
    )

    residue_names = {(residue["resseq"], residue["resname"]) for residue in site_model["residues"]}
    assert residue_names == {(10, "HIS"), (14, "HIS"), (301, "ZN")}
    assert "GLY" not in {residue["resname"] for residue in site_model["residues"]}


def test_build_metal_site_model_caps_outer_peptide_boundaries(tmp_path: Path) -> None:
    pdb = "".join(
        [
            _pdb_atom("ATOM", 1, "N", "GLY", "A", 9, -3.6, 0.0, 0.0, "N"),
            _pdb_atom("ATOM", 2, "CA", "GLY", "A", 9, -2.6, 0.0, 0.0, "C"),
            _pdb_atom("ATOM", 3, "C", "GLY", "A", 9, -1.3, 0.0, 0.0, "C"),
            _pdb_atom("ATOM", 4, "O", "GLY", "A", 9, -0.3, 0.0, 0.0, "O"),
            _pdb_atom("ATOM", 5, "N", "HIS", "A", 10, 0.0, 0.0, 0.0, "N"),
            _pdb_atom("ATOM", 6, "CA", "HIS", "A", 10, 1.2, 0.0, 0.0, "C"),
            _pdb_atom("ATOM", 7, "C", "HIS", "A", 10, 2.4, 0.0, 0.0, "C"),
            _pdb_atom("ATOM", 8, "O", "HIS", "A", 10, 3.4, 0.2, 0.0, "O"),
            _pdb_atom("ATOM", 9, "CB", "HIS", "A", 10, 1.2, -1.2, 0.0, "C"),
            _pdb_atom("ATOM", 10, "CG", "HIS", "A", 10, 0.4, -2.2, 0.0, "C"),
            _pdb_atom("HETATM", 11, "ZN", "ZN", "A", 301, 1.2, -3.0, 0.0, "ZN"),
            _pdb_atom("ATOM", 12, "N", "SER", "A", 11, 3.7, 0.0, 0.0, "N"),
            _pdb_atom("ATOM", 13, "CA", "SER", "A", 11, 4.9, 0.0, 0.0, "C"),
            _pdb_atom("ATOM", 14, "C", "SER", "A", 11, 6.1, 0.0, 0.0, "C"),
            _pdb_atom("ATOM", 15, "O", "SER", "A", 11, 7.1, 0.0, 0.0, "O"),
            "END\n",
        ]
    )
    pdb_path = _write_text(tmp_path / "metal_site_caps.pdb", pdb)
    structure = read_pdb(pdb_path)

    site_model = build_metal_site_model(
        structure,
        [residue for residue in structure["residues"] if residue["resseq"] in {10, 301}],
    )

    resnames = [residue["resname"] for residue in site_model["residues"]]
    assert "ACE" not in resnames
    assert "NME" not in resnames
    assert "HIS" in resnames
    assert "ZN" in resnames


def test_abinitio_metal_route_writes_models_and_outputs(monkeypatch, tmp_path: Path) -> None:
    pdb_path, atoms, output, output_root = _make_metal_route_inputs(tmp_path)
    base = output.with_suffix("")

    monkeypatch.setattr(metal_workflow_module, "run_resp_pipeline", _fake_metal_large_resp_pipeline(str(output)))
    job = Abinitio(
        output=str(output),
        atoms=atoms,
        params={
            "pdb": pdb_path,
            "target": "A301",
            "add_resid": "A401",
            "cluster_cutoff": 4.0,
            "chgmod": 1,
        },
    )

    job.run()

    assert (output_root / f"{base.name}_metal_large_raw.pdb").is_file()
    assert (output_root / f"{base.name}_metal_large_opt.pdb").is_file()
    assert (output_root / f"{base.name}_metal_site_opt.pdb").is_file()
    assert (output_root / "metalaa" / f"{base.name}_metal_large_resp.gjf").is_file()
    assert (output_root / f"{base.name}_metal_site.mol2").is_file()
    assert (output_root / f"{base.name}_metal.frcmod").is_file()
    assert (output_root / f"{base.name}_metal_tleap.pdb").is_file()
    assert (output_root / f"{base.name}_metal_tleap.in").is_file()
    assert (output_root / "CS1.mol2").is_file()
    assert (output_root / "ZN1.mol2").is_file()

    site_pdb = (output_root / f"{base.name}_metal_site_opt.pdb").read_text(encoding="utf-8")
    site_mol2 = (output_root / f"{base.name}_metal_site.mol2").read_text(encoding="utf-8")
    metal_mol2 = (output_root / "ZN1.mol2").read_text(encoding="utf-8")
    tleap_input = (output_root / f"{base.name}_metal_tleap.in").read_text(encoding="utf-8")
    route_log = output.read_text(encoding="utf-8")

    assert " ACE " not in site_pdb
    assert " NME " not in site_pdb
    assert "M1" in site_mol2
    assert "Y1" in site_mol2
    assert "ZN1" in metal_mol2
    assert "    2.000000" in metal_mol2
    assert "source leaprc.protein.ff14SB" in tleap_input
    assert "source leaprc.gaff2" in tleap_input
    assert "source leaprc.water.tip3p" in tleap_input
    assert "ZN1 = loadmol2 ZN1.mol2" in tleap_input
    assert f"loadamberparams {base.name}_metal.frcmod" in tleap_input
    assert f"mol = loadpdb {base.name}_metal_tleap.pdb" in tleap_input
    assert "bond mol.2.SG mol.1.ZN" in tleap_input
    assert f"savepdb mol {base.name}_metal_tleap_dry.pdb" in tleap_input
    assert f"saveamberparm mol {base.name}_metal_tleap_dry.prmtop {base.name}_metal_tleap_dry.inpcrd" in tleap_input
    assert "solvatebox mol TIP3PBOX 10.0" in tleap_input
    assert "addions mol Na+ 0" in tleap_input
    assert "addions mol Cl- 0" in tleap_input
    assert f"savepdb mol {base.name}_metal_tleap_solvated.pdb" in tleap_input
    assert f"saveamberparm mol {base.name}_metal_tleap_solvated.prmtop {base.name}_metal_tleap_solvated.inpcrd" in tleap_input
    assert "quit" in tleap_input
    assert "PARMFIT ABINITIO RESULT" in route_log
    assert "metal site charge/mult: 2 1" in route_log
    assert "Renamed atom types:" in route_log
    assert "A301:ZN" in route_log and "M1" in route_log
    assert "large RESP/Gaussian ESP" in route_log
    assert "Hessian + mSeminario" in route_log
    assert f"tleap -s -f {base.name}_metal_tleap.in" in route_log


def test_metal_workflow_constrains_non_deployment_residue_charge_groups(monkeypatch, tmp_path: Path) -> None:
    pdb_path, atoms, output, _output_root = _make_metal_route_inputs(tmp_path)
    structure = read_pdb(pdb_path)
    captured: dict[str, object] = {}
    base_fake = _fake_metal_large_resp_pipeline(str(output))

    def fake_run_resp_pipeline(**kwargs):
        charge_groups = kwargs.get("charge_groups")
        captured["charge_groups"] = charge_groups
        flattened = model_module.flatten_model_atoms(kwargs["model"])
        captured["group_residue_names"] = [
            {
                flattened[atom_index - 1][0]["resname"]
                for atom_index in atom_indices
            }
            for atom_indices, _target in (charge_groups or [])
        ]
        return base_fake(**kwargs)

    monkeypatch.setattr(metal_workflow_module, "run_resp_pipeline", fake_run_resp_pipeline)

    metal_workflow_module.run_metal_abinitio(
        output=str(output),
        source_atoms=atoms,
        structure=structure,
        config=parse_metal_abinitio_config(
            {
                "pdb": pdb_path,
                "target": "A301",
                "add_resid": "A401",
                "cluster_cutoff": 4.0,
                "chgmod": 1,
            },
            pdb_path=pdb_path,
            target="A301",
            charge=2,
            mult=1,
            target_residue=structure["residues"][0],
        ),
        log_info=lambda lines: None,
    )

    assert captured["charge_groups"]
    assert any(target == pytest.approx(-1.0) for _indices, target in captured["charge_groups"])
    assert {"GLU"} in captured["group_residue_names"]


def test_metal_workflow_fails_when_deployment_resp_charge_is_not_integer(monkeypatch, tmp_path: Path) -> None:
    pdb_path, atoms, output, _output_root = _make_metal_route_inputs(tmp_path)
    structure = read_pdb(pdb_path)
    base_fake = _fake_metal_large_resp_pipeline(str(output))

    def fake_run_resp_pipeline(**kwargs):
        result = base_fake(**kwargs)
        first_ion = True
        for residue in result.model["residues"]:
            for atom in residue["atoms"]:
                atom["charge"] = 1.25 if first_ion and residue.get("kind") == "ion" else 0.0
                first_ion = False if residue.get("kind") == "ion" else first_ion
        return result

    monkeypatch.setattr(metal_workflow_module, "run_resp_pipeline", fake_run_resp_pipeline)

    with pytest.raises(ValueError, match="Deployment RESP charge"):
        metal_workflow_module.run_metal_abinitio(
            output=str(output),
            source_atoms=atoms,
            structure=structure,
            config=parse_metal_abinitio_config(
                {
                    "pdb": pdb_path,
                    "target": "A301",
                    "add_resid": "A401",
                    "cluster_cutoff": 4.0,
                    "chgmod": 1,
                },
                pdb_path=pdb_path,
                target="A301",
                charge=2,
                mult=1,
                target_residue=structure["residues"][0],
            ),
            log_info=lambda lines: None,
        )


def test_metal_workflow_refreshes_donors_from_optimized_large_model(monkeypatch, tmp_path: Path) -> None:
    pdb_path, atoms, output, output_root = _make_carboxylate_route_inputs(tmp_path)
    input_structure = read_pdb(pdb_path)
    captured_bond_pairs: list[tuple[int, int]] = []

    def fake_optimize_model_geometry(model, **_kwargs):
        optimized = deepcopy(model)
        glu = next(residue for residue in optimized["residues"] if residue["resname"] == "GLU")
        search_atom(glu, "OE1")["xyz"] = np.array((1.80, 0.00, 0.00))
        search_atom(glu, "OE2")["xyz"] = np.array((0.00, 1.80, 0.00))
        refresh_resid(glu)
        return model_module.rebuild_model_index(optimized)

    real_build_bond_angle_terms = model_module.build_bond_angle_terms

    def spy_build_bond_angle_terms(model, source_structure=None, bond_policy="auto", bond_pairs=None):
        if model.get("name") == "large_model":
            captured_bond_pairs[:] = list(bond_pairs or [])
        if bond_pairs is None:
            return real_build_bond_angle_terms(model, source_structure=source_structure, bond_policy=bond_policy)
        return real_build_bond_angle_terms(
            model,
            source_structure=source_structure,
            bond_policy=bond_policy,
            bond_pairs=bond_pairs,
        )

    monkeypatch.setattr(metal_workflow_module, "optimize_model_geometry", fake_optimize_model_geometry)
    monkeypatch.setattr(metal_workflow_module, "run_resp_pipeline", _fake_metal_large_resp_pipeline(str(output)))
    monkeypatch.setattr(metal_workflow_module, "build_bond_angle_terms", spy_build_bond_angle_terms)

    result = metal_workflow_module.run_metal_abinitio(
        output=str(output),
        source_atoms=atoms,
        structure=input_structure,
        config=parse_metal_abinitio_config(
            {
                "pdb": pdb_path,
                "target": "A301",
                "add_resid": "A10",
                "donor_cutoff": 2.5,
            },
            pdb_path=pdb_path,
            target="A301",
            charge=2,
            mult=1,
            target_residue=input_structure["residues"][0],
        ),
        log_info=lambda lines: None,
    )

    flattened = model_module.flatten_model_atoms(result.site_model)
    serial_to_site_xyz = {atom["serial"]: atom["xyz"] for _residue, atom in flattened}
    assert serial_to_site_xyz[9] == pytest.approx(np.array((1.80, 0.00, 0.00)))
    assert serial_to_site_xyz[10] == pytest.approx(np.array((0.00, 1.80, 0.00)))
    assert result.site_model["donor_atoms"][("A", 10, "")] == ["OE1", "OE2"]

    site_mol2 = (output_root / f"{output.with_suffix('').name}_metal_site.mol2").read_text(encoding="utf-8")
    assert "OE1      1.8000    0.0000    0.0000" in site_mol2
    assert "OE2      0.0000    1.8000    0.0000" in site_mol2
    assert "bond mol.2.OE1 mol.1.ZN\n" in result.artifacts.tleap_lines
    assert "bond mol.2.OE2 mol.1.ZN\n" in result.artifacts.tleap_lines

    large_serial_to_index = {
        atom["serial"]: atom_index
        for atom_index, (_residue, atom) in enumerate(model_module.flatten_model_atoms(result.large_model), start=1)
    }
    metal_index = large_serial_to_index[1]
    assert tuple(sorted((metal_index, large_serial_to_index[9]))) in captured_bond_pairs
    assert tuple(sorted((metal_index, large_serial_to_index[10]))) in captured_bond_pairs


def test_metal_workflow_set_bonded_keeps_explicit_donors_after_optimization(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pdb_path, atoms, output, _output_root = _make_carboxylate_route_inputs(tmp_path)
    input_structure = read_pdb(pdb_path)
    captured_bond_pairs: list[tuple[int, int]] = []

    def fake_optimize_model_geometry(model, **_kwargs):
        optimized = deepcopy(model)
        glu = next(residue for residue in optimized["residues"] if residue["resname"] == "GLU")
        search_atom(glu, "OE1")["xyz"] = np.array((1.80, 0.00, 0.00))
        search_atom(glu, "OE2")["xyz"] = np.array((0.00, 1.80, 0.00))
        refresh_resid(glu)
        return model_module.rebuild_model_index(optimized)

    real_build_bond_angle_terms = model_module.build_bond_angle_terms

    def spy_build_bond_angle_terms(model, source_structure=None, bond_policy="auto", bond_pairs=None):
        if model.get("name") == "large_model":
            captured_bond_pairs[:] = list(bond_pairs or [])
        if bond_pairs is None:
            return real_build_bond_angle_terms(model, source_structure=source_structure, bond_policy=bond_policy)
        return real_build_bond_angle_terms(
            model,
            source_structure=source_structure,
            bond_policy=bond_policy,
            bond_pairs=bond_pairs,
        )

    monkeypatch.setattr(metal_workflow_module, "optimize_model_geometry", fake_optimize_model_geometry)
    monkeypatch.setattr(metal_workflow_module, "run_resp_pipeline", _fake_metal_large_resp_pipeline(str(output)))
    monkeypatch.setattr(metal_workflow_module, "build_bond_angle_terms", spy_build_bond_angle_terms)

    result = metal_workflow_module.run_metal_abinitio(
        output=str(output),
        source_atoms=atoms,
        structure=input_structure,
        config=parse_metal_abinitio_config(
            {
                "pdb": pdb_path,
                "target": "A301",
                "set_bonded": "9-1",
                "donor_cutoff": 2.5,
            },
            pdb_path=pdb_path,
            target="A301",
            charge=2,
            mult=1,
            target_residue=input_structure["residues"][0],
        ),
        log_info=lambda lines: None,
    )

    assert result.site_model["donor_atoms"][("A", 10, "")] == ["OE1"]
    assert "bond mol.2.OE1 mol.1.ZN\n" in result.artifacts.tleap_lines
    assert "bond mol.2.OE2 mol.1.ZN\n" not in result.artifacts.tleap_lines

    large_serial_to_index = {
        atom["serial"]: atom_index
        for atom_index, (_residue, atom) in enumerate(model_module.flatten_model_atoms(result.large_model), start=1)
    }
    metal_index = large_serial_to_index[1]
    assert tuple(sorted((metal_index, large_serial_to_index[9]))) in captured_bond_pairs
    assert tuple(sorted((metal_index, large_serial_to_index[10]))) not in captured_bond_pairs


def test_run_metal_abinitio_returns_workflow_result(monkeypatch, tmp_path: Path) -> None:
    pdb_path, atoms, output, _ = _make_metal_route_inputs(tmp_path)
    structure = read_pdb(pdb_path)
    monkeypatch.setattr(metal_workflow_module, "run_resp_pipeline", _fake_metal_large_resp_pipeline(str(output)))
    config = parse_metal_abinitio_config(
        {
            "pdb": pdb_path,
            "target": "A301",
            "add_resid": "A401",
            "cluster_cutoff": 4.0,
            "chgmod": 1,
        },
        pdb_path=pdb_path,
        target="A301",
        charge=2,
        mult=1,
        target_residue=structure["residues"][0],
    )

    result = metal_workflow_module.run_metal_abinitio(
        output=str(output),
        source_atoms=atoms,
        structure=structure,
        config=config,
        log_info=lambda lines: None,
    )

    assert isinstance(result, MetalWorkflowResult)
    assert result.core_info["target"]["resname"] == "ZN"
    assert result.files["site_mol2"].endswith("_metal_site.mol2")
    assert result.files["frcmod"].endswith("_metal.frcmod")
    assert result.files["tleap_input"].endswith("_metal_tleap.in")
    assert result.artifacts.mol2_files["ZN1"].endswith("ZN1.mol2")
    assert result.resp_files["resp2_chg"].endswith("resp2.chg")
    assert isinstance(result.site_model, dict)
    assert {residue["resname"] for residue in result.site_model["residues"]} == {"ZN", "CYS", "HOH"}


def test_abinitio_metal_keeps_large_raw_pdb_if_optimization_fails(monkeypatch, tmp_path: Path) -> None:
    pdb_path, atoms, output, output_root = _make_metal_route_inputs(tmp_path)
    base_name = output.with_suffix("").name

    def fail_optimize(*args, **kwargs):
        raise RuntimeError("opt failed")

    monkeypatch.setattr(metal_workflow_module, "optimize_model_geometry", fail_optimize)
    job = Abinitio(
        output=str(output),
        atoms=atoms,
        params={
            "pdb": pdb_path,
            "target": "A301",
            "add_resid": "A401",
            "cluster_cutoff": 4.0,
            "chgmod": 1,
        },
    )

    with pytest.raises(RuntimeError, match="opt failed"):
        job.run()

    assert (output_root / f"{base_name}_metal_large_raw.pdb").is_file()
    assert not (output_root / f"{base_name}_metal_large_opt.pdb").exists()
    assert not (output_root / f"{base_name}_metal_site_opt.pdb").exists()
    assert not (output_root / f"{base_name}_metal_site.mol2").exists()
    assert not (output_root / f"{base_name}_metal_site.frcmod").exists()


def test_abinitio_metal_keeps_site_files_if_hessian_stage_fails(monkeypatch, tmp_path: Path) -> None:
    pdb_path, atoms, output, output_root = _make_metal_route_inputs(tmp_path)
    base_name = output.with_suffix("").name

    monkeypatch.setattr(metal_workflow_module, "run_resp_pipeline", _fake_metal_large_resp_pipeline(str(output)))
    monkeypatch.setattr(metal_workflow_module, "get_cartesian_hessian", lambda atoms: (_ for _ in ()).throw(RuntimeError("hess failed")))
    job = Abinitio(
        output=str(output),
        atoms=atoms,
        params={
            "pdb": pdb_path,
            "target": "A301",
            "add_resid": "A401",
            "cluster_cutoff": 4.0,
            "chgmod": 1,
        },
    )

    with pytest.raises(RuntimeError, match="hess failed"):
        job.run()

    assert (output_root / f"{base_name}_metal_large_raw.pdb").is_file()
    assert (output_root / f"{base_name}_metal_large_opt.pdb").is_file()
    assert (output_root / f"{base_name}_metal_site_opt.pdb").is_file()
    assert (output_root / f"{base_name}_metal_site.mol2").is_file()
    assert not (output_root / f"{base_name}_metal.frcmod").exists()


def test_abinitio_reads_pdb_once_before_routing(monkeypatch, tmp_path: Path) -> None:
    structure = {
        "path": "dummy.pdb",
        "residues": [make_residue("A", 301, "", "ZN", [make_atom(1, "ZN", "ZN", np.array((0.0, 0.0, 0.0)))], kind="ion")],
        "serial_to_atom": {},
        "serial_to_residue": {},
        "explicit_pairs": set(),
        "_pair_cache": {},
    }
    structure["serial_to_atom"][1] = structure["residues"][0]["atoms"][0]
    structure["serial_to_residue"][1] = structure["residues"][0]
    calls = {"read_pdb": 0, "parse_metal_abinitio_config": 0, "run_metal_abinitio": 0}

    def fake_read_pdb(path):
        calls["read_pdb"] += 1
        return structure

    def fake_parse_metal_abinitio_config(raw, *, pdb_path, target, charge, mult, target_residue, oxy=None):
        calls["parse_metal_abinitio_config"] += 1
        assert raw["pdb"] == str(tmp_path / "single_read.pdb")
        assert pdb_path == str(tmp_path / "single_read.pdb")
        assert target == "A301"
        assert charge == 2
        assert mult == 1
        assert oxy is None
        assert target_residue["resname"] == "ZN"
        return object()

    def fake_run_metal_abinitio(*, output, source_atoms, structure, config, log_info):
        calls["run_metal_abinitio"] += 1
        assert structure is not None
        assert config is not None

    monkeypatch.setattr(abinitio_module, "read_pdb", fake_read_pdb)
    monkeypatch.setattr(abinitio_module, "parse_metal_abinitio_config", fake_parse_metal_abinitio_config)
    monkeypatch.setattr(abinitio_module, "run_metal_abinitio", fake_run_metal_abinitio)

    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    atoms.info["charge"] = 2
    atoms.info["mult"] = 1
    atoms.calc = ZeroHessianCalculator()

    job = Abinitio(
        output=str(tmp_path / "single_read.out"),
        atoms=atoms,
        params={"pdb": str(tmp_path / "single_read.pdb"), "target": "A301"},
    )
    job.run()

    assert calls == {"read_pdb": 1, "parse_metal_abinitio_config": 1, "run_metal_abinitio": 1}


def test_abinitio_metal_route_writes_user_summary_and_returns_result(monkeypatch, tmp_path: Path) -> None:
    pdb = "".join(
        [
            _pdb_atom("HETATM", 1, "ZN", "ZN", "A", 301, 0.0, 0.0, 0.0, "ZN"),
            "END\n",
        ]
    )
    pdb_path = _write_text(tmp_path / "metal_summary.pdb", pdb)
    output = tmp_path / "metal_summary.out"

    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    atoms.info["charge"] = -1
    atoms.info["mult"] = 6
    atoms.info["oxy"] = 3
    atoms.calc = ZeroHessianCalculator()

    target_residue = make_residue("A", 301, "", "ZN", [make_atom(1, "ZN", "ZN", np.array((0.0, 0.0, 0.0)))], kind="ion")
    core_residue = make_residue("A", 10, "", "CYS", [make_atom(2, "SG", "S", np.array((2.0, 0.0, 0.0)))], kind="protein")
    artifacts = SimpleNamespace(
        files={
            "large_raw_pdb": "metal_large_raw.pdb",
            "large_opt_pdb": "metal_large_opt.pdb",
            "site_pdb": "metal_site_opt.pdb",
            "mol2": "metal_site.mol2",
            "tleap_pdb": "metal_tleap.pdb",
            "tleap_input": "metal_tleap.in",
            "frcmod": "metal.frcmod",
        },
        mol2_files={"HM1": "HM1.mol2"},
        cofactor_frcmods=["metal_HEM_1_orig.frcmod"],
    )
    expected_result = SimpleNamespace(
        selection=SimpleNamespace(core_residues=[target_residue, core_residue], warnings=["selection warning"]),
        large_model={"charge": -2, "mult": 6},
        site_model={"warnings": ["site warning"]},
        artifacts=artifacts,
        mseminario_warning="seminario warning",
    )

    def fake_parse_metal_abinitio_config(raw, *, pdb_path, target, charge, mult, target_residue, oxy=None):
        return SimpleNamespace(
            target=target,
            charge=charge,
            mult=mult,
            oxy=oxy,
            cfmol2=["HEM.mol2"],
        )

    def fake_run_metal_abinitio(**kwargs):
        return expected_result

    monkeypatch.setattr(abinitio_module, "parse_metal_abinitio_config", fake_parse_metal_abinitio_config)
    monkeypatch.setattr(abinitio_module, "run_metal_abinitio", fake_run_metal_abinitio)

    result = Abinitio(
        output=str(output),
        atoms=atoms,
        params={"pdb": pdb_path, "target": "A301"},
    ).run()

    text = output.read_text(encoding="utf-8")
    assert result is expected_result
    assert "PARMFIT ABINITIO RESULT" in text
    assert "Route: MetalAA" in text
    assert "Target: A301:ZN" in text
    assert "Charge/mult/oxidation: -1 6 3" in text
    assert "Large model charge/mult: -2 6" in text
    assert "Core residues: A301:ZN, A10:CYS" in text
    assert "mol2 files:  HM1.mol2" in text
    assert "final frcmod: metal.frcmod" in text
    assert "Tleap status:" in text
    assert "input written, not executed" in text
    assert "selection warning" in text
    assert "site warning" in text
    assert "seminario warning" in text


def test_abinitio_routes_protein_target_to_ncaa(monkeypatch, tmp_path: Path) -> None:
    pdb = "".join(
        [
            _pdb_atom("ATOM", 1, "N", "GLY", "A", 1, 0.0, 0.0, 0.0, "N"),
            _pdb_atom("ATOM", 2, "CA", "GLY", "A", 1, 1.4, 0.0, 0.0, "C"),
            _pdb_atom("ATOM", 3, "C", "GLY", "A", 1, 2.4, 1.0, 0.0, "C"),
            _pdb_atom("ATOM", 4, "O", "GLY", "A", 1, 3.4, 1.0, 0.0, "O"),
            _pdb_atom("ATOM", 5, "CB", "GLY", "A", 1, 1.5, -1.2, 0.0, "C"),
            "END\n",
        ]
    )
    pdb_path = _write_text(tmp_path / "non_ion.pdb", pdb)

    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    atoms.info["charge"] = 0
    atoms.info["mult"] = 1
    atoms.calc = ZeroHessianCalculator()

    calls = {"parse": 0, "run": 0}

    def fake_parse_ncaa_abinitio_config(raw, *, pdb_path, target, charge, mult):
        calls["parse"] += 1
        assert raw["pdb"] == pdb_path
        assert target == "A1"
        assert charge == 0
        assert mult == 1
        return object()

    def fake_run_ncaa_abinitio(*, output, source_atoms, structure, target_residue, config, log_info):
        del output, source_atoms, log_info
        calls["run"] += 1
        assert len(structure["residues"]) == 1
        assert target_residue["resname"] == "GLY"
        assert config is not None

    monkeypatch.setattr(abinitio_module, "parse_ncaa_abinitio_config", fake_parse_ncaa_abinitio_config)
    monkeypatch.setattr(abinitio_module, "run_ncaa_abinitio", fake_run_ncaa_abinitio)

    job = Abinitio(
        output=str(tmp_path / "non_ion.out"),
        atoms=atoms,
        params={"pdb": pdb_path, "target": "A1"},
    )

    job.run()
    assert calls == {"parse": 1, "run": 1}


def test_abinitio_ncaa_route_writes_user_summary_and_returns_result(monkeypatch, tmp_path: Path) -> None:
    pdb = "".join(
        [
            _pdb_atom("ATOM", 1, "N", "SER", "A", 1, 0.0, 0.0, 0.0, "N"),
            _pdb_atom("ATOM", 2, "CA", "SER", "A", 1, 1.4, 0.0, 0.0, "C"),
            _pdb_atom("ATOM", 3, "C", "SER", "A", 1, 2.4, 1.0, 0.0, "C"),
            _pdb_atom("ATOM", 4, "O", "SER", "A", 1, 3.4, 1.0, 0.0, "O"),
            _pdb_atom("ATOM", 5, "CB", "SER", "A", 1, 1.5, -1.2, 0.0, "C"),
            "END\n",
        ]
    )
    pdb_path = _write_text(tmp_path / "ncaa_summary.pdb", pdb)
    output = tmp_path / "ncaa_summary.out"

    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    atoms.info["charge"] = 0
    atoms.info["mult"] = 1
    atoms.calc = ZeroHessianCalculator()
    tleap_input = tmp_path / "NAA_tleap.in"
    tleap_input.write_text("quit\n", encoding="utf-8")
    tleap_input.with_suffix(".out").write_text("Errors = 0; Warnings = 2; Notes = 1\n", encoding="utf-8")

    artifacts = SimpleNamespace(
        files={
            "capped_mol2": "NAA_capped.mol2",
            "ac": "NAA.ac",
            "prepin": "NAA.prepin",
            "frcmod": "NAA.frcmod",
            "refined_prepin": "NAA_maple.prepin",
            "refined_frcmod": "NAA_maple.frcmod",
            "tleap_pdb": "NAA_tleap.pdb",
            "tleap_input": str(tleap_input),
            "target_capped_pdb": "target_capped.pdb",
            "alpha_capped_pdb": "alpha_capped.pdb",
            "beta_capped_pdb": "beta_capped.pdb",
        }
    )
    expected_result = SimpleNamespace(
        identity=SimpleNamespace(chirality="L"),
        representative=SimpleNamespace(label="ref"),
        conformers=[
            SimpleNamespace(label="alpha"),
            SimpleNamespace(label="beta"),
        ],
        artifacts=artifacts,
    )

    def fake_parse_ncaa_abinitio_config(raw, *, pdb_path, target, charge, mult):
        return SimpleNamespace(target=target, charge=charge, mult=mult, rn="NAA")

    def fake_run_ncaa_abinitio(**kwargs):
        return expected_result

    monkeypatch.setattr(abinitio_module, "parse_ncaa_abinitio_config", fake_parse_ncaa_abinitio_config)
    monkeypatch.setattr(abinitio_module, "run_ncaa_abinitio", fake_run_ncaa_abinitio)

    result = Abinitio(
        output=str(output),
        atoms=atoms,
        params={"pdb": pdb_path, "target": "A1"},
    ).run()

    text = output.read_text(encoding="utf-8")
    assert result is expected_result
    assert "PARMFIT ABINITIO RESULT" in text
    assert "Route: NCAA" in text
    assert "Target: A1:SER" in text
    assert "Residue name: NAA" in text
    assert "Chirality: L" in text
    assert "Protein model: ff14SB" in text
    assert "Charge/mult: 0 1" in text
    assert "Representative conformer: ref" in text
    assert "RESP conformers: alpha, beta" in text
    assert "refined prepin: NAA_maple.prepin" in text
    assert "refined frcmod: NAA_maple.frcmod" in text
    assert "tleap input:" in text
    assert "NAA_tleap.in" in text
    assert "tleap PDB:      NAA_tleap.pdb" in text
    assert "Errors: 0" in text
    assert "Warnings: 2" in text
    assert "Notes: 1" in text
    assert "input written, not executed" not in text
    assert "tleap -s -f NAA_tleap.in" in text
    assert "ff14SB manual note:" not in text
    assert "ff19SB note:" not in text


def test_abinitio_routes_no_chain_residue_name_target_to_ncaa(monkeypatch, tmp_path: Path) -> None:
    pdb = "".join(
        [
            _pdb_atom("ATOM", 1, "N", "GLY", "", 1, 0.0, 0.0, 0.0, "N"),
            _pdb_atom("ATOM", 2, "CA", "GLY", "", 1, 1.4, 0.0, 0.0, "C"),
            _pdb_atom("ATOM", 3, "C", "GLY", "", 1, 2.4, 1.0, 0.0, "C"),
            _pdb_atom("ATOM", 4, "O", "GLY", "", 1, 3.4, 1.0, 0.0, "O"),
            _pdb_atom("ATOM", 5, "CB", "GLY", "", 1, 1.5, -1.2, 0.0, "C"),
            _pdb_atom("ATOM", 6, "N", "SER", "", 2, 4.0, 1.0, 0.0, "N"),
            _pdb_atom("ATOM", 7, "CA", "SER", "", 2, 5.4, 1.0, 0.0, "C"),
            _pdb_atom("ATOM", 8, "C", "SER", "", 2, 6.4, 2.0, 0.0, "C"),
            _pdb_atom("ATOM", 9, "O", "SER", "", 2, 7.4, 2.0, 0.0, "O"),
            _pdb_atom("ATOM", 10, "CB", "SER", "", 2, 5.4, -0.2, 0.0, "C"),
            "END\n",
        ]
    )
    pdb_path = _write_text(tmp_path / "no_chain_ncaa.pdb", pdb)

    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    atoms.info["charge"] = 0
    atoms.info["mult"] = 1
    atoms.calc = ZeroHessianCalculator()

    calls = {"parse": 0, "run": 0}

    def fake_parse_ncaa_abinitio_config(raw, *, pdb_path, target, charge, mult):
        calls["parse"] += 1
        assert raw["pdb"] == pdb_path
        assert target == "SER2"
        assert charge == 0
        assert mult == 1
        return object()

    def fake_run_ncaa_abinitio(*, output, source_atoms, structure, target_residue, config, log_info):
        del output, source_atoms, log_info
        calls["run"] += 1
        assert len(structure["residues"]) == 2
        assert target_residue["chain"] == "_"
        assert target_residue["resname"] == "SER"
        assert config is not None

    monkeypatch.setattr(abinitio_module, "parse_ncaa_abinitio_config", fake_parse_ncaa_abinitio_config)
    monkeypatch.setattr(abinitio_module, "run_ncaa_abinitio", fake_run_ncaa_abinitio)

    job = Abinitio(
        output=str(tmp_path / "no_chain_ncaa.out"),
        atoms=atoms,
        params={"pdb": pdb_path, "target": "SER2"},
    )

    job.run()
    assert calls == {"parse": 1, "run": 1}


def test_abinitio_routes_no_chain_residue_name_target_to_metal(monkeypatch, tmp_path: Path) -> None:
    pdb = "".join(
        [
            _pdb_atom("HETATM", 1, "FE", "FE", "", 291, 0.0, 0.0, 0.0, "FE"),
            "END\n",
        ]
    )
    pdb_path = _write_text(tmp_path / "no_chain_metal.pdb", pdb)

    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    atoms.info["charge"] = 3
    atoms.info["mult"] = 2
    atoms.info["oxy"] = 3
    atoms.calc = ZeroHessianCalculator()

    calls = {"parse": 0, "run": 0}

    def fake_parse_metal_abinitio_config(raw, *, pdb_path, target, charge, mult, target_residue, oxy=None):
        calls["parse"] += 1
        assert raw["pdb"] == pdb_path
        assert target == "FE291"
        assert charge == 3
        assert mult == 2
        assert oxy == 3
        assert target_residue["chain"] == "_"
        assert target_residue["resname"] == "FE"
        return object()

    def fake_run_metal_abinitio(*, output, source_atoms, structure, config, log_info):
        del output, source_atoms, log_info
        calls["run"] += 1
        assert len(structure["residues"]) == 1
        assert config is not None

    monkeypatch.setattr(abinitio_module, "parse_metal_abinitio_config", fake_parse_metal_abinitio_config)
    monkeypatch.setattr(abinitio_module, "run_metal_abinitio", fake_run_metal_abinitio)

    job = Abinitio(
        output=str(tmp_path / "no_chain_metal.out"),
        atoms=atoms,
        params={"pdb": pdb_path, "target": "FE291"},
    )

    job.run()
    assert calls == {"parse": 1, "run": 1}
