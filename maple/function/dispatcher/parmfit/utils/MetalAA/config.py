"""Usage: parse user-facing MetalAA input options."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..interface import RespConfig, set_method
from ..QMInterface import QMReferenceConfig, build_qm_reference_config
from ...runconfig import as_tracked


SUPPORTED_PRO_FF = ("ff14SB", "ff19SB")
SUPPORTED_WATER_MODELS = ("tip3p", "spce", "tip4pew", "opc3", "opc", "fb3", "fb4")
SUPPORTED_ION_PARAMETER_SETS = ("hfe", "cm", "iod", "12_6", "12_6_4")
SUPPORTED_BONDED_METHODS = ("mseminario", "seminario")


@dataclass(frozen=True)
class MetalAbinitioConfig:
    pdb_path: str
    ion_resids: str
    ion_charges: int
    ion_mults: int
    target_residue: dict
    resp: RespConfig
    qm: QMReferenceConfig = field(default_factory=QMReferenceConfig)
    ncaa_resids: list[str] = field(default_factory=list)
    ncaa_charges: list[int] = field(default_factory=list)
    ncaa_resnames: list[str] = field(default_factory=list)
    ncaa_mults: list[int] = field(default_factory=list)
    ncaa_chgfit: str = "resp"
    ncaa_chglevel: str = "HF/6-31G(d)"
    ncaa_chgroute: str = ""
    lig_resids: list[str] = field(default_factory=list)
    lig_charges: list[int] = field(default_factory=list)
    lig_mults: list[int] = field(default_factory=list)
    lig_chgfit: str = "abcg2"
    lig_chglevel: str = "HF/6-31G(d)"
    lig_chgroute: str = ""
    add_resid: list[str] = field(default_factory=list)
    set_bonded: list[tuple[int, int]] = field(default_factory=list)
    cluster_cutoff: float = 3.0
    donor_cutoff: float = 2.7
    vib_scale: float = 1.0
    opt_max_iter: int = 256
    opt_max_step: float = 0.2
    wat_ff: str = "tip3p"
    ion_ff: str = "12_6"
    pro_ff: str = "ff14SB"
    bonded: str = "mseminario"


def _split_entries(value) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(token).strip() for token in value if str(token).strip()]
    return [token.strip() for token in str(value).split(",") if token.strip()]


def _parse_set_bonded(value: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for token in _split_entries(value):
        parts = token.split("-")
        if len(parts) != 2:
            raise ValueError(f"Invalid set_bonded pair {token!r}; expected SERIAL-SERIAL.")
        try:
            left, right = int(parts[0]), int(parts[1])
        except ValueError as exc:
            raise ValueError(f"Invalid set_bonded pair {token!r}; expected integer SERIAL-SERIAL.") from exc
        pairs.append((left, right))
    return pairs


def parse_metal_abinitio_config(
    raw: dict | None,
    *,
    pdb_path: str,
    ion_resids: str,
    target_residue: dict,
) -> MetalAbinitioConfig:

    raw = as_tracked(raw)
    vib_scale = float(raw.get("vib_scale", 1.0))
    raw.set_group("MetalAA run", "Site composition, force field, RESP settings and ligand declarations")
    raw_pro_ff = str(raw.get("pro_ff", "ff14SB")).strip()
    resp_backend = str(raw.get("resp_backend", "gaussian")).strip().lower()
    if resp_backend not in {"gaussian"}:
        raise ValueError(f"Unsupported RESP backend {resp_backend!r}; expected one of 'gaussian'.")

    pro_ff = next((item for item in SUPPORTED_PRO_FF if item.lower() == raw_pro_ff.lower()), None)
    if pro_ff is None:
        raise ValueError(f"Unsupported protein ff {raw_pro_ff!r}; expected one of {', '.join(SUPPORTED_PRO_FF)}.")

    wat_ff = str(raw.get("wat_ff", "tip3p")).strip().lower()
    if wat_ff not in SUPPORTED_WATER_MODELS:
        raise ValueError(f"Unsupported water ff {wat_ff!r}; expected one of {', '.join(SUPPORTED_WATER_MODELS)}.")

    ion_ff = str(raw.get("ion_ff", "12_6")).strip().lower()
    if ion_ff not in SUPPORTED_ION_PARAMETER_SETS:
        raise ValueError(
            f"Unsupported ion ff {ion_ff!r}; expected one of {', '.join(SUPPORTED_ION_PARAMETER_SETS)}."
        )

    bonded = str(raw.get("bonded", "mseminario")).strip().lower()
    if bonded not in SUPPORTED_BONDED_METHODS:
        raise ValueError(
            f"Unsupported bonded method {bonded!r}; expected one of {', '.join(SUPPORTED_BONDED_METHODS)}."
        )

    chgmod = int(raw.get("chgmod", 1))
    if chgmod not in {0, 1, 2, 3}:
        raise ValueError("chgmod must be one of 0, 1, 2, or 3.")

    chg_level = str(raw.get("chg_level", "PBE1PBE/def2SVP")).strip()
    charge_theory, _sep, charge_basis = (part.strip() for part in chg_level.partition("/"))
    if not charge_theory or not charge_basis:
        raise ValueError(f"chg_level {chg_level!r} must use METHOD/BASIS syntax.")

    ion_charges = int(raw.get("ion_charges", 0))
    ion_mults = int(raw.get("ion_mults", 1))
    chg_route = str(raw.get("chg_route", "")).strip()
    fixchg_resids = _split_entries(raw.get("fixchg_resids", ""))
    add_resid = _split_entries(raw.get("add_resid", ""))
    set_bonded = _parse_set_bonded(raw.get("set_bonded", ""))
    cluster_cutoff = float(raw.get("cluster_cutoff", 3.0))
    donor_cutoff = float(raw.get("donor_cutoff", 2.7))
    lig_resids = _split_entries(raw.get("lig_resids", ""))
    lig_charges = [int(token) for token in _split_entries(raw.get("lig_charges", ""))]
    if len(lig_charges) != len(lig_resids):
        raise ValueError(
            f"lig_charges ({len(lig_charges)} entries) must pair 1:1 with lig_resids ({len(lig_resids)} entries)."
        )
    lig_mults = [int(token) for token in _split_entries(raw.get("lig_mults", ""))]
    if lig_mults and len(lig_mults) != len(lig_resids):
        raise ValueError(
            f"lig_mults ({len(lig_mults)} entries) must pair 1:1 with lig_resids ({len(lig_resids)} entries)."
        )
    lig_chgfit = str(raw.get("lig_chgfit", "abcg2")).strip()
    lig_chglevel = str(raw.get("lig_chglevel", "HF/6-31G(d)")).strip()
    lig_chgroute = str(raw.get("lig_chgroute", "")).strip()

    qm = build_qm_reference_config(raw)

    raw.set_group("NCAA run", "Non-standard amino acid declarations built through the native NCAA flow; charges frozen into the site RESP")
    ncaa_resids = _split_entries(raw.get("ncaa_resids", ""))
    ncaa_charges_text = str(raw.get("ncaa_charges", "")).strip()
    ncaa_charges = [int(token) for token in _split_entries(ncaa_charges_text)]
    if len(ncaa_charges) != len(ncaa_resids):
        raise ValueError(
            f"ncaa_charges ({len(ncaa_charges)} entries) must pair 1:1 with ncaa_resids ({len(ncaa_resids)} entries)."
        )
    ncaa_resnames = _split_entries(raw.get("ncaa_resnames", ""))
    if ncaa_resnames and len(ncaa_resnames) != len(ncaa_resids):
        raise ValueError(
            f"ncaa_resnames ({len(ncaa_resnames)} entries) must pair 1:1 with ncaa_resids ({len(ncaa_resids)} entries)."
        )
    ncaa_mults = [int(token) for token in _split_entries(raw.get("ncaa_mults", ""))]
    if ncaa_mults and len(ncaa_mults) != len(ncaa_resids):
        raise ValueError(
            f"ncaa_mults ({len(ncaa_mults)} entries) must pair 1:1 with ncaa_resids ({len(ncaa_resids)} entries)."
        )
    ncaa_chgfit = str(raw.get("ncaa_chgfit", "resp")).strip()
    ncaa_chglevel = str(raw.get("ncaa_chglevel", "HF/6-31G(d)")).strip()
    ncaa_chgroute = str(raw.get("ncaa_chgroute", "")).strip()

    raw.set_group("MLIP optimization", "MLIP geometry optimization controls")
    opt_max_iter = int(raw.get("opt_max_iter", 256))
    opt_max_step = float(raw.get("opt_max_step", 0.2))

    return MetalAbinitioConfig(
        pdb_path=pdb_path,
        ion_resids=ion_resids,
        ion_charges=ion_charges,
        ion_mults=ion_mults,
        ncaa_resids=ncaa_resids,
        ncaa_charges=ncaa_charges,
        ncaa_resnames=ncaa_resnames,
        ncaa_mults=ncaa_mults,
        ncaa_chgfit=ncaa_chgfit,
        ncaa_chglevel=ncaa_chglevel,
        ncaa_chgroute=ncaa_chgroute,
        lig_resids=lig_resids,
        lig_charges=lig_charges,
        lig_mults=lig_mults,
        lig_chgfit=lig_chgfit,
        lig_chglevel=lig_chglevel,
        lig_chgroute=lig_chgroute,
        target_residue=target_residue,
        resp=RespConfig(
            qm=set_method(
                {
                    "backend": resp_backend,
                    "theory": charge_theory,
                    "basis": charge_basis,
                    "route": chg_route,
                    "nproc": qm.qm_nproc,
                    "mem": qm.qm_mem,
                }
            ),
            chgmod=chgmod,
            fixchg_resids=fixchg_resids,
            wat_ff=wat_ff,
            pro_ff=pro_ff,
        ),
        qm=qm,
        add_resid=add_resid,
        set_bonded=set_bonded,
        cluster_cutoff=cluster_cutoff,
        donor_cutoff=donor_cutoff,
        vib_scale=vib_scale,
        opt_max_iter=opt_max_iter,
        opt_max_step=opt_max_step,
        wat_ff=wat_ff,
        ion_ff=ion_ff,
        pro_ff=pro_ff,
        bonded=bonded,
    )
