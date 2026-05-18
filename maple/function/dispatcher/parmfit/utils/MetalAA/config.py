"""Usage: parse user-facing MetalAA input options."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..interface import RespConfig, set_method

SUPPORTED_WATER_MODELS = ("tip3p", "spce", "tip4pew", "opc3", "opc", "fb3", "fb4")
SUPPORTED_ION_PARAMETER_SETS = ("hfe", "cm", "iod", "12_6", "12_6_4")

@dataclass(frozen=True)
class MetalAbinitioConfig:
    pdb_path: str
    target: str
    charge: int
    mult: int
    target_residue: dict
    resp: RespConfig
    oxy: int | None = None
    add_resid: list[str] = field(default_factory=list)
    cfmol2: list[str] = field(default_factory=list)
    cluster_cutoff: float = 3.0
    donor_cutoff: float = 3.0
    opt_max_iter: int = 256
    opt_max_step: float = 0.2
    ionm: str = "12_6"


def parse_metal_abinitio_config(
    raw: dict | None,
    *,
    pdb_path: str,
    target: str,
    charge: int,
    mult: int,
    target_residue: dict,
    oxy: int | None = None,
) -> MetalAbinitioConfig:
    raw = dict(raw or {})

    watm = raw.get("watm", "opc").strip().lower()
    if watm not in SUPPORTED_WATER_MODELS:
        raise ValueError(f"Unsupported water model {watm!r}; expected one of {', '.join(SUPPORTED_WATER_MODELS)}.")

    ionm = raw.get("ionm", "12_6").strip().lower()
    if ionm not in SUPPORTED_ION_PARAMETER_SETS:
        raise ValueError(
            f"Unsupported ion parameter set {ionm!r}; expected one of {', '.join(SUPPORTED_ION_PARAMETER_SETS)}."
        )

    chgmod = int(raw.get("chgmod", 1))
    if chgmod not in {0, 1, 2, 3}:
        raise ValueError("chgmod must be one of 0, 1, 2, or 3.")

    return MetalAbinitioConfig(
        pdb_path=pdb_path,
        target=target,
        charge=charge,
        mult=mult,
        oxy=oxy,
        target_residue=target_residue,
        resp=RespConfig(
            qm=set_method(
                {
                    "backend": "gaussian",
                    "theory": raw.get("theory", "PBE1PBE").strip(),
                    "basis": raw.get("basis", "def2SVP").strip(),
                    "route": raw.get("route", "").strip(),
                    "nproc": int(raw.get("nproc", 8)),
                    "mem": int(raw.get("mem", 16)),
                }
            ),
            chgmod=chgmod,
            fixchg_resids=raw.get("fixchg_resids", "").split(),
            watm=watm,
        ),
        add_resid=raw.get("add_resid", "").split(),
        cfmol2=raw.get("cfmol2", "").split(),
        cluster_cutoff=float(raw.get("cluster_cutoff", 3.0)),
        donor_cutoff=float(raw.get("donor_cutoff", 3.0)),
        opt_max_iter=256,
        opt_max_step=0.2,
        ionm=ionm,
    )
