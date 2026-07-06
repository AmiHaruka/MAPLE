"""Usage: parse user-facing NCAA input options."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..interface import RespConfig, set_method
from ..QMInterface import QMReferenceConfig, build_qm_reference_config
from ..TorsionFit import TorsionFitParams, build_torsion_fit_params


SUPPORTED_PROM = ("ff14SB", "ff19SB")
SUPPORTED_WATER_MODELS = ("tip3p", "spce", "tip4pew", "opc3", "opc", "fb3", "fb4")
SUPPORTED_ION_PARAMETER_SETS = ("hfe", "cm", "iod", "12_6", "12_6_4")
SUPPORTED_BONDED_METHODS = ("mseminario", "seminario", "none")


@dataclass(frozen=True)
class NCAAAbinitioConfig:
    pdb_path: str
    target: str
    charge: int
    mult: int
    resp: RespConfig
    qm: QMReferenceConfig = field(default_factory=QMReferenceConfig)
    rn: str = "MOL"
    vib_scale: float = 1.0
    opt_max_iter: int = 256
    opt_max_step: float = 0.2
    torsion: TorsionFitParams = field(default_factory=TorsionFitParams)
    watm: str = "tip3p"
    ionm: str = "12_6"
    prom: str = "ff14SB"
    bonded: str = "mseminario"


def parse_ncaa_abinitio_config(
    raw: dict | None,
    *,
    pdb_path: str,
    target: str,
    charge: int,
    mult: int,
) -> NCAAAbinitioConfig:

    raw = dict(raw or {})
    raw_prom = raw.get("prom", "ff14SB").strip()
    resp_backend = raw.get("resp_backend", "gaussian").strip().lower()
    if resp_backend not in {"gaussian"}:
        raise ValueError(f"Unsupported RESP backend {resp_backend!r}; expected one of 'gaussian'.")

    prom = next((item for item in SUPPORTED_PROM if item.lower() == raw_prom.lower()), None)
    if prom is None:
        raise ValueError(f"Unsupported protein model {raw_prom!r}; expected one of {', '.join(SUPPORTED_PROM)}.")

    watm = raw.get("watm", "tip3p").strip().lower()
    if watm not in SUPPORTED_WATER_MODELS:
        raise ValueError(f"Unsupported water model {watm!r}; expected one of {', '.join(SUPPORTED_WATER_MODELS)}.")

    ionm = raw.get("ionm", "12_6").strip().lower()
    # NOTE: 
    # We allow unsupported ion parameter sets to be specified, 
    # since users may have custom parameters that they want to use. 
    # We just won't do any validation on them.
    
    #if ionm not in SUPPORTED_ION_PARAMETER_SETS:    
        # raise ValueError(
        #     f"Unsupported ion parameter set {ionm!r}; expected one of {', '.join(SUPPORTED_ION_PARAMETER_SETS)}."
        # )

    bonded = raw.get("bonded", "mseminario").strip().lower()
    if bonded not in SUPPORTED_BONDED_METHODS:
        raise ValueError(
            f"Unsupported bonded method {bonded!r}; expected one of {', '.join(SUPPORTED_BONDED_METHODS)}."
        )
    
    torsion = build_torsion_fit_params(raw)
    torsion.torsion_ensemble = False
    torsion._refresh_derived()

    qm = build_qm_reference_config(raw)
    charge_theory, charge_basis = (
        part.strip() for part in str(raw.get("chg_level", "HF/6-31G(d)")).strip().split("/", 1)
    )

    return NCAAAbinitioConfig(
        pdb_path=pdb_path,
        target=target,
        charge=charge,
        mult=mult,
        resp=RespConfig(
            qm=set_method(
                {
                    "backend": resp_backend,
                    "theory": charge_theory,
                    "basis": charge_basis,
                    "route": raw.get("chg_route", "").strip(),
                    "nproc": int(raw.get("qm_nproc", 8)),
                    "mem": int(raw.get("qm_mem", 24)),
                }
            ),
            watm=watm,
            prom=prom,
        ),
        qm=qm,
        rn=raw.get("rn", "MOL").strip().upper(),
        vib_scale=float(raw.get("vib_scale", 1.0)),
        opt_max_iter=int(raw.get("opt_max_iter", 256)),
        opt_max_step=float(raw.get("opt_max_step", 0.2)),
        torsion=torsion,
        watm=watm,
        ionm=ionm,
        prom=prom,
        bonded=bonded,
    )
