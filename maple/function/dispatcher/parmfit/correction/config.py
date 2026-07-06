"""Usage: parse user-facing correction parameters."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

from ...jobABC import JobABC
from ..utils.QMInterface import QMReferenceConfig, build_qm_reference_config
from ..utils.Scan.optimizer import LBFGSParams
from ..utils.TorsionFit import TorsionFitParams, build_torsion_fit_params


SUPPORTED_BONDED_METHODS = ("mseminario", "seminario", "none")


@dataclass
class CorrectionConfig:
    mol2: str = ""
    bonded: str = "mseminario"
    vib_scale: float = 1.0
    qm: QMReferenceConfig = field(default_factory=QMReferenceConfig)
    torsion: TorsionFitParams = field(default_factory=TorsionFitParams)
    lbfgs: LBFGSParams = field(default_factory=LBFGSParams)
    scan_opt: LBFGSParams = field(default_factory=LBFGSParams)


def build_correction_config(raw_params: Optional[dict]) -> CorrectionConfig:
    params = raw_params if isinstance(raw_params, dict) else {}
    config = CorrectionConfig()
    sub_dict = JobABC._select_subdict(params, ("parmfit", "correction"))
    JobABC._update_dataclass_from_dict(config, sub_dict)
    config.bonded = str(config.bonded).strip().lower()
    if config.bonded not in SUPPORTED_BONDED_METHODS:
        raise ValueError(
            f"Unsupported bonded method {config.bonded!r}; expected one of {', '.join(SUPPORTED_BONDED_METHODS)}."
        )
    qm_source = dict(params)
    qm_source.update(sub_dict)
    config.qm = build_qm_reference_config(qm_source)
    config.torsion = build_torsion_fit_params(params)

    lbfgs = LBFGSParams()
    JobABC._update_dataclass_from_dict(lbfgs, JobABC._select_subdict(params, ("lbfgs", "LBFGS", "opt")))
    config.lbfgs = lbfgs
    config.scan_opt = deepcopy(lbfgs)
    config.scan_opt.write_traj = False
    config.scan_opt.verbose = 0
    return config
