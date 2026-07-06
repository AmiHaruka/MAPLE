"""Usage: expose parmfit correction as the dispatcher-facing job class."""

from __future__ import annotations

from typing import Optional

from ase import Atoms

from ...jobABC import JobABC
from ..utils.readparm import CorrectionParameterSet
from ..utils.TorsionFit import run_torsion_workflow
from .artifacts import CorrectionWorkflowResult
from .config import CorrectionConfig, build_correction_config
from .workflow import run_correction_workflow

from maple.function.timer import timer


CorrectionParams = CorrectionConfig


class Correction(JobABC):
    def __init__(self, output: str, atoms: Atoms, params: Optional[dict] = None):
        super().__init__(output)
        self.atoms = atoms
        self.params = build_correction_config(params if isinstance(params, dict) else {})
        self.workflow_result: CorrectionWorkflowResult | None = None

    @property
    def result(self) -> Optional[CorrectionParameterSet]:
        if self.workflow_result is None:
            return None
        return self.workflow_result.final_parmset

    def run(self) -> CorrectionWorkflowResult:
        with timer("Parmfit correction"):
            self.workflow_result = run_correction_workflow(
                output=self.output,
                atoms=self.atoms,
                config=self.params,
                log_info=self.log_info,
                torsion_workflow_fn=run_torsion_workflow,
            )
            return self.workflow_result
