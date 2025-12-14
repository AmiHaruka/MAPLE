import importlib
import torch
import numpy as np
from typing import Literal
from ase import Atoms
from ase.calculators.calculator import all_changes
from ase.calculators.calculator import Calculator

try:
    from fairchem.core import pretrained_mlip
    from fairchem.core.calculate.ase_calculator import FAIRChemCalculator
    from fairchem.core.datasets import data_list_collater
except ImportError:
    raise ImportError("fairchem-core is not installed. Please install it first.")

EV2HARTREE = 1.0 / 27.211386245988


class UMACalculator(FAIRChemCalculator):
    """
    UMA Calculator: extends Meta's FAIRChemCalculator with additional methods for
    total energy and Hessian calculation. Default task is 'omol' (molecular systems).
    """

    def __init__(
        self,
        device: torch.device,
        model: str = "uma",
        overrides: dict | None = None,
        implicit: Literal["gbsa", "none"] = "gbsa",
        solvent: str = 'none',
    ):
        """
        Initialize UMA Calculator.

        Args:
            device (torch.device): Target device ('cuda' or 'cpu').
            model (str): UMA model name or local checkpoint path.
            overrides (dict, optional): Additional inference configuration overrides.
        """
        UMA_MODELS_MAP = {"uma": "uma-s-1p1"}
        model = UMA_MODELS_MAP.get(model)

        device = str(device)
        device = "cuda" if device.startswith("cuda") else "cpu"

        if not importlib.util.find_spec("fairchem"):
            raise ImportError("fairchem-core is not installed. Please install it first.")

        predictor = pretrained_mlip.get_predict_unit(
            model,
            inference_settings="default",
            overrides=overrides,
            device=device,
        )
        super().__init__(predict_unit=predictor, task_name="omol")
        self.device = torch.device(device)
        
        if implicit == "gbsa" and solvent != 'none':

            # GBSA solvent correction and QEq charge calculator
            from ..extra_correction import GBSA
            from ..extra_correction import QEqTorch

            self.solvent_correction = GBSA(solvent=solvent, device=self.device)
        
            self.chargecalc = QEqTorch(device=self.device)
        else:
            self.solvent_correction = None

    def get_energy(self, atoms: Atoms) -> torch.Tensor:
        """
        Compute total energy (Hartree) for a given atomic structure.

        Args:
            atoms (ase.Atoms): Atomic structure.

        Returns:
            torch.Tensor: Total energy in Hartree.
        """
        self.calculate(atoms, properties=["energy"], system_changes=all_changes)
        energy_value = self.results["energy"]

        if self.solvent_correction:
            solvent_energy = self.solvent_correction.get_energy(atoms)
            energy_value += solvent_energy

        return torch.tensor(energy_value, dtype=torch.float32, device=self.device)
    
    def get_hessian(
        self,
        atoms: Atoms,
        delta: float = 0.002,
        dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        """
        Compute Hessian by finite difference of UMA forces.
        Returns a (3N, 3N) torch.Tensor on self.device.
        """
        from ase.constraints import FixAtoms  # 已经添加了这个导入
        from ase.calculators.calculator import all_changes

        # ------------------------------------------------------
        # 1. Prepare
        # ------------------------------------------------------
        N = len(atoms)
        pos0 = atoms.get_positions()
        device = self.device

        # Find fixed atoms
        fixed = {
            i for c in atoms.constraints if isinstance(c, FixAtoms)
            for i in c.get_indices()
        }
        movable = [i for i in range(N) if i not in fixed]

        # If all atoms are fixed
        if len(movable) == 0:
            return torch.zeros((3*N, 3*N), dtype=dtype, device=device)

        # Output Hessian
        H = torch.zeros((3*N, 3*N), dtype=dtype, device=device)

        # Helper: get force using UMA (returns numpy array)
        def eval_force(positions: np.ndarray) -> torch.Tensor:
            atoms_tmp = atoms.copy()
            atoms_tmp.set_positions(positions)
            self.calculate(atoms_tmp, properties=["forces"], system_changes=all_changes)
            F = self.results["forces"]  # numpy (N, 3)
            return torch.tensor(F, dtype=dtype, device=device)

        # ------------------------------------------------------
        # 2. Finite difference: loop over movable atoms
        # ------------------------------------------------------
        for a in movable:
            for k in range(3):      # x/y/z direction

                # Displace +delta
                pos_p = pos0.copy()
                pos_p[a, k] += delta
                Fp = eval_force(pos_p)

                # Displace -delta
                pos_m = pos0.copy()
                pos_m[a, k] -= delta
                Fm = eval_force(pos_m)

                # Hessian: H = -∂F/∂x ≈ -(F(+δ) - F(-δ)) / (2δ)
                dF = (Fp - Fm) / (2.0 * delta)  # (N, 3)
                
                # Fill row for this DOF
                row = 3 * a + k
                H[row, :] = (-dF).reshape(-1)  # 明确添加负号，与ANI一致

        return H





    
    def calculate(self, atoms, properties=None, system_changes=None):
        """
        Override base calculate() to convert energy and forces into Hartree.

        Args:
            atoms (ase.Atoms): Atomic structure.
            properties (list, optional): Properties to calculate.
            system_changes (list, optional): System changes to consider.

        Returns:
            dict: Calculation results with energy and forces converted to Hartree.
        """
        super().calculate(atoms, properties, system_changes)

        if "energy" in self.results:
            self.results["energy"] *= EV2HARTREE
        if "free_energy" in self.results:
            self.results["free_energy"] *= EV2HARTREE
        if "forces" in self.results:
            self.results["forces"] *= EV2HARTREE

        if self.solvent_correction:
            atoms.atomic_charges = self.chargecalc(atoms)
            solvent_energy, solvent_force = self.solvent_correction.get_energy_and_force(atoms)
            solvent_force = solvent_force.detach().cpu().numpy()
            self.results["energy"] += solvent_energy.item()
            self.results["free_energy"] += solvent_energy.item()

            # Add solvent forces to the results
            self.results["forces"] += solvent_force


        return self.results
