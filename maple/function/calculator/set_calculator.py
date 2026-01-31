import torch
from typing import Optional

import ase
from ase import Atoms

from .ani._ani_calculator import ANICalculator
from .mace._mace_calculator import MACECalculator

IMPLEMENTATION_MODELs = [
            'ani2x',
            'ani1x',
            'ani1ccx',
            'ani1xnr',
            'maceoff23s',
            'maceoff23m',
            'maceoff23l',
            'egret',
            'aimnet2',
            'uma',
            'maceomol',
            'aimnet2nse',
        ]

class SetClaculator():

    def __init__(self, device: torch.device,
            model:str,
            output: str,
            atoms: Optional[Atoms] = None,
            d4:bool=False,
            implicit:str = 'None',
            solvent: str = 'None',
            model_params: Optional[dict] = None) -> None:
        self.output = output
        self.model = model
        self.d4 = d4
        self.device = device
        self.model = model
        self.atoms = atoms
        self.implicit = implicit
        self.solvent = solvent
        self.model_params = model_params

    def set_calculator(self) -> ase.calculators.calculator.Calculator:
        if self.model not in IMPLEMENTATION_MODELs:
            error_message = f"\n [ERROR] Unsupported model: {self.model}\n"
            self.log_error(error_message)
            raise ValueError(error_message)

        # Initialize calculator based on model
        if self.model in ['ani2x', 'ani1x', 'ani1ccx', 'ani1xnr']:
            calculator = ANICalculator(model=self.model, d4=self.d4, device=self.device, implicit = self.implicit, solvent = self.solvent)
        else:
            if self.d4 == True : self.log_info([f"\n [WARNING:] D4 is not supported for model {self.model}. D4 will be ignored.\n"])
            if self.model in ['maceoff23s', 'maceoff23m', 'maceoff23l','egret']:
                from .mace._mace_calculator import MACECalculator
                calculator = MACECalculator(model=self.model, device=self.device, implicit = self.implicit, solvent = self.solvent)
            elif self.model in ['aimnet2', 'aimnet2nse']:
                from .aimnet._aimnet2_calculator import AIMNet2Calculator
                calculator = AIMNet2Calculator(model=self.model, device=self.device, implicit = self.implicit, solvent = self.solvent)
            elif self.model in ['uma']:
                from .uma._uma_calculator import UMACalculator
                # Extract UMA-specific parameters if model_params exists
                uma_task = None
                uma_size = None
                if self.model_params is not None:
                    uma_task = self.model_params.get('task', 'omol')
                    uma_size = self.model_params.get('size', 'uma-s-1p1')
                calculator = UMACalculator(
                    model=self.model,
                    device=self.device,
                    implicit=self.implicit,
                    solvent=self.solvent,
                    task=uma_task,
                    size=uma_size
                )
            elif self.model in ['maceomol']:
                from .mace._mace_general_calculator import MACEModelCalculator
                calculator = MACEModelCalculator(model=self.model, device=self.device, implicit = self.implicit, solvent = self.solvent)
            else:
                raise ValueError(f"Model '{self.model}' is not implemented yet.")

        # Check charge/mult compatibility and issue warning if needed
        if self.atoms is not None:
            has_charge = 'charge' in self.atoms.info and self.atoms.info['charge'] != 0
            has_mult = 'mult' in self.atoms.info and self.atoms.info['mult'] != 1

            if has_charge or has_mult:
                # Models that do NOT support charge/mult
                unsupported_models = ['ani2x', 'ani1x', 'ani1ccx', 'ani1xnr',
                                     'maceoff23s', 'maceoff23m', 'maceoff23l',
                                     'egret', 'maceomol']

                if self.model in unsupported_models:
                    charge_val = self.atoms.info.get('charge', 0)
                    mult_val = self.atoms.info.get('mult', 1)
                    self.log_info([
                        f"\n [WARNING] Model '{self.model}' does not support charge ({charge_val}) "
                        f"and multiplicity ({mult_val}) parameters.\n"
                        f"           These values will be IGNORED by the calculator.\n"
                        f"           Supported models: aimnet2, aimnet2nse, uma\n"
                    ])

        return calculator
        

    def log_error(self, error_message: str) -> None:
        """
        Logs error messages to the output file.

        Args:
            error_message: The error message to log.
        """
        with open(self.output, 'a') as file:
            file.write(f"ERROR: {error_message}\n")

    def log_info(self, info_message: list) -> None:
        """
        Logs info messages to the output file.

        Args:
            info_message: The info message to log.
        """
        with open(self.output, 'a') as file:
            for info in info_message:   
                file.write(f"{info}")