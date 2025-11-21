# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
from typing import Dict, Union, Sequence, Optional
from ase.calculators.calculator import all_changes
from ..calculator_base import CalcABC
from typing import Literal

EV2HARTREE = 1.0 / 27.211386245988

# ------------------------ Basic helpers ------------------------

_SYMBOL2Z = {
    "H":1, "He":2, "Li":3, "Be":4, "B":5, "C":6, "N":7, "O":8, "F":9, "Ne":10,
    "Na":11, "Mg":12, "Al":13, "Si":14, "P":15, "S":16, "Cl":17, "Ar":18,
    "K":19, "Ca":20, "Sc":21, "Ti":22, "V":23, "Cr":24, "Mn":25, "Fe":26,
    "Co":27, "Ni":28, "Cu":29, "Zn":30
}

def _one_hot_node_attrs(Z: torch.Tensor, atomic_number_table: list, dtype=torch.float64) -> torch.Tensor:
    table = torch.tensor(atomic_number_table, dtype=torch.long, device=Z.device)
    eq = (Z[:, None] == table[None, :])
    if not torch.all(eq.any(dim=1)):
        miss = Z[~eq.any(dim=1)].unique().tolist()
        raise ValueError(f"Atomic number(s) {miss} not in AtomicNumberTable {atomic_number_table}")
    return eq.to(dtype)

def _radius_graph_no_pbc(positions: torch.Tensor, r_max: float):
    N = positions.size(0)
    rij = positions[:, None, :] - positions[None, :, :]
    d2 = (rij * rij).sum(dim=-1)
    mask = torch.ones((N, N), dtype=torch.bool, device=positions.device)
    mask.fill_diagonal_(False)
    mask &= (d2 <= (r_max + 1e-12) ** 2)
    iu, ju = torch.nonzero(torch.triu(mask), as_tuple=True)
    src = torch.cat([iu, ju], dim=0)
    dst = torch.cat([ju, iu], dim=0)
    edge_index = torch.stack([src, dst], dim=0).to(torch.long)
    shifts = torch.zeros(edge_index.size(1), 3, dtype=positions.dtype, device=positions.device)
    return edge_index, shifts

def build_data_from_atoms(atoms, model, device="cpu"):
    device = torch.device(device)

    pos = torch.tensor(atoms.get_positions(), dtype=torch.float64, device=device)
    Z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long, device=device)
    r_max = float(model.r_max)
    atomic_number_table = [int(z) for z in model.atomic_numbers]

    node_attrs = _one_hot_node_attrs(Z, atomic_number_table)
    edge_index, shifts = _radius_graph_no_pbc(pos, r_max)

    N = pos.size(0)
    batch = torch.zeros(N, dtype=torch.int64, device=device)
    cell = torch.zeros(3, 3, dtype=torch.float64, device=device)
    charge = torch.zeros(N, dtype=torch.float64, device=device)
    dipole = torch.zeros(1, 3, dtype=torch.float64, device=device)
    energy = torch.tensor([0.0], dtype=torch.float64, device=device)
    energy_weight = torch.tensor([0.0], dtype=torch.float64, device=device)
    force = torch.zeros(N, 3, dtype=torch.float64, device=device)
    forces_weight = torch.tensor([0.0], dtype=torch.float64, device=device)
    ptr = torch.tensor([0, N], dtype=torch.int64, device=device)
    stress = torch.zeros(1, 3, 3, dtype=torch.float64, device=device)
    stress_weight = torch.tensor([0.0], dtype=torch.float64, device=device)
    unit_shifts = torch.zeros(edge_index.size(1), 3, dtype=torch.float64, device=device)
    virials = torch.zeros(1, 3, 3, dtype=torch.float64, device=device)
    virials_weight = torch.tensor([0.0], dtype=torch.float64, device=device)
    weight = torch.tensor([1.0], dtype=torch.float64, device=device)

    data_dict = {
        'batch': batch,
        'cell': cell,
        'charges': charge,
        'dipole': dipole,
        'edge_index': edge_index,
        'energy': energy,
        'energy_weight': energy_weight,
        'forces': force,
        'forces_weight': forces_weight,
        'node_attrs': node_attrs,
        'positions': pos,
        'ptr': ptr,
        'shifts': shifts,
        'stress': stress,
        'stress_weight': stress_weight,
        'unit_shifts': unit_shifts,
        'virials': virials,
        'virials_weight': virials_weight,
        'weight': weight
    }

    local_or_ghost = torch.ones(N, dtype=torch.float64, device=device)
    return data_dict, local_or_ghost


# ======================================================================
# ------------------------ NEW MACE CALCULATOR -------------------------
# ======================================================================

class MACEModelCalculator(CalcABC):
    """
    ASE calculator: directly loads raw `.model` (PyTorch MACE model).
    No TorchScript needed.
    """

    implemented_properties = ['energy', 'forces', 'free_energy']

    def __init__(
        self,
        device: torch.device,
        model: str,
        implicit: Literal["gbsa", "none"] = "gbsa",
        solvent: str = 'none',
    ):
        super().__init__()

        # --------------------- Load raw .model ------------------------
        model_dir = os.path.dirname(os.path.realpath(__file__))
        model_dir = os.path.dirname(model_dir)
        model_path = os.path.join(model_dir, 'model', f'{model}.model')
        self.model = torch.load(model_path, map_location=device, weights_only=False)
        self.model.to(device)
        self.model.eval()

        for p in self.model.parameters():
            p.requires_grad_(False)

        self.device = device
        self.dtype = torch.float64

        self.r_max = float(self.model.r_max)
        self.atomic_numbers = [int(z) for z in self.model.atomic_numbers]

        # Initialize implicit solvent
        self.implicit_solv_init(implicit=implicit, solvent=solvent)

    # ==================================================================
    def calculate(self, atoms=None, properties=['energy', 'forces'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        data_dict, local_or_ghost = build_data_from_atoms(
            atoms, self.model, device=self.device
        )

        # ----------------- Forward energy (MACE raw model) -----------------
        out = self.model(
            data_dict,
            training=False,
            compute_force=False,
            compute_virials=False,
            compute_stress=False,
            compute_displacement=False,
        )

        # node_energy is present in all MACE models
        node_energy = out["node_energy"]
        total_energy = torch.sum(node_energy)

        energy = total_energy * EV2HARTREE

        # ------------------- Optional implicit solvent ---------------------
        if self.solvent_correction:
            energy += self.implicit_solv_energy(atoms)

        self.results['energy'] = energy.item()
        self.results['free_energy'] = energy.item()

        # =========================== Forces ================================
        if 'forces' in properties:
            data_dict['positions'].requires_grad_(True)

            out = self.model(
                data_dict,
                training=False,
                compute_force=False,
                compute_virials=False,
                compute_stress=False,
                compute_displacement=False,
            )
            ene = out["node_energy"].sum() * EV2HARTREE

            forces = -torch.autograd.grad(
                ene, data_dict['positions'],
                create_graph=False, retain_graph=False
            )[0]

            if self.solvent_correction:
                sol_e, sol_f = self.implicit_solv_energy_and_force(atoms)
                forces += sol_f

            self.results['forces'] = forces.detach().cpu().numpy()

        # =========================== Hessian ===============================
        if "hessian" in properties:
            if self.solvent_correction:
                raise NotImplementedError("Hessian + solvent not implemented.")

            self.results["hessian"] = self.get_hessian(atoms)

    # ==================================================================
    # Hessian
    # ==================================================================
    @staticmethod
    def compute_hessian(coords: torch.Tensor, energy: torch.Tensor) -> torch.Tensor:
        num_atoms = coords.shape[0]
        hessian = torch.zeros((3*num_atoms, 3*num_atoms),
                              dtype=coords.dtype, device=coords.device)

        grad = torch.autograd.grad(energy, coords, create_graph=True)[0].view(-1)

        for i in range(3*num_atoms):
            grad2 = torch.autograd.grad(grad[i], coords, retain_graph=True)[0].view(-1)
            hessian[i, :] = grad2
        return hessian

    def get_hessian(self, atoms):
        coords = torch.tensor(
            atoms.get_positions(), dtype=self.dtype,
            device=self.device, requires_grad=True
        )

        data_dict, local_or_ghost = build_data_from_atoms(
            atoms, self.model, device=self.device
        )
        out = self.model(
            data_dict,
            training=False,
            compute_force=False,
            compute_virials=False,
            compute_stress=False,
            compute_displacement=False,
        )

        energy = out["node_energy"].sum() * EV2HARTREE
        H = self.compute_hessian(coords, energy)
        return H.detach().cpu().numpy()
