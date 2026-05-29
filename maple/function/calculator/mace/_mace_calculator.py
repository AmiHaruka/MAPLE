import os
import torch
import numpy as np
from typing import Dict, Union, Sequence, Optional
from ase.calculators.calculator import all_changes
from ..calculator_base import CalcABC, IMPLICIT_SOLVENT_FORCE_ERROR
from typing import Literal
EV2HARTREE = 1.0 / 27.211386245988

# ------------------------ Basic helpers ------------------------

_SYMBOL2Z = {
    "H":1, "He":2, "Li":3, "Be":4, "B":5, "C":6, "N":7, "O":8, "F":9, "Ne":10,
    "Na":11, "Mg":12, "Al":13, "Si":14, "P":15, "S":16, "Cl":17, "Ar":18,
    "K":19, "Ca":20, "Sc":21, "Ti":22, "V":23, "Cr":24, "Mn":25, "Fe":26, "Co":27, "Ni":28, "Cu":29, "Zn":30
}

def _symbols_to_Z(symbols: Sequence[Union[str,int]]) -> list:
    """Convert element symbols to atomic numbers."""
    out = []
    for s in symbols:
        if isinstance(s, int):
            out.append(int(s))
        else:
            z = _SYMBOL2Z.get(str(s))
            if z is None:
                raise ValueError(f"Unknown element symbol: {s}")
            out.append(z)
    return out

def _one_hot_node_attrs(Z: torch.Tensor, atomic_number_table: list, dtype=torch.float64) -> torch.Tensor:
    """Convert atomic numbers into one-hot vectors aligned with atomic_number_table."""
    table = torch.tensor(atomic_number_table, dtype=torch.long, device=Z.device)
    eq = (Z[:, None] == table[None, :])
    if not torch.all(eq.any(dim=1)):
        miss = Z[~eq.any(dim=1)].unique().tolist()
        raise ValueError(f"Atomic number(s) {miss} not in AtomicNumberTable {atomic_number_table}")
    return eq.to(dtype)

def _radius_graph_no_pbc(positions: torch.Tensor, r_max: float):
    """Construct a simple O(N^2) radius graph without periodic boundaries."""
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
    shifts = torch.zeros((edge_index.size(1), 3), dtype=positions.dtype, device=positions.device)
    return edge_index, shifts

def build_data_from_atoms(atoms, model, device="cpu"):
    """Build a data_dict for Wrapper.forward() from an ASE Atoms object."""
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


# ------------------------ Calculator ------------------------

class MACECalculator(CalcABC):
    """ASE-style calculator wrapping a scripted Wrapper MACE model."""

    implemented_properties = ['energy', 'forces', 'free_energy']
    supported_hessian_modes = ("analytic", "numerical")

    def __init__(self, 
        device: torch.device, 
        model: str = 'maceoff23s', 
        model_path: Optional[str] = None,
        overwrite: bool = False,
        implicit: Literal["gbsa", "none"] = "gbsa",
        solvent: str = 'none',
        ):

        """
        Args:
            device (torch.device): Torch device.
            model (str): Name of the model (expects `<model>.pt` under `model/`).
            model_path (str, optional): Explicit path to the scripted model file.
            overwrite (bool): Whether to overwrite existing models (unused).
        """
        super().__init__()
        if model_path is None:
            model_dir = os.path.dirname(os.path.realpath(__file__))
            model_dir = os.path.dirname(model_dir)
            model_path = os.path.join(model_dir, 'model', f'{model}.pt')

        # Load the scripted wrapper model
        self.model = torch.jit.load(model_path, map_location=device)
        self.model.eval()

        for p in self.model.parameters():
            p.requires_grad_(False)

        self.device = device
        self.dtype = torch.float64
        self.overwrite = overwrite

        self.r_max = float(self.model.r_max)
        self.atomic_numbers = [int(z) for z in self.model.atomic_numbers]
        self.hessian = "analytic"

        # Initialize implicit solvent
        self.implicit_solv_init(implicit=implicit, solvent=solvent)

    def calculate(self, atoms=None, properties=['energy','forces'], system_changes=all_changes):
        """Main ASE calculation entry point."""
        self._reject_implicit_solvent_derivatives(properties)
        super().calculate(atoms, properties, system_changes)

        data_dict, local_or_ghost = build_data_from_atoms(
            atoms, self.model, device=self.device
        )

        # Forward pass (Wrapper returns total_energy_local tensor)
        total_energy_local = self.model.forward(
            data=data_dict,
            local_or_ghost=local_or_ghost,
            compute_virials=False
        )

        energy = total_energy_local.sum()
        energy = energy * EV2HARTREE  # Convert eV to Hartree

        if self.solvent_correction:
            solvent_energy = self.implicit_solv_energy(atoms)
            energy += solvent_energy

        self.results['energy'] = energy.item()
        self.results['free_energy'] = energy.item()

        # Compute forces by autograd if requested
        if 'forces' in properties:
            data_dict['positions'].requires_grad_(True)
            total_energy_local = self.model.forward(
                data=data_dict,
                local_or_ghost=local_or_ghost,
                compute_virials=False
            )
            forces = -torch.autograd.grad(
                total_energy_local.sum(),
                data_dict['positions'],
                create_graph=False,
                retain_graph=False
            )[0]
            forces = forces * EV2HARTREE

            if self.solvent_correction:
                solvent_energy, solvent_force = self.implicit_solv_energy_and_force(atoms)
                forces += solvent_force

            self.results['forces'] = forces.detach().cpu().numpy()

        if "hessian" in properties:
            self.results["hessian"] = self.get_hessian(atoms)

    def get_energy(self, atoms) -> torch.Tensor:
        """Compute total energy as a torch scalar."""
        data_dict, local_or_ghost = build_data_from_atoms(
            atoms, self.model, device=self.device
        )
        total_energy_local = self.model.forward(
            data=data_dict,
            local_or_ghost=local_or_ghost,
            compute_virials=False
        )
        return total_energy_local.sum()

    @staticmethod
    def compute_hessian(coords: torch.Tensor, energy: torch.Tensor) -> torch.Tensor:
        """Compute the Hessian matrix (3N x 3N) by second derivatives."""
        num_atoms = coords.shape[0]
        hessian = torch.zeros((3 * num_atoms, 3 * num_atoms), dtype=coords.dtype, device=coords.device)
        grad = torch.autograd.grad(energy, coords, create_graph=True)[0].view(-1)
        for i in range(3 * num_atoms):
            grad2 = torch.autograd.grad(grad[i], coords, retain_graph=True)[0].view(-1)
            hessian[i, :] = grad2
        return hessian

    def _build_graph_inputs(self, atoms, positions: Optional[torch.Tensor] = None):
        if positions is None:
            positions = torch.tensor(
                atoms.get_positions(), dtype=self.dtype, device=self.device
            )
        Z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long, device=self.device)
        node_attrs = _one_hot_node_attrs(Z, self.atomic_numbers, dtype=self.dtype)
        edge_index, shifts = _radius_graph_no_pbc(positions, self.r_max)
        N = positions.size(0)
        batch = torch.zeros(N, dtype=torch.int64, device=self.device)
        cell = torch.zeros(3, 3, dtype=self.dtype, device=self.device)
        charge = torch.zeros(N, dtype=self.dtype, device=self.device)
        dipole = torch.zeros(1, 3, dtype=self.dtype, device=self.device)
        energy = torch.tensor([0.0], dtype=self.dtype, device=self.device)
        energy_weight = torch.tensor([0.0], dtype=self.dtype, device=self.device)
        force = torch.zeros(N, 3, dtype=self.dtype, device=self.device)
        forces_weight = torch.tensor([0.0], dtype=self.dtype, device=self.device)
        ptr = torch.tensor([0, N], dtype=torch.int64, device=self.device)
        stress = torch.zeros(1, 3, 3, dtype=self.dtype, device=self.device)
        stress_weight = torch.tensor([0.0], dtype=self.dtype, device=self.device)
        unit_shifts = torch.zeros(edge_index.size(1), 3, dtype=self.dtype, device=self.device)
        virials = torch.zeros(1, 3, 3, dtype=self.dtype, device=self.device)
        virials_weight = torch.tensor([0.0], dtype=self.dtype, device=self.device)
        weight = torch.tensor([1.0], dtype=self.dtype, device=self.device)

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
            'positions': positions,
            'ptr': ptr,
            'shifts': shifts,
            'stress': stress,
            'stress_weight': stress_weight,
            'unit_shifts': unit_shifts,
            'virials': virials,
            'virials_weight': virials_weight,
            'weight': weight
        }
        local_or_ghost = torch.ones(N, dtype=self.dtype, device=self.device)
        return data_dict, local_or_ghost

    def _get_hessian_analytic(self, atoms=None) -> np.ndarray:
        """Compute the Hessian matrix for an ASE Atoms object."""
        positions = torch.tensor(
            atoms.get_positions(),
            dtype=self.dtype,
            device=self.device,
            requires_grad=True
        )

        data_dict, local_or_ghost = self._build_graph_inputs(atoms, positions=positions)
        total_energy_local = self.model.forward(
            data=data_dict,
            local_or_ghost=local_or_ghost,
            compute_virials=False
        )
        energy = total_energy_local.sum() * EV2HARTREE

        hessian = self.compute_hessian(positions, energy)
        return hessian.detach().cpu().numpy()

    def _get_hessian_numerical(self, atoms, delta: float = 0.002) -> np.ndarray:
        from ase.constraints import FixAtoms

        N = len(atoms)
        pos0 = atoms.get_positions().copy()
        fixed = {
            i for c in getattr(atoms, "constraints", [])
            if isinstance(c, FixAtoms)
            for i in c.get_indices()
        }
        movable = [i for i in range(N) if i not in fixed]
        H = np.zeros((3 * N, 3 * N), dtype=np.float64)

        if len(movable) == 0:
            return H

        def force_at(positions: np.ndarray) -> np.ndarray:
            atoms_tmp = atoms.copy()
            atoms_tmp.set_positions(positions)
            if getattr(atoms, "constraints", None):
                atoms_tmp.set_constraint(atoms.constraints)
            self.calculate(atoms_tmp, properties=["forces"], system_changes=all_changes)
            return np.asarray(self.results["forces"], dtype=np.float64)

        for a in movable:
            for k in range(3):
                row = 3 * a + k
                pos_p = pos0.copy()
                pos_p[a, k] += delta
                Fp = force_at(pos_p)

                pos_m = pos0.copy()
                pos_m[a, k] -= delta
                Fm = force_at(pos_m)

                H[row, :] = (-(Fp - Fm) / (2.0 * delta)).reshape(-1)

        return H

    def get_hessian(self, atoms=None, delta: float = 0.002) -> np.ndarray:
        if self.solvent_correction:
            raise NotImplementedError(IMPLICIT_SOLVENT_FORCE_ERROR)

        if self.hessian == "analytic":
            return self._get_hessian_analytic(atoms)
        if self.hessian == "numerical":
            return self._get_hessian_numerical(atoms, delta)
        raise ValueError(
            f"Unknown hessian method: {self.hessian}. Must be 'analytic' or 'numerical'"
        )
