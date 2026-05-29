import os
import torch
import numpy as np
from typing import Dict, Union, Sequence, Optional
from ase.calculators.calculator import all_changes
from ..calculator_base import CalcABC, IMPLICIT_SOLVENT_FORCE_ERROR
from typing import Literal

EV2HARTREE = 1.0 / 27.211386245988

# ------------------------ 基础辅助函数 ------------------------

_SYMBOL2Z = {
    "H":1, "He":2, "Li":3, "Be":4, "B":5, "C":6, "N":7, "O":8, "F":9, "Ne":10,
    "Na":11, "Mg":12, "Al":13, "Si":14, "P":15, "S":16, "Cl":17, "Ar":18,
    "K":19, "Ca":20, "Sc":21, "Ti":22, "V":23, "Cr":24, "Mn":25, "Fe":26, "Co":27, "Ni":28, "Cu":29, "Zn":30
}

def _one_hot_node_attrs(Z: torch.Tensor, atomic_number_table: list, dtype=torch.float64) -> torch.Tensor:
    """将原子序数转换为one-hot向量"""
    table = torch.tensor(atomic_number_table, dtype=torch.long, device=Z.device)
    eq = (Z[:, None] == table[None, :])
    if not torch.all(eq.any(dim=1)):
        miss = Z[~eq.any(dim=1)].unique().tolist()
        raise ValueError(f"原子序数 {miss} 不在AtomicNumberTable {atomic_number_table} 中")
    return eq.to(dtype)

def _radius_graph_no_pbc(positions: torch.Tensor, r_max: float):
    """构建无周期性边界的半径图"""
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

def build_inputs_from_atoms(atoms, model, device="cpu"):
    """
    从ASE Atoms对象构建模型输入
    返回: (positions, node_attrs, edge_index, shifts, batch, ptr)
    
    注意: 新版MACE可能需要total_charge和total_spin,但这些在wrapper内部已经处理
    """
    device = torch.device(device)
    pos = torch.tensor(atoms.get_positions(), dtype=torch.float64, device=device)
    Z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long, device=device)
    r_max = float(model.r_max)
    atomic_number_table = [int(z) for z in model.atomic_numbers]

    node_attrs = _one_hot_node_attrs(Z, atomic_number_table)
    edge_index, shifts = _radius_graph_no_pbc(pos, r_max)

    N = pos.size(0)
    batch = torch.zeros(N, dtype=torch.int64, device=device)
    ptr = torch.tensor([0, N], dtype=torch.int64, device=device)
    
    return pos, node_attrs, edge_index, shifts, batch, ptr


# ------------------------ Calculator ------------------------

class MACEModelCalculator(CalcABC):
    """使用traced MACE模型的ASE计算器"""

    implemented_properties = ['energy', 'forces', 'free_energy', 'hessian']
    supported_hessian_modes = ("analytic", "numerical")

    def __init__(self, 
        device: torch.device, 
        model: str = 'maceomol', 
        overwrite: bool = False,
        implicit: Literal["gbsa", "none"] = "gbsa",
        solvent: str = 'none',
        ):
        """
        参数:
            device (torch.device): 计算设备
            model (str): 模型名称 (期望在 model/ 目录下有 <model>.pt 文件)
            overwrite (bool): 是否覆盖现有模型
            implicit (str): 隐式溶剂模型类型
            solvent (str): 溶剂类型
        """
        super().__init__()
        model_dir = os.path.dirname(os.path.realpath(__file__))
        model_dir = os.path.dirname(model_dir)
        model_path = os.path.join(model_dir, 'model', f'{model}.pt')

        # 加载traced模型
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

        # 初始化隐式溶剂
        self.implicit_solv_init(implicit=implicit, solvent=solvent)

    def calculate(self, atoms=None, properties=['energy','forces'], system_changes=all_changes):
        """主计算入口"""
        properties = self._reject_implicit_solvent_derivatives(properties)
        super().calculate(atoms, properties, system_changes)

        # 构建输入
        inputs = build_inputs_from_atoms(atoms, self.model, device=self.device)
        
        # 前向传播计算能量
        with torch.no_grad():
            total_energy = self.model(*inputs)  # 返回 [num_graphs] 的tensor

        energy = total_energy.sum()
        energy = energy * EV2HARTREE  # eV转Hartree

        if self.solvent_correction:
            solvent_energy = self.implicit_solv_energy(atoms)
            energy += solvent_energy

        self.results['energy'] = energy.item()
        self.results['free_energy'] = energy.item()

        # 如果需要计算力
        if 'forces' in properties:
            # 重新构建输入,positions需要梯度
            positions_grad = torch.tensor(
                atoms.get_positions(), 
                dtype=torch.float64, 
                device=self.device,
                requires_grad=True
            )
            
            # 重建其他输入
            Z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long, device=self.device)
            atomic_number_table = [int(z) for z in self.model.atomic_numbers]
            node_attrs = _one_hot_node_attrs(Z, atomic_number_table)
            edge_index, shifts = _radius_graph_no_pbc(positions_grad, self.r_max)
            N = positions_grad.size(0)
            batch = torch.zeros(N, dtype=torch.int64, device=self.device)
            ptr = torch.tensor([0, N], dtype=torch.int64, device=self.device)
            
            # 前向传播
            total_energy = self.model(positions_grad, node_attrs, edge_index, shifts, batch, ptr)
            
            # 计算梯度
            forces = -torch.autograd.grad(
                total_energy.sum(),
                positions_grad,
                create_graph=False,
                retain_graph=False
            )[0]
            forces = forces * EV2HARTREE

            self.results['forces'] = forces.detach().cpu().numpy()

        if "hessian" in properties:
            self.results["hessian"] = self.get_hessian(atoms)

    def get_energy(self, atoms) -> torch.Tensor:
        """计算总能量 (返回torch标量)"""
        inputs = build_inputs_from_atoms(atoms, self.model, device=self.device)
        with torch.no_grad():
            total_energy = self.model(*inputs)
        return total_energy.sum()

    @staticmethod
    def compute_hessian(coords: torch.Tensor, energy: torch.Tensor) -> torch.Tensor:
        """计算Hessian矩阵 (3N x 3N)"""
        num_atoms = coords.shape[0]
        hessian = torch.zeros((3 * num_atoms, 3 * num_atoms), dtype=coords.dtype, device=coords.device)
        grad = torch.autograd.grad(energy, coords, create_graph=True)[0].view(-1)
        for i in range(3 * num_atoms):
            grad2 = torch.autograd.grad(grad[i], coords, retain_graph=True)[0].view(-1)
            hessian[i, :] = grad2
        return hessian

    def _get_hessian_analytic(self, atoms=None) -> np.ndarray:
        """使用自动微分计算Hessian矩阵"""
        positions = torch.tensor(
            atoms.get_positions(),
            dtype=self.dtype,
            device=self.device,
            requires_grad=True
        )

        # 构建输入
        Z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long, device=self.device)
        atomic_number_table = [int(z) for z in self.model.atomic_numbers]
        node_attrs = _one_hot_node_attrs(Z, atomic_number_table)
        edge_index, shifts = _radius_graph_no_pbc(positions, self.r_max)
        N = positions.size(0)
        batch = torch.zeros(N, dtype=torch.int64, device=self.device)
        ptr = torch.tensor([0, N], dtype=torch.int64, device=self.device)
        
        # 计算能量
        total_energy = self.model(positions, node_attrs, edge_index, shifts, batch, ptr)
        energy = total_energy.sum() * EV2HARTREE

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
