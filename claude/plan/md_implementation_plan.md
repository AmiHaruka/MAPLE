# MAPLE 分子动力学 (MD) 模块实现计划

**作者**: Claude
**日期**: 2026-01-30 (创建) | 2026-01-30 (更新)
**目标**: 实现基于 MLP 的 NVE 和 NVT 分子动力学模拟功能
**当前状态**: Phase 1 完成 ✅ | Phase 2 进行中 ⏳
**完成报告**: `/home/user/software/MAPLE/claude/done/md_phase1_completion.md`

---

## 目录

1. [需求概述](#需求概述)
2. [架构设计](#架构设计)
3. [模块详细设计](#模块详细设计)
4. [文件修改清单](#文件修改清单)
5. [实现步骤](#实现步骤)
6. [测试验证](#测试验证)
7. [未来扩展预留](#未来扩展预留)

---

## 需求概述

### 核心功能

1. **NVE 系综**（微正则系综）
   - 经典 Velocity Verlet 积分器
   - 能量守恒检验
   - 无温度控制

2. **NVT 系综**（正则系综）
   - Velocity Verlet 积分器
   - Langevin 热浴（摩擦力 + 随机力）
   - 温度控制和监控

3. **能量和力计算**
   - 从现有 MLP Calculator 获取（ANI, MACE, UMA, AIMNet2）
   - 支持所有 MAPLE 已有模型
   - PBC 由 Calculator 处理（无需 MD 模块关心）

4. **模块化设计**
   - 热浴模块可开关
   - 未来预留 NPT、Nosé-Hoover 等扩展接口
   - 清晰的代码组织结构

### 输入参数

```bash
#md(ensemble=nvt, timestep=0.5, steps=10000, temperature=300,
    thermostat=langevin, friction=0.01, traj_every=10, log_every=100)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `ensemble` | str | 'nve' | nve, nvt, (npt预留) |
| `timestep` | float | 0.5 | 时间步长 (fs) |
| `steps` | int | 10000 | 总步数 |
| `temperature` | float | 300.0 | 目标温度 (K) |
| `thermostat` | str | 'langevin' | langevin, berendsen, (nose-hoover预留) |
| `friction` | float | 0.01 | Langevin摩擦系数 (1/fs) |
| `tau_t` | float | 100.0 | Berendsen耦合时间常数 (fs) |
| `traj_every` | int | 10 | 轨迹写入频率 |
| `log_every` | int | 100 | 日志输出频率 |
| `verbose` | int | 1 | 0=简洁, 1=详细, 2=调试 |

### 输出文件

```
task.out              # 主日志文件
task_md_traj.xyz      # MD 轨迹（XYZ格式，包含能量）
task_md_thermo.dat    # 热力学量（时间序列）
task_md_summary.txt   # 最终统计摘要
```

---

## 架构设计

### 总体架构

```
maple/function/dispatcher/md/
├── __init__.py                # 模块导出
├── md.py                      # MD 主调度器
├── ensemble/                  # 系综模块
│   ├── __init__.py
│   ├── nve.py                 # NVE 系综
│   ├── nvt.py                 # NVT 系综
│   └── ensemble_base.py       # 系综基类（未来扩展）
├── integrator/                # 积分器模块
│   ├── __init__.py
│   ├── velocity_verlet.py     # Velocity Verlet 积分器
│   └── integrator_base.py     # 积分器基类（未来扩展）
├── thermostat/                # 热浴模块
│   ├── __init__.py
│   ├── langevin.py            # Langevin 动力学
│   ├── berendsen.py           # Berendsen 热浴
│   └── thermostat_base.py     # 热浴基类（未来扩展 Nosé-Hoover）
├── utils.py                   # 工具函数（能量计算、温度计算等）
└── logger.py                  # MD 专用日志输出
```

### 类继承关系

```
JobABC (抽象基类)
    ↑
MDDispatcher (md.py) - 顶层调度器，选择系综
    ↓
    ├── NVEEnsemble (nve.py)
    │       ↓
    │   VelocityVerlet (integrator)
    │
    └── NVTEnsemble (nvt.py)
            ↓
        VelocityVerlet + LangevinThermostat/BerendsenThermostat
```

### 设计模式

1. **JobABC 继承**：遵循 MAPLE 现有模式
2. **组合模式**：Ensemble 组合 Integrator + Thermostat
3. **策略模式**：可切换的热浴实现
4. **模板方法**：基类定义流程，子类实现细节

---

## 模块详细设计

### 1. MD 主调度器 (`md.py`)

**职责**：
- 解析用户参数
- 选择合适的系综类
- 初始化速度
- 输出参数日志

**代码框架**：
```python
from dataclasses import dataclass
from typing import Optional
from ase import Atoms
from ..jobABC import JobABC

@dataclass
class MDParams:
    ensemble: str = 'nve'
    timestep: float = 0.5
    steps: int = 10000
    temperature: float = 300.0
    thermostat: str = 'langevin'
    friction: float = 0.01
    tau_t: float = 100.0
    traj_every: int = 10
    log_every: int = 100
    verbose: int = 1
    write_mode: str = 'incremental'  # 'incremental' 或 'batch'
    restart_file: Optional[str] = None  # 未来：重启功能

class MDDispatcher(JobABC):
    """分子动力学主调度器"""

    def __init__(self, atoms: Atoms, output: str, paras: Optional[dict] = None):
        super().__init__(output)
        self.atoms = atoms
        self.params = self._init_params(MDParams, paras, ("md", "MD"))
        self._log_parameters()

    def run(self):
        """选择系综并运行"""
        ensemble_type = self.params.ensemble.lower()

        if ensemble_type == 'nve':
            from .ensemble import NVEEnsemble
            ensemble = NVEEnsemble(self.atoms, self.output, self.params)

        elif ensemble_type == 'nvt':
            from .ensemble import NVTEnsemble
            ensemble = NVTEnsemble(self.atoms, self.output, self.params)

        else:
            raise ValueError(f"Unsupported ensemble: {ensemble_type}. "
                           f"Valid options: nve, nvt")

        ensemble.run()
```

**关键方法**：
- `_log_parameters()`: 输出参数到日志文件
- `_initialize_velocities()`: 从 Maxwell-Boltzmann 分布生成初始速度
- `run()`: 根据 ensemble 参数分发到具体系综

---

### 2. NVE 系综 (`ensemble/nve.py`)

**职责**：
- 使用 Velocity Verlet 积分器
- 监控能量守恒
- 记录轨迹和热力学量

**理论基础**：
- **守恒量**：总能量 E = KE + PE
- **积分器**：Velocity Verlet（辛算法，保持相空间体积）
- **能量漂移**：理想情况下应为零，实际由于数值误差会有小漂移

**代码框架**：
```python
import numpy as np
from ase import Atoms
from ..integrator import VelocityVerlet
from ..utils import calculate_temperature, calculate_kinetic_energy

class NVEEnsemble:
    """NVE (微正则) 系综"""

    def __init__(self, atoms: Atoms, output: str, params):
        self.atoms = atoms
        self.output = output
        self.params = params

        # 初始化积分器
        self.integrator = VelocityVerlet(
            atoms=atoms,
            timestep=params.timestep
        )

        # 初始化速度
        self.velocities = self._initialize_velocities()

        # 能量监控
        self.initial_energy = None
        self.max_energy_drift = 0.01  # Hartree

    def run(self):
        """主循环"""
        # 文件准备
        traj_file = self._prepare_files()

        # 初始能量
        self.initial_energy = self._calculate_total_energy()

        # 日志表头
        self._write_log_header()

        # 主循环
        for step in range(self.params.steps):
            # 1. 执行一步积分
            self.velocities = self.integrator.step(self.velocities)

            # 2. 计算热力学量
            kinetic = calculate_kinetic_energy(self.atoms, self.velocities)
            potential = self.atoms.get_potential_energy(force_consistent=True)
            total = kinetic + potential
            temperature = calculate_temperature(self.atoms, self.velocities)

            # 3. 能量漂移检查
            energy_drift = abs(total - self.initial_energy)
            if energy_drift > self.max_energy_drift:
                warning = f"WARNING: Energy drift {energy_drift:.6f} exceeds threshold\n"
                self._log_info([warning])

            # 4. 轨迹写入
            if step % self.params.traj_every == 0:
                self._write_trajectory(step, total)

            # 5. 日志输出
            if step % self.params.log_every == 0:
                self._write_log(step, temperature, kinetic, potential, total)

        # 最终统计
        self._write_summary()

    def _initialize_velocities(self):
        """从 Maxwell-Boltzmann 分布初始化"""
        kT = self.params.temperature * KELVIN_TO_HARTREE
        masses = self.atoms.get_masses() * AMU_TO_AU

        velocities = np.random.randn(len(self.atoms), 3)
        for i, mass in enumerate(masses):
            sigma = np.sqrt(kT / mass)
            velocities[i] *= sigma

        # 移除质心运动
        total_momentum = np.sum(masses[:, np.newaxis] * velocities, axis=0)
        velocities -= total_momentum / np.sum(masses)

        return velocities
```

---

### 3. NVT 系综 (`ensemble/nvt.py`)

**职责**：
- 在 NVE 基础上添加热浴
- 控制温度稳定在目标值
- 监控温度涨落

**理论基础**：
- **Langevin 动力学**：添加摩擦力和随机力
  ```
  m*dv/dt = F_potential - γ*m*v + R(t)
  ```
  - γ: 摩擦系数
  - R(t): 随机力（高斯白噪声）

- **Berendsen 热浴**：速度缩放
  ```
  v_new = v * sqrt(1 + dt/τ * (T_target/T_current - 1))
  ```

**代码框架**：
```python
from ..thermostat import LangevinThermostat, BerendsenThermostat

class NVTEnsemble:
    """NVT (正则) 系综"""

    def __init__(self, atoms: Atoms, output: str, params):
        self.atoms = atoms
        self.output = output
        self.params = params

        # 积分器
        self.integrator = VelocityVerlet(atoms, params.timestep)

        # 热浴选择
        if params.thermostat == 'langevin':
            self.thermostat = LangevinThermostat(
                atoms=atoms,
                temperature=params.temperature,
                friction=params.friction,
                timestep=params.timestep
            )
        elif params.thermostat == 'berendsen':
            self.thermostat = BerendsenThermostat(
                atoms=atoms,
                temperature=params.temperature,
                tau=params.tau_t,
                timestep=params.timestep
            )
        else:
            raise ValueError(f"Unknown thermostat: {params.thermostat}")

        self.velocities = self._initialize_velocities()

    def run(self):
        """主循环（与NVE类似但加入热浴）"""
        for step in range(self.params.steps):
            # 1. 半步积分
            self.velocities = self.integrator.half_step(self.velocities)

            # 2. 应用热浴
            self.velocities = self.thermostat.apply(self.velocities)

            # 3. 完成积分
            self.velocities = self.integrator.complete_step(self.velocities)

            # 4. 热力学量计算和输出
            # ... (同NVE) ...
```

---

### 4. Velocity Verlet 积分器 (`integrator/velocity_verlet.py`)

**职责**：
- 实现辛积分算法
- 更新位置和速度
- 调用 Calculator 计算力

**算法流程**：
```
1. v(t+dt/2) = v(t) + (F(t)/m) * dt/2
2. r(t+dt) = r(t) + v(t+dt/2) * dt
3. 计算 F(t+dt)
4. v(t+dt) = v(t+dt/2) + (F(t+dt)/m) * dt/2
```

**代码框架**：
```python
class VelocityVerlet:
    """Velocity Verlet 辛积分器"""

    # 单位转换常数
    FS_TO_AU = 41.341374575751
    AMU_TO_AU = 1822.888486209

    def __init__(self, atoms: Atoms, timestep: float):
        self.atoms = atoms
        self.timestep = timestep * self.FS_TO_AU  # fs → a.u.
        self.masses = atoms.get_masses() * self.AMU_TO_AU

    def step(self, velocities: np.ndarray) -> np.ndarray:
        """完整的一步积分"""
        dt = self.timestep
        masses = self.masses[:, np.newaxis]

        # 半步速度更新
        forces = self.atoms.get_forces()  # ← Calculator调用
        velocities += 0.5 * forces / masses * dt

        # 全步位置更新
        positions = self.atoms.get_positions()
        positions += velocities * dt
        self.atoms.set_positions(positions)

        # 重新计算力
        forces = self.atoms.get_forces()  # ← Calculator再次调用

        # 半步速度更新
        velocities += 0.5 * forces / masses * dt

        return velocities

    def half_step(self, velocities: np.ndarray) -> np.ndarray:
        """半步更新（用于分离热浴应用）"""
        dt = self.timestep
        masses = self.masses[:, np.newaxis]

        forces = self.atoms.get_forces()
        velocities += 0.5 * forces / masses * dt

        positions = self.atoms.get_positions()
        positions += velocities * dt
        self.atoms.set_positions(positions)

        return velocities

    def complete_step(self, velocities: np.ndarray) -> np.ndarray:
        """完成剩余的半步"""
        dt = self.timestep
        masses = self.masses[:, np.newaxis]

        forces = self.atoms.get_forces()
        velocities += 0.5 * forces / masses * dt

        return velocities
```

---

### 5. Langevin 热浴 (`thermostat/langevin.py`)

**职责**：
- 添加摩擦力和随机力
- 维持目标温度

**理论**：
```
F_langevin = -γ*m*v + sqrt(2*γ*m*kT/dt) * R
```
其中 R 是标准正态分布随机数

**代码框架**：
```python
class LangevinThermostat:
    """Langevin 动力学热浴"""

    KELVIN_TO_HARTREE = 3.1668114e-6
    AMU_TO_AU = 1822.888486209
    FS_TO_AU = 41.341374575751

    def __init__(self, atoms: Atoms, temperature: float,
                 friction: float, timestep: float):
        self.atoms = atoms
        self.temperature = temperature
        self.friction = friction  # 1/fs
        self.timestep = timestep * self.FS_TO_AU
        self.masses = atoms.get_masses() * self.AMU_TO_AU

        # 预计算常数
        self.kT = temperature * self.KELVIN_TO_HARTREE
        self.gamma_dt = friction * timestep

    def apply(self, velocities: np.ndarray) -> np.ndarray:
        """应用 Langevin 热浴"""
        dt = self.timestep
        masses = self.masses[:, np.newaxis]

        # 摩擦力项
        friction_force = -self.gamma_dt * velocities

        # 随机力项（高斯白噪声）
        random_force = np.random.randn(len(self.atoms), 3)
        random_force *= np.sqrt(2.0 * self.gamma_dt * self.kT / masses)

        # 更新速度
        velocities += friction_force + random_force

        return velocities
```

---

### 6. Berendsen 热浴 (`thermostat/berendsen.py`)

**职责**：
- 速度缩放控温
- 简单且数值稳定

**理论**：
```
λ = sqrt(1 + dt/τ * (T_target/T_current - 1))
v_new = v * λ
```

**代码框架**：
```python
class BerendsenThermostat:
    """Berendsen 速度缩放热浴"""

    KELVIN_TO_HARTREE = 3.1668114e-6
    AMU_TO_AU = 1822.888486209
    FS_TO_AU = 41.341374575751

    def __init__(self, atoms: Atoms, temperature: float,
                 tau: float, timestep: float):
        self.atoms = atoms
        self.temperature = temperature
        self.tau = tau * self.FS_TO_AU  # fs → a.u.
        self.timestep = timestep * self.FS_TO_AU
        self.masses = atoms.get_masses() * self.AMU_TO_AU

    def apply(self, velocities: np.ndarray) -> np.ndarray:
        """应用 Berendsen 热浴"""
        from ..utils import calculate_temperature

        current_temp = calculate_temperature(self.atoms, velocities)

        # 计算缩放因子
        lambda_factor = np.sqrt(
            1.0 + self.timestep / self.tau *
            (self.temperature / current_temp - 1.0)
        )

        # 缩放速度
        velocities *= lambda_factor

        return velocities
```

---

### 7. 工具函数 (`utils.py`)

**职责**：
- 温度计算
- 动能计算
- 单位转换

**代码框架**：
```python
import numpy as np
from ase import Atoms

# 单位转换常数
KELVIN_TO_HARTREE = 3.1668114e-6
HARTREE_TO_KELVIN = 1.0 / KELVIN_TO_HARTREE
AMU_TO_AU = 1822.888486209
FS_TO_AU = 41.341374575751
BOHR_TO_ANGSTROM = 0.529177249

def calculate_temperature(atoms: Atoms, velocities: np.ndarray) -> float:
    """计算瞬时温度

    T = 2*KE / (N_dof * k_B)
    N_dof = 3N - 3 (减去质心平动自由度)
    """
    masses = atoms.get_masses() * AMU_TO_AU
    kinetic = 0.5 * np.sum(masses[:, np.newaxis] * velocities**2)

    n_atoms = len(atoms)
    n_dof = 3 * n_atoms - 3  # 减去质心平动

    temperature = 2.0 * kinetic / (n_dof * KELVIN_TO_HARTREE)
    return temperature

def calculate_kinetic_energy(atoms: Atoms, velocities: np.ndarray) -> float:
    """计算动能 (Hartree)"""
    masses = atoms.get_masses() * AMU_TO_AU
    kinetic = 0.5 * np.sum(masses[:, np.newaxis] * velocities**2)
    return kinetic

def initialize_velocities(atoms: Atoms, temperature: float,
                         remove_com: bool = True) -> np.ndarray:
    """从 Maxwell-Boltzmann 分布初始化速度"""
    kT = temperature * KELVIN_TO_HARTREE
    masses = atoms.get_masses() * AMU_TO_AU
    n_atoms = len(atoms)

    # 生成随机速度
    velocities = np.random.randn(n_atoms, 3)

    # 缩放到目标温度
    for i, mass in enumerate(masses):
        sigma = np.sqrt(kT / mass)
        velocities[i] *= sigma

    # 移除质心运动
    if remove_com:
        total_momentum = np.sum(masses[:, np.newaxis] * velocities, axis=0)
        total_mass = np.sum(masses)
        velocities -= total_momentum / total_mass

    return velocities

def write_xyz_frame(file_handle, atoms: Atoms, energy: float,
                   frame_number: int):
    """写入单帧XYZ格式"""
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()

    file_handle.write(f"{len(symbols)}\n")
    file_handle.write(f"Frame {frame_number}  Energy = {energy:.10f} Hartree\n")

    for symbol, (x, y, z) in zip(symbols, positions):
        file_handle.write(f"{symbol:2s} {x:15.8f} {y:15.8f} {z:15.8f}\n")
```

---

### 8. MD 日志输出 (`logger.py`)

**职责**：
- 格式化 MD 输出
- 轨迹文件写入
- 热力学数据记录

**代码框架**：
```python
import os

class MDLogger:
    """MD 模拟日志管理器"""

    def __init__(self, output_file: str, verbose: int = 1):
        self.output = output_file
        self.verbose = verbose

        base, _ = os.path.splitext(output_file)
        self.traj_file = base + "_md_traj.xyz"
        self.thermo_file = base + "_md_thermo.dat"
        self.summary_file = base + "_md_summary.txt"

        # 打开文件句柄
        self.traj_handle = None
        self.thermo_handle = None

    def open_files(self):
        """打开输出文件"""
        self.traj_handle = open(self.traj_file, 'w')
        self.thermo_handle = open(self.thermo_file, 'w')

        # 写入热力学文件表头
        self.thermo_handle.write(
            f"# {'Step':>8} {'Time(fs)':>12} {'Temp(K)':>10} "
            f"{'KE(Eh)':>14} {'PE(Eh)':>14} {'TE(Eh)':>14}\n"
        )

    def close_files(self):
        """关闭文件句柄"""
        if self.traj_handle:
            self.traj_handle.close()
        if self.thermo_handle:
            self.thermo_handle.close()

    def write_trajectory_frame(self, atoms, energy, frame_number):
        """写入轨迹帧"""
        from .utils import write_xyz_frame
        write_xyz_frame(self.traj_handle, atoms, energy, frame_number)

    def write_thermo(self, step, time, temp, ke, pe, te):
        """写入热力学数据"""
        self.thermo_handle.write(
            f"  {step:8d} {time:12.3f} {temp:10.2f} "
            f"{ke:14.8f} {pe:14.8f} {te:14.8f}\n"
        )
        self.thermo_handle.flush()  # 确保实时写入

    def write_log(self, step, time, temp, ke, pe, te):
        """写入主日志"""
        if self.verbose == 0:
            # 简洁模式
            msg = (f"{step:8d} {time:10.3f} {temp:8.2f} "
                  f"{ke:12.6f} {pe:12.6f} {te:12.6f}\n")
        else:
            # 详细模式
            msg = (
                f"\n{'-' * 70}\n"
                f"{'MD Step: ' + str(step):^70}\n\n"
                f"Time:              {time:12.3f} fs\n"
                f"Temperature:       {temp:12.2f} K\n"
                f"Kinetic Energy:    {ke:12.6f} Hartree\n"
                f"Potential Energy:  {pe:12.6f} Hartree\n"
                f"Total Energy:      {te:12.6f} Hartree\n"
            )

        with open(self.output, 'a') as f:
            f.write(msg)

    def write_summary(self, total_steps, total_time, initial_energy,
                     final_energy, avg_temp, std_temp):
        """写入最终统计摘要"""
        energy_drift = abs(final_energy - initial_energy)

        summary = [
            "\n" + "=" * 70 + "\n",
            "MD Simulation Summary\n",
            "=" * 70 + "\n",
            f"Total steps:        {total_steps}\n",
            f"Total time:         {total_time:.3f} fs\n",
            f"Initial energy:     {initial_energy:.10f} Hartree\n",
            f"Final energy:       {final_energy:.10f} Hartree\n",
            f"Energy drift:       {energy_drift:.10f} Hartree\n",
            f"Average temp:       {avg_temp:.2f} K\n",
            f"Temp std dev:       {std_temp:.2f} K\n",
            f"\nOutput files:\n",
            f"  Trajectory:       {self.traj_file}\n",
            f"  Thermodynamics:   {self.thermo_file}\n",
            f"  Summary:          {self.summary_file}\n",
            "=" * 70 + "\n",
        ]

        # 写入主日志
        with open(self.output, 'a') as f:
            f.writelines(summary)

        # 写入单独摘要文件
        with open(self.summary_file, 'w') as f:
            f.writelines(summary)
```

---

## 文件修改清单

### 新增文件

```
maple/function/dispatcher/md/
├── __init__.py                         # NEW: 模块导出
├── md.py                               # NEW: MD主调度器
├── ensemble/
│   ├── __init__.py                     # NEW
│   ├── nve.py                          # NEW: NVE系综
│   ├── nvt.py                          # NEW: NVT系综
│   └── ensemble_base.py                # NEW: 基类（预留）
├── integrator/
│   ├── __init__.py                     # NEW
│   ├── velocity_verlet.py              # NEW: VV积分器
│   └── integrator_base.py              # NEW: 基类（预留）
├── thermostat/
│   ├── __init__.py                     # NEW
│   ├── langevin.py                     # NEW: Langevin热浴
│   ├── berendsen.py                    # NEW: Berendsen热浴
│   └── thermostat_base.py              # NEW: 基类（预留）
├── utils.py                            # NEW: 工具函数
└── logger.py                           # NEW: 日志管理
```

### 修改现有文件

#### 1. `/home/user/software/MAPLE/maple/function/read/command_control.py`

**修改位置**: 第 10-17 行（SUPPORTED_TASKS 和 DEFAULTS）

**修改内容**:
```python
# 添加 MD 到支持的任务类型
SUPPORTED_TASKS = {"sp", "opt", "ts", "scan", "freq", "irc", "md"}  # ← 添加 md

# 添加 MD 默认参数
DEFAULTS = {
    "sp": {},
    "opt": {"method": "lbfgs"},
    "scan": {"method": "lbfgs"},
    # ... 其他现有默认值 ...

    # NEW: MD 默认参数
    "md": {
        "ensemble": "nve",
        "timestep": 0.5,
        "steps": 10000,
        "temperature": 300.0,
        "thermostat": "langevin",
        "friction": 0.01,
        "tau_t": 100.0,
        "traj_every": 10,
        "log_every": 100,
        "verbose": 1
    }
}
```

#### 2. `/home/user/software/MAPLE/maple/function/dispatcher/dispatcher.py`

**修改位置**: `__call__` 方法（约第 40-80 行）

**修改内容**:
```python
def __call__(self, commandcontrol, jobtype: int, atoms, output: str, extra: dict = None):
    # ... 现有代码 ...

    # NEW: 添加 MD 调度
    elif jobtype == 'md':
        from .md import MDDispatcher
        md_job = MDDispatcher(
            atoms=atoms,
            output=output,
            paras=commandcontrol.params
        )
        md_job.run()

    # ... 其他任务类型 ...
```

#### 3. `/home/user/software/MAPLE/maple/function/dispatcher/__init__.py`

**修改位置**: 导入部分

**修改内容**:
```python
from .optimization import Optimization
from .ts import TransitionState
# ... 其他导入 ...

# NEW: 导入 MD
from .md import MDDispatcher
```

---

## 实现步骤

### ✅ 阶段 1: 基础设施（完成于 2026-01-30）

**目标**: 搭建模块框架和工具函数

1. ✅ 创建目录结构 - **完成**
2. ✅ 实现 `utils.py`（温度、动能计算等）- **完成** (380 行)
3. ⏸️ 实现 `logger.py`（日志管理）- **推迟到 Phase 2**
4. ✅ 实现 `velocity_verlet.py`（积分器）- **完成** (280 行)
5. ✅ 创建所有 `__init__.py` 文件 - **完成** (4 个)

**已完成文件** (7 个):
- `maple/function/dispatcher/md/__init__.py`
- `maple/function/dispatcher/md/utils.py` ⭐
- `maple/function/dispatcher/md/integrator/__init__.py`
- `maple/function/dispatcher/md/integrator/velocity_verlet.py` ⭐
- `maple/function/dispatcher/md/ensemble/__init__.py`
- `maple/function/dispatcher/md/thermostat/__init__.py`

**代码统计**:
- 总行数: ~660 行
- 核心代码: 2 个文件（utils.py + velocity_verlet.py）
- 支持代码: 5 个 __init__.py

**验证方法**:
```python
# 测试温度计算
from maple.function.dispatcher.md import calculate_temperature, initialize_velocities
from ase.build import molecule

atoms = molecule('H2O')
velocities = initialize_velocities(atoms, temperature=300.0)
temp = calculate_temperature(atoms, velocities)
print(f"Temperature: {temp:.2f} K")  # 应约等于 300K

# 测试积分器（需要在 Phase 2 中完成）
# from maple.function.dispatcher.md.integrator import integrate_nve
# ...
```

### ⏳ 阶段 2: NVE 系综（当前阶段）

**目标**: 实现完整的 NVE 模拟功能

#### 待实现任务

1. ⬜ 实现 `logger.py` - MD 专用日志管理器
   - MDLogger 类
   - 热力学量输出 (thermo.dat)
   - XYZ 轨迹管理
   - 最终摘要输出

2. ⬜ 实现 `nve.py` - NVE 系综类
   - NVEEnsemble 类
   - 使用 VelocityVerlet 积分器
   - 能量守恒监控
   - 轨迹记录逻辑

3. ⬜ 实现 `md.py` - MD 主调度器
   - MDDispatcher 类（继承 JobABC）
   - MDParams dataclass 定义
   - 速度初始化逻辑
   - 系综选择和分发

4. ⬜ 修改 `command_control.py`
   - 添加 'md' 到 SUPPORTED_TASKS
   - 添加 MD 默认参数到 DEFAULTS

5. ⬜ 修改 `dispatcher.py`
   - 添加 MD 任务路由
   - 调用 MDDispatcher

6. ⬜ 修改 `dispatcher/__init__.py`
   - 导出 MDDispatcher

7. ⬜ 集成测试（小分子 NVE 轨迹）
   - 创建测试输入文件
   - 运行 NVE 模拟
   - 验证能量守恒

#### Phase 2 验证计划

**测试输入文件** (`test_nve.inp`):
```bash
#model=ani2x
#md(ensemble=nve, timestep=1.0, steps=1000, temperature=300,
    traj_every=10, log_every=100)
#device=cpu

O  0.000000  0.000000  0.000000
H  0.957200  0.000000  0.000000
H -0.239987  0.927663  0.000000
```

**运行测试**:
```bash
maple test_nve.inp
```

**预期输出文件**:
- `test_nve.out` - 主日志文件
- `test_nve_md_traj.xyz` - XYZ 轨迹（每 10 步）
- `test_nve_md_thermo.dat` - 热力学数据（每步）
- `test_nve_md_summary.txt` - 最终统计

**验证标准**:
```bash
# 检查能量守恒（能量漂移 < 0.01 Hartree）
python scripts/check_energy_conservation.py test_nve_md_thermo.dat

# 检查温度分布
python scripts/analyze_temperature.py test_nve_md_thermo.dat
```

**成功标准**:
- ✅ 模拟正常运行完成 1000 步
- ✅ 总能量漂移 < 1%
- ✅ XYZ 轨迹可用 VMD/Ovito 可视化
- ✅ 热力学数据格式正确

### 阶段 3: 热浴模块（第 5-6 天）

**目标**: 实现温度控制

1. ✅ 实现 `langevin.py`
2. ✅ 实现 `berendsen.py`
3. ✅ 实现 `nvt.py`
4. ✅ 集成测试（NVT 温度分布）

**验证**:
```bash
# 测试 Langevin NVT
#model=ani2x
#md(ensemble=nvt, thermostat=langevin, temperature=300,
    timestep=1.0, steps=5000, friction=0.01)

# 运行
maple test_nvt_langevin.inp

# 检查温度分布（平均温度 ≈ 300K, 标准差 < 20K）
python scripts/analyze_temperature.py test_nvt_md_thermo.dat
```

### 阶段 4: 优化和文档（第 7 天）

1. ✅ 性能优化（减少重复计算）
2. ✅ 添加 docstring 文档
3. ✅ 编写用户手册
4. ✅ 创建示例输入文件

---

## 测试验证

### 单元测试

**文件**: `maple/function/dispatcher/md/tests/`

```python
# test_integrator.py
def test_velocity_verlet_energy_conservation():
    """测试 VV 积分器的能量守恒（简谐振子）"""
    # 构建简谐势
    # 运行 1000 步
    # 检查能量漂移 < 1e-6

# test_temperature.py
def test_maxwell_boltzmann_distribution():
    """测试速度初始化符合 MB 分布"""
    # 生成 10000 个原子的速度
    # 检查温度分布是否符合理论

# test_thermostat.py
def test_langevin_temperature_control():
    """测试 Langevin 热浴的温度控制"""
    # 运行 NVT 模拟
    # 检查平均温度和涨落
```

### 集成测试

**测试用例 1: 小分子 NVE**
```
系统: H2O 分子
模型: ANI-2x
参数: dt=0.5fs, 10000 步
预期: 能量漂移 < 0.01 Hartree
```

**测试用例 2: 小分子 NVT (Langevin)**
```
系统: CH4 分子
模型: ANI-2x
参数: T=300K, friction=0.01, dt=1.0fs, 5000 步
预期: <T> = 300±10K
```

**测试用例 3: 周期性体系 NVE**
```
系统: Cu(111) 表面 + 吸附分子
模型: UMA (task=oc20)
PBC: (18.19, 18.19, 24.90, 90, 90, 90)
参数: dt=1.0fs, 5000 步
预期: 能量守恒，温度稳定
```

### 性能基准

**目标**:
- 1000 原子体系，1000 步 MD < 10 分钟（GPU）
- 内存占用 < 2GB
- 轨迹文件大小合理（每 10 步保存一帧）

---

## 未来扩展预留

### 1. NPT 系综（恒温恒压）

**新增模块**:
```
maple/function/dispatcher/md/ensemble/npt.py
maple/function/dispatcher/md/barostat/
    ├── berendsen_barostat.py
    ├── parrinello_rahman.py
    └── barostat_base.py
```

**参数扩展**:
```python
@dataclass
class MDParams:
    # ... 现有参数 ...
    pressure: float = 1.0        # atm
    barostat: str = 'berendsen'  # berendsen, parrinello-rahman
    tau_p: float = 1000.0        # 压力耦合时间常数 (fs)
```

**接口设计**:
```python
class NPTEnsemble:
    def __init__(self, atoms, output, params):
        self.integrator = VelocityVerlet(...)
        self.thermostat = LangevinThermostat(...)
        self.barostat = BerendsenBarostat(...)  # NEW
```

### 2. Nosé-Hoover 热浴

**优点**: 正则系综严格采样（不像 Berendsen）

**新增模块**:
```python
# maple/function/dispatcher/md/thermostat/nose_hoover.py
class NoseHooverThermostat:
    def __init__(self, atoms, temperature, tau):
        self.Q = self._calculate_thermal_mass(tau)  # 热浴质量
        self.xi = 0.0  # 热浴自由度
        self.v_xi = 0.0  # 热浴速度
```

### 3. 约束动力学（SHAKE/RATTLE）

**用途**: 固定键长/键角（如 H-O 键）

**新增模块**:
```python
# maple/function/dispatcher/md/constraints/
class SHAKEConstraint:
    def apply(self, positions, velocities):
        # 迭代求解约束
        pass
```

### 4. 增强采样方法

**潜在扩展**:
- **Umbrella Sampling**: 沿反应坐标施加偏置势
- **Metadynamics**: 填充自由能景观
- **Replica Exchange MD (REMD)**: 并行温度交换

**接口设计**:
```python
# maple/function/dispatcher/md/enhanced_sampling/
class UmbrellaSampling:
    def __init__(self, atoms, collective_variable, force_constant):
        self.cv = collective_variable
        self.k = force_constant

    def apply_bias(self, atoms):
        # 计算偏置力
        pass
```

### 5. 轨迹分析工具

**新增模块**:
```python
# maple/function/dispatcher/md/analysis/
class TrajectoryAnalyzer:
    def radial_distribution_function(self, traj_file):
        """径向分布函数 g(r)"""
        pass

    def mean_square_displacement(self, traj_file):
        """均方位移 MSD"""
        pass

    def autocorrelation(self, property_series):
        """自相关函数"""
        pass
```

### 6. 并行化和GPU加速

**策略**:
- ASE Atoms 和 Calculator 已支持 GPU（通过 PyTorch）
- 轨迹写入异步化（避免 I/O 阻塞）
- 多副本并行（REMD）

**预留接口**:
```python
class MDDispatcher:
    def run_parallel(self, n_replicas: int):
        """并行运行多个副本"""
        pass
```

### 7. 重启和检查点

**功能**:
- 保存 MD 状态（位置、速度、时间）
- 从检查点恢复模拟

**文件格式**:
```python
# checkpoint.pkl
{
    'step': 5000,
    'time': 2500.0,  # fs
    'positions': np.array(...),
    'velocities': np.array(...),
    'thermostat_state': {...},
    'rng_state': np.random.get_state()
}
```

**接口**:
```python
@dataclass
class MDParams:
    restart_file: Optional[str] = None
    save_checkpoint_every: int = 1000

class MDDispatcher:
    def save_checkpoint(self, step, filename):
        pass

    def load_checkpoint(self, filename):
        pass
```

---

## 接口设计原则

为确保未来扩展性，遵循以下设计原则：

### 1. 开闭原则（OCP）
- 对扩展开放，对修改关闭
- 新增热浴/积分器不修改现有代码

### 2. 依赖倒置（DIP）
- Ensemble 依赖抽象接口（ThermostatBase），而非具体实现

```python
# 好的设计
class NVTEnsemble:
    def __init__(self, thermostat: ThermostatBase):
        self.thermostat = thermostat  # 接受任何热浴实现

# 而非
class NVTEnsemble:
    def __init__(self, friction: float):
        self.thermostat = LangevinThermostat(friction)  # 硬编码
```

### 3. 单一职责（SRP）
- Integrator 只负责积分
- Thermostat 只负责温度控制
- Logger 只负责输出

### 4. 组合优于继承
- Ensemble 组合 Integrator + Thermostat
- 避免深层继承树

---

## 性能优化考虑

### 1. 减少 Calculator 调用

**问题**: `atoms.get_forces()` 是最昂贵的操作

**优化**:
```python
# 不好：重复调用
forces1 = atoms.get_forces()
energy = atoms.get_potential_energy()
forces2 = atoms.get_forces()  # ← 重复计算！

# 好：缓存结果
forces = atoms.get_forces()
energy = atoms.calc.results['energy']  # ← 使用缓存
```

### 2. 向量化操作

**使用 NumPy 向量化**:
```python
# 不好：循环
for i in range(n_atoms):
    velocities[i] += forces[i] / masses[i] * dt

# 好：向量化
velocities += forces / masses[:, np.newaxis] * dt
```

### 3. 异步 I/O

**避免轨迹写入阻塞主循环**:
```python
import threading

class AsyncTrajectoryWriter:
    def __init__(self, filename):
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self._writer_loop)
        self.thread.start()

    def write(self, atoms, energy):
        self.queue.put((atoms.copy(), energy))

    def _writer_loop(self):
        while True:
            atoms, energy = self.queue.get()
            # 写入文件
```

---

## 总结

本计划提供了完整的 MD 模块实现方案，包括：

✅ **清晰的架构设计**：模块化、可扩展
✅ **详细的代码框架**：直接可用的实现模板
✅ **完整的文件清单**：新增和修改的文件
✅ **渐进式实现步骤**：分阶段开发和测试
✅ **全面的未来预留**：NPT、约束、增强采样等

下一步：根据计划开始实施第一阶段（基础设施）。
