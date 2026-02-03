# MAPLE 代码库总结文档

**生成日期**: 2026-01-30
**代码库位置**: /home/user/software/MAPLE
**版本**: 0.1.0

---

## 目录

1. [MAPLE 概述](#maple-概述)
2. [UMA Calculator 详解](#uma-calculator-详解)
3. [PBC (周期边界条件) 设置详解](#pbc-周期边界条件-设置详解)

---

# MAPLE 概述

## 1.1 什么是 MAPLE

**MAPLE** = **MAchine-learning Potential for Landscape Exploration**
（基于机器学习势能面的分子景观探索工具）

MAPLE 是匹兹堡大学开发的基于 Python 的计算化学工具包，利用机器学习势能函数进行高效的分子模拟。它将 ML 势能函数与传统量子化学工作流程相结合，提供基于 ASE 的界面用于分子性质计算、结构优化、过渡态搜索和反应路径分析。

**核心目的**: 用快速的 ML 势能函数替代昂贵的量子化学计算，同时保持熟悉的输入格式和全面的算法能力。

**许可证**: BSD 3-Clause + Creative Commons BY（学术使用）

---

## 1.2 主要功能和能力

| 类别 | 方法 |
|------|------|
| **几何优化** | LBFGS, RFO, SD, DIIS |
| **过渡态搜索** | NEB, CI-NEB, PRFO, BPRFO, Dimer, String/GSM, AFIR, DescAFIR, AutoNEB |
| **反应路径** | IRC (Gonzalez-Schlegel 方法) |
| **分析** | 频率分析（质量加权）、势能面扫描、单点能计算 |
| **ML 势能函数** | AIMNet2, ANI (1x, 1ccx, 1xnr, 2x), MACE (MACE-OFF23s/m/l), EGRET, UMA |
| **物理修正** | DFT-D4 色散修正, GBSA 隐式溶剂化, 电荷修正 (QEq, EEq) |

---

## 1.3 目录结构

```
/home/user/software/MAPLE/
├── maple/                          # 主 Python 包 (84 个 .py 文件)
│   ├── main.py                     # CLI 入口点
│   ├── function/
│   │   ├── engine.py               # 核心协调器
│   │   ├── read/                   # 输入解析子系统
│   │   │   ├── input_reader.py     # 主输入文件解析器
│   │   │   ├── command_control.py  # 分层参数系统
│   │   │   ├── header/             # 头部指令解析
│   │   │   └── filereader/         # XYZ 坐标文件读取器
│   │   ├── calculator/             # ML 势能函数计算器（工厂模式）
│   │   │   ├── set_calculator.py   # 计算器工厂
│   │   │   ├── aimnet/             # AIMNet2 实现
│   │   │   ├── ani/                # ANI 实现
│   │   │   ├── mace/               # MACE 实现
│   │   │   ├── uma/                # UMA 实现 ⭐
│   │   │   ├── model/              # 预训练模型权重 (~550MB)
│   │   │   └── extra_correction/   # D4, GBSA, QEq 修正
│   │   ├── dispatcher/             # 任务路由和算法实现
│   │   │   ├── dispatcher.py       # 主任务路由器
│   │   │   ├── optimization/       # 几何优化
│   │   │   ├── ts/                 # 过渡态方法
│   │   │   ├── irc/                # IRC 计算
│   │   │   ├── frequency/          # 频率分析
│   │   │   ├── scan/               # 势能面扫描
│   │   │   └── sp/                 # 单点能
│   │   └── utility/                # 工具类
│   └── __init__.py
├── example/                        # 测试用例和示例 (8 个测试场景)
│   ├── opt/lbfgs/                  # LBFGS 优化测试
│   │   ├── pbc_cu.inp              # PBC 示例 ⭐
│   │   └── pbc_cu.xyz
│   ├── ts/neb/                     # NEB 过渡态测试
│   └── ...
├── ARCHITECTURE.md                 # 详细技术架构文档
├── README.md                       # 用户文档
└── pyproject.toml                  # 包配置
```

---

## 1.4 可用的 ML 势能函数

| 势能函数 | 变体 | 目标系统 | 特性 |
|---------|------|----------|------|
| **ANI** | ani2x, ani1x, ani1ccx, ani1xnr | CHNO 分子 | 快速，对有机物精确 |
| **AIMNet2** | aimnet2, aimnet2nse | 有机分子 | 支持电荷/多重度 |
| **MACE** | maceoff23s/m/l, egret, maceomol | 通用材料 | 消息传递等变 |
| **UMA** | uma (可配置: omol/oc20/omat/omc/odac) | 通用材料 | 多任务学习 ⭐ |

---

## 1.5 输入文件格式

```
# 头部指令（不区分大小写）
#model=<uma|aimnet2|ani|mace>
#<jobtype>(method=<method>,param=value)
#device=<gpu0|gpu1|cpu>

# 坐标（两种选项）
C   0.000   0.000   0.000      # 选项 1: 内联
H   1.089   0.000   0.000

# 或
XYZ /path/to/structure.xyz     # 选项 2: 外部文件
```

---

## 1.6 工作流程

```
main.py (CLI)
    ↓
engine()
    ├─→ InputReader: 解析输入文件
    ├─→ SetCalculator: 初始化 ML 势能函数
    └─→ Dispatcher: 路由到任务处理器
        ├─→ opt (LBFGS/RFO/SD/DIIS)
        ├─→ ts (NEB/PRFO/Dimer/String/AFIR/等)
        ├─→ irc (Gonzalez-Schlegel)
        ├─→ freq (振动分析)
        ├─→ scan (势能面扫描)
        └─→ sp (单点能)
```

---

# UMA Calculator 详解

## 2.1 UMA 是什么

**UMA** = **Universal Materials Atom**（通用材料原子模型）

UMA 是一个多任务机器学习势能函数，由 FAIRChem (Meta) 开发，能够处理分子和材料体系。它是 MAPLE 支持的最灵活的势能函数之一，支持周期性边界条件和多种任务类型。

---

## 2.2 文件位置和类结构

**主实现文件**:
[`maple/function/calculator/uma/_uma_calculator.py`](maple/function/calculator/uma/_uma_calculator.py) (220 行)

**类继承关系**:
```python
FAIRChemCalculator (来自 fairchem.core)
    ↑
UMACalculator
```

---

## 2.3 UMACalculator 类详解

### 构造函数参数

```python
UMACalculator(
    device: torch.device,           # 'cuda' 或 'cpu'
    model: str = "uma",             # 模型名称（默认: "uma"）
    overrides: dict | None = None,  # 推理配置覆盖
    implicit: Literal["gbsa", "none"] = "gbsa",  # 隐式溶剂模型
    solvent: str = 'none',          # GBSA 溶剂名称
    task: str | None = None,        # UMA 任务类型（默认: "omol"）
    size: str | None = None,        # 模型大小（默认: "uma-s-1p1"）
)
```

### 关键属性

- `self.task_name`: 正在使用的 UMA 任务
- `self.model_name`: 特定的 UMA 模型大小
- `self.device`: PyTorch 设备
- `self.solvent_correction`: GBSA 模块（如果启用）
- `self.chargecalc`: QEq 电荷计算器（如果启用溶剂化）

---

## 2.4 核心方法

### 2.4.1 能量计算

```python
get_energy(atoms: Atoms) -> torch.Tensor
```

- 计算总能量（单位：Hartree）
- 自动包含溶剂修正（如果配置）
- 返回：在 self.device 上的 torch.Tensor

### 2.4.2 Hessian 计算（有限差分法）

```python
get_hessian(
    atoms: Atoms,
    delta: float = 0.002,           # 位移大小
    dtype: torch.dtype = torch.float64,  # 输出精度
) -> torch.Tensor
```

- 通过力的有限差分计算 Hessian 矩阵
- 形状：(3N, 3N) 张量
- 尊重 `FixAtoms` 约束（跳过固定原子）
- 固定原子返回零行
- 使用公式：`H = -∂F/∂x ≈ -(F(+δ) - F(-δ)) / (2δ)`

**代码位置**: [_uma_calculator.py:103-185](_uma_calculator.py#L103-L185)

### 2.4.3 能量和力计算

```python
calculate(atoms, properties=None, system_changes=None)
```

- 覆盖基类方法，转换单位 eV → Hartree
- 对能量和力应用溶剂修正
- 如果启用 GBSA，计算 QEq 电荷
- 将溶剂化贡献添加到结果字典

**关键单位转换**:
```python
EV2HARTREE = 1.0 / 27.211386245988  ≈ 0.036749
```

---

## 2.5 UMA 任务类型和模型大小

### 支持的 UMA 任务

| 任务 | 描述 | PBC 兼容性 |
|------|------|------------|
| `omol` | 分子系统（默认） | ❌ 否 |
| `oc20` | OpenCatalyst 2020 数据集 | ✅ 是 |
| `omat` | 材料数据库 | ✅ 是 |
| `omc` | 金属有机化合物 | ✅ 是 |
| `odac` | 领域自适应化合物 | ✅ 是 |

**代码位置**: [command_control.py:214](command_control.py#L214)

### 支持的模型大小

| 大小 | 全名 | 状态 |
|------|------|------|
| `uma-s-1p1` | 小型模型（默认） | 活跃 |
| `uma-m-1p1` | 中型模型 | 活跃 |

**验证逻辑位置**: [command_control.py:206-226](command_control.py#L206-L226)

---

## 2.6 输入文件语法

### 示例 1: 简单优化

```
#model=uma
#opt(method=lbfgs)
#device=gpu0

C      -0.77812600     -1.06756100      0.32105900
C       1.30255300      0.05212000     -0.02829900
H       0.50000000      0.00000000      0.00000000
```

### 示例 2: 带 UMA 参数

```
#model=uma(task=oc20, size=uma-m-1p1)
#opt(method=lbfgs)
#device=gpu0
#pbc(18.19160,18.19160,24.90230)

XYZ /path/to/structure.xyz
```

### 示例 3: 带溶剂化

```
#model=uma
#opt(lbfgs)
#device=gpu0
#solv(method=gbsa, implicit=water)

H      0.0  0.0  0.0
O      0.0  0.0  1.0
H      0.0  1.0  0.0
```

### 示例 4: 势能面扫描

```
#model=uma
#scan
#device=gpu0

C  0.0  0.0  0.0
O  1.5  0.0  0.0
...

S 1 2 -0.05 50      # 扫描: 原子 1-2, 步长=-0.05, 50 步
```

---

## 2.7 溶剂化集成 (GBSA)

### GBSA (广义 Born 表面积) 修正

**配置**:
```python
#solv(method=gbsa, implicit=<solvent>)
```

**支持的溶剂 (16 种)**:
- water (默认)
- acetone, acetonitrile
- benzene
- dichloromethane
- dimethylformamide (dmf)
- dimethyl sulfoxide (dmso)
- diethyl ether (ether)
- carbon disulfide (cs2)
- toluene
- tetrahydrofuran (thf)
- methanol
- n-hexane (nhexan)
- chloroform (trichloromethane)

**实现位置**:
[`maple/function/calculator/extra_correction/solvent/gbsa/`](maple/function/calculator/extra_correction/solvent/gbsa/)

### 组件

1. **GBSA 模块** ([gbsa.py](gbsa.py))
   - 广义 Born (GB) 极性项
   - OBC-II 有效 Born 半径模型
   - 几何依赖半径计算
   - 输入：坐标（Å）
   - 输出：能量（Hartree），力（Hartree/Å）

2. **QEq 电荷计算器** ([qeq.py](qeq.py))
   - Rappé and Goddard III 方法 (J. Phys. Chem. 1991)
   - 读取每个元素的参数（电负性、硬度、高斯半径）
   - GPU 兼容的 PyTorch 实现
   - 输出原子电荷

**GBSA 参数文件位置**:
[`maple/function/calculator/extra_correction/solvent/gbsa/data/`](maple/function/calculator/extra_correction/solvent/gbsa/data/)

每个溶剂有一个 `.dat` 文件，包含：
```
第 1 行: 8 个常数
  [0] eps        - 介电常数
  [1] molarMass  - 摩尔质量
  [2] refracIndex - 折射率
  [3] gamma
  [4] beta
  [5] born_scale
  [6] born_offset
  [7] reserved

第 2+ 行: 两个数组（array1, array2）
  array1: 元素偏移值
  array2: 元素 vdw_ref
```

### UMA 中的溶剂化流程

```python
if self.solvent_correction:
    # 1. 计算 QEq 电荷
    atoms.atomic_charges = self.chargecalc(atoms)

    # 2. 获取 GBSA 能量和力
    solvent_energy, solvent_force = self.solvent_correction.get_energy_and_force(atoms)

    # 3. 添加到结果
    self.results["energy"] += solvent_energy.item()
    self.results["forces"] += solvent_force  # numpy array
```

**代码位置**: [_uma_calculator.py:59-95](_uma_calculator.py#L59-L95)

---

## 2.8 与 MAPLE 其他组件的集成

### 计算器工厂 (set_calculator.py)

```python
if self.model in ['uma']:
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
```

**代码位置**: [set_calculator.py](set_calculator.py)

### 支持的任务类型

- `sp` - 单点能量计算
- `opt` - 几何优化 (LBFGS, RFO, CG)
- `freq` - 频率分析（使用 Hessian）
- `scan` - 势能面扫描
- `ts` - 过渡态搜索 (NEB, String, Dimer, AFIR 等)
- `irc` - 内禀反应坐标

---

## 2.9 特殊功能

### 2.9.1 带固定原子的 Hessian

- 通过 `FixAtoms` 约束识别固定原子
- 仅循环可移动原子
- 对固定自由度返回 (3N, 3N) 矩阵的零
- 用于优化和频率分析

**代码位置**: [_uma_calculator.py:103-185](_uma_calculator.py#L103-L185)

### 2.9.2 设备灵活性

```python
device = "cuda" if device.startswith("cuda") else "cpu"
```

- 自动处理 GPU 设备选择
- 如果 CUDA 不可用，回退到 CPU
- 所有计算移动到指定设备

### 2.9.3 模型大小选择

默认映射（如果未指定大小）：
```python
UMA_MODELS_MAP = {
    "uma-s-1p1": "uma-s-1p1",      # 小
    "uma-m-1p1": "uma-m-1p1",      # 中
    "uma": "uma-s-1p1"              # 默认为小
}
```

### 2.9.4 电荷支持

- UMA 支持带电分子
- 在输入文件中指定：`charge multiplicity` 行
- 存储在 `atoms.info['charge']` 和 `atoms.info['mult']`
- 注意：某些模型（ANI, MACE）不支持电荷/多重度

---

## 2.10 限制和约束

### 已知限制

1. **PBC 不兼容性**:
   - `task='omol'`（默认）不能使用 PBC
   - 如果两者一起使用，验证期间会引发错误
   - 对于周期性系统，必须使用材料任务（oc20, omat 等）

2. **优化约束**:
   - 每个优化任务只能有一个 Atoms 对象
   - 不能对 opt, scan, freq, irc, ts (PRFO/Newton) 使用 Molecules/list
   - 多结构任务仅适用于 TS 中的 NEB, String, Dimer 方法

3. **溶剂限制**:
   - 仅 16 种溶剂选项可用
   - GBSA 需要原子电荷（如果全为零则失败）
   - 已移除非极性项（仅极性项）

4. **模型可用性**:
   - 依赖于 FAIRChem/Meta 的预训练模型可用性
   - 必须使用 `fairchem-core` 库加载模型

5. **Hessian 计算**:
   - 有限差分方法（非解析）
   - 默认 delta=0.002 Å（可调）
   - 对大型系统计算成本高

---

## 2.11 输入验证

**验证位置**: [command_control.py:206-248](command_control.py#L206-L248)

- 模型验证：检查 SUPPORTED_MODELS 列表
- 任务验证：限制为 {omol, oc20, omat, omc, odac}
- 大小验证：标准化（删除 '-', '_'）以进行比较
- PBC 验证：必须是 3 个正浮点数，与 omol 不兼容
- 设备验证：如果可用，自动选择 GPU，否则回退到 CPU

---

# PBC (周期边界条件) 设置详解

## 3.1 PBC 在代码中的实现位置

### 核心实现文件

1. **命令解析和验证**:
   [`maple/function/read/command_control.py:102-248`](maple/function/read/command_control.py#L102-L248)

2. **PBC 应用到原子对象**:
   [`maple/function/read/input_reader.py:320,422-449,511-518`](maple/function/read/input_reader.py#L320)

3. **UMA 计算器实现**:
   [`maple/function/calculator/uma/_uma_calculator.py`](maple/function/calculator/uma/_uma_calculator.py)

4. **计算器工厂**:
   [`maple/function/calculator/set_calculator.py`](maple/function/calculator/set_calculator.py)

5. **MACE 计算器单元处理**:
   [`maple/function/calculator/mace/_mace_calculator.py`](maple/function/calculator/mace/_mace_calculator.py)
   [`maple/function/calculator/mace/_mace_general_calculator.py`](maple/function/calculator/mace/_mace_general_calculator.py)

---

## 3.2 PBC 配置方法

### 输入文件语法

PBC 使用设置部分的单个命令指定：

```
#pbc(X, Y, Z)
或
#pbc=(X, Y, Z)
```

其中 X, Y, Z 是正交（立方/矩形）单元格的单元格尺寸（单位：Å）。

### 示例（来自 pbc_cu.inp）

```
#model=uma
#opt(method=lbfgs)
#device=gpu0
#pbc(18.19160,18.19160,24.90230)

XYZ 0 1 /home/user/software/MAPLE/example/opt/lbfgs/pbc_cu.xyz
```

**文件位置**: [`example/opt/lbfgs/pbc_cu.inp`](example/opt/lbfgs/pbc_cu.inp)

### 解析细节

**代码位置**: [command_control.py:102-118](command_control.py#L102-L118)

- 使用正则表达式模式：`#\s*pbc\s*(?:=\s*)?(?:\((.*)?\))`
- 期望恰好 3 个浮点值（X, Y, Z）
- 使用 `float()` 解析值，支持科学计数法
- 所有值必须为正 (>0)

---

## 3.3 PBC 设置可用选项和工作原理

### PBC 应用

**代码位置**: [input_reader.py:422-449](input_reader.py#L422-L449)

当指定 PBC 时，代码对 ASE Atoms 对象执行三个操作：

1. **启用周期性边界**:
   ```python
   atoms.set_pbc([True, True, True])
   ```

2. **设置正交单元格**:
   ```python
   cell = [[X, 0, 0],
           [0, Y, 0],
           [0, 0, Z]]
   atoms.set_cell(cell)
   ```

3. **记录 PBC 应用**:
   - 写入输出文件的确认消息
   - 记录单元格尺寸以供验证

### 当前限制

- **仅支持正交单元格**（无倾斜/三斜单元格）
- 单元格始终是轴对齐的（对角矩阵形式）
- 在坐标解析期间应用，在任何计算之前

### 应用于多种结构类型

- 内联原子坐标
- 轨迹（XYZTRAJ 文件）- 应用于每一帧
- 外部 XYZ 文件

---

## 3.4 PBC 与计算器的交互（特别是 UMA）

### UMA 计算器集成

UMA 计算器是专门设计考虑到 PBC 支持的：

**文件**: [`maple/function/calculator/uma/_uma_calculator.py`](maple/function/calculator/uma/_uma_calculator.py)

- 扩展 fairchem-core 库的 FAIRChemCalculator
- 默认任务：`'omol'`（分子系统 - **与 PBC 不兼容**）
- 支持 PBC 的任务：`'oc20'`, `'omat'`, `'omc'`, `'odac'`（材料/晶体系统）
- 支持的大小：`'uma-s-1p1'`（小）, `'uma-m-1p1'`（中）

### UMA 模型参数化

**代码位置**: [command_control.py:206-226](command_control.py#L206-L226)

```
#model=uma(task=oc20, size=uma-m-1p1)
```

支持的 UMA 任务及其 PBC 兼容性：
- `omol`: 分子系统 - **不支持 PBC** ❌
- `oc20`: 材料/晶体 - **支持 PBC** ✅
- `omat`: 材料 - **支持 PBC** ✅
- `omc`: 材料/催化 - **支持 PBC** ✅
- `odac`: 直接吸附复合物 - **支持 PBC** ✅

### MACE 计算器

MACE 计算器（maceoff23s/m/l, egret, maceomol）：
- 当前在半径图构建中不实现 PBC 支持
- 使用 `_radius_graph_no_pbc()` 函数（[_mace_calculator.py:40-53](_mace_calculator.py#L40-L53)）
- 图构建时不考虑周期性镜像

### 验证

**代码位置**: [command_control.py:240-248](command_control.py#L240-L248)

PBC 验证防止不兼容的配置：

```python
if model_name == 'uma' and task_param == 'omol':
    raise ValueError("PBC is incompatible with UMA task='omol'")
```

---

## 3.5 示例文件和使用

### 示例文件

**文件**: [`example/opt/lbfgs/pbc_cu.inp`](example/opt/lbfgs/pbc_cu.inp)

这演示了 Cu 表面/晶体优化：

- **模型**: UMA（默认：task=omol, size=uma-s-1p1）
- **任务**: opt（LBFGS 优化）
- **设备**: gpu0
- **PBC 单元格**: 18.1916 × 18.1916 × 24.9023 Å（正交）
- **结构**: 245 个 Cu + O 原子（可能是 CuO 表面）

### 相关文件

- [`example/opt/lbfgs/pbc_cu.xyz`](example/opt/lbfgs/pbc_cu.xyz): 输入结构（245 个原子）
- [`example/opt/lbfgs/pbc_cu.out`](example/opt/lbfgs/pbc_cu.out): 输出日志

### 输出日志信息

来自 pbc_cu.out：
```
Parsing # commands...
Global parameter: model = uma
Task set to 'opt'
Global parameter: device = gpu0
Global parameter: pbc = [18.1916, 18.1916, 24.9023]
```

---

## 3.6 架构流程

```
输入文件 (.inp)
    ↓
InputReader.settings_command()
    ↓
CommandControl.from_settings() - 解析 #pbc(X,Y,Z)
    ↓
InputReader.element_and_coordinates() - 将 PBC 应用于 atoms
    ├─ atoms.set_pbc([True, True, True])
    └─ atoms.set_cell([[X,0,0], [0,Y,0], [0,0,Z]])
    ↓
engine._mlp_initiator() - 创建计算器
    ↓
SetCalculator.set_calculator()
    ├─ 对于 UMA: UMACalculator(task=..., size=...)
    └─ 验证任务必须是 oc20/omat/omc/odac（不是 omol）
    ↓
Atoms 对象传递给带 PBC+cell 的 Job 调度器
```

---

## 3.7 关键验证规则

**验证位置**: [command_control.py:229-248](command_control.py#L229-L248)

1. **PBC 必须是 3 个值的列表** - `[X, Y, Z]`
2. **所有尺寸必须为正** - `> 0`
3. **对于 UMA 模型**:
   - 仅与以下任务兼容：`oc20`, `omat`, `omc`, `odac`
   - 与 `omol` 任务（分子系统）不兼容
4. **错误处理**:
   - 如果不是恰好 3 个值，抛出 ValueError
   - 如果任何值 ≤ 0，抛出 ValueError
   - 如果使用 UMA+omol 组合，抛出 ValueError

---

## 3.8 当前分支信息

**来自** `.collaboration_notes`:

- **分支**: `feat/pbc-md`
- **状态**: 开发中（截至 2026-01-26）
- **修改的文件**: 共 5 个文件（command_control.py, input_reader.py, set_calculator.py, uma/_uma_calculator.py, engine.py）

这是一个用于具有分子动力学能力的 PBC 支持的功能分支的一部分，尽管当前实现侧重于 PBC 单元格设置而不是 MD 动力学。

---

## 3.9 实际使用示例

### 示例 1: 分子优化（无 PBC）

```
#model=uma
#opt(method=lbfgs)
#device=gpu0

C  0.0  0.0  0.0
O  1.5  0.0  0.0
```

✅ 有效：默认 task=omol，无 PBC

### 示例 2: 周期性材料优化

```
#model=uma(task=oc20, size=uma-m-1p1)
#opt(method=lbfgs)
#device=gpu0
#pbc(20.0, 20.0, 20.0)

XYZ /path/to/crystal.xyz
```

✅ 有效：task=oc20 支持 PBC

### 示例 3: 错误配置

```
#model=uma
#opt(method=lbfgs)
#device=gpu0
#pbc(18.0, 18.0, 18.0)

C  0.0  0.0  0.0
```

❌ 错误：默认 task=omol 与 PBC 不兼容

**错误消息**: "PBC is incompatible with UMA task='omol'"

### 示例 4: 正确的周期性分子系统

```
#model=uma(task=omat)
#opt(method=lbfgs)
#device=gpu0
#pbc(25.0, 25.0, 25.0)

XYZ /path/to/periodic_structure.xyz
```

✅ 有效：task=omat 支持 PBC

---

## 3.10 技术实现细节

### PBC 数据结构

在 CommandControl 中解析后：
```python
self.pbc = [X, Y, Z]  # 浮点数列表
```

在 ASE Atoms 对象中应用后：
```python
atoms.pbc = [True, True, True]     # 布尔数组
atoms.cell = [[X,0,0],             # 3x3 矩阵
              [0,Y,0],
              [0,0,Z]]
```

### 正则表达式模式

**代码位置**: [command_control.py:102](command_control.py#L102)

```python
pattern = r'#\s*pbc\s*(?:=\s*)?(?:\((.*)?\))'
```

匹配：
- `#pbc(10, 20, 30)`
- `#pbc=(10, 20, 30)`
- `# pbc (10, 20, 30)`
- 等等

### 值解析

**代码位置**: [command_control.py:108-118](command_control.py#L108-L118)

```python
pbc_values = [float(x.strip()) for x in matched.split(',')]
if len(pbc_values) != 3:
    raise ValueError("PBC must have exactly 3 values")
if any(v <= 0 for v in pbc_values):
    raise ValueError("All PBC dimensions must be positive")
```

---

## 3.11 未来发展方向

根据分支名称 `feat/pbc-md`，未来可能的发展包括：

1. **分子动力学 (MD) 集成**
   - 使用 PBC 的 NVE/NVT/NPT 系综
   - 温度和压力控制
   - 轨迹生成和分析

2. **扩展单元格类型**
   - 支持非正交单元格（三斜）
   - 允许倾斜单元格向量

3. **增强的周期性处理**
   - 最小镜像约定
   - Ewald 求和（如果适用）
   - 周期性约束

4. **与更多计算器的集成**
   - 为 MACE 添加 PBC 支持
   - 为 ANI/AIMNet2 添加 PBC 支持（如果适用）

---

# 总结

## 代码质量亮点

1. ✅ **良好的文档**: 使用 docstring 完善
2. ✅ **关注点分离**: UMA 核心、溶剂化、电荷清晰分离
3. ✅ **适当的错误处理**: 提供信息丰富的消息
4. ✅ **PyTorch 集成**: GPU 加速
5. ✅ **一致的单位转换**: eV ↔ Hartree 处理一致

## 关键要点

### UMA Calculator

- 多任务学习势能函数，支持分子和材料
- 5 种任务类型：omol, oc20, omat, omc, odac
- 2 种模型大小：uma-s-1p1（小），uma-m-1p1（中）
- 支持 GBSA 溶剂化（16 种溶剂）
- 提供能量、力和 Hessian（有限差分）计算
- 与 MAPLE 所有任务类型完全集成

### PBC 设置

- 输入语法：`#pbc(X, Y, Z)`
- 仅支持正交单元格
- 与 UMA task=omol 不兼容
- 与 UMA task=oc20/omat/omc/odac 兼容
- 正在 feat/pbc-md 分支中积极开发
- 当前限制：无倾斜单元格，无 MD 功能（尚未）

### MAPLE 生态系统

- 综合性计算化学工具包
- 支持 12 种 ML 势能函数变体
- 9 种过渡态搜索方法
- 4 种几何优化算法
- 完整的频率和 IRC 分析
- 熟悉的输入格式和工作流程

---

**文档结束**

如需更多详细信息，请参阅：
- [ARCHITECTURE.md](ARCHITECTURE.md) - 详细技术架构
- [README.md](README.md) - 用户文档和示例
- [command_control.py](maple/function/read/command_control.py) - 参数验证
- [_uma_calculator.py](maple/function/calculator/uma/_uma_calculator.py) - UMA 实现
- [input_reader.py](maple/function/read/input_reader.py) - PBC 应用
