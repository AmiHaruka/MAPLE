# SD 和 DIIS 优化算法重构计划

## 目标
将 SD (Steepest Descent) 和 DIIS 优化算法重构为类式设计，与 LBFGS/RFO 保持一致的代码风格。

---

## 当前问题分析

### 1. SD.py 问题
- **函数式设计**：直接是 `SD()` 函数，不是类
- **参数硬编码**：`max_step_size=0.2`, `maxiterations=128` 作为函数参数
- **没有继承 JobABC**：无法使用 `_init_params()` 等工具方法
- **单位转换不一致**：使用 `g_au` 转换，而 LBFGS 不使用
- **DIIS 集成粗糙**：每 10 次迭代在循环内部 `from .DIIS import DIIS`
- **没有轨迹输出**：缺少 `write_xyz` 功能

### 2. DIIS.py 问题
- **函数式设计**：`DIIS()` 是函数而非类
- **职责不清**：DIIS 应该是加速器组件，不是独立优化器
- **单位转换问题**：同样使用 `g_au`

### 3. optimization.py 问题
- **SD 调用不一致**：`SD(self.atoms, output=self.output)` 没有传 `paras` 参数

---

## 重构设计

### 1. DIISAccelerator 类（加速器模式）

将 DIIS 重新定位为**加速器组件**，可被任何优化器复用：

```python
@dataclass
class DIISParams:
    memory: int = 10          # 历史向量数量
    min_vectors: int = 3      # 最少向量数才能外推

class DIISAccelerator:
    def __init__(self, params: DIISParams = None)
    def store(self, x: np.ndarray, error: np.ndarray) -> None
    def can_extrapolate(self) -> bool
    def extrapolate(self) -> Optional[np.ndarray]
    def reset(self) -> None
```

### 2. SD 类设计（继承 JobABC）

```python
@dataclass
class SDParams:
    max_step: float = 0.2           # 最大步长 (Angstrom)
    max_iter: int = 256             # 最大迭代次数
    diis_enabled: bool = True       # 是否启用 DIIS 加速
    diis_interval: int = 10         # DIIS 触发间隔
    diis_memory: int = 10           # DIIS 历史向量数
    write_traj: bool = False        # 是否写轨迹
    traj_every: int = 1             # 轨迹写入间隔
    verbose: int = 1                # 日志详细程度

class SD(JobABC):
    def __init__(self, atoms, output, paras=None)
    def run(self) -> Atoms
    def _clip_step(self, step) -> np.ndarray
    def _try_diis_acceleration(self, forces) -> Optional[np.ndarray]
    def _build_iter_message(...) -> List[str]
    def _log_iter(self, info_message) -> None
    def _check_convergence(self) -> bool
```

---

## 需要修改的文件

| 文件 | 修改内容 |
|------|---------|
| `maple/function/dispatcher/optimization/algorithm/DIIS.py` | 重构为 DIISAccelerator 类 + 保留旧函数兼容 |
| `maple/function/dispatcher/optimization/algorithm/SD.py` | 完全重写为类式设计 |
| `maple/function/dispatcher/optimization/algorithm/__init__.py` | 更新导出 |
| `maple/function/dispatcher/optimization/optimization.py` | 更新 SD 调用方式 |

---

## 详细实现步骤

### Step 1: 重构 DIIS.py

1. 创建 `DIISParams` dataclass
2. 创建 `DIISAccelerator` 类
   - `store()`: 存储位置和误差向量
   - `extrapolate()`: 执行 DIIS 外推（构建 B 矩阵，求解线性系统）
   - `reset()`: 重置历史
3. 保留旧的 `OptimizationStorage` 和 `DIIS()` 函数（标记为 deprecated）

### Step 2: 重构 SD.py

1. 创建 `SDParams` dataclass（参考 LBFGSParams）
2. 创建 `SD` 类继承 `JobABC`
3. 实现核心方法：
   - `__init__()`: 初始化参数，创建 DIISAccelerator
   - `_clip_step()`: 限制步长
   - `_try_diis_acceleration()`: 尝试 DIIS 加速
   - `_build_iter_message()`: 构建迭代日志（与 LBFGS 格式一致）
   - `_check_convergence()`: 检查收敛条件
   - `run()`: 主优化循环
4. 添加 `write_xyz()` 轨迹输出
5. **移除 `g_au` 单位转换**，与 LBFGS 保持一致

### Step 3: 更新 __init__.py

```python
from .DIIS import DIISAccelerator, DIISParams, OptimizationStorage, DIIS
from .SD import SD, SDParams
```

### Step 4: 更新 optimization.py

```python
elif self.commandcontrol.get('method').lower() == 'sd':
    from .algorithm import SD
    opt = SD(self.atoms, output=self.output, paras=self.commandcontrol)
    return opt.run()
```

---

## 关键设计决策

### 1. DIIS 作为加速器而非独立优化器
- DIIS 本身不计算搜索方向，只做位置外推
- 作为组件可被 SD、CG 等优化器复用

### 2. 移除 `g_au` 单位转换
- LBFGS/RFO 直接使用 ASE 返回的单位 (eV, eV/Å)
- SD/DIIS 应该保持一致

### 3. 保留向后兼容性
- 保留旧的 `DIIS()` 函数和 `OptimizationStorage` 类
- 使用 DeprecationWarning 提示迁移

### 4. 统一的收敛检查
- 四项标准：max_f, rms_f, max_dp, rms_dp
- 与 LBFGS 使用相同的阈值属性

---

## 验证方案

1. **运行测试**：使用简单分子（如 H2O）测试 SD 优化
2. **对比收敛**：验证 DIIS 加速确实提高收敛速度
3. **检查输出**：确保日志格式与 LBFGS 一致
4. **回归测试**：确保现有功能不受影响

---

## 预期输出文件

优化完成后将生成：
- `{base}_traj.xyz` - 完整轨迹
- `{base}_opt.xyz` - 最终优化结构
- `{base}_opt_traj.xyz` - 详细轨迹（当 write_traj=True）
