# Collaboration Knowledge Base

This document captures collaboration experience, preferred working styles, and lessons learned while working on the MAPLE project.

---

## Preferred Collaboration Style

### Communication Preferences
- **语言**: 日常交流使用中文（用户为中文母语者）
- **专业术语**: 使用英文保持准确性（如 NEB, PRFO, Hessian, convergence, trajectory 等）
- **代码讨论**:
  - 中文解释逻辑和架构
  - 英文保留 function names, class names, variable names
  - 混合使用以实现快速理解

### Working Style
- **行动导向**: 用户偏好直接修复问题，而不是先询问权限
- **深度优先**: 需要找到根本原因，不只是表面修复
- **主动性**: 发现问题后立即修复，不等待进一步指示
- **知识管理**: 每次会话后更新知识库，记录经验教训

### Documentation Style
- **文档语言**: 中英混用，优先考虑快速理解
- **专业词汇**: 保留英文术语（避免生硬的中文翻译）
- **结构**:
  - 使用中文标题和说明
  - Code blocks 和 technical terms 使用英文
  - 示例和注释根据上下文混用
- **目标**: 快速传达信息，减少理解成本

### Code Review Preferences
- *To be documented as we work together*

---

## Project-Specific Knowledge

### MAPLE Codebase Insights
- Main computational chemistry toolkit using ML potentials
- Focus on transition state search methods (8 algorithms)
- Input format follows Gaussian-style conventions
- ASE (Atomic Simulation Environment) as core dependency
- **PBC Support**: Flexible periodic boundary conditions (2/3/6 parameters)
- **ML Potentials**: ANI, AIMNet2, MACE, UMA (Universal Materials Atom)
  - UMA: Multi-task model with 5 task types (omol, oc20, omat, omc, odac)
  - PBC compatibility: omol (分子) 不支持，oc20/omat/omc/odac (材料) 支持

### Key Design Patterns
- **JobABC**: Abstract base class for all job types
- **Dataclass parameters**: Type-safe, case-insensitive parsing
- **Factory pattern**: SetCalculator for ML potential initialization
- **Template method**: _init_params() for consistent parameter handling

### Critical Files to Reference
- [jobABC.py](../../maple/function/dispatcher/jobABC.py) - Base class for all jobs
- [engine.py](../../maple/function/engine.py) - Core orchestration
- [dispatcher.py](../../maple/function/dispatcher/dispatcher.py) - Job routing
- [neb.py](../../maple/function/dispatcher/ts/algorithm/neb.py) - Most comprehensive TS implementation

---

## Common Pitfalls and Mistakes

### Past Issues

#### 1. Python Bytecode Cache 导致旧代码执行 (2026-01-30)
**问题**: 修改了 Python 代码后，运行时仍然执行旧版本的代码
**原因**: Python 的 `__pycache__` 目录缓存了编译后的 .pyc 文件
**解决方案**:
```bash
# 清除所有 Python 缓存
find /home/user/software/MAPLE -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
```
**教训**: 修改代码后如果行为异常，首先清除 Python 缓存

#### 2. 代码流程中的 continue 语句导致逻辑跳过 (2026-01-30)
**问题**: `#model=uma(task=oc20)` 解析失败，`model_name` 为空
**原因**:
- 代码有两个处理括号的块：通用块（行 105-128）和特殊块（行 131-147）
- 通用块在匹配 `paren_val` 后执行 `continue`，跳过了后续的 model 特殊处理
- 导致只解析了 `{'task': 'oc20'}`，缺少 `'name': 'uma'`

**修复**: 在通用块之前添加 model 的特殊处理
```python
# 在 if paren_val: 块中，PBC 特殊处理之后添加：
if key == 'model' and assign_val:
    value = cls._auto_cast(assign_val.strip())
    model_params = {'name': value}
    cls._parse_nested(model_params, paren_val)
    params[key] = model_params
    continue
```
**教训**:
- 处理条件分支时注意 `continue` 语句的影响
- 特殊情况的处理要在通用逻辑之前
- 使用调试输出追踪代码执行流程

#### 3. 只读属性赋值错误 (2026-01-30)
**问题**: `AttributeError: can't set attribute 'task_name'`
**原因**: 父类 `FAIRChemCalculator` 的 `task_name` 是只读 property，在 `super().__init__()` 后不能再次赋值
**修复**: 删除重复的属性赋值语句
**教训**:
- 调用 `super().__init__()` 后，检查哪些属性已经被父类设置
- 不要重复设置只读属性

### Things to Avoid

1. **不要猜测代码执行流程** - 使用调试输出或仔细追踪逻辑
2. **不要忽视 Python 缓存** - 修改代码后异常时先清除缓存
3. **不要在没有 conda 环境的情况下运行测试** - 需要 ASE、PyTorch 等依赖
4. **不要假设正则表达式的捕获组** - 用实际输入测试验证
5. **不要忽略代码中的 `continue` 和 `return`** - 它们会改变控制流

### Best Practices

1. **调试策略**:
   - 添加 `print()` 语句追踪变量值
   - 在关键分支点输出执行路径
   - 完成调试后移除所有调试输出

2. **代码修改流程**:
   - 先理解现有代码的完整逻辑流程
   - 识别问题的根本原因
   - 实施最小化修改
   - 清除 Python 缓存
   - 在正确的环境中测试

3. **解析复杂输入**:
   - 先用简单的测试用例验证正则表达式
   - 处理特殊情况要在通用逻辑之前
   - 记录每个分支的处理逻辑

---

## Development Workflow

### Environment Setup

**Conda 安装路径**: `/home/user/software/miniconda3`
**Conda Environment**: `mlp`
**Python 路径**: `/home/user/software/miniconda3/envs/mlp/bin/python`

```bash
# ⚠️ 在 Bash tool 中激活 conda 环境的正确方式（必须先 source）
source /home/user/software/miniconda3/etc/profile.d/conda.sh && conda activate mlp

# 示例：激活环境后运行 MAPLE
source /home/user/software/miniconda3/etc/profile.d/conda.sh && conda activate mlp && maple input.inp

# 检查是否在正确环境
which python  # 应该指向 /home/user/software/miniconda3/envs/mlp/bin/python
```

**关键依赖**:
- Python >= 3.9
- ASE (Atomic Simulation Environment)
- PyTorch >= 2.0
- fairchem-core (for UMA calculator)
- NumPy, SciPy

**重要提示**:
- ⚠️ **Bash tool 中不能直接 `conda activate`**，必须先 `source .../conda.sh`
- ⚠️ **不要在没有激活 mlp 环境的情况下运行代码**
- ⚠️ **系统的 `maple` 命令可能指向其他包（malepso），需要在项目目录中运行**
- ⚠️ **调试脚本需要在 mlp 环境中运行才能 import ASE 等模块**

### Testing Strategy
- Example test cases in `example/` directory
- Use `test.py` for running built-in tests
- Command: `maple --test <test_number>`

**测试流程**:
```bash
cd /home/user/software/MAPLE
conda activate mlp

# 清除 Python 缓存（如果修改了代码）
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# 运行测试
cd example/opt/lbfgs
maple pbc_cu.inp
```

### Common Development Tasks
- Adding new ML potential: Implement calculator in `maple/function/calculator/`
- Adding new job type: Inherit from JobABC, register in Dispatcher
- Adding new TS algorithm: Inherit from JobABC, add to TransitionState router
- Modifying parameters: Update corresponding @dataclass definition

---

## Future Enhancements

### Potential Improvements
- *To be documented as ideas emerge*

### Feature Requests
- *To be documented as needs arise*

---

## Session History

### Session 1: Initial Codebase Exploration (2026-01-23)
**Tasks Completed:**
- Comprehensive exploration of MAPLE codebase structure
- Detailed analysis of TS module architecture
- Created documentation in `/home/user/software/MAPLE/claude/`:
  - `summary/framework_overview.md` - Complete framework summary
  - `summary/ts_module_detailed.md` - In-depth TS module technical analysis
  - `knowledge/collaboration_notes.md` - This template for future collaboration

**Key Findings:**
- MAPLE is a well-architected computational chemistry toolkit (~4,000+ lines in TS module alone)
- 8 distinct TS search algorithms (NEB, PRFO, Dimer, String, AFIR, etc.)
- Clean separation of concerns with JobABC base class
- Sophisticated mathematical implementations (Kabsch alignment, IDPP, dynamic springs, etc.)
- Production-quality features (logging, trajectories, convergence checks)

**Tools Used:**
- Explore agents for codebase understanding
- Read tool for file inspection
- Documentation written to project directory

### Session 2: Bug Fixes & PBC Enhancement (2026-01-30)
**Tasks Completed:**
1. **创建 MAPLE 代码库总结** (中文)
   - 文件: `/home/user/software/MAPLE/summary.md` (24KB)
   - 内容: MAPLE 概述、UMA Calculator 详解、PBC 设置详解
   - 重点: UMA 与 PBC 的兼容性、任务类型、溶剂化集成

2. **修复 Bug #1: UMACalculator 只读属性错误**
   - 文件: `maple/function/calculator/uma/_uma_calculator.py:80-81`
   - 问题: `AttributeError: can't set attribute 'task_name'`
   - 原因: 父类 `FAIRChemCalculator` 的 `task_name` 是只读 property
   - 修复: 删除重复的 `self.task_name` 和 `self.model_name` 赋值

3. **修复 Bug #2: Model 参数解析失败**
   - 文件: `maple/function/read/command_control.py:123-132`
   - 问题: `#model=uma(task=oc20)` 导致 `Unsupported model: ''`
   - 原因: 通用的 `if paren_val:` 块在 model 特殊处理之前执行 `continue`
   - 根本原因: 代码流程中的条件分支和 `continue` 语句导致逻辑跳过
   - 修复: 在 PBC 特殊处理后添加 model 特殊处理，确保在通用块之前执行
   - 调试方法: 添加多处 `print()` 调试输出追踪执行流程

4. **增强功能: PBC 参数灵活性**
   - 文件:
     - `maple/function/read/command_control.py:106-138` (解析)
     - `maple/function/read/input_reader.py:423-449, 512-518` (应用)
   - 新功能:
     - `#pbc(a,b,c,α,β,γ)` - 6 参数，完整晶胞（支持三斜晶系）
     - `#pbc(a,b,c)` - 3 参数，默认 α=β=γ=90°（正交晶胞）
     - `#pbc(a,b)` - 2 参数，默认 c=1000, α=β=γ=90°（模拟 2D 周期性）
   - 实现: 使用 `ase.cell.Cell.fromcellpar()` 构建晶胞矩阵
   - 向后兼容: 保持原有 3 参数格式正常工作

**遇到的挑战:**
- Python bytecode 缓存导致修改后的代码不生效
- 代码逻辑流程中的 `continue` 语句影响了预期的执行路径
- 需要在用户的 conda 环境中测试，无法直接运行测试

**解决方案:**
- 清除 `__pycache__` 目录
- 添加调试输出追踪代码执行流程
- 重构代码逻辑，将特殊情况处理放在通用逻辑之前
- 完成后移除所有调试输出

**经验教训:**
- 修改代码后出现异常行为，首先清除 Python 缓存
- 使用调试输出追踪变量值和执行路径
- 注意条件分支中的 `continue` 和 `return` 对控制流的影响
- 特殊情况的处理要在通用逻辑之前

**Tools Used:**
- Explore agents (parallel) 用于理解 PBC 和 UMA 实现
- Read tool 检查代码逻辑
- Grep 查找相关代码位置
- Edit tool 修复 bug 和增强功能
- Bash 清除 Python 缓存

### Session 3: SD/DIIS 优化算法重构 & GDIIS 改进 (2026-02-02)
**Tasks Completed:**

1. **SD.py 重构为类式设计**
   - 从函数 `SD()` 改为 `class SD(JobABC)`
   - 新增 `SDParams` dataclass（与 LBFGSParams/RFOParams 风格一致）
   - 参数通过 `_init_params()` 支持大小写不敏感解析
   - 新增 `write_xyz` 轨迹输出、收敛检查、日志系统
   - 移除了 `g_au` 单位转换（与 LBFGS 保持一致）

2. **DIIS.py 重构为 GDIIS 加速器类**
   - 从函数 `DIIS()` 改为 `class DIISAccelerator`
   - 新增 `DIISParams` dataclass
   - 保留旧函数和 `OptimizationStorage`（标记为 deprecated）

3. **GDIIS 算法改进**（核心改进，原始 naive DIIS 效果差）
   - **问题诊断**:
     - 原始 naive DIIS 直接外推位置 `x_new = Σ c_i * x_i`，导致能量升高
     - 每步存储导致误差向量高度线性相关（B 矩阵几乎奇异）
     - 每次 DIIS 后 reset 丢失历史
   - **GDIIS 公式**: 外推修正坐标 `x_tilde_i = x_i + step_scale * f_i`
   - **间隔存储**: 每 `diis_store_every=5` 步才存储快照，确保向量多样性
   - **Barzilai-Borwein step_scale**: 自适应估计局部曲率作为 H⁻¹ 近似
   - **步骤验证**: GDIIS 后检查能量/力，拒绝差的步骤并回退 SD
   - **Tikhonov 正则化**: `B += ε * max(diag(B)) * I` 改善数值稳定性
   - **lstsq 代替 solve**: 更稳定的线性方程组求解
   - **结果**: SD+GDIIS 在 116 步收敛（纯 naive DIIS 256 步未收敛）

4. **注册 SD 到 command_control.py**
   - 在 `IMPLEMENTATION_MAP["opt"]` 中添加 `"sd"`

5. **更新 optimization.py**
   - SD 调用方式与 LBFGS/RFO 一致:
     `opt = SD(self.atoms, output=self.output, paras=self.commandcontrol)`

6. **创建 SD 测试用例**
   - 目录: `example/opt/sd/`
   - 输入: `inp1.inp`（与 lbfgs/rfo 同一分子、同风格）

**修改的文件:**
- `maple/function/dispatcher/optimization/algorithm/DIIS.py` - 完全重写
- `maple/function/dispatcher/optimization/algorithm/SD.py` - 完全重写
- `maple/function/dispatcher/optimization/algorithm/__init__.py` - 更新导出
- `maple/function/dispatcher/optimization/optimization.py` - 更新 SD 调用
- `maple/function/read/command_control.py` - 注册 SD 方法

**关键经验教训:**
- GDIIS 配合 SD 的核心挑战是 B 矩阵条件差（SD 步方向单调）
- 间隔存储（而非每步存储）是让 GDIIS 在 SD 上工作的关键
- 系数检查应该放宽或去掉——用最终步长验证替代
- GDIIS 本设计用于 quasi-Newton，配 SD 需要特殊处理

**GDIIS 算法参考文献:**
- Csaszar & Pulay, J. Mol. Struct. (Theochem) 114, 31-34 (1984)
- Farkas & Schlegel, PCCP 4, 11-15 (2002) - Controlled GDIIS

---

## Notes for Future Sessions

### Quick Start Checklist
1. Reference `summary/framework_overview.md` for overall architecture
2. Reference `summary/ts_module_detailed.md` for TS-specific details
3. Check this file for collaboration preferences and past lessons
4. Review recent session history for context

### Useful Commands
```bash
# ⚠️ 所有命令都需要先激活 conda 环境
source /home/user/software/miniconda3/etc/profile.d/conda.sh && conda activate mlp

# Run MAPLE
maple input.inp

# Run built-in tests
maple --test 1  # LBFGS optimization
maple --test 3  # NEB transition state
maple --test 5  # RFO optimization
maple --test 7  # Frequency analysis

# Run example tests
cd /home/user/software/MAPLE/example/opt/lbfgs && maple inp1.inp
cd /home/user/software/MAPLE/example/opt/rfo && maple inp1.inp
cd /home/user/software/MAPLE/example/opt/sd && maple inp1.inp

# Install
pip install -e .              # Full installation
pip install -e ".[minimal]"   # Minimal dependencies

# 清除 Python 缓存（修改代码后）
find /home/user/software/MAPLE -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
```

### PBC 使用快速参考
```bash
# 完整晶胞参数（6 个值）
#pbc(a, b, c, alpha, beta, gamma)
#pbc(10, 15, 20, 90, 90, 120)  # 六方晶系

# 正交晶胞（3 个值，默认 α=β=γ=90°）
#pbc(a, b, c)
#pbc(18.19160, 18.19160, 24.90230)

# 2D 周期性（2 个值，c=1000 模拟无周期性）
#pbc(a, b)
#pbc(20.0, 20.0)  # z 方向很大，相当于 2D 材料
```

### UMA Calculator 快速参考
```bash
# 分子系统（默认，不支持 PBC）
#model=uma
#model=uma(task=omol, size=uma-s-1p1)

# 材料系统（支持 PBC）
#model=uma(task=oc20)         # OpenCatalyst 2020
#model=uma(task=omat)         # Materials database
#model=uma(task=omc)          # Metal-organic compounds
#model=uma(task=odac)         # Domain adaptation

# 模型大小
size=uma-s-1p1  # Small (默认)
size=uma-m-1p1  # Medium
```

### Optimization 算法快速参考
```bash
# LBFGS (默认, 推荐)
#opt(method=lbfgs)
#opt(method=lbfgs, memory=5, max_step=0.2, max_iter=256)

# RFO (信任域方法)
#opt(method=rfo)
#opt(method=rfo, trust_radius_init=0.2, max_iter=256)

# SD + GDIIS (最速下降 + GDIIS 加速)
#opt(method=sd)
#opt(method=sd, max_step=0.2, diis_enabled=true, diis_store_every=5, diis_memory=6)
#opt(method=sd, diis_enabled=false)  # 纯 SD, 不用 GDIIS
```

### 代码架构快速参考
```
optimization/
├── optimization.py     # 入口: Optmization.run() 路由到具体算法
└── algorithm/
    ├── __init__.py     # 导出: LBFGS, RFO, SD, DIISAccelerator 等
    ├── LBFGS.py        # L-BFGS 优化器 (class LBFGS(JobABC))
    ├── RFO.py          # RFO 优化器 (class RFO(JobABC))
    ├── SD.py           # SD+GDIIS 优化器 (class SD(JobABC))
    ├── DIIS.py         # GDIIS 加速器 (class DIISAccelerator)
    └── logger.py       # 日志工具

所有优化器遵循统一模式:
  - 继承 JobABC
  - @dataclass 定义 XxxParams
  - _init_params() 初始化参数 (大小写不敏感)
  - run() 返回优化后的 Atoms
  - write_xyz() 输出轨迹
```

---

## Contact and Resources

### Documentation Locations
- Main README: `/home/user/software/MAPLE/README.md`
- Architecture docs: `/home/user/software/MAPLE/ARCHITECTURE.md`
- Claude summaries: `/home/user/software/MAPLE/claude/summary/`
- Collaboration notes: `/home/user/software/MAPLE/claude/knowledge/` (this file)
- Development plans: `/home/user/software/MAPLE/claude/plan/`

### External Resources
- ASE Documentation: https://wiki.fysik.dtu.dk/ase/
- PyTorch: https://pytorch.org/
- fairchem-core: ML potential models

---

*This document will be updated continuously as we collaborate and learn more about working together on the MAPLE project.*
