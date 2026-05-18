# MAPLE Parmfit Metal Ab Initio 架构

顶层 Parmfit 架构见 [ARCH_PARMFIT.md](./ARCH_PARMFIT.md)。

这份文档描述当前 `parmfit(method=abinitio)` 的金属路线。它是源码现状说明，不定义未来行为。

## 1. 定位

当前金属 ab initio 参数构建的 canonical implementation 是 `utils/MetalAA/`。

`metalcc` 只是示例输入、输出文件名前缀或任务命名习惯，不是独立 Python 模块。实际入口仍然是：

```text
parmfit(method=abinitio, pdb=..., target=..., cmo=..., cfmol2=...)
        |
        v
abinitio/abinitio.py
        |
        v
utils/MetalAA/workflow.py
```

`utils/MCPB/` 是范本和参考包。MetalAA 继承了 MCPB 对金属位点参数化问题的分层思想，但不在运行期调用 MCPB。

当前 MetalAA 使用两层模型：

- `large_model`
  - 用于几何优化、RESP、Hessian 和 modified Seminario。
- `site_model`
  - 定义最终部署 residue 集合，用于 RESP 电荷回填、per-residue mol2、renamed PDB、frcmod 和 `tleap.in` 导出。

这比 MCPB 更小：

- 没有 small / standard / large 三模型系统。
- 没有 fingerprint 文件契约。
- 没有 small-model-specific QM 分支。
- 没有自动运行 `tleap` 或生成 topology/coordinate 文件。
- 没有 multi-metal coupled workflow。

## 2. 入口与路由

### 2.1 `Parmfit.run()`

`Parmfit.run()` 负责最外层分发：

- `method="abinitio"` 时要求 `params["pdb"]` 存在并规范化为绝对路径。
- 从 `params["cmo"]` 解析 charge / multiplicity / optional oxidation，默认 `"0 1"`。
- 把 charge、mult、spin 和可选 `oxy` 写入 bootstrap `Atoms.info`。
- 实例化 `Abinitio(output, atoms, params)` 并调用 `run()`。

这里的 inline coordinate block 只用于 MAPLE 全局 reader 和 calculator bootstrap。金属路线的真实结构来自 `pdb`。

### 2.2 `Abinitio.run()`

`Abinitio.run()` 是薄 route orchestrator：

1. 读取 `params["pdb"]`。
2. 用 `params["target"]` 在 PDB 结构中定位唯一 residue。
3. 调用 `classify_kind(target_residue)`。
4. 当 `target_kind == "ion"` 时进入 MetalAA。
5. 当 `target_kind == "protein"` 时进入 NCAA。

MetalAA 路由只接受 `ion` target。其它 target kind 不会进入 `utils/MetalAA/`。

### 2.3 `MetalAbinitioConfig`

`parse_metal_abinitio_config(...)` 把 route 参数收束进 `MetalAbinitioConfig`：

- 必要来源：
  - `pdb_path`
  - `target`
  - `charge`
  - `mult`
  - `target_residue`
- MetalAA 参数：
  - `add_resid`
  - `cluster_cutoff`
  - `donor_cutoff`
  - `chgmod`
  - `fixchg_resids`
  - `watm`
  - `ionm`
- QM 参数：
  - `theory`
  - `basis`
  - `route`
  - `nproc`
  - `mem`

当前默认值：

```text
cluster_cutoff = 3.0
donor_cutoff   = 3.0
chgmod         = 1
watm           = opc
ionm           = 12_6
opt_max_iter   = 256
opt_max_step   = 0.2
QM             = gaussian / PBE1PBE / def2SVP
```

`watm` 和 `ionm` 都会校验。`watm` 影响 Gaussian `ReadRadii`；`watm/ionm` 共同决定生成的 ion frcmod 加载行，但 workflow 不自动运行 `tleap`。

## 3. 模块职责

Metal-specific code 位于 `utils/MetalAA/`：

- `config.py`
  - 定义 `MetalAbinitioConfig`。
  - 解析 MetalAA 输入面和 QM 选项。
- `core.py`
  - 定义 `MetalSiteSelection`。
  - 识别 target metal、自动 donor residues、手动 `add_resid`。
  - 执行单金属邻近校验。
- `models.py`
  - 构建 `large_model` 和 `site_model`。
  - 管理 `MetalModelBundle`。
- `charges.py`
  - 推断 `large_model` 总电荷。
  - 把 RESP 电荷从 `large_model` 投影回 `site_model`。
  - 投影键使用原始 PDB atom serial；找不到部署 atom 的 RESP charge 直接失败。
- `workflow.py`
  - 串起完整 MetalAA workflow。
  - 返回 `MetalWorkflowResult`。
- `export.py`
  - 规划文件路径。
  - 导出 large/site 过程文件、per-residue mol2、renamed full PDB、final frcmod 和完整 `tleap.in`。
  - 只给 metal atom 和 direct donor atoms 分配局部 atom type，例如 `M1`、`Y1`。
- `report.py`
  - 输出开始/结束报告和 `tleap reference` 片段。

MetalAA 使用的共享层：

- `utils/structure.py`
  - PDB parsing、residue key、target kind、formal charge 常量。
- `utils/context.py`
  - residue selector、peptide 前后邻居、cutoff environment。
- `utils/model.py`
  - `build_capped_selected_model(...)`、`model <-> ASE Atoms`、bond/angle 推断、PDB 写出。
- `utils/runtime.py`
  - `SilentLBFGS`、geometry optimization、Hessian helper、RESP pipeline。
- `utils/resp.py`
  - RESP 输入、固定电荷规则、RESP charge 读写、mol2 写出。
- `utils/interface.py`
  - Gaussian ESP input 和 Gaussian 调用。
- `utils/ionparams.py`
  - 本地离子半径表、Gaussian `ReadRadii` entries、离子 frcmod 名称推断。
- `utils/mSeminario.py`
  - modified Seminario bond / angle 力常数填充。

## 4. 对象与数据流

MetalAA 的核心数据流是：

```text
PDB
 |
 v
structure
 |
 v
MetalSiteSelection
 |
 v
large_model
 |
 | optimize_model_geometry
 | run_resp_pipeline
 v
charged large_model
 |
 v
site_model
 |
 | project RESP charges
 | optimize_model_geometry
 | Hessian + mSeminario
 v
MetalWorkflowResult + exported files
```

### 4.1 `structure`

`structure` 来自 `read_pdb(...)`，包含：

- `residues`
- `serial_to_atom`
- `serial_to_residue`
- `explicit_pairs`
- `_pair_cache`

每个 residue 通过 `classify_kind(...)` 标记为 `ion`、`protein`、`water`、`cofactor` 或 `ligand`。

### 4.2 `MetalSiteSelection`

`find_metal_site_core(...)` 生成 `MetalSiteSelection`：

- `target`
  - 目标金属 ion residue。
- `auto_core_residues`
  - donor cutoff 内自动识别到的 protein donor residues。
- `manual_core_residues`
  - 用户通过 `add_resid` 强制加入的 residues。
- `core_residues`
  - `target + auto_core_residues + manual_core_residues`。
- `donor_atoms`
  - key 为 residue key，value 为 donor atom name 列表。
- `warnings`
  - 手动加入但无自动 donor 的 residue、环境中潜在带电 residue 等提示。

自动 donor 元素来自 `METAL_SITE_DONOR_ELEMENTS`：`N/O/S/P/SE/F/CL/BR/I`。

如果 donor cutoff 内存在另一个 ion，当前 workflow 直接失败，因为 MetalAA 只支持单金属中心。

### 4.3 `large_model`

`build_metal_large_model(...)` 构建 `large_model`：

- 包含 metal core。
- 加入 `cluster_cutoff` 内、未在 core 中的 environment residues。
- 默认不把 water 作为 cutoff environment 加入。
- 对 peptide fragment 进行必要的 GLY bridge 和 ACE/NME cap。

`large_model` 的关键字段：

- `name = "large_model"`
- `target_key`
- `core_keys`
- `environment_keys`
- `donor_atoms`
- `warnings`
- `cluster_cutoff`
- `charge`
- `mult`

`large_model` 是优化和 RESP 的对象，不是最终正式导出的 site parameter object。

### 4.4 `site_model`

`site_model` 从原始 PDB `structure` 和 `selection.core_residues` 中构建：

- 包含 target metal。
- 包含 donor residues。
- 包含手动 `add_resid` residues。
- 不包含 GLY bridge。
- 不包含外边界 ACE/NME cap。
- 不包含 large-only environment residues。

`site_model` 的关键字段：

- `name = "site_model"`
- `target_key`
- `core_keys`
- `donor_atoms`
- `charge`
- `mult`
- `warnings`

`site_model` 是最终部署对象：RESP 电荷按原始 serial 回填到这一层；per-residue mol2、renamed PDB、final frcmod 和 `tleap.in` 都围绕这一层导出。Hessian 和 mSeminario 在 optimized `large_model` 上完成，然后只把能映射到部署 site 的 terms 写入 final frcmod。

### 4.5 `RespPipelineResult`

`run_resp_pipeline(...)` 返回 `RespPipelineResult`：

- `model`
  - 已写入 RESP charges 的 `large_model`。
- `files`
  - 当前主要包含 Gaussian input 和 RESP large mol2。
- `resp_files`
  - Gaussian log、ESP、resp1/resp2 输入输出、charge、calculated ESP 等 sidecar。
- `decision`
  - 使用的 `QMMethod`。

MetalAA 不把 RESP 产生的 large-model mol2 作为最终 mol2。最终 mol2 总是从 charge-updated、re-optimized 的 `site_model` 写出。

### 4.6 `MetalWorkflowResult`

`run_metal_abinitio(...)` 最终返回 `MetalWorkflowResult`：

- `selection`
- `large_model`
- `site_model`
- `artifacts`
- `site_typing`

它提供：

- `core_info`
  - `selection.to_dict()`。
- `files`
  - 正式输出路径。
- `resp_files`
  - RESP / Gaussian sidecar 路径。

## 5. 电荷链路

### 5.1 `cmo` 的 MetalAA 语义

用户通过 `cmo="q m"` 或 `cmo="q m oxy"` 提供 charge / multiplicity / optional oxidation。

在 MetalAA 中，`q` 是 target metal + 非蛋白配位片段的总电荷，`m` 是 metal site QM multiplicity，`oxy` 是可选的金属氧化数：

```text
nonprotein_site_charge = config.charge
site_mult              = config.mult
metal_formal_charge    = config.oxy if config.oxy is not None else config.charge
```

`q` 不等于 whole large model charge；标准蛋白残基电荷仍单独累加。离子元素来自 PDB atom `element/name`；离子形式电荷优先来自 `cmo[2]`，没有第三项时 fallback 到 `cmo[0]`。该形式电荷会传给 Gaussian `ReadRadii` 和 `watm/ionm` 对应的 ion frcmod 选择。

### 5.2 `large_charge`

`large_charge` 自动推断：

```text
large_charge = nonprotein_site_charge + sum(formal_charge(residue) for protein residues in large_model)
large_mult   = site_mult
```

标准 residue 形式电荷取自 `CHARGED_STANDARD_RESIDUES`：

```text
ASP -1
GLU -1
LYS +1
ARG +1
HIP +1
CYM -1
```

ACE/NME/GLY cap/bridge、水和未带电标准 residue 贡献 0。当前实现描述的是现状：large model 的未知非标准 residue 没有独立电荷数据库时按 0 处理；真正进入最终部署且需要重命名/导出的非标准 residue 必须能提供 atom type 来源，否则导出阶段失败。

### 5.3 `chgmod` 和 `fixchg_resids`

`chgmod` 传入 `resp.write_resp_input_files(...)`，控制 RESP 阶段固定哪些标准 backbone 电荷：

- `0`
  - 不按 backbone policy 自动固定。
- `1`
  - 固定 `N/CA/C/O/OXT`。
- `2`
  - 固定 `N/H/HA/CA/C/O/OXT`。
- `3`
  - 固定 `N/H/HA/CA/CB/C/O/OXT`。

`fixchg_resids` 会把指定 residue 的所有可查标准 Amber 电荷固定到 reference library 值。如果指定 residue 没有标准参考定义，RESP 输入生成会失败。

### 5.4 RESP 运行对象

RESP 在 optimized `large_model` 上运行：

1. `prepare_gaussian_esp_input(...)` 写 Gaussian ESP input。
2. 如果模型包含 ion，Gaussian route 使用 `Pop(MK,ReadRadii)`。
3. `collect_gaussian_readradii_entries(...)` 根据 `watm` 和 ion identity 追加 `Element radius` 行。
4. `run_gaussian(...)` 运行 Gaussian。
5. `espgen` 从 Gaussian log 抽 ESP。
6. `resp` 运行两阶段拟合。
7. `read_resp_charges(...)` 读取 `resp2.chg`。
8. `apply_resp_charges(...)` 写回 `large_model`。

### 5.5 RESP 回填到部署 `site_model`

`project_resp_charges_onto_site_model(site_model, charged_large_model)` 的规则：

- 构建 `charge_map[original_pdb_serial] = RESP charge`。
- `site_model` 是部署模型，只含 target metal、自动 donor residues 和用户 `add_resid` core residues。
- `site_model` 不包含 ACE/NME cap，也不包含 large-only GLY bridge。
- 对部署 atom：
  - 必须在 `charged_large_model` 中找到相同原始 PDB serial。
  - 找不到即失败。
  - charge 完全来自 RESP，不做后归一化。
- cap/bridge/environment atom 没有最终部署目标，不参与回填。

这个实现不使用 MCPB fingerprint；它依赖原始 PDB atom serial 在 large/site 两层之间保持稳定。

## 6. 执行链路

`run_metal_abinitio(...)` 的顺序是：

1. `plan_metal_artifacts(output)`
   - 生成正式输出路径和 RESP sidecar 容器。
2. `find_metal_site_core(...)`
   - 识别 metal core 和 donor atoms。
3. `build_metal_model_bundle(...)`
   - 构建 `large_model`。
4. `infer_large_model_charge(...)`
   - 计算 `large_charge`。
5. `write_large_pdb(..., optimized=False)`
   - 写 raw large PDB。
6. `optimize_model_geometry(large_model, ...)`
   - 用当前 `source_atoms.calc` 优化 large model。
7. `write_large_pdb(..., optimized=True)`
   - 写 optimized large PDB。
8. `infer_bond_pairs(large_model, ...)`
   - 给 RESP large mol2 准备 bond pairs。
9. `run_resp_pipeline(...)`
    - Gaussian ESP + espgen + RESP。
10. `build_metal_site_model(...)`
    - 从原始 PDB structure 和 `selection.core_residues` 构建部署 site model。
11. `project_resp_charges_onto_site_model(...)`
    - 按原始 PDB atom serial 把 RESP charges 映射回部署 site model。
12. `write_site_model_files(...)`
    - 写 diagnostic site PDB/mol2、per-residue mol2、renamed full PDB、`tleap.in`，并生成 `MetalSiteTyping`。
13. `model_to_atoms(large_model, ...)`
    - 把 optimized large model 转成 ASE Atoms。
14. `get_cartesian_hessian(large_atoms)`
    - 从 calculator 获取 large model Hessian。
15. `apply_mseminario(...)`
    - 填充 bond / angle 平衡值和力常数。
16. remap large terms to deployment site indices
    - 只保留所有 atom 都能映射到部署 site model 的 bond / angle terms。
17. `write_site_frcmod(...)`
    - 写最终 `<base>_metal.frcmod`。
18. `format_metal_final_lines(...)`
    - 写最终报告和 `tleap reference`。

## 7. 导出链路

### 7.1 正式输出

正式输出写到 `"<base>_work/"`：

```text
<base>_work/<base>_metal_large_raw.pdb
<base>_work/<base>_metal_large_opt.pdb
<base>_work/<base>_metal_site_opt.pdb
<base>_work/<base>_metal_site.mol2
<base>_work/<base>_metal.frcmod
<base>_work/<base>_metal_tleap.pdb
<base>_work/<base>_metal_tleap.in
<base>_work/HD1.mol2
<base>_work/GU1.mol2
<base>_work/FE1.mol2
```

`*_metal_site.mol2` 是 diagnostic/check artifact；正式 `tleap` 导入目标是 MCPB-style per-residue mol2，例如 `HD1.mol2`、`GU1.mol2`、`FE1.mol2`。

### 7.2 过程文件

RESP / Gaussian sidecar 写到：

```text
<base>_work/metalaa/
```

典型文件包括：

```text
<base>_metal_large_resp.gjf
<base>_metal_large_resp.log
metal_large_resp.esp
metal_large_resp.mol2
resp1.in
resp1.out
resp1.pch
resp1.chg
resp1_calc.esp
resp2.in
resp2.out
resp2.pch
resp2.chg
resp2_calc.esp
```

### 7.3 Atom type remapping

`_build_site_typing(...)` 会给 final site 局部重新分配 atom type：

- ion atom 使用 `M1`, `M2`, ...。
- direct donor atoms 使用 `Y1`, `Y2`, ...，再继续到 `Z/U/V/...`。
- donor 所在 residue 的其他 atoms 保持标准 Amber / GAFF type。
- cap / bridge atoms 不进入部署 site model，也不写 per-residue mol2。

`frcmod` 只写和 renamed atom type 相关的：

- `MASS`
- `BOND`
- `ANGLE`

当前 `DIHE`、`IMPROPER`、`NONBON` section 会保留为空 section。离子的 nonbonded 参数预期由报告里的标准 ion frcmod 提供。

### 7.4 `tleap reference`

MetalAA 不执行 `tleap`。它生成完整 `<base>_metal_tleap.in`，并在报告里复制同一段参考内容：

- `source leaprc.protein.ff19SB`
- `source leaprc.gaff2`
- `source leaprc.water.<watm>`
- `addAtomTypes { ... }`
- `<RES> = loadmol2 <RES>.mol2`
- `loadamberparams <ion frcmod>`
- `loadamberparams <base>_metal.frcmod`
- `mol = loadpdb <base>_metal_tleap.pdb`
- `bond ...` metal-donor bonds、renamed protein residue peptide reconnect bonds、disulfide bonds
- `quit`

这些文件是用户后续组装 AMBER 系统的输入；workflow 当前只生成输入，不自动运行 `tleap`。

## 8. 失败点和边界

当前 MetalAA 会在以下情况下失败或提前停止：

- `target` 缺失或找不到唯一 residue。
- target residue 不是 `ion`。
- donor cutoff 内发现另一个 ion。
- `watm` 不在支持列表中。
- `ionm` 不在支持列表中。
- `chgmod` 不是 `0/1/2/3`。
- geometry optimization 不收敛。
- Gaussian、espgen 或 resp 外部程序失败。
- RESP 返回 charge 数和 large model atom 数不匹配。
- 部署 atom 无法从 charged large model 按原始 PDB serial 找到 RESP charge。
- 部署 residue 无法解析出 Amber/GAFF old atom type。
- `source_atoms.calc` 不提供 `get_hessian()`。
- Hessian 形状不是 square 2D matrix。

`apply_mseminario(...)` 的部分数值异常会被捕获并记录 warning；workflow 仍会写 `frcmod`，但缺失或无法确定的 terms 不会被强行填入。

## 9. MCPB 对照

MCPB.py 是文件驱动的四步流水线：

1. Step 1
   - 识别 metal site。
   - 构建 `small / standard / large` 三模型。
   - 写 fingerprint 和 QM 输入。
2. Step 2
   - 生成 pre frcmod。
   - 用 empirical / Seminario / modified Seminario / Z-matrix / blank 等方式补 metal bond / angle。
3. Step 3
   - 在 large model 上做 RESP 或 FQ。
   - 通过 fingerprint 把电荷回填到 standard model。
   - 输出每个 metal-center residue 的 mol2。
4. Step 4
   - 写 `tleap` 输入。
   - 组合 mol2、frcmod 和原始 PDB。
   - 生成最终 AMBER 建模文件。

MetalAA 的对应关系：

| MCPB 概念 | MetalAA 当前实现 |
|---|---|
| Step 1 metal site 识别 | `find_metal_site_core(...)` |
| Step 1 large model | `build_metal_large_model(...)` |
| Step 1 standard model | `site_model` 承担最终导出对象职责 |
| Step 1 small model | 未实现独立 small model |
| fingerprint | 未实现；使用 original PDB atom serial 映射 |
| Step 2 modified Seminario | `apply_mseminario(...)` |
| Step 3 RESP | `run_resp_pipeline(...)` |
| Step 3 charge 回填 | `project_resp_charges_onto_site_model(...)` |
| Step 4 tleap assembly | 生成 per-residue mol2、renamed full PDB、final frcmod 和 `<base>_metal_tleap.in`；不自动运行 `tleap` |

MetalAA 继承 MCPB 的关键思想：

- 把“位点识别”、“电荷拟合”、“bond/angle 补参”、“最终导出”分开。
- RESP 使用更大的 capped environment，而不是只在最小 site 上拟合。
- 金属相关 atom types 需要局部重命名，避免污染标准力场类型。
- 离子 Gaussian `ReadRadii` 和 ion frcmod 选择应与 water model 相关。
- 最终部署文件按 MCPB-style 分 residue 导出，并用 `bond` commands 恢复 metal-donor、peptide reconnect 和 disulfide 连接。

MetalAA 刻意没有继承 MCPB 的部分：

- fingerprint 文件边界。
- 三模型系统。
- small-model QM Hessian 分支。
- 多个外部 step 手工串联。
- GAMESS / SQM / FQ branch。
- 自动 whole-protein `tleap` assembly。
- multi-metal coupled site 支持。

## 10. 当前设计取舍

当前 MetalAA 优先追求短链路和 MAPLE 内部可组合性：

- `Abinitio` 只做 route selection。
- `MetalAA` 只做 metal chemistry。
- `runtime` 只做外部程序 orchestration。
- `resp` 只做 RESP 输入/输出规则。
- `mSeminario` 只做 Hessian 到 bond/angle terms 的数值填充。

因此，MetalAA 的正式产物是 site-level 参数文件，而不是完整蛋白部署包。后续如果要扩展到 whole-protein AMBER assembly，应作为新的明确 workflow stage 设计，而不是隐式塞进现有导出步骤。
