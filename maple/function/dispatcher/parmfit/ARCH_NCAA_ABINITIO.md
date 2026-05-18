# MAPLE Parmfit NCAA Ab Initio 架构

顶层 Parmfit 架构见 [ARCH_PARMFIT.md](./ARCH_PARMFIT.md)。金属专项架构见 [ARCH_METAL_ABINITIO.md](./ARCH_METAL_ABINITIO.md)。

这份文档描述当前 `parmfit(method=abinitio)` 的 NCAA 路线。它是源码现状说明，不定义未来行为。

## 1. 定位

当前单残基 NCAA ab initio 参数构建的 canonical implementation 是 `utils/NCAA/`。

实际入口是：

```text
parmfit(method=abinitio, pdb=..., target=..., cmo=...)
        |
        v
abinitio/abinitio.py
        |
        v
utils/NCAA/workflow.py
```

NCAA route 处理的是 `classify_kind(target_residue) == "protein"` 的 target residue。源码当前不再额外区分标准氨基酸和非标准氨基酸；只要 target 被分类为 protein，就会进入 `utils/NCAA/`。

NCAA 不是 MetalAA / MCPB 风格的 metal site route。它的核心对象不是 metal core、donor residues 或 site model，而是一个目标残基的 capped representative：

```text
ACE-target-NME
```

当前 NCAA workflow 围绕四件事组织：

- 构建并优化 `ACE-target-NME` capped representative。
- 从 representative 生成 alpha / beta RESP conformers。
- 对 alpha / beta conformers 做多构象 RESP，并把电荷回填到 representative ordering。
- 通过 AmberTools 生成 `prepin/frcmod`，再用 mSeminario 和 shared torsion refinement 生成 Maple refined template。

如果 README 或总架构文档与当前源码不一致，本专项文档以源码和测试所表达的当前实现为准。一个已知例子是 `rn`：当前 `NCAAAbinitioConfig` 默认 `rn = "MOL"`，不是 target 原残基名。

## 2. 入口与路由

### 2.1 `Parmfit.run()`

`Parmfit.run()` 负责最外层分发：

- `method="abinitio"` 时要求 `params["pdb"]` 存在并规范化为绝对路径。
- 从 `params["cmo"]` 解析 charge / multiplicity，默认 `"0 1"`。第三项 oxidation 只供 MetalAA 使用，NCAA route 忽略它。
- 把 charge、mult、spin 写入 bootstrap `Atoms.info`。
- 实例化 `Abinitio(output, atoms, params)` 并调用 `run()`。

这里传入的 `Atoms` 主要负责携带 calculator、charge / multiplicity 和优化阈值。NCAA route 的真实结构来自 `pdb`，不是 inline coordinate block。

### 2.2 `Abinitio.run()`

`Abinitio.run()` 是薄 route orchestrator：

1. 读取 `params["pdb"]`。
2. 用 `params["target"]` 在 PDB 结构中定位唯一 residue。
3. 调用 `classify_kind(target_residue)`。
4. 当 `target_kind == "ion"` 时进入 MetalAA。
5. 当 `target_kind == "protein"` 时进入 NCAA。

NCAA 路由只接受 `protein` target。`ion` target 不会进入 NCAA，其它 target kind 当前直接报 `NotImplementedError`。

### 2.3 `NCAAAbinitioConfig`

`parse_ncaa_abinitio_config(...)` 把 route 参数收束进 `NCAAAbinitioConfig`：

- 必要来源：
  - `pdb_path`
  - `target`
  - `charge`
  - `mult`
- NCAA 参数：
  - `rn`
  - `vib_scale`
- QM 参数：
  - `qm_backend`
  - `resp_backend`
  - `theory`
  - `basis`
  - `route`
  - `nproc`
  - `mem`
- shared torsion 参数：
  - `torsionfit`
  - `torsion_bonds`
  - `torsion_steps`
  - `torsion_refine_rounds`
  - `torsion_refine_max_iter`
  - `torsion_refine_tol`
  - `backend`
  - `constraint_mode`

当前默认值：

```text
rn            = MOL
vib_scale     = 1.0
QM            = gaussian / HF / 6-31G(d)
nproc         = 8
mem           = 16
torsionfit    = true
torsion_steps = 72
```

`qm_backend` / `resp_backend` 属于 QM/RESP 配置，当前默认 `gaussian`。shared torsion scan backend 使用 TorsionFit 的 `backend` 字段，默认来自 `TorsionFitParams`，当前为 `cgbs`。不要把 torsion backend 和 QM backend 混用。

## 3. 模块职责

NCAA-specific code 位于 `utils/NCAA/`：

- `config.py`
  - 定义 `NCAAAbinitioConfig`。
  - 解析 NCAA 输入面、QM 选项和 shared torsion 参数。
- `models.py`
  - 定义 `NCAAIdentity` 和 `NCAAConformer`。
  - 识别 chirality 和 sidechain anchor。
  - 构建 `ACE-target-NME` capped model。
  - 优化 representative，生成 alpha / beta conformers。
  - 推断 mainchain path，并为 torsion refinement 构建 sidechain center-bond filter。
- `workflow.py`
  - 串起完整 NCAA workflow。
  - 返回 `NCAAWorkflowResult`。
- `artifacts.py`
  - 调用 `antechamber`、`prepgen`、`parmchk2`。
  - 写 mainchain `.mc`。
  - 规划 `NCAAAmberArtifacts` 和 `NCAAArtifacts`。
  - 从 original `prepin` 解析 residue atom rows。
  - 分配 Maple atom types，例如 `Z0`、`Z1`。
  - 将 full capped parameter set 映射成 residue-only refined template。
  - 插入 peptide boundary cross terms。
  - 写 refined `prepin/frcmod`、capped PDB 和 NCAA `tleap.in`。
- `report.py`
  - 输出开始/结束报告、阶段耗时、Maple atom-type remapping 和 `tleap reference` 片段。

NCAA 使用的共享层：

- `utils/structure.py`
  - PDB parsing、residue key、target kind、坐标和二面角工具。
- `utils/context.py`
  - residue selector 和唯一 target residue 校验。
- `utils/capping.py`
  - `ACE` / `NME` cap 构建。
- `utils/model.py`
  - `model <-> ASE Atoms`、bond/angle 推断、PDB / mol2 写出辅助。
- `utils/runtime.py`
  - geometry optimization、Hessian helper、多构象 RESP pipeline。
- `utils/resp.py`
  - 多构象 RESP 输入、ESP merge、charge 读写、RESP mol2 写出。
- `utils/interface.py`
  - Gaussian、AmberTools、frcmod 写出和 patch helpers。
- `utils/readparm.py`
  - 从 typed mol2 + frcmod 构建 `CorrectionParameterSet`。
- `utils/mSeminario.py`
  - modified Seminario bond / angle 力常数填充。
- `utils/TorsionFit/`
  - shared torsion scan 和 torsion parameter refinement。

## 4. 对象与数据流

NCAA 的核心数据流是：

```text
PDB
 |
 v
structure
 |
 v
target_residue
 |
 v
NCAAIdentity
 |
 v
capped_model
 |
 | optimize_capped_reference
 v
representative
 |
 | build_resp_conformers_from_reference
 v
alpha / beta conformers
 |
 | run_multiconformer_resp
 v
MultiRespPipelineResult
 |
 v
charged representative_model
 |
 | antechamber / prepgen / parmchk2
 v
NCAAAmberArtifacts
 |
 | mSeminario + torsion refinement
 | Maple atom-type remapping
 v
refined prepin/frcmod + NCAA tleap input
 |
 v
NCAAWorkflowResult + exported files
```

这条后半段与 `AutoNACC` 的运行逻辑一致：RESP charge 之后交给 AmberTools 生成 residue template，再给用户用于最终 protein `tleap` 组装。MAPLE 这里不接管 AutoNACC 的前期 PDB 准备、标准化、补氢或对齐；用户需要先提供可用的 protein PDB。

### 4.1 `structure` 和 `target_residue`

`structure` 来自 `read_pdb(...)`，包含 PDB residues、serial maps、explicit bonds 和 pair cache。

`Abinitio.run()` 用 `find_unique_residue(...)` 从 `structure` 找到 `target_residue`。`target_residue` 必须唯一匹配 `params["target"]`，并且必须被 `classify_kind(...)` 识别为 `protein`。

`run_ncaa_abinitio(...)` 当前接收 `structure`，但 workflow 内部主要使用已经解析好的 `target_residue`。

### 4.2 `NCAAIdentity`

`identity_ncaa(target_residue)` 生成 `NCAAIdentity`：

- `residue_key`
  - target residue 的 `(chain, resseq, icode)`。
- `resname`
  - PDB 中原始 residue name。
- `chirality`
  - 通过 `N/CA/C/sidechain_anchor` 的 signed volume 判定 `L` 或 `D`。
- `sidechain_anchor`
  - 与 `CA` 共价相连的最近 sidechain heavy atom。

如果 target residue 缺少 `N`、`CA` 或 `C`，chirality detection 会失败。如果 `CA` 没有可识别的 sidechain heavy atom，当前实现会在取第一个 candidate 时失败。

### 4.3 `capped_model`

`build_capped_ncaa_model(target_residue, config.rn)` 构建 `ACE-target-NME`：

- target residue 会被复制，并把 resname 改为 `config.rn`。
- N 端 cap 来自 `build_ace_cap(...)`。
- C 端 cap 来自 `build_nme_cap(...)`。
- `target_key` 仍指向复制后的 target residue key。
- `segment_sizes` 记录 `ace`、`residue`、`nme` 的 atom 数。

`_prepare_ncaa_models(...)` 随后把 `config.charge` 和 `config.mult` 写到 `capped_model`。这里的 charge / multiplicity 是整个 capped model 的 QM / RESP 权威值。

### 4.4 `NCAAConformer`

`NCAAConformer` 是 representative 和 RESP conformer 共用的数据结构：

- `label`
- `phi_deg`
- `psi_deg`
- `energy`
- `model`

`optimize_capped_reference(...)` 返回 representative conformer。源码默认 label 是 `"ref"`，`phi_deg` 和 `psi_deg` 记录为 `0.0`，其 `model` 是优化后的 capped model。

`build_resp_conformers_from_reference(...)` 根据 chirality 生成 RESP conformers：

```text
L: alpha = (-60, -40), beta = (-120, -140)
D: alpha = ( 60,  40), beta = ( 120,  140)
```

每个 conformer 先通过 Scan API 的 `run_silent_scan(...)` 把 backbone `phi/psi` 转到目标角度，再用 `FixInternals` 固定 `phi/psi` 做优化。

### 4.5 `NCAAModelBundle`

`_prepare_ncaa_models(...)` 返回 `NCAAModelBundle`：

- `identity`
  - target identity 和 chirality。
- `representative`
  - 优化后的 capped representative。
- `conformers`
  - alpha / beta RESP conformers。

这个 bundle 是 workflow 前半段的临时对象，不直接导出。

### 4.6 `MultiRespPipelineResult`

`run_multiconformer_resp(...)` 返回 `MultiRespPipelineResult`：

- `model`
  - 已写入 RESP charges 的 representative model。
- `files`
  - 当前主要包含 capped RESP mol2。
- `resp_files`
  - merged ESP、resp1/resp2 输入输出、charge、calculated ESP、每个 conformer 的 Gaussian input/log/ESP、`target_chg`。
- `conformers`
  - 每个 RESP conformer 的 Gaussian / ESP sidecar metadata。
- `decision`
  - 使用的 `QMMethod`。

当前源码中，多构象 RESP 的 ESP conformer list 只包含 alpha / beta。representative model 作为 ordering reference 传入，用来校验 atom 数、承接第一套 RESP charge 并写出 capped mol2；它本身不是当前 `conformers` list 中的 Gaussian ESP conformer。

### 4.7 `NCAAAmberArtifacts` 和 `NCAAArtifacts`

`NCAAAmberArtifacts` 描述 AmberTools 和 refined template 相关路径：

- `capped_mol2`
- `gaff2_mol2`
- `ac`
- `mc`
- `prepin`
- `refined_prepin`
- `res`
- `newpdb`
- `frcmod`
- `refined_frcmod`

`NCAAArtifacts` 包装 `NCAAAmberArtifacts`，并增加：

- `target_capped_pdb`
- `tleap_input`
- `conformer_capped_pdbs`
- `files`
  - 面向 caller / report 的统一路径字典。

### 4.8 `NCAAWorkflowResult`

`run_ncaa_abinitio(...)` 最终返回 `NCAAWorkflowResult`：

- `identity`
- `representative`
- `conformers`
- `parameter_set`
- `torsion`
- `artifacts`
- `resp_files`
- `representative_model`
- `residue_model`
- `stage_timings`

它提供：

- `chirality`
  - `identity.chirality`。
- `representative_conformer`
  - `representative.label`。
- `files`
  - `artifacts.files`。

`residue_model` 是从 charged representative 中复制出的 target residue-only model，主要表达最终 residue template 对象，不是 RESP 或 mSeminario 的运行对象。

## 5. 构象与 RESP 链路

### 5.1 representative 优化

`optimize_capped_reference(...)` 在 `ACE-target-NME` 上运行几何优化：

1. `model_to_atoms(...)` 把 capped model 转成 ASE `Atoms`。
2. 从 `source_atoms` 复制优化阈值。
3. 使用 `source_atoms.calc` 作为 calculator。
4. 调用 `optimize_model_geometry(...)` 或带 `FixAtoms` 的 `optimize_atoms_geometry(...)`。
5. 通过 `get_potential_energy(...)` 记录优化后能量。

默认优化参数当前固定在 helper 默认值中：

```text
max_iter = 256
max_step = 0.2
```

这些不是 `NCAAAbinitioConfig` 当前公开字段。

### 5.2 alpha / beta conformer 生成

`build_resp_conformers_from_reference(...)` 从 representative model 生成 alpha / beta：

1. 根据 chirality 选择目标 `phi/psi`。
2. `_build_backbone_rotation_map(...)` 定位 `omega_pre`、`phi`、`psi`、`omega_post` 的 atom indices。
3. 计算从当前二面角到目标二面角的最短 360 度等价旋转。
4. 用 Scan API 的 `run_silent_scan(...)` 做 2D rigid scan。
5. 取 scan 最后一帧。
6. 用 `FixInternals` 固定 `phi/psi` 做最终优化。

scan 输出和 conformer 优化输出写在 `"<base>_work/ncaa/"` 下，文件名前缀来自 `parmfit_work_prefix(output, "ncaa")`。

### 5.3 多构象 RESP

`run_multiconformer_resp(...)` 运行 NCAA 的 RESP：

1. 为每个 conformer 写 Gaussian ESP input：
   - `<work_prefix>_alpha_resp.gjf`
   - `<work_prefix>_beta_resp.gjf`
2. 调用 Gaussian。
3. 用 `espgen` 从 Gaussian log 抽 ESP。
4. 合并所有 ESP 到 `<base>_all.esp`。
5. `resp.write_multiconformer_resp_input_files(...)` 写 `resp1.in` 和 `resp2.in`。
6. 运行两阶段 `resp`。
7. 从 `resp2.chg` 读取 charges。
8. 取第一套 representative atom ordering 对应的 charges。
9. 写 `target_chg`，供 AmberTools `antechamber -c rc -cf` 使用。
10. 把 charges 写回 `representative_model`。
11. 写 capped RESP mol2。

多构象 RESP 的化学约束规则在 `utils/resp.py`：

- 所有 models 必须有相同 residue / atom ordering。
- target residue atoms 会形成 group charge constraint，总电荷为 `config.charge`。
- stage 1 做 equivalent hydrogens grouping。
- stage 2 释放 `_free_stage2_atoms(...)` 选择的 atoms，并做 interstructure equivalencing。

NCAA route 当前没有 MetalAA 的 `chgmod` / `fixchg_resids` 语义。

## 6. AmberTools / prepin / frcmod 链路

### 6.1 `antechamber`

`build_ncaa_amber_artifacts(...)` 先调用 `_run_ambertools_build(...)`。

`_run_ambertools_build(...)` 对 RESP capped mol2 调两次 `antechamber`：

1. typed mol2：
   - input format: `mol2`
   - output format: `mol2`
   - atom type: `gaff2`
   - charge mode: `rc`
   - charge file: `target_chg`
   - 输出记录为 `gaff2_mol2`
2. AC file：
   - input format: `mol2`
   - output format: `ac`
   - atom type: `gaff2`
   - charge mode: `rc`
   - charge file: `target_chg`
   - 输出记录为 `ac`

`target_chg` 是 RESP 回填给 representative ordering 的 charge 文件。AmberTools 后续模板生成使用这套 charge。

### 6.2 mainchain `.mc`

`write_mainchain_mc(...)` 写 `RN.mc`：

- `HEAD_NAME N`
- `TAIL_NAME C`
- `MAIN_CHAIN ...`
- `OMIT_NAME ...`
- `PRE_HEAD_TYPE C`
- `POST_TAIL_TYPE N`
- `CHARGE <config.charge>`

`infer_mainchain_names(charged_residue)` 会在 target residue 内从 `N` 到 `C` 找 heavy-atom shortest path，并把 path 中间 atom names 写成 `MAIN_CHAIN`。因此 `N` 和 `C` 本身不会作为 `MAIN_CHAIN` 行写入。

`infer_terminal_omit_names(...)` 会处理 terminal extra hydrogens 或 extra carboxyl oxygens，避免 prepgen 把末端形式误带入最终 residue template。

### 6.3 `prepgen` 和 `parmchk2`

`run_prepgen(...)` 用 `ac + mc` 生成：

- `RN.prepin`
- `RN.res`
- `NEWPDB.PDB`

`run_parmchk2(...)` 随后用 `prepi` 格式检查 `RN.prepin`，生成：

- `RN.frcmod`

`_materialize_amber_artifacts(...)` 会把 `prepgen` 和 `parmchk2` 的正式 template 复制到 `"<base>_work/"` 根目录：

- `<base>_work/RN.prepin`
- `<base>_work/RN.frcmod`

中间 `ac/mc/res/newpdb/gaff2_mol2` 保留在 `"<base>_work/ncaa/"`。

### 6.4 Maple refined template

`build_ncaa_export_bundle(...)` 调用 `_write_refined_amber_files(...)` 生成 Maple refined template：

1. 读取 original `RN.prepin`。
2. 跳过 `DUMM` 行，解析 prepin atom rows。
3. 以 target residue atom names 对应 representative global indices。
4. 从 final `CorrectionParameterSet` 读取 old atom type、RESP charge 和 mass。
5. 用 `allocate_maple_atom_types(...)` 分配 Maple atom types。
6. 重写 `prepin` atom type，输出 `RN_maple.prepin`。
7. 从 full capped parameter set 切出 residue-only bonds / angles / dihedrals / impropers / nonbonds。
8. 写 `RN_maple.frcmod`。
9. 插入 ff14SB / gaff2 peptide boundary cross terms。

Maple refined template 的目标是只包含 target residue 的 refined parameters，同时保留它与标准 peptide backbone 拼接时需要的边界参数。

## 7. mSeminario 与 torsion refinement 链路

### 7.1 stage 0 parameter set

`_refine_ncaa_parameters(...)` 先把 charged representative model 转回 ASE `Atoms`：

```text
representative_model -> representative_atoms
```

然后：

1. 复用 `source_atoms.calc`。
2. 复制优化阈值。
3. `patch_frcmod_crossterms(frcmod_path)` patch AmberTools frcmod 中的 cross terms。
4. `build_correction_parameter_set(representative_atoms, typed_mol2_path, frcmod_path)` 构建 stage 0 `CorrectionParameterSet`。

这里的 `typed_mol2_path` 是 `gaff2_mol2`，`frcmod_path` 是 original `RN.frcmod`。

### 7.2 mSeminario

NCAA 在 representative capped model 上做 Hessian / mSeminario：

1. `get_cartesian_hessian(representative_atoms)` 从 calculator 取 Cartesian Hessian。
2. `apply_mseminario(...)` 更新 stage 0 bonds 和 angles。
3. `config.vib_scale` 作为 modified Seminario 缩放因子传入。

如果 calculator 没有 `get_hessian()` 或返回的 Hessian shape 不合法，NCAA 会在这个阶段失败。

### 7.3 shared torsion refinement

mSeminario 之后进入 shared torsion workflow：

```text
run_torsion_workflow(
    atoms=representative_atoms,
    parameter_set=stage0_result,
    params=config.torsion,
    runtime=TorsionScanRuntime(...),
    center_bond_filter=build_ncaa_center_bond_filter(representative_model),
)
```

NCAA 的 `center_bond_filter` 只保留 target residue 内部、至少一侧属于 sidechain R group、并且不是纯 backbone-backbone 的 center bonds。它会排除 cap bonds 和 target backbone 主链键。

`run_torsion_workflow(...)` 返回 `TorsionWorkflowResult`。NCAA 取 `torsion.final_parameter_set` 作为最终 `parameter_set`，再写 refined `prepin/frcmod`。

## 8. 输出与失败点

### 8.1 正式输出

NCAA route 的正式输出位于 `"<base>_work/"` 根目录：

```text
<base>_work/RN.prepin
<base>_work/RN.frcmod
<base>_work/RN_maple.prepin
<base>_work/RN_maple.frcmod
<base>_work/<base>_ncaa_tleap.in
```

这里的 `RN` 来自 `config.rn`，默认是 `MOL`。

### 8.2 过程文件

NCAA 过程文件主要位于 `"<base>_work/ncaa/"`：

- capped RESP mol2
- gaff2 typed mol2
- `RN.ac`
- `RN.mc`
- `RN.res`
- `NEWPDB.PDB`
- target capped PDB
- alpha / beta capped optimized PDB
- alpha / beta Gaussian ESP input / log / ESP
- merged ESP
- resp1 / resp2 输入输出
- `target_chg`

shared torsion scan 和 refinement 文件位于 `"<base>_work/torsionfit/"`。

### 8.3 log 输出

`format_ncaa_start_lines(...)` 输出：

- target selector
- output residue name
- chirality
- charge / multiplicity
- QM ESP method

`format_ncaa_final_lines(...)` 输出：

- route completed
- Maple atom-type remapping
- `tleap reference`
- NCAA route stage timings
- capped mol2 / ac
- original Amber template
- refined Amber template
- `tleap.in`

当前 NCAA route 会写一个简单的 `tleap.in`，用于加载用户已准备好的 protein PDB、`RN_maple.prepin`、`RN_maple.frcmod`、OPC water 和 HFE ion 参数。log 中的 `tleap reference` 仍保留，用于快速查看关键加载行。

### 8.4 关键失败点

当前实现中常见失败点包括：

- `pdb` 缺失、路径不存在或 PDB 无法解析。
- `target` 不能唯一定位 residue。
- target residue 不是 `protein` kind。
- target residue 缺少 `N/CA/C`。
- target residue 没有与 `CA` 相连的 sidechain heavy atom。
- representative geometry optimization 不收敛。
- alpha / beta scan 或 constrained optimization 不收敛。
- 多构象 RESP models 的 residue / atom ordering 不一致。
- RESP 返回 charge 数量不能对应 representative model atom 数。
- Gaussian、`espgen`、`resp`、`antechamber`、`prepgen`、`parmchk2` 缺失或失败。
- `infer_mainchain_names(...)` 无法从 target residue 的 `N` 到 `C` 找到 heavy-atom path。
- `prepgen` 输出的 `prepin` atom rows 无法与 charged residue atom names 对应。
- final `CorrectionParameterSet` 缺少 prepin atom 对应的 nonbond 或 mass 参数。
- calculator 缺少 forces / energy，导致优化或 scan 失败。
- calculator 缺少 Hessian，导致 mSeminario 失败。
- `format_tleap_add_atom_types_lines(...)` 无法从 old atom type 推断 hybridization。

## 9. 与 MetalAA / MCPB 思路对照

### 9.1 与 MetalAA 的共同点

NCAA 和 MetalAA 都是 `parmfit(method=abinitio)` 的正式路线：

- 都从 `Parmfit.run()` 进入 `Abinitio.run()`。
- 都依赖 PDB target selector。
- 都使用 MAPLE 的 `source_atoms.calc` 做优化和 Hessian。
- 都使用 Gaussian / ESP / RESP 生成 QM-derived charges。
- 都通过 modified Seminario 更新 bond / angle 力常数。
- 都导出 Amber 可用的参数文件。

### 9.2 与 MetalAA 的关键区别

NCAA 的路线更接近 nonstandard residue template construction：

- MetalAA target 是 `ion`，NCAA target 是 `protein` residue。
- MetalAA 自动识别 donor core 和 environment，NCAA 只围绕一个 target residue。
- MetalAA 有 `large_model/site_model` 两层模型，NCAA 有 `capped representative` 和 `residue template` 两层语义。
- MetalAA 的 RESP 在 optimized `large_model` 上跑，再投影到 `site_model`。
- NCAA 的 RESP 在 alpha / beta capped conformers 上跑，再回填到 representative ordering。
- MetalAA 导出 site `mol2/frcmod`。
- NCAA 导出 Amber `prepin/frcmod` 和 Maple refined `prepin/frcmod`。
- MetalAA 不运行 shared torsion refinement。
- NCAA 在 mSeminario 之后运行 shared torsion refinement。

### 9.3 与 MCPB 的关系

`utils/MCPB/` 是 MetalAA 的历史范本和参考包，不是 NCAA 运行期依赖。

NCAA 与 MCPB 的关系更弱。它继承的是 broader Amber 参数化思路：

- 用 QM / RESP 获得 residue charges。
- 用 AmberTools 生成 template。
- 用 frcmod 表达缺失或 refined parameters。
- 用 final files 给 `tleap` 消费。

NCAA 没有继承 MCPB 的 metal-specific 机制：

- 没有 metal center。
- 没有 donor atom / metal bond graph。
- 没有 MCPB fingerprint 文件。
- 没有 small / standard / large metal model 系统。
- 没有 bonded / nonbonded metal ion model 切换。
- 没有 multi-metal coupled workflow。
- 没有自动 `tleap` assembly。

## 10. 设计取舍

当前 NCAA route 的主要取舍是：

- 单 target residue
  - workflow 假设只参数化一个 residue selector，不处理多个 NCAA 同时耦合拟合。
- capped representative
  - 用 `ACE-target-NME` 模拟 peptide 环境，同时让最终导出回到 residue template。
- alpha / beta RESP coverage
  - 用 chirality 决定两套代表性 backbone conformers，避免只对单一构象拟合 charges。
- AmberTools template ownership
  - 去帽、head/tail、mainchain 和 residue template 结构交给 `prepgen`。
- Maple refinement ownership
  - AmberTools 负责原始 `prepin/frcmod`，MAPLE 负责 mSeminario、torsion refinement、Maple atom type remapping 和 refined `frcmod`。
- 写出简单 `tleap.in`
  - workflow 只写加载脚本，不自动运行 `tleap`，也不接管 protein PDB 准备。
- 不引入 MetalAA 电荷策略
  - NCAA 不使用 `chgmod`、`fixchg_resids`、`large_charge` 或 site-only cap 标准电荷回填。
- 不设计未来 backend
  - 当前 QM backend 是 Gaussian。其它 backend 需要先在共享 QM interface 中实现。
