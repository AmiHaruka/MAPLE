# MAPLE Parmfit Dispatcher 架构

本文档描述 `parmfit` dispatcher 的当前源码结构。它是实现说明，不定义未来行为。

## 1. 设计定位

`parmfit` 的职责是把用户输入转成一条清楚的参数化 workflow。它不直接实现金属识别、RESP、电荷投影、AmberTools 模板构建或 torsion fitting。

```text
parmfit.py
 |
 +-- correction/correction.py
 |
 `-- abinitio/abinitio.py
      +-- utils/MetalAA/
      `-- utils/NCAA/
```

当前有效 route：

| Route | 触发条件 | 核心对象 |
|---|---|---|
| Correction | `method=correction` | `CorrectionParameterSet` |
| MetalAA | `method=abinitio` 且 target kind 为 `ion` | `large_model` / `site_model` |
| NCAA | `method=abinitio` 且 target kind 为 `protein` | `ACE-target-NME` / residue template |

## 2. 顶层入口

`Parmfit.run()` 做四件事：

1. 读取外部 parmfit config。
2. 规范化 `mol2` / `pdb` 路径。
3. 根据 `method` 选择 correction 或 abinitio。
4. 返回子 workflow 的 result。

`abinitio` 额外解析：

```text
cmo = "<charge> <multiplicity> [oxidation]"
```

并写入：

```text
Atoms.info["charge"]
Atoms.info["mult"]
Atoms.info["spin"] = (mult - 1) / 2
Atoms.info["oxy"]  # optional
```

`cm` 不再是当前接口语义；当前接口是 `cmo`。

## 3. Ab Initio Route Selector

`abinitio/abinitio.py` 是薄 selector：

```text
read PDB
 -> find target residue
 -> classify target kind
 -> MetalAA if ion
 -> NCAA if protein
```

它不处理具体化学。MetalAA / NCAA 结束后，`abinitio/report.py` 生成统一结尾块：

```text
PARMFIT ABINITIO RESULT
Status / Route / Target / Main products / Stage timing / Next step / Tleap status
```

MetalAA 和 NCAA 都会生成并运行 tleap input；summary 从 `*_tleap.out` 中提取 `Errors / Warnings / Notes` 和关键检查信息。

## 4. Active Package Layout

```text
parmfit/
|-- parmfit.py
|-- abinitio/
|   |-- abinitio.py
|   `-- report.py
|-- correction/
|   |-- correction.py
|   |-- config.py
|   |-- parameters.py
|   |-- workflow.py
|   |-- artifacts.py
|   `-- report.py
|-- utils/
|   |-- MetalAA/
|   |-- NCAA/
|   |-- TorsionFit/
|   |-- Scan/
|   |-- runtime.py
|   |-- interface.py
|   |-- resp.py
|   |-- readparm.py
|   |-- outputparm.py
|   |-- mechanics.py
|   |-- mSeminario.py
|   |-- structure.py
|   |-- context.py
|   |-- model.py
|   `-- capping.py
`-- test/
```

当前文件边界：

| 文件/包 | 责任 |
|---|---|
| `correction/config.py` | correction 用户输入解析 |
| `correction/parameters.py` | `parmchk2 -> CorrectionParameterSet` |
| `correction/workflow.py` | correction 主流程 |
| `correction/artifacts.py` | correction result 与 Amber/GROMACS 文件导出 |
| `utils/MetalAA/recognize.py` | metal core / donor / cfmol2 识别 |
| `utils/MetalAA/models.py` | large/site model 构建 |
| `utils/MetalAA/charges.py` | large charge 推断与 RESP charge 投影 |
| `utils/MetalAA/artifacts.py` | MetalAA typing、frcmod、mol2、tleap 文件 |
| `utils/NCAA/models.py` | NCAA identity、capping、conformer、residue graph |
| `utils/NCAA/artifacts.py` | NCAA AmberTools、remap、frcmod、tleap 文件 |
| `utils/TorsionFit/workflow.py` | scan/cache/filter 与 fitting 调用 |
| `utils/TorsionFit/basis.py` | local/global problem 与 fast MM profile cache |
| `utils/TorsionFit/stage1.py` | spectral shared-group restrained LLS |
| `utils/TorsionFit/stage2.py` | global Stage2 loss refinement with k/phase and coeff_ab optimizer paths |
| `utils/TorsionFit/ensemble.py` | optional rigid rotor ensemble target for Stage2 |
| `utils/TorsionFit/fit.py` | Stage1/Stage2/cycle 组合入口 |
| `utils/Scan/optimizer.py` | parmfit 内部 LBFGS、projected optimizer、CGWS/CGBS |

历史备份目录和 `utils/?/` 不属于 active workflow，除非当前源码显式 import。

## 5. Shared Parameter Object

`CorrectionParameterSet` 是 correction、NCAA 和 TorsionFit 的共享参数对象：

```text
mol2 topology
frcmod database
bonds
angles
dihedrals
impropers
nonbonds
unmatched terms
```

它承担两类角色：

- 作为内部 MM energy 和 TorsionFit 的计算对象。
- 作为 Amber/GROMACS export 的源对象。

因此任何修正必须最终落回对应 section：

```text
BOND     -> Bond.kBond / Bond.rEq
ANGLE    -> Angle.kTheta / Angle.thetaEq
DIHE     -> FourierTerm.kPhi / period / phase
IMPROPER -> FourierTerm.kPhi / period / phase
NONBON   -> charge / rmin_half / epsilon
```

## 6. Shared MM Semantics

`utils/mechanics.py` 是内部 MM energy evaluator。它用于 TorsionFit profile、scan filtering 和 fast cycle。

```text
E_MM = E_bond + E_angle + E_proper + E_improper + E_vdw + E_elec
```

Bond/angle/proper torsion：

```text
E_b = k_b (r - r0)^2
E_a = k_a (theta - theta0)^2
E_t = sum_n k_n [1 + cos(n phi - gamma_n)]
```

Torsion angle 使用 AMBER convention。`phase` 在内部以 radian 保存，导出时转换为 degree，不做额外 offset。

1-4 scaling：

```text
electrostatic: / 1.2
vdw:           / 2.0
```

## 7. Scan 与 Optimizer 边界

`utils/Scan/` 是 parmfit 的正式 scan subsystem。

Public entry:

```python
run_silent_scan(...)
read_scan_final_atoms(...)
```

当前 parmfit 不依赖 optimization dispatcher 的 LBFGS。普通几何优化和 scan projected optimization 都使用：

```text
utils/Scan/optimizer.py::LBFGS
```

用法区别：

| 用途 | `use_projection` | `use_line_search` | 输出 |
|---|---:|---:|---|
| 普通 geometry optimization | false | false | 不写 scan trajectory |
| `fixinternals` relaxed scan | false | false | 写 scan final xyz |
| projected scan | true | optional | 写 scan final xyz |

## 8. Route Dataflow

### 8.1 Correction

```text
mol2
 -> parmchk2
 -> initial parameter set
 -> geometry optimization
 -> Hessian + mSeminario
 -> stage0 parameter set
 -> TorsionFit
 -> final parameter set
 -> Amber/GROMACS artifacts
 -> CorrectionWorkflowResult
```

`Correction.run()` 保存并返回 `CorrectionWorkflowResult`。`Correction.result` 是兼容属性，返回 `workflow_result.final_parameter_set`。

### 8.2 MetalAA

```text
PDB target ion
 -> recognize core/cofactor/donor atoms
 -> large_model
 -> optimization + RESP
 -> optimized donor re-selection
 -> site_model
 -> RESP charge projection
 -> Hessian + mSeminario
 -> final metal frcmod
 -> tleap validation
 -> MetalWorkflowResult
```

MetalAA 不用 TorsionFit。金属相关 BOND/ANGLE 来自 Hessian + mSeminario；cofactor 内部参数从 cfmol2 orig frcmod 继承和 remap。

### 8.3 NCAA

```text
PDB target residue
 -> NCAA identity and chirality
 -> ACE-target-NME representative
 -> alpha/beta conformers
 -> multiconformer RESP
 -> AmberTools template
 -> Hessian + mSeminario
 -> sidechain TorsionFit
 -> refined prepin/frcmod
 -> processed protein PDB
 -> tleap validation
 -> NCAAWorkflowResult
```

NCAA 使用 TorsionFit，但通过 center-bond filter 排除纯 backbone/cap bonds。

## 9. Output Contract

Correction output:

```text
<base>_work/<base>_original.frcmod
<base>_work/<base>_maple.mol2
<base>_work/<base>_maple.frcmod
<base>_work/<base>_maple.top
<base>_work/<base>_maple.gro
```

`corr.out` 同时给出精简 torsion energy trace：

```text
MLIP_ref / orig_ref / stage0_ref / stage1_ref / stage2_ref
```

`orig_ref` 是原始 GAFF2/parmchk2 relative MM profile；`stage2_ref` 是最终接受参数对 scan frames 的 relative MM profile。

MetalAA output:

```text
<base>_work/<base>_metal_large_raw.pdb
<base>_work/<base>_metal_large_opt.pdb
<base>_work/<base>_metal_site.mol2
<base>_work/<base>_metal.frcmod
<base>_work/<base>_metal_tleap.pdb
<base>_work/<base>_metal_tleap.in
<base>_work/<base>_metal_tleap.out
```

NCAA output:

```text
<base>_work/RN.prepin
<base>_work/RN.frcmod
<base>_work/RN_maple.prepin
<base>_work/RN_maple.frcmod
<base>_work/<base>_ncaa_tleap.pdb
<base>_work/<base>_ncaa_tleap.in
<base>_work/<base>_ncaa_tleap.out
```

## 10. Design Boundaries

- `Parmfit` 保持 dispatcher，不承载化学算法。
- `Abinitio` 保持 target route selector。
- `MetalAA` 与 `NCAA` 保持独立模型定义；二者都用 RESP，但 charge projection 和 deployment 语义不同。
- `TorsionFit` 只修正 selected proper torsions。
- `mSeminario` 只修正 BOND/ANGLE。
- `mechanics.py` 是内部快速 MM evaluator，不替代 sander/pmemd 的生产能量引擎。
- `tleap` 成功只说明 AMBER pattern 匹配完整；参数数值正确性仍由 source parameter set、mSeminario、TorsionFit 和外部能量对照共同决定。
