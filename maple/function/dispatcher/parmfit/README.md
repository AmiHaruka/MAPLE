# Parmfit Dispatcher 技术报告

`parmfit` 是 MAPLE 中负责力场参数构建与参数修正的 dispatcher。它本身只处理输入解释、route 分发和结果返回；具体化学模型由 `correction`、`MetalAA` 和 `NCAA` 子系统承担。

专项说明：

- [ARCH_PARMFIT.md](./ARCH_PARMFIT.md)
- [CORRECTION.md](./CORRECTION.md)
- [ARCH_METAL_ABINITIO.md](./ARCH_METAL_ABINITIO.md)
- [ARCH_NCAA_ABINITIO.md](./ARCH_NCAA_ABINITIO.md)

## 1. 总览

当前 `parmfit` 有三条正式 workflow：

| Route | 入口条件 | 领域实现 | 主要产物 |
|---|---|---|---|
| `method=correction` | 输入 `mol2` 和带 calculator 的结构 | `correction/` + `utils/TorsionFit/` | refined Amber mol2/frcmod 与 GROMACS top/gro |
| `method=abinitio`, target 为 ion | PDB 中 target 被识别为金属/离子 | `utils/MetalAA/` | metal site mol2、final metal frcmod、tleap PDB/input/out |
| `method=abinitio`, target 为 protein residue | PDB 中 target 被识别为蛋白残基 | `utils/NCAA/` | NCAA prepin/frcmod、processed protein PDB、tleap input/out |

顶层分发关系：

```text
Parmfit
+-- method=correction
|   `-- correction/
`-- method=abinitio
    +-- target_kind=ion     -> utils/MetalAA/
    `-- target_kind=protein -> utils/NCAA/
```

`abinitio` 只支持 `ion` 和 `protein` target。其它 target kind 会明确失败。

## 2. 输入契约

`Parmfit.run()` 的 `method` 识别顺序是：

```text
constructor method > params["method"] > "correction"
```

输入文件规则：

- `method=correction` 必须提供 `mol2`。
- `method=abinitio` 必须提供 `pdb` 和 `target`。
- `mol2`、`pdb` 会被解析为绝对路径，并要求文件存在。
- `abinitio` route 的真实结构来自 PDB；inline coordinate block 只用于 calculator bootstrap。

`abinitio` 使用统一的 charge/multiplicity 输入：

```text
cmo = "<charge> <multiplicity> [oxidation]"
```

- 前两项写入 `Atoms.info["charge"]` 和 `Atoms.info["mult"]`。
- 第三项 `oxidation` 只供 MetalAA 识别金属氧化态。
- `cmo` 未提供时默认 `"0 1"`。

## 3. 共用力场表达

MAPLE 的内部 classical MM 计算遵循 AMBER 风格。总能量写作：

```text
E_MM = E_bond + E_angle + E_proper + E_improper + E_vdw + E_elec
```

其中 bonded terms 为：

```text
E_bond    = sum_b k_b (r_b - r0_b)^2
E_angle   = sum_a k_a (theta_a - theta0_a)^2
E_torsion = sum_t sum_n k_n [1 + cos(n phi_t - gamma_n)]
```

非键项使用 AMBER 常见的 12-6 Lennard-Jones 和 Coulomb 形式：

```text
E_vdw  = epsilon_ij [(R_ij/r_ij)^12 - 2(R_ij/r_ij)^6]
E_elec = 332.05221729 q_i q_j / r_ij
```

1-2 和 1-3 nonbonded pair 被排除，1-4 pair 使用：

```text
SCEE = 1.2
SCNB = 2.0
```

`mechanics.py` 的 torsion angle 已按 AMBER/sander/cpptraj convention 修正；`readparm.py` 和 `outputparm.py` 只做 degree/radian 转换，不额外改变 phase。

## 4. Correction Route

Correction 从已有 ligand/cofactor `mol2` 出发，自动生成初始 GAFF2 参数并修正局部 bonded terms。

```text
mol2
 -> parmchk2
 -> initial CorrectionParameterSet
 -> geometry optimization
 -> Hessian + modified Seminario
 -> stage0 parameter set
 -> TorsionFit
 -> final parameter set
 -> Amber/GROMACS export
```

关键点：

- `parmchk2` 自动生成 `<base>_original.frcmod`。
- `mSeminario` 只更新 BOND/ANGLE：`kBond/rEq/kTheta/thetaEq`。
- DIHE 来自原始 GAFF2/parmchk2 与 TorsionFit，不由 mSeminario 更新。
- TorsionFit 只改变 selected proper torsion，不改变 nonbonded、improper 或拓扑。
- Amber 导出采用 Maple atom type rename，确保 tleap 能按 exact atom type pattern 找到局部修正项。
- GROMACS 导出写 explicit interaction rows，可作为检查内部参数对象是否完整的独立视角。

## 5. TorsionFit Route

TorsionFit 被 Correction 和 NCAA 复用。它的目标是拟合固定 scan frames 上的相对能量 profile，而不是假设体系是理想单自由度 torsion。

对每个 center bond：

```text
QM_rel_i = QM_i - min(QM)
fit_target_rel_i = QM_rel_i - MM_base_rel_i
```

`MM_base_rel` 表示把当前 center-bond proper torsion contribution 移除后的 MM relative profile。

Stage1 是 restrained linear least squares：

```text
min ||B k - y||^2 + lambda sum_j w_j (k_j - k0_j)^2
```

当前 Stage1 特点：

- existing term 进入弱 prior refit，可被压到接近 0。
- zero-amplitude 原始 term 不进入 active fitting，但会保留，避免 Amber 缺项。
- spectral helper 从 `n = 1, 2, 3, 4, 6` 中选择候选 period。
- shared group 内通过 phase migration 把 representative path 的 spectral phase 迁移到实际 torsion paths。
- 单个 center bond 默认最多新增 3 个 spectral slots。

Stage2 是 global direct `kPhi/phase` refinement：

```text
x = [k_1, gamma_1, k_2, gamma_2, ...]

a_j = k_j cos(gamma_j)
b_j = k_j sin(gamma_j)

MM_s(x) = constant_s + cos_basis_s @ a + sin_basis_s @ b
```

Stage2 不新增 slot；它用 L-BFGS-B 直接优化 AMBER torsion 的 `kPhi/phase`，并用 coefficient-space prior 约束参数不要无意义偏离 Stage1。`torsion_refine_rounds > 1` 时执行 fast MM cycle：不重新 scan、不重新跑 MLIP，只在已有 frames 上用 cached phi 和 torsion basis 快速重算 MM profiles。每轮只有在 finite total loss 改善时才接受，否则保留 Stage1 或上一轮 best result。

可选 `torsion_ensemble` 只进入 Stage2：从优化后结构做 15 degree rigid random rotor sampling，经 CPU MM filter 选出少量构象，再对 selected frames 做 MLIP single point。ensemble target 与 scan reference 对齐，作为 `w_ensemble L_ensemble` 加入 Stage2 loss；`corr.out` 仍只展开 scan torsion energy trace。

## 6. MetalAA Route

MetalAA 处理 `target_kind=ion` 的金属位点。

```text
PDB + target ion
 -> metal/donor/cofactor recognition
 -> large_model
 -> geometry optimization + RESP
 -> optimized donor re-selection
 -> site_model
 -> RESP charge projection
 -> Hessian + mSeminario
 -> final metal frcmod + tleap validation
```

MetalAA 的 charge 语义：

```text
config.charge = metal + nonprotein cofactor/ligand fragments
config.mult   = QM multiplicity
config.oxy    = optional metal oxidation identity
```

large model 总电荷为：

```text
Q_large = config.charge + sum(protein formal residue charges)
```

`oxy` 不参与 total charge，只用于金属形式电荷、Gaussian ReadRadii 和离子参数身份。

`cfmol2` 用于 HEM 等 cofactor：

- 注入 cofactor atom type 和 charge。
- 注入 explicit bonds，避免大环只靠距离猜键。
- 通过 `parmchk2 -s gaff2 -a Y` 生成 cofactor orig frcmod。
- orig frcmod 只作为中间参数源；final `*_metal.frcmod` 是唯一 cofactor/metal 参数入口。

MetalAA final frcmod 合并：

```text
cofactor orig full parameters
+ renamed donor inherited parameters
+ metal-donor BOND/ANGLE fitted terms
+ metal-related zero DIHE
```

## 7. NCAA Route

NCAA 处理 `target_kind=protein` 的单残基 residue template 构建。

```text
PDB + target residue
 -> ACE-target-NME representative
 -> representative optimization
 -> alpha/beta conformers
 -> multiconformer RESP
 -> antechamber/prepgen/parmchk2
 -> Hessian + mSeminario
 -> sidechain TorsionFit
 -> RN_maple.prepin/RN_maple.frcmod
 -> processed protein PDB + tleap validation
```

NCAA 与 AutoNACC 的关系：

- 两者后半段都走 AmberTools template：RESP charge -> antechamber -> prepgen -> parmchk2 -> prepin/frcmod。
- MAPLE 不接管 AutoNACC 的前期 PDB 准备；用户需要提供已准备好的 protein PDB。
- MAPLE 额外加入 alpha/beta 多构象 RESP、mSeminario 和 TorsionFit refinement。

NCAA 只拟合 target residue 内部、与 R group 相关的 center bond。主链 peptide boundary 通过 exact cross terms 补齐：

```text
previous C - target N
target C   - next N
```

这些补项按真实上下游残基动态生成，不全局铺一批无实际边界的组合。

## 8. 结果输出

Correction 输出：

```text
<base>_work/<base>_original.frcmod
<base>_work/<base>_maple.mol2
<base>_work/<base>_maple.frcmod
<base>_work/<base>_maple.top
<base>_work/<base>_maple.gro
```

`corr.out` 的 torsion energy trace 使用同一 scan reference 汇总：

```text
MLIP_ref / orig_ref / stage0_ref / stage1_ref / stage2_ref
```

其中 `orig_ref` 是原始 GAFF2/parmchk2 MM profile，`stage2_ref` 是最终接受参数的 profile。

MetalAA 输出：

```text
<base>_work/<base>_metal_large_raw.pdb
<base>_work/<base>_metal_large_opt.pdb
<base>_work/<base>_metal_site.mol2
<base>_work/<base>_metal.frcmod
<base>_work/<base>_metal_tleap.pdb
<base>_work/<base>_metal_tleap.in
<base>_work/<base>_metal_tleap.out
```

NCAA 输出：

```text
<base>_work/RN.prepin
<base>_work/RN.frcmod
<base>_work/RN_maple.prepin
<base>_work/RN_maple.frcmod
<base>_work/<base>_ncaa_tleap.pdb
<base>_work/<base>_ncaa_tleap.in
<base>_work/<base>_ncaa_tleap.out
```

`abinitio` 结尾统一输出 `PARMFIT ABINITIO RESULT`，包括 route、target、主要文件、耗时、tleap status 和下一步命令。

## 9. 当前默认值

重要默认值以当前源码为准：

```text
method                  = correction
cmo                     = "0 1"
torsionfit              = true
torsion_steps           = 72
torsion backend         = cgbs
constraint_mode         = projected
torsion_refine_rounds   = 2
torsion_refine_max_iter = 10
torsion_refine_tol      = 1.0e-6
report_debug            = false
torsion_ensemble        = true
torsion_ensemble_ratio  = 0.3
torsion_ensemble_weight = 0.50
NCAA rn                 = MOL
NCAA prom/watm/ionm     = ff14SB / tip3p / 12_6
NCAA QM                 = gaussian / HF / 6-31G(d)
MetalAA prom/watm/ionm  = ff14SB / tip3p / 12_6
MetalAA QM              = gaussian / PBE1PBE / def2SVP
```

`qm_backend` / `resp_backend` 属于 QM/RESP 配置；torsion scan backend 是 TorsionFit 的 `backend` 字段。
