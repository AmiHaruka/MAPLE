# MAPLE Parmfit NCAA Ab Initio 架构

顶层架构见 [ARCH_PARMFIT.md](./ARCH_PARMFIT.md)。本文档描述 `parmfit(method=abinitio)` 中 target kind 为 `protein` 时的 NCAA route。

## 1. 定位

NCAA route 为单个目标蛋白残基构建 AMBER residue template：

```text
PDB + target residue
 -> ACE-target-NME model
 -> multiconformer RESP
 -> AmberTools template
 -> mSeminario
 -> sidechain TorsionFit
 -> RN_maple.prepin/RN_maple.frcmod
 -> processed protein PDB + tleap validation
```

它不是 MetalAA/MCPB route。NCAA 的核心对象是目标残基的 capped model，而不是 metal core。

## 2. 模块边界

| 文件 | 责任 |
|---|---|
| `config.py` | 解析 NCAA 输入、QM/RESP、protein/water/ion model、TorsionFit 参数 |
| `models.py` | target identity、chirality、ACE/NME capped model、alpha/beta conformers、residue graph |
| `artifacts.py` | AmberTools、prepin/frcmod、Maple atom type remap、boundary terms、tleap PDB/input |
| `workflow.py` | 串联 NCAA 主流程 |
| `report.py` | route progress 文案 |

共享层：

- `runtime.py`: optimization、Hessian、多构象 RESP。
- `resp.py`: RESP 输入、ESP 合并、charge 回填。
- `interface.py`: Gaussian、AmberTools、tleap。
- `mSeminario.py`: BOND/ANGLE 参数。
- `TorsionFit/`: target residue sidechain torsion refinement。

## 3. 输入语义

核心输入：

```text
pdb
target
cmo = "<charge> <multiplicity>"
rn  = output residue name
```

当前默认值：

```text
rn       = MOL
prom     = ff14SB
watm     = tip3p
ionm     = 12_6
QM       = gaussian / HF / 6-31G(d)
bonded   = mseminario
```

`prom` 支持 `ff14SB` 和 `ff19SB`。`ionm` 当前允许用户指定自定义值；NCAA config 不强制校验 ion parameter set。

## 4. NCAA Identity

`identity_ncaa(target_residue)` 提取：

- residue key。
- 原始 PDB resname。
- chirality。
- sidechain anchor。

chirality 由 `N/CA/C/sidechain_anchor` 的 signed volume 判定。缺少 `N/CA/C` 或缺少与 `CA` 相连的 sidechain heavy atom 会失败。

## 5. Capped Representative

目标残基被复制为：

```text
ACE - target(rn) - NME
```

`build_capped_ncaa_model(...)` 会记录：

```text
segment_sizes = {ace, residue, nme}
target_key
charge
mult
```

representative 优化使用 `source_atoms.calc`。优化后的 capped model 是后续 RESP ordering、mSeminario 和 TorsionFit 的共同参考结构。

## 6. Alpha/Beta RESP Conformers

NCAA 生成 alpha/beta 两个 RESP conformers。

当前目标 backbone angles：

```text
L: alpha = (-60,  -40), beta = (-120, -140)
D: alpha = ( 60,   40), beta = ( 120,  140)
```

每个 conformer：

1. 从 optimized representative 复制。
2. 用 Scan API 把 `phi/psi` 转到目标角度。
3. 用 `FixInternals` 固定 `phi/psi` 做局部优化。
4. 写 conformer capped PDB。

## 7. 多构象 RESP

多构象 RESP 的对象是 alpha/beta conformers。流程：

```text
alpha/beta models
 -> Gaussian ESP input/log
 -> espgen
 -> merged ESP
 -> resp stage 1/2
 -> target_chg
 -> charged representative model
 -> capped RESP mol2
```

RESP 约束：

- 所有 conformers 必须有一致 residue/atom ordering。
- target residue atoms 作为 group，总电荷为 `config.charge`。
- stage 1 做等价 H grouping。
- stage 2 释放可拟合 atom，并做 interstructure equivalencing。

NCAA 不使用 MetalAA 的 `chgmod/fixchg_resids`。

## 8. AmberTools Template

AmberTools 后半段与 AutoNACC 思路一致：

```text
capped RESP mol2
 -> antechamber gaff2 mol2
 -> antechamber ac
 -> mainchain .mc
 -> prepgen
 -> RN.prepin
 -> parmchk2
 -> RN.frcmod
```

`.mc` 文件定义：

```text
HEAD_NAME N
TAIL_NAME C
MAIN_CHAIN ...
OMIT_NAME ...
PRE_HEAD_TYPE C
POST_TAIL_TYPE N
CHARGE <config.charge>
```

`infer_mainchain_names(...)` 在 target residue 内找 `N -> C` heavy-atom shortest path。`infer_terminal_omit_names(...)` 去除末端多余 H/O，避免 prepgen 生成 terminal template。

## 9. mSeminario 与 Stage0

`_refine_ncaa_parameters(...)` 用 representative capped model 构建 stage0 parameter set：

```text
gaff2 mol2 + RN.frcmod
 -> CorrectionParameterSet
 -> Hessian
 -> mSeminario/Seminario
 -> stage0 parameter set
```

mSeminario 只更新：

```text
BOND:  kBond, rEq
ANGLE: kTheta, thetaEq
```

不更新 DIHE/IMPROPER/NONBON。

## 10. NCAA TorsionFit

NCAA 复用 shared `TorsionFit`，但会加 center-bond filter。

保留的 center bond 必须：

- 属于 target residue 内部。
- 至少一侧与 sidechain R group 相关。
- 不是纯 backbone-backbone。
- 不属于 cap。

目标是只修正 R group 相关 torsion，不用 TorsionFit 拟合主链 peptide potential。

TorsionFit 数据链：

```text
stage0 parameter set
 -> sidechain center bonds
 -> constrained scan frames
 -> spectral shared-group Stage1
 -> Stage2 direct k/phase fast MM cycles
 -> final parameter set
```

Stage1 target：

```text
fit_target_rel = QM_rel - MM_base_rel
```

Stage2 target：

```text
L = L_scan + w_ensemble L_ensemble + L_prior
```

Stage2 不再做 checkpoint / per-center rollback；它直接优化 Stage1 slot 的 AMBER `kPhi/phase`，只有 finite total loss 改善时才接受。开启 `torsion_ensemble` 时，NCAA ensemble 只移动 R group mobile atoms，backbone/cap 保持冻结，extra frames 只作为 Stage2 额外约束。

## 11. Maple Atom Type Remap

`write_ncaa_amber_files(...)` 将 final capped parameter set 切成 target residue-only template。

步骤：

1. 读取 original `RN.prepin`。
2. 解析 non-DUMM atom rows。
3. 用 target residue atom names 对应 representative global indices。
4. 从 final `CorrectionParameterSet` 读取 old atom type、charge、mass。
5. 分配 Maple atom types，例如 `Z0/Z1/...`。
6. 重写 `RN_maple.prepin`。
7. 切出 residue 内部 BOND/ANGLE/DIHE/IMPROPER/NONBON。
8. 写 `RN_maple.frcmod`。
9. 插入 peptide boundary exact cross terms。

当前实现会重命名 target residue 的 template atom types。这个设计是为了让 mSeminario/TorsionFit 的局部参数能够以 exact atom type pattern 写入 frcmod，而不是污染全局 GAFF2/Amber atom type。

## 12. Peptide Boundary Exact Terms

NCAA residue 作为 custom `RN` template 后，tleap 无法自动用标准 amino acid library 的全部 peptide boundary pattern。MAPLE 因此动态补真实边界上的 exact terms。

只补：

```text
previous C - target N
target C   - next N
```

不补不存在的组合。

### 12.1 Head side

若上一残基与 target `N` 成肽键：

```text
prev C - target N
```

写入：

```text
BOND:  prev_C - target_N
ANGLE: prev_O - prev_C - target_N
ANGLE: prev_CA - prev_C - target_N
ANGLE: prev_C - target_N - target_CA
DIHE:  prev_O/prev_CA - prev_C - target_N - target_CA/H/CA-neighbor
```

上一残基的 `CA` atom type 按 `prom` 和 terminal 状态解析：

```text
ff14SB internal CA -> CX
ff19SB internal CA -> XC
N/C terminal CA    -> CX
```

### 12.2 Tail side

若 target `C` 与下一残基 `N` 成肽键：

```text
target C - next N
```

写入：

```text
BOND:  target_C - next_N
ANGLE: target_O - target_C - next_N
ANGLE: target_CA - target_C - next_N
ANGLE: target_C - next_N - next substituent
DIHE:  target_O/target_CA - target_C - next_N - substituent
DIHE:  target side - target_CA - target_C - next_N
```

普通下一残基 substituent 通常是 `H/HN/CA`；PRO 下一残基使用 `CA/CD`，因此会出现 `N-XC` 和 `N-CT` 边界项。

### 12.3 AutoNACC 思路

AutoNACC 的固定 cross-term 表本质是“补匹配”，不是重新拟合 peptide barrier。MAPLE 沿用这个思想，但按真实 PDB boundary 动态生成 exact terms，不全局铺一批 `CX/XC/CT` 组合。

## 13. ff14SB / ff19SB / CMAP

NCAA 支持 `prom=ff14SB` 和 `prom=ff19SB`。

重要边界：

- `RN_maple.prepin/frcmod` 不生成 CMAP section。
- target NCAA residue 的主链行为由 custom template + exact boundary terms + GAFF2/Maple residue terms 描述。
- 标准上下游残基仍由 protein library 处理；tleap 对标准 residue 的内部 backbone 参数照常来自 `leaprc.protein.<prom>`。
- ff19SB 的 CMAP 是 protein library 层面的 backbone correction；custom NCAA residue 不会自动获得一套新 CMAP，除非后续显式实现 CMAP template 支持。

因此：

- `prom=ff14SB` 是当前更保守的默认路线。
- `prom=ff19SB` 可用，但需要检查 tleap 输出与后续 MD 需求。
- abinitio summary 会提示 ff19SB NCAA 的 CMAP 注意事项。

## 14. Processed Protein PDB 与 tleap

NCAA 不让 tleap 直接读原始 PDB。它写：

```text
<base>_ncaa_tleap.pdb
```

处理规则：

- 完整复制输入 protein PDB。
- 只把 target residue 的 resname 改成 `config.rn`。
- atom names 保持与 prepin 对应。
- 保留链、编号、TER/END 语义。

tleap input：

```text
source leaprc.protein.<prom>
source leaprc.gaff2
source leaprc.water.<watm>
addAtomTypes ...
loadamberprep RN_maple.prepin
loadamberparams RN_maple.frcmod
loadamberparams ion frcmod
mol = loadpdb <base>_ncaa_tleap.pdb
check mol
charge mol
solvatebox ...
addions ...
saveamberparm ...
```

workflow 会自动运行 tleap 并写：

```text
<base>_ncaa_tleap.out
```

abinitio summary 会报告 `Errors / Warnings / Notes`。

## 15. Workflow Result

`NCAAWorkflowResult` 包含：

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

`result.files` 汇总正式文件路径，包括:

```text
target_capped_pdb
tleap_pdb
tleap_input
capped_mol2
gaff2_mol2
prepin
frcmod
refined_prepin
refined_frcmod
alpha/beta capped pdb
```

## 16. 输出与报告

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

过程文件：

```text
<base>_work/ncaa/
<base>_work/torsionfit/
```

`PARMFIT ABINITIO RESULT` 中显示：

- route。
- target。
- residue name。
- chirality。
- representative conformer。
- RESP conformers。
- refined prepin/frcmod。
- tleap PDB/input/status。
- stage timing。

## 17. 失败点

常见失败来源：

- target 不是 protein kind。
- target residue 缺 `N/CA/C`。
- target residue 无 sidechain anchor。
- representative 或 alpha/beta conformer 优化不收敛。
- 多构象 RESP atom ordering 不一致。
- Gaussian / espgen / resp / antechamber / prepgen / parmchk2 / tleap 失败。
- `N -> C` mainchain path 推断失败。
- prepin atom names 与 target residue atom names 不一致。
- final parameter set 缺 prepin atom 对应 MASS/NONBON 或 bonded term。
- peptide boundary exact terms 不覆盖实际 tleap 枚举的边界 pattern。
- 使用 ff19SB 时，custom NCAA residue 的 CMAP 语义不符合生产需求。

## 18. 方法学定位

NCAA route 的方法定位可以概括为：

```text
multiconformer RESP for charge
+ Hessian-derived BOND/ANGLE refinement
+ sidechain torsion scan fitting
+ exact peptide-boundary repair
+ automatic tleap validation
```

它不是全蛋白自动修复器。PDB 准备、其它 ligand/cofactor template、protonation state 和生产 MD protocol 仍由用户控制。
