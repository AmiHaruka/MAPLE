# Parmfit Dispatcher 技术报告

`parmfit` 是 MAPLE 中负责参数构建与参数修正的 dispatcher。它不直接承载所有化学逻辑，而是根据输入 route 把任务交给 `correction`、`MetalAA` 或 `NCAA` 子系统。

这份 README 描述当前源码的用户可见契约、数据链路和拟合语义。更细的专项说明见：

- [ARCH_PARMFIT.md](./ARCH_PARMFIT.md)
- [CORRECTION.md](./CORRECTION.md)
- [ARCH_METAL_ABINITIO.md](./ARCH_METAL_ABINITIO.md)
- [ARCH_NCAA_ABINITIO.md](./ARCH_NCAA_ABINITIO.md)

## 1. Overview

当前 `parmfit` 有三条正式 workflow：

| Route | 入口条件 | 领域实现 | 主要产物 |
|---|---|---|---|
| `method=correction` | 输入 `mol2` | `correction/` + `utils/TorsionFit/` | Maple Amber/GROMACS refined parameters |
| `method=abinitio`, `target_kind=ion` | PDB target 被识别为 ion | `utils/MetalAA/` | metal site mol2/frcmod/tleap input |
| `method=abinitio`, `target_kind=protein` | PDB target 被识别为 protein residue | `utils/NCAA/` | NCAA original/refined Amber residue template |

dispatcher 结构保持很薄：

```text
Parmfit
├── correction
└── abinitio
    ├── ion     -> MetalAA
    └── protein -> NCAA
```

`abinitio` 当前只支持 `ion` 和 `protein` target。其它 target kind 会提前失败，而不是降级到 `correction`。

## 2. Dispatcher Contract

`Parmfit` 接收 `output`、带 calculator 的 ASE `Atoms` 和 `params`。`method` 识别顺序是：

```text
constructor method > params["method"] > "correction"
```

输入文件约束：

- `method=correction` 必须提供 `mol2`。
- `method=abinitio` 必须提供 `pdb` 和 `target`。
- `mol2` / `pdb` 会被规范化为绝对路径，并要求文件存在。
- `abinitio` route 的真实结构来自 `pdb`，不是输入文件中的 xyz block。

`cmo` 只在 ab initio 路线解析：

```text
cmo = "charge multiplicity [oxidation]"
```

- `charge` / `multiplicity` 写入 bootstrap `Atoms.info`。
- 第三项 `oxidation` 只被 MetalAA 使用。
- 未提供 `cmo` 时默认 `"0 1"`。

`route` 参数是 Gaussian route 字符串，不是 dispatcher route。

## 3. Route Dataflow

### 3.1 Correction

Correction 从已有 `mol2` 和当前 `atoms.calc` 出发，修正 bonded 参数并导出最终参数。

```text
mol2 + atoms.calc
        |
        v
parmchk2 -> <base>_original.frcmod
        |
        v
CorrectionParameterSet
        |
        v
geometry optimization
        |
        v
Cartesian Hessian -> modified Seminario bond/angle
        |
        v
TorsionFit scan + Stage1 restrained LLS + Stage2 refinement
        |
        v
Maple GROMACS + Maple Amber export
```

Correction 不接受用户手工 `frcmod` 作为正式输入；当前会自动用 `parmchk2` 生成 `<base>_original.frcmod`。

### 3.2 MetalAA

MetalAA 处理 `target_kind=ion` 的金属位点参数化。

```text
PDB + target ion
        |
        v
metal core detection
        |
        v
large_model = core + cutoff environment + caps/bridges
        |
        v
large_model geometry optimization + RESP
        |
        v
RESP charge projection onto site_model
        |
        v
site deployment files + large-model Hessian/mSeminario
        |
        v
<base>_metal.frcmod + per-residue mol2 + tleap input
```

`large_model` 是优化、RESP、Hessian 和 mSeminario 的对象。`site_model` 是最终部署对象，用于 RESP charge 回填、per-residue mol2、renamed PDB、final frcmod 和 `tleap.in`。

### 3.3 NCAA

NCAA 处理 `target_kind=protein` 的单残基 residue template 构建。

```text
PDB + target residue
        |
        v
ACE-target-NME capped representative
        |
        v
representative optimization
        |
        v
alpha / beta RESP conformers
        |
        v
multiconformer Gaussian ESP + RESP
        |
        v
antechamber / prepgen / parmchk2 -> RN.prepin / RN.frcmod
        |
        v
mSeminario + shared TorsionFit
        |
        v
RN_maple.prepin / RN_maple.frcmod
        |
        v
simple NCAA tleap input
```

`rn` 是输出 residue name，当前默认 `MOL`。NCAA 的后半段与 AutoNACC 相同，都是 RESP 后经 AmberTools 生成 residue template；MAPLE 不处理 AutoNACC 的前期 PDB 准备，用户需要先提供可用的 protein PDB。

## 4. Current TorsionFit Semantics

TorsionFit 当前被 `correction` 和 `NCAA` 复用。它的输入不是抽象的平滑 torsion potential，而是一组通过 scan 得到的构象 `X_i` 和参考相对能量 `QM_rel(X_i)`。

scan 的意义是获取结构和相对能量数据。构象可以出现分支跳变、位阻释放或其它 relaxed geometry 变化；这类数据仍会保留在报告和误差评估中，但不应该被额外 high-order Fourier term 强行解释。

当前代码边界按物理链路组织：

| 文件 | 责任 |
|---|---|
| `utils/TorsionFit/workflow.py` | 串联 center bond 选择、scan 复用/执行、scan 点过滤和 loss-mode 拟合 |
| `utils/TorsionFit/core.py` | 处理 torsion 拓扑选择、center bond 识别、ring 判断和 fitted torsion 写回 |
| `utils/TorsionFit/problem.py` | 把 scan data 和 parameter set 转成 local/global fitting problem |
| `utils/TorsionFit/profiles.py` | 处理 scan 权重、profile 质量、RMSE 和尺度归一化 |
| `utils/TorsionFit/stage1.py` | 执行 Stage1 local fixed-phase restrained LLS、新 slot 试探和 local report 数据组装 |
| `utils/TorsionFit/stage2.py` | 执行 Stage2 global continuous-phase refinement、checkpoint、guard、rollback 和 cos/sin delta |
| `utils/TorsionFit/fit.py` | 提供 fitting entrypoints，连接 local problem、Stage1 和 report 组装 |
| `utils/TorsionFit/mode_loss.py` | 把 Stage1 与 Stage2 组合成 correction/NCAA 共用的 loss-mode 路线 |
| `utils/TorsionFit/report.py` | 输出用户可读的 torsion fit 报告 |

核心数据链路保持为：

```text
run_torsion_workflow
-> scan data
-> run_loss_mode
-> fit_torsion_scan
-> build_local_torsion_problem
-> Stage1 local fit
-> apply_fitted_torsion
-> build_global_torsion_problem
-> refine_torsion_scans_global
-> apply_global_delta
-> TorsionWorkflowResult.final_parameter_set
```

### 4.1 Scan 数据

每个 selected center bond 会选择一个 representative dihedral 生成 scan。scan xyz 中记录：

```text
theta_i      scan angle
X_i          optimized or rigid frame
QM_i         reference energy
```

TorsionFit 使用相对能量：

```text
QM_rel_i = QM_i - min(QM)
```

拟合前会用 original MM relative energy 过滤明显异常的高能点，默认阈值是 `50 kcal/mol`。

### 4.2 Stage1: Fixed-Phase Restrained LLS

Stage1 是 per-center 的 restrained linear least squares。默认 fixed phase，只拟 `k`：

```text
target = QM_rel - MM_zeroed_rel
min_k || W(Ak - target) ||^2 + || R(k - k0) ||^2
```

语义：

- existing torsion slot 默认参与拟合。
- phase 在 Stage1 不自由优化。
- 缺失的 canonical `k1..k4` slot 会逐个试探。
- candidate 只有在 restrained LLS residual 有实际改善且参数合法时才加入 active set。
- 单个 center 当前最多新增有限数量的 slot。
- selected target terms 最终硬约束 `k <= 3.0`。

Stage1 不负责判断“是否回到原始 torsion”。它负责给出 restrained LLS 初拟合结果。

### 4.3 Stage2: Global Continuous-Phase Refinement

Stage2 从 Stage1 参数出发，不新增 slot，只优化 active slot 的 cos/sin coefficient delta：

```text
MM_s(x) = constant_s
        + cos_basis_s @ (orig_cos + delta_cos)
        + sin_basis_s @ (orig_sin + delta_sin)

total_loss = mean_s(weighted_relative_energy_loss_s) + prior_loss
```

语义：

- `torsion_refine_rounds <= 0` 时跳过 Stage2。
- optimization 使用 checkpoint。
- checkpoint 必须满足 finite、`k <= 3.0`、cancellation guard 等硬条件。
- 最终按 center 判断 Stage2 block 是否接受。
- 被拒绝的 center 回到 Stage1 block。
- `k_efficiency` 是诊断指标，不承担 hard cap 语义。
- `geometry_jump` center 会冻结 Stage2 更新，保留 Stage1 结果。

### 4.4 Generalization Note

Top20 中 chonf25/chonf26 这类体系的核心风险是：scan training distribution 上的 local fit 可以变好，但对外部 500-frame ensemble 的能量排序未必泛化。这属于训练构象代表性和 torsion 可识别性问题，不是 `k` cap 未生效，也不是 optimizer 没有运行。

## 5. Scan API

scan 能力集中在 `utils/Scan/`，正式入口是：

```python
from maple.function.dispatcher.parmfit.utils.Scan import run_silent_scan, read_scan_final_atoms
```

public models：

- `ScanConstraint`
- `SilentScanOptions`
- `SilentScanResult`

路由语义：

| `constraint_mode` | `mode` | `backend` | 行为 |
|---|---|---|---|
| `fixinternals` | `rigid` | any | 只做 rigid geometry，不跑 optimizer |
| `fixinternals` | `relaxed` | `lbfgs` | ASE `FixInternals` + global classic LBFGS |
| `fixinternals` | `relaxed` | `cgws/cgbs` | ASE `FixInternals` + Scan CG optimizer |
| `projected` | `relaxed` | `lbfgs/cgws/cgbs` | Scan projected optimizer |
| `projected` | `rigid` | any | 抛 `ValueError` |

约束输入使用 1-based atom index。scan 支持 1D/2D/3D，最终 xyz 写为：

```text
<base>_scan_final.xyz
```

`projected` 是 force projection 近似，不是严格 holonomic constraint solver。

## 6. Input Reference

### 6.1 Common Keys

| Key | Route | Required | Default | Meaning |
|---|---|---:|---|---|
| `method` | all | no | `correction` | dispatcher route |
| `mol2` | correction | yes |  | initial Amber/GAFF2 mol2 |
| `pdb` | abinitio | yes |  | input PDB |
| `target` | abinitio | yes |  | residue selector, e.g. `A301` |
| `cmo` | abinitio | no | `"0 1"` | charge / multiplicity / optional oxidation |

### 6.2 Correction / Shared Torsion Keys

| Key | Required | Default | Meaning |
|---|---:|---|---|
| `vib_scale` | no | `1.0` | mSeminario scaling factor |
| `torsionfit` | no | `true` | enable shared torsion refinement |
| `torsion_bonds` | no | auto | center bonds, e.g. `8-15,3-7` |
| `torsion_steps` | no | `72` | scan increments around 360 degrees |
| `backend` | no | `cgbs` | scan optimizer backend: `lbfgs/cgws/cgbs` |
| `constraint_mode` | no | `projected` | `fixinternals` or `projected` |
| `torsion_refine_rounds` | no | `2` | Stage2 refinement rounds |
| `torsion_refine_max_iter` | no | `10` | Stage2 optimizer max iterations |
| `torsion_refine_tol` | no | `1.0e-6` | Stage2 tolerance |
| `torsion_report_debug` | no | `false` | emit candidate/debug diagnostics |

### 6.3 Ab Initio Shared Keys

| Key | Required | Default | Meaning |
|---|---:|---|---|
| `theory` | no | route-specific | Gaussian theory |
| `basis` | no | route-specific | Gaussian basis |
| `route` | no | `""` | extra Gaussian route string |
| `nproc` | no | `8` | Gaussian CPU count |
| `mem` | no | `16` | Gaussian memory in GB |
| `qm_backend` | no | `gaussian` | QM backend selector |
| `resp_backend` | no | `gaussian` | legacy RESP/QM backend selector |

`qm_backend/resp_backend` belong to QM/RESP. Torsion scan backend uses `backend` from TorsionFit params.

### 6.4 MetalAA Keys

| Key | Required | Default | Meaning |
|---|---:|---|---|
| `add_resid` | no | `""` | manually add residues to metal core |
| `cfmol2` | no | `""` | cofactor mol2 templates |
| `cluster_cutoff` | no | `3.0` | large model environment radius in Angstrom |
| `donor_cutoff` | no | `3.0` | donor detection cutoff in Angstrom |
| `chgmod` | no | `1` | RESP charge constraint mode, `0/1/2/3` |
| `fixchg_resids` | no | `""` | residues with fixed reference charges |
| `watm` | no | `opc` | water model for ReadRadii / tleap reference |
| `ionm` | no | `12_6` | ion parameter set selector |
| `basis` | no | `def2SVP` | MetalAA Gaussian basis |

### 6.5 NCAA Keys

| Key | Required | Default | Meaning |
|---|---:|---|---|
| `rn` | no | `MOL` | output Amber residue name |
| `vib_scale` | no | `1.0` | NCAA mSeminario scaling factor |
| `basis` | no | `6-31G(d)` | NCAA Gaussian basis |
| `theory` | no | `HF` | NCAA Gaussian theory |
| shared torsion keys | no | see above | sidechain torsion refinement |

## 7. Output Contract

All formal outputs are written under:

```text
<base>_work/
```

### 7.1 Correction

Formal outputs:

- `<base>_work/<base>_original.frcmod`
- `<base>_work/<base>_maple.top`
- `<base>_work/<base>_maple.gro`
- `<base>_work/<base>_maple.mol2`
- `<base>_work/<base>_maple.frcmod`

Process files:

- `<base>_work/torsionfit/`

### 7.2 MetalAA

Formal outputs:

- `<base>_work/<base>_metal_large_raw.pdb`
- `<base>_work/<base>_metal_large_opt.pdb`
- `<base>_work/<base>_metal_site_opt.pdb`
- `<base>_work/<base>_metal_site.mol2`
- `<base>_work/<base>_metal.frcmod`
- `<base>_work/<base>_metal_tleap.pdb`
- `<base>_work/<base>_metal_tleap.in`
- per-residue mol2 files, e.g. `FE1.mol2`, `HD1.mol2`

Process files:

- `<base>_work/metalaa/`

### 7.3 NCAA

Formal outputs:

- `<base>_work/RN.prepin`
- `<base>_work/RN.frcmod`
- `<base>_work/RN_maple.prepin`
- `<base>_work/RN_maple.frcmod`
- `<base>_work/<base>_ncaa_tleap.in`

Process files:

- `<base>_work/ncaa/`
- `<base>_work/torsionfit/`

## 8. Project Structure

```text
parmfit/
├── parmfit.py
├── abinitio/
│   └── abinitio.py
├── correction/
│   ├── correction.py
│   ├── config.py
│   ├── workflow.py
│   └── report.py
├── utils/
│   ├── MetalAA/
│   ├── NCAA/
│   ├── Scan/
│   ├── TorsionFit/
│   ├── runtime.py
│   ├── interface.py
│   ├── resp.py
│   ├── structure.py
│   ├── context.py
│   ├── model.py
│   ├── readparm.py
│   ├── mechanics.py
│   ├── mSeminario.py
│   ├── outputparm.py
│   └── capping.py
├── test/
├── ARCH_PARMFIT.md
├── CORRECTION.md
├── ARCH_METAL_ABINITIO.md
└── ARCH_NCAA_ABINITIO.md
```

Reference snapshots and historical directories are not part of active runtime unless explicitly imported by current code.

## 9. Known Limits

- ab initio currently supports `ion` and `protein` target kinds only.
- MetalAA supports one metal center per workflow.
- MetalAA generates site-level parameter files and `tleap` input, but does not run whole-protein `tleap` assembly.
- NCAA assumes the target can be represented as `ACE-target-NME`.
- TorsionFit only modifies selected proper torsion terms; it does not repair wrong topology, nonbonded parameters, impropers, or bad input mol2 bond types.
- A scan fit that improves local training profiles may still fail on broad conformer ensembles if the scan set is not representative.
