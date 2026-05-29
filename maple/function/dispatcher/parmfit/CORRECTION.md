# Parmfit Correction 技术说明

本文档描述 `method=correction` 的当前实现。Correction 从已有 `mol2` 和 MAPLE calculator 出发，修正 bonded parameters，并导出 Maple Amber/GROMACS 参数。

## 1. Workflow

```text
input mol2 + atoms.calc
        |
        v
parmchk2 -> <base>_original.frcmod
        |
        v
initial CorrectionParameterSet
        |
        v
LBFGS geometry optimization
        |
        v
Cartesian Hessian -> modified Seminario bond/angle
        |
        v
stage0 parameter set
        |
        v
TorsionFit scan + spectral shared-group Stage1
        |
        v
Stage2 direct k/phase fast MM cycles
        |
        v
final parameter set
        |
        v
Amber mol2/frcmod + GROMACS top/gro
```

正式输出：

```text
<base>_work/<base>_original.frcmod
<base>_work/<base>_maple.mol2
<base>_work/<base>_maple.frcmod
<base>_work/<base>_maple.top
<base>_work/<base>_maple.gro
```

Torsion scan 文件写到：

```text
<base>_work/torsionfit/
```

## 2. 模块边界

| 文件 | 责任 |
|---|---|
| `correction/config.py` | 解析 `mol2`、LBFGS 和 TorsionFit 参数 |
| `correction/parameters.py` | 运行 `parmchk2` 并构建初始 `CorrectionParameterSet` |
| `correction/workflow.py` | 串联 correction 主流程 |
| `correction/artifacts.py` | 保存 workflow result，写 Amber/GROMACS 文件 |
| `correction/report.py` | 输出 setup、参数变化、energy trace 和 result block |
| `correction/correction.py` | dispatcher-facing thin facade |

TorsionFit 当前文件边界：

| 文件 | 责任 |
|---|---|
| `workflow.py` | center bond 选择、scan 复用/执行、scan 点过滤、调用 fitting |
| `topology.py` | rotatable center bond、proper torsion 拓扑、参数写回 |
| `basis.py` | local/global fitting problem、shared group basis、fast MM profile cache |
| `quality.py` | scan profile 权重、尺度、RMSE、geometry jump 判断 |
| `spectral.py` | FFT/DiFT-style candidate period 和 shared-group phase seed |
| `ensemble.py` | optional rigid rotor ensemble sampling and Stage2 extra targets |
| `stage1.py` | spectral shared-group restrained LLS |
| `stage2.py` | global direct k/phase loss refinement |
| `fit.py` | Stage1/Stage2/cycle 组合入口 |
| `records.py` | scan、report、cycle、workflow result 数据对象 |
| `report.py` | torsion fit 报告 |

## 3. 初始参数链路

`correction/parameters.py` 执行：

```text
mol2
 -> parmchk2 -s gaff2 -a Y
 -> <base>_original.frcmod
 -> build_correction_parameter_set(...)
```

`CorrectionParameterSet` 中每个实际 atom pair/tuple 都应有对应参数或明确的 unmatched 记录：

```text
Bond      -> atom pair
Angle     -> atom triple
Dihedral  -> atom quartet
Improper  -> atom quartet
Nonbond   -> atom
```

Amber 导出前会检查缺项。缺 MASS/NONBON/DIHE/IMPROPER 不会静默写出。

## 4. mSeminario Stage

几何优化后，workflow 读取 Cartesian Hessian：

```text
H_cartesian = d^2E / dx_i dx_j
```

`apply_mseminario(...)` 把 Hessian 投影到 internal coordinate，更新：

```text
Bond:  kBond, rEq
Angle: kTheta, thetaEq
```

不更新：

```text
DIHE.kPhi / phase / period
IMPROPER
NONBON
```

因此参数来源分层是：

```text
initial GAFF2/parmchk2
 -> mSeminario changes BOND/ANGLE
 -> TorsionFit changes selected DIHE
```

## 5. Classical MM 语义

Correction 的内部 profile 计算使用 `utils/mechanics.py`：

```text
E_MM = E_bond + E_angle + E_proper + E_improper + E_vdw + E_elec
```

Bond/angle/torsion：

```text
E_bond    = sum_b k_b (r_b - r0_b)^2
E_angle   = sum_a k_a (theta_a - theta0_a)^2
E_torsion = sum_t sum_n k_n [1 + cos(n phi_t - gamma_n)]
```

非键项：

```text
E_vdw  = epsilon_ij [(R_ij/r_ij)^12 - 2(R_ij/r_ij)^6]
E_elec = 332.05221729 q_i q_j / r_ij
```

AMBER exclusions：

```text
1-2 pair: excluded
1-3 pair: excluded
1-4 electrostatic: / 1.2
1-4 vdw:           / 2.0
```

`dihedral_radians(...)` 返回 AMBER/sander/cpptraj convention 下的 `phi`。`phase` 在内部是 radian，写 frcmod/top 时转成 degree。

## 6. Center Bond Selection

自动选择 center bond 时要求：

1. 它是 proper torsion 的中心键。
2. 它不在 ring 中。
3. mol2 bond type 表示单键。

可旋转单键识别：

```text
{"1", "1.0", "s", "single"}
```

如果输入 mol2 把 amide、芳香键或双键错误标成单键，Correction 会按 mol2 执行。此类问题应修正 mol2，而不是让 TorsionFit 猜拓扑。

显式 `torsion_bonds` 可以指定 center bond。非单键会报错；ring center bond 当前允许但会给 warning。

## 7. Scan 数据

每个 center bond 选择一个 representative dihedral，并生成 scan frames：

```text
theta_i      scan angle
X_i          optimized / constrained frame
Q_i          MLIP/QM reference energy
```

内部相对能量：

```text
QM_rel_i = Q_i - min(Q)
```

拟合前使用 original MM 过滤明显异常 frame：

```text
MM_orig_rel_i <= 50.0 kcal/mol
```

scan 数据可以包含 relaxed geometry 的耦合变化。`geometry_jump` 只作为 profile 诊断和人工解读信息；当前 Stage2 不把它作为参数接受/拒绝条件。

## 8. Shared Group

TorsionFit 不做 atom-id 级别 per-instance fitting。同一 center bond 下的 proper torsion instances 会按 shared group 共同拟合。

当前 shared key：

```text
canonical torsion atom types + one-hop non-torsion environment
```

对 scan frame `i`、group `g`、slot `s`：

```text
B_i,g,s = sum_m [1 + cos(n_s phi_i,m - gamma_g,s)]
```

其中 `m` 是 group 内 torsion path。这样同一化学环境共享一套参数，同时允许不同 one-hop environment 分裂。

## 9. Stage1: Spectral Shared-Group Restrained LLS

Stage1 的 target 是当前 center bond 的 torsion contribution：

```text
mm_base_rel    = MM relative profile with this center-bond proper torsion removed
fit_target_rel = QM_rel - mm_base_rel
```

Stage1 解 restrained least squares：

```text
min ||B k - fit_target_rel||^2 + lambda sum_j w_j (k_j - k0_j)^2
```

语义：

- existing term 进入 weak-prior refit。
- existing term 可以被 refit 到接近 0。
- old zero-amplitude term 不进 active fitting，但保留到最终参数，避免 Amber 缺项。
- spectral helper 从 `n = 1, 2, 3, 4, 6` 中选 candidate。
- 默认不主动新增 `n=5`；若原始参数已有 `n=5`，会作为 existing term 保留。
- 单个 center bond 默认最多新增 3 个 spectral slots。

Spectral candidate 的代表 profile 近似写作：

```text
fit_target_rel(phi) ~= a_n cos(n phi) + b_n sin(n phi)
```

并转成 AMBER form：

```text
k_n = sqrt(a_n^2 + b_n^2)
gamma_n = atan2(b_n, a_n)
```

Shared-group phase migration 使用：

```text
response_g,n = mean_frames(mean_paths exp(i n (phi_path - phi_rep)))
coherence_g,n = |response_g,n|
phase_seed_g,n = phase_fft,n + arg(response_g,n)
```

coherence 过低表示 group 内 path cancellation，candidate 不会激活。

## 10. Stage2: Global Direct K/Phase Refinement

Stage2 不新增 slot，不做 candidate gate；它直接优化 Stage1 已确定 slot 的 AMBER torsion 参数：

```text
x = [k_1, gamma_1, k_2, gamma_2, ...]
```

计算时仍使用 cached cos/sin basis，以避免逐 frame 重算完整 MM：

```text
a_j = k_j cos(gamma_j)
b_j = k_j sin(gamma_j)

MM_s(x) = constant_s + cos_basis_s @ a + sin_basis_s @ b
```

目标函数：

```text
L = L_scan + w_ensemble L_ensemble + L_prior
```

其中 scan 和 ensemble target 使用同一套相对 reference 约定；`torsion_ensemble_weight` 只作为 ensemble loss 的权重。prior 在 coefficient space 约束 Stage2 不要无意义偏离 Stage1：

```text
a0_j = k0_j cos(gamma0_j)
b0_j = k0_j sin(gamma0_j)

L_prior = lambda sum_j p_j [(a_j-a0_j)^2 + (b_j-b0_j)^2] / scale_j^2
```

Stage2 的接受规则是全局 loss 判断：

```text
if finite(final_total_loss) and final_total_loss < initial_total_loss - tol:
    accept optimized k/phase
else:
    keep Stage1
```

数值边界：

- `0 <= kPhi <= k_cap`，由 L-BFGS-B bounds 约束。
- `phase` 优化时不设硬边界，写回前 wrap 到 `(-pi, pi]`。
- nonfinite loss / vector 返回大 penalty；最终非有限则拒绝。
- 不再使用 checkpoint selection、per-center rollback、geometry jump/cancellation/torsion-unidentifiable hard guard。

`stage2_ref` 在报告中表示最终 accepted parameter set 对 scan frames 的 MM profile。如果 Stage2 没有降低 total loss，最终参数保持 Stage1。

## 11. Optional Torsion Ensemble Target

`torsion_ensemble=true` 时，Correction 可以给 Stage2 增加额外构象约束。它不改变 Stage1，不改变 scan 文件，也不在 `corr.out` 展开额外 profile 表。

当前 ensemble 采样链路：

```text
optimized starting structure
 -> 15 degree rigid random rotor trials
 -> CPU MM energy / clash / duplicate filtering
 -> selected frames only
 -> MLIP single point
 -> Stage2 extra target
```

触发边界：

- 至少需要两个 fitted center bonds；只有一个 center bond 时跳过 ensemble。
- correction 可以采全分子 eligible rotors；NCAA 可传入 R-group mobile mask，冻结 backbone/cap。
- trial 不做 `FixInternals` relaxed optimization，不占用 GPU 做结构优化。
- 最终只写一个 ensemble xyz：

```text
<base>_work/torsionfit/<base>_torsionfit_ensemble.xyz
```

ensemble target 与 scan 使用同一个 reference convention。对某个 center bond：

```text
qm_rel_ens      = MLIP_abs_ens - MLIP_abs_scan_ref
constant_rel    = constant_abs_ens - constant_abs_scan_ref
cos_basis_rel   = cos_basis_abs_ens - cos_basis_abs_scan_ref
sin_basis_rel   = sin_basis_abs_ens - sin_basis_abs_scan_ref
```

因此 ensemble frame 如果比 scan reference 更低能，`qm_rel_ens` 可以为负。这是允许的；它表示 scan 没覆盖到的更低能构象。

`torsion_ensemble_ratio` 控制目标 ensemble 规模：

```text
target_count ~= ceil(number_of_center_bonds * (torsion_steps + 1) * torsion_ensemble_ratio)
```

最终进入 Stage2 的 frame 数还会受 MM filter、MLIP high-energy filter 和每个 center 至少 2 帧的要求影响。`torsion_ensemble_weight` 只在 Stage2 loss 中缩放 ensemble target。

## 12. Fast MM Cycle

`torsion_refine_rounds` 当前语义：

```text
0: Stage1 only
1: Stage1 + Stage2 once
N: 最多 N 轮 Stage1 + Stage2 fast MM cycle
```

每轮：

```text
current parameter set
 -> cached MM profile refresh
 -> Stage1 refit
 -> Stage2 direct k/phase refinement
 -> inner Stage2 accepts only if total loss improves
 -> outer fast-cycle accepts only if this round improves the best scan score
 -> otherwise keep previous best and stop
```

不做：

- 不重新跑 MLIP。
- 不重新生成 scan。
- 不重新 constrained optimization。
- 不做 relaxed-rescan loop。

`basis.py::_MMProfileCache` 保存：

```text
full MM reference energy
phi(frame) for fitted center-bond proper torsions
base profile after removing all fitted-center torsions
```

因此任意 current parameter set 下：

```text
full_rel = base_rel + sum(all fitted-center torsion_rel)
center_zeroed_rel(A) = full_rel - torsion_rel(A)
```

这保证多 center-bond cycle 中，先前 center 的新参数会影响后续 center 的 base profile。

## 13. Amber / GROMACS Export

Amber 导出：

- 写 refined mol2。
- 写 refined frcmod。
- 对 Maple atom type 的 MASS/NONBON/BOND/ANGLE/DIHE/IMPROPER 做完整检查。
- 依赖 Amber/tleap 的 atom type pattern 匹配。

GROMACS 导出：

- 写 explicit atom、bond、angle、proper、improper、nonbond rows。
- 不依赖 tleap wildcard。
- 单位转换按 GROMACS 语义：
  - bond length: Angstrom -> nm
  - bond force: kcal/mol/A^2 -> kJ/mol/nm^2
  - angle/torsion energy: kcal/mol -> kJ/mol
  - phase: rad -> degree

## 14. Report

`corr.out` 当前采用：

```text
short stage progress
-> mSeminario bond/angle changes
-> TorsionFit dihedral changes
-> PARMFIT CORRECTION RESULT
```

结尾块包含：

- final Amber/GROMACS files。
- stage timing。
- mSeminario 修改数量。
- TorsionFit 修改数量。
- Torsion energy trace。
- `MLIP_ref vs stage2_final` MAE/RMSE。
- warnings。

Torsion energy trace 列：

```text
angle_deg
MLIP_ref
orig_ref
stage0_ref
stage1_ref
stage2_ref
```

其中：

- `MLIP_ref = TorsionFitReport.curves.qm_rel`
- `orig_ref = mm_orig_rel`，即原始 GAFF2/parmchk2 参数在同一 scan reference 下的 MM relative profile
- `stage0_ref = mm_stage0_rel`
- `stage1_ref = mm_stage1_rel`
- `stage2_ref = mm_stage2_rel`，即最终接受参数的 profile；如果 Stage2 被拒绝，它等价于最终保留的 Stage1/best profile

## 15. 结果解释

### 15.1 Stage1 好，外部 ensemble 差

这通常表示 scan training distribution 与外部 conformer ensemble 不一致。TorsionFit 优化 scan frames 上的相对能量，不保证自动改善外部 500-frame ensemble 排序。

### 15.2 高 RMSE 或 profile 分支变化

这不是 optimizer 没运行。常见原因：

- branch hop。
- steric clash release。
- nonbonded/improper/bond-angle 耦合主导误差。
- relaxed scan 进入另一条构象分支。

当前策略是保留这些数据进入评估；Stage2 不因为 geometry jump 自动拒绝参数，但最终仍要求 total loss 有数值改善。

### 15.3 Stage2 rejected

Stage2 rejected 表示直接 `k/phase` 优化没有降低 finite total loss。最终参数使用 Stage1 或上一轮 best accepted result，而不是 blindly 使用最后一轮 optimizer 输出。

## 16. Debug 顺序

建议排查顺序：

1. 检查 mol2 bond type。
2. 检查 selected center bonds。
3. 检查 scan xyz 是否对应当前输入和当前 torsion settings。
4. 查看 high-energy frame filter warning。
5. 比较 `MLIP_ref / orig_ref / stage0_ref / stage1_ref / stage2_ref`。
6. 查看 active slots、spectral candidate、coherence、k cap 和 Stage2 loss summary。
7. 用 `torsion_refine_rounds=0` 单独判断 Stage1。
8. 如果 scan 内改善但外部 ensemble 变差，优先检查 scan 覆盖和 torsion 可识别性。
