# Parmfit Correction 技术说明

本文档描述 `method=correction` 的当前实现。Correction 的职责是从已有 `mol2` 和当前 MAPLE calculator 出发，修正 bonded parameters，并导出 Maple Amber/GROMACS 参数。

Correction 的 torsion correction 只有一条正式路径：基于 constrained scan 数据的 shared TorsionFit。没有独立的 torsion fitting mode 选择，也不在 workflow 中写额外 JSON 曲线文件。

## 1. Workflow

```text
input mol2 + atoms.calc
        |
        v
parmchk2 -> <base>_original.frcmod
        |
        v
original CorrectionParameterSet
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
rotatable center bond selection
        |
        v
Scan API constrained torsion scans
        |
        v
MM_orig high-energy frame filter
        |
        v
Stage1 per-center fixed-phase restrained LLS
        |
        v
Stage2 global continuous-phase refinement
        |
        v
final parameter set + report + exports
```

正式输出：

- `<base>_work/<base>_original.frcmod`
- `<base>_work/<base>_maple.top`
- `<base>_work/<base>_maple.gro`
- `<base>_work/<base>_maple.mol2`
- `<base>_work/<base>_maple.frcmod`

Torsion scan 过程文件写到：

```text
<base>_work/torsionfit/
```

TorsionFit 模块的当前责任划分：

- `workflow.py`：center bond 选择、scan 复用/执行、scan 点过滤。
- `core.py`：proper torsion 拓扑、center bond 判断和参数写回。
- `problem.py`：构建 local/global fitting problem。
- `profiles.py`：scan profile 权重、质量判断、RMSE 和尺度归一化。
- `stage1.py`：Stage1 local fixed-phase restrained LLS、active set、新 slot 试探和 local report 数据。
- `stage2.py`：Stage2 global continuous-phase refinement、checkpoint、guard、rollback 和 cos/sin delta。
- `fit.py`：public fitting entrypoints，连接 local problem、Stage1 和 report 组装。
- `mode_loss.py`：把 Stage1 与 Stage2 连接成 correction/NCAA 共用路线。
- `report.py`：格式化报告文字。

## 2. Center Bond Selection

自动选择 center bond 时必须同时满足：

1. 该键是 proper torsion 的中心键。
2. 该键不在 ring 中。
3. 该键在 mol2 bond table 中标记为单键。

当前可旋转性只信任 mol2 `bond_type`：

```text
rotatable bond_type in {"1", "1.0", "s", "single"}
```

如果输入 mol2 把双键、芳香键或 amide-like 键错误标成单键，Correction 会把它当作可旋转键。修正方式是修正 mol2，而不是让 TorsionFit 用额外规则猜拓扑。

显式 `torsion_bonds` 可以指定 center bond。非 mol2 单键会报错；ring center bond 按当前策略允许但给出 warning。

## 3. Scan 数据语义

scan 是构象和相对能量数据来源，不是对“理想单自由度 torsion potential”的承诺。对每个 selected center bond，workflow 选择一个 representative dihedral，生成一组 frames：

```text
theta_i      scan angle in degrees
X_i          generated structure
Q_i          reference energy
```

内部使用相对能量：

```text
QM_rel_i = Q_i - min(Q)
```

这些点可以包含 relaxed geometry 的耦合变化。构象突跳、位阻释放或其它 branch change 不会让数据自动失效；它们只影响后续是否允许新增自由度或 Stage2 更新。

Stage1 前会用 original MM 参数过滤明显异常的高能 frames：

```text
MM_orig_rel_i <= 50.0 kcal/mol
```

这个 filter 用来避免明显不物理的 MM frame 主导拟合，不表示高 QM 能量点必然无效。

## 4. Shared Torsion Group

TorsionFit 不做 per-instance torsion fit。一个 center bond 下的 proper torsion instances 会按 shared group 拟合，同一 group-slot 的参数写回 group 内所有 instances。

shared key 当前由两部分组成：

```text
canonical torsion atom types + one-hop non-torsion environment signature
```

这比单纯 GAFF atom type 更细，但仍避免退化成 atom-id 级别拟合。它的目标是放松错误共享，而不是完全放弃参数经济性。

对 scan frame `i`、shared group `g`、slot `s`，Stage1 basis 是 group 内 instances 的贡献和：

```text
B_i,g,s = sum_m [1 + cos(n_s * phi_i,m - gamma_s)]
```

其中 `m` 是 group 内 torsion instance，`n_s` 是 period，`gamma_s` 是 Stage1 固定 phase。

## 5. Stage1: Fixed-Phase Restrained LLS

Stage1 对每个 center bond 独立构建 local linear problem。核心目标是让 selected center bond 的 torsion contribution 解释：

```text
target = QM_rel - MM_zeroed_rel
```

其中 `MM_zeroed_rel` 是把当前 center bond 的待拟合 proper torsion contribution 置零后的 MM relative profile。

Stage1 默认 fixed phase，只优化 `k`：

```text
min_k || W(Ak - target) ||^2 + || R(k - k0) ||^2
```

含义：

- `A` 是 active group-slot basis。
- `W` 是 profile weights，低能点权重大，高能点仍保留。
- `R(k-k0)` 是 restrained prior，防止欠定方向任意漂移。
- `k0` 来自当前 parameter set。

Stage1 active-set 语义：

1. existing slots 默认参与 restrained LLS。
2. 缺失 slot 只从 canonical `k1..k4` 试探。
3. candidate 必须让 restrained LLS residual 有实际改善。
4. candidate 必须满足 k cap、贡献幅度、rank/cancellation 等检查。
5. 每个 center 只允许有限数量的 new slots。
6. `geometry_jump` center 不扩新 slot，但已有 slot 仍做 restrained LLS。

Stage1 不负责“回退到原始 torsion”。它是初拟合器，负责稳定地产生 restrained LLS 参数。最终是否接受 Stage2 的进一步更新由 Stage2 per-center selector 处理。

负系数会折叠成：

```text
k = abs(k)
phase = phase + pi
```

selected target terms 最终硬约束：

```text
k <= 3.0
```

## 6. Geometry Jump Diagnostic

`geometry_jump` 是 profile-level 风险标记，表示相邻 scan 点的 reference relative energy 出现异常突跳。它不等于“数据无效”，也不要求把该 scan 丢弃。

当前语义：

- scan 数据保留。
- 误差评估保留。
- Stage1 不为该 center 新增 slot。
- Stage1 已有 slot 仍做 restrained LLS。
- Stage2 冻结该 center 的 active variables，保留 Stage1 block。

这样处理的原因是：proper torsion Fourier terms 可以拟合一组构象上的相对能量趋势，但不应该用额外高阶项去硬解释 relaxed branch hop 或其它非 torsion-only 的结构事件。

## 7. Stage2: Global Continuous-Phase Refinement

Stage2 从 Stage1 parameter set 出发，构建跨所有 scan 的 global problem。它不新增 inactive slot，只优化 Stage1 active slots 的 cos/sin coefficient delta：

```text
MM_s(x) = constant_s
        + cos_basis_s @ (orig_cos + delta_cos)
        + sin_basis_s @ (orig_sin + delta_sin)
```

每个 scan 的 loss 是 weighted relative-energy error，global objective 是所有 scan 的平均 data loss 加 coefficient prior：

```text
total_loss = mean_s(data_loss_s) + prior_loss
```

Stage2 的作用不是替代 Stage1，而是在 Stage1 已确定 active set 后进行连续 phase/amplitude 微调。

### 7.1 Stage2 Guard

Stage2 使用 checkpoint + guard + per-center selector。

硬条件：

- objective 和参数必须 finite。
- selected target term 必须满足 `k <= 3.0`。
- cancellation ratio 不能过大。
- 被判定为 `geometry_jump` 的 center 不允许 Stage2 更新。
- 标记为 unidentifiable 的 torsion 不允许被无意义扰动。

诊断条件：

- `k_efficiency` 是 diagnostic。
- global total loss 中途上升不直接失败。
- per-scan data loss 的变化会进入报告，用于 per-center 接受判断。

最终选择：

- optimizer 可以记录多个合法 checkpoint。
- 全局先选 hard-safe checkpoint。
- 然后按 center 比较 Stage1 block 与 Stage2 block。
- Stage2 没有改善或违反 center guard 的 block 回到 Stage1。

## 8. Scan API Usage

Correction 不再直接依赖旧 `runtime.SilentScan`。scan 入口是：

```python
run_silent_scan(...)
```

来自：

```text
utils/Scan/
```

默认 TorsionFit scan 参数：

```text
backend         = cgbs
constraint_mode = projected
torsion_steps   = 72
```

路由要点：

- `fixinternals + rigid`：只做 rigid geometry。
- `fixinternals + relaxed + lbfgs`：走 global classic LBFGS。
- `projected + relaxed`：走 `utils/Scan/optimizer.py`。
- `projected + rigid`：抛 `ValueError`。
- scan 支持 1D/2D/3D。

## 9. Report 输出

Correction 主报告包含：

```text
Parmfit Correction Summary
Parmfit Torsion Scan Fit
Parmfit Torsion Stage-2 Final Fit
Parmfit Correction Parameter Report
```

curve table 使用五条曲线：

```text
angle_deg
QM_ref
MM_orig
MM_stage0
MM_stage1
MM_stage2
```

默认报告保持简洁。`torsion_report_debug=true` 时才输出 candidate trial、gate reason、rank、cap 等 debug 细节。

## 10. 结果解释

### 10.1 Stage1 有改善但外部 ensemble 变差

这通常说明 scan training distribution 与外部 conformer ensemble 不一致。TorsionFit 在 scan 构象上拟合相对能量，但没有外部 500-frame 数据时，不能保证 broad ensemble 排序一定改善。

chonf25/chonf26 类问题属于这个范畴：scan-union 上可以改善，但外部 ensemble 的目标 torsion correction 解释力弱。

### 10.2 高 RMSE 且 `geometry_jump`

这是保守失败，不是 optimizer 没运行。常见来源：

- branch hop
- steric clash release
- nonbonded/improper/bond-angle 耦合主导误差
- relaxed scan 进入另一条构象分支

当前策略是不让 proper torsion 用额外 new slots 或 Stage2 自由 phase 去硬拟合这种突跳。

### 10.3 Stage2 被拒绝

Stage2 被拒绝通常表示该 center 的 Stage2 block 没有比 Stage1 更好，或违反了硬条件。最终参数会回到 Stage1 block，而不是原始 torsion。

## 11. Debug 顺序

建议按以下顺序排查：

1. 检查 mol2 bond type。
2. 检查 selected center bonds。
3. 检查 scan xyz 是否对应当前输入和当前 scan 设置。
4. 查看 high-energy frame filter warning。
5. 比较 `QM_ref`、`MM_orig`、`MM_stage0`、`MM_stage1`、`MM_stage2`。
6. 查看 `geometry_jump`、active slots、rank、cancellation、capped terms。
7. 如 Stage2 异常，设置 `torsion_refine_rounds=0` 判断 Stage1 是否合理。
8. 如 scan fit 好但外部 ensemble 差，优先考虑 scan 覆盖和 torsion 可识别性，而不是放宽 `k` 上限。
