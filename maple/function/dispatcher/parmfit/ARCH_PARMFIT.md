# MAPLE Parmfit Dispatcher 架构

本文档描述 `parmfit` dispatcher 的当前源码架构。它是现状说明，不定义未来行为。

## 1. 定位

`parmfit` 是 MAPLE 的 parameter fitting dispatcher。它负责 route 识别、输入路径归一化和 workflow orchestration；具体化学逻辑放在领域包中。

当前正式 route：

```text
Parmfit
├── method=correction
│   └── correction/
└── method=abinitio
    ├── target_kind=ion     -> utils/MetalAA/
    └── target_kind=protein -> utils/NCAA/
```

`abinitio/abinitio.py` 是薄 route orchestrator。它读取 PDB、定位 target residue、调用 `classify_kind(...)`，然后把任务交给 MetalAA 或 NCAA；它不承载 metal chemistry、NCAA capping、RESP 或 torsion fitting 逻辑。

## 2. Route Contract

### 2.1 `method=correction`

输入：

- `mol2`
- 带 calculator 的 `Atoms`

数据链路：

```text
mol2 + atoms.calc
        |
        v
correction/config.py
        |
        v
correction/correction.py
        |
        v
correction/workflow.py
        |
        v
utils/TorsionFit/
        |
        v
Maple Amber/GROMACS export
```

Correction 负责：

- 自动 `parmchk2` 生成 `<base>_original.frcmod`。
- 几何优化。
- Hessian -> modified Seminario 修正 bond/angle。
- 调 shared TorsionFit 修正 selected proper torsion。
- 导出 Maple Amber/GROMACS 文件。

### 2.2 `method=abinitio`, `target_kind=ion`

输入：

- `pdb`
- `target`
- `cmo="charge multiplicity [oxidation]"`
- MetalAA-specific options

数据链路：

```text
PDB + target ion
        |
        v
abinitio/abinitio.py
        |
        v
utils/MetalAA/config.py
        |
        v
utils/MetalAA/workflow.py
        |
        v
large_model RESP/Hessian + site_model deployment
```

MetalAA 负责：

- metal core 和 donor residues 识别。
- `large_model` 构建、几何优化和 RESP。
- RESP charge 映射回 `site_model`。
- large-model Hessian / mSeminario。
- 部署 site 的 per-residue mol2、renamed PDB、`<base>_metal.frcmod` 和 `tleap.in`。

### 2.3 `method=abinitio`, `target_kind=protein`

输入：

- `pdb`
- `target`
- `cmo="charge multiplicity"`
- NCAA-specific options

数据链路：

```text
PDB + target residue
        |
        v
abinitio/abinitio.py
        |
        v
utils/NCAA/config.py
        |
        v
utils/NCAA/workflow.py
        |
        v
ACE-target-NME RESP + Amber template + TorsionFit
```

NCAA 负责：

- 构建 `ACE-target-NME` capped representative。
- 生成 alpha/beta RESP conformers。
- 多构象 RESP。
- AmberTools 生成 original `RN.prepin/RN.frcmod`。
- mSeminario + shared TorsionFit。
- 生成 refined `RN_maple.prepin/RN_maple.frcmod`。

## 3. Active Package Layout

```text
parmfit/
├── parmfit.py
├── abinitio/
│   ├── abinitio.py
│   └── report.py
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
│   ├── mSeminario.py
│   ├── readparm.py
│   ├── outputparm.py
│   ├── mechanics.py
│   ├── structure.py
│   ├── context.py
│   ├── model.py
│   └── capping.py
└── test/
```

正式职责边界：

- `utils/MetalAA/`：metal site chemistry and deployment。
- `utils/NCAA/`：capped residue template construction。
- `utils/TorsionFit/`：shared torsion scan data fitting and refinement。
- `utils/Scan/`：scan API、projected optimizer、CGWS/CGBS/projected LBFGS。
- `utils/runtime.py`：workdir、geometry optimization、Hessian、RESP runtime helpers。
- `utils/interface.py`：Gaussian/AmberTools command interface and frcmod helpers。
- `utils/resp.py`：RESP input/output and charge rules。

历史 snapshot、reference 目录和 `utils/?/` 不是 active workflow 层，除非当前源码显式 import。

## 4. Shared Scan Subsystem

`utils/Scan/` 是正式 scan 子系统。public API：

```python
run_silent_scan(...)
read_scan_final_atoms(...)
```

核心文件：

- `models.py`：`ScanConstraint`、`SilentScanOptions`、`SilentScanResult`。
- `api.py`：public entrypoint。
- `engine.py`：scan engine。
- `optimizer.py`：projected helpers、projected LBFGS、CG_WS、CG_BS。

Scan 路由：

| Constraint mode | Mode | Optimizer path |
|---|---|---|
| `fixinternals` | `rigid` | rigid geometry only |
| `fixinternals` | `relaxed` + `lbfgs` | global classic LBFGS |
| `fixinternals` | `relaxed` + `cgws/cgbs` | Scan CG optimizer |
| `projected` | `relaxed` | Scan projected optimizer |
| `projected` | `rigid` | invalid |

Scan 支持 1D/2D/3D，输出 `<base>_scan_final.xyz`。

## 5. TorsionFit Architecture

TorsionFit 是 `correction` 和 `NCAA` 的 shared subsystem。它的工作对象是一组 scan frames 和相对能量，不是必须连续的理想 torsion curve。

代码边界：

- `workflow.py` 只负责外层流程：center bond、scan、过滤、loss-mode 调用。
- `core.py` 负责拓扑选择和 fitted torsion 写回。
- `problem.py` 负责把 scan/profile/parameter set 构造成 local/global problem。
- `fit.py` 负责 Stage1、Stage2、guard 和 report 数据。
- `mode_loss.py` 负责把 Stage1 结果交给 Stage2，并返回最终 parameter set。
- `report.py` 负责文字输出。

主数据链：

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

当前语义：

- center bond 自动选择依赖 proper torsion、非 ring、mol2 single bond。
- Stage1 是 per-center fixed-phase restrained LLS。
- Stage1 只拟 `k`，可试探少量 canonical `k1..k4` missing slots。
- Stage2 是 global continuous-phase refinement，变量是 cos/sin coefficient delta。
- selected target terms 硬约束 `k <= 3.0`。
- `geometry_jump` 数据保留评估，但 Stage2 冻结对应 center。
- 最终按 center 接受 Stage2 block；未接受的 block 回到 Stage1。

详细数学和 report 语义见 [CORRECTION.md](./CORRECTION.md)。

## 6. Output Contract

### Correction

- `<base>_work/<base>_original.frcmod`
- `<base>_work/<base>_maple.top`
- `<base>_work/<base>_maple.gro`
- `<base>_work/<base>_maple.mol2`
- `<base>_work/<base>_maple.frcmod`

### MetalAA

- `<base>_work/<base>_metal_large_raw.pdb`
- `<base>_work/<base>_metal_large_opt.pdb`
- `<base>_work/<base>_metal_site_opt.pdb`
- `<base>_work/<base>_metal_site.mol2`
- `<base>_work/<base>_metal.frcmod`
- `<base>_work/<base>_metal_tleap.pdb`
- `<base>_work/<base>_metal_tleap.in`
- per-residue mol2 files

### NCAA

- `<base>_work/RN.prepin`
- `<base>_work/RN.frcmod`
- `<base>_work/RN_maple.prepin`
- `<base>_work/RN_maple.frcmod`

## 7. Configuration Defaults

Important current defaults:

```text
method                  = correction
cmo                     = "0 1"
torsion_steps           = 72
backend                 = cgbs
constraint_mode         = projected
torsion_refine_rounds   = 2
torsion_refine_max_iter = 10
torsion_refine_tol      = 1.0e-6
torsion_report_debug    = false
NCAA rn                 = MOL
NCAA QM                 = gaussian / HF / 6-31G(d)
MetalAA QM              = gaussian / PBE1PBE / def2SVP
MetalAA watm            = opc
MetalAA ionm            = 12_6
```

`qm_backend` / `resp_backend` belong to QM/RESP configuration. Torsion scan backend is the TorsionFit `backend`.

## 8. Design Boundaries

- `Parmfit` should remain a dispatcher, not a chemistry implementation.
- `abinitio` should remain a route selector.
- MetalAA and NCAA should keep independent chemistry semantics; they both use RESP, but their model definitions differ.
- Scan is shared infrastructure; consumers should call `run_silent_scan(...)`.
- TorsionFit modifies selected proper torsions only; it does not repair topology, nonbonded terms, impropers, or bad mol2 bond types.
- Whole-protein AMBER assembly is not part of current MetalAA route; MetalAA emits deployment files and `tleap` input.
