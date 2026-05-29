# MAPLE Parmfit MetalAA Ab Initio 架构

顶层架构见 [ARCH_PARMFIT.md](./ARCH_PARMFIT.md)。本文档描述 `parmfit(method=abinitio)` 中 target kind 为 `ion` 时的 MetalAA route。

## 1. 定位

MetalAA 的目标是为金属位点生成 AMBER 可加载的局部参数。它参考 MCPB 的方法思想，但不调用 MCPB。

```text
PDB + target ion
 -> metal site recognition
 -> large_model
 -> RESP
 -> site_model deployment
 -> Hessian + mSeminario
 -> final metal frcmod
 -> tleap validation
```

MetalAA 当前只支持单金属中心。donor cutoff 内发现另一个 ion 会失败。

## 2. 模块边界

| 文件 | 责任 |
|---|---|
| `config.py` | 解析 MetalAA 用户输入 |
| `recognize.py` | metal core、donor、cofactor mol2、orig frcmod 准备 |
| `models.py` | `large_model` / `site_model` 构建 |
| `charges.py` | large charge 推断和 RESP charge 投影 |
| `parameters.py` | Amber/GAFF/ion 参数读取与匹配 |
| `artifacts.py` | atom typing、frcmod merge/remap、mol2/PDB/tleap 文件 |
| `workflow.py` | 串联 MetalAA 主流程 |
| `report.py` | route progress 文案 |

共享层：

- `runtime.py`: geometry optimization、Hessian、RESP runtime。
- `resp.py`: RESP 输入/输出和固定电荷规则。
- `interface.py`: Gaussian、AmberTools、tleap。
- `mSeminario.py`: Hessian 到 bond/angle 参数。
- `structure.py` / `context.py` / `model.py`: PDB、residue、model 和文件写出。

## 3. 输入语义

### 3.1 `cmo`

MetalAA 使用：

```text
cmo = "<charge> <multiplicity> [oxidation]"
```

含义：

```text
config.charge = metal + nonprotein ligand/cofactor fragments
config.mult   = QM multiplicity
config.oxy    = optional metal oxidation identity
```

`oxy` 不参与 total charge。它只用于 metal formal charge / ion identity：

```text
metal_formal_charge = config.oxy if config.oxy is not None else config.charge
```

large model charge：

```text
Q_large = config.charge + sum(formal_charge(protein residue in large_model))
M_large = config.mult
```

标准蛋白残基形式电荷来自内部表，例如：

```text
ASP -1
GLU -1
LYS +1
ARG +1
HIP +1
CYM -1
```

### 3.2 `cfmol2`

`cfmol2` 用于 HEM 等非蛋白 cofactor：

- 读取 mol2 atom names、atom types、charges、bonds。
- 用 atom names 匹配 PDB 中 ligand/cofactor residue。
- 把 mol2 atom type/charge 注入 PDB residue atom dict。
- 把 mol2 bonds 映射为 explicit PDB serial pairs。
- 对每个 cfmol2 调 `parmchk2 -s gaff2 -a Y` 生成 cofactor orig frcmod。

cofactor orig frcmod 是中间参数源，不在 final `tleap.in` 里单独加载。

### 3.3 `set_bonded`

`set_bonded` 显式指定 metal-donor serial pair：

```text
set_bonded = "FE_SERIAL-DONOR_SERIAL ..."
```

如果设置了它，MetalAA 使用这些 pair 作为直接 donor 关系；否则按 donor cutoff 自动识别。

### 3.4 RESP 固定电荷

`chgmod` 控制 RESP 中标准 backbone atom 的固定策略：

| `chgmod` | 固定 atom names |
|---:|---|
| 0 | 不按 backbone policy 自动固定 |
| 1 | `N/CA/C/O/OXT` |
| 2 | `N/H/HA/CA/C/O/OXT` |
| 3 | `N/H/HA/CA/CB/C/O/OXT` |

`fixchg_resids` 可指定完整 residue 固定到标准 Amber 电荷。指定非标准 residue 时，如果无法查到参考电荷会失败。

## 4. 模型定义

### 4.1 `MetalSiteSelection`

`find_metal_site_core(...)` 产生：

- target metal residue。
- direct donor atoms。
- 自动 core residues。
- 用户 `add_resid` core residues。
- selection warnings。

自动 donor 元素：

```text
N O S P SE F CL BR I
```

如果 donor 是 ligand/cofactor 且没有 atom type，workflow 会要求提供对应 `cfmol2`。

### 4.2 `large_model`

`large_model` 是优化、RESP、Hessian 和 mSeminario 的对象：

```text
large_model = metal core + cutoff environment + bridge/caps
```

包含：

- target metal。
- donor residues。
- user-added residues。
- cutoff environment。
- peptide bridge/cap。

不作为最终 AMBER 部署对象。

### 4.3 `site_model`

`site_model` 是最终部署对象：

```text
site_model = target metal + final core residues
```

它不包含 large-only environment 和 caps。RESP charges 从 `charged_large_model` 按 original PDB serial 映射回来：

```text
charge_map[original_serial] -> site atom charge
```

找不到对应 serial 会失败。

## 5. Workflow

当前 `run_metal_abinitio(...)` 顺序：

```text
plan artifacts
 -> apply cfmol2 templates
 -> build cofactor orig frcmods
 -> recognize initial metal core
 -> build large_model
 -> annotate metal formal charge
 -> infer large charge
 -> write raw large PDB
 -> optimize large_model
 -> reselect optimized core donors
 -> build RESP problem
 -> run Gaussian ESP + RESP
 -> deploy site_model
 -> project RESP charges
 -> write site deployment files
 -> Hessian + mSeminario/Seminario
 -> write final metal frcmod
 -> run tleap validation
 -> return MetalWorkflowResult
```

优化后会基于 optimized coordinates 重新判断 donor/core，除非用户通过 `set_bonded` 显式固定 metal-donor pair。

## 6. RESP Charge Chain

RESP 在 optimized `large_model` 上运行：

```text
large_model
 -> Gaussian ESP input
 -> Gaussian log
 -> espgen
 -> resp stage 1/2
 -> resp2.chg
 -> charged_large_model
```

如果包含 ion，Gaussian route 使用 `ReadRadii`。radii entries 根据：

```text
element + metal_formal_charge + water model
```

生成。

部署阶段：

```text
charged_large_model charges
 -> original serial mapping
 -> site_model charges
 -> per-residue mol2
```

不做额外归一化。

## 7. Bonded Parameter Chain

MetalAA 不运行 TorsionFit。金属相关 bonded 参数来自 Hessian：

```text
large_model optimized coordinates
 -> Cartesian Hessian
 -> mSeminario/Seminario
 -> fitted BOND/ANGLE
 -> remap to site_model atom indices
```

mSeminario 更新：

```text
BOND:  kBond, rEq
ANGLE: kTheta, thetaEq
```

金属相关 DIHE 当前作为 zero torsion 写入，用于满足 AMBER 的 bonded pattern 需求，不表示额外拟合了 torsion barrier。

## 8. Atom Type Remap

`artifacts.py` 给最终 site 局部重命名：

- metal atom: `M1`, `M2`, ...
- direct donor atom: `Y1`, `Y2`, ...
- donor residue 其它原子保持原 Amber/GAFF atom type。

旧 atom type 来源：

- protein residue: protein force-field library。
- water/ion: built-in lookup。
- cofactor/ligand: `cfmol2` 注入。

缺旧 atom type 会失败，而不是猜测。

## 9. Final FRCMOD

最终 `<base>_metal.frcmod` 是唯一 cofactor/metal 参数入口：

```text
cofactor orig full frcmod
+ renamed donor MASS/NONBON
+ renamed donor inherited BOND/ANGLE/DIHE/IMPROPER
+ metal fitted BOND/ANGLE
+ metal-related zero DIHE
```

DIHE 继承遵循 AMBER/tleap wildcard 语义：

- exact source -> exact renamed term。
- wildcard source, 如 `X-cc-nd-X` -> wildcard-aware renamed term，如 `X-cc-Y2-X`。
- metal-related torsion -> zero torsion。
- nonmetal renamed torsion 找不到来源时失败。

IMPROPER 继承只从已有参数源来，不对任意 improper 泛化默认值。

## 10. Tleap Deployment

MetalAA 生成：

```text
<base>_metal_tleap.pdb
<base>_metal_tleap.in
<base>_metal_tleap.out
```

`tleap.in` 包含：

```text
source leaprc.protein.<prom>
source leaprc.gaff2
source leaprc.water.<watm>
addAtomTypes ...
loadmol2 per-residue files
loadamberparams ion frcmods
loadamberparams <base>_metal.frcmod
mol = loadpdb <base>_metal_tleap.pdb
bond metal-donor pairs
bond peptide reconnect pairs
bond disulfide pairs
check/charge/solvate/addions/saveamberparm
```

MetalAA 会自动运行：

```text
tleap -s -f <base>_metal_tleap.in |tee <base>_metal_tleap.out
```

abinitio summary 读取 tleap out 并报告 `Errors / Warnings / Notes`。

## 11. MCPB 对照

| MCPB 概念 | MetalAA 当前实现 |
|---|---|
| site recognition | `recognize.py` |
| large model | `large_model` |
| standard model | `site_model` |
| small model | 未单独实现 |
| fingerprint | 不使用；按 original PDB serial 映射 |
| modified Seminario | `apply_mseminario(...)` |
| RESP charge fitting | `run_resp_pipeline(...)` |
| charge projection | `project_resp_charges_onto_site_model(...)` |
| final tleap | `artifacts.py` 生成并验证 |

MetalAA 与 MCPB 的共同点：

- 金属相关 atom type 局部重命名。
- 用较大环境做 RESP。
- 用 Hessian/Seminario 补 metal BOND/ANGLE。
- 用 explicit `bond` commands 恢复 metal-donor 连接。

区别：

- MetalAA 不拆 small/standard/large 三模型。
- MetalAA 不使用 MCPB fingerprint 文件。
- MetalAA final frcmod 合并 cofactor orig 与 metal fitted 项，用户不需要分批加载 cofactor orig frcmod。

## 12. CMAP 与蛋白力场

MetalAA 只重命名金属和 direct donor atom type。参与金属配位的通常是 sidechain atom，不改变标准蛋白 backbone atom type。因此标准蛋白 residue 的 ff19SB/ff14SB backbone 行为仍由 protein library 决定。

MetalAA 不生成 CMAP。对于金属位点参数化，这通常不是问题，因为 MetalAA 的新增项集中在 metal-donor 局部 bonded terms，而不是把标准主链 residue 改成新的 NCAA template。

## 13. 失败点

常见失败来源：

- target 不是 ion。
- donor cutoff 内有另一个 ion。
- ligand/cofactor donor 缺 atom type 且未提供 `cfmol2`。
- `cfmol2` atom names 无法匹配 PDB residue。
- Gaussian / espgen / resp / parmchk2 / tleap 外部程序失败。
- RESP charge 数量与 large model atom 数不一致。
- site atom 无法按 original serial 从 charged large model 找到 charge。
- Hessian 缺失或 shape 不合法。
- final frcmod 缺 inherited nonmetal renamed parameter。

这些失败通常说明输入结构、cofactor template 或参数来源不完整，不应通过静默猜参绕过。
