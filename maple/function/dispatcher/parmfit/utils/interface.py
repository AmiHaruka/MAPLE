"""Usage: run external QM and AmberTools commands for parmfit workflows."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from math import degrees
from pathlib import Path
import shlex
import shutil
import subprocess as sp

from .ionparams import collect_gaussian_readradii_entries
from .readparm import CorrectionParameterSet
from .structure import parse_pdb_coord

"""
Origin idea from: 
Hsuchein  | https://github.com/Hsuchein/AutoNACC
HighRelax | J. Chem. Theory Comput. 2026, 22, 6, 3093–3102
"""

# =============================================================================
# QM backend adapter
# =============================================================================

@dataclass(frozen=True)
class QMMethod:
    backend: str
    theory: str
    basis: str
    nproc: int
    mem: int
    route: str


@dataclass(frozen=True)
class RespConfig:
    qm: QMMethod
    chgmod: int = 1
    fixchg_resids: list[str] = field(default_factory=list)
    watm: str | None = None


@dataclass(frozen=True)
class AntechamberResult:
    ac_path: str
    input_path: str
    input_format: str
    residue_name: str


@dataclass(frozen=True)
class PrepgenResult:
    prepin_path: str
    res_path: str
    newpdb_path: str
    mainchain_path: str
    residue_name: str


@dataclass(frozen=True)
class Parmchk2Result:
    frcmod_path: str
    input_path: str
    residue_name: str


@dataclass(frozen=True)
class TleapResult:
    input_path: str
    output_path: str
    returncode: int
    command: str


def set_method(params: dict, *, backend: str = "gaussian") -> QMMethod:
    raw = dict(params)
    resolved_backend = str(raw.get("backend", backend)).strip().lower()
    if resolved_backend != "gaussian":
        raise NotImplementedError(f"Unsupported QM backend {resolved_backend!r}; only Gaussian is implemented.")
    theory = str(raw.get("theory", "HF")).strip()
    basis = str(raw.get("basis", "6-31G(d)")).strip()
    nproc = int(raw.get("nproc", 8))
    mem = int(raw.get("mem", 16))
    route = str(raw.get("route", "")).strip()
    return QMMethod(
        backend=resolved_backend,
        theory=theory,
        basis=basis,
        nproc=nproc,
        mem=mem,
        route=route,
    )


# =============================================================================
# Gaussian interface
# =============================================================================

def _resolve_gaussian_command(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    for candidate in ("g16", "g09"):
        resolved = shutil.which(candidate)
        if resolved: return resolved
    raise RuntimeError("Gaussian executable was not found. Expected one of: g16, g09.")


def prepare_gaussian_esp_input(
    path: str,
    model: dict,
    *,
    total_charge: int,
    multiplicity: int,
    decision: QMMethod,
    title: str = "MAPLE RESP",
    watm: str | None = None,
) -> str:
    atoms_flat = [atom for residue in model["residues"] for atom in sorted(residue["atoms"], key=lambda item: item["serial"])]
    radii_entries = collect_gaussian_readradii_entries(model, watm=watm)
    chk_name = Path(path).with_suffix(".chk").name
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"%chk={chk_name}\n")
        handle.write(f"%nproc={decision.nproc}\n")
        handle.write(f"%mem={decision.mem}GB\n")
        route_terms = [
            f"{decision.theory}/{decision.basis}",
            "Pop(MK,ReadRadii)" if radii_entries else "Pop=MK",
            "IOp(6/33=2) Guess=Read",
            "SCF=Tight",
        ]
        if decision.route:
            route_terms.append(decision.route)
        handle.write(f"#P {' '.join(route_terms)}\n")
        handle.write(f"\n{title}\n\n")
        handle.write(f"{int(total_charge)} {int(multiplicity)}\n")
        for atom in atoms_flat:
            x, y, z = atom["xyz"]
            handle.write(f" {atom['element']:<2s} {x:16.8f} {y:16.8f} {z:16.8f}\n")
        handle.write("\n")
        for element, radius in radii_entries:
            handle.write(f"{element} {radius:.3f}\n")
        if radii_entries:
            handle.write("\n")
    return path


def _ensure_gaussian_normal_termination(log_file: str) -> str:
    candidates = [log_file, os.path.splitext(log_file)[0] + ".out"]
    for candidate in candidates:
        if not os.path.isfile(candidate):
            continue
        with open(candidate, "r", encoding="utf-8", errors="replace") as handle:
            content = handle.read()
        if "Error termination" in content:
            raise RuntimeError(f"Gaussian terminated abnormally. See {candidate}.")
        if "Normal termination" in content:
            return candidate
    raise RuntimeError(f"Gaussian did not finish normally; missing a successful log/out file for {log_file}.")


def run_gaussian(gjf: str, decision: QMMethod) -> str:
    explicit_cmd = None
    command = _resolve_gaussian_command(explicit=str(explicit_cmd) if explicit_cmd else None)
    log_file = os.path.splitext(gjf)[0] + ".log"
    cwd = os.path.dirname(os.path.abspath(gjf))
    result = sp.run(
        [command, os.path.basename(gjf)],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(f"Gaussian execution failed for {gjf}: {stderr}")
    _ensure_gaussian_normal_termination(log_file)
    return log_file


def monitor_gaussian(log_file, interval=60):
    del interval
    try:
        return True, _ensure_gaussian_normal_termination(log_file)
    except RuntimeError as exc:
        print(f"  [ERROR] {exc}")
        return False, log_file


def write_gaussian_input(path, elems, coords, cfg):
    decision = set_method(cfg)
    atoms = [{"element": element, "xyz": xyz} for element, xyz in zip(elems, coords)]
    model = {"residues": [{"atoms": atoms}]}
    return prepare_gaussian_esp_input(
        path,
        model,
        total_charge=int(cfg["net_charge"]),
        multiplicity=int(cfg.get("multiplicity", 1)),
        decision=decision,
        title=f"ACE-Target-NME ESP",
    )


# =============================================================================
# AmberTools interface
# =============================================================================

def _run_cmd(cmd, cwd=None):
    print(f"  [CMD] {cmd}")
    r = sp.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
    if r.returncode != 0:
        print(f"    STDERR: {r.stderr[:800]}")
    return r.returncode, r.stdout, r.stderr


def _result_path(path: str, cwd: str | None) -> str:
    if os.path.isabs(path) or cwd is None:
        return os.path.abspath(path)
    return os.path.abspath(os.path.join(cwd, path))


def read_ac_names(ac_path):
    names = []
    with open(ac_path) as f:
        for line in f:
            if line.startswith("ATOM"):
                names.append(line.split()[2])
    return names


def write_residue_pdb(ac_path, out_pdb, n_ace, n_res, rn, cfg):
    """从 AC 文件提取残基原子, 用原始坐标输出 PDB (原子名与 prepin 一致)."""
    # 读取残基输入 PDB 的坐标 (未经 Gaussian 优化的原始坐标)
    res_file = cfg["residue_file"]
    with open(res_file) as f:
        res_lines = [l for l in f if l[:6].strip() in ("ATOM", "HETATM")]
    res_coords = [parse_pdb_coord(line) for line in res_lines]

    # 读取 AC 文件中残基部分的原子名
    ac_atoms = []
    with open(ac_path) as f:
        for line in f:
            if line.startswith("ATOM"):
                name = line.split()[2]
                # 元素从原子名提取: 取前缀字母部分
                elem = ''.join(c for c in name if c.isalpha())
                if len(elem) > 2:
                    elem = elem[:1]
                ac_atoms.append((name, elem.capitalize()))
    res_ac = ac_atoms[n_ace:n_ace + n_res]

    if len(res_ac) != len(res_coords):
        print(f"  [WARNING] AC 残基原子数 {len(res_ac)} != 输入 PDB {len(res_coords)}")
        return

    with open(out_pdb, 'w') as f:
        for i, ((name, elem), (x, y, z)) in enumerate(zip(res_ac, res_coords)):
            serial = i + 1
            if len(name) <= 3:
                name_field = f" {name:<3s}"
            else:
                name_field = f"{name:<4s}"
            f.write(f"ATOM  {serial:5d} {name_field}"
                    f" {rn:>3s} A   1    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}"
                    f"{1.0:6.2f}{0.0:6.2f}"
                    f"          {elem:>2s}  \n")
        f.write("END\n")


def run_antechamber(
    input_file,
    cfg,
    workdir,
    *,
    input_format: str = "gout",
    output_format: str = "ac",
    charge_mode: str | None = None,
    charge_file: str | None = None,
):
    rn = cfg["residue_name"]
    output_name = f"{rn}.{output_format}"
    cmd = (
        f"{cfg.get('antechamber', 'antechamber')} "
        f"-i {input_file} -fi {input_format} "
        f"-o {output_name} -fo {output_format} "
        f"-rn {rn} -at gaff2 "
        f"-nc {cfg['net_charge']} -pf y"
    )
    if charge_mode is not None:
        cmd += f" -c {charge_mode}"
    if charge_file is not None:
        cmd += f" -cf {charge_file}"
    if input_format == "gout" and charge_mode is None:
        cmd += " -c resp -s 2"
    rc, _, err = _run_cmd(cmd, cwd=workdir)
    if rc != 0:
        raise RuntimeError(f"antechamber 失败:\n{err}")
    return AntechamberResult(
        ac_path=os.path.join(workdir, output_name),
        input_path=_result_path(input_file, workdir),
        input_format=input_format,
        residue_name=rn,
    )


def write_mainchain_mc(path, ace_names, nme_names,
                       head, tail, mainchain, charge, extra_omit_names=None):
    extra_omit_names = list(extra_omit_names or [])
    with open(path, 'w') as f:
        f.write(f"HEAD_NAME {head}\n")
        f.write(f"TAIL_NAME {tail}\n")
        for mc in mainchain:
            f.write(f"MAIN_CHAIN {mc}\n")
        for nm in ace_names:
            f.write(f"OMIT_NAME {nm}\n")
        for nm in nme_names:
            f.write(f"OMIT_NAME {nm}\n")
        for nm in extra_omit_names:
            f.write(f"OMIT_NAME {nm}\n")
        f.write("PRE_HEAD_TYPE C\n")
        f.write("POST_TAIL_TYPE N\n")
        f.write(f"CHARGE {charge:.1f}\n")


def run_prepgen(ac_file, mc_file, cfg, workdir):
    rn = cfg["residue_name"]
    out = f"{rn}.prepin"
    res = f"{rn}.res"
    cmd = (f"{cfg.get('prepgen', 'prepgen')} "
           f"-i {ac_file} "
           f"-o {out} "
           f"-m {mc_file} "
           f"-rn {rn} "
           f"-rf {res}")
    rc, _, err = _run_cmd(cmd, cwd=workdir)
    if rc != 0:
        raise RuntimeError(f"prepgen 失败:\n{err}")
    return PrepgenResult(
        prepin_path=os.path.join(workdir, out),
        res_path=os.path.join(workdir, res),
        newpdb_path=os.path.join(workdir, "NEWPDB.PDB"),
        mainchain_path=_result_path(mc_file, workdir),
        residue_name=rn,
    )


def run_parmchk2(input_file, cfg, ifmol2, workdir):
    rn = cfg["residue_name"]
    out = f"{rn}.frcmod"
    cmd = (f"{cfg.get('parmchk2', 'parmchk2')} "
           f"-i {input_file} -f {'mol2' if ifmol2 else 'prepi'} "
           f"-a Y -s gaff2 "
           f"-o {out} ")
    rc, _, err = _run_cmd(cmd, cwd=workdir)
    if rc != 0:
        raise RuntimeError(f"parmchk2 失败:\n{err}")
    return Parmchk2Result(
        frcmod_path=os.path.join(workdir, out),
        input_path=_result_path(input_file, workdir),
        residue_name=rn,
    )


def run_tleap(input_file: str, workdir: str | None = None, executable: str = "tleap") -> TleapResult:
    input_path = os.path.abspath(input_file if workdir is None else _result_path(input_file, workdir))
    resolved_workdir = os.path.abspath(workdir or os.path.dirname(input_path) or ".")
    input_name = os.path.basename(input_path)
    output_name = os.path.splitext(input_name)[0] + ".out"
    output_path = os.path.join(resolved_workdir, output_name)
    command_input = input_name if os.path.dirname(input_path) == resolved_workdir else input_path
    display = f"{executable} -s -f {command_input} |tee {output_name}"
    shell_command = (
        "set -o pipefail; "
        f"{shlex.quote(executable)} -s -f {shlex.quote(command_input)} "
        f"2>&1 |tee {shlex.quote(output_name)}"
    )
    print(f"  [CMD] {display}")
    result = sp.run(
        ["bash", "-lc", shell_command],
        cwd=resolved_workdir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        with open(output_path, "a", encoding="utf-8") as handle:
            handle.write(f"\nMAPLE_TLEAP_RETURN_CODE = {int(result.returncode)}\n")
    return TleapResult(
        input_path=input_path,
        output_path=output_path,
        returncode=int(result.returncode),
        command=display,
    )



# =============================================================================
# prepin 电荷校验
# =============================================================================

def verify_prepin_charge(prepin_path, expected_charge):
    """校验 prepin 文件中残基总电荷是否为整数且等于期望值."""
    total = 0.0
    n_atoms = 0
    has_nan = False
    with open(prepin_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 11:
                continue
            # 跳过 DUMM 行
            if parts[1] == 'DUMM':
                continue
            try:
                int(parts[0])
                charge = float(parts[10])
                total += charge
                n_atoms += 1
            except (ValueError, IndexError):
                continue
            # 检测 nan
            if 'nan' in line.lower():
                has_nan = True

    ok = True
    if has_nan:
        print(f"  [WARNING] prepin 含有 nan, prepgen 内坐标树构建可能有问题!")
        ok = False

    rounded = round(total)
    if abs(total - rounded) > 0.01:
        print(f"  [WARNING] prepin 总电荷 {total:.6f} 不是整数! ({n_atoms} atoms)")
        ok = False
    if rounded != expected_charge:
        print(f"  [WARNING] prepin 电荷 {rounded} != 配置 residue_charge {expected_charge}")
        ok = False
    if ok:
        print(f"  电荷校验通过: {total:.4f} ≈ {expected_charge} ({n_atoms} atoms)")
    return ok


def fix_prepin_charge(prepin_path, expected_charge):
    """修正 prepin 电荷: 将误差均匀分配到主链 (M 标记) 原子上.

    仅在总电荷与期望值差异 > 1e-4 时修正.
    """
    with open(prepin_path) as f:
        lines = f.readlines()

    # 第一遍: 计算总电荷, 找主链原子行号
    total = 0.0
    backbone_indices = []   # line indices of M-flagged atoms
    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) < 11:
            continue
        if parts[1] == 'DUMM':
            continue
        try:
            int(parts[0])
            charge = float(parts[10])
        except (ValueError, IndexError):
            continue
        total += charge
        # parts[3] 是 tree 标志: M=mainchain, S=sidechain, B=branch, E=end
        if parts[3] == 'M':
            backbone_indices.append(i)

    error = expected_charge - total
    if abs(error) < 1e-4:
        return False   # 无需修正

    if not backbone_indices:
        print(f"  [WARNING] 无主链原子 (M 标记), 无法分配电荷修正 {error:.6f}")
        return False

    correction = error / len(backbone_indices)
    print(f"  电荷修正: {total:.6f} → {expected_charge} "
          f"(误差 {error:+.6f}, 分配到 {len(backbone_indices)} 个主链原子, "
          f"每个 {correction:+.6f})")

    # 第二遍: 修正主链原子电荷
    for i in backbone_indices:
        line = lines[i]
        parts = line.split()
        old_charge = float(parts[10])
        new_charge = old_charge + correction

        # prepin 电荷字段在最后一列, 找到其起始位置并替换
        # 格式: 最后一个字段是电荷, 宽度一般为 10 字符 (%10.6f)
        last_space = line.rstrip('\n').rfind(' ')
        lines[i] = line[:last_space + 1] + f"{new_charge:.6f}\n"

    with open(prepin_path, 'w') as f:
        f.writelines(lines)

    return True


# =============================================================================
# ff14SB / gaff2 交叉参数库
# =============================================================================
# 非标准氨基酸 (gaff2) 接入蛋白 (ff14SB) 时, 肽键连接处产生大小写
# 混合的原子类型参数. 此库自动补充缺失的交叉项.

def _bond_key(line):
    if len(line) < 5 or line[2] != '-':
        return None
    t = (line[0:2].strip(), line[3:5].strip())
    if not t[0] or not t[1]:
        return None
    return min(t, t[::-1])

def _angle_key(line):
    if len(line) < 8 or line[2] != '-' or line[5] != '-':
        return None
    t = (line[0:2].strip(), line[3:5].strip(), line[6:8].strip())
    if not all(t):
        return None
    return min(t, t[::-1])

def _dihe_key(line):
    if len(line) < 11 or line[2] != '-' or line[5] != '-' or line[8] != '-':
        return None
    t = (line[0:2].strip(), line[3:5].strip(), line[6:8].strip(), line[9:11].strip())
    if not all(t):
        return None
    return min(t, t[::-1])


# --- 交叉项参数 ---
# HEAD 连接: ff14SB C(=O) → gaff2 ns   (蛋白前一残基 → 非标准残基 N端)
# TAIL 连接: gaff2 c(=O) → ff14SB N(-H) (非标准残基 C端 → 蛋白后一残基)

_CROSSTERM_BOND = [
    "C -ns  490.000   1.335       ff14SB/gaff2 peptide bond\n",
    "c -N   490.000   1.335       ff14SB/gaff2 peptide bond\n",
]

_CROSSTERM_ANGLE = [
    # HEAD: protein C(=O) → residue ns
    "O -C -ns    80.000     122.900   ff14SB(O,C)/gaff2(ns)\n",
    "C -ns-hn    50.000     120.000   ff14SB(C)/gaff2(ns,hn)\n",
    "C -ns-c3    50.000     121.900   ff14SB(C)/gaff2(ns,c3)\n",
    "CX-C -ns    70.000     116.600   ff14SB(CX,C)/gaff2(ns)\n",
    # TAIL: residue c(=O) → protein N(-H)
    "o -c -N     80.000     122.900   gaff2(o,c)/ff14SB(N)\n",
    "c -N -H     80.000     122.900   gaff2(c)/ff14SB(N,H)\n",
    "c -N -CX    50.000     121.900   gaff2(c)/ff14SB(N,CX)\n",
    "c3-c -N     70.000     116.600   gaff2(c3,c)/ff14SB(N)\n",
]

_CROSSTERM_DIHE = [
    # HEAD: central bond C-ns (ff14SB C=O to gaff2 amide N)
    "O -C -ns-hn   1    2.500       180.000          -2.000      ff14SB/gaff2\n",
    "O -C -ns-hn   1    2.000         0.000           1.000      ff14SB/gaff2\n",
    "O -C -ns-c3   4   10.000       180.000           2.000      ff14SB/gaff2\n",
    "CX-C -ns-c3   4   10.000       180.000           2.000      ff14SB/gaff2\n",
    "CX-C -ns-hn   4   10.000       180.000           2.000      ff14SB/gaff2\n",
    # HEAD: central bond ns-c3
    "C -ns-c3-c    6    0.000         0.000           2.000      ff14SB/gaff2\n",
    "C -ns-c3-c3   6    0.000         0.000           2.000      ff14SB/gaff2\n",
    "C -ns-c3-h1   6    0.000         0.000           2.000      ff14SB/gaff2\n",
    # TAIL: central bond c-N (gaff2 C=O to ff14SB amide N)
    "o -c -N -H    1    2.500       180.000          -2.000      ff14SB/gaff2\n",
    "o -c -N -H    1    2.000         0.000           1.000      ff14SB/gaff2\n",
    "o -c -N -CX   4   10.000       180.000           2.000      ff14SB/gaff2\n",
    "c3-c -N -H    4   10.000       180.000           2.000      ff14SB/gaff2\n",
    "c3-c -N -CX   4   10.000       180.000           2.000      ff14SB/gaff2\n",
    # TAIL: central bond N-CX
    "c -N -CX-C    4   10.000       180.000           2.000      ff14SB/gaff2\n",
    "c -N -CX-H1   4   10.000       180.000           2.000      ff14SB/gaff2\n",
]


def patch_frcmod_crossterms(frcmod_path):
    """补充 ff14SB/gaff2 交叉参数到 frcmod."""
    with open(frcmod_path) as f:
        lines = f.readlines()

    SECTIONS = {'MASS', 'BOND', 'ANGLE', 'ANGL', 'DIHE', 'IMPROPER', 'IMPR', 'NONBON'}
    current = None
    existing = {'BOND': set(), 'ANGLE': set(), 'DIHE': set()}
    section_end = {}   # section_name -> line index of terminating blank line

    for i, line in enumerate(lines):
        s = line.strip()
        if s in SECTIONS:
            current = {'ANGL': 'ANGLE', 'IMPR': 'IMPROPER'}.get(s, s)
        elif s == '' and current:
            section_end[current] = i
            current = None
        elif current in existing:
            fn = {'BOND': _bond_key, 'ANGLE': _angle_key, 'DIHE': _dihe_key}[current]
            k = fn(line)
            if k:
                existing[current].add(k)

    # 缺失的 BOND / ANGLE
    missing = {'BOND': [], 'ANGLE': [], 'DIHE': []}

    for line in _CROSSTERM_BOND:
        k = _bond_key(line)
        if k and k not in existing['BOND']:
            missing['BOND'].append(line)
            existing['BOND'].add(k)

    for line in _CROSSTERM_ANGLE:
        k = _angle_key(line)
        if k and k not in existing['ANGLE']:
            missing['ANGLE'].append(line)
            existing['ANGLE'].add(k)

    # 缺失的 DIHE (支持多项式: 同一 key 可能有多行)
    dihe_groups = {}
    for line in _CROSSTERM_DIHE:
        k = _dihe_key(line)
        if k:
            dihe_groups.setdefault(k, []).append(line)
    for k, grp in dihe_groups.items():
        if k not in existing['DIHE']:
            missing['DIHE'].extend(grp)
            existing['DIHE'].add(k)

    total = sum(len(v) for v in missing.values())
    if total == 0:
        return 0

    # 在各 section 的结束空行前插入交叉项
    output = []
    for i, line in enumerate(lines):
        for sec in ('BOND', 'ANGLE', 'DIHE'):
            if i == section_end.get(sec) and missing[sec]:
                for ml in missing[sec]:
                    output.append(ml)
        output.append(line)

    with open(frcmod_path, 'w') as f:
        f.writelines(output)
    return total


def write_refined_frcmod(
    parameter_set: CorrectionParameterSet,
    frcmod_path: str,
    *,
    mass_params: dict[str, float],
    remark: str = "REMARK MAPLE refined frcmod",
) -> str:
    _validate_refined_frcmod_inputs(parameter_set, mass_params)

    bond_params: dict[tuple[str, str], tuple[float, float]] = {}
    angle_params: dict[tuple[str, str, str], tuple[float, float]] = {}
    dihedral_params: dict[tuple[str, str, str, str], list] = {}
    improper_params: dict[tuple[str, str, str, str], list] = {}
    nonbond_params: dict[str, tuple[float, float]] = {}

    for bond in parameter_set.bonds:
        if bond.kBond is None or bond.rEq is None:
            continue
        reverse = (bond.atom_types[1], bond.atom_types[0])
        key = bond.atom_types if bond.atom_types <= reverse else reverse
        bond_params[key] = (float(bond.kBond), float(bond.rEq))

    for angle in parameter_set.angles:
        if angle.kTheta is None or angle.thetaEq is None:
            continue
        reverse = (angle.atom_types[2], angle.atom_types[1], angle.atom_types[0])
        key = angle.atom_types if angle.atom_types <= reverse else reverse
        angle_params[key] = (float(angle.kTheta), float(angle.thetaEq))

    for dihedral in parameter_set.dihedrals:
        if not dihedral.terms:
            continue
        reverse = tuple(reversed(dihedral.atom_types))
        key = dihedral.atom_types if dihedral.atom_types <= reverse else reverse
        dihedral_params[key] = list(dihedral.terms)

    for improper in parameter_set.impropers:
        if not improper.terms:
            continue
        improper_params[improper.atom_types] = list(improper.terms)

    for nonbond in parameter_set.nonbonds:
        if nonbond.rmin_half is None or nonbond.epsilon is None:
            continue
        nonbond_params[nonbond.atom_type] = (float(nonbond.rmin_half), float(nonbond.epsilon))

    lines = [f"{remark}\n", "\n", "MASS\n"]
    for atom_type, mass in mass_params.items():
        lines.append(f"{atom_type:<2s}  {mass:10.3f}\n")

    lines.extend(["\n", "BOND\n"])
    for atom_types, (k_bond, r_eq) in sorted(bond_params.items()):
        lines.append(f"{atom_types[0]:<2s}-{atom_types[1]:<2s}  {k_bond:10.3f}  {r_eq:8.4f}\n")

    lines.extend(["\n", "ANGLE\n"])
    for atom_types, (k_theta, theta_eq) in sorted(angle_params.items()):
        lines.append(
            f"{atom_types[0]:<2s}-{atom_types[1]:<2s}-{atom_types[2]:<2s}  {k_theta:10.3f}  {degrees(theta_eq):9.3f}\n"
        )

    lines.extend(["\n", "DIHE\n"])
    for atom_types, terms in sorted(dihedral_params.items()):
        for term in terms:
            lines.append(
                f"{atom_types[0]:<2s}-{atom_types[1]:<2s}-{atom_types[2]:<2s}-{atom_types[3]:<2s}"
                f"  {1:4d}  {float(term.kPhi):10.4f}  {degrees(float(term.phase)):9.3f}  {float(term.period):8.3f}\n"
            )

    lines.extend(["\n", "IMPROPER\n"])
    for atom_types, terms in sorted(improper_params.items()):
        for term in terms:
            lines.append(
                f"{atom_types[0]:<2s}-{atom_types[1]:<2s}-{atom_types[2]:<2s}-{atom_types[3]:<2s}"
                f"  {float(term.kPhi):10.4f}  {degrees(float(term.phase)):9.3f}  {float(term.period):8.3f}\n"
            )

    lines.extend(["\n", "NONBON\n"])
    for atom_type, (rmin_half, epsilon) in sorted(nonbond_params.items()):
        lines.append(f"{atom_type:<2s}  {rmin_half:10.6f}  {epsilon:10.6f}\n")
    lines.append("\n")

    with open(frcmod_path, "w", encoding="utf-8") as handle:
        handle.writelines(lines)
    return frcmod_path


def _validate_refined_frcmod_inputs(
    parameter_set: CorrectionParameterSet,
    mass_params: dict[str, float],
) -> None:
    for nonbond in parameter_set.nonbonds:
        if nonbond.atom_type not in mass_params:
            raise ValueError(f"Cannot write refined frcmod MASS for atom type {nonbond.atom_type!r}.")
        if nonbond.rmin_half is None or nonbond.epsilon is None:
            raise ValueError(
                f"Cannot write refined frcmod NONBON for atom {nonbond.atom} ({nonbond.atom_type}): "
                "missing rmin_half/epsilon."
            )

    for bond in parameter_set.bonds:
        if bond.kBond is None or bond.rEq is None:
            raise ValueError(f"Cannot write refined frcmod BOND {'-'.join(map(str, bond.atoms))}: missing kBond/rEq.")

    for angle in parameter_set.angles:
        if angle.kTheta is None or angle.thetaEq is None:
            raise ValueError(
                f"Cannot write refined frcmod ANGLE {'-'.join(map(str, angle.atoms))}: missing kTheta/thetaEq."
            )

    for dihedral in parameter_set.dihedrals:
        if not dihedral.terms:
            raise ValueError(
                f"Cannot write refined frcmod DIHE {'-'.join(map(str, dihedral.atoms))}: missing torsion terms."
            )

    for improper in parameter_set.impropers:
        if not improper.terms:
            raise ValueError(
                f"Cannot write refined frcmod IMPROPER {'-'.join(map(str, improper.atoms))}: missing torsion terms."
            )
