#!/bin/bash
# 用法: ./plot_md.sh <*_md_thermo.dat>
#
# 生成 MD 模拟科学验证图（每项一个独立窗口），用于论文发表前的结果校验。
# 参考文献：
#   [1] Páll et al., J. Chem. Theory Comput. 2020 — 能量漂移率标准
#   [2] Case et al., AMBER 2022 Manual §3   — 能量涨落标准
#   [3] Shirts & Chodera, JCP 129, 124105 (2008) — KE-PE 反相关
#   [4] Allen & Tildesley, Computer Simulation of Liquids 2nd ed. 2017 §2.4

if [ $# -ne 1 ]; then
    echo "用法: $0 <*_md_thermo.dat>"
    exit 1
fi

input_file="$1"

if [ ! -f "$input_file" ]; then
    echo "错误：文件 '$input_file' 不存在。"
    exit 1
fi

python3 - "$input_file" <<'EOF'
import re, sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from pathlib import Path
from scipy.stats import norm

filename = sys.argv[1]

# ─── 解析 thermo.dat ──────────────────────────────────────────────────────
col_names, rows = [], []
ensemble = 'MD'

with open(filename) as fh:
    for line in fh:
        s = line.strip()
        if not s:
            continue
        if s.startswith('#'):
            m = re.search(r'MD Simulation - (\w+) Ensemble', s)
            if m:
                ensemble = m.group(1)
            cand = s.lstrip('#').split()
            if cand and re.match(r'(?i)step|time', cand[0]):
                col_names = cand
            continue
        try:
            rows.append([float(v) for v in s.split()])
        except ValueError:
            continue

if not rows:
    print("错误：未能解析数据。")
    sys.exit(1)

data = np.array(rows)
n = data.shape[1]
col_names = (col_names + [f"Col{i+1}" for i in range(len(col_names), n)])[:n]

def col(pat):
    for i, c in enumerate(col_names):
        if re.search(pat, c, re.I):
            return data[:, i]
    return None

time  = col(r'time\(fs\)|^time$')
step  = col(r'^step$')
T     = col(r'temp')
KE    = col(r'^ke\b|^ke\(')
PE    = col(r'^pe\b|^pe\(')
TE    = col(r'^te\b|^te\(')
press = col(r'press')
vol   = col(r'vol')

if TE is None or T is None:
    print("错误：文件中缺少必要列（Temp / TE）。")
    sys.exit(1)

xdata  = time if time is not None else step
xlabel = 'Time (fs)' if time is not None else 'Step'

# ─── 统计量 ───────────────────────────────────────────────────────────────
Ha_to_kJ = 2625.4996
fs_to_ns  = 1e-6

coef        = np.polyfit(xdata, TE, 1)
te_fit      = np.polyval(coef, xdata)
te_fluct    = TE.std() / abs(TE.mean())
drift_kJ_ns = coef[0] * Ha_to_kJ / fs_to_ns
residual    = (TE - te_fit) * Ha_to_kJ * 1000   # J/mol
rms_resid   = np.sqrt(np.mean(residual**2))

r_kepe  = float(np.corrcoef(KE, PE)[0, 1]) if KE is not None and PE is not None else None
T_mean  = T.mean()
T_std   = T.std()
is_npt  = press is not None and vol is not None

# ─── 打印摘要 ─────────────────────────────────────────────────────────────
SEP = "=" * 60
print(SEP)
print(f" MD Validation Report  —  {ensemble} Ensemble")
print(SEP)
print(f" File   : {filename}")
print(f" Steps  : {len(xdata)}   Time: {xdata[-1]:.1f} fs")
print()
print(f" Energy Conservation:")
print(f"   TE mean         = {TE.mean():.8f} Ha")
print(f"   TE std          = {TE.std():.8f} Ha")
print(f"   TE drift (Δ)    = {TE[-1]-TE[0]:.8f} Ha")
print(f"   σ(TE)/|⟨TE⟩|    = {te_fluct:.2e}   [AMBER: < 1e-4]")
print(f"   Linear drift    = {drift_kJ_ns:.4g} kJ/mol/ns (total)")
print()
print(f" Temperature:")
print(f"   ⟨T⟩ = {T_mean:.2f} K   σ = {T_std:.2f} K   σ/⟨T⟩ = {T_std/T_mean:.4f}")
if r_kepe is not None:
    print()
    print(f" Integrator  [Shirts & Chodera JCP 2008]:")
    is_nve = ensemble.upper() == 'NVE'
    if is_nve:
        r_expect = f"r ≈ -1 expected (NVE: energy conservation forces anti-correlation)"
        r_ok_str = "✓" if r_kepe < -0.95 else "✗  (expected r ≈ -1 for NVE)"
    else:
        r_expect = f"r ≈ 0 expected ({ensemble} thermostat randomises KE each step)"
        r_ok_str = "✓" if r_kepe > -0.5 else "✗  (unexpectedly strong anti-correlation)"
    print(f"   KE-PE r = {r_kepe:.4f}   {r_ok_str}")
    print(f"   Note: {r_expect}")
print(SEP)
print()

# ─── 通用样式 ─────────────────────────────────────────────────────────────
BLUE  = '#1f77b4'
RED   = '#d62728'
GREEN = '#2ca02c'
GRAY  = '#7f7f7f'
title_base = Path(filename).stem.replace('_md_thermo', '')
sup = f"{title_base}  |  {ensemble} MD"

def new_fig(subtitle):
    """每个指标独立窗口，固定尺寸。"""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    fig.suptitle(sup, fontsize=11, fontweight='bold', y=1.0)
    ax.set_title(subtitle, fontsize=9, loc='left', pad=5)
    return fig, ax

def finish(ax, ylabel, xl=None):
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xlabel(xl or xlabel, fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.ticklabel_format(useOffset=False, style='plain', axis='y')
    ax.figure.tight_layout()

# ══════════════════════════════════════════════════════════════════════════
# 图 1 — Total Energy + linear drift
# ══════════════════════════════════════════════════════════════════════════
ok_fluct = te_fluct < 1e-4
fig1, ax1 = new_fig(
    f'Total Energy  |  σ/|⟨E⟩| = {te_fluct:.2e}'
    + ('  ✓' if ok_fluct else '  ✗  (>1×10⁻⁴, AMBER §3)')
)
ax1.plot(xdata, TE, color=BLUE, lw=0.9,
         marker='o', markersize=1.5, markevery=1, label='TE')
ax1.plot(xdata, te_fit, color=RED, lw=1.5, ls='--',
         label=f'Linear fit  ({drift_kJ_ns:.3g} kJ/mol/ns)')
ax1.legend(fontsize=8, loc='upper right')
finish(ax1, 'E_total (Ha)')

# ══════════════════════════════════════════════════════════════════════════
# 图 2 — TE residual（积分器噪声）
# ══════════════════════════════════════════════════════════════════════════
fig2, ax2 = new_fig(
    f'TE residual (drift removed)  RMS = {rms_resid:.2g} J/mol'
    '  —  integrator noise'
)
ax2.plot(xdata, residual, color=GREEN, lw=0.9,
         marker='o', markersize=1.5, markevery=1)
ax2.axhline(0, color=GRAY, lw=0.8, ls='--')
finish(ax2, 'ΔE (J/mol)')

# ══════════════════════════════════════════════════════════════════════════
# 图 3 — Temperature
# ══════════════════════════════════════════════════════════════════════════
fig3, ax3 = new_fig(
    f'Temperature  |  ⟨T⟩={T_mean:.1f} K'
    f'  σ={T_std:.1f} K  σ/⟨T⟩={T_std/T_mean:.3f}'
)
ax3.plot(xdata, T, color=RED, lw=0.9,
         marker='o', markersize=1.5, markevery=1)
ax3.axhline(T_mean, color=GRAY, lw=1.2, ls='--',
            label=f'⟨T⟩ = {T_mean:.1f} K')
ax3.fill_between(xdata, T_mean - T_std, T_mean + T_std,
                 alpha=0.15, color=RED, label=f'±σ = ±{T_std:.1f} K')
ax3.legend(fontsize=8, loc='upper right')
finish(ax3, 'T (K)')

# ══════════════════════════════════════════════════════════════════════════
# 图 4 — KE vs PE time series (twin Y-axes)
# ══════════════════════════════════════════════════════════════════════════
r_str = f'r = {r_kepe:.4f}' if r_kepe is not None else ''
is_nve = ensemble.upper() == 'NVE'
if r_kepe is None:
    ok_r = False
elif is_nve:
    ok_r = r_kepe < -0.95          # NVE: 必须强反相关
else:
    ok_r = r_kepe > -0.5           # NVT/NPT: Langevin 随机化 KE，r ≈ 0 才正常
fig4, ax4 = new_fig(
    f'KE – PE anti-correlation  {r_str}' + ('  ✓' if ok_r else '  ✗')
)
ax4r = ax4.twinx()
l1, = ax4.plot(xdata, KE, color=BLUE, lw=0.9,
               marker='o', markersize=1.5, markevery=1, label='KE')
l2, = ax4r.plot(xdata, PE, color=RED, lw=0.9,
                marker='o', markersize=1.5, markevery=1, alpha=0.7, label='PE')
ax4.set_ylabel('KE (Ha)', color=BLUE, fontsize=9)
ax4r.set_ylabel('PE (Ha)', color=RED, fontsize=9)
ax4.tick_params(axis='y', labelcolor=BLUE)
ax4r.tick_params(axis='y', labelcolor=RED)
ax4.ticklabel_format(useOffset=False, style='plain', axis='y')
ax4r.ticklabel_format(useOffset=False, style='plain', axis='y')
ax4.set_xlabel(xlabel, fontsize=9)
ax4.grid(True, linestyle='--', alpha=0.35)
ax4.xaxis.set_minor_locator(AutoMinorLocator())
ax4.legend(handles=[l1, l2], fontsize=8, loc='upper right')
fig4.tight_layout()

# ══════════════════════════════════════════════════════════════════════════
# 图 5 — NPT: Pressure / NVE+NVT: KE-PE scatter
# ══════════════════════════════════════════════════════════════════════════
if is_npt:
    fig5, ax5 = new_fig(
        f'Pressure  |  ⟨P⟩={press.mean():.1f} bar  σ={press.std():.1f} bar'
    )
    ax5.plot(xdata, press, color=BLUE, lw=0.9,
             marker='o', markersize=1.5, markevery=1)
    ax5.axhline(press.mean(), color=GRAY, lw=1.2, ls='--',
                label=f'⟨P⟩ = {press.mean():.1f} bar')
    ax5.legend(fontsize=8)
    finish(ax5, 'P (bar)')
else:
    if r_kepe is not None:
        scatter_note = (
            'NVE: r ≈ -1 confirms symplecticity'
            if is_nve else
            f'NVT/NPT: r ≈ 0 expected (thermostat decouples KE from PE)'
        )
        fig5, ax5 = new_fig(
            f'KE vs PE scatter  —  {scatter_note}'
        )
        ax5.scatter(KE, PE, s=5, alpha=0.45, color=BLUE, edgecolors='none')
        m_s, b_s = np.polyfit(KE, PE, 1)
        kx = np.linspace(KE.min(), KE.max(), 200)
        ax5.plot(kx, m_s * kx + b_s, color=RED, lw=1.5,
                 label=f'slope = {m_s:.2f}   r = {r_kepe:.4f}')
        ax5.legend(fontsize=8)
        finish(ax5, 'PE (Ha)', 'KE (Ha)')

# ══════════════════════════════════════════════════════════════════════════
# 图 6 — NPT: Volume / NVE+NVT: Temperature histogram
# ══════════════════════════════════════════════════════════════════════════
if is_npt:
    fig6, ax6 = new_fig(
        f'Volume  |  ⟨V⟩={vol.mean():.2f} Å³  σ={vol.std():.2f} Å³'
    )
    ax6.plot(xdata, vol, color=GREEN, lw=0.9,
             marker='o', markersize=1.5, markevery=1)
    ax6.axhline(vol.mean(), color=GRAY, lw=1.2, ls='--',
                label=f'⟨V⟩ = {vol.mean():.2f} Å³')
    ax6.legend(fontsize=8)
    finish(ax6, 'V (Å³)')
else:
    fig6, ax6 = new_fig(
        'Temperature distribution  '
        '[Allen & Tildesley, Computer Simulation of Liquids, 2017 §2.4]'
    )
    ax6.hist(T, bins=40, color=BLUE, alpha=0.6,
             edgecolor='white', lw=0.4,
             density=True, label='Simulation')
    tx = np.linspace(T.min(), T.max(), 300)
    ax6.plot(tx, norm.pdf(tx, T_mean, T_std),
             color=RED, lw=1.8,
             label=f'Gaussian  μ={T_mean:.1f}  σ={T_std:.1f}')
    ax6.legend(fontsize=8)
    finish(ax6, 'Probability density', 'T (K)')

# ─── 显示所有窗口 ─────────────────────────────────────────────────────────
try:
    plt.show()
except Exception as e:
    # 无图形界面时逐图保存
    stem = Path(filename).stem
    for i, fig in enumerate(map(plt.figure, plt.get_fignums()), 1):
        out = f"{stem}_validation_{i}.png"
        fig.savefig(out, bbox_inches='tight', dpi=150)
        print(f"  已保存: {out}")
EOF
