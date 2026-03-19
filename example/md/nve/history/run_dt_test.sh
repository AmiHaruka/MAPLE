#!/bin/bash
# 时间步长对比测试：0.5 / 0.25 / 0.125 fs
# 用法：cd example/md/nve && bash run_dt_test.sh
# 需要 conda 环境 maple 已激活，或修改下面的 MAPLE_CMD

set -e
MAPLE_CMD="maple"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

run_and_report() {
    local dt=$1
    local inp="dt_test_${dt}.inp"

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Running Δt = ${dt} fs  →  ${inp}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # MAPLE auto-generates: dt_test_0.5.inp → dt_test_0.5.out
    # and logger uses that stem: dt_test_0.5_md_thermo.dat etc.
    $MAPLE_CMD "$inp" 2>&1 | tail -5

    echo
}

run_and_report 0.5
run_and_report 0.25
run_and_report 0.125

# ─── 汇总比较 ─────────────────────────────────────────────────────────────
echo "════════════════════════════════════════"
echo "   Timestep Drift Comparison"
echo "════════════════════════════════════════"

python3 - <<'PY'
import numpy as np, re, sys
from pathlib import Path

Ha_to_kJ = 2625.4996
fs_to_ns  = 1e-6

results = []
for dt_str in ["0.5", "0.25", "0.125"]:
    fname = f"dt_test_{dt_str}_md_thermo.dat"
    p = Path(fname)
    if not p.exists():
        print(f"  {fname}: 文件不存在，跳过")
        continue

    rows = []
    with open(p) as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            try:
                rows.append([float(v) for v in s.split()])
            except ValueError:
                continue

    if not rows:
        continue

    data = np.array(rows)
    # columns: Step Time(fs) Temp KE PE TE
    time = data[:, 1]
    TE   = data[:, 5]

    coef       = np.polyfit(time, TE, 1)
    drift_rate = coef[0] * Ha_to_kJ / fs_to_ns   # kJ/mol/ns
    te_fluct   = TE.std() / abs(TE.mean())
    delta_E    = TE[-1] - TE[0]

    results.append((float(dt_str), len(rows), time[-1], delta_E, drift_rate, te_fluct))
    print(f"  Δt={dt_str:5s} fs | steps={len(rows):5d} | time={time[-1]:.1f} fs"
          f" | ΔE={delta_E:+.6f} Ha"
          f" | drift={drift_rate:.3g} kJ/mol/ns"
          f" | σ/|E|={te_fluct:.2e}")

if len(results) >= 2:
    print()
    print("  Drift scaling check (expect drift ∝ Δt²):")
    for i in range(1, len(results)):
        dt_ratio  = results[i-1][0] / results[i][0]
        dr_ratio  = abs(results[i-1][4]) / abs(results[i][4]) if results[i][4] != 0 else float('nan')
        expected  = dt_ratio**2
        print(f"    Δt ratio {dt_ratio:.1f}x → drift ratio {dr_ratio:.2f}x  (expected {expected:.1f}x for Δt² scaling)")
PY

echo "════════════════════════════════════════"
