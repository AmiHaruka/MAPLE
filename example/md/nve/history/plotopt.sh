#!/bin/bash
# 用法: ./plot.sh <输出文件>

if [ $# -ne 1 ]; then
    echo "错误：请指定输出文件作为参数。"
    exit 1
fi

input_file="$1"

if [ ! -f "$input_file" ]; then
    echo "错误：文件 '$input_file' 不存在。"
    exit 1
fi

# 将文件内容作为参数传递给 Python 脚本
python3 - "$input_file" <<'EOF'
import re
import matplotlib.pyplot as plt
import sys

filename = sys.argv[1]

steps = []
energies = []
step_counter = 0
energy_pattern = re.compile(r'Energy\s*=\s*([-\d.]+)')
step_pattern = re.compile(r'(?:Image|Iteration)\s+(\d+)')

with open(filename, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        energy_match = energy_pattern.search(line)
        if energy_match:
            energy = float(energy_match.group(1))
            step_match = step_pattern.search(line)
            if step_match:
                step = int(step_match.group(1))
            else:
                step = step_counter
                step_counter += 1
            steps.append(step)
            energies.append(energy)

if not steps:
    print("错误：文件中未找到能量信息。")
    sys.exit(1)

print(f"找到 {len(steps)} 个优化步，能量范围：{min(energies):.8f} ～ {max(energies):.8f}")

plt.figure(figsize=(8, 5))
plt.plot(steps, energies, marker='o', linestyle='-', color='b')
plt.xlabel('Optimization Step')          # 英文标签
plt.ylabel('Total Energy (Hartree)')      # 英文标签
plt.title('Energy Convergence Curve')      # 英文标签
plt.grid(True)
plt.tight_layout()
plt.show()
EOF