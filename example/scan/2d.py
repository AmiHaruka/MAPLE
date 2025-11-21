#!/usr/bin/env python3
"""
提取格点扫描能量数据并绘制高质量等高线图
从Hartree转换为kcal/mol
增强版：支持多种平滑方法
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import griddata, RBFInterpolator
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

# 转换因子: 1 Hartree = 627.509474 kcal/mol
HARTREE_TO_KCAL = 627.509474

def extract_scan_data(filename):
    """
    从输出文件中提取格点扫描的坐标和能量数据
    
    参数:
        filename: 输出文件路径
    
    返回:
        data: 包含格点坐标和能量的列表
    """
    data = []
    
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 使用正则表达式匹配每个扫描点的信息
    pattern = r'Scanning combination \d+/\d+: \[([\d.]+), ([\d.]+)\].*?Energy:\s+([-\d.]+)'
    
    matches = re.findall(pattern, content, re.DOTALL)
    
    for match in matches:
        coord1 = float(match[0])
        coord2 = float(match[1])
        energy_hartree = float(match[2])
        energy_kcal = energy_hartree * HARTREE_TO_KCAL
        
        data.append({
            'coord1': coord1,
            'coord2': coord2,
            'energy_hartree': energy_hartree,
            'energy_kcal': energy_kcal
        })
    
    return data

def apply_smoothing(grid_energy, method='gaussian', sigma=1.0):
    """
    对网格能量数据应用平滑
    
    参数:
        grid_energy: 2D能量网格
        method: 平滑方法 ('gaussian', 'none')
        sigma: 高斯平滑参数（值越大越平滑）
    
    返回:
        平滑后的能量网格
    """
    if method == 'none':
        return grid_energy
    
    elif method == 'gaussian':
        # 高斯滤波平滑
        return gaussian_filter(grid_energy, sigma=sigma, mode='nearest')
    
    else:
        return grid_energy

def plot_contour(data, output_file='energy_contour.png', 
                method='linear', smooth='gaussian', sigma=1.5):
    """
    绘制高质量能量等高线图
    
    参数:
        data: 提取的数据列表
        output_file: 输出图片文件名
        method: 插值方法 ('linear', 'cubic', 'rbf', 'nearest')
        smooth: 平滑方法 ('gaussian', 'none')
        sigma: 高斯平滑强度 (0.5-3.0, 推荐1.0-2.0)
    """
    # 提取坐标和能量
    coord1 = np.array([d['coord1'] for d in data])
    coord2 = np.array([d['coord2'] for d in data])
    energies = np.array([d['energy_kcal'] for d in data])
    
    # 转换为相对能量 (kcal/mol)
    min_energy = energies.min()
    relative_energies = energies - min_energy
    
    print(f"\n绘图参数:")
    print(f"  插值方法: {method}")
    print(f"  平滑方法: {smooth}")
    if smooth == 'gaussian':
        print(f"  平滑强度: {sigma}")
    print(f"\n数据统计:")
    print(f"  Coord1 范围: [{coord1.min():.2f}, {coord1.max():.2f}]")
    print(f"  Coord2 范围: [{coord2.min():.2f}, {coord2.max():.2f}]")
    print(f"  能量范围: [0.00, {relative_energies.max():.2f}] kcal/mol")
    print(f"  数据点数: {len(data)}")
    
    # 检查数据是否规则网格
    coord1_unique = np.unique(coord1)
    coord2_unique = np.unique(coord2)
    is_regular_grid = len(data) == len(coord1_unique) * len(coord2_unique)
    
    if is_regular_grid:
        print(f"  网格类型: 规则网格 {len(coord1_unique)} x {len(coord2_unique)}")
    
    # 创建网格用于插值
    n_points = 300  # 高密度网格
    grid_coord1, grid_coord2 = np.meshgrid(
        np.linspace(coord1.min(), coord1.max(), n_points),
        np.linspace(coord2.min(), coord2.max(), n_points)
    )
    
    # 根据选择的方法进行插值
    if method == 'rbf':
        # 使用径向基函数插值
        try:
            rbf = RBFInterpolator(
                np.column_stack([coord1, coord2]), 
                relative_energies,
                kernel='thin_plate_spline',
                smoothing=0.0
            )
            grid_energy = rbf(np.column_stack([grid_coord1.ravel(), grid_coord2.ravel()]))
            grid_energy = grid_energy.reshape(grid_coord1.shape)
        except:
            print("  警告: RBF插值失败，切换到linear方法")
            method = 'linear'
    
    if method in ['linear', 'cubic', 'nearest']:
        # 使用scipy的griddata
        grid_energy = griddata(
            (coord1, coord2), 
            relative_energies, 
            (grid_coord1, grid_coord2), 
            method=method,
            fill_value=np.nan  # 使用NaN标记外推区域
        )
        
        # 处理NaN值（如果有的话）
        if np.any(np.isnan(grid_energy)):
            # 用最近邻填充NaN
            mask = np.isnan(grid_energy)
            grid_energy[mask] = griddata(
                (coord1, coord2), 
                relative_energies, 
                (grid_coord1[mask], grid_coord2[mask]), 
                method='nearest'
            )
    
    # 应用平滑
    if smooth != 'none':
        grid_energy = apply_smoothing(grid_energy, method=smooth, sigma=sigma)
        print(f"  已应用{smooth}平滑")
    
    # 设置绘图样式
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 绘制填充等高线 - 使用viridis配色
    n_levels = 15  # 减少等高线密度，让图更清晰
    levels = np.linspace(np.nanmin(grid_energy), np.nanmax(grid_energy), n_levels)
    contourf = ax.contourf(grid_coord1, grid_coord2, grid_energy, 
                           levels=levels, cmap='viridis', extend='both')
    
    # 添加颜色条
    cbar = plt.colorbar(contourf, ax=ax)
    cbar.ax.tick_params(labelsize=11)
    
    # 设置坐标轴标签
    ax.set_xlabel('X-axis', fontsize=13)
    ax.set_ylabel('Y-axis', fontsize=13)
    
    # 构建标题
    title = 'Energy Contour Plot'
    if smooth != 'none':
        title += f' ({smooth.capitalize()} Smoothed)'
    ax.set_title(title, fontsize=14)
    
    # 设置刻度
    ax.tick_params(labelsize=11)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"\n等高线图已保存到 {output_file}")
    
    # 也保存为PDF格式（矢量图）
    pdf_file = output_file.rsplit('.', 1)[0] + '.pdf'
    plt.savefig(pdf_file, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"PDF格式已保存到 {pdf_file}")
    
    plt.close()

def plot_scatter(data, output_file='energy_scatter.png'):
    """
    绘制原始数据点的散点图
    """
    coord1 = np.array([d['coord1'] for d in data])
    coord2 = np.array([d['coord2'] for d in data])
    energies = np.array([d['energy_kcal'] for d in data])
    relative_energies = energies - energies.min()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    scatter = ax.scatter(coord1, coord2, c=relative_energies, 
                        s=100, cmap='viridis', edgecolors='black', linewidths=1)
    
    # 在每个点上标注能量值
    for i in range(len(data)):
        ax.text(coord1[i], coord2[i], f'{relative_energies[i]:.1f}', 
                fontsize=7, ha='center', va='center', color='white',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5))
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Relative Energy (kcal/mol)', fontsize=11)
    
    ax.set_xlabel('X-axis', fontsize=13)
    ax.set_ylabel('Y-axis', fontsize=13)
    ax.set_title('Original Data Points', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"散点图已保存到 {output_file}")
    plt.close()

def save_results(data, output_file='scan_energies.txt'):
    """
    保存提取的数据到文件
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"{'Coord1':>10s} {'Coord2':>10s} {'Energy(Hartree)':>20s} {'Energy(kcal/mol)':>20s} {'Relative(kcal/mol)':>20s}\n")
        f.write("-" * 90 + "\n")
        
        # 找到最低能量作为参考
        min_energy_kcal = min(d['energy_kcal'] for d in data)
        
        for d in data:
            relative_energy = d['energy_kcal'] - min_energy_kcal
            f.write(f"{d['coord1']:>10.2f} {d['coord2']:>10.2f} "
                   f"{d['energy_hartree']:>20.6f} {d['energy_kcal']:>20.6f} "
                   f"{relative_energy:>20.6f}\n")
    
    print(f"数据文件已保存到 {output_file}")

def print_summary(data):
    """
    打印能量数据的统计摘要
    """
    if not data:
        print("未找到数据")
        return
    
    energies_kcal = [d['energy_kcal'] for d in data]
    min_energy = min(energies_kcal)
    max_energy = max(energies_kcal)
    
    # 找到最低能量点
    min_point = min(data, key=lambda x: x['energy_kcal'])
    
    print("\n" + "=" * 60)
    print("格点扫描能量数据统计")
    print("=" * 60)
    print(f"总扫描点数: {len(data)}")
    print(f"最低能量: {min_energy:.6f} kcal/mol")
    print(f"  对应坐标: [{min_point['coord1']:.2f}, {min_point['coord2']:.2f}]")
    print(f"  (Hartree: {min_point['energy_hartree']:.6f})")
    print(f"最高能量: {max_energy:.6f} kcal/mol")
    print(f"能量范围: {max_energy - min_energy:.6f} kcal/mol")
    print("=" * 60)

def main():
    """主函数"""
    import sys
    
    if len(sys.argv) < 2:
        print("=" * 70)
        print("格点扫描能量等高线图绘制工具 (增强版)")
        print("=" * 70)
        print("\n用法:")
        print("  python script.py <文件> [插值方法] [平滑方法] [平滑强度]")
        print("\n示例:")
        print("  python script.py data.txt linear gaussian 1.5  # 推荐")
        print("  python script.py data.txt rbf gaussian 1.0")
        print("  python script.py data.txt linear none          # 不平滑")
        print("\n参数说明:")
        print("\n  [插值方法] - 可选:")
        print("    linear  - 线性插值 (默认，推荐)")
        print("    rbf     - 径向基函数 (最平滑)")
        print("    cubic   - 三次插值 (可能振荡)")
        print("    nearest - 最近邻插值")
        print("\n  [平滑方法] - 可选:")
        print("    gaussian - 高斯平滑 (默认，推荐)")
        print("    none     - 不平滑")
        print("\n  [平滑强度] - 可选 (仅gaussian时有效):")
        print("    0.5-1.0  - 轻微平滑")
        print("    1.0-2.0  - 中等平滑 (推荐)")
        print("    2.0-3.0  - 强平滑")
        print("    默认: 1.5")
        print("\n💡 提示:")
        print("  - 如果图像有噪声，增大平滑强度")
        print("  - 如果需要保留更多细节，减小平滑强度")
        print("  - 先用 linear + gaussian 1.5 试试看效果")
        print("=" * 70)
        sys.exit(1)
    
    input_file = sys.argv[1]
    method = sys.argv[2] if len(sys.argv) > 2 else 'linear'
    smooth = sys.argv[3] if len(sys.argv) > 3 else 'gaussian'
    sigma = float(sys.argv[4]) if len(sys.argv) > 4 else 1.5
    
    if method not in ['linear', 'cubic', 'rbf', 'nearest']:
        print(f"错误: 未知的插值方法 '{method}'")
        print("可用方法: linear, cubic, rbf, nearest")
        sys.exit(1)
    
    if smooth not in ['gaussian', 'none']:
        print(f"错误: 未知的平滑方法 '{smooth}'")
        print("可用方法: gaussian, none")
        sys.exit(1)
    
    print(f"正在读取文件: {input_file}")
    
    # 提取数据
    data = extract_scan_data(input_file)
    
    if not data:
        print("错误: 未能提取到数据")
        sys.exit(1)
    
    # 打印统计信息
    print_summary(data)
    
    # 保存文本结果
    output_file = input_file.rsplit('.', 1)[0] + '_energies.txt'
    save_results(data, output_file)
    
    # 绘制原始数据散点图
    scatter_file = input_file.rsplit('.', 1)[0] + '_scatter.png'
    plot_scatter(data, scatter_file)
    
    # 绘制等高线图
    suffix = f'{method}_{smooth}'
    if smooth == 'gaussian':
        suffix += f'_{sigma:.1f}'
    plot_file = input_file.rsplit('.', 1)[0] + f'_contour_{suffix}.png'
    plot_contour(data, plot_file, method=method, smooth=smooth, sigma=sigma)
    
    print("\n✅ 完成!")

if __name__ == "__main__":
    main()