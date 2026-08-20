import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def generate_aptos_class_distribution():
    # 1. 自动定位项目根目录与 outputs/report_figures 保存路径
    current_dir = Path(__file__).resolve().parent
    code_dir = current_dir.parent if current_dir.name == 'scripts' else current_dir
    save_dir = code_dir / "outputs" / "report_figures"
    save_dir.mkdir(parents=True, exist_ok=True)

    # 2. 准备 APTOS 2019 训练集数据
    grades = ['Grade 0\n(No DR)', 'Grade 1\n(Mild)', 'Grade 2\n(Moderate)', 'Grade 3\n(Severe)',
              'Grade 4\n(Proliferative)']
    counts = [1805, 370, 999, 193, 295]  # APTOS 2019 训练集样本数
    total = sum(counts)
    percentages = [(c / total) * 100 for c in counts]

    # 3. 绘制高清柱状图
    plt.figure(figsize=(8.5, 5.5), dpi=300, facecolor='white')

    # 使用学术深蓝配色
    bars = plt.bar(grades, percentages, color='#1f77b4', width=0.52, edgecolor='#134b73', linewidth=1.0)

    # 坐标轴与标题设置
    plt.ylabel('Share of Images (%)', fontsize=12, fontweight='bold')
    plt.xlabel('Diabetic Retinopathy Grade', fontsize=12, fontweight='bold')
    plt.title('Class Distribution of APTOS 2019 Dataset', fontsize=14, fontweight='bold', pad=15)
    plt.ylim(0, 60)
    plt.grid(axis='y', linestyle='--', alpha=0.4)

    # 4. 在每根柱子上方标出：百分比与具体样本数 (如: 49.3% \n (1805))
    for bar, count, pct in zip(bars, counts, percentages):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 1.2,
                 f'{pct:.1f}%\n({count})',
                 ha='center', va='bottom', fontsize=10, fontweight='bold', color='#1f77b4')

    plt.tight_layout()

    # 5. 保存图片至 report_figures
    output_path = save_dir / "class_distribution.png"
    plt.savefig(output_path, dpi=300, facecolor='white')
    print(f"✅ 单数据集 (APTOS) 类别分布图已成功保存至:\n{output_path.resolve()}")


if __name__ == '__main__':
    generate_aptos_class_distribution()