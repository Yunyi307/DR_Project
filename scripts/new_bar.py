import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def generate_updated_bar_chart():
    # 1. 确定保存路径 (outputs/report_figures)
    current_dir = Path(__file__).resolve().parent
    code_dir = current_dir.parent if current_dir.name == 'scripts' else current_dir
    save_dir = code_dir / "outputs" / "report_figures"
    save_dir.mkdir(parents=True, exist_ok=True)

    # 2. 准备数据 (已将 Referable AUC 统一更新为 0.9808)
    metrics = [
        'QWK\n(Ord. Kappa)',
        'Macro\nAccuracy',
        'Referable\nSensitivity',
        'Referable\nSpecificity',
        'Referable\nAUC'
    ]

    baseline_scores = [0.8549, 0.6364, 0.8430, 0.9480, 0.9723]
    ensemble_scores = [0.8924, 0.6451, 0.9327, 0.9358, 0.9808]  # 更新 0.9788 -> 0.9808

    x = np.arange(len(metrics))
    width = 0.35

    # 3. 绘图
    plt.figure(figsize=(10, 6), dpi=300, facecolor='white')

    rects1 = plt.bar(x - width / 2, baseline_scores, width, label='Single Model Baseline (CE)', color='#2b5c8f')
    rects2 = plt.bar(x + width / 2, ensemble_scores, width, label='5-Fold Soft Voting Ensemble', color='#d95f02')

    plt.ylabel('Score (0.0 - 1.0)', fontsize=12, fontweight='bold')
    plt.title('Comprehensive Metric Comparison: Baseline vs. 5-Fold Ensemble', fontsize=14, fontweight='bold', pad=15)
    plt.xticks(x, metrics, fontsize=11)
    plt.ylim(0, 1.15)
    plt.legend(loc='upper left', fontsize=11, framealpha=0.95)
    plt.grid(axis='y', linestyle='--', alpha=0.4)

    # 在柱状图上方标数值
    def autolabel(rects, is_bold=False):
        for rect in rects:
            height = rect.get_height()
            weight = 'bold' if is_bold else 'normal'
            plt.annotate(f'{height:.4f}',
                         xy=(rect.get_x() + rect.get_width() / 2, height),
                         xytext=(0, 4),  # 4pts vertical offset
                         textcoords="offset points",
                         ha='center', va='bottom',
                         fontsize=10.5, fontweight=weight)

    autolabel(rects1, is_bold=False)
    autolabel(rects2, is_bold=True)

    plt.tight_layout()

    # 4. 保存图片
    output_path = save_dir / "fig2_overall_metrics_comparison.png"
    plt.savefig(output_path, dpi=300, facecolor='white')
    print(f"✅ 柱状图已更新并保存至:\n{output_path.resolve()}")


if __name__ == '__main__':
    generate_updated_bar_chart()