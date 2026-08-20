import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def generate_referable_roc_comparison():
    # 1. 定位项目根目录
    current_dir = Path(__file__).resolve().parent
    code_dir = current_dir.parent if current_dir.name == 'scripts' else current_dir
    outputs_dir = code_dir / "outputs"

    # 2. 读取预测文件（准确指向 predictions_5fold.npz）
    baseline_path = outputs_dir / "effnet_b4_baseline" / "predictions.npz"
    ensemble_path = outputs_dir / "effnet_b4_baseline_5fold" / "predictions_5fold.npz"
    save_dir = outputs_dir / "report_figures"

    # 自动兼容 effnet_b4_ce 路径
    if not baseline_path.exists():
        baseline_path = outputs_dir / "effnet_b4_ce" / "predictions.npz"

    print(f"🔍 Baseline 文件: {baseline_path}")
    print(f"🔍 5-Fold 集成文件: {ensemble_path}")

    # 3. 加载预测矩阵
    data_base = np.load(baseline_path)
    y_true = data_base['y_true']
    y_prob_base = data_base['y_prob']

    data_ens = np.load(ensemble_path)
    y_prob_ens = data_ens['y_prob']

    # 4. 构建 Referable DR 二分类 (Grade >= 2 为 1，Grade < 2 为 0)
    y_true_ref = (y_true >= 2).astype(int)
    prob_ref_base = y_prob_base[:, 2:].sum(axis=1)
    prob_ref_ens = y_prob_ens[:, 2:].sum(axis=1)

    # 5. 计算 ROC 曲线与 AUC
    fpr_base, tpr_base, _ = roc_curve(y_true_ref, prob_ref_base)
    auc_base = auc(fpr_base, tpr_base)

    fpr_ens, tpr_ens, _ = roc_curve(y_true_ref, prob_ref_ens)
    auc_ens = auc(fpr_ens, tpr_ens)

    # 6. 绘图
    plt.figure(figsize=(8, 6.5), dpi=300, facecolor='white')
    plt.plot(fpr_base, tpr_base, color='#1f77b4', lw=2.5,
             label=f'Baseline (CE) (AUC = {auc_base:.4f})')
    plt.plot(fpr_ens, tpr_ens, color='#d95f02', lw=2.5,
             label=f'5-Fold Ensemble (AUC = {auc_ens:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.5, label='Random Guess')

    plt.xlim([-0.01, 1.0])
    plt.ylim([-0.01, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12, fontweight='bold')
    plt.title('Referable DR ROC Curve Comparison (Grades 0-1 vs. 2-4)', fontsize=13, fontweight='bold', pad=12)

    plt.legend(loc="lower right", fontsize=11, framealpha=0.95)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()

    # 7. 保存图片
    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / "fig4_roc_referable_comparison.png"
    plt.savefig(output_path, dpi=300, facecolor='white')

    print(f"✅ 图片已成功生成并保存至:\n{output_path.resolve()}")


if __name__ == '__main__':
    generate_referable_roc_comparison()