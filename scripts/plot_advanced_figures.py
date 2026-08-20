"""Generate publication-quality 5-Class ROC and Specialized Confusion Matrices.

Usage:
    python scripts/plot_advanced_figures.py --run-dir outputs/effnet_b4_weighted_ce_5fold_5fold
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize

# 5类视网膜病变标准命名[cite: 7]
CLASS_NAMES_5 = ["No DR (0)", "Mild (1)", "Moderate (2)", "Severe (3)", "Proliferative (4)"]
# 剔除 No DR 后的 4类命名[cite: 7]
CLASS_NAMES_4 = ["Mild (1)", "Moderate (2)", "Severe (3)", "Proliferative (4)"]
# 二分类 (正常 vs 患病) 命名[cite: 7]
CLASS_NAMES_2 = ["No DR (Grade 0)", "DR (Grades 1-4)"]

# ROC 绘图定制配色 (5类不同颜色)[cite: 6, 7]
ROC_COLORS = ["#2c3e50", "#27ae60", "#2980b9", "#d35400", "#8e44ad"]


def plot_5class_roc(y_true: np.ndarray, y_prob: np.ndarray, save_dir: Path) -> None:
    """1. 绘制 5部分类 One-vs-Rest (OvR) ROC 曲线 (300 DPI + PNG/PDF)[cite: 6, 7]"""
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3, 4])
    y_prob = np.array(y_prob)

    fig, ax = plt.subplots(figsize=(7.5, 6.0), dpi=300)

    for i in range(5):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(
            fpr, tpr, color=ROC_COLORS[i], lw=2.2,
            label=f"{CLASS_NAMES_5[i]} (AUC = {roc_auc:.4f})"
        )

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.6, label="Random Guess (AUC = 0.5000)")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.04])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    ax.set_title("Multiclass ROC Curves on Unseen Domain (One-vs-Rest)", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=9, frameon=True)
    ax.grid(alpha=0.25, linestyle="--")
    fig.tight_layout()

    fig.savefig(save_dir / "roc_curve_multiclass.png", dpi=300, bbox_inches="tight")
    fig.savefig(save_dir / "roc_curve_multiclass.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ [1/4] 生成 5 分类 ROC 曲线 -> {save_dir / 'roc_curve_multiclass.png'}")


def plot_confusion_heatmap(
    cm: np.ndarray,
    class_names: list[str],
    title: str,
    save_dir: Path,
    file_prefix: str,
    figsize: tuple[int, int] = (7.5, 6.5)
) -> None:
    """通用学术混淆矩阵绘图函数：统一使用 Blues 渐变 + 双层数值比例标注 (300 DPI)"""
    cm = np.array(cm, dtype=float)
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1e-6)

    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    sns.heatmap(
        cm_norm, cmap="Blues", vmin=0, vmax=1, ax=ax,
        xticklabels=class_names, yticklabels=class_names,
        cbar_kws={"label": "Row-Normalized Ratio (Recall)"}
    )

    # 格内文字自适应：>0.55 为纯白，其余为黑色
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm_norm[i, j] > 0.55 else "black"
            text = f"{int(cm[i, j])}\n({cm_norm[i, j]*100:.1f}%)"
            ax.text(
                j + 0.5, i + 0.5, text,
                ha="center", va="center",
                color=color, fontsize=10, fontweight="bold"
            )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("True Severity Grade", fontsize=11)
    ax.set_xlabel("Predicted Severity Grade", fontsize=11)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=10)
    plt.setp(ax.get_yticklabels(), fontsize=10)
    fig.tight_layout()

    fig.savefig(save_dir / f"{file_prefix}.png", dpi=300, bbox_inches="tight")
    fig.savefig(save_dir / f"{file_prefix}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_cm_without_nodr(y_true: np.ndarray, y_pred: np.ndarray, save_dir: Path) -> None:
    """2. 去掉 No DR (0级) 的 4x4 混淆矩阵 (仅展示 1~4 级内部混淆)[cite: 7]"""
    cm_all = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])
    cm_no_dr = cm_all[1:, 1:]

    plot_confusion_heatmap(
        cm=cm_no_dr,
        class_names=CLASS_NAMES_4,
        title="Confusion Matrix without 'No DR' (Grades 1-4 Only)",
        save_dir=save_dir,
        file_prefix="confusion_matrix_no_dr",
        figsize=(7.0, 6.0)
    )
    print(f"✅ [2/4] 生成剔除 No DR 的 4×4 混淆矩阵 -> {save_dir / 'confusion_matrix_no_dr.png'}")


def plot_binary_dr_cm(y_true: np.ndarray, y_pred: np.ndarray, save_dir: Path) -> None:
    """3. DR vs No-DR 二分类 2x2 混淆矩阵 (0级 vs 1,2,3,4级)[cite: 7]"""
    y_true_bin = (y_true >= 1).astype(int)
    y_pred_bin = (y_pred >= 1).astype(int)
    cm_bin = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])

    plot_confusion_heatmap(
        cm=cm_bin,
        class_names=CLASS_NAMES_2,
        title="Binary Confusion Matrix (No DR vs. Retinopathy)",
        save_dir=save_dir,
        file_prefix="confusion_matrix_binary_dr",
        figsize=(6.0, 5.0)
    )
    print(f"✅ [3/4] 生成 DR vs. No-DR 二分类混淆矩阵 -> {save_dir / 'confusion_matrix_binary_dr.png'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate publication-quality 5-Class ROC and Specialized Confusion Matrices"
    )
    parser.add_argument(
        "--run-dir", type=str, required=True,
        help="Experiment output dir containing predictions_5fold.npz (or predictions.npz)"
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    fig_dir = run_dir / "figures_advanced"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. 兼容加载 5 折预测文件或单次训练预测文件[cite: 7]
    npz_path = run_dir / "predictions_5fold.npz"
    if not npz_path.exists():
        npz_path = run_dir / "predictions.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"找不到预测数据，请检查路径中是否有 predictions_5fold.npz: {run_dir}")

    data = np.load(npz_path)
    y_true = data["y_true"]
    y_prob = data["y_prob"]
    y_pred = np.argmax(y_prob, axis=1)

    print(f"📂 正在解析模型结果 -> {run_dir.name} (总测试集样本 N = {len(y_true)})")

    # 2. 生成 4 张精美学术图表 (统一 Blues 渐变 + 300 DPI + PNG/PDF)
    plot_5class_roc(y_true, y_prob, fig_dir)
    plot_cm_without_nodr(y_true, y_pred, fig_dir)
    plot_binary_dr_cm(y_true, y_pred, fig_dir)

    # ④ 完整 5 分类混淆矩阵[cite: 7]
    cm_full = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])
    plot_confusion_heatmap(
        cm=cm_full,
        class_names=CLASS_NAMES_5,
        title="Full 5-Class Retinopathy Grade Confusion Matrix",
        save_dir=fig_dir,
        file_prefix="confusion_matrix_5class",
        figsize=(7.5, 6.5)
    )
    print(f"✅ [4/4] 生成完整 5×5 混淆矩阵 -> {fig_dir / 'confusion_matrix_5class.png'}")

    print(f"\n🎉 所有图表已按同款出版级规范（300 DPI, PNG & PDF, All-Blues）生成完毕！\n   保存目录: {fig_dir}")


if __name__ == "__main__":
    main()