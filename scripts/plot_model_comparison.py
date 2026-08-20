"""Generate publication-quality comparative figures: Baseline vs. 5-Fold Ensemble.

Usage:
    python scripts/plot_model_comparison.py \
        --dir-baseline outputs/effnet_b4_baseline \
        --dir-5fold outputs/effnet_b4_weighted_ce_5fold_5fold \
        --out-dir outputs/comparison_baseline_vs_5fold
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import auc, cohen_kappa_score, confusion_matrix, roc_curve

# 5类视网膜病变标准命名
CLASS_NAMES = ["No DR (0)", "Mild (1)", "Moderate (2)", "Severe (3)", "Proliferative (4)"]

# 学术定制主题颜色
COLOR_BASE = "#2980b9"   # 深海蓝：代表 Single Model Baseline (CE)
COLOR_5FOLD = "#d35400"  # 警示红橙：代表 5-Fold Soft Voting Ensemble


def load_prediction_data(run_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """兼容加载 predictions.npz 或 predictions_5fold.npz"""
    npz_paths = [
        run_dir / "predictions_5fold.npz",
        run_dir / "predictions.npz",
    ]
    for p in npz_paths:
        if p.exists():
            data = np.load(p)
            y_true = data["y_true"]
            y_prob = data["y_prob"]
            y_pred = data["y_pred"] if "y_pred" in data else np.argmax(y_prob, axis=1)
            return y_true, y_prob, y_pred

    raise FileNotFoundError(f"在 {run_dir} 下未找到预测数组（predictions*.npz），请先执行评估推断。")


def calculate_metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> dict:
    """自动测算各类学术评价指标"""
    # 1. QWK (二次加权卡佩系数)
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")

    # 2. Macro Accuracy (宏准确率 / 五分类平均对角线召回率)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    recalls = cm.diagonal() / np.maximum(cm.sum(axis=1), 1e-6)
    macro_acc = np.mean(recalls)

    # 3. Referable DR 二分类 (0,1 级 vs 2,3,4 级)
    y_true_bin = (y_true >= 2).astype(int)
    y_prob_ref = np.sum(y_prob[:, 2:], axis=1)
    y_pred_bin = (y_pred >= 2).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()
    ref_sens = tp / max(tp + fn, 1e-6)
    ref_spec = tn / max(tn + fp, 1e-6)

    fpr, tpr, _ = roc_curve(y_true_bin, y_prob_ref)
    ref_auc = auc(fpr, tpr)

    return {
        "qwk": qwk,
        "macro_acc": macro_acc,
        "ref_sens": ref_sens,
        "ref_spec": ref_spec,
        "ref_auc": ref_auc,
        "per_class_recalls": recalls,
        "cm": cm,
        "fpr": fpr,
        "tpr": tpr,
    }


def plot_cm_side_by_side(res_base: dict, res_5fold: dict, save_dir: Path):
    """图 1：左右混淆矩阵对比图 (Side-by-Side Confusion Matrix)"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), dpi=300)

    for idx, (ax, res, title) in enumerate([
        (axes[0], res_base, "Single Model Baseline (CE)"),
        (axes[1], res_5fold, "5-Fold Soft Voting Ensemble")
    ]):
        cm = res["cm"]
        cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1e-6)

        sns.heatmap(
            cm_norm, cmap="Blues", vmin=0, vmax=1, ax=ax,
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            cbar=(idx == 1),
            cbar_kws={"label": "Row-Normalized Ratio (Recall)"} if idx == 1 else {}
        )

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = "white" if cm_norm[i, j] > 0.55 else "black"
                text = f"{int(cm[i, j])}\n({cm_norm[i, j]*100:.1f}%)"
                ax.text(j + 0.5, i + 0.5, text, ha="center", va="center",
                        color=color, fontsize=10, fontweight="bold")

        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax.set_ylabel("True Severity Grade" if idx == 0 else "", fontsize=11)
        ax.set_xlabel("Predicted Severity Grade", fontsize=11)
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=10)
        plt.setp(ax.get_yticklabels(), fontsize=10)

    fig.suptitle("Comparison of Confusion Matrices: Baseline vs. 5-Fold Ensemble",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_dir / "fig1_cm_side_by_side.png", dpi=300, bbox_inches="tight")
    fig.savefig(save_dir / "fig1_cm_side_by_side.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_overall_metrics(res_base: dict, res_5fold: dict, save_dir: Path):
    """图 2：核心评价指标分组柱状图 (Grouped Bar Chart - Overall Metrics)"""
    labels = ["QWK\n(Ord. Kappa)", "Macro\nAccuracy", "Referable\nSensitivity", "Referable\nSpecificity", "Referable\nAUC"]
    base_vals = [res_base["qwk"], res_base["macro_acc"], res_base["ref_sens"], res_base["ref_spec"], res_base["ref_auc"]]
    fold5_vals = [res_5fold["qwk"], res_5fold["macro_acc"], res_5fold["ref_sens"], res_5fold["ref_spec"], res_5fold["ref_auc"]]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    rects1 = ax.bar(x - width/2, base_vals, width, label="Single Model Baseline (CE)", color=COLOR_BASE)
    rects2 = ax.bar(x + width/2, fold5_vals, width, label="5-Fold Soft Voting Ensemble", color=COLOR_5FOLD)

    ax.set_ylabel("Score (0.0 - 1.0)", fontsize=12)
    ax.set_title("Comprehensive Metric Comparison: Baseline vs. 5-Fold Ensemble", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim([0.0, 1.15])
    ax.legend(loc="upper left", fontsize=11, frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    ax.bar_label(rects1, fmt="%.4f", padding=3, fontsize=9.5)
    ax.bar_label(rects2, fmt="%.4f", padding=3, fontsize=9.5, fontweight="bold")

    fig.tight_layout()
    fig.savefig(save_dir / "fig2_overall_metrics_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(save_dir / "fig2_overall_metrics_comparison.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_sensitivity(res_base: dict, res_5fold: dict, save_dir: Path):
    """图 3：0~4 级各类别灵敏度对比图 (Per-Class Sensitivity Comparison)"""
    x = np.arange(len(CLASS_NAMES))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9.5, 5.5), dpi=300)
    rects1 = ax.bar(x - width/2, res_base["per_class_recalls"], width, label="Single Model Baseline (CE)", color=COLOR_BASE)
    rects2 = ax.bar(x + width/2, res_5fold["per_class_recalls"], width, label="5-Fold Soft Voting Ensemble", color=COLOR_5FOLD)

    ax.set_ylabel("Sensitivity (Recall)", fontsize=12)
    ax.set_title("Per-Class Sensitivity (Recall) Comparison by Retinopathy Grade", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, fontsize=11, rotation=10)
    ax.set_ylim([0.0, 1.15])
    ax.legend(loc="upper right", fontsize=11, frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    ax.bar_label(rects1, fmt="%.3f", padding=3, fontsize=9.5)
    ax.bar_label(rects2, fmt="%.3f", padding=3, fontsize=9.5, fontweight="bold")

    fig.tight_layout()
    fig.savefig(save_dir / "fig3_per_class_sensitivity_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(save_dir / "fig3_per_class_sensitivity_comparison.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_roc_referable_comparison(res_base: dict, res_5fold: dict, save_dir: Path):
    """图 4：转诊 DR 二分类 ROC 曲线叠加对比图 (Overlay Referable DR ROC)"""
    fig, ax = plt.subplots(figsize=(7.0, 6.0), dpi=300)

    ax.plot(
        res_base["fpr"], res_base["tpr"], color=COLOR_BASE, lw=2.2,
        label=f"Baseline (CE) (AUC = {res_base['ref_auc']:.4f})"
    )
    ax.plot(
        res_5fold["fpr"], res_5fold["tpr"], color=COLOR_5FOLD, lw=2.5,
        label=f"5-Fold Ensemble (AUC = {res_5fold['ref_auc']:.4f})"
    )
    ax.plot([0, 1], [0, 1], color="#7f8c8d", lw=1.5, linestyle="--", label="Random Guess")

    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.04])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    ax.set_title("Referable DR ROC Curve Comparison (Grades 0-1 vs. 2-4)", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=10.5, frameon=True)
    ax.grid(alpha=0.25, linestyle="--")

    fig.tight_layout()
    fig.savefig(save_dir / "fig4_roc_referable_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(save_dir / "fig4_roc_referable_comparison.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate publication-quality comparative figures between Baseline and 5-Fold.")
    parser.add_argument("--dir-baseline", type=str, required=True, help="Path to single Baseline model output dir")
    parser.add_argument("--dir-5fold", type=str, required=True, help="Path to 5-Fold Ensemble output dir")
    parser.add_argument("--out-dir", type=str, default="outputs/comparison_baseline_vs_5fold", help="Directory to save generated plots")
    args = parser.parse_args()

    dir_base = Path(args.dir_baseline)
    dir_5fold = Path(args.dir_5fold)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== 正在加载实验预测数据 ===")
    print(f"1. Single Baseline (CE) -> {dir_base}")
    y_true_b, y_prob_b, y_pred_b = load_prediction_data(dir_base)
    res_base = calculate_metrics(y_true_b, y_prob_b, y_pred_b)

    print(f"2. 5-Fold Ensemble -> {dir_5fold}")
    y_true_5, y_prob_5, y_pred_5 = load_prediction_data(dir_5fold)
    res_5fold = calculate_metrics(y_true_5, y_prob_5, y_pred_5)

    print("\n=== 正在生成 300 DPI 论文级对比图表 ===")
    plot_cm_side_by_side(res_base, res_5fold, out_dir)
    print("✅ [1/4] 生成左右混淆矩阵对比图 -> fig1_cm_side_by_side.png / .pdf")

    plot_overall_metrics(res_base, res_5fold, out_dir)
    print("✅ [2/4] 生成综合核心指标分组柱状图 -> fig2_overall_metrics_comparison.png / .pdf")

    plot_per_class_sensitivity(res_base, res_5fold, out_dir)
    print("✅ [3/4] 生成各等级类别灵敏度对比图 -> fig3_per_class_sensitivity_comparison.png / .pdf")

    plot_roc_referable_comparison(res_base, res_5fold, out_dir)
    print("✅ [4/4] 生成转诊 DR 二分类 ROC 曲线叠加图 -> fig4_roc_referable_comparison.png / .pdf")

    print(f"\n🎉 所有对比图表已成功生成！保存目录: {out_dir.resolve()}")


if __name__ == "__main__":
    main()