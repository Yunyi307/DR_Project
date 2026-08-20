"""Generate publication-ready figures for External Dataset (e.g. IDRiD) evaluation.

Supports switching between 'expected_score' and 'argmax' directly in plotting,
ensuring QWK, Referable AUC, Balanced Accuracy, and Confusion Matrix are 100% accurate.

Usage:
    python scripts/make_external_figures.py --run-dir outputs/effnet_b4_weighted_ce_5fold_5fold --dataset IDRiD --pred-method expected_score --thresholds 0.45 1.35 2.25 3.15
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    auc,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

CLASS_NAMES = ["No DR (0)", "Mild (1)", "Moderate (2)", "Severe (3)", "Proliferative (4)"]
COLORS = ["#2b5c8f", "#d95f02", "#7570b3", "#e7298a", "#66a61e"]


def get_predictions(y_prob: np.ndarray, pred_method: str, thresholds: list[float]) -> np.ndarray:
    """按选定策略计算预测级别 (0-4)"""
    if pred_method == "expected_score":
        weights = np.arange(y_prob.shape[1], dtype=np.float32)
        scores = np.sum(y_prob * weights, axis=1)
        return np.digitize(scores, thresholds).astype(int)
    return np.argmax(y_prob, axis=1)


def plot_multiclass_roc(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path, dataset_name: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    # 左图：二分类转诊 DR ROC (Referable DR: grade >= 2)
    y_true_ref = (y_true >= 2).astype(int)
    y_prob_ref = np.sum(y_prob[:, 2:], axis=1)
    fpr_ref, tpr_ref, _ = roc_curve(y_true_ref, y_prob_ref)
    auc_ref = auc(fpr_ref, tpr_ref)

    ax1.plot(fpr_ref, tpr_ref, color="#d95f02", lw=2.5, label=f"Referable DR (AUC = {auc_ref:.4f})")
    ax1.plot([0, 1], [0, 1], color="#7f8c8d", lw=1.5, linestyle="--")
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    ax1.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    ax1.set_title(f"Referable DR ROC Curve ({dataset_name})", fontsize=13, fontweight="bold")
    ax1.legend(loc="lower right", fontsize=11)
    ax1.grid(alpha=0.25)

    # 右图：5分类 One-vs-Rest ROC
    for i, name in enumerate(CLASS_NAMES):
        y_true_binary = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(y_true_binary, y_prob[:, i])
        class_auc = auc(fpr, tpr)
        ax2.plot(fpr, tpr, color=COLORS[i], lw=2, label=f"{name} (AUC = {class_auc:.3f})")

    ax2.plot([0, 1], [0, 1], color="#7f8c8d", lw=1.5, linestyle="--")
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel("False Positive Rate", fontsize=12)
    ax2.set_ylabel("True Positive Rate", fontsize=12)
    ax2.set_title(f"Per-Class (One-vs-Rest) ROC Curves ({dataset_name})", fontsize=13, fontweight="bold")
    ax2.legend(loc="lower right", fontsize=10)
    ax2.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_sensitivity_specificity(sens: dict, spec: dict, out_path: Path, dataset_name: str):
    names = [k.split(" ")[0] for k in sens.keys()]
    y_sens = [sens[k] for k in sens.keys()]
    y_spec = [spec[k] for k in spec.keys()]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    rects1 = ax.bar(x - width/2, y_sens, width, label="Sensitivity (Recall)", color="#2b5c8f", alpha=0.9)
    rects2 = ax.bar(x + width/2, y_spec, width, label="Specificity (True Negative Rate)", color="#2ca02c", alpha=0.9)

    ax.set_ylabel("Score (0.0 - 1.0)", fontsize=12)
    ax.set_title(f"Per-Class Sensitivity and Specificity on {dataset_name}", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, fontsize=11)
    ax.set_ylim([0.0, 1.15])
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.2f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, out_path: Path, dataset_name: str):
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    cax = ax.matshow(cm_norm, cmap=plt.cm.Blues, alpha=0.85)
    fig.colorbar(cax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(CLASS_NAMES, rotation=25, ha="left", fontsize=10)
    ax.set_yticklabels(CLASS_NAMES, fontsize=10)
    ax.set_xlabel("Predicted Grade", fontsize=12, labelpad=10)
    ax.set_ylabel("True Grade", fontsize=12)
    ax.set_title(f"Confusion Matrix ({dataset_name} External Test)", fontsize=14, fontweight="bold", pad=20)

    for i in range(5):
        for j in range(5):
            pct = cm_norm[i, j]
            count = cm[i, j]
            color = "white" if pct > 0.55 else "black"
            text = f"{pct*100:.1f}%\n({count})"
            ax.text(j, i, text, va="center", ha="center", color=color, fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_overall_summary_card(metrics: dict, out_path: Path, dataset_name: str):
    labels = [
        "QWK\n(Ord. Kappa)",
        "Referable\nAUC",
        "Referable\nSensitivity",
        "Referable\nSpecificity",
        "Balanced\nAccuracy",
        "Macro\nAccuracy"
    ]
    scores = [
        metrics["qwk"],
        metrics["referable_auc"],
        metrics["referable_sensitivity"],
        metrics["referable_specificity"],
        metrics["balanced_accuracy"],
        metrics["macro_accuracy"]
    ]
    bar_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#8c564b", "#9467bd", "#7f7f7f"]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    bars = ax.bar(labels, scores, color=bar_colors, width=0.5, alpha=0.9)

    ax.set_ylim([0.0, 1.15])
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"Comprehensive Metric Summary on {dataset_name} (5-Fold Ensemble)", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.4f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 5), textcoords="offset points",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate publication-ready DR grading plots.")
    parser.add_argument("--run-dir", type=str, required=True,
                        help="Path to trained output folder")
    parser.add_argument("--dataset", type=str, default="IDRiD")
    parser.add_argument("--pred-method", type=str, default="expected_score", choices=["expected_score", "argmax"])
    parser.add_argument("--thresholds", type=float, nargs=4, default=[0.45, 1.35, 2.25, 3.15])
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    dataset_name = args.dataset.upper()

    # 1. 加载预测概率矩阵 .npz
    npz_files = list(run_dir.glob("external_*_predictions.npz"))
    if not npz_files:
        raise FileNotFoundError(f"找不到外部预测的 npz 文件，请检查 {run_dir} 目录下是否有 external_*_predictions.npz")
    data = np.load(npz_files[0])
    y_true = data["y_true"]
    y_prob = data["y_prob"]

    # 2. 按命中的模式（默认 expected_score）进行预测类别测算
    y_pred = get_predictions(y_prob, args.pred_method, args.thresholds)

    # 3. 从底向上一对一精确测算核心指标，绝对不让旧版 JSON 干扰
    qwk = float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    macro_acc = float(np.mean(y_true == y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    ref_true = (y_true >= 2).astype(int)
    ref_prob = np.sum(y_prob[:, 2:], axis=1)
    ref_pred = (y_pred >= 2).astype(int)
    ref_auc = float(roc_auc_score(ref_true, ref_prob))

    cm_ref = confusion_matrix(ref_true, ref_pred)
    tn, fp, fn, tp = cm_ref.ravel()
    ref_sens = float(tp / (tp + fn))
    ref_spec = float(tn / (tn + fp))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(5)))
    sens_dict, spec_dict = {}, {}
    total = cm.sum()
    for i, name in enumerate(CLASS_NAMES):
        c_tp = cm[i, i]
        c_fn = cm[i, :].sum() - c_tp
        c_fp = cm[:, i].sum() - c_tp
        c_tn = total - c_tp - c_fn - c_fp
        sens_dict[name] = float(c_tp / (c_tp + c_fn)) if (c_tp + c_fn) > 0 else 0.0
        spec_dict[name] = float(c_tn / (c_tn + c_fp)) if (c_tn + c_fp) > 0 else 0.0

    metrics = {
        "qwk": qwk,
        "referable_auc": ref_auc,
        "referable_sensitivity": ref_sens,
        "referable_specificity": ref_spec,
        "balanced_accuracy": bal_acc,
        "macro_accuracy": macro_acc
    }

    print(f"\n================ FULL METRICS ({dataset_name} | {args.pred_method.upper()}) ================")
    print(f"QWK                   : {qwk:.4f}")
    print(f"Referable AUC         : {ref_auc:.4f}")
    print(f"Referable Sensitivity : {ref_sens:.4f}")
    print(f"Referable Specificity : {ref_spec:.4f}")
    print(f"Balanced Accuracy     : {bal_acc:.4f}  <-- [NEW]")
    print(f"Macro Accuracy        : {macro_acc:.4f}")

    # 4. 生成 4 张精美图表
    f1 = run_dir / f"fig1_{args.dataset}_roc_curves.png"
    f2 = run_dir / f"fig2_{args.dataset}_sensitivity_specificity.png"
    f3 = run_dir / f"fig3_{args.dataset}_confusion_matrix.png"
    f4 = run_dir / f"fig4_{args.dataset}_overall_metrics.png"

    plot_multiclass_roc(y_true, y_prob, f1, dataset_name)
    plot_sensitivity_specificity(sens_dict, spec_dict, f2, dataset_name)
    plot_confusion_matrix(cm, f3, dataset_name)
    plot_overall_summary_card(metrics, f4, dataset_name)

    print("\n--> 4 张标准学术发表级图表已完成导出:")
    print(f"    [ROC 曲线]          : {f1.name}")
    print(f"    [灵敏度&特异度 bar] : {f2.name}")
    print(f"    [混淆矩阵热力图]    : {f3.name}")
    print(f"    [全核心指标汇总柱图]: {f4.name}")


if __name__ == "__main__":
    main()