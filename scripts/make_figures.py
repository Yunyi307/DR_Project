from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import auc, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import CLASS_NAMES, OUTPUT_DIR  # noqa: E402


def training_curves(history: list[dict], out_path: Path, title_suffix: str = "") -> None:
    """绘制训练收敛曲线 (支持单次或 5 折平均指标)"""
    epochs = [h["epoch"] for h in history]
    fig, ax1 = plt.subplots(figsize=(7.5, 4.8), dpi=200)

    ax1.plot(epochs, [h["val_qwk"] for h in history], "-o", color="#2166ac",
             label="val QWK", markersize=4, lw=1.8)
    ax1.plot(epochs, [h["val_macro_acc"] for h in history], "-s", color="#4393c3",
             label="val accuracy", markersize=4, lw=1.8)
    ax1.axhline(0.85, ls="--", color="grey", lw=1.2, label="QWK target 0.85")

    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Validation Metric", fontsize=11)
    ax1.set_ylim(0.4, 1.0)
    ax1.grid(alpha=0.25, linestyle="--")

    ax2 = ax1.twinx()
    ax2.plot(epochs, [h["train_loss"] for h in history], "-^", color="#b2182b",
             label="train loss", markersize=4, lw=1.8, alpha=0.75)
    ax2.set_ylabel("Train Loss", color="#b2182b", fontsize=11)

    # 组合双坐标轴图例
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="lower right", fontsize=9, frameon=True)
    ax1.set_title(f"Training Convergence Curves {title_suffix}".strip(), fontsize=12, fontweight="bold", pad=10)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def confusion_heatmap(cm: np.ndarray, out_path: Path) -> None:
    """绘制蓝色系列混淆矩阵，格内同时呈现【样本数 + 归一化百分比】"""
    cm = np.array(cm, dtype=float)
    # 按行(真实病变等级)归一化，得到每个类别预测为哪一级的百分比 (Recall)
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1e-6)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=200)
    sns.heatmap(
        cm_norm, cmap="Blues", vmin=0, vmax=1, ax=ax,
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        cbar_kws={"label": "Row-Normalized Ratio (Recall)"}
    )

    # 遍历每个方格，打印粗体双标注：数值 + 百分比
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm_norm[i, j] > 0.55 else "black"
            text = f"{int(cm[i, j])}\n({cm_norm[i, j] * 100:.1f}%)"
            ax.text(
                j + 0.5, i + 0.5, text,
                ha="center", va="center",
                color=color, fontsize=10, fontweight="bold"
            )

    ax.set_xlabel("Predicted Grade", fontsize=11)
    ax.set_ylabel("True Grade", fontsize=11)
    ax.set_title("Confusion Matrix (Counts & Recall Percentage)", fontsize=13, fontweight="bold", pad=12)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=10)
    plt.setp(ax.get_yticklabels(), fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path) -> None:
    """绘制 Referable DR (0,1 vs 2,3,4) 二分类 ROC 曲线"""
    y_true_binary = (y_true >= 2).astype(int)
    y_prob_referable = np.sum(y_prob[:, 2:], axis=1)

    fpr, tpr, _ = roc_curve(y_true_binary, y_prob_referable)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=200)
    ax.plot(fpr, tpr, color="#d35400", lw=2.2, label=f"Referable DR (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#2c3e50", lw=1.5, ls="--", label="Random Guess")

    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.04])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    ax.set_title("ROC Curve for Referable DR (Grades 2-4)", fontsize=13, fontweight="bold", pad=10)
    ax.legend(loc="lower right", fontsize=10, frameon=True)
    ax.grid(alpha=0.25, linestyle="--")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def load_history_data(run_dir: Path) -> tuple[list[dict] | None, str]:
    """智能载入历史数据：首先找单次 history.json，没有则尝试取 5 折均值"""
    single_hist = run_dir / "history.json"
    if single_hist.exists():
        return json.loads(single_hist.read_text()), ""

    # 如果是 5 折实验，查找所有 history_fold_*.json 并把每一轮指标做均值融合
    fold_files = sorted(run_dir.glob("history_fold_*.json"))
    if not fold_files:
        return None, ""

    all_folds = [json.loads(p.read_text()) for p in fold_files]
    max_epochs = max(len(f) for f in all_folds)
    avg_history = []

    for ep_idx in range(max_epochs):
        ep_num = ep_idx + 1
        val_qwks, val_accs, train_losses = [], [], []
        for fold_hist in all_folds:
            if ep_idx < len(fold_hist):
                val_qwks.append(fold_hist[ep_idx]["val_qwk"])
                val_accs.append(fold_hist[ep_idx]["val_macro_acc"])
                train_losses.append(fold_hist[ep_idx]["train_loss"])

        avg_history.append({
            "epoch": ep_num,
            "val_qwk": float(np.mean(val_qwks)),
            "val_macro_acc": float(np.mean(val_accs)),
            "train_loss": float(np.mean(train_losses)),
        })

    return avg_history, f"(5-Fold Mean, n={len(fold_files)})"


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "effnet_b4_focal"
    run_dir = OUTPUT_DIR / name
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. 兼容读取 5 折或单次训练的 test_metrics JSON 文件
    metrics_path = run_dir / "test_metrics_5fold.json"
    if not metrics_path.exists():
        metrics_path = run_dir / "test_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"未找到评估报告文件 test_metrics.json 或 test_metrics_5fold.json: {run_dir}")

    metrics = json.loads(metrics_path.read_text())

    # 2. 生成蓝色系列混淆矩阵 (数值 + 百分比)
    confusion_heatmap(metrics["confusion"], fig_dir / "confusion_matrix.png")
    print("✅ 已生成混淆矩阵 -> confusion_matrix.png")

    # 3. 兼容生成 Referable DR 二分类 ROC 曲线
    npz_path = run_dir / "predictions_5fold.npz"
    if not npz_path.exists():
        npz_path = run_dir / "predictions.npz"
    if npz_path.exists():
        data = np.load(npz_path)
        plot_roc_curve(data["y_true"], data["y_prob"], fig_dir / "roc_curve.png")
        print("✅ 已生成转诊 ROC 曲线 -> roc_curve.png")

    # 4. 智能判断并绘制训练曲线 (单次模型 / 5 折平均训练曲线)
    history, suffix = load_history_data(run_dir)
    if history is not None:
        training_curves(history, fig_dir / "training_curves.png", title_suffix=suffix)
        print("✅ 已生成训练收敛曲线 -> training_curves.png")

    print(f"\n🎉 所有图片导出完毕！已保存至文件夹:\n   {fig_dir}")


if __name__ == "__main__":
    main()