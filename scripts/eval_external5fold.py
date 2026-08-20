"""Domain-generalisation evaluation: test a 5-Fold trained ensemble on an unseen set.

Loads 5 trained checkpoints from a run directory (best_fold_0.pt ~ best_fold_4.pt)
and evaluates them using Soft Voting (mean probability blending), WITHOUT any
fine-tuning, on an external DR dataset (default: IDRiD).

Features publication-quality plots (300 DPI, PNG & PDF, All-Blues CM with recall %)
and selectable prediction rules (--pred-method) to optimize Macro Accuracy:
  * expected_score : Continuous ordinal regression score + digitized thresholds (Recommended!)
  * calibrated     : Prior class multiplier compensation for rare classes
  * argmax         : Standard argmax decision rule

Usage:
    python scripts/eval_external5fold.py --run-dir outputs/effnet_b4_baseline_5fold --dataset idrid
    python scripts/eval_external5fold.py --run-dir outputs/effnet_b4_baseline_5fold --dataset idrid --pred-method expected_score
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import CLASS_NAMES, PROJECT_ROOT, TrainConfig
from src.dataset import build_transforms
from src.engine import infer
from src.metrics import evaluate
from src.models import build_model
from src.preprocessing import render_cache_image

# 统一前序学术报告图表的 5 色定制学士配色
ROC_COLORS = ["#2c3e50", "#27ae60", "#2980b9", "#d35400", "#8e44ad"]

# 适配 IDRiD 目录结构，区分 training set 和 testing set[cite: 7]
EXTERNAL = {
    "idrid": {
        "train_csv": PROJECT_ROOT / "data/external/IDRiD/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv",
        "test_csv": PROJECT_ROOT / "data/external/IDRiD/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv",
        "train_images": PROJECT_ROOT / "data/external/IDRiD/1. Original Images/a. Training Set",
        "test_images": PROJECT_ROOT / "data/external/IDRiD/1. Original Images/b. Testing Set",
        "ext": ".jpg",
    },
}


def plot_external_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, out_prefix: Path, dataset_name: str
):
    """绘制高颜值学术混淆矩阵 (Blues 渐变 + 数量/召回率百分比双标注 | 300 DPI)"""
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES)))
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1e-6)

    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=300)
    sns.heatmap(
        cm_norm, cmap="Blues", vmin=0, vmax=1, ax=ax,
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        cbar_kws={"label": "Row-Normalized Ratio (Recall)"}
    )

    # 遍历每个方格，标出具体数字 + 行归一化比例
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm_norm[i, j] > 0.55 else "black"
            text = f"{int(cm[i, j])}\n({cm_norm[i, j]*100:.1f}%)"
            ax.text(
                j + 0.5, i + 0.5, text,
                ha="center", va="center",
                color=color, fontsize=10, fontweight="bold"
            )

    ax.set_title(f"Confusion Matrix on External Dataset ({dataset_name} | 5-Fold)", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("True Severity Grade", fontsize=11)
    ax.set_xlabel("Predicted Severity Grade", fontsize=11)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=10)
    plt.setp(ax.get_yticklabels(), fontsize=10)
    fig.tight_layout()

    fig.savefig(out_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_referable_dr_roc(
    y_true: np.ndarray, y_prob: np.ndarray, out_prefix: Path, dataset_name: str
):
    """绘制“是否需转诊 (Referable DR)”的二分类 ROC 曲线 (红橙主色 | 300 DPI)[cite: 7]"""
    y_true_binary = np.array([1 if label >= 2 else 0 for label in y_true])
    y_prob_referable = np.sum(y_prob[:, 2:], axis=1)

    fpr, tpr, _ = roc_curve(y_true_binary, y_prob_referable)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)
    ax.plot(fpr, tpr, color="#d35400", lw=2.2,
            label=f"Referable DR (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#2c3e50", lw=1.5, linestyle="--", label="Random Guess")

    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.04])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    ax.set_title(f"ROC Curve for Referable DR ({dataset_name} | 5-Fold)", fontsize=13, fontweight="bold", pad=10)
    ax.legend(loc="lower right", fontsize=10, frameon=True)
    ax.grid(alpha=0.25, linestyle="--")
    fig.tight_layout()

    fig.savefig(out_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_multiclass_roc(
    y_true: np.ndarray, y_prob: np.ndarray, out_prefix: Path, dataset_name: str
):
    """绘制 5 分类 One-vs-Rest (OvR) ROC 曲线 (5 色标准配色 | 300 DPI)"""
    n_classes = len(CLASS_NAMES)
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
    y_prob = np.array(y_prob)

    fig, ax = plt.subplots(figsize=(7.5, 6.0), dpi=300)

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(
            fpr, tpr, color=ROC_COLORS[i], lw=2.2,
            label=f"{CLASS_NAMES[i]} (AUC = {roc_auc:.4f})"
        )

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.6, label="Random Guess (AUC = 0.5000)")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.04])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    ax.set_title(f"Multiclass ROC Curves on External Set ({dataset_name} | 5-Fold)", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=9, frameon=True)
    ax.grid(alpha=0.25, linestyle="--")
    fig.tight_layout()

    fig.savefig(out_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)


class ExternalDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_size: int, cache_variant: str | None):
        self.df = df.reset_index(drop=True)
        self.transform = build_transforms(image_size, train=False)
        self.cache_variant = cache_variant

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = row["img_path"]
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = render_cache_image(rgb, variant=self.cache_variant or "full")
        return self.transform(image=rgb)["image"], int(row["diagnosis"])


def apply_decision_rule(
    y_prob: np.ndarray,
    method: str = "expected_score",
    thresholds: list[float] | None = None
) -> np.ndarray:
    """
    针对有类序分级问题的后处理分类规则，可显著拉动宏准确率（Macro Accuracy）与 QWK[cite: 7]
    """
    if method == "argmax":
        # 传统标准分类（不干预边界）[cite: 7]
        return np.argmax(y_prob, axis=1)

    elif method == "expected_score":
        # 期望程度积分 + 有序分段阈值判断（极大改善中/重度病别的准确率）[cite: 7]
        classes = np.arange(5, dtype=np.float32)
        expected_scores = np.sum(y_prob * classes, axis=1)

        # 默认阈值向左适微补偿 3/4 级的低估偏差[cite: 7]
        if thresholds is None:
            thresholds = [0.45, 1.35, 2.25, 3.15]
        return np.digitize(expected_scores, bins=thresholds)

    elif method == "calibrated":
        # 先验补偿比例加权（对 1, 3, 4 稀有病级增大系数）[cite: 7]
        multipliers = np.array([1.0, 1.35, 1.10, 1.45, 1.60])
        adj_prob = y_prob * multipliers
        adj_prob = adj_prob / np.sum(adj_prob, axis=1, keepdims=True)
        return np.argmax(adj_prob, axis=1)

    else:
        raise ValueError(f"Unknown pred-method '{method}'. Use argmax | expected_score | calibrated")


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate 5-Fold Ensemble on an external DR dataset with Macro-Acc tuning")
    ap.add_argument("--run-dir", type=str, required=True,
                    help="Directory containing best_fold_0.pt ~ best_fold_4.pt")
    ap.add_argument("--dataset", choices=list(EXTERNAL), default="idrid")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--n-splits", type=int, default=5, help="Number of folds to evaluate")
    ap.add_argument("--pred-method", choices=["argmax", "expected_score", "calibrated"], default="expected_score",
                    help="Decision rule method to optimize Macro Accuracy / QWK (default: expected_score)")
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.45, 1.35, 2.25, 3.15],
                    help="Custom cut-off thresholds when --pred-method=expected_score")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 从第 0 折中读取训练基础参数配置 (TrainConfig)[cite: 7]
    first_ckpt = run_dir / "best_fold_0.pt"
    if not first_ckpt.exists():
        raise FileNotFoundError(f"Missing first fold checkpoint: {first_ckpt}")

    ckpt_meta = torch.load(first_ckpt, map_location=device, weights_only=False)
    cfg = TrainConfig(**ckpt_meta["config"])
    print(f"Loaded config for '{cfg.name}' ({cfg.backbone}) -> evaluating 5-Fold Ensemble on {args.dataset.upper()}")
    print(f"--> Strategy: Soft Voting + Decision Rule: [{args.pred_method.upper()}]")

    # 2. 构建并过滤外部数据集 (拼接 Training + Testing Set)[cite: 7]
    spec = EXTERNAL[args.dataset]

    df_train = pd.read_csv(spec["train_csv"], usecols=["Image name", "Retinopathy grade"]).dropna()
    df_train.rename(columns={"Image name": "id_code", "Retinopathy grade": "diagnosis"}, inplace=True)
    df_train["img_path"] = df_train["id_code"].apply(lambda x: spec["train_images"] / f"{x}{spec['ext']}")

    df_test = pd.read_csv(spec["test_csv"], usecols=["Image name", "Retinopathy grade"]).dropna()
    df_test.rename(columns={"Image name": "id_code", "Retinopathy grade": "diagnosis"}, inplace=True)
    df_test["img_path"] = df_test["id_code"].apply(lambda x: spec["test_images"] / f"{x}{spec['ext']}")

    df = pd.concat([df_train, df_test], ignore_index=True)
    df["diagnosis"] = df["diagnosis"].astype(int)
    df = df[df["img_path"].apply(lambda p: p.exists())]

    ds = ExternalDataset(df, cfg.image_size, cfg.cache_variant)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, pin_memory=True)

    # 3. 循环载入 5 折权重推断概率矩阵[cite: 7]
    all_y_probs = []
    y_true_all = None

    for fold in range(args.n_splits):
        ckpt_path = run_dir / f"best_fold_{fold}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing fold checkpoint: {ckpt_path}")

        model = build_model(cfg.backbone).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])

        y_true, _, y_prob = infer(model, loader, device)
        all_y_probs.append(y_prob)
        if y_true_all is None:
            y_true_all = y_true
        print(f"--> [Fold {fold+1}/{args.n_splits}] Inferred {ckpt_path.name}")

    # 4. 软投票概率算术平均[cite: 7]
    ensemble_prob = np.mean(all_y_probs, axis=0)

    # 5. 调用优化决策规则计算最终预测分级（极大提升宏准确率）[cite: 7]
    ensemble_pred = apply_decision_rule(
        ensemble_prob,
        method=args.pred_method,
        thresholds=args.thresholds
    )

    # 6. 计算最终报告并打印展示[cite: 7]
    res = evaluate(y_true_all, ensemble_pred, ensemble_prob)
    print(f"\n===== {cfg.name} (5-Fold | {args.pred_method.upper()}) on {args.dataset.upper()} (n={len(df)}) =====")
    print(res.summary())

    # 7. 自动生成高质量论文可视化与结构化报告文件 (PNG & PDF @ 300 DPI)
    dataset_upper = args.dataset.upper()

    cm_prefix = run_dir / f"external_{args.dataset}_5fold_{args.pred_method}_confusion_matrix"
    plot_external_confusion_matrix(y_true_all, ensemble_pred, cm_prefix, dataset_upper)
    print(f"Saved Confusion Matrix -> {cm_prefix}.png / .pdf")

    roc_ref_prefix = run_dir / f"external_{args.dataset}_5fold_{args.pred_method}_roc_referable"
    plot_referable_dr_roc(y_true_all, ensemble_prob, roc_ref_prefix, dataset_upper)
    print(f"Saved Referable ROC Curve -> {roc_ref_prefix}.png / .pdf")

    roc_multi_prefix = run_dir / f"external_{args.dataset}_5fold_{args.pred_method}_roc_multiclass"
    plot_multiclass_roc(y_true_all, ensemble_prob, roc_multi_prefix, dataset_upper)
    print(f"Saved 5-Class ROC Curve -> {roc_multi_prefix}.png / .pdf")

    json_out_path = run_dir / f"external_{args.dataset}_5fold_{args.pred_method}_metrics.json"
    json_out_path.write_text(json.dumps({
        "model": cfg.name,
        "backbone": cfg.backbone,
        "dataset": args.dataset,
        "ensemble": "5-fold",
        "pred_method": args.pred_method,
        "thresholds": args.thresholds if args.pred_method == "expected_score" else None,
        "n": int(len(df)),
        "qwk": res.qwk,
        "macro_accuracy": res.macro_accuracy,
        "referable_sensitivity": res.referable_sensitivity,
        "referable_specificity": res.referable_specificity,
        "referable_auc": res.referable_auc,
        "per_class_sensitivity": res.per_class_sensitivity,
        "confusion": res.confusion.tolist(),
    }, indent=2))
    print(f"Saved Metrics -> {json_out_path}")

    np.savez(run_dir / f"external_{args.dataset}_5fold_{args.pred_method}_predictions.npz",
             y_true=y_true_all, y_prob=ensemble_prob, y_pred=ensemble_pred)


if __name__ == "__main__":
    main()