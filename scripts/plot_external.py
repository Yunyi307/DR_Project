"""Generate publication-quality Confusion Matrix and ROC Curves (Binary & Multiclass).

Usage:
    python scripts/plot_external.py outputs/effnet_b4_focal/best.pt --dataset idrid_all
"""
from __future__ import annotations

import argparse
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
from src.config import PROJECT_ROOT, CLASS_NAMES, TrainConfig  # noqa: E402
from src.dataset import build_transforms  # noqa: E402
from src.engine import infer  # noqa: E402
from src.models import build_model  # noqa: E402
from src.preprocessing import render_cache_image  # noqa: E402

# 统一前序图表的 5 色定制学士配色
ROC_COLORS = ["#2c3e50", "#27ae60", "#2980b9", "#d35400", "#8e44ad"]

EXTERNAL = {
    "idrid_all": {
        "csv": [
            PROJECT_ROOT / "data/external/IDRiD/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv",
            PROJECT_ROOT / "data/external/IDRiD/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv"
        ],
        "images": [
            PROJECT_ROOT / "data/external/IDRiD/1. Original Images/a. Training Set",
            PROJECT_ROOT / "data/external/IDRiD/1. Original Images/b. Testing Set"
        ],
    }
}


class ExternalDataset(Dataset):
    def __init__(self, df, image_size, cache_variant):
        self.df = df.reset_index(drop=True)
        self.transform = build_transforms(image_size, train=False)
        self.cache_variant = cache_variant

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["img_path"]
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = render_cache_image(rgb, variant=self.cache_variant or "full")
        return self.transform(image=rgb)["image"], int(row["diagnosis"])


def plot_confusion_matrix(y_true, y_pred, save_dir: Path):
    """绘制高颜值学术混淆矩阵 (蓝色语系 + 数量/召回百分比双标注)"""
    cm = confusion_matrix(y_true, y_pred)
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

    ax.set_title("Confusion Matrix on Unseen Domain (IDRiD)", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("True Severity Grade", fontsize=11)
    ax.set_xlabel("Predicted Severity Grade", fontsize=11)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=10)
    plt.setp(ax.get_yticklabels(), fontsize=10)
    fig.tight_layout()

    fig.savefig(save_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    fig.savefig(save_dir / "confusion_matrix.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_binary_roc_curve(y_true, y_prob, save_dir: Path):
    """绘制二分类 ROC 曲线 (Referable DR) - 保持同款橙红配色与样式"""
    y_true_binary = (np.array(y_true) >= 2).astype(int)
    y_prob_referable = np.sum(np.array(y_prob)[:, 2:], axis=1)

    fpr, tpr, _ = roc_curve(y_true_binary, y_prob_referable)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)
    ax.plot(fpr, tpr, color="#d35400", lw=2.2,
            label=f"Referable DR (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#2c3e50", lw=1.5, ls="--", label="Random Guess")

    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.04])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    ax.set_title("ROC Curve for Referable DR Screening (Binary)", fontsize=13, fontweight="bold", pad=10)
    ax.legend(loc="lower right", fontsize=10, frameon=True)
    ax.grid(alpha=0.25, linestyle="--")
    fig.tight_layout()

    fig.savefig(save_dir / "roc_curve_binary_referable.png", dpi=300, bbox_inches="tight")
    fig.savefig(save_dir / "roc_curve_binary_referable.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_multiclass_roc_curve(y_true, y_prob, save_dir: Path):
    """绘制五分类 ROC 曲线 (One-vs-Rest) - 匹配 5 色定制配色表[cite: 5]"""
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
    ax.set_title("Multiclass ROC Curves on Unseen Domain (One-vs-Rest)", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=9, frameon=True)
    ax.grid(alpha=0.25, linestyle="--")
    fig.tight_layout()

    fig.savefig(save_dir / "roc_curve_multiclass.png", dpi=300, bbox_inches="tight")
    fig.savefig(save_dir / "roc_curve_multiclass.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=str)
    ap.add_argument("--dataset", choices=list(EXTERNAL), default="idrid_all")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = TrainConfig(**ckpt["config"])
    model = build_model(cfg.backbone).to(device)
    model.load_state_dict(ckpt["model"])

    print(f"Generating academic plots for {cfg.backbone} on {args.dataset.upper()}...")

    spec = EXTERNAL[args.dataset]
    df_list = []
    for csv_path, img_dir in zip(spec["csv"], spec["images"]):
        temp_df = pd.read_csv(csv_path)
        if "Image name" in temp_df.columns:
            temp_df = temp_df.rename(columns={"Image name": "id_code", "Retinopathy grade": "diagnosis"})
        temp_df = temp_df[["id_code", "diagnosis"]].dropna()
        temp_df["diagnosis"] = temp_df["diagnosis"].astype(int)

        valid_rows = []
        for _, row in temp_df.iterrows():
            img_id = str(row["id_code"]).strip()
            p_jpg = img_dir / f"{img_id}.jpg"
            p_tif = img_dir / f"{img_id}.tif"
            if p_jpg.exists():
                row["img_path"] = p_jpg
                valid_rows.append(row)
            elif p_tif.exists():
                row["img_path"] = p_tif
                valid_rows.append(row)
        df_list.append(pd.DataFrame(valid_rows))

    df = pd.concat(df_list, ignore_index=True)
    ds = ExternalDataset(df, cfg.image_size, cfg.cache_variant)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, pin_memory=True)

    y_true, y_pred, y_prob = infer(model, loader, device)

    save_dir = Path(args.checkpoint).parent

    print("Drawing Confusion Matrix...")
    plot_confusion_matrix(y_true, y_pred, save_dir)

    print("Drawing Binary ROC Curve (Referable DR)...")
    plot_binary_roc_curve(y_true, y_prob, save_dir)

    print("Drawing Multiclass ROC Curves (5-Class)...")
    plot_multiclass_roc_curve(y_true, y_prob, save_dir)

    print(f"\n✅ All plots saved successfully in: {save_dir}")
    print("Files generated:")
    print("  - confusion_matrix.png / .pdf")
    print("  - roc_curve_binary_referable.png / .pdf")
    print("  - roc_curve_multiclass.png / .pdf")


if __name__ == "__main__":
    main()