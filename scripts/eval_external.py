"""Domain-generalisation evaluation: test an APTOS-trained model on an unseen set.

Loads a trained checkpoint and evaluates it, WITHOUT any fine-tuning, on an
external DR dataset (default: IDRiD). The external images are preprocessed with
exactly the same pipeline the model was trained on, so any performance drop
reflects genuine domain shift (different camera / population / grade distribution)
rather than a preprocessing mismatch.

Usage:
    python scripts/eval_external.py outputs/effnet_b4_focal/best.pt
    python scripts/eval_external.py outputs/swin_b_focal/best.pt --dataset idrid
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import PROJECT_ROOT, TrainConfig  # noqa: E402
from src.dataset import build_transforms  # noqa: E402
from src.engine import infer  # noqa: E402
from src.metrics import evaluate  # noqa: E402
from src.models import build_model  # noqa: E402
from src.preprocessing import render_cache_image  # noqa: E402

# Registry of external datasets: label csv + image dir + filename builder.
# [UPDATED] 适配最新的 IDRiD 目录结构，区分了 training set 和 testing set
EXTERNAL = {
    "idrid": {
        "train_csv": PROJECT_ROOT / "data/external/IDRiD/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv",
        "test_csv": PROJECT_ROOT / "data/external/IDRiD/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv",
        "train_images": PROJECT_ROOT / "data/external/IDRiD/1. Original Images/a. Training Set",
        "test_images": PROJECT_ROOT / "data/external/IDRiD/1. Original Images/b. Testing Set",
        "ext": ".jpg",
    },
}


def plot_referable_dr_roc(y_true, y_prob, out_path, dataset_name):
    """
    绘制“是否需转诊 (Referable DR)”的二分类 ROC 曲线
    """
    y_true_binary = np.array([1 if label >= 2 else 0 for label in y_true])
    y_prob_referable = np.sum(y_prob[:, 2:], axis=1)

    fpr, tpr, _ = roc_curve(y_true_binary, y_prob_referable)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6), dpi=150)
    plt.plot(fpr, tpr, color='#d35400', lw=2.5,
             label=f'Referable DR (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='#2c3e50', lw=2, linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title(f'ROC Curve for Referable DR ({dataset_name})', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(alpha=0.3)

    plt.savefig(out_path, bbox_inches='tight')
    plt.close()


class ExternalDataset(Dataset):
    def __init__(self, df, image_size, cache_variant):
        self.df = df.reset_index(drop=True)
        self.transform = build_transforms(image_size, train=False)
        self.cache_variant = cache_variant

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # 直接使用预先拼接好的绝对路径
        path = row['img_path']
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = render_cache_image(rgb, variant=self.cache_variant or "full")
        return self.transform(image=rgb)["image"], int(row["diagnosis"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=str)
    ap.add_argument("--dataset", choices=list(EXTERNAL), default="idrid")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = TrainConfig(**ckpt["config"])
    model = build_model(cfg.backbone).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded {cfg.backbone} (val_metric={ckpt.get('val_metric'):.4f}) "
          f"-> evaluating on {args.dataset} (no fine-tuning)")

    spec = EXTERNAL[args.dataset]

    # 1. 加载并处理 Training Set 标签
    df_train = pd.read_csv(spec["train_csv"], usecols=["Image name", "Retinopathy grade"]).dropna()
    df_train.rename(columns={"Image name": "id_code", "Retinopathy grade": "diagnosis"}, inplace=True)
    df_train["img_path"] = df_train["id_code"].apply(lambda x: spec["train_images"] / f"{x}{spec['ext']}")

    # 2. 加载并处理 Testing Set 标签
    df_test = pd.read_csv(spec["test_csv"], usecols=["Image name", "Retinopathy grade"]).dropna()
    df_test.rename(columns={"Image name": "id_code", "Retinopathy grade": "diagnosis"}, inplace=True)
    df_test["img_path"] = df_test["id_code"].apply(lambda x: spec["test_images"] / f"{x}{spec['ext']}")

    # 3. 将两者拼接在一起，在整个 IDRiD 集合上进行评估测试
    df = pd.concat([df_train, df_test], ignore_index=True)
    df["diagnosis"] = df["diagnosis"].astype(int)

    # 4. 过滤掉硬盘上实际不存在的图片，防止报错
    df = df[df["img_path"].apply(lambda p: p.exists())]

    # 初始化 Dataset 与 Dataloader
    ds = ExternalDataset(df, cfg.image_size, cfg.cache_variant)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, pin_memory=True)
    y_true, y_pred, y_prob = infer(model, loader, device)
    res = evaluate(y_true, y_pred, y_prob)

    print(f"\n===== {cfg.name} on {args.dataset.upper()} (n={len(df)}) =====")
    print(res.summary())

    out_path = Path(args.checkpoint).parent / f"external_{args.dataset}_metrics.json"
    out_path.write_text(json.dumps({
        "model": cfg.name, "backbone": cfg.backbone, "dataset": args.dataset,
        "n": int(len(df)), "qwk": res.qwk, "macro_accuracy": res.macro_accuracy,
        "referable_sensitivity": res.referable_sensitivity,
        "referable_specificity": res.referable_specificity,
        "referable_auc": res.referable_auc,
        "per_class_sensitivity": res.per_class_sensitivity,
        "confusion": res.confusion.tolist(),
    }, indent=2))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()