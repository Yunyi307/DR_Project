"""Generate the remaining figures used in the thesis (preprocessing, EDA, DG)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import CLASS_NAMES, OUTPUT_DIR, PROJECT_ROOT, SPLIT_CSV, TRAIN_IMAGES_DIR
from src.preprocessing import apply_clahe, ben_graham, circular_mask, crop_to_roi

FIG = PROJECT_ROOT / "report_figures"
FIG.mkdir(exist_ok=True)


def preprocessing_stages() -> None:
    df = pd.read_csv(SPLIT_CSV)
    row = df[df["diagnosis"] == 2].iloc[0]           # a Moderate case shows lesions
    rgb = cv2.cvtColor(cv2.imread(str(TRAIN_IMAGES_DIR / f"{row['id_code']}.png")),
                       cv2.COLOR_BGR2RGB)
    crop = crop_to_roi(rgb)
    crop = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_AREA)
    clahe = apply_clahe(crop)
    full = circular_mask(ben_graham(clahe))
    stages = [(rgb, "(a) Original"), (crop, "(b) Cropped ROI"),
              (clahe, "(c) + CLAHE"), (full, "(d) + Ben Graham")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for ax, (img, title) in zip(axes, stages):
        ax.imshow(img); ax.set_title(title, fontsize=12); ax.axis("off")
    fig.tight_layout(); fig.savefig(FIG / "preprocessing_stages.png", dpi=150); plt.close(fig)


def class_distribution() -> None:
    # 1. 计算 APTOS 的类别分布
    aptos = pd.read_csv(SPLIT_CSV)["diagnosis"].value_counts(normalize=True).reindex(range(5), fill_value=0)

    # 2. 适配最新的 IDRiD 目录结构，分别读取 Training 和 Testing 标签
    idrid_train_csv = PROJECT_ROOT / "data/external/IDRiD/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv"
    idrid_test_csv = PROJECT_ROOT / "data/external/IDRiD/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv"

    df_train = pd.read_csv(idrid_train_csv, usecols=["Retinopathy grade"]).dropna()
    df_test = pd.read_csv(idrid_test_csv, usecols=["Retinopathy grade"]).dropna()

    # 3. 合并计算 IDRiD 的总分布
    idrid_df = pd.concat([df_train, df_test], ignore_index=True)
    idrid = idrid_df["Retinopathy grade"].astype(int).value_counts(normalize=True).reindex(range(5), fill_value=0)

    # 4. 绘图
    x = np.arange(5); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w/2, aptos.values * 100, w, label="APTOS (train)", color="#2166ac")
    ax.bar(x + w/2, idrid.values * 100, w, label="IDRiD (external)", color="#b2182b")
    ax.set_xticks(x); ax.set_xticklabels([f"{i}\n{CLASS_NAMES[i]}" for i in range(5)], fontsize=9)
    ax.set_ylabel("Share of images (%)"); ax.set_title("DR grade distribution: APTOS vs. IDRiD")
    ax.legend(); fig.tight_layout(); fig.savefig(FIG / "class_distribution.png", dpi=150); plt.close(fig)


def dg_gap() -> None:
    # 定义你计划要画的所有模型（已添加 effnet_b4_ordinal）
    models = [("effnet_b4_ce", "EffNet-CE"), ("effnet_b4_focal", "EffNet-Focal"),
              ("effnet_b4_ordinal", "EffNet-Ordinal"),
              ("vit_b16_focal", "ViT-B/16"), ("swin_b_focal", "Swin-B")]

    ind, ext, valid_labels = [], [], []

    # 动态检查文件是否存在
    for name, label in models:
        test_path = OUTPUT_DIR / name / "test_metrics.json"
        ext_path = OUTPUT_DIR / name / "external_idrid_metrics.json"

        if test_path.exists() and ext_path.exists():
            ind.append(json.loads(test_path.read_text())["qwk"])
            ext.append(json.loads(ext_path.read_text())["qwk"])
            valid_labels.append(label)
        else:
            print(f"⚠️ 跳过 {label}: 找不到测试结果文件 ({name})")

    if not valid_labels:
        print("❌ 没有任何模型有完整的测试数据，无法绘制 dg_gap 图。")
        return

    x = np.arange(len(valid_labels));
    w = 0.38
    fig, ax = plt.subplots(figsize=(10, 4.5))

    # 接收 bar() 的返回值
    rects1 = ax.bar(x - w / 2, ind, w, label="APTOS test (in-domain)", color="#2166ac")
    rects2 = ax.bar(x + w / 2, ext, w, label="IDRiD (out-of-domain)", color="#f4a582")

    # 在柱子顶部添加数字标签 (fmt='%.3f' 表示保留3位小数，padding是距离顶部的间距)
    ax.bar_label(rects1, padding=3, fmt='%.3f', fontsize=8)
    ax.bar_label(rects2, padding=3, fmt='%.3f', fontsize=8)

    ax.axhline(0.85, ls="--", color="grey", lw=1, label="QWK target 0.85")
    ax.set_xticks(x);
    ax.set_xticklabels(valid_labels)
    ax.set_ylabel("Quadratic Weighted Kappa");
    ax.set_ylim(0.6, 0.98)  # 稍微调高 y 轴上限，防止数字被遮挡
    ax.set_title("Domain-generalisation gap (in-domain vs. unseen dataset)")
    ax.legend(fontsize=8);
    fig.tight_layout();
    fig.savefig(FIG / "dg_gap.png", dpi=150);
    plt.close(fig)


def dg_gap_accuracy() -> None:
    # 定义你计划要画的所有模型（已添加 effnet_b4_ordinal）
    models = [("effnet_b4_ce", "EffNet-CE"), ("effnet_b4_focal", "EffNet-Focal"),
              ("effnet_b4_ordinal", "EffNet-Ordinal"),
              ("vit_b16_focal", "ViT-B/16"), ("swin_b_focal", "Swin-B")]

    ind, ext, valid_labels = [], [], []

    # 动态检查文件是否存在
    for name, label in models:
        test_path = OUTPUT_DIR / name / "test_metrics.json"
        ext_path = OUTPUT_DIR / name / "external_idrid_metrics.json"

        if test_path.exists() and ext_path.exists():
            # 读取宏平均准确率 (macro_accuracy)
            ind.append(json.loads(test_path.read_text())["macro_accuracy"])
            ext.append(json.loads(ext_path.read_text())["macro_accuracy"])
            valid_labels.append(label)
        else:
            # 静默跳过，避免与 dg_gap 的警告重复打印
            pass

    if not valid_labels:
        return

    x = np.arange(len(valid_labels));
    w = 0.38
    fig, ax = plt.subplots(figsize=(10, 4.5))

    # 接收 bar() 的返回值
    rects1 = ax.bar(x - w / 2, ind, w, label="APTOS test (in-domain)", color="#2166ac")
    rects2 = ax.bar(x + w / 2, ext, w, label="IDRiD (out-of-domain)", color="#f4a582")

    # 在柱子顶部添加数字标签
    ax.bar_label(rects1, padding=3, fmt='%.3f', fontsize=8)
    ax.bar_label(rects2, padding=3, fmt='%.3f', fontsize=8)

    ax.set_xticks(x);
    ax.set_xticklabels(valid_labels)
    ax.set_ylabel("Macro Accuracy");
    ax.set_ylim(0.4, 1.05)  # 稍微调高 y 轴上限，防止数字被遮挡
    ax.set_title("Domain-generalisation gap (Accuracy)")
    ax.legend(fontsize=8);
    fig.tight_layout()

    fig.savefig(FIG / "dg_gap_accuracy.png", dpi=150);
    plt.close(fig)

if __name__ == "__main__":
    preprocessing_stages()
    class_distribution()
    dg_gap()  # 生成 QWK 泛化差距图
    dg_gap_accuracy()  # 生成 Accuracy 泛化差距图
    print(f"Wrote report figures to {FIG}")