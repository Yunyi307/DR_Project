"""Generate a Grad-CAM panel: one correctly-graded example per DR severity.

For each grade 0-4 we pick a validation image the model grades correctly and show
the original (preprocessed) image beside its Grad-CAM overlay. On DR-positive
grades the heatmap should concentrate on lesions (haemorrhages, exudates,
neovascularisation) rather than the optic disc - the qualitative evidence for the
interpretability section of the report.

Usage:
    python scripts/make_gradcam.py outputs/effnet_b4_focal/best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import CLASS_NAMES, PROCESSED_DIR, SPLIT_CSV, TrainConfig  # noqa: E402
from src.dataset import build_transforms  # noqa: E402
from src.gradcam import GradCAM, overlay_cam  # noqa: E402
from src.models import build_model, find_target_layer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=str)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = TrainConfig(**ckpt["config"])
    model = build_model(cfg.backbone).to(device)
    model.load_state_dict(ckpt["model"])
    cam_engine = GradCAM(model, find_target_layer(model))

    df = pd.read_csv(SPLIT_CSV)
    val = df[df["split"] == "val"]
    tf = build_transforms(cfg.image_size, train=False)
    cache_dir = PROCESSED_DIR / (cfg.cache_variant or "full")

    fig, axes = plt.subplots(2, 5, figsize=(16, 6.5))
    for grade in range(5):
        rows = val[val["diagnosis"] == grade]
        chosen, img_rgb, cam = None, None, None
        for _, row in rows.iterrows():
            p = cache_dir / f"{row['id_code']}.png"
            rgb = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
            x = tf(image=rgb)["image"].unsqueeze(0).to(device)
            heat, pred = cam_engine(x, class_idx=grade)
            if pred == grade:  # prefer a correctly-graded example
                chosen, img_rgb, cam = row["id_code"], rgb, heat
                break
        if chosen is None:  # fall back to the first available
            row = rows.iloc[0]
            rgb = cv2.cvtColor(cv2.imread(str(cache_dir / f"{row['id_code']}.png")),
                               cv2.COLOR_BGR2RGB)
            x = tf(image=rgb)["image"].unsqueeze(0).to(device)
            cam, _ = cam_engine(x, class_idx=grade)
            img_rgb = rgb

        axes[0, grade].imshow(img_rgb); axes[0, grade].set_title(f"{grade}: {CLASS_NAMES[grade]}")
        axes[1, grade].imshow(overlay_cam(img_rgb, cam))
        for ax in (axes[0, grade], axes[1, grade]):
            ax.axis("off")
    axes[0, 0].set_ylabel("input", fontsize=11)
    fig.suptitle(f"Grad-CAM by DR grade - {cfg.name}", fontsize=13)
    fig.tight_layout()

    out = Path(args.checkpoint).parent / "figures" / "gradcam_panel.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
