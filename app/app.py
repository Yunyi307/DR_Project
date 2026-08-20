"""Interactive DR grading demo (Streamlit).

Upload a retinal fundus photograph and the app will:
  1. show the preprocessing pipeline (crop -> CLAHE -> Ben Graham),
  2. predict the DR grade (0-4) with per-class confidence,
  3. give the binary "referable DR" screening decision, and
  4. overlay a Grad-CAM heatmap of the regions driving the prediction.

Run with:
    streamlit run app/app.py

The pure-Python helpers (load_model, predict, make_gradcam) are importable and
unit-tested separately in tests; only main() touches Streamlit.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import CLASS_NAMES, OUTPUT_DIR, REFERABLE_THRESHOLD, TrainConfig
from src.dataset import build_transforms
from src.gradcam import GradCAM, overlay_cam
from src.models import build_model, find_target_layer
from src.preprocessing import render_cache_image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def list_checkpoints() -> dict[str, Path]:
    """Map experiment name -> best.pt path for every trained model on disk."""
    return {p.parent.name: p for p in sorted(OUTPUT_DIR.glob("*/best.pt"))}


def load_model(checkpoint: Path):
    """Load a checkpoint; return (model, config). Cached by callers."""
    ckpt = torch.load(checkpoint, map_location=DEVICE, weights_only=False)
    cfg = TrainConfig(**ckpt["config"])
    model = build_model(cfg.backbone).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def preprocess(image_rgb: np.ndarray, cfg: TrainConfig) -> tuple[np.ndarray, torch.Tensor]:
    """Return (display image at cache resolution, model-input tensor)."""
    processed = render_cache_image(image_rgb, variant=cfg.cache_variant or "full")
    tensor = build_transforms(cfg.image_size, train=False)(image=processed)["image"]
    return processed, tensor.unsqueeze(0).to(DEVICE)


@torch.no_grad()
def predict(model, tensor: torch.Tensor) -> np.ndarray:
    """Return class-probability vector (length 5)."""
    return torch.softmax(model(tensor), dim=1).cpu().numpy()[0]


def make_gradcam(model, cfg: TrainConfig, tensor: torch.Tensor,
                 display_img: np.ndarray, class_idx: int) -> np.ndarray | None:
    """Grad-CAM overlay for CNN backbones; None for pure ViTs (not supported)."""
    if "vit" in cfg.backbone and "swin" not in cfg.backbone:
        return None
    cam, _ = GradCAM(model, find_target_layer(model))(tensor, class_idx=class_idx)
    return overlay_cam(display_img, cam)


def referable_decision(probs: np.ndarray) -> tuple[bool, float]:
    """Binary referable-DR call (grade >= threshold) and its probability."""
    p_ref = float(probs[REFERABLE_THRESHOLD:].sum())
    return p_ref >= 0.5, p_ref


# --------------------------------------------------------------------------- #
# Streamlit UI
# --------------------------------------------------------------------------- #
def main() -> None:
    import pandas as pd
    import streamlit as st

    st.set_page_config(page_title="DR Grading Demo", page_icon="👁", layout="wide")
    st.title("👁 Diabetic Retinopathy Grading")
    st.caption("Upload a retinal fundus image to grade DR severity (0-4) with "
               "Grad-CAM explanation. Research demo - not for clinical use.")

    checkpoints = list_checkpoints()
    if not checkpoints:
        st.error("No trained models found in outputs/. Train a model first.")
        return

    with st.sidebar:
        st.header("Model")
        default = "effnet_b4_ce" if "effnet_b4_ce" in checkpoints else list(checkpoints)[0]
        name = st.selectbox("Checkpoint", list(checkpoints),
                            index=list(checkpoints).index(default))
        st.caption(f"Device: **{DEVICE}**")

    model, cfg = st.cache_resource(load_model)(checkpoints[name])

    uploaded = st.file_uploader("Fundus image", type=["png", "jpg", "jpeg"])
    if uploaded is None:
        st.info("Upload a fundus photograph to begin.")
        return

    data = np.frombuffer(uploaded.read(), np.uint8)
    image_rgb = cv2.cvtColor(cv2.imdecode(data, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    display_img, tensor = preprocess(image_rgb, cfg)
    probs = predict(model, tensor)
    pred = int(probs.argmax())
    referable, p_ref = referable_decision(probs)

    c1, c2, c3 = st.columns(3)
    c1.image(image_rgb, caption="Original", use_container_width=True)
    c2.image(display_img, caption="Preprocessed (crop + CLAHE + Ben Graham)",
             use_container_width=True)
    overlay = make_gradcam(model, cfg, tensor, display_img, pred)
    if overlay is not None:
        c3.image(overlay, caption="Grad-CAM (regions driving the prediction)",
                 use_container_width=True)
    else:
        c3.info("Grad-CAM not available for pure ViT backbones.")

    st.subheader(f"Prediction: Grade {pred} — {CLASS_NAMES[pred]}  ({probs[pred]*100:.1f}%)")
    if referable:
        st.error(f"⚠ Referable DR (grade ≥ {REFERABLE_THRESHOLD}) — "
                 f"refer to ophthalmologist. P(referable) = {p_ref*100:.1f}%")
    else:
        st.success(f"✓ Non-referable (grade < {REFERABLE_THRESHOLD}). "
                   f"P(referable) = {p_ref*100:.1f}%")

    st.bar_chart(pd.DataFrame({"probability": probs}, index=CLASS_NAMES))


if __name__ == "__main__":
    main()
