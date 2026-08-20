"""Train a DR grading model. Entry point for experiments E1-E4.

Usage:
    python train.py                         # EfficientNet-B4 default
    python train.py --loss ce --name effnet_ce   # quick per-flag overrides
"""

from __future__ import annotations
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import time
from pathlib import Path

import torch

from src.config import OUTPUT_DIR, load_config
from src.dataset import build_dataloaders
from src.engine import evaluate_loader, train_one_epoch, infer
from src.losses import build_loss
from src.models import build_model, count_parameters
from src.metrics import evaluate
import numpy as np


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the training script."""
    p = argparse.ArgumentParser(description="Train a DR grading model")
    p.add_argument("--config", type=str, default=None, help="YAML config path")
    p.add_argument("--name", type=str, default=None, help="Override experiment name")
    p.add_argument("--backbone", type=str, default=None, help="Backbone model architecture")
    p.add_argument("--loss", type=str, default=None, choices=["ce", "weighted_ce", "focal", "ordinal"])
    p.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    p.add_argument("--batch-size", type=int, default=None, help="Batch size")
    return p.parse_args()


def main() -> None:
    """Main execution entry point for single-run model training."""
    args = parse_args()
    cfg = load_config(args.config)
    # CLI flags override YAML/defaults.
    for k in ("name", "backbone", "loss", "epochs"):
        v = getattr(args, k)
        if v is not None:
            setattr(cfg, k, v)
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = OUTPUT_DIR / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
    print(f"Experiment '{cfg.name}' on {device}\n{json.dumps(cfg.to_dict(), indent=2)}")

    train_loader, val_loader, test_loader, counts = build_dataloaders(cfg)
    print(f"Train class counts: {counts.tolist()}")

    model = build_model(cfg.backbone).to(device)
    print(f"Model '{cfg.backbone}': {count_parameters(model):,} trainable params")

    criterion = build_loss(cfg.loss, class_counts=counts.to(device),
                           focal_gamma=cfg.focal_gamma,
                           label_smoothing=cfg.label_smoothing).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = torch.amp.GradScaler("cuda") if (cfg.mixed_precision and device == "cuda") else None

    best_metric, best_epoch, patience = -1.0, -1, 0
    history = []
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer,
                                     device, scaler, epoch)
        scheduler.step()
        val = evaluate_loader(model, val_loader, device)
        score = getattr(val, "qwk" if cfg.monitor_metric == "qwk" else "macro_accuracy")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_qwk": val.qwk,
                        "val_macro_acc": val.macro_accuracy})
        print(f"[{epoch:02d}/{cfg.epochs}] loss={train_loss:.4f} "
              f"val_qwk={val.qwk:.4f} val_acc={val.macro_accuracy:.4f} "
              f"({time.time()-t0:.0f}s)")

        if score > best_metric:
            best_metric, best_epoch, patience = score, epoch, 0
            torch.save({"model": model.state_dict(), "config": cfg.to_dict(),
                        "epoch": epoch, "val_metric": score}, run_dir / "best.pt")
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                print(f"Early stopping at epoch {epoch} (best {cfg.monitor_metric}="
                      f"{best_metric:.4f} @ epoch {best_epoch})")
                break

    (run_dir / "history.json").write_text(json.dumps(history, indent=2))

    # Final test-set evaluation using the best checkpoint.
    ckpt = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])

    # Use infer to get raw prediction data
    y_true, y_pred, y_prob = infer(model, test_loader, device)
    test = evaluate(y_true, y_pred, y_prob)

    # Save raw prediction data for make_figures.py to generate ROC curves
    np.savez(run_dir / "predictions.npz", y_true=y_true, y_prob=y_prob)

    test = evaluate_loader(model, test_loader, device)
    print("\n===== TEST SET =====")
    print(test.summary())
    (run_dir / "test_metrics.json").write_text(json.dumps({
        "qwk": test.qwk, "macro_accuracy": test.macro_accuracy,
        "referable_sensitivity": test.referable_sensitivity,
        "referable_specificity": test.referable_specificity,
        "referable_auc": test.referable_auc,
        "per_class_sensitivity": test.per_class_sensitivity,
        "per_class_specificity": test.per_class_specificity,
        "confusion": test.confusion.tolist(),
    }, indent=2))
    print(f"\nSaved results to {run_dir}")


if __name__ == "__main__":
    main()