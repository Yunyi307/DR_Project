"""Train a DR grading model with 5-Fold Cross-Validation Ensemble."""
"""Usage:
   python train_5fold.py --config configs/effnet_b4.yaml
"""


import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import DataLoader

from src.config import OUTPUT_DIR, load_config
from src.dataset import APTOSDataset, load_split, class_counts
from src.engine import evaluate_loader, train_one_epoch, infer
from src.losses import build_loss
from src.models import build_model, count_parameters
from src.metrics import evaluate


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the 5-fold cross-validation training script."""
    p = argparse.ArgumentParser(description="Train a DR grading model with 5-Fold Ensemble")
    p.add_argument("--config", type=str, default=None, help="YAML config path")
    p.add_argument("--name", type=str, default=None, help="Override experiment name")
    p.add_argument("--backbone", type=str, default=None, help="Backbone model architecture")
    p.add_argument("--loss", type=str, default=None, choices=["ce", "weighted_ce", "focal", "ordinal"])
    p.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    p.add_argument("--batch-size", type=int, default=None, help="Batch size")
    p.add_argument("--n-splits", type=int, default=5, help="Number of folds for cross-validation")
    return p.parse_args()


def build_fold_loaders(df_train_val: pd.DataFrame, df_test: pd.DataFrame, cfg, train_idx, val_idx):
    """Build Train / Val / Test Dataloaders for a single fold based on dataset indices."""
    train_sub = df_train_val.iloc[train_idx].reset_index(drop=True)
    val_sub = df_train_val.iloc[val_idx].reset_index(drop=True)

    train_ds = APTOSDataset(train_sub, cfg, train=True)
    val_ds = APTOSDataset(val_sub, cfg, train=False)
    test_ds = APTOSDataset(df_test, cfg, train=False)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=False
    )
    return train_loader, val_loader, test_loader, class_counts(train_sub)


def train_single_fold(fold: int, cfg, train_loader, val_loader, counts, device, run_dir: Path):
    """Train a single fold model and save the best checkpoint as best_fold_{fold}.pt."""
    print(f"\n=================== FOLD {fold + 1}/{cfg.n_splits} ===================")
    model = build_model(cfg.backbone).to(device)
    print(f"Model '{cfg.backbone}' (Fold {fold + 1}): {count_parameters(model):,} trainable params")

    criterion = build_loss(
        cfg.loss, class_counts=counts.to(device),
        focal_gamma=cfg.focal_gamma, label_smoothing=cfg.label_smoothing
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = torch.amp.GradScaler("cuda") if (cfg.mixed_precision and device == "cuda") else None

    best_metric, best_epoch, patience = -1.0, -1, 0
    history = []
    ckpt_path = run_dir / f"best_fold_{fold}.pt"

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, epoch)
        scheduler.step()
        val = evaluate_loader(model, val_loader, device)
        score = getattr(val, "qwk" if cfg.monitor_metric == "qwk" else "macro_accuracy")

        history.append({
            "fold": fold, "epoch": epoch, "train_loss": train_loss,
            "val_qwk": val.qwk, "val_macro_acc": val.macro_accuracy
        })
        print(f"[Fold {fold + 1} | {epoch:02d}/{cfg.epochs}] loss={train_loss:.4f} "
              f"val_qwk={val.qwk:.4f} val_acc={val.macro_accuracy:.4f} ({time.time() - t0:.0f}s)")

        if score > best_metric:
            best_metric, best_epoch, patience = score, epoch, 0
            torch.save({
                "model": model.state_dict(), "config": cfg.to_dict(),
                "fold": fold, "epoch": epoch, "val_metric": score
            }, ckpt_path)
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                print(
                    f"Early stopping Fold {fold + 1} at epoch {epoch} (best {cfg.monitor_metric}={best_metric:.4f} @ epoch {best_epoch})")
                break

    (run_dir / f"history_fold_{fold}.json").write_text(json.dumps(history, indent=2))
    return ckpt_path


def main() -> None:
    """Main execution entry point for 5-fold cross-validation training and soft voting ensemble inference."""
    args = parse_args()
    cfg = load_config(args.config)
    for k in ("name", "backbone", "loss", "epochs"):
        v = getattr(args, k)
        if v is not None:
            setattr(cfg, k, v)
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    cfg.n_splits = args.n_splits

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = OUTPUT_DIR / f"{cfg.name}_5fold"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
    print(f"Experiment '{cfg.name}' (5-Fold Ensemble) on {device}")

    # 1. Prepare and combine datasets (combine train+val for cross-validation, keep test set independent)
    df_all = load_split()
    df_train_val = df_all[df_all["split"].isin(["train", "val"])].reset_index(drop=True)
    df_test = df_all[df_all["split"] == "test"].reset_index(drop=True)

    skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)

    # 2. Sequentially train 5 individual fold models
    fold_ckpt_paths = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(df_train_val, df_train_val["diagnosis"])):
        train_loader, val_loader, test_loader, counts = build_fold_loaders(
            df_train_val, df_test, cfg, train_idx, val_idx
        )
        ckpt_path = train_single_fold(fold, cfg, train_loader, val_loader, counts, device, run_dir)
        fold_ckpt_paths.append(ckpt_path)

    # 3. Perform 5-fold Soft Voting ensemble inference
    print("\n================= RUNNING 5-FOLD ENSEMBLE INFERENCE =================")
    all_y_probs = []
    y_true_test = None

    for fold, ckpt_path in enumerate(fold_ckpt_paths):
        model = build_model(cfg.backbone).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])

        y_true, _, y_prob = infer(model, test_loader, device)
        all_y_probs.append(y_prob)
        if y_true_test is None:
            y_true_test = y_true
        print(f"--> Fold {fold + 1} Test Inference Completed.")

    # 4. Compute probability arithmetic mean (Soft Voting: P_ensemble = \frac{1}{K} \sum_{k=1}^K P_k)
    ensemble_prob = np.mean(all_y_probs, axis=0)
    ensemble_pred = np.argmax(ensemble_prob, axis=1)

    # 5. Output evaluation metrics using native evaluation module and save predictions
    test_eval = evaluate(y_true_test, ensemble_pred, ensemble_prob)
    np.savez(run_dir / "predictions_5fold.npz", y_true=y_true_test, y_prob=ensemble_prob)

    print("\n===== 5-FOLD ENSEMBLE TEST SET RESULTS =====")
    print(test_eval.summary())
    (run_dir / "test_metrics_5fold.json").write_text(json.dumps({
        "qwk": test_eval.qwk,
        "macro_accuracy": test_eval.macro_accuracy,
        "referable_sensitivity": test_eval.referable_sensitivity,
        "referable_specificity": test_eval.referable_specificity,
        "referable_auc": test_eval.referable_auc,
        "per_class_sensitivity": test_eval.per_class_sensitivity,
        "per_class_specificity": test_eval.per_class_specificity,
        "confusion": test_eval.confusion.tolist(),
    }, indent=2))
    print(f"\nSaved 5-Fold ensemble results to {run_dir}")


if __name__ == "__main__":
    main()