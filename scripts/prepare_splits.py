"""Create a reproducible, stratified train/val/test split and print class EDA.

Writes ``data/splits.csv`` with columns: id_code, diagnosis, split.
Stratifying on ``diagnosis`` preserves the (imbalanced) class distribution across
all three splits, which matters for honest evaluation of the rare grades 3 and 4.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import (  # noqa: E402
    CLASS_NAMES, SPLIT_CSV, TRAIN_CSV, TrainConfig,
)


def main() -> None:
    cfg = TrainConfig()
    if not TRAIN_CSV.exists():
        raise FileNotFoundError(
            f"{TRAIN_CSV} not found - run scripts/download_data.py first."
        )
    df = pd.read_csv(TRAIN_CSV)

    # First carve out the test set, then split the remainder into train/val.
    train_val, test = train_test_split(
        df, test_size=cfg.test_frac, stratify=df["diagnosis"], random_state=cfg.seed,
    )
    val_ratio = cfg.val_frac / (1.0 - cfg.test_frac)
    train, val = train_test_split(
        train_val, test_size=val_ratio, stratify=train_val["diagnosis"],
        random_state=cfg.seed,
    )

    train = train.assign(split="train")
    val = val.assign(split="val")
    test = test.assign(split="test")
    out = pd.concat([train, val, test]).sort_values("id_code").reset_index(drop=True)
    SPLIT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(SPLIT_CSV, index=False)

    print(f"Wrote {SPLIT_CSV}  (n={len(out)})\n")
    print("Class distribution by split (row-normalised %):")
    dist = (
        out.groupby("split")["diagnosis"]
        .value_counts(normalize=True)
        .mul(100).round(1)
        .unstack(fill_value=0.0)
    )
    dist.columns = [f"{c}:{CLASS_NAMES[c]}" for c in dist.columns]
    print(dist.to_string())
    print("\nCounts by split:")
    print(out.groupby("split")["diagnosis"].value_counts().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
