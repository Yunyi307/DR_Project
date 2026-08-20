"""PyTorch dataset and dataloaders for APTOS fundus grading.

Preprocessing (crop / Ben Graham / CLAHE) is applied on the fly in ``__getitem__``
and cached in memory-mapped form is intentionally avoided to keep the pipeline
transparent for the report; if throughput becomes a bottleneck we can pre-render
processed images to disk. Training augmentation uses albumentations.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

from .config import PROCESSED_DIR, SPLIT_CSV, TRAIN_IMAGES_DIR, TrainConfig
from .preprocessing import preprocess_fundus
import cv2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(image_size: int, train: bool) -> A.Compose:
    """Augmentation for training; deterministic resize/normalise for eval."""
    if train:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=0.7, border_mode=cv2.BORDER_CONSTANT),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=0,
                               p=0.3, border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


class APTOSDataset(Dataset):
    """Reads id_code/diagnosis rows, applies preprocessing + transforms."""

    def __init__(self, df: pd.DataFrame, cfg: TrainConfig, train: bool,
                 images_dir: Path = TRAIN_IMAGES_DIR):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.transform = build_transforms(cfg.image_size, train)
        # When a cache variant is set, read pre-rendered images and skip the
        # expensive on-the-fly preprocessing (see scripts/cache_preprocess.py).
        self.use_cache = cfg.cache_variant is not None
        if self.use_cache:
            self.images_dir = PROCESSED_DIR / cfg.cache_variant
            if not self.images_dir.exists():
                raise FileNotFoundError(
                    f"Cache dir {self.images_dir} missing - run "
                    f"scripts/cache_preprocess.py --variant {cfg.cache_variant}"
                )
        else:
            self.images_dir = Path(images_dir)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = self.images_dir / f"{row['id_code']}.png"
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if not self.use_cache:
            rgb = preprocess_fundus(
                rgb, use_ben_graham=self.cfg.use_ben_graham,
                use_clahe=self.cfg.use_clahe,
            )
        img = self.transform(image=rgb)["image"]
        label = int(row["diagnosis"])
        return img, label


def load_split(split_csv: Path = SPLIT_CSV) -> pd.DataFrame:
    if not Path(split_csv).exists():
        raise FileNotFoundError(
            f"{split_csv} not found - run scripts/prepare_splits.py first."
        )
    return pd.read_csv(split_csv)


def class_counts(df: pd.DataFrame, num_classes: int = 5) -> torch.Tensor:
    counts = df["diagnosis"].value_counts().reindex(range(num_classes), fill_value=0)
    return torch.tensor(counts.values, dtype=torch.long)


def build_dataloaders(cfg: TrainConfig) -> tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """Return (train, val, test) loaders and the training-set class counts."""
    df = load_split()
    splits = {s: df[df["split"] == s] for s in ("train", "val", "test")}

    loaders = {}
    for name, sub in splits.items():
        ds = APTOSDataset(sub, cfg, train=(name == "train"))
        loaders[name] = DataLoader(
            ds, batch_size=cfg.batch_size, shuffle=(name == "train"),
            num_workers=cfg.num_workers, pin_memory=True,
            drop_last=(name == "train"),
        )
    return loaders["train"], loaders["val"], loaders["test"], class_counts(splits["train"])
