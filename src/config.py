"""Central configuration for the DR grading project.

All paths and default hyper-parameters are defined here to ensure experiments are
reproducible and to eliminate hardcoded values throughout the codebase.
Per-experiment overrides are provided via YAML files in ``configs/`` and merged
on top of these defaults using ``load_config``.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import yaml

# --------------------------------------------------------------------------- #
# File System Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "aptos2019"          # Raw data from Kaggle
TRAIN_IMAGES_DIR = DATA_DIR / "train_images"            # Fundus images (*.png)
TRAIN_CSV = DATA_DIR / "train.csv"                      # Meta-data (id_code, diagnosis)
SPLIT_CSV = PROJECT_ROOT / "data" / "splits.csv"        # Stratified train/val/test splits
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"     # Cached pre-rendered images
OUTPUT_DIR = PROJECT_ROOT / "outputs"                   # Checkpoints, logs, and figures

# --------------------------------------------------------------------------- #
# Task definition (APTOS / APAC 5-class DR severity scale)
# --------------------------------------------------------------------------- #
NUM_CLASSES = 5
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
# Binary "referable DR" threshold: {0, 1} -> non-referable, {2, 3, 4} -> referable
REFERABLE_THRESHOLD = 2


@dataclass
class TrainConfig:
    """Data class containing all hyperparameters for a training run."""

    # Experiment metadata
    name: str = "effnet_b4_baseline"
    backbone: str = "tf_efficientnet_b4"     # Timm model name
    image_size: int = 380                    # Input resolution (380 for B4, 224 for ViT/Swin)

    # Data splitting
    seed: int = 42
    val_frac: float = 0.15
    test_frac: float = 0.15
    num_workers: int = 6

    # Preprocessing pipeline
    use_ben_graham: bool = True
    use_clahe: bool = True
    # If set, read pre-rendered images from data/processed/<cache_variant>/
    # to skip expensive on-the-fly transformations (e.g., crop/CLAHE).
    cache_variant: str | None = "full"

    # Optimization settings
    batch_size: int = 16
    epochs: int = 25
    lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 2
    loss: str = "focal"                      # Options: {"ce", "weighted_ce", "focal"}
    focal_gamma: float = 2.0
    label_smoothing: float = 0.0
    mixed_precision: bool = True             # Automatic Mixed Precision (AMP)

    # Miscellaneous
    early_stop_patience: int = 6
    monitor_metric: str = "qwk"              # Metric used for checkpoint selection

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return dataclasses.asdict(self)


def load_config(yaml_path: str | Path | None = None) -> TrainConfig:
    """Loads default configuration and optionally overrides it with a YAML file.

    """
    cfg = TrainConfig()
    if yaml_path is not None:
        with open(yaml_path, "r", encoding="utf-8") as fh:
            overrides = yaml.safe_load(fh) or {}
        unknown = set(overrides) - {f.name for f in dataclasses.fields(cfg)}
        if unknown:
            raise ValueError(f"Unknown config keys in {yaml_path}: {sorted(unknown)}")
        cfg = dataclasses.replace(cfg, **overrides)
    return cfg