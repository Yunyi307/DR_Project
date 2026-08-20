"""Loss functions for class-imbalanced DR grading.

APTOS is heavily skewed towards grade 0 (No DR); grades 3 (Severe) and 4 (PDR)
are rare but clinically the most important not to miss. We therefore support:

  * ``ce``           - plain cross-entropy (baseline).
  * ``weighted_ce``  - cross-entropy weighted by inverse class frequency.
  * ``focal``        - Focal Loss (Lin et al., 2017), down-weighting easy
                       examples so training focuses on hard/rare grades.
  * ``ordinal``      - Combined CE and MSE to penalize distant misclassifications,
                       optimizing for QWK in ordinal grading tasks.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class FocalLoss(nn.Module):
    """Multi-class Focal Loss with optional per-class alpha weighting."""

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha if alpha is not None else None)
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # ce per-sample without reduction; pt = model prob of the true class
        ce = F.cross_entropy(
            logits, target, weight=self.alpha,
            reduction="none", label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce)
        loss = (1.0 - pt) ** self.gamma * ce
        return loss.mean()


class OrdinalLoss(nn.Module):
    """Combines Cross Entropy with a distance-based penalty (MSE) to optimize QWK."""

    def __init__(self, alpha: torch.Tensor | None = None, mse_weight: float = 0.5,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=alpha, label_smoothing=label_smoothing)
        self.mse_weight = mse_weight
        # Pre-define the class value matrix [0, 1, 2, 3, 4]
        self.register_buffer("classes", torch.arange(5, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 1. Base classification loss
        loss_ce = self.ce(logits, target)

        # 2. Distance penalty (regression loss, greatly optimizing QWK)
        probs = F.softmax(logits, dim=1)

        # [Key fix]: Dynamically get the device of logits (e.g., cuda:0) and move classes there
        device_classes = self.classes.to(logits.device)

        # Calculate the expected predicted grade
        expected_pred = torch.sum(probs * device_classes, dim=1)

        # Calculate the mean squared error between the expected prediction and the true grade
        loss_mse = F.mse_loss(expected_pred, target.float())

        return loss_ce + self.mse_weight * loss_mse


def class_weights_from_counts(counts: torch.Tensor) -> torch.Tensor:
    """Inverse-frequency weights, normalised to mean 1 for stable learning rates."""
    counts = counts.float().clamp(min=1.0)
    w = counts.sum() / (len(counts) * counts)
    return w / w.mean()


def build_loss(name: str, class_counts: torch.Tensor | None = None,
               focal_gamma: float = 2.0, label_smoothing: float = 0.0) -> nn.Module:
    """Factory mapping a config string to a loss module."""
    alpha = None
    # Include ordinal in the list of losses that can use inverse-frequency weights
    if class_counts is not None and name in {"weighted_ce", "focal", "ordinal"}:
        alpha = class_weights_from_counts(class_counts)

    if name == "ce":
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if name == "weighted_ce":
        return nn.CrossEntropyLoss(weight=alpha, label_smoothing=label_smoothing)
    if name == "focal":
        return FocalLoss(gamma=focal_gamma, alpha=alpha, label_smoothing=label_smoothing)
    if name == "ordinal":
        return OrdinalLoss(alpha=alpha, mse_weight=0.5, label_smoothing=label_smoothing)

    raise ValueError(f"Unknown loss '{name}' (expected ce|weighted_ce|focal|ordinal)")