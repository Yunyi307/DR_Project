"""Evaluation metrics for DR grading.

The headline metric for APTOS / ophthalmic grading is Quadratic Weighted Kappa
(QWK), which rewards predictions that are *close* to the true grade more than
distant ones - appropriate for an ordinal severity scale. We also report macro
accuracy and, crucially for a screening tool, per-class sensitivity (recall) and
specificity, plus a binary "referable DR" summary.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, confusion_matrix, roc_auc_score,
)

from .config import CLASS_NAMES, NUM_CLASSES, REFERABLE_THRESHOLD


@dataclass
class EvalResult:
    qwk: float
    macro_accuracy: float
    per_class_sensitivity: dict[str, float]
    per_class_specificity: dict[str, float]
    referable_sensitivity: float
    referable_specificity: float
    referable_auc: float
    confusion: np.ndarray = field(repr=False)

    def summary(self) -> str:
        lines = [
            f"QWK                    : {self.qwk:.4f}",
            f"Macro accuracy         : {self.macro_accuracy:.4f}",
            f"Referable sensitivity  : {self.referable_sensitivity:.4f}",
            f"Referable specificity  : {self.referable_specificity:.4f}",
            f"Referable AUC          : {self.referable_auc:.4f}",
            "Per-class sensitivity / specificity:",
        ]
        for name in CLASS_NAMES:
            lines.append(
                f"  {name:<14} sens={self.per_class_sensitivity[name]:.3f} "
                f"spec={self.per_class_specificity[name]:.3f}"
            )
        return "\n".join(lines)


def _per_class_sens_spec(cm: np.ndarray) -> tuple[dict, dict]:
    """Sensitivity (recall) and specificity per class from a confusion matrix."""
    sens, spec = {}, {}
    total = cm.sum()
    for i, name in enumerate(CLASS_NAMES):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        sens[name] = tp / (tp + fn) if (tp + fn) else 0.0
        spec[name] = tn / (tn + fp) if (tn + fp) else 0.0
    return sens, spec


def evaluate(y_true: np.ndarray, y_pred: np.ndarray,
             y_prob: np.ndarray | None = None) -> EvalResult:
    """Compute the full metric suite.

    Args:
        y_true: int array of ground-truth grades, shape (N,).
        y_pred: int array of predicted grades, shape (N,).
        y_prob: optional float array of class probabilities, shape (N, 5), used
            for the referable-DR AUC. If ``None``, AUC is reported as NaN.
    """
    labels = list(range(NUM_CLASSES))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    sens, spec = _per_class_sens_spec(cm)

    # Binary referable-DR view: grade >= REFERABLE_THRESHOLD is "referable".
    ref_true = (y_true >= REFERABLE_THRESHOLD).astype(int)
    ref_pred = (y_pred >= REFERABLE_THRESHOLD).astype(int)
    tp = int(((ref_pred == 1) & (ref_true == 1)).sum())
    fn = int(((ref_pred == 0) & (ref_true == 1)).sum())
    fp = int(((ref_pred == 1) & (ref_true == 0)).sum())
    tn = int(((ref_pred == 0) & (ref_true == 0)).sum())
    ref_sens = tp / (tp + fn) if (tp + fn) else 0.0
    ref_spec = tn / (tn + fp) if (tn + fp) else 0.0

    if y_prob is not None and len(np.unique(ref_true)) == 2:
        ref_score = y_prob[:, REFERABLE_THRESHOLD:].sum(axis=1)
        ref_auc = float(roc_auc_score(ref_true, ref_score))
    else:
        ref_auc = float("nan")

    return EvalResult(
        qwk=float(cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=labels)),
        macro_accuracy=float(accuracy_score(y_true, y_pred)),
        per_class_sensitivity=sens,
        per_class_specificity=spec,
        referable_sensitivity=ref_sens,
        referable_specificity=ref_spec,
        referable_auc=ref_auc,
        confusion=cm,
    )
