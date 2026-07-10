"""Evaluation utilities — unified metrics for all models and experiments.

Owner: Alex Pellet
All experiment evaluation must go through compute_metrics() for consistency.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute the full evaluation metric suite for dropout-risk prediction.

    Args:
        y_true: Ground-truth binary labels, shape (N,). Values in {0, 1}.
        y_pred_prob: Predicted probabilities P(dropout risk), shape (N,). Values in [0, 1].
        threshold: Classification threshold. Default 0.5.

    Returns:
        Dict with keys: auc_roc, accuracy, f1_macro, f1_binary, precision,
                        recall, brier_score, avg_precision, tn, fp, fn, tp.

    Example:
        metrics = compute_metrics(y_true, y_pred_prob)
        print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
    """
    y_pred = (y_pred_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    both_classes = len(np.unique(y_true)) > 1
    auc = float(roc_auc_score(y_true, y_pred_prob)) if both_classes else float("nan")
    ap = float(average_precision_score(y_true, y_pred_prob)) if both_classes else float("nan")


    return {
        "auc_roc": auc,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_binary": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, y_pred_prob)),
        "avg_precision": ap,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def print_metrics_table(metrics: dict[str, float], label: str = "") -> None:
    """Pretty-print a metrics dict to stdout.

    Args:
        metrics: Output of compute_metrics().
        label: Optional model/config label to print as header.
    """
    header = f"=== {label} ===" if label else "=== Metrics ==="
    print(header)
    for k in ["auc_roc", "accuracy", "f1_macro", "precision", "recall", "brier_score"]:
        print(f"  {k:<18}: {metrics[k]:.4f}")
    print(
        f"  {'confusion matrix':<18}: TN={metrics['tn']} FP={metrics['fp']} "
        f"FN={metrics['fn']} TP={metrics['tp']}"
    )
