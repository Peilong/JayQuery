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

# ── Prediction-health thresholds ───────────────────────────────────────────────
# A healthy classifier on this task spreads its probabilities out (pred_std
# typically lands in 0.15-0.30). A model that has collapsed to predicting one
# class outputs a near-constant probability, so pred_std ~ 0. These two named
# thresholds define the `pred_collapsed` flag; they are diagnostics, not tuned
# hyperparameters.
PRED_STD_FLOOR = 0.05
"""Below this std of predicted probabilities, output is effectively constant."""

PRED_MINORITY_FRAC_FLOOR = 0.01
"""If the minority *predicted* class holds < this fraction of predictions at the
threshold, the model is predicting (essentially) a single class."""


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
                        recall, brier_score, avg_precision, tn, fp, fn, tp,
                        plus prediction-health keys:
            pred_std       — std of y_pred_prob. ~0 means constant output
                             (one-class collapse). Healthy: >= PRED_STD_FLOOR.
            pred_mean      — mean of y_pred_prob. Should sit near base_rate for
                             a calibrated model.
            pred_pos_rate  — fraction predicted positive at `threshold`.
                             Compare to base_rate; 0.0 or 1.0 = one-class.
            base_rate      — actual positive fraction of y_true.
            pred_collapsed — 1.0 if pred_std < PRED_STD_FLOOR or the minority
                             predicted class holds < PRED_MINORITY_FRAC_FLOOR
                             of predictions; else 0.0. Instant failure flag.

    Example:
        metrics = compute_metrics(y_true, y_pred_prob)
        print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
        if metrics["pred_collapsed"]:
            print("WARNING: model is predicting a single class")
    """
    y_pred = (y_pred_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    both_classes = len(np.unique(y_true)) > 1
    auc = float(roc_auc_score(y_true, y_pred_prob)) if both_classes else float("nan")
    ap = float(average_precision_score(y_true, y_pred_prob)) if both_classes else float("nan")

    pred_std = float(np.std(y_pred_prob))
    pred_pos_rate = float(np.mean(y_pred))
    minority_frac = min(pred_pos_rate, 1.0 - pred_pos_rate)
    collapsed = pred_std < PRED_STD_FLOOR or minority_frac < PRED_MINORITY_FRAC_FLOOR

    return {
        "auc_roc": auc,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_binary": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, y_pred_prob)),
        "avg_precision": ap,
        "pred_std": pred_std,
        "pred_mean": float(np.mean(y_pred_prob)),
        "pred_pos_rate": pred_pos_rate,
        "base_rate": float(np.mean(y_true)),
        "pred_collapsed": float(collapsed),
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
    if "pred_std" in metrics:
        print(
            f"  {'pred health':<18}: std={metrics['pred_std']:.4f} "
            f"pos_rate={metrics['pred_pos_rate']:.4f} (base={metrics['base_rate']:.4f})"
        )
        if metrics.get("pred_collapsed"):
            print(
                f"  *** COLLAPSED: predictions are (near-)constant — "
                f"pred_std < {PRED_STD_FLOOR} or one-class output ***"
            )
