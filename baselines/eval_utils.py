"""Evaluation utilities — unified metrics for all models and experiments.

Owner: Alex Pellet
All experiment evaluation must go through compute_metrics() for consistency.
"""

import math

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# ── Prediction-health thresholds ───────────────────────────────────────────────
# `pred_collapsed` means the model carries NO usable ranking signal — its ROC-AUC
# is at (or below) chance. That is a genuine failure. It is deliberately NOT the
# same as "predicts one class at a 0.5 threshold": a DP model routinely outputs a
# tight band of probabilities that all fall on one side of 0.5 (low pred_std) while
# still ranking students correctly (AUC well above chance). That is a calibration /
# threshold problem, fixed by `tune_threshold`, not a dead model — so it is surfaced
# as the separate, non-fatal `one_class_at_threshold` / `pred_low_variance` flags.
AUC_COLLAPSE_FLOOR = 0.55
"""At or below this ROC-AUC the model cannot rank — genuine collapse."""

PRED_STD_FLOOR = 0.05
"""Below this std of predicted probabilities the output band is very tight
(diagnostic only — expected under DP noise, not a failure on its own)."""

PRED_MINORITY_FRAC_FLOOR = 0.01
"""If the minority *predicted* class holds < this fraction of predictions at the
threshold, the model decides a single class at that threshold (diagnostic only)."""


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
            pred_collapsed — 1.0 only if the model carries no ranking signal
                             (ROC-AUC <= AUC_COLLAPSE_FLOOR); the true failure flag.
            one_class_at_threshold — 1.0 if every prediction lands on one side of
                             `threshold` (all-positive or all-negative). Diagnostic:
                             usually a threshold/calibration issue, not a dead model.
            pred_low_variance — 1.0 if pred_std < PRED_STD_FLOOR. Diagnostic: a tight
                             probability band, expected under DP noise.

    Example:
        metrics = compute_metrics(y_true, y_pred_prob)
        print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
        if metrics["pred_collapsed"]:
            print("WARNING: model has no ranking signal (AUC at chance)")
        elif metrics["one_class_at_threshold"]:
            print("Ranks fine but the 0.5 cutoff is off — see tune_threshold()")
    """
    y_pred = (y_pred_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    both_classes = len(np.unique(y_true)) > 1
    auc = float(roc_auc_score(y_true, y_pred_prob)) if both_classes else float("nan")
    ap = float(average_precision_score(y_true, y_pred_prob)) if both_classes else float("nan")

    pred_std = float(np.std(y_pred_prob))
    pred_pos_rate = float(np.mean(y_pred))
    minority_frac = min(pred_pos_rate, 1.0 - pred_pos_rate)
    # Collapse = no usable ranking signal. Low variance / one-class-at-threshold are
    # kept as separate, non-fatal diagnostics (a DP model hits them while ranking fine).
    collapsed = (not both_classes) or math.isnan(auc) or auc <= AUC_COLLAPSE_FLOOR

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
        "one_class_at_threshold": float(minority_frac < PRED_MINORITY_FRAC_FLOOR),
        "pred_low_variance": float(pred_std < PRED_STD_FLOOR),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def tune_threshold(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    objective: str = "f1",
    target_recall: float | None = None,
) -> float:
    """Pick a decision threshold from (validation) predictions.

    The default 0.5 cutoff is meaningless for a miscalibrated model: DP-SGD shrinks
    logits toward zero, so a model that ranks students correctly can still put every
    probability below 0.5 and flag nobody. This chooses a data-driven operating point
    on a held-out (validation) split, which the caller then applies to test.

    Args:
        y_true: Ground-truth binary labels, shape (N,).
        y_pred_prob: Predicted probabilities, shape (N,).
        objective: "f1" maximizes binary F1; "youden" maximizes TPR - FPR (balanced
            error); "recall" returns the highest-precision threshold whose recall is
            still >= ``target_recall``.
        target_recall: Required when ``objective == "recall"``.

    Returns:
        A threshold in [0, 1]. Falls back to the median probability when the scores
        are degenerate (single class, or all-equal probabilities).

    Raises:
        ValueError: If ``objective`` is unknown, or "recall" without ``target_recall``.
    """
    fallback = float(np.median(y_pred_prob))
    if len(np.unique(y_true)) < 2 or np.ptp(y_pred_prob) == 0.0:
        return fallback

    if objective == "youden":
        fpr, tpr, thr = roc_curve(y_true, y_pred_prob)
        best = thr[int(np.argmax(tpr - fpr))]
        return fallback if not math.isfinite(best) else float(best)

    precision, recall, thr = precision_recall_curve(y_true, y_pred_prob)
    if thr.size == 0:
        return fallback

    if objective == "f1":
        f1 = np.divide(
            2 * precision * recall, precision + recall,
            out=np.zeros_like(precision), where=(precision + recall) > 0,
        )
        return float(thr[int(np.argmax(f1[:-1]))])

    if objective == "recall":
        if target_recall is None:
            raise ValueError("objective='recall' requires target_recall")
        feasible = recall[:-1] >= target_recall
        return float(thr[feasible][-1]) if feasible.any() else float(thr[0])

    raise ValueError(f"unknown objective {objective!r} (use 'f1', 'youden', or 'recall')")


def precision_at_k(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    k_frac: float = 0.10,
) -> float:
    """Precision among the top ``k_frac`` of students by predicted risk.

    The deployment view of an early-warning model: an advisor can only reach a
    fixed number of students, so what matters is how many of the highest-risk
    flagged are genuinely at risk.

    This depends only on the RANKING, not on calibration, which is why it
    survives DP. DP-SGD shrinks logits toward zero, so a model that still
    ranks students correctly can put every probability below 0.5 and flag
    nobody at the default cutoff while remaining perfectly usable as a list.

    Args:
        y_true: Binary ground-truth labels.
        y_pred_prob: Predicted positive probabilities (or any monotone risk score).
        k_frac: Fraction of the cohort to flag, in (0, 1]. Default 0.10.

    Returns:
        Precision within the flagged group. Compare against the base rate: a
        model with no ranking signal scores roughly the prevalence.

    Raises:
        ValueError: If ``k_frac`` is outside (0, 1] or the inputs differ in length.
    """
    if not 0.0 < k_frac <= 1.0:
        raise ValueError(f"k_frac must be in (0, 1], got {k_frac}")
    y_true = np.asarray(y_true)
    y_pred_prob = np.asarray(y_pred_prob)
    if y_true.shape != y_pred_prob.shape:
        raise ValueError(
            f"length mismatch: y_true {y_true.shape} vs y_pred_prob {y_pred_prob.shape}"
        )

    # At least one student, so a small cohort cannot divide by zero.
    k = max(1, int(round(k_frac * len(y_true))))
    top = np.argsort(-y_pred_prob, kind="stable")[:k]
    return float(np.mean(y_true[top]))


def recall_at_precision(
    y_true: np.ndarray, y_pred_prob: np.ndarray, min_precision: float = 0.6
) -> float:
    """Best recall achievable while holding precision >= ``min_precision``.

    Answers the deployment question directly: "if advisors will only accept flagged
    students at >= min_precision, what fraction of real dropouts do we still catch?"
    A random ranker maxes out at recall≈1 only when min_precision <= base_rate, so
    this rewards genuine ranking lift.

    Args:
        y_true: Ground-truth binary labels, shape (N,).
        y_pred_prob: Predicted probabilities, shape (N,).
        min_precision: Precision floor the operating point must satisfy.

    Returns:
        The maximum recall over thresholds meeting the precision floor, or 0.0 if none.
    """
    if len(np.unique(y_true)) < 2:
        return 0.0
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    meets = precision >= min_precision
    return float(recall[meets].max()) if meets.any() else 0.0


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
                f"  *** COLLAPSED: no ranking signal — "
                f"AUC <= {AUC_COLLAPSE_FLOOR} (chance) ***"
            )
        elif metrics.get("one_class_at_threshold"):
            print(
                "  (note) ranks fine but all predictions fall one side of the "
                "threshold — retune with tune_threshold()"
            )
