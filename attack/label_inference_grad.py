"""Label inference from the gradient the coordinator returns to a silo.

The peer silo never receives labels, only the gradient of the loss with
respect to its own cut-layer embedding. For a binary head trained with BCE,
dL/dz = sigma(z) - y, so the returned vector is a per-example scalar times a
direction shared across the batch, and that scalar changes sign with the
label. The returned gradients therefore fall into two antipodal groups and
recovering the labels is a sign problem rather than a learning problem
(Li et al. 2022).

This is the adversary the label channel is supposed to bound. The gradient is
computed from whatever labels the server holds, so with label DP enabled it
carries the randomized labels and not the true ones: the attack should recover
the released labels nearly perfectly and the true labels only at the
mechanism's keep rate.

References:
  - Li et al. 2022 (ICLR), Label leakage and protection in two-party split
    learning.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class LabelInferenceResult:
    """Outcome of one gradient-based label inference attempt.

    Attributes:
        accuracy: Fraction of records whose label was recovered.
        baseline: Majority-class rate, what a constant predictor scores.
        advantage: accuracy - baseline, the leakage above guessing the
            majority class.
        n: Records attacked.
    """

    accuracy: float
    baseline: float
    advantage: float
    n: int


def infer_labels_from_gradients(
    gradients: torch.Tensor,
    y_true: np.ndarray,
    base_rate: float,
) -> LabelInferenceResult:
    """Recover labels from returned gradients and score against ground truth.

    The gradients span one dominant direction, so their projection onto the
    first principal component separates the two labels. That projection is
    sign-ambiguous: the attacker resolves it with the base rate, which an
    institution's published statistics give it, never with the labels it is
    trying to recover.

    Args:
        gradients: Per-record gradient w.r.t. the silo's embedding, (n, d).
        y_true: Ground-truth binary labels, length n. Used only for scoring.
        base_rate: Positive-class prevalence the attacker is assumed to know.

    Returns:
        A :class:`LabelInferenceResult`.

    Raises:
        ValueError: If the gradient and label row counts disagree.
    """
    g = gradients.detach().cpu().numpy().astype(np.float64)
    y = np.asarray(y_true).astype(int)
    if len(g) != len(y):
        raise ValueError(
            f"gradients and labels must have the same number of rows, "
            f"got {len(g)} and {len(y)}"
        )

    centered = g - g.mean(axis=0, keepdims=True)
    # First right-singular vector: the direction the per-example scalar scales.
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    proj = centered @ vt[0]

    # dL/dz = sigma(z) - y is MORE NEGATIVE for positives, but the singular
    # vector's sign is arbitrary, so pick the orientation whose predicted
    # positive rate is closer to the prevalence the attacker knows.
    cand = [(proj < 0).astype(int), (proj > 0).astype(int)]
    pred = min(cand, key=lambda c: abs(c.mean() - base_rate))

    counts = np.bincount(y, minlength=2)
    baseline = float(counts.max()) / float(len(y))
    accuracy = float((pred == y).mean())
    return LabelInferenceResult(
        accuracy=accuracy,
        baseline=baseline,
        advantage=accuracy - baseline,
        n=int(len(y)),
    )
