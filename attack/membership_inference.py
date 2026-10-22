"""Loss-threshold membership inference against the released model.

The second adversary in the per-adversary evaluation. Embedding inversion
(:mod:`attack.embedding_inversion`) asks whether the coordinator can read a
student's *features* off the wire. This asks whether whoever receives the
trained model can tell that a given student was *in its training set* -- the
exposure server DP-SGD is meant to bound.

Method (Yeom et al. 2018): a model fits its training data more closely than
held-out data, so training members tend to have lower per-example loss.
Thresholding that loss separates members from non-members.

Reported threshold-free, as ROC-AUC over the loss scores, plus the Yeom
advantage ``max(TPR - FPR)``. Scoring at a threshold tuned on the evaluation
set would report the tuning as leakage, and raw accuracy drifts with the
member/non-member ratio in a way AUC does not.

Chance is 0.5. A model that leaks nothing sits there.

Note on label DP: the attacker is assumed to know the candidate record it is
testing, including its true label, so losses are plain BCE against the labels
supplied. The randomized-response correction used during training is NOT
applied here -- that correction describes the noisy targets the server
optimized, not what an adversary holding a real record would compute.

References:
  - Yeom et al. 2018 (CSF) -- Privacy Risk in Machine Learning
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve

#: (X_silo_a, X_silo_b, labels)
Split = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass
class MembershipResult:
    """Outcome of one membership inference attempt.

    Attributes:
        auc: ROC-AUC of the loss score at separating members from
            non-members. 0.5 is chance; above 0.5 means members have lower
            loss, which is the expected direction.
        advantage: Yeom advantage, ``max(TPR - FPR)`` over all thresholds.
        accuracy: Balanced accuracy at the advantage-maximizing threshold.
            Reported for comparability with prior work; ``auc`` is the
            primary number because it needs no threshold.
        baseline: Chance level, always 0.5.
        n_members: Training records tested.
        n_nonmembers: Held-out records tested.
    """

    auc: float
    advantage: float
    accuracy: float
    baseline: float
    n_members: int
    n_nonmembers: int


def per_example_losses(
    top_model: nn.Module,
    bottom_a: nn.Module,
    bottom_b: nn.Module,
    X_a: torch.Tensor,
    X_b: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Per-record binary cross-entropy under the released model.

    Runs the full VFL forward -- both bottom models and the top model -- since
    a prediction for one student needs both silos' features.

    Args:
        top_model: The released top model.
        bottom_a: Silo A's bottom model.
        bottom_b: Silo B's bottom model.
        X_a: Silo A features, shape (n, input_dim_a).
        X_b: Silo B features, shape (n, input_dim_b).
        y: True labels, shape (n,).

    Returns:
        Losses of shape (n,), one per record. Lower means the model fits that
        record more closely.
    """
    for m in (top_model, bottom_a, bottom_b):
        m.eval()
    with torch.no_grad():
        logits = top_model(bottom_a(X_a), bottom_b(X_b)).squeeze(1)
        return F.binary_cross_entropy_with_logits(
            logits, y.float(), reduction="none"
        )


def run_loss_threshold_mia(
    top_model: nn.Module,
    bottom_a: nn.Module,
    bottom_b: nn.Module,
    members: Split,
    nonmembers: Split,
) -> MembershipResult:
    """Score how well per-example loss separates members from non-members.

    Args:
        top_model: The released top model.
        bottom_a: Silo A's bottom model.
        bottom_b: Silo B's bottom model.
        members: Records the model was trained on.
        nonmembers: Held-out records from the same distribution.

    Returns:
        A :class:`MembershipResult`.

    Raises:
        ValueError: If either split is empty.
    """
    if len(members[2]) == 0 or len(nonmembers[2]) == 0:
        raise ValueError(
            "membership inference needs non-empty member and non-member sets"
        )

    loss_m = per_example_losses(top_model, bottom_a, bottom_b, *members)
    loss_n = per_example_losses(top_model, bottom_a, bottom_b, *nonmembers)

    # Members are the positive class. Lower loss should indicate membership,
    # so the score is NEGATED loss: a higher score must mean "more likely a
    # member" for AUC to come out above 0.5 in the leaking direction.
    scores = torch.cat([-loss_m, -loss_n]).numpy()
    truth = np.concatenate([np.ones(len(loss_m)), np.zeros(len(loss_n))])

    auc = float(roc_auc_score(truth, scores))
    fpr, tpr, _ = roc_curve(truth, scores)
    idx = int(np.argmax(tpr - fpr))
    advantage = float(tpr[idx] - fpr[idx])
    # Balanced, so the number does not drift with the member/non-member ratio.
    accuracy = float((tpr[idx] + (1.0 - fpr[idx])) / 2.0)

    return MembershipResult(
        auc=auc,
        advantage=advantage,
        accuracy=accuracy,
        baseline=0.5,
        n_members=int(len(loss_m)),
        n_nonmembers=int(len(loss_n)),
    )
