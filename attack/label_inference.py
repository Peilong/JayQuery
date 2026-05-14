"""Label Inference Attack — server infers Silo B labels from returned gradients.

Owner: Wyatt Wilfong
Status: SCAFFOLD — implement in Week 9.

Reference: Li et al. 2022 (ICLR) — "Label Leakage and Protection in Two-party Split Learning"

Attacker model: The FL server (honest-but-curious) observes the gradient signal
dL/d_emb_b that it sends back to Silo B each round. Without ever seeing Silo B's
labels, can it infer them from the gradient direction?

Key insight (Li 2022): For binary classification with BCELoss, the gradient of the
loss w.r.t. the logit is (predicted - true_label). If the server knows the predicted
probability, it can rearrange to estimate true_label.
"""

import torch
import torch.nn as nn


def infer_labels_from_gradient(
    grad_emb_b: torch.Tensor,
    top_model_weights: nn.Module,
    predicted_probs: torch.Tensor,
) -> torch.Tensor:
    """Attempt to infer Silo B binary labels from the gradient returned by the server.

    Args:
        grad_emb_b: Gradient tensor the server returns to Silo B, shape (batch_size, 32).
        top_model_weights: The server's Top Model (attacker has access to this).
        predicted_probs: Predicted probabilities from the Top Model, shape (batch_size,).

    Returns:
        Inferred binary labels, shape (batch_size,) with values in {0, 1}.

    TODO (Wyatt, Week 9): Implement using the Li 2022 gradient-sign heuristic.
        Hint: For BCEWithLogitsLoss, sign(grad) correlates with (pred - label).
              If pred > 0.5 and grad is positive → label is likely 0 (not at risk).
    """
    raise NotImplementedError("TODO (Wyatt, Week 9): implement infer_labels_from_gradient()")
