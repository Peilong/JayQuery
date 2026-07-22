"""Label Inference Attack — server infers Silo B labels from returned gradients.

Owner: Wyatt Wilfong
Status: IMPLEMENTED — Week 9.

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

    Uses the Li 2022 gradient-sign heuristic. The returned embedding gradient is the
    per-sample logit gradient ``(p_i - y_i)`` pushed through the top model, so its
    sign — read off relative to the batch's dominant gradient direction — leaks the
    label: samples on the negative side are the at-risk (positive) class. The sign
    convention is verified against ``fl_server/server_app.py``, which returns
    ``emb_b.grad`` un-negated.

    Args:
        grad_emb_b: Gradient tensor the server returns to Silo B, shape (batch_size, 32).
        top_model_weights: The server's Top Model (attacker has access to this).
        predicted_probs: Predicted probabilities from the Top Model, shape (batch_size,).

    Returns:
        Inferred binary labels, shape (batch_size,) with values in {0, 1}.
    """
    grad = grad_emb_b.detach()
    p = predicted_probs.detach().reshape(-1)
    emb_dim = grad.shape[1]

    # Reference direction: the way emb_b must move to raise the logit, i.e. the axis the
    # per-sample gradient (p - y) * d(logit)/d(emb_b) lies along. The attacker owns the
    # top model, so it reads this off directly via a backward pass at a neutral probe.
    was_training = top_model_weights.training
    top_model_weights.eval()
    probe_a = torch.zeros(1, emb_dim, requires_grad=True)
    probe_b = torch.zeros(1, emb_dim, requires_grad=True)
    top_model_weights(probe_a, probe_b).reshape(()).backward()
    ref = probe_b.grad.detach().reshape(-1)
    if was_training:
        top_model_weights.train()
    ref = ref / ref.norm().clamp_min(1e-12)

    # score ~ (p - y): positive aligns with y=0, negative is the at-risk class.
    scores = grad @ ref
    labels = (scores < 0).long()

    # Fall back to the prediction where the gradient carries no usable signal.
    ambiguous = scores.abs() < 1e-8
    labels[ambiguous] = (p[ambiguous] > 0.5).long()
    return labels
