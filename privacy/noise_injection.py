"""Manual Gaussian noise injection for embeddings — alternative to DP-SGD.

Owner: Wyatt Wilfong
Status: SCAFFOLD — implement add_dp_noise_to_embedding() in Week 6.

This module provides a simpler (but less tight) DP mechanism:
inject noise directly into the embedding tensor at the cut layer,
rather than using Opacus DP-SGD on the optimizer.

Used in: Experiment 4 (Gaussian vs. Laplace noise comparison).

References:
  - Dwork & Roth 2014 — Gaussian mechanism (Chapter 3)
  - CLAUDE.md DP guide Step 5
"""

import torch
import math


def add_dp_noise_to_embedding(
    embedding: torch.Tensor,
    sensitivity: float,
    epsilon: float,
    delta: float = 1e-5,
) -> tuple[torch.Tensor,float]:
    """Add calibrated Gaussian noise to an embedding tensor.

    Uses the Gaussian mechanism: noise ~ N(0, sigma^2 * I) where
    sigma = sensitivity * sqrt(2 * ln(1.25/delta)) / epsilon

    Args:
        embedding: The cut-layer embedding tensor, shape (batch_size, embedding_dim).
        sensitivity: L2 sensitivity of the embedding function (e.g., max_grad_norm).
        epsilon: Privacy budget for this noise addition.
        delta: DP delta parameter. Default 1e-5.

    Returns:
        Noisy embedding tensor of the same shape.

    Raises:
        ValueError: If epsilon <= 0 or sensitivity <= 0.

    Example:
        noisy_emb = add_dp_noise_to_embedding(embedding, sensitivity=1.0, epsilon=1.0)
    """

    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}")
    if sensitivity <= 0:
        raise ValueError(f"sensitivity must be > 0, got {sensitivity}")

    sigma = sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon
    noise = torch.randn_like(embedding) * sigma
    return embedding + noise, sigma


def add_laplace_noise_to_embedding(
    embedding: torch.Tensor,
    sensitivity: float,
    epsilon: float,
) -> torch.Tensor:
    """Add calibrated Laplace noise to an embedding tensor.

    Uses the Laplace mechanism: noise ~ Laplace(0, sensitivity/epsilon).
    Provides pure epsilon-DP (no delta term).

    Args:
        embedding: The cut-layer embedding tensor.
        sensitivity: L1 sensitivity of the embedding function.
        epsilon: Privacy budget.

    Returns:
        Noisy embedding tensor of the same shape.

    TODO (Wyatt, Week 6): implement using torch.distributions.Laplace.
    """
    laplace = torch.distributions.Laplace(loc=0.0, scale=sensitivity / epsilon)
    noise = laplace.sample(embedding.shape)
    return embedding + noise
