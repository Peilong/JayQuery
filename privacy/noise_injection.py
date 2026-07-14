"""Cut-layer noise for embeddings — the CLIENT-side DP mechanism.

Owner: Wyatt Wilfong

Applied in fl_client/common.py:generate_embeddings to the sent embedding only.
The noise std is sigma * max_norm, where sigma comes from
``privacy.privacy_accountant.solve_noise_multiplier`` and ``clip_embedding_rows``
enforces the sensitivity bound the accounting assumes.

References:
  - Dwork & Roth 2014 — Gaussian mechanism (Chapter 3)
"""

import torch


def clip_embedding_rows(embedding: torch.Tensor, max_norm: float) -> torch.Tensor:
    """Clip each row to L2 norm <= max_norm (enforces the DP sensitivity bound).

    Args:
        embedding: Cut-layer embedding, shape (batch_size, embedding_dim).
        max_norm: Per-row L2 bound. Must be > 0.

    Returns:
        A new clipped tensor of the same shape.
    """
    if max_norm <= 0:
        raise ValueError(f"max_norm must be > 0, got {max_norm}")
    norms = embedding.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
    return embedding * (max_norm / norms).clamp(max=1.0)


def add_gaussian_noise(embedding: torch.Tensor, noise_std: float) -> torch.Tensor:
    """Add Gaussian noise at a fixed std (sigma * sensitivity, caller-computed).

    Args:
        embedding: Cut-layer embedding tensor (clip rows first).
        noise_std: Noise standard deviation. Must be > 0.

    Returns:
        A new noised tensor of the same shape.
    """
    if noise_std <= 0:
        raise ValueError(f"noise_std must be > 0, got {noise_std}")
    return embedding + torch.randn_like(embedding) * noise_std


def add_laplace_noise_to_embedding(
    embedding: torch.Tensor,
    sensitivity: float,
    epsilon: float,
) -> torch.Tensor:
    """Laplace mechanism (pure epsilon-DP) — kept for the noise-comparison experiment.

    Args:
        embedding: Cut-layer embedding tensor.
        sensitivity: L1 sensitivity of the embedding function.
        epsilon: Privacy budget.

    Returns:
        A new noised tensor of the same shape.
    """
    laplace = torch.distributions.Laplace(loc=0.0, scale=sensitivity / epsilon)
    return embedding + laplace.sample(embedding.shape)
