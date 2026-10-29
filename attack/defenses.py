"""Non-DP cut-layer defenses, as baselines for the DP placements.

Hashing or otherwise transforming the embedding before sending it is the
common alternative to privatizing it. Neither defense here carries a privacy
budget, so each bounds a particular attack rather than a class of
adversaries; running them against the same learned inverse the DP arms face
quantifies what that distinction is worth.

Both follow how they are proposed in the literature: a secret fixed transform
the attacker is assumed not to know, and sparsification borrowed from
gradient-compression work. An attacker who can collect (transformed
embedding, attribute) pairs trains against the transform rather than around
it, and an orthogonal map preserves every distance a learned inverse relies
on.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

#: Default sparsity for :func:`top_k_sparsify`, a quarter of a 32-wide layer.
DEFAULT_K = 8


def random_projection(embedding: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """Apply a fixed secret orthogonal map to each row.

    The silo draws one matrix at setup and reuses it, so the coordinator sees
    a consistently rotated view. Orthogonality means the top model loses
    nothing, which is why the defense is attractive and why it does not work:
    distances, and therefore everything a learned inverse uses, are preserved
    exactly.

    Args:
        embedding: Rows to transform, shape (n, d).
        seed: The silo's secret key for the draw.

    Returns:
        A new tensor of the same shape.
    """
    d = embedding.shape[1]
    gen = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(d, d, generator=gen))
    return embedding @ q


def top_k_sparsify(embedding: torch.Tensor, k: int = DEFAULT_K) -> torch.Tensor:
    """Keep each row's ``k`` largest-magnitude coordinates, zero the rest.

    Borrowed from gradient compression, where it is proposed as privacy
    protection on the grounds that most of the signal is discarded.

    Args:
        embedding: Rows to sparsify, shape (n, d).
        k: Coordinates to keep per row.

    Returns:
        A new tensor of the same shape.

    Raises:
        ValueError: If ``k`` is outside 1..d.
    """
    d = embedding.shape[1]
    if not 1 <= k <= d:
        raise ValueError(f"k must be in 1..{d}, got {k}")
    if k == d:
        return embedding.clone()
    idx = embedding.abs().topk(k, dim=1).indices
    out = torch.zeros_like(embedding)
    return out.scatter(1, idx, embedding.gather(1, idx))


#: Name -> transform, for the comparison harness. "none" is the zero point.
DEFENSES: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "none": lambda e: e,
    "random-projection": random_projection,
    "top-k": top_k_sparsify,
}
