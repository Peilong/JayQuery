"""Noise-matched learned inversion of the defended cut-layer channel.

This models the adversary the threat model actually names: an
honest-but-curious coordinator. It sees the embeddings the silos send -- after
clipping and Gaussian noise -- holds auxiliary records whose protected
attribute it knows, and trains an inverse mapping

    noised embedding  ->  protected attribute

on that auxiliary data, then applies it to the embeddings it intercepts.

How this differs from :mod:`attack.inversion`, and why both exist:

  * ``attack.inversion`` is an optimization attack (iDLG-style) that HOLDS the
    silo's trained bottom model and searches for an input reproducing a target
    embedding. That grants the attacker more than the threat model does; it is
    an upper bound on leakage.
  * This module grants only what the coordinator really has. It is the number
    that answers "does cut-layer noise protect features in deployment?"

The load-bearing detail is that the attacker's noise pipeline is the SAME one
the client applies -- it calls :mod:`privacy.noise_injection` rather than
reimplementing it. An inverse trained on clean embeddings and scored on noised
ones measures the train/test mismatch, not the protection, and would show
attack success falling with epsilon for entirely the wrong reason.

References:
  - He et al. 2019 (ACSAC) -- model inversion against collaborative inference
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from privacy.noise_injection import add_gaussian_noise, clip_embedding_rows


@dataclass
class InversionResult:
    """Outcome of one inversion attempt against one configuration.

    Attributes:
        accuracy: Fraction of target records whose attribute was recovered.
        baseline: Majority-class rate on the target set -- what an attacker
            achieves by always guessing the most common value. Accuracy alone
            is meaningless on a skewed attribute.
        lift: accuracy - baseline. This is the leakage.
        n_target: Number of records attacked.
        sigma: Noise multiplier the attack was calibrated to.
        n_observations: Noisy views averaged per record.
    """

    accuracy: float
    baseline: float
    lift: float
    n_target: int
    sigma: float
    n_observations: int


def noised_embeddings(
    bottom_model: nn.Module,
    X: torch.Tensor,
    max_norm: float,
    sigma: float,
    seed: int,
    n_observations: int = 1,
) -> torch.Tensor:
    """Push features through the bottom model and the client's noise pipeline.

    Clip-then-noise, in that order, via :mod:`privacy.noise_injection` -- the
    same calls ``fl_client.common.generate_embeddings`` makes. Reversing the
    order would not enforce the sensitivity bound the accounting assumes.

    Args:
        bottom_model: The silo's trained bottom model.
        X: Feature matrix, shape (n, input_dim).
        max_norm: Per-row L2 clip, the mechanism's sensitivity.
        sigma: Noise multiplier; noise std is ``sigma * max_norm``.
        seed: Seed for the noise draw.
        n_observations: Noisy views to average per record. A coordinator sees
            each student every round, so averaging is available to it; >1
            models that denoising.

    Returns:
        Embeddings of shape (n, embedding_dim), averaged over observations.
    """
    bottom_model.eval()
    with torch.no_grad():
        clean = bottom_model(X)
    clipped = clip_embedding_rows(clean, max_norm)

    torch.manual_seed(seed)
    noise_std = sigma * max_norm
    acc = torch.zeros_like(clipped)
    for _ in range(max(1, n_observations)):
        acc += add_gaussian_noise(clipped, noise_std=noise_std)
    return acc / max(1, n_observations)


class _InverseNet(nn.Module):
    """Small MLP mapping an embedding back to the protected attribute."""

    def __init__(self, embedding_dim: int, n_classes: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Class logits for each embedding."""
        return self.net(x)


def train_inverse_model(
    embeddings: torch.Tensor,
    targets: torch.Tensor,
    n_classes: int,
    epochs: int = 120,
    lr: float = 1e-3,
    hidden_dim: int = 64,
    seed: int = 0,
) -> nn.Module:
    """Fit the attacker's inverse mapping on auxiliary data.

    Args:
        embeddings: Noised embeddings of the auxiliary records.
        targets: Known attribute values for those records.
        n_classes: Number of attribute values.
        epochs: Training epochs.
        lr: Adam learning rate.
        hidden_dim: Width of the inverse network.
        seed: Seed for initialization.

    Returns:
        The trained inverse model, in eval mode.
    """
    torch.manual_seed(seed)
    model = _InverseNet(embeddings.shape[1], n_classes, hidden_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss_fn(model(embeddings), targets).backward()
        opt.step()
    model.eval()
    return model


def _overlaps(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Whether any row of ``a`` also appears in ``b``."""
    if a.shape[1:] != b.shape[1:]:
        return False
    rows_b = {r.numpy().tobytes() for r in b}
    return any(r.numpy().tobytes() in rows_b for r in a)


def run_learned_inversion(
    bottom_model: nn.Module,
    X_aux: torch.Tensor,
    y_aux: torch.Tensor,
    X_target: torch.Tensor,
    y_target: torch.Tensor,
    n_classes: int,
    max_norm: float,
    sigma: float,
    n_observations: int = 1,
    epochs: int = 120,
    seed: int = 0,
) -> InversionResult:
    """Train the inverse on auxiliary records, then attack the target ones.

    Args:
        bottom_model: The silo's trained bottom model (used only to produce
            embeddings; the attacker never sees its weights).
        X_aux, y_aux: Auxiliary records and their known attribute values.
        X_target, y_target: The records under attack and their ground truth,
            used only for scoring.
        n_classes: Number of attribute values.
        max_norm: Per-row clip used by the silo.
        sigma: Noise multiplier used by the silo.
        n_observations: Noisy views averaged per record.
        epochs: Inverse-model training epochs.
        seed: Seed for noise and initialization.

    Returns:
        An :class:`InversionResult`.

    Raises:
        ValueError: If the auxiliary and target sets overlap. Scoring an
            inverse on records it trained on reports memorization as leakage.
    """
    if _overlaps(X_target, X_aux):
        raise ValueError(
            "auxiliary and target records overlap; they must be disjoint, or "
            "the attack scores memorization rather than leakage"
        )

    emb_aux = noised_embeddings(
        bottom_model, X_aux, max_norm, sigma, seed=seed, n_observations=n_observations
    )
    inverse = train_inverse_model(
        emb_aux, y_aux, n_classes=n_classes, epochs=epochs, seed=seed
    )

    emb_tgt = noised_embeddings(
        bottom_model, X_target, max_norm, sigma,
        seed=seed + 1, n_observations=n_observations,
    )
    with torch.no_grad():
        predicted = inverse(emb_tgt).argmax(dim=1)

    accuracy = float((predicted == y_target).float().mean())
    counts = torch.bincount(y_target, minlength=n_classes)
    baseline = float(counts.max()) / float(len(y_target))

    return InversionResult(
        accuracy=accuracy,
        baseline=baseline,
        lift=accuracy - baseline,
        n_target=int(len(y_target)),
        sigma=sigma,
        n_observations=max(1, n_observations),
    )


def noise_params_from_meta(
    checkpoint_dir: str | Path, channel: str = "silo_a"
) -> tuple[float, int]:
    """Read a run's REALIZED noise parameters from its meta.json.

    The configured horizon is not what a run spends when it stops early, so
    calibrating a post-hoc attack from the config would mis-size the noise for
    exactly those runs. ``privacy_primitives`` records what was actually used.

    Args:
        checkpoint_dir: Directory containing the run's meta.json.
        channel: Which channel's primitives to read ("silo_a", "silo_b",
            "server").

    Returns:
        ``(sigma, realized_steps)``.

    Raises:
        KeyError: If the run recorded no primitives for that channel.
    """
    meta = json.loads((Path(checkpoint_dir) / "meta.json").read_text())
    prims = (meta.get("privacy_primitives") or {}).get(channel)
    if not prims:
        raise KeyError(
            f"no privacy_primitives for channel {channel!r} in "
            f"{checkpoint_dir}/meta.json — the attack cannot be calibrated to "
            f"a run whose realized noise was not recorded."
        )
    return float(prims["sigma"]), int(prims["steps"])
