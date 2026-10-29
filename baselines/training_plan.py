"""Resolve a baseline's training budget from the experiment config.

The baseline MLPs are the paper's architecture-matched controls, so their
optimization budget must match the VFL run's; otherwise the comparison
measures training effort rather than architecture. Both MLPs resolve their
budget here so they cannot drift apart, and ``tests/test_training_plan.py``
pins the mapping from config keys to the budget actually used.

Config keys honored: ``gradient_steps``, ``batch_fraction``,
``class_weighting``, and the legacy ``epochs`` / ``batch_size`` pair when no
step budget is given.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Accepted values for the class_weighting policy. "none" matches the VFL
#: grid, which runs pos-weight = 0.
_POLICIES = ("none", "balanced")


@dataclass(frozen=True)
class TrainingPlan:
    """A resolved training budget.

    Attributes:
        steps: Optimizer steps to take. Not epochs: the VFL run's horizon is
            counted in rounds, and one round is one top-model step.
        batch_size: Rows per step.
        pos_weight: Positive-class weight for BCEWithLogitsLoss, or None for
            no weighting (what the VFL run does).
    """

    steps: int
    batch_size: int
    pos_weight: float | None


def resolve_plan(
    cfg: dict,
    n_train: int,
    n_pos: int | None = None,
    n_neg: int | None = None,
) -> TrainingPlan:
    """Turn a config into a concrete budget.

    ``gradient_steps`` and ``batch_fraction`` mirror the VFL run directly: the
    grid takes ``num-server-rounds`` steps, each on a Poisson sample of
    ``batch-sample-rate`` of the training set. When ``gradient_steps`` is
    absent the legacy ``epochs`` x ``batch_size`` path is used, so existing
    callers keep their behavior rather than silently training for one step.

    Args:
        cfg: Config fragment. Recognizes gradient_steps, batch_fraction,
            epochs, batch_size and class_weighting.
        n_train: Training-set size, used to turn a fraction into a row count.
        n_pos: Positive-class count, required for ``class_weighting:
            balanced``.
        n_neg: Negative-class count, required for ``class_weighting:
            balanced``.

    Returns:
        The resolved :class:`TrainingPlan`.

    Raises:
        ValueError: If ``gradient_steps`` is not positive, if
            ``class_weighting`` is unrecognized, or if ``balanced`` was asked
            for without the class counts needed to compute the weight.
    """
    if "batch_fraction" in cfg:
        batch_size = max(1, min(n_train, round(float(cfg["batch_fraction"]) * n_train)))
    else:
        batch_size = int(cfg.get("batch_size", 64))

    if "gradient_steps" in cfg:
        steps = int(cfg["gradient_steps"])
        if steps <= 0:
            raise ValueError(f"gradient_steps must be > 0, got {steps}")
    else:
        # Legacy path: epochs over the full set at batch_size.
        steps = int(cfg.get("epochs", 50)) * max(1, n_train // max(1, batch_size))

    policy = str(cfg.get("class_weighting", "none")).lower()
    if policy not in _POLICIES:
        raise ValueError(
            f"class_weighting must be one of {_POLICIES}, got {policy!r}"
        )
    pos_weight: float | None = None
    if policy == "balanced":
        if not n_pos or n_neg is None:
            raise ValueError(
                "class_weighting='balanced' needs n_pos and n_neg to compute "
                "the weight; pass them or use 'none' to match the VFL run"
            )
        pos_weight = float(n_neg) / float(max(n_pos, 1))

    return TrainingPlan(steps=steps, batch_size=batch_size, pos_weight=pos_weight)
