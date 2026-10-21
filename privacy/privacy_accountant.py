"""Privacy budget accountant — composes cumulative epsilon with autodp.

Owner: Wyatt Wilfong

Epsilon is always recomputed from plain primitives (sigma, sample_rate, steps),
never accumulated, so tracker state survives serialization in context.state.

References:
  - Mironov 2017 (CSF) — Rényi DP composition
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

from autodp import mechanism_zoo, transformer_zoo


class PrivacyBudgetExceededError(Exception):
    """Raised when the cumulative epsilon expenditure exceeds the target budget."""


class PrivacyBudgetTracker:
    """autodp-based cumulative-epsilon accountant over plain primitives.

    ``sigma`` is the normalized noise multiplier (noise_std / sensitivity) —
    callers do the division: the server passes the DP optimizer's
    ``noise_multiplier``; the client cut layer passes ``noise_std / sensitivity``.
    ``sample_rate < 1.0`` applies Poisson-subsampling amplification (mini-batch);
    ``1.0`` uses exact Gaussian composition (full-batch).

    Args:
        target_epsilon: Budget ceiling; step() raises past it.
        sigma: Noise multiplier of the per-step Gaussian mechanism.
        sample_rate: Per-step sampling probability (batch_size / dataset_size).
        delta: DP delta parameter.
    """

    def __init__(
        self,
        target_epsilon: float,
        sigma: float = 1.0,
        sample_rate: float = 1.0,
        delta: float = 1e-5,
    ) -> None:
        if target_epsilon <= 0:
            raise ValueError(f"target_epsilon must be > 0, got {target_epsilon}")
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        if not 0.0 < sample_rate <= 1.0:
            raise ValueError(f"sample_rate must be in (0, 1], got {sample_rate}")
        if not 0.0 < delta < 1.0:
            raise ValueError(f"delta must be in (0, 1), got {delta}")
        self.target_epsilon = target_epsilon
        self.sigma = sigma
        self.sample_rate = sample_rate
        self.delta = delta
        self.steps = 0

    def step(self) -> None:
        """Advance one training step and enforce the budget.

        Raises:
            PrivacyBudgetExceededError: If cumulative epsilon exceeds target_epsilon.
        """
        self.steps += 1
        current = self.get_epsilon()
        if current > self.target_epsilon:
            raise PrivacyBudgetExceededError(
                f"Privacy budget exceeded: {current:.4f} > {self.target_epsilon}"
            )

    def get_epsilon(self) -> float:
        """Recompute cumulative epsilon from the primitives (pure — no mutation).

        Returns:
            Cumulative epsilon at ``delta`` after ``steps`` compositions.
        """
        return combine_epsilon([self], delta=self.delta)

    def would_exceed(self, n: int = 1) -> bool:
        """Whether spending ``n`` more steps would exceed the budget (no mutation)."""
        probe = PrivacyBudgetTracker(
            target_epsilon=self.target_epsilon,
            sigma=self.sigma,
            sample_rate=self.sample_rate,
            delta=self.delta,
        )
        probe.steps = self.steps + n
        return probe.get_epsilon() > self.target_epsilon

    def to_dict(self) -> dict[str, float | int]:
        """Plain-primitive state for context.state persistence."""
        return {
            "target_epsilon": self.target_epsilon,
            "sigma": self.sigma,
            "sample_rate": self.sample_rate,
            "delta": self.delta,
            "steps": self.steps,
        }

    @classmethod
    def from_dict(cls, state: dict[str, float | int]) -> PrivacyBudgetTracker:
        """Rebuild a tracker from to_dict() output."""
        tracker = cls(
            target_epsilon=float(state["target_epsilon"]),
            sigma=float(state["sigma"]),
            sample_rate=float(state["sample_rate"]),
            delta=float(state["delta"]),
        )
        tracker.steps = int(state["steps"])
        return tracker


def _per_step_mechanism(sigma: float, sample_rate: float) -> transformer_zoo.Mechanism:
    """One noisy step as an autodp mechanism, subsampling-amplified when applicable."""
    base = mechanism_zoo.GaussianMechanism(sigma=sigma)
    if sample_rate == 1.0:
        return base
    return transformer_zoo.AmplificationBySampling(PoissonSampling=True)(
        base, sample_rate, improved_bound_flag=True
    )


def combine_epsilon(trackers: Sequence[PrivacyBudgetTracker], delta: float) -> float:
    """Composed epsilon across mechanisms (e.g. client cut-layer + server DP-SGD).

    Every tracker must be a subsampled-Gaussian :class:`PrivacyBudgetTracker`
    over the SAME (example-adjacent) neighboring relation. The label channel's
    randomized response is pure epsilon-DP over a *label-adjacent* relation, so
    it is rejected here and reported separately instead -- a single number
    mixing the two would bound neither guarantee.

    Args:
        trackers: Accountants for each budget source; zero-step trackers contribute nothing.
        delta: DP delta at which the combined guarantee is stated.

    Returns:
        Epsilon of the composition of every tracker's spend, at ``delta``.

    Raises:
        ValueError: If ``delta`` is outside (0, 1).
        TypeError: If any tracker is not a ``PrivacyBudgetTracker`` -- notably a
            ``privacy.label_dp.LabelRRAccountant``.
    """
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    for t in trackers:
        if not isinstance(t, PrivacyBudgetTracker):
            raise TypeError(
                f"combine_epsilon() composes subsampled-Gaussian PrivacyBudgetTrackers "
                f"over an example-adjacent neighboring relation, but got "
                f"{type(t).__name__}. The label randomized-response channel is pure "
                f"epsilon-DP over a label-adjacent relation (see privacy/label_dp.py) "
                f"and must be REPORTED SEPARATELY, not composed into this number."
            )
    active = [t for t in trackers if t.steps > 0]
    if not active:
        return 0.0
    with warnings.catch_warnings():
        # autodp's internal Brent search trips benign scipy RuntimeWarnings.
        warnings.simplefilter("ignore", RuntimeWarning)
        if all(t.sample_rate == 1.0 for t in active):
            # Exact composition of Gaussians — tighter than the RDP bound.
            composed = transformer_zoo.ComposeGaussian().compose(
                mechanism_list=[mechanism_zoo.GaussianMechanism(sigma=t.sigma) for t in active],
                coeff_list=[t.steps for t in active],
            )
        else:
            composed = transformer_zoo.Composition().compose(
                mechanism_list=[_per_step_mechanism(t.sigma, t.sample_rate) for t in active],
                coeff_list=[t.steps for t in active],
            )
        return float(composed.get_eps(delta=delta))


def solve_noise_multiplier(
    target_epsilon: float,
    sample_rate: float,
    steps: int,
    delta: float = 1e-5,
    sigma_hi: float = 1e3,
    tol: float = 1e-3,
) -> float:
    """Smallest sigma whose autodp-composed epsilon over ``steps`` stays within budget.

    Uses the exact accounting of :class:`PrivacyBudgetTracker`, so a run of exactly
    ``steps`` steps at the returned sigma will not trip the tracker. ``get_epsilon`` is
    monotone decreasing in sigma (more noise -> less epsilon), so we bisect for the least
    noise (best utility) that is still feasible.

    Args:
        target_epsilon: Budget ceiling to stay within.
        sample_rate: Per-step sampling probability (batch_size / dataset_size).
        steps: Number of optimizer steps that will actually run.
        delta: DP delta.
        sigma_hi: Upper bracket for the search; must be feasible.
        tol: Absolute tolerance on the returned sigma.

    Returns:
        The least sigma with composed epsilon <= target_epsilon over ``steps`` steps.

    Raises:
        ValueError: If ``steps`` <= 0, or if ``sigma_hi`` is still over budget.
    """
    if steps <= 0:
        raise ValueError(f"steps must be > 0, got {steps}")

    def eps_at(sigma: float) -> float:
        tracker = PrivacyBudgetTracker(
            target_epsilon=target_epsilon,
            sigma=sigma,
            sample_rate=sample_rate,
            delta=delta,
        )
        tracker.steps = steps
        return tracker.get_epsilon()

    if eps_at(sigma_hi) > target_epsilon:
        raise ValueError(
            f"sigma_hi={sigma_hi} still exceeds budget "
            f"(eps={eps_at(sigma_hi):.4f} > {target_epsilon}); raise sigma_hi."
        )

    lo, hi = 1e-3, sigma_hi
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if eps_at(mid) > target_epsilon:
            lo = mid  # not enough noise -> need larger sigma
        else:
            hi = mid  # feasible -> try less noise
    return hi  # feasible endpoint
