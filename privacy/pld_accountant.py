"""Privacy-loss-distribution accounting (Connect-the-Dots).

The R\\'enyi accountant in :mod:`privacy.privacy_accountant` composes an RDP
bound and converts to (eps, delta) at the end. That conversion is lossy. PLD
accounting composes the privacy loss *distribution* itself and reads epsilon
off the result, so for the subsampled Gaussian it reports a strictly smaller
epsilon for the same noise.

Nothing about the mechanism changes. The silos clip and noise exactly as
before; only the measurement of what that costs gets tighter. Every arm's
reported privacy price drops, which matters for a paper whose headline is a
privacy/utility trade-off: a loose accountant makes DP look more expensive
than it is.

Connect-the-Dots (Doroshenko et al., PETS 2022) is the discretization used
here, via Google's ``dp_accounting``. One caveat worth knowing: PLD is not
universally tighter --- for *fixed-size* batch DP-SGD an improved RDP bound can
win. This system Poisson-subsamples, which is the regime PLD is tight in.

The label channel is excluded by construction, exactly as in the RDP path:
randomized response is pure epsilon-DP over a label-adjacent relation, and
composing it here would produce a number bounding neither guarantee.
"""

from __future__ import annotations

from collections.abc import Sequence

from dp_accounting.pld import privacy_loss_distribution as pld_lib

from privacy.privacy_accountant import PrivacyBudgetTracker

#: Discretization granularity. Smaller is tighter and slower; 1e-4 is the
#: library's default working point and is well inside our reporting precision.
_DISCRETIZATION = 1e-4


def _validate_delta(delta: float) -> None:
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must be in (0, 1), got {delta}")


def _channel_pld(sigma: float, sample_rate: float, steps: int):
    """PLD for ``steps`` rounds of one subsampled-Gaussian channel."""
    single = pld_lib.from_gaussian_mechanism(
        standard_deviation=sigma,
        sensitivity=1.0,            # sigma is already the noise multiplier
        sampling_prob=sample_rate,
        use_connect_dots=True,
        value_discretization_interval=_DISCRETIZATION,
    )
    return single.self_compose(steps)


def pld_epsilon(
    sigma: float, sample_rate: float, steps: int, delta: float = 1e-5
) -> float:
    """Epsilon for one subsampled-Gaussian channel under PLD accounting.

    Args:
        sigma: Noise multiplier (noise std / sensitivity).
        sample_rate: Poisson sampling probability per round.
        steps: Rounds actually taken.
        delta: Target delta.

    Returns:
        Epsilon at ``delta``. Zero when no steps were taken.

    Raises:
        ValueError: If ``delta`` is outside (0, 1).
    """
    _validate_delta(delta)
    if steps <= 0:
        return 0.0
    return float(_channel_pld(sigma, sample_rate, steps).get_epsilon_for_delta(delta))


def pld_compose(channels: Sequence[dict], delta: float = 1e-5) -> float:
    """Compose several subsampled-Gaussian channels under PLD.

    Args:
        channels: One mapping per channel with ``sigma``, ``sample_rate`` and
            ``steps``. A :class:`PrivacyBudgetTracker` may be passed instead and
            its primitives are read off.
        delta: Target delta for the composed guarantee.

    Returns:
        Composed epsilon at ``delta``; zero for an empty sequence.

    Raises:
        ValueError: If ``delta`` is outside (0, 1).
        TypeError: If a channel is a label accountant, whose pure epsilon-DP
            guarantee over a label-adjacent relation must be reported
            separately rather than composed here.
    """
    _validate_delta(delta)

    specs: list[dict] = []
    for ch in channels:
        if isinstance(ch, PrivacyBudgetTracker):
            specs.append(
                {"sigma": ch.sigma, "sample_rate": ch.sample_rate, "steps": ch.steps}
            )
        elif isinstance(ch, dict):
            specs.append(ch)
        else:
            raise TypeError(
                f"pld_compose() composes subsampled-Gaussian channels over an "
                f"example-adjacent neighboring relation, but got "
                f"{type(ch).__name__}. The label randomized-response channel is "
                f"pure epsilon-DP over a label-adjacent relation (see "
                f"privacy/label_dp.py) and must be REPORTED SEPARATELY."
            )

    active = [s for s in specs if int(s["steps"]) > 0]
    if not active:
        return 0.0

    composed = _channel_pld(
        float(active[0]["sigma"]),
        float(active[0]["sample_rate"]),
        int(active[0]["steps"]),
    )
    for spec in active[1:]:
        composed = composed.compose(
            _channel_pld(
                float(spec["sigma"]), float(spec["sample_rate"]), int(spec["steps"])
            )
        )
    return float(composed.get_epsilon_for_delta(delta))
