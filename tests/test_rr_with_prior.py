"""Tests for RRWithPrior adapted to split VFL (Ghazi et al. 2021).

Run: pytest tests/test_rr_with_prior.py -v

Plain randomized response spends epsilon flipping labels blindly. RRWithPrior
spends it only where the prior is uncertain. For BINARY labels the mechanism
collapses to a clean rule -- let q = e^eps/(e^eps + 1) be the RR keep rate and
p1 the prior's top probability:

  * p1 > q : emit the prior's argmax, IGNORING the true label. The output is
             independent of the label, so this costs nothing and is more
             accurate than flipping.
  * p1 <= q: fall back to plain RR.

The VFL twist: the label holder has only 18 of the 44 features, so its own
prior is weak. The COORDINATOR holds the joint model and already sends
predicted probabilities to the holder for scoring. Those predictions are the
prior. They are computed from other students' already-randomized labels, never
from this student's label, which is what keeps the guarantee intact.
"""

import math

import pytest
import torch


def _q(eps: float) -> float:
    return math.exp(eps) / (math.exp(eps) + 1.0)


@pytest.mark.dp
class TestRRWithPriorBinary:
    """The k*=1 / k*=2 rule and its consequences."""

    def test_uniform_prior_recovers_plain_rr(self) -> None:
        """With no information the mechanism must not do better than RR --
        that would mean epsilon was not actually being spent."""
        from privacy.label_dp import randomized_response, rr_with_prior

        labels = torch.randint(0, 2, (20_000,)).float()
        prior = torch.full((20_000, 2), 0.5)
        out = rr_with_prior(labels, prior, epsilon=1.0, seed=3)
        plain = randomized_response(labels, epsilon=1.0, seed=3)

        agree_prior = (out == labels).float().mean().item()
        agree_plain = (plain == labels).float().mean().item()
        assert agree_prior == pytest.approx(agree_plain, abs=0.02)

    def test_confident_prior_emits_argmax_and_ignores_the_label(self) -> None:
        """THE mechanism. When the prior beats the RR keep rate the output no
        longer depends on the true label at all, so no budget is spent on it."""
        from privacy.label_dp import rr_with_prior

        n = 2000
        prior = torch.zeros(n, 2)
        prior[:, 1] = 0.97           # confident the label is 1
        prior[:, 0] = 0.03
        all_ones = torch.ones(n)
        all_zeros = torch.zeros(n)

        out_a = rr_with_prior(all_ones, prior, epsilon=1.0, seed=7)
        out_b = rr_with_prior(all_zeros, prior, epsilon=1.0, seed=7)

        assert torch.equal(out_a, out_b), (
            "output changed with the true label despite a confident prior — "
            "the k*=1 branch is still leaking"
        )
        assert (out_a == 1).all(), "did not emit the prior's argmax"

    def test_informative_prior_beats_plain_rr_on_utility(self) -> None:
        """The whole point: more usable labels at the SAME epsilon."""
        from privacy.label_dp import randomized_response, rr_with_prior

        torch.manual_seed(0)
        n = 20_000
        labels = (torch.rand(n) < 0.46).float()
        # a decent but imperfect prior, as a trained joint model would give
        conf = torch.where(labels == 1, 0.85, 0.15) + torch.randn(n) * 0.05
        conf = conf.clamp(0.02, 0.98)
        prior = torch.stack([1 - conf, conf], dim=1)

        eps = 1.0
        smart = (rr_with_prior(labels, prior, epsilon=eps, seed=1) == labels).float().mean()
        plain = (randomized_response(labels, epsilon=eps, seed=1) == labels).float().mean()
        assert smart > plain + 0.05, (
            f"prior bought nothing: {smart:.3f} vs plain RR {plain:.3f}"
        )

    def test_falls_back_to_rr_when_prior_is_weak(self) -> None:
        """A barely-informative prior must not trigger the argmax branch, or
        the mechanism would emit confident garbage."""
        from privacy.label_dp import rr_with_prior

        n = 4000
        eps = 2.0
        weak = _q(eps) - 0.05            # just below the threshold
        prior = torch.stack([torch.full((n,), 1 - weak), torch.full((n,), weak)], dim=1)
        labels = (torch.rand(n) < 0.5).float()

        out = rr_with_prior(labels, prior, epsilon=eps, seed=2)
        agree = (out == labels).float().mean().item()
        # RR would keep ~q; argmax-of-prior would agree ~50%. Expect RR-like.
        assert agree == pytest.approx(_q(eps), abs=0.03)

    def test_threshold_moves_with_epsilon(self) -> None:
        """At large epsilon RR is already accurate, so the prior has to be very
        confident before ignoring the label wins."""
        from privacy.label_dp import rr_with_prior

        n = 3000
        labels = torch.ones(n)
        flipped = torch.zeros(n)
        prior = torch.zeros(n, 2)
        prior[:, 1] = 0.80
        prior[:, 0] = 0.20

        # eps=0.5 -> q=0.62 < 0.80, so argmax branch: label-independent
        lo_a = rr_with_prior(labels, prior, epsilon=0.5, seed=1)
        lo_b = rr_with_prior(flipped, prior, epsilon=0.5, seed=1)
        assert torch.equal(lo_a, lo_b)

        # eps=4 -> q=0.982 > 0.80, so RR branch: output must track the label
        hi_a = rr_with_prior(labels, prior, epsilon=4.0, seed=1)
        hi_b = rr_with_prior(flipped, prior, epsilon=4.0, seed=1)
        assert not torch.equal(hi_a, hi_b)

    def test_output_stays_binary_and_shaped(self) -> None:
        from privacy.label_dp import rr_with_prior

        labels = torch.randint(0, 2, (256,)).float()
        prior = torch.rand(256, 2)
        prior = prior / prior.sum(dim=1, keepdim=True)
        out = rr_with_prior(labels, prior, epsilon=1.0, seed=0)
        assert out.shape == labels.shape
        assert set(out.unique().tolist()) <= {0.0, 1.0}

    def test_deterministic_given_seed(self) -> None:
        """One-shot reuse still requires reproducibility across restarts."""
        from privacy.label_dp import rr_with_prior

        labels = torch.randint(0, 2, (500,)).float()
        prior = torch.rand(500, 2)
        prior = prior / prior.sum(1, keepdim=True)
        a = rr_with_prior(labels, prior, epsilon=1.0, seed=11)
        b = rr_with_prior(labels, prior, epsilon=1.0, seed=11)
        assert torch.equal(a, b)

    def test_rejects_malformed_prior(self) -> None:
        from privacy.label_dp import rr_with_prior

        labels = torch.randint(0, 2, (10,)).float()
        with pytest.raises(ValueError, match="prior"):
            rr_with_prior(labels, torch.rand(10, 3), epsilon=1.0, seed=0)
        with pytest.raises(ValueError, match="prior"):
            rr_with_prior(labels, torch.rand(5, 2), epsilon=1.0, seed=0)
