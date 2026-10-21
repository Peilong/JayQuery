"""Label-DP unit tests — randomized response on labels (ICC C1, third channel).

Run: pytest tests/test_label_dp.py -v

The label channel differs from the other two in ways that are easy to get
wrong, so these tests pin the differences down:

  * It is **pure epsilon-DP** (delta = 0), not (eps, delta) Gaussian.
  * It is **one-shot**: labels are randomized once after PSI and reused every
    round, so epsilon does NOT grow with training steps. Re-randomizing per
    round would compose and inflate the real budget.
  * Its neighboring relation is **label-adjacent** (one student's label
    changes), not example-adjacent. Composing it into the Gaussian
    accountant's number would report a quantity that bounds nothing.
"""

import math

import pytest
import torch


@pytest.mark.dp
class TestKeepProbability:
    """p = e^eps / (1 + e^eps): the probability a label survives unflipped."""

    def test_known_values(self) -> None:
        from privacy.label_dp import keep_probability

        # eps = ln(3) -> e^eps = 3 -> p = 3/4
        assert keep_probability(math.log(3)) == pytest.approx(0.75)
        # eps = 0 -> p = 1/2: the output is independent of the true label
        assert keep_probability(0.0) == pytest.approx(0.5)

    def test_monotone_in_epsilon(self) -> None:
        from privacy.label_dp import keep_probability

        ps = [keep_probability(e) for e in (0.1, 0.5, 1.0, 2.0, 4.0, 8.0)]
        assert ps == sorted(ps)
        assert all(0.5 <= p < 1.0 for p in ps)

    def test_large_epsilon_approaches_no_noise(self) -> None:
        from privacy.label_dp import keep_probability

        assert keep_probability(20.0) > 0.999

    def test_rejects_negative_epsilon(self) -> None:
        from privacy.label_dp import keep_probability

        with pytest.raises(ValueError, match="epsilon"):
            keep_probability(-1.0)


@pytest.mark.dp
class TestRandomizedResponse:
    """The mechanism itself: flip each binary label independently."""

    def test_shape_and_values_preserved(self) -> None:
        from privacy.label_dp import randomized_response

        labels = torch.tensor([0.0, 1.0, 1.0, 0.0, 1.0])
        out = randomized_response(labels, epsilon=1.0, seed=0)
        assert out.shape == labels.shape
        assert set(out.unique().tolist()) <= {0.0, 1.0}, "output must stay binary"
        assert out.dtype == labels.dtype

    def test_does_not_mutate_input(self) -> None:
        """The label holder keeps its true labels for evaluation."""
        from privacy.label_dp import randomized_response

        labels = torch.tensor([0.0, 1.0, 1.0, 0.0])
        before = labels.clone()
        randomized_response(labels, epsilon=1.0, seed=0)
        assert torch.equal(labels, before), "true labels were mutated in place"

    def test_deterministic_given_seed(self) -> None:
        """Labels are randomized ONCE and reused; a silo restarting mid-run
        must reproduce the same noisy labels or epsilon silently doubles."""
        from privacy.label_dp import randomized_response

        labels = torch.randint(0, 2, (500,)).float()
        a = randomized_response(labels, epsilon=1.0, seed=42)
        b = randomized_response(labels, epsilon=1.0, seed=42)
        assert torch.equal(a, b)

    def test_different_seeds_differ(self) -> None:
        from privacy.label_dp import randomized_response

        labels = torch.randint(0, 2, (500,)).float()
        a = randomized_response(labels, epsilon=1.0, seed=1)
        b = randomized_response(labels, epsilon=1.0, seed=2)
        assert not torch.equal(a, b)

    def test_flip_rate_matches_epsilon(self) -> None:
        """Empirical flip rate must track 1 - p, or the accounted epsilon is
        not the epsilon actually spent."""
        from privacy.label_dp import keep_probability, randomized_response

        labels = torch.randint(0, 2, (20_000,)).float()
        for eps in (0.5, 1.0, 4.0):
            out = randomized_response(labels, epsilon=eps, seed=7)
            flipped = (out != labels).float().mean().item()
            expected = 1.0 - keep_probability(eps)
            assert flipped == pytest.approx(expected, abs=0.02), (
                f"eps={eps}: flipped {flipped:.3f}, expected ~{expected:.3f}"
            )

    def test_tiny_epsilon_destroys_signal(self) -> None:
        """At eps -> 0 the noisy labels are ~independent of the truth."""
        from privacy.label_dp import randomized_response

        labels = torch.randint(0, 2, (20_000,)).float()
        out = randomized_response(labels, epsilon=1e-6, seed=3)
        agreement = (out == labels).float().mean().item()
        assert agreement == pytest.approx(0.5, abs=0.02)

    def test_large_epsilon_preserves_labels(self) -> None:
        from privacy.label_dp import randomized_response

        labels = torch.randint(0, 2, (5_000,)).float()
        out = randomized_response(labels, epsilon=15.0, seed=3)
        assert (out == labels).float().mean().item() > 0.99

    def test_rejects_non_binary_labels(self) -> None:
        """RR is defined for binary labels; silently accepting a 0/1/2 vector
        would produce garbage with a confident-looking epsilon attached."""
        from privacy.label_dp import randomized_response

        with pytest.raises(ValueError, match="binary"):
            randomized_response(torch.tensor([0.0, 1.0, 2.0]), epsilon=1.0, seed=0)


@pytest.mark.dp
class TestForwardCorrection:
    """Train on noisy labels but correct the likelihood for the known flip rate."""

    def test_no_noise_is_plain_sigmoid(self) -> None:
        from privacy.label_dp import corrected_positive_prob

        logits = torch.tensor([-2.0, 0.0, 2.0])
        out = corrected_positive_prob(logits, keep_prob=1.0)
        assert torch.allclose(out, torch.sigmoid(logits), atol=1e-6)

    def test_maximum_noise_is_uninformative(self) -> None:
        """At p = 0.5 the observed label carries no information, so the
        corrected probability is 0.5 whatever the model predicts."""
        from privacy.label_dp import corrected_positive_prob

        logits = torch.tensor([-5.0, 0.0, 5.0])
        out = corrected_positive_prob(logits, keep_prob=0.5)
        assert torch.allclose(out, torch.full_like(out, 0.5), atol=1e-6)

    def test_output_is_a_valid_probability(self) -> None:
        from privacy.label_dp import corrected_positive_prob

        logits = torch.linspace(-20, 20, 101)
        for p in (0.5, 0.6, 0.75, 0.9, 1.0):
            out = corrected_positive_prob(logits, keep_prob=p)
            assert torch.all((out >= 0.0) & (out <= 1.0)), f"p={p}"

    def test_shrinks_toward_half_as_noise_grows(self) -> None:
        from privacy.label_dp import corrected_positive_prob

        logits = torch.tensor([3.0])
        clean = corrected_positive_prob(logits, keep_prob=1.0).item()
        noisy = corrected_positive_prob(logits, keep_prob=0.75).item()
        assert 0.5 < noisy < clean

    def test_correction_recovers_the_clean_optimum(self) -> None:
        """THE utility invariant. Fitting a bias term to RR-corrupted labels
        with the correction must recover the TRUE base rate; fitting without
        it converges to the corrupted rate instead."""
        from privacy.label_dp import (
            corrected_positive_prob,
            keep_probability,
            randomized_response,
        )

        torch.manual_seed(0)
        true_rate = 0.30
        n = 40_000
        labels = (torch.rand(n) < true_rate).float()
        eps = 1.0
        p = keep_probability(eps)
        noisy = randomized_response(labels, epsilon=eps, seed=11)

        def fit(corrected: bool) -> float:
            bias = torch.zeros(1, requires_grad=True)
            opt = torch.optim.Adam([bias], lr=0.05)
            for _ in range(600):
                opt.zero_grad()
                logits = bias.expand(n)
                if corrected:
                    prob = corrected_positive_prob(logits, keep_prob=p)
                    loss = torch.nn.functional.binary_cross_entropy(prob, noisy)
                else:
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        logits, noisy
                    )
                loss.backward()
                opt.step()
            return torch.sigmoid(bias.detach()).item()

        corrected_rate = fit(corrected=True)
        naive_rate = fit(corrected=False)
        observed_rate = noisy.mean().item()

        assert corrected_rate == pytest.approx(true_rate, abs=0.02), (
            f"corrected fit {corrected_rate:.3f} should recover {true_rate}"
        )
        assert naive_rate == pytest.approx(observed_rate, abs=0.02), (
            "uncorrected fit should converge to the corrupted rate"
        )
        assert abs(corrected_rate - true_rate) < abs(naive_rate - true_rate)


@pytest.mark.dp
class TestLabelRRAccountant:
    """One-shot pure-epsilon accounting, deliberately NOT Gaussian-composable."""

    def test_reports_its_epsilon(self) -> None:
        from privacy.label_dp import LabelRRAccountant

        acct = LabelRRAccountant(epsilon=2.0)
        assert acct.get_epsilon() == pytest.approx(2.0)
        assert acct.delta == 0.0, "randomized response is pure epsilon-DP"

    def test_epsilon_does_not_grow_with_rounds(self) -> None:
        """THE one-shot invariant. Labels are randomized once and reused, so
        200 rounds of training spend exactly the same label budget as one. An
        accountant that advanced per round would overstate the cost and, if
        someone 'fixed' it by re-randomizing, would understate the real one."""
        from privacy.label_dp import LabelRRAccountant

        acct = LabelRRAccountant(epsilon=2.0)
        before = acct.get_epsilon()
        for _ in range(200):
            acct.step()
        assert acct.get_epsilon() == pytest.approx(before)

    def test_neighboring_relation_is_stated(self) -> None:
        """Table I reports each channel's relation; it must come from the code,
        not from a caption typed by hand."""
        from privacy.label_dp import LabelRRAccountant

        assert "label" in LabelRRAccountant(epsilon=1.0).neighboring_relation.lower()

    def test_state_roundtrip(self) -> None:
        from privacy.label_dp import LabelRRAccountant

        acct = LabelRRAccountant(epsilon=1.5, seed=9)
        restored = LabelRRAccountant.from_dict(acct.to_dict())
        assert restored.get_epsilon() == pytest.approx(acct.get_epsilon())
        assert restored.seed == acct.seed

    def test_rejects_invalid_epsilon(self) -> None:
        from privacy.label_dp import LabelRRAccountant

        with pytest.raises(ValueError, match="epsilon"):
            LabelRRAccountant(epsilon=0.0)

    def test_keep_prob_matches_epsilon(self) -> None:
        from privacy.label_dp import LabelRRAccountant, keep_probability

        acct = LabelRRAccountant(epsilon=1.0)
        assert acct.keep_prob == pytest.approx(keep_probability(1.0))


@pytest.mark.dp
class TestGaussianCompositionGuard:
    """The label channel must never be folded into the Gaussian composition."""

    def test_combine_epsilon_rejects_a_label_accountant(self) -> None:
        """THE accounting invariant. combine_epsilon() composes subsampled
        Gaussian mechanisms over an example-adjacent relation. A label
        accountant has a different mechanism AND a different relation, so a
        combined number including it would bound nothing. Make the mistake
        impossible rather than merely documented."""
        from privacy.label_dp import LabelRRAccountant
        from privacy.privacy_accountant import PrivacyBudgetTracker, combine_epsilon

        gaussian = PrivacyBudgetTracker(
            target_epsilon=10.0, sigma=2.0, sample_rate=0.1, delta=1e-5
        )
        gaussian.steps = 5

        with pytest.raises(TypeError, match="label"):
            combine_epsilon([gaussian, LabelRRAccountant(epsilon=1.0)], delta=1e-5)

    def test_gaussian_only_composition_still_works(self) -> None:
        from privacy.privacy_accountant import PrivacyBudgetTracker, combine_epsilon

        trackers = []
        for _ in range(2):
            t = PrivacyBudgetTracker(
                target_epsilon=10.0, sigma=2.0, sample_rate=0.1, delta=1e-5
            )
            t.steps = 5
            trackers.append(t)
        assert combine_epsilon(trackers, delta=1e-5) > 0


@pytest.mark.dp
class TestTopModelLabelCorrection:
    """The server's loss must account for the known label flip rate."""

    def _model(self):
        from fl_server.top_model import TopModel

        torch.manual_seed(0)
        return TopModel(embedding_dim=4, hidden_dim=8)

    def test_defaults_to_no_correction(self) -> None:
        """Label DP off must leave the original BCE-with-logits behavior."""
        import torch.nn as nn

        top = self._model()
        logits = torch.tensor([-1.0, 0.5, 2.0])
        labels = torch.tensor([0.0, 1.0, 1.0])

        expected = nn.BCEWithLogitsLoss(pos_weight=top.pos_weight)(logits, labels)
        assert top.loss_fn(logits, labels) == pytest.approx(expected.item(), abs=1e-6)

    def test_set_label_noise_changes_the_loss(self) -> None:
        top = self._model()
        logits = torch.tensor([-1.0, 0.5, 2.0])
        labels = torch.tensor([0.0, 1.0, 1.0])

        before = top.loss_fn(logits, labels).item()
        top.set_label_noise(keep_prob=0.73)
        after = top.loss_fn(logits, labels).item()
        assert after != pytest.approx(before)

    def test_keep_prob_one_is_identical_to_off(self) -> None:
        """eps -> infinity means no flips, so the corrected loss must collapse
        back onto the uncorrected one."""
        top = self._model()
        logits = torch.tensor([-1.0, 0.5, 2.0])
        labels = torch.tensor([0.0, 1.0, 1.0])

        plain = top.loss_fn(logits, labels).item()
        top.set_label_noise(keep_prob=1.0)
        assert top.loss_fn(logits, labels).item() == pytest.approx(plain, abs=1e-5)

    def test_gradients_still_flow_to_both_silos(self) -> None:
        """The corrected loss must remain differentiable w.r.t. both inputs --
        it sits on the only path the silos learn from."""
        top = self._model()
        top.set_label_noise(keep_prob=0.73)
        emb_a = torch.randn(8, 4, requires_grad=True)
        emb_b = torch.randn(8, 4, requires_grad=True)
        labels = torch.randint(0, 2, (8,)).float()

        loss = top.loss_fn(top(emb_a, emb_b).squeeze(1), labels)
        loss.backward()

        assert emb_a.grad is not None and emb_b.grad is not None
        assert torch.isfinite(emb_a.grad).all()
        assert abs(emb_a.grad).sum() > 0

    def test_loss_is_finite_at_extreme_logits(self) -> None:
        """A saturated logit must not produce log(0) = -inf and poison the run."""
        top = self._model()
        top.set_label_noise(keep_prob=0.6)
        logits = torch.tensor([-60.0, 60.0, 0.0])
        for labels in (torch.tensor([1.0, 0.0, 1.0]), torch.tensor([0.0, 1.0, 0.0])):
            assert torch.isfinite(top.loss_fn(logits, labels)).all()

    def test_pos_weight_still_applies(self) -> None:
        """Class imbalance weighting must survive the correction."""
        top = self._model()
        top.set_label_noise(keep_prob=0.75)
        logits = torch.tensor([0.0, 0.0])
        labels = torch.tensor([1.0, 0.0])

        base = top.loss_fn(logits, labels).item()
        top.set_pos_weight(n_neg=9, n_pos=1)  # weight 9.0
        assert top.loss_fn(logits, labels).item() > base

    def test_rejects_impossible_keep_prob(self) -> None:
        top = self._model()
        with pytest.raises(ValueError, match="keep_prob"):
            top.set_label_noise(keep_prob=0.2)
