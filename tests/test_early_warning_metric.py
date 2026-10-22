"""Tests for the early-warning operating metric (ICC Row 11).

Run: pytest tests/test_early_warning_metric.py -v

An advisor cannot contact everyone. The deployment question is: of the
students a model ranks in the top 10% of risk, what fraction are genuinely at
risk? That is precision@k, and it ties DP's score compression to practice --
DP shrinks predicted-risk spread but can preserve the ranking, in which case
a ranked list still works while a 0.5 cutoff flags nobody.
"""

import numpy as np
import pytest


@pytest.mark.model
class TestPrecisionAtK:
    """Precision among the top-k fraction by predicted risk."""

    def test_perfect_ranker_scores_one(self) -> None:
        from baselines.eval_utils import precision_at_k

        y = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        probs = y.astype(float)  # ranks the positives first
        assert precision_at_k(y, probs, k_frac=0.3) == pytest.approx(1.0)

    def test_inverse_ranker_scores_zero(self) -> None:
        from baselines.eval_utils import precision_at_k

        y = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        assert precision_at_k(y, 1.0 - y.astype(float), k_frac=0.3) == pytest.approx(0.0)

    def test_uninformative_ranker_lands_near_base_rate(self) -> None:
        """The number only means something against the base rate: a useless
        model scores roughly the prevalence, not zero."""
        from baselines.eval_utils import precision_at_k

        rng = np.random.default_rng(0)
        y = (rng.random(4000) < 0.25).astype(int)
        got = precision_at_k(y, rng.random(4000), k_frac=0.10)
        assert got == pytest.approx(0.25, abs=0.06)

    def test_k_of_one_equals_base_rate(self) -> None:
        """Selecting everyone cannot beat prevalence."""
        from baselines.eval_utils import precision_at_k

        y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 1, 0])
        rng = np.random.default_rng(1)
        assert precision_at_k(y, rng.random(10), k_frac=1.0) == pytest.approx(y.mean())

    def test_selects_at_least_one_student(self) -> None:
        """A tiny cohort must not select zero and divide by zero."""
        from baselines.eval_utils import precision_at_k

        y = np.array([1, 0, 0])
        got = precision_at_k(y, np.array([0.9, 0.2, 0.1]), k_frac=0.01)
        assert got == pytest.approx(1.0)

    def test_ranking_not_calibration_is_what_matters(self) -> None:
        """THE DP-relevant property. Squashing every probability toward 0.5
        leaves precision@k unchanged as long as the ORDER survives -- which is
        exactly the regime DP puts the model in."""
        from baselines.eval_utils import precision_at_k

        rng = np.random.default_rng(2)
        y = (rng.random(1000) < 0.3).astype(int)
        probs = np.clip(y * 0.6 + rng.random(1000) * 0.4, 0, 1)
        squashed = 0.5 + (probs - probs.mean()) * 0.001  # same order, no spread

        assert precision_at_k(y, squashed, k_frac=0.1) == pytest.approx(
            precision_at_k(y, probs, k_frac=0.1)
        )

    def test_rejects_invalid_k(self) -> None:
        from baselines.eval_utils import precision_at_k

        y = np.array([1, 0, 1, 0])
        for bad in (0.0, -0.1, 1.5):
            with pytest.raises(ValueError, match="k_frac"):
                precision_at_k(y, np.array([0.1, 0.2, 0.3, 0.4]), k_frac=bad)

    def test_rejects_length_mismatch(self) -> None:
        from baselines.eval_utils import precision_at_k

        with pytest.raises(ValueError, match="length|shape"):
            precision_at_k(np.array([1, 0, 1]), np.array([0.1, 0.2]), k_frac=0.5)
