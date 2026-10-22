"""Tests for the PLD (Connect-the-Dots) accountant.

Run: pytest tests/test_pld_accountant.py -v

The RDP accountant composes a Renyi bound and converts to (eps, delta) at the
end; that conversion is lossy. Privacy-loss-distribution accounting composes
the distribution itself, so for the subsampled Gaussian it reports a smaller
epsilon for the SAME noise. Nothing about the mechanism changes -- only how
tightly its cost is measured.

This matters here because the paper reports composed epsilon across three
Gaussian channels, and a loose accountant overstates the privacy price of
every arm.
"""

import pytest


@pytest.mark.dp
class TestPLDEpsilon:
    """Composed epsilon under PLD."""

    def test_matches_rdp_order_of_magnitude(self) -> None:
        """Sanity: PLD must be in the same ballpark, not a different units bug."""
        from privacy.pld_accountant import pld_epsilon
        from privacy.privacy_accountant import PrivacyBudgetTracker

        t = PrivacyBudgetTracker(target_epsilon=100.0, sigma=2.0,
                                 sample_rate=0.05, delta=1e-5)
        t.steps = 100
        rdp = t.get_epsilon()
        pld = pld_epsilon(sigma=2.0, sample_rate=0.05, steps=100, delta=1e-5)
        assert 0.2 * rdp < pld < 1.5 * rdp, f"PLD {pld:.4f} vs RDP {rdp:.4f}"

    def test_is_tighter_than_rdp(self) -> None:
        """THE reason to adopt it: same noise, smaller reported epsilon."""
        from privacy.pld_accountant import pld_epsilon
        from privacy.privacy_accountant import PrivacyBudgetTracker

        for sigma, q, steps in ((2.0, 0.05, 100), (1.5, 0.05, 100), (3.0, 0.1, 50)):
            t = PrivacyBudgetTracker(target_epsilon=1e6, sigma=sigma,
                                     sample_rate=q, delta=1e-5)
            t.steps = steps
            rdp = t.get_epsilon()
            pld = pld_epsilon(sigma=sigma, sample_rate=q, steps=steps, delta=1e-5)
            assert pld <= rdp + 1e-9, (
                f"PLD ({pld:.4f}) should not exceed RDP ({rdp:.4f}) at "
                f"sigma={sigma}, q={q}, steps={steps}"
            )

    def test_monotone_in_steps(self) -> None:
        from privacy.pld_accountant import pld_epsilon

        eps = [pld_epsilon(sigma=2.0, sample_rate=0.05, steps=s, delta=1e-5)
               for s in (10, 50, 100, 200)]
        assert eps == sorted(eps)

    def test_decreasing_in_sigma(self) -> None:
        from privacy.pld_accountant import pld_epsilon

        eps = [pld_epsilon(sigma=s, sample_rate=0.05, steps=100, delta=1e-5)
               for s in (1.0, 2.0, 4.0, 8.0)]
        assert eps == sorted(eps, reverse=True)

    def test_zero_steps_spends_nothing(self) -> None:
        from privacy.pld_accountant import pld_epsilon

        assert pld_epsilon(sigma=2.0, sample_rate=0.05, steps=0,
                           delta=1e-5) == pytest.approx(0.0)

    def test_composes_heterogeneous_channels(self) -> None:
        """The three Gaussian channels have different sigmas and step counts;
        composing them must cost more than any one alone."""
        from privacy.pld_accountant import pld_compose

        chans = [
            {"sigma": 2.0, "sample_rate": 0.05, "steps": 100},
            {"sigma": 2.5, "sample_rate": 0.05, "steps": 100},
            {"sigma": 1.8, "sample_rate": 0.05, "steps": 100},
        ]
        combined = pld_compose(chans, delta=1e-5)
        singles = [pld_compose([c], delta=1e-5) for c in chans]
        assert combined >= max(singles) - 1e-9
        assert combined <= sum(singles) + 1e-9, "composition worse than naive sum"

    def test_empty_composition_is_zero(self) -> None:
        from privacy.pld_accountant import pld_compose

        assert pld_compose([], delta=1e-5) == pytest.approx(0.0)

    def test_rejects_a_label_channel(self) -> None:
        """Same guard as the RDP path: randomized response is pure eps-DP over
        a label-adjacent relation and must never enter this composition."""
        from privacy.label_dp import LabelRRAccountant
        from privacy.pld_accountant import pld_compose

        with pytest.raises(TypeError, match="label"):
            pld_compose([LabelRRAccountant(epsilon=1.0)], delta=1e-5)

    def test_rejects_invalid_delta(self) -> None:
        from privacy.pld_accountant import pld_epsilon

        for bad in (0.0, 1.0, -0.1):
            with pytest.raises(ValueError, match="delta"):
                pld_epsilon(sigma=2.0, sample_rate=0.05, steps=10, delta=bad)
