"""Tests for the loss-threshold membership inference attack (ICC C2, Row 10).

Run: pytest tests/test_membership_inference.py -v

The second adversary in the per-adversary evaluation. Where embedding
inversion asks "can the coordinator read a student's features off the wire?",
this asks "can whoever receives the trained model tell whether a given student
was in its training set?" -- which is the exposure server DP-SGD is supposed
to bound (Table I).

The attack is Yeom et al.'s: a training member tends to have lower loss than a
non-member, so thresholding the per-example loss separates them. Reported
threshold-free as AUC, because an attack scored at a threshold tuned on the
evaluation set reports its own tuning as leakage.
"""

import pytest
import torch


def _models(emb_dim: int = 6, in_a: int = 8, in_b: int = 10):
    from fl_server.top_model import TopModel
    from silo_a.bottom_model_a import BottomModelA
    from silo_b.bottom_model_b import BottomModelB

    torch.manual_seed(0)
    a = BottomModelA(input_dim=in_a, embedding_dim=emb_dim, dropout_rate=0.0, hidden_dim=16)
    b = BottomModelB(input_dim=in_b, embedding_dim=emb_dim, dropout_rate=0.0, hidden_dim=16)
    top = TopModel(embedding_dim=emb_dim, dropout_rate=0.0, hidden_dim=16)
    for m in (a, b, top):
        m.eval()
    return a, b, top


def _split(n: int, in_a: int = 8, in_b: int = 10, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    Xa = torch.randn(n, in_a, generator=g)
    Xb = torch.randn(n, in_b, generator=g)
    y = (torch.rand(n, generator=g) < 0.45).float()
    return Xa, Xb, y


def _overfit(models, data, epochs: int = 400):
    """Drive the model to memorize `data` so members become separable."""
    a, b, top = models
    Xa, Xb, y = data
    for m in (a, b, top):
        m.train()
    opt = torch.optim.Adam(
        list(a.parameters()) + list(b.parameters()) + list(top.parameters()), lr=5e-3
    )
    for _ in range(epochs):
        opt.zero_grad()
        logits = top(a(Xa), b(Xb)).squeeze(1)
        torch.nn.functional.binary_cross_entropy_with_logits(logits, y).backward()
        opt.step()
    for m in (a, b, top):
        m.eval()
    return a, b, top


@pytest.mark.slow
class TestPerExampleLosses:
    """The signal the attack thresholds."""

    def test_returns_one_loss_per_record(self) -> None:
        from attack.membership_inference import per_example_losses

        a, b, top = _models()
        Xa, Xb, y = _split(32)
        losses = per_example_losses(top, a, b, Xa, Xb, y)
        assert losses.shape == (32,)
        assert torch.isfinite(losses).all()

    def test_losses_are_nonnegative(self) -> None:
        from attack.membership_inference import per_example_losses

        a, b, top = _models()
        Xa, Xb, y = _split(32)
        assert (per_example_losses(top, a, b, Xa, Xb, y) >= 0).all()

    def test_confident_correct_predictions_have_lower_loss(self) -> None:
        """Direction sanity: the whole attack rests on this ordering."""
        from attack.membership_inference import per_example_losses

        a, b, top = _models()
        Xa, Xb, _ = _split(64)
        with torch.no_grad():
            probs = torch.sigmoid(top(a(Xa), b(Xb)).squeeze(1))
        agree = (probs > 0.5).float()      # labels the model already predicts
        disagree = 1.0 - agree
        assert per_example_losses(top, a, b, Xa, Xb, agree).mean() < \
               per_example_losses(top, a, b, Xa, Xb, disagree).mean()


@pytest.mark.slow
class TestLossThresholdAttack:
    """Membership inference against a released model."""

    def test_untrained_model_leaks_nothing(self) -> None:
        """Members and non-members drawn from the same distribution through an
        untrained model must be indistinguishable: AUC at chance."""
        from attack.membership_inference import run_loss_threshold_mia

        a, b, top = _models()
        res = run_loss_threshold_mia(top, a, b, _split(300, seed=1), _split(300, seed=2))
        assert res.auc == pytest.approx(0.5, abs=0.07), (
            f"found membership signal in an untrained model (AUC {res.auc:.3f})"
        )

    def test_overfit_model_leaks(self) -> None:
        """THE attack. A model driven to memorize its training set must be
        detectably vulnerable, or the attack cannot measure protection."""
        from attack.membership_inference import run_loss_threshold_mia

        members = _split(200, seed=1)
        nonmembers = _split(200, seed=2)
        a, b, top = _overfit(_models(), members)

        res = run_loss_threshold_mia(top, a, b, members, nonmembers)
        assert res.auc > 0.60, (
            f"memorized training data was not detectable (AUC {res.auc:.3f}) — "
            f"the attack has no power and cannot certify any defense"
        )
        assert res.advantage > 0.15

    def test_auc_above_half_means_members_have_lower_loss(self) -> None:
        """Sign check. With the comparison inverted an overfit model scores
        AUC < 0.5; reporting |AUC - 0.5| would hide that as success."""
        from attack.membership_inference import per_example_losses, run_loss_threshold_mia

        members = _split(200, seed=1)
        nonmembers = _split(200, seed=2)
        a, b, top = _overfit(_models(), members)

        m_loss = per_example_losses(top, a, b, *members).mean()
        n_loss = per_example_losses(top, a, b, *nonmembers).mean()
        res = run_loss_threshold_mia(top, a, b, members, nonmembers)

        assert m_loss < n_loss, "overfitting did not lower member loss"
        assert res.auc > 0.5, "AUC is inverted relative to the loss ordering"

    def test_reports_chance_baseline_and_counts(self) -> None:
        from attack.membership_inference import run_loss_threshold_mia

        a, b, top = _models()
        res = run_loss_threshold_mia(top, a, b, _split(120, seed=1), _split(80, seed=2))
        assert res.baseline == 0.5
        assert res.n_members == 120
        assert res.n_nonmembers == 80
        assert 0.0 <= res.auc <= 1.0

    def test_advantage_is_zero_without_leakage(self) -> None:
        """Yeom advantage = max(TPR - FPR); a non-leaking model sits near 0."""
        from attack.membership_inference import run_loss_threshold_mia

        a, b, top = _models()
        res = run_loss_threshold_mia(top, a, b, _split(300, seed=1), _split(300, seed=2))
        assert res.advantage < 0.15

    def test_handles_imbalanced_sets(self) -> None:
        """AUC must not drift with the member/non-member ratio the way raw
        accuracy would."""
        from attack.membership_inference import run_loss_threshold_mia

        members = _split(200, seed=1)
        a, b, top = _overfit(_models(), members)
        balanced = run_loss_threshold_mia(top, a, b, members, _split(200, seed=2))
        skewed = run_loss_threshold_mia(top, a, b, members, _split(40, seed=2))
        assert abs(balanced.auc - skewed.auc) < 0.12

    def test_rejects_empty_input(self) -> None:
        from attack.membership_inference import run_loss_threshold_mia

        a, b, top = _models()
        empty = (torch.zeros(0, 8), torch.zeros(0, 10), torch.zeros(0))
        with pytest.raises(ValueError, match="empty|non-empty"):
            run_loss_threshold_mia(top, a, b, empty, _split(10))

    def test_accuracy_is_balanced(self) -> None:
        """Balanced accuracy satisfies acc = 0.5 + advantage/2 exactly. Raw
        TPR does not, and would report a defense as weaker than it is on a
        skewed member/non-member split."""
        from attack.membership_inference import run_loss_threshold_mia

        members = _split(200, seed=1)
        a, b, top = _overfit(_models(), members)
        for nonmembers in (_split(200, seed=2), _split(40, seed=2)):
            res = run_loss_threshold_mia(top, a, b, members, nonmembers)
            assert res.accuracy == pytest.approx(0.5 + res.advantage / 2, abs=1e-6)

    def test_accuracy_near_chance_without_leakage(self) -> None:
        from attack.membership_inference import run_loss_threshold_mia

        a, b, top = _models()
        res = run_loss_threshold_mia(top, a, b, _split(300, seed=1), _split(300, seed=2))
        assert res.accuracy == pytest.approx(0.5, abs=0.08)
