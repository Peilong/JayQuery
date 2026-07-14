"""Differential Privacy unit tests — owned by Wyatt Wilfong.

Run: pytest tests/test_dp.py -v
"""

import pytest
import torch
import torch.nn as nn


def _make_tiny_model() -> tuple[nn.Module, torch.optim.Optimizer]:
    model = nn.Sequential(nn.Linear(12, 8), nn.ReLU(), nn.Linear(8, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return model, optimizer


# ─────────────────────────────────────────────────────────────────────────────
# DP wrapper (server top model)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.dp
def test_dp_wrapper_basic() -> None:
    """make_private_with_noise() returns a usable (dp_model, dp_optimizer) pair."""
    from opacus.grad_sample import GradSampleModule
    from opacus.optimizers import DPOptimizer

    from privacy.dp_wrapper import make_private_with_noise

    model, optimizer = _make_tiny_model()
    dp_model, dp_optimizer = make_private_with_noise(
        model=model, optimizer=optimizer,
        noise_multiplier=1.0, max_grad_norm=1.0, expected_batch_size=32,
    )
    assert isinstance(dp_model, GradSampleModule)
    assert isinstance(dp_optimizer, DPOptimizer)
    assert dp_model._module is model


@pytest.mark.dp
def test_dp_wrapper_rejects_invalid_params() -> None:
    """make_private_with_noise() validates all three DP parameters."""
    from privacy.dp_wrapper import make_private_with_noise

    model, optimizer = _make_tiny_model()
    with pytest.raises(ValueError, match="noise_multiplier"):
        make_private_with_noise(model=model, optimizer=optimizer,
                                noise_multiplier=0.0, max_grad_norm=1.0,
                                expected_batch_size=32)
    with pytest.raises(ValueError, match="max_grad_norm"):
        make_private_with_noise(model=model, optimizer=optimizer,
                                noise_multiplier=1.0, max_grad_norm=-1.0,
                                expected_batch_size=32)
    with pytest.raises(ValueError, match="expected_batch_size"):
        make_private_with_noise(model=model, optimizer=optimizer,
                                noise_multiplier=1.0, max_grad_norm=1.0,
                                expected_batch_size=0)


@pytest.mark.dp
def test_dp_wrapper_input_grads_flow() -> None:
    """The wrapped model still exposes input gradients (the VFL cut-layer grads)."""
    from privacy.dp_wrapper import make_private_with_noise

    model, optimizer = _make_tiny_model()
    dp_model, dp_optimizer = make_private_with_noise(
        model=model, optimizer=optimizer,
        noise_multiplier=1.0, max_grad_norm=1.0, expected_batch_size=8,
    )
    x = torch.randn(8, 12).requires_grad_(True)
    dp_optimizer.zero_grad()
    loss = dp_model(x).square().mean()
    loss.backward()
    assert x.grad is not None
    dp_optimizer.step()


# ─────────────────────────────────────────────────────────────────────────────
# Privacy accountant
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.dp
def test_accountant_budget_exceeded() -> None:
    """PrivacyBudgetTracker raises PrivacyBudgetExceededError when over budget.

    NOTE: This test mocks get_epsilon() to force a budget exceeded condition.
    """
    from unittest.mock import patch

    from privacy.privacy_accountant import PrivacyBudgetExceededError, PrivacyBudgetTracker

    tracker = PrivacyBudgetTracker(target_epsilon=1.0, sigma=1.0, delta=1e-5)

    with patch.object(tracker, "get_epsilon", return_value=1.5):
        with pytest.raises(PrivacyBudgetExceededError):
            tracker.step()


@pytest.mark.dp
def test_accountant_get_epsilon() -> None:
    """PrivacyBudgetTracker.get_epsilon() returns a float."""
    from privacy.privacy_accountant import PrivacyBudgetTracker

    tracker = PrivacyBudgetTracker(target_epsilon=1.0, sigma=1.0, delta=1e-5)
    tracker.steps = 2
    eps = tracker.get_epsilon()
    assert isinstance(eps, float), f"get_epsilon() returned {type(eps)}, expected float"


@pytest.mark.dp
def test_accountant_epsilon_monotone_sublinear() -> None:
    """Epsilon grows monotonically and sub-linearly in steps (composition sanity)."""
    from privacy.privacy_accountant import PrivacyBudgetTracker

    tracker = PrivacyBudgetTracker(target_epsilon=1e6, sigma=2.0, delta=1e-5)
    assert tracker.get_epsilon() == 0.0, "epsilon must be 0.0 before any step"

    eps_seen = []
    for _ in range(100):
        tracker.step()
        eps_seen.append(tracker.get_epsilon())

    assert all(b >= a for a, b in zip(eps_seen, eps_seen[1:])), "epsilon must be monotone"
    assert eps_seen[99] < 100 * eps_seen[0], (
        f"eps(100)={eps_seen[99]:.4f} must be < 100 * eps(1)={eps_seen[0]:.4f} — "
        "composition should grow sub-linearly"
    )


@pytest.mark.dp
def test_accountant_subsampling_tightens_epsilon() -> None:
    """Mini-batch (sample_rate < 1) yields lower epsilon than full-batch at equal sigma/steps."""
    from privacy.privacy_accountant import PrivacyBudgetTracker

    full = PrivacyBudgetTracker(target_epsilon=1e6, sigma=2.0, sample_rate=1.0, delta=1e-5)
    mini = PrivacyBudgetTracker(target_epsilon=1e6, sigma=2.0, sample_rate=0.1, delta=1e-5)
    full.steps = mini.steps = 50

    eps_full, eps_mini = full.get_epsilon(), mini.get_epsilon()
    assert eps_mini < eps_full, (
        f"subsampled eps ({eps_mini:.4f}) should beat full-batch eps ({eps_full:.4f})"
    )


@pytest.mark.dp
def test_accountant_state_roundtrip() -> None:
    """to_dict()/from_dict() rebuild an identical tracker from plain primitives."""
    from privacy.privacy_accountant import PrivacyBudgetTracker

    tracker = PrivacyBudgetTracker(target_epsilon=5.0, sigma=1.5, sample_rate=0.25, delta=1e-5)
    tracker.steps = 7

    state = tracker.to_dict()
    assert all(isinstance(v, int | float) for v in state.values()), (
        "persisted state must be plain primitives only"
    )

    restored = PrivacyBudgetTracker.from_dict(state)
    assert restored.steps == tracker.steps
    assert restored.get_epsilon() == pytest.approx(tracker.get_epsilon())


@pytest.mark.dp
def test_accountant_rejects_invalid_params() -> None:
    """Constructor validates target_epsilon, sigma, sample_rate, and delta."""
    from privacy.privacy_accountant import PrivacyBudgetTracker

    with pytest.raises(ValueError, match="target_epsilon"):
        PrivacyBudgetTracker(target_epsilon=0.0)
    with pytest.raises(ValueError, match="sigma"):
        PrivacyBudgetTracker(target_epsilon=1.0, sigma=-1.0)
    with pytest.raises(ValueError, match="sample_rate"):
        PrivacyBudgetTracker(target_epsilon=1.0, sample_rate=1.5)
    with pytest.raises(ValueError, match="delta"):
        PrivacyBudgetTracker(target_epsilon=1.0, delta=0.0)


@pytest.mark.dp
def test_accountant_would_exceed_gates_before_spend() -> None:
    """would_exceed() flags the next step while current epsilon is still within budget."""
    from privacy.privacy_accountant import PrivacyBudgetTracker

    tracker = PrivacyBudgetTracker(target_epsilon=1.0, sigma=6.45, sample_rate=1.0, delta=1e-5)
    while not tracker.would_exceed():
        tracker.step()
    assert tracker.get_epsilon() <= tracker.target_epsilon, (
        "would_exceed() must trip before the budget is actually crossed"
    )
    assert tracker.steps > 0


@pytest.mark.dp
def test_solve_noise_multiplier_fits_budget() -> None:
    """Solved sigma stays within budget over the horizon, and tightly so."""
    from privacy.privacy_accountant import PrivacyBudgetTracker, solve_noise_multiplier

    for q in (1.0, 0.1):
        sigma = solve_noise_multiplier(target_epsilon=1.0, sample_rate=q, steps=50, delta=1e-5)
        tracker = PrivacyBudgetTracker(target_epsilon=1.0, sigma=sigma, sample_rate=q, delta=1e-5)
        tracker.steps = 50
        eps = tracker.get_epsilon()
        assert eps <= 1.0, f"q={q}: eps={eps:.4f} exceeds budget"
        assert eps > 0.9, f"q={q}: eps={eps:.4f} — sigma is wastefully large"


# ─────────────────────────────────────────────────────────────────────────────
# Combined (client + server) epsilon
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.dp
def test_combine_epsilon_dominates_each_source() -> None:
    """The composed total exceeds every individual source's epsilon."""
    from privacy.privacy_accountant import PrivacyBudgetTracker, combine_epsilon

    silo_a = PrivacyBudgetTracker(target_epsilon=1e6, sigma=2.0, sample_rate=0.1, delta=1e-5)
    silo_b = PrivacyBudgetTracker(target_epsilon=1e6, sigma=3.0, sample_rate=0.1, delta=1e-5)
    server = PrivacyBudgetTracker(target_epsilon=1e6, sigma=2.0, sample_rate=1.0, delta=1e-5)
    silo_a.steps = silo_b.steps = server.steps = 50

    combined = combine_epsilon([silo_a, silo_b, server], delta=1e-5)
    for t in (silo_a, silo_b, server):
        assert combined >= t.get_epsilon(), (
            f"combined ({combined:.4f}) must dominate source ({t.get_epsilon():.4f})"
        )


@pytest.mark.dp
def test_combine_epsilon_single_tracker_matches_get_epsilon() -> None:
    from privacy.privacy_accountant import PrivacyBudgetTracker, combine_epsilon

    tracker = PrivacyBudgetTracker(target_epsilon=1e6, sigma=2.0, sample_rate=0.2, delta=1e-5)
    tracker.steps = 10
    assert combine_epsilon([tracker], delta=1e-5) == pytest.approx(tracker.get_epsilon())


@pytest.mark.dp
def test_combine_epsilon_ignores_inactive_trackers() -> None:
    """Zero-step trackers contribute nothing; no active trackers means zero epsilon."""
    from privacy.privacy_accountant import PrivacyBudgetTracker, combine_epsilon

    idle = PrivacyBudgetTracker(target_epsilon=1.0, sigma=1.0)
    active = PrivacyBudgetTracker(target_epsilon=1e6, sigma=2.0)
    active.steps = 5

    assert combine_epsilon([], delta=1e-5) == 0.0
    assert combine_epsilon([idle], delta=1e-5) == 0.0
    assert combine_epsilon([idle, active], delta=1e-5) == pytest.approx(active.get_epsilon())


@pytest.mark.dp
def test_combine_epsilon_rejects_invalid_delta() -> None:
    from privacy.privacy_accountant import combine_epsilon

    with pytest.raises(ValueError, match="delta"):
        combine_epsilon([], delta=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Cut-layer noise (client mechanism)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.dp
def test_clip_embedding_rows_enforces_bound() -> None:
    """Rows over the bound are scaled to it; rows under it are untouched."""
    from privacy.noise_injection import clip_embedding_rows

    emb = torch.zeros(2, 4)
    emb[0, 0] = 10.0   # norm 10 -> clipped to 1
    emb[1, 0] = 0.5    # norm 0.5 -> unchanged
    clipped = clip_embedding_rows(emb, max_norm=1.0)

    norms = clipped.norm(p=2, dim=1)
    assert norms[0] == pytest.approx(1.0)
    assert torch.allclose(clipped[1], emb[1])
    assert (norms <= 1.0 + 1e-6).all()


@pytest.mark.dp
def test_add_gaussian_noise_changes_and_preserves_shape() -> None:
    from privacy.noise_injection import add_gaussian_noise

    emb = torch.ones(16, 32)
    noisy = add_gaussian_noise(emb, noise_std=1.0)
    assert noisy.shape == emb.shape
    assert not torch.allclose(emb, noisy)
    assert torch.allclose(emb, torch.ones(16, 32)), "input must not be modified in place"


@pytest.mark.dp
def test_noise_fns_reject_invalid_params() -> None:
    from privacy.noise_injection import add_gaussian_noise, clip_embedding_rows

    emb = torch.ones(2, 4)
    with pytest.raises(ValueError, match="max_norm"):
        clip_embedding_rows(emb, max_norm=0.0)
    with pytest.raises(ValueError, match="noise_std"):
        add_gaussian_noise(emb, noise_std=0.0)
