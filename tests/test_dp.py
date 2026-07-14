"""Differential Privacy unit tests — owned by Wyatt Wilfong.

Run: pytest tests/test_dp.py -v

Tests progress from Week 4 (dp_wrapper) to Week 6 (noise_injection).
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def _make_tiny_model_and_loader() -> tuple[nn.Module, torch.optim.Optimizer, DataLoader]:
    """Helper: create a tiny 2-layer MLP + Adam + DataLoader for testing."""
    model = nn.Sequential(nn.Linear(12, 8), nn.ReLU(), nn.Linear(8, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    X = torch.randn(200, 12)
    y = torch.randint(0, 2, (200,)).float()
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32)
    return model, optimizer, loader


# ─────────────────────────────────────────────────────────────────────────────
# Week 4 Tests — DP Wrapper
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.dp
def test_dp_wrapper_basic() -> None:
    """make_private() returns a wrapped model, optimizer, and loader.
    Expected: PASSES after Week 4 Step 3.
    """
    from privacy.dp_wrapper import make_private
    model, optimizer, loader = _make_tiny_model_and_loader()

    dp_model, dp_optimizer, dp_loader, priv_engine = make_private(
        model=model, optimizer=optimizer, data_loader=loader,
        target_epsilon=5.0, target_delta=1e-5,
        max_grad_norm=1.0, num_epochs=10,
    )

    assert dp_model is not None, "make_private() returned None for dp_model"
    assert dp_optimizer is not None, "make_private() returned None for dp_optimizer"
    assert dp_loader is not None, "make_private() returned None for dp_loader"
    assert priv_engine is not None, "make_private() returned None for priv_engine"
    print("\n✅  make_private() PASSED — Everything returned successfully")


@pytest.mark.dp
def test_dp_wrapper_rejects_invalid_epsilon() -> None:
    """make_private() raises ValueError for epsilon <= 0.
    Expected: PASSES after Week 4 Step 1 (input validation).
    """
    from privacy.dp_wrapper import make_private
    model, optimizer, loader = _make_tiny_model_and_loader()

    with pytest.raises(ValueError, match="epsilon"):
        make_private(
            model=model, optimizer=optimizer, data_loader=loader,
            target_epsilon=-1.0, target_delta=1e-5,
            max_grad_norm=1.0, num_epochs=5,
        )


@pytest.mark.dp
def test_dp_wrapper_rejects_invalid_delta() -> None:
    """make_private() raises ValueError for delta <= 0."""
    from privacy.dp_wrapper import make_private
    model, optimizer, loader = _make_tiny_model_and_loader()

    with pytest.raises(ValueError, match="delta"):
        make_private(
            model=model, optimizer=optimizer, data_loader=loader,
            target_epsilon=1.0, target_delta=-1e-5,
            max_grad_norm=1.0, num_epochs=5,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Week 5 Tests — Privacy Accountant
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.dp
def test_accountant_budget_exceeded() -> None:
    """PrivacyBudgetTracker raises PrivacyBudgetExceededError when over budget.

    NOTE: This test mocks get_epsilon() to force a budget exceeded condition.
    """
    from unittest.mock import patch

    from privacy.privacy_accountant import PrivacyBudgetExceededError, PrivacyBudgetTracker

    tracker = PrivacyBudgetTracker(target_epsilon=1.0, sigma=1.0, delta=1e-5)

    # Patch get_epsilon to simulate exceeding the budget
    with patch.object(tracker, "get_epsilon", return_value=1.5):
        with pytest.raises(PrivacyBudgetExceededError):
            tracker.step()
    print("\n✅  PrivacyBudgetExceededError raised correctly")


@pytest.mark.dp
def test_accountant_get_epsilon() -> None:
    """PrivacyBudgetTracker.get_epsilon() returns a float."""
    from privacy.privacy_accountant import PrivacyBudgetTracker

    tracker = PrivacyBudgetTracker(target_epsilon=1.0, sigma=1.0, delta=1e-5)
    tracker.steps = 2
    eps = tracker.get_epsilon()
    assert isinstance(eps, float), f"get_epsilon() returned {type(eps)}, expected float"
    print(f"\n✅  get_epsilon() returned a float: {eps:.4f}")


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
    print(f"\n✅  50 steps @ sigma=2.0: full-batch eps={eps_full:.4f}, q=0.1 eps={eps_mini:.4f}")


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


# ─────────────────────────────────────────────────────────────────────────────
# Week 6 Tests — Noise Injection
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.dp
def test_noise_injection_changes_embedding() -> None:
    """add_dp_noise_to_embedding() modifies the tensor with low epsilon.
    Expected: PASSES after Week 6 Step 5.
    """
    from privacy.noise_injection import add_dp_noise_to_embedding
    embedding = torch.ones(4, 32)
    noisy, _ = add_dp_noise_to_embedding(embedding, sensitivity=1.0, epsilon=0.01)

    assert not torch.allclose(embedding, noisy), (
        "With epsilon=0.01, noise should significantly alter the embedding"
    )


@pytest.mark.dp
def test_noise_injection_preserves_shape() -> None:
    """add_dp_noise_to_embedding() returns a tensor with the same shape."""
    from privacy.noise_injection import add_dp_noise_to_embedding
    embedding = torch.randn(16, 32)
    noisy, _ = add_dp_noise_to_embedding(embedding, sensitivity=1.0, epsilon=1.0)
    assert noisy.shape == embedding.shape, (
        f"Shape mismatch: input {embedding.shape}, output {noisy.shape}"
    )


@pytest.mark.dp
def test_noise_injection_high_epsilon_near_original() -> None:
    """With very high epsilon (little privacy), noise should be near zero."""
    from privacy.noise_injection import add_dp_noise_to_embedding
    embedding = torch.ones(8, 32)
    noisy, _ = add_dp_noise_to_embedding(embedding, sensitivity=1.0, epsilon=1e6)
    # Mean absolute difference should be tiny
    mad = (noisy - embedding).abs().mean().item()
    assert mad < 0.01, f"With epsilon=1e6, noise should be near 0, got mean abs diff = {mad:.6f}"
