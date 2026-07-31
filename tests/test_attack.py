"""Unit tests for the privacy-attack module — owned by Wyatt Wilfong.

Covers the three attack primitives in attack/:
  - inversion.py:run_inversion_attack        (iDLG gradient-matching)
  - label_inference.py:infer_labels_from_gradient
  - gradient_monitor.py:GradientNormMonitor

Run: pytest tests/test_attack.py -v

These are white-box unit tests: each exercises one primitive in isolation with
small, seeded models so the assertions are deterministic and fast (no OULAD, no
checkpoints, no Flower). They do NOT measure attack success rate on real data —
that is the job of scripts/run_attack.py.
"""

import pytest
import torch

from attack.gradient_monitor import GradientNormMonitor
from attack.inversion import IMD_BAND_COL_IDX, run_inversion_attack
from attack.label_inference import infer_labels_from_gradient
from fl_server.top_model import TopModel
from silo_a.bottom_model_a import BottomModelA

# Silo A carries 18 features (see bottom_model_a.py / CLAUDE.md run-config).
SILO_A_INPUT_DIM = 18
EMB_DIM = 8  # small embedding keeps the tests fast; the code is dim-agnostic


def _seed(seed: int = 42) -> None:
    """Seed torch so every attack run in this suite is reproducible."""
    torch.manual_seed(seed)


def _eval_bottom_a(input_dim: int = SILO_A_INPUT_DIM, embedding_dim: int = EMB_DIM) -> BottomModelA:
    """A seeded BottomModelA in eval mode (dropout off, LayerNorm deterministic)."""
    _seed()
    model = BottomModelA(input_dim=input_dim, embedding_dim=embedding_dim)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Model inversion (iDLG)
# ─────────────────────────────────────────────────────────────────────────────


def test_inversion_returns_expected_schema() -> None:
    """run_inversion_attack returns the documented keys with sane types/ranges."""
    _seed()
    model = _eval_bottom_a()
    x = torch.randn(1, SILO_A_INPUT_DIM)
    real_emb = model(x)
    true_band = int(x[0, IMD_BAND_COL_IDX].round().clamp(0, 9).item())

    result = run_inversion_attack(model, real_emb, true_band, num_iters=50)

    assert set(result) == {
        "success", "predicted_imd_band", "true_imd_band", "final_loss", "iters",
        "reconstruction",
    }
    assert isinstance(result["success"], bool)
    assert isinstance(result["predicted_imd_band"], int)
    assert 0 <= result["predicted_imd_band"] <= 9
    assert result["true_imd_band"] == true_band
    assert isinstance(result["final_loss"], float)
    assert result["iters"] == 50
    # The full recovered row, not just the one column that used to be read out.
    assert result["reconstruction"].shape == (SILO_A_INPUT_DIM,)


def test_inversion_infers_input_dim_from_model() -> None:
    """Regression: the dummy input width is read from the model, not hard-coded to 12.

    A non-18 feature width (here 15) must not raise a shape-mismatch error, proving
    run_inversion_attack sizes dummy_input to model.encoder[0].in_features.
    """
    _seed()
    for input_dim in (12, 15, SILO_A_INPUT_DIM):
        model = _eval_bottom_a(input_dim=input_dim)
        x = torch.randn(1, input_dim)
        real_emb = model(x)
        # Should complete without a RuntimeError about matmul shapes.
        result = run_inversion_attack(model, real_emb, true_imd_band=0, num_iters=10)
        assert result["iters"] == 10


def test_inversion_converges_toward_target_embedding() -> None:
    """The gradient-matching objective drives ||dummy_emb - real_emb||^2 down.

    Against an eval-mode (deterministic) model the target embedding is reachable, so
    a converged attack should reach a far smaller loss than a random guess at the
    same input width.
    """
    _seed()
    model = _eval_bottom_a()
    x = torch.randn(1, SILO_A_INPUT_DIM)
    real_emb = model(x)

    # Baseline: matching loss of an arbitrary random input (no optimization).
    random_guess_loss = ((model(torch.randn(1, SILO_A_INPUT_DIM)) - real_emb) ** 2).sum().item()

    result = run_inversion_attack(model, real_emb, true_imd_band=0, num_iters=300, lr=0.5)

    assert result["final_loss"] < random_guess_loss, (
        f"attack loss {result['final_loss']:.4g} did not improve on the random "
        f"baseline {random_guess_loss:.4g}"
    )
    assert result["final_loss"] < 1.0, f"attack failed to converge (loss {result['final_loss']:.4g})"


def test_inversion_success_flag_tracks_prediction() -> None:
    """success is exactly (predicted_imd_band == true_imd_band)."""
    _seed()
    model = _eval_bottom_a()
    x = torch.randn(1, SILO_A_INPUT_DIM)
    real_emb = model(x)

    hit = run_inversion_attack(model, real_emb, true_imd_band=3, num_iters=20)
    assert hit["success"] == (hit["predicted_imd_band"] == hit["true_imd_band"])


# ─────────────────────────────────────────────────────────────────────────────
# Label inference
# ─────────────────────────────────────────────────────────────────────────────


def _top_ref_direction(top: TopModel, emb_dim: int) -> torch.Tensor:
    """Reproduce the attack's internal reference direction: d(top output)/d(emb_b)
    at the zero probe. Used to craft gradients with a known, attackable sign."""
    probe_a = torch.zeros(1, emb_dim, requires_grad=True)
    probe_b = torch.zeros(1, emb_dim, requires_grad=True)
    top(probe_a, probe_b).reshape(()).backward()
    return probe_b.grad.detach().reshape(-1)


def test_label_inference_output_shape_and_domain() -> None:
    """Returns one binary label per row as a long tensor in {0, 1}."""
    _seed()
    top = TopModel(embedding_dim=EMB_DIM)
    top.eval()
    grad = torch.randn(6, EMB_DIM)
    probs = torch.rand(6)

    labels = infer_labels_from_gradient(grad, top, probs)

    assert labels.shape == (6,)
    assert labels.dtype == torch.long
    assert set(labels.tolist()) <= {0, 1}


def test_label_inference_recovers_labels_from_signed_gradient() -> None:
    """With gradients aligned to (p - y) along the top model's reference direction,
    the sign heuristic recovers the labels exactly.

    The returned per-sample gradient is (p_i - y_i) pushed through the top model:
    y=0 aligns with +ref, y=1 with -ref. We craft exactly that and check recovery.
    """
    _seed()
    top = TopModel(embedding_dim=EMB_DIM)
    top.eval()
    ref = _top_ref_direction(top, EMB_DIM)
    assert ref.norm() > 0, "degenerate reference direction — pick another seed"

    y = torch.tensor([0, 1, 0, 1, 1, 0])
    signs = torch.where(y == 0, 1.0, -1.0)  # (p - y): +ve for y=0, -ve for y=1
    grad = signs.unsqueeze(1) * ref.unsqueeze(0)  # each row = sign_i * ref
    probs = torch.full((6,), 0.5)  # irrelevant here; gradient carries the signal

    inferred = infer_labels_from_gradient(grad, top, probs)
    assert torch.equal(inferred, y), f"expected {y.tolist()}, got {inferred.tolist()}"


def test_label_inference_falls_back_to_probs_when_gradient_uninformative() -> None:
    """Zero gradients carry no signal, so the attack backs off to (p > 0.5)."""
    _seed()
    top = TopModel(embedding_dim=EMB_DIM)
    top.eval()
    grad = torch.zeros(4, EMB_DIM)
    probs = torch.tensor([0.2, 0.8, 0.6, 0.4])

    inferred = infer_labels_from_gradient(grad, top, probs)
    assert torch.equal(inferred, torch.tensor([0, 1, 1, 0]))


def test_label_inference_preserves_top_model_training_mode() -> None:
    """The attack must not leave the top model flipped into eval/train."""
    _seed()
    top = TopModel(embedding_dim=EMB_DIM)
    top.train()
    infer_labels_from_gradient(torch.randn(3, EMB_DIM), top, torch.rand(3))
    assert top.training is True

    top.eval()
    infer_labels_from_gradient(torch.randn(3, EMB_DIM), top, torch.rand(3))
    assert top.training is False


# ─────────────────────────────────────────────────────────────────────────────
# Gradient-norm monitor (defense)
# ─────────────────────────────────────────────────────────────────────────────


def test_monitor_reports_l2_norm() -> None:
    """update returns the exact L2 norm of the gradient."""
    monitor = GradientNormMonitor()
    norm, _ = monitor.update(torch.tensor([3.0, 4.0]))
    assert norm == pytest.approx(5.0)


def test_monitor_cold_start_never_flags() -> None:
    """Fewer than three observations cannot be anomalous (no running mean yet)."""
    monitor = GradientNormMonitor()
    _, a0 = monitor.update(torch.tensor([1.0]))
    _, a1 = monitor.update(torch.tensor([50.0]))  # even a spike can't flag yet
    assert a0 is False
    assert a1 is False


def test_monitor_flags_spike_against_running_mean() -> None:
    """A gradient far above the recent mean is flagged; steady norms are not."""
    monitor = GradientNormMonitor(window=10, threshold_multiplier=3.0)
    # Three steady rounds (norm ~1.0): establishes the running mean, no flags.
    flags = [monitor.update(torch.tensor([1.0]))[1] for _ in range(3)]
    assert flags == [False, False, False]

    # A 10x spike relative to the ~1.0 mean must trip the detector.
    _, spiked = monitor.update(torch.tensor([10.0]))
    assert spiked is True

    # Returning to baseline should not flag (the spike is now one of ~5 samples,
    # but the current small norm is below threshold * mean).
    _, calm = monitor.update(torch.tensor([1.0]))
    assert calm is False


def test_monitor_steady_stream_is_never_anomalous() -> None:
    """Constant-norm gradients never trip the threshold."""
    monitor = GradientNormMonitor(threshold_multiplier=3.0)
    any_flag = any(monitor.update(torch.tensor([2.0]))[1] for _ in range(20))
    assert any_flag is False


def test_monitor_history_capped_to_window() -> None:
    """The running-average history is trimmed to the configured window."""
    monitor = GradientNormMonitor(window=3)
    for _ in range(6):
        monitor.update(torch.tensor([1.0]))
    assert len(monitor.history) == 3


def test_monitor_history_returns_a_copy() -> None:
    """The history property hands back a copy, not the internal list."""
    monitor = GradientNormMonitor()
    monitor.update(torch.tensor([1.0]))
    snapshot = monitor.history
    snapshot.append(999.0)
    assert len(monitor.history) == 1, "mutating the returned list must not affect the monitor"


def test_monitor_threshold_multiplier_is_respected() -> None:
    """A lower multiplier flags a moderate rise that a higher one tolerates."""
    strict = GradientNormMonitor(threshold_multiplier=1.5)
    lax = GradientNormMonitor(threshold_multiplier=5.0)
    for m in (strict, lax):
        for _ in range(3):
            m.update(torch.tensor([1.0]))  # mean ~1.0
    _, strict_flag = strict.update(torch.tensor([2.0]))  # 2x mean
    _, lax_flag = lax.update(torch.tensor([2.0]))
    assert strict_flag is True
    assert lax_flag is False
