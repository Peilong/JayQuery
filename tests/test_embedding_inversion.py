"""Tests for the noise-matched learned inversion attack (ICC C2, Row 2).

Run: pytest tests/test_embedding_inversion.py -v

The existing attack (attack/inversion.py) optimizes a dummy input to match a
target embedding while HOLDING the bottom model. This one models the actual
coordinator: it sees noised embeddings, holds auxiliary records, and trains an
inverse mapping embedding -> protected attribute.

The property that makes the measurement meaningful is that the attacker's
noise pipeline is the *same* one the client applies. An inverse trained on
clean embeddings and evaluated on noised ones measures the mismatch, not the
protection, and would report attack success falling with epsilon for entirely
the wrong reason.
"""

import pytest
import torch


def _bottom(input_dim: int = 8, embedding_dim: int = 6):
    from silo_a.bottom_model_a import BottomModelA

    torch.manual_seed(0)
    m = BottomModelA(
        input_dim=input_dim, embedding_dim=embedding_dim, dropout_rate=0.0, hidden_dim=16
    )
    m.eval()
    return m


def _data(n: int = 400, input_dim: int = 8, n_classes: int = 4, seed: int = 0):
    """Features whose first column determines the protected attribute."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, input_dim, generator=g)
    y = (X[:, 0] * 1.5).clamp(-2, 2).add(2).div(4).mul(n_classes).long().clamp(0, n_classes - 1)
    return X, y


@pytest.mark.slow
class TestNoisePipelineMatch:
    """The attacker must reproduce the client's clip+noise exactly."""

    def test_matches_privacy_noise_injection(self) -> None:
        """THE calibration invariant. If this drifts from what
        fl_client/common.py sends, every attack-success-vs-epsilon number is
        measuring the wrong thing."""
        from attack.embedding_inversion import noised_embeddings
        from privacy.noise_injection import add_gaussian_noise, clip_embedding_rows

        model = _bottom()
        X, _ = _data(n=32)
        max_norm, sigma = 1.0, 0.5

        got = noised_embeddings(model, X, max_norm=max_norm, sigma=sigma, seed=7)

        with torch.no_grad():
            clean = model(X)
        torch.manual_seed(7)
        expected = add_gaussian_noise(
            clip_embedding_rows(clean, max_norm), noise_std=sigma * max_norm
        )
        assert torch.allclose(got, expected, atol=1e-6)

    def test_clips_before_noising(self) -> None:
        """Order matters: clipping after noising would not enforce the
        sensitivity bound the accounting assumes."""
        from attack.embedding_inversion import noised_embeddings

        model = _bottom()
        X, _ = _data(n=64)
        clean_rows = noised_embeddings(model, X, max_norm=0.1, sigma=1e-9, seed=1)
        assert clean_rows.norm(dim=1).max() <= 0.1 + 1e-3

    def test_deterministic_given_seed(self) -> None:
        from attack.embedding_inversion import noised_embeddings

        model = _bottom()
        X, _ = _data(n=32)
        a = noised_embeddings(model, X, max_norm=1.0, sigma=0.3, seed=5)
        b = noised_embeddings(model, X, max_norm=1.0, sigma=0.3, seed=5)
        assert torch.equal(a, b)

    def test_more_noise_moves_embeddings_further(self) -> None:
        from attack.embedding_inversion import noised_embeddings

        model = _bottom()
        X, _ = _data(n=128)
        clean = noised_embeddings(model, X, max_norm=1.0, sigma=1e-9, seed=2)
        low = noised_embeddings(model, X, max_norm=1.0, sigma=0.2, seed=2)
        high = noised_embeddings(model, X, max_norm=1.0, sigma=1.5, seed=2)
        assert (high - clean).norm() > (low - clean).norm()


@pytest.mark.slow
class TestLearnedInversion:
    """Attack success must track the noise, and be honestly measured."""

    def _run(self, sigma: float, n_obs: int = 1, seed: int = 0):
        from attack.embedding_inversion import run_learned_inversion

        model = _bottom()
        X_aux, y_aux = _data(n=600, seed=1)
        X_tgt, y_tgt = _data(n=200, seed=2)
        return run_learned_inversion(
            bottom_model=model,
            X_aux=X_aux, y_aux=y_aux,
            X_target=X_tgt, y_target=y_tgt,
            n_classes=4, max_norm=1.0, sigma=sigma,
            n_observations=n_obs, seed=seed, epochs=60,
        )

    def test_reports_accuracy_against_a_guessing_baseline(self) -> None:
        """Raw accuracy is meaningless without the majority-class rate: on a
        skewed attribute a constant predictor can look like a strong attack."""
        res = self._run(sigma=0.05)
        assert 0.0 <= res.accuracy <= 1.0
        assert 0.0 <= res.baseline <= 1.0
        assert res.lift == pytest.approx(res.accuracy - res.baseline)
        assert res.n_target == 200

    def test_recovers_the_attribute_with_little_noise(self) -> None:
        res = self._run(sigma=0.02)
        assert res.accuracy > res.baseline + 0.10, (
            f"inverse learned nothing at near-zero noise "
            f"(acc {res.accuracy:.3f} vs baseline {res.baseline:.3f})"
        )

    def test_heavy_noise_collapses_the_attack(self) -> None:
        res = self._run(sigma=12.0)
        assert res.accuracy <= res.baseline + 0.08, (
            f"attack still succeeded under heavy noise (acc {res.accuracy:.3f} "
            f"vs baseline {res.baseline:.3f}) — the noise is not reaching the "
            f"embeddings the attacker sees"
        )

    def test_success_decreases_with_noise(self) -> None:
        """The monotone trend IS the C2 result."""
        lifts = [self._run(sigma=s).lift for s in (0.02, 0.5, 12.0)]
        assert lifts[0] > lifts[-1], f"not decreasing in sigma: {lifts}"

    def test_averaging_observations_helps_the_attacker(self) -> None:
        """A coordinator sees each student every round and can average to
        denoise. Ignoring that would understate the leak."""
        one = self._run(sigma=1.5, n_obs=1, seed=3)
        many = self._run(sigma=1.5, n_obs=10, seed=3)
        assert many.lift >= one.lift - 0.02, (
            f"averaging 10 observations did not help (lift {one.lift:.3f} -> "
            f"{many.lift:.3f})"
        )


@pytest.mark.slow
class TestEvaluationHonesty:
    """Guards against inflating the attack's apparent success."""

    def test_rejects_overlap_between_auxiliary_and_target(self) -> None:
        """Training the inverse on records it is then scored against reports
        memorization as leakage."""
        from attack.embedding_inversion import run_learned_inversion

        X, y = _data(n=200)
        with pytest.raises(ValueError, match="overlap|disjoint"):
            run_learned_inversion(
                bottom_model=_bottom(),
                X_aux=X, y_aux=y,
                X_target=X, y_target=y,  # same records
                n_classes=4, max_norm=1.0, sigma=0.1, seed=0, epochs=5,
            )

    def test_baseline_is_the_majority_class_rate(self) -> None:
        from attack.embedding_inversion import run_learned_inversion

        X_aux, y_aux = _data(n=300, seed=1)
        X_tgt, _ = _data(n=100, seed=2)
        y_tgt = torch.zeros(100, dtype=torch.long)
        y_tgt[:80] = 1  # 80% majority

        res = run_learned_inversion(
            bottom_model=_bottom(), X_aux=X_aux, y_aux=y_aux,
            X_target=X_tgt, y_target=y_tgt,
            n_classes=4, max_norm=1.0, sigma=0.1, seed=0, epochs=5,
        )
        assert res.baseline == pytest.approx(0.80)


@pytest.mark.slow
class TestCheckpointCalibration:
    """The attack must use the noise a run ACTUALLY spent."""

    def test_reads_realized_sigma_from_meta(self, tmp_path) -> None:
        """meta.json records realized sigma and steps precisely so a post-hoc
        attack is calibrated to the run rather than to the config, which
        diverges whenever a run stops early."""
        import json

        from attack.embedding_inversion import noise_params_from_meta

        meta = {
            "num_rounds": 50,
            "stopped_early": True,
            "privacy_primitives": {
                "silo_a": {"sigma": 2.75, "sample_rate": 0.05, "steps": 31,
                           "delta": 1e-5, "target_epsilon": 1.0},
            },
            "communication": {},
        }
        (tmp_path / "meta.json").write_text(json.dumps(meta))

        sigma, steps = noise_params_from_meta(tmp_path, channel="silo_a")
        assert sigma == pytest.approx(2.75)
        assert steps == 31, "used the configured horizon instead of realized steps"

    def test_missing_channel_raises(self, tmp_path) -> None:
        import json

        from attack.embedding_inversion import noise_params_from_meta

        (tmp_path / "meta.json").write_text(json.dumps({"privacy_primitives": {}}))
        with pytest.raises(KeyError, match="silo_a|channel"):
            noise_params_from_meta(tmp_path, channel="silo_a")
