"""Tests for attack/label_inference_grad.py.

The peer silo never sees labels, only the gradient the coordinator returns for
its own embedding. For a binary head trained with BCE, dL/dz = sigma(z) - y,
so that gradient's sign flips with the label and the returned vectors fall
into two antipodal groups. Recovering the labels is then a sign problem, not a
learning problem.

What the tests pin down is the part that makes the measurement meaningful: the
attack must recover labels from gradients alone, must resolve the global sign
without being told the answer, and must be scored against the majority-class
rate rather than chance.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


class TestGradientSignAttack:
    def _grads(self, y: np.ndarray, dim: int = 8, seed: int = 0) -> torch.Tensor:
        """Gradients of the shape a BCE head produces: (sigma(z) - y) * w."""
        g = torch.Generator().manual_seed(seed)
        w = torch.randn(dim, generator=g)
        resid = torch.tensor(0.5 - y, dtype=torch.float32)       # sigma(z)=0.5
        noise = 0.02 * torch.randn(len(y), dim, generator=g)
        return resid[:, None] * w[None, :] + noise

    def test_recovers_labels_from_gradients_alone(self) -> None:
        from attack.label_inference_grad import infer_labels_from_gradients

        y = np.array([1, 0, 1, 1, 0, 0, 1, 0] * 12)
        r = infer_labels_from_gradients(self._grads(y), y, base_rate=y.mean())
        assert r.accuracy > 0.95, "an undefended gradient should leak the label"

    def test_resolves_the_global_sign_without_being_told(self) -> None:
        """Projection onto the principal direction is sign-ambiguous. The
        attacker breaks the tie with the base rate, which it knows, not with
        the labels it is trying to recover."""
        from attack.label_inference_grad import infer_labels_from_gradients

        for flip in (False, True):
            y = np.array([1] * 30 + [0] * 70)
            g = self._grads(y)
            if flip:
                g = -g
            r = infer_labels_from_gradients(g, y, base_rate=0.30)
            assert r.accuracy > 0.95, f"sign resolution failed (flip={flip})"

    def test_reports_advantage_over_the_majority_class(self) -> None:
        """On a skewed label a constant predictor already scores well, so
        accuracy alone would overstate the leak."""
        from attack.label_inference_grad import infer_labels_from_gradients

        y = np.array([1] * 10 + [0] * 90)
        r = infer_labels_from_gradients(self._grads(y), y, base_rate=0.10)
        assert r.baseline == pytest.approx(0.90)
        assert r.advantage == pytest.approx(r.accuracy - 0.90)

    def test_pure_noise_gradients_do_not_beat_the_baseline(self) -> None:
        """The control: with no label signal in the gradient, the attack must
        land near the majority rate, not above it."""
        from attack.label_inference_grad import infer_labels_from_gradients

        rng = np.random.default_rng(0)
        y = (rng.random(400) < 0.45).astype(int)
        g = torch.tensor(rng.normal(size=(400, 8)), dtype=torch.float32)
        r = infer_labels_from_gradients(g, y, base_rate=0.45)
        assert r.advantage < 0.10, f"spurious leak: advantage {r.advantage}"

    def test_rejects_mismatched_shapes(self) -> None:
        from attack.label_inference_grad import infer_labels_from_gradients

        with pytest.raises(ValueError, match="rows"):
            infer_labels_from_gradients(torch.zeros(5, 4), np.zeros(6), 0.5)
