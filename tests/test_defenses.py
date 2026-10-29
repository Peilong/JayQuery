"""Tests for attack/defenses.py, the non-DP cut-layer baselines.

These defenses are compared against the DP placements under the same learned
inverse, so the properties that matter are the ones that keep that comparison
fair: a defense must alter what is sent, must be deterministic given its key
so the attacker sees a consistent transform, and must preserve the row count.
"""

from __future__ import annotations

import pytest
import torch


class TestRandomProjection:
    """A fixed secret orthogonal map, the archetypal 'transform it' defense."""

    def test_changes_the_embedding(self) -> None:
        from attack.defenses import random_projection

        emb = torch.randn(16, 32, generator=torch.Generator().manual_seed(0))
        out = random_projection(emb, seed=1)
        assert not torch.allclose(out, emb)

    def test_is_deterministic_in_its_key(self) -> None:
        """The silo applies one fixed map for the whole run, so the attacker
        sees a consistent transform. A fresh draw per call would make this a
        noise mechanism by accident."""
        from attack.defenses import random_projection

        emb = torch.randn(8, 32, generator=torch.Generator().manual_seed(0))
        assert torch.allclose(random_projection(emb, seed=7),
                              random_projection(emb, seed=7))
        assert not torch.allclose(random_projection(emb, seed=7),
                                  random_projection(emb, seed=8))

    def test_preserves_geometry(self) -> None:
        """Orthogonal, so norms and distances survive. That is precisely why
        it costs no utility and precisely why it cannot stop a learned
        inverse.

        Asserted on norms rather than through torch.cdist, whose float32
        accumulation is itself good only to about 1e-3 at this scale and so
        cannot witness a 1e-6 property.
        """
        from attack.defenses import random_projection

        emb = torch.randn(32, 32, generator=torch.Generator().manual_seed(3))
        out = random_projection(emb, seed=2)
        assert torch.allclose(out.norm(dim=1), emb.norm(dim=1), atol=1e-5)
        inner_before = emb @ emb.T
        inner_after = out @ out.T
        assert torch.allclose(inner_before, inner_after, atol=1e-4)


class TestTopK:
    """Sparsification, used in federated learning for compression."""

    def test_keeps_exactly_k_per_row(self) -> None:
        from attack.defenses import top_k_sparsify

        emb = torch.tensor([[5.0, 1.0, 4.0, 2.0], [0.1, 9.0, 0.2, 0.3]])
        out = top_k_sparsify(emb, k=2)
        assert (out != 0).sum(dim=1).tolist() == [2, 2]

    def test_keeps_the_largest_magnitudes(self) -> None:
        from attack.defenses import top_k_sparsify

        emb = torch.tensor([[5.0, 1.0, -4.0, 2.0]])
        out = top_k_sparsify(emb, k=2)
        assert out[0, 0] == 5.0 and out[0, 2] == -4.0
        assert out[0, 1] == 0.0 and out[0, 3] == 0.0

    def test_k_at_full_width_is_a_no_op(self) -> None:
        from attack.defenses import top_k_sparsify

        emb = torch.randn(4, 8, generator=torch.Generator().manual_seed(0))
        assert torch.allclose(top_k_sparsify(emb, k=8), emb)

    def test_rejects_a_nonsensical_k(self) -> None:
        from attack.defenses import top_k_sparsify

        emb = torch.randn(4, 8)
        for bad in (0, -1, 9):
            with pytest.raises(ValueError, match="k"):
                top_k_sparsify(emb, k=bad)


class TestRegistry:
    """Every defense must be interchangeable in the comparison harness."""

    def test_all_preserve_shape(self) -> None:
        from attack.defenses import DEFENSES

        emb = torch.randn(10, 32, generator=torch.Generator().manual_seed(0))
        for name, fn in DEFENSES.items():
            assert fn(emb).shape == emb.shape, name

    def test_none_is_the_identity(self) -> None:
        """The undefended control must be exactly the raw embedding, or the
        comparison has no zero point."""
        from attack.defenses import DEFENSES

        emb = torch.randn(10, 32, generator=torch.Generator().manual_seed(0))
        assert torch.equal(DEFENSES["none"](emb), emb)
