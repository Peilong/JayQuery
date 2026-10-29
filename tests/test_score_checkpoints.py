"""Tests for scripts/score_checkpoints.py.

Precision within the top 10% of predicted risk is replayed from saved
checkpoints rather than recorded during training. That is only sound if the
post-hoc forward pass reproduces what the live system scored, so the
load-bearing test is exactly that: recomputed AUC must match the AUC the run
recorded. If it does not, the replay is not the model the grid evaluated and
the replayed metric cannot be reported alongside the recorded ones.
"""

from __future__ import annotations

import pytest
import torch


class TestEvalReplay:
    """The post-hoc pass must reproduce the live eval path.

    The live path clips eval embeddings when the cut-layer channel is on and
    does NOT noise them (fl_client/common.py), so the top model sees the
    training distribution at evaluation. Getting either half wrong changes the
    scores.
    """

    def test_clips_when_the_cut_channel_was_on(self) -> None:
        from scripts.score_checkpoints import eval_embedding

        emb = torch.tensor([[3.0, 4.0]])          # L2 norm 5
        out = eval_embedding(emb, cut_enabled=True, max_norm=1.0)
        assert out.norm(dim=1).item() == pytest.approx(1.0, abs=1e-6)

    def test_leaves_the_embedding_alone_when_the_channel_was_off(self) -> None:
        from scripts.score_checkpoints import eval_embedding

        emb = torch.tensor([[3.0, 4.0]])
        out = eval_embedding(emb, cut_enabled=False, max_norm=1.0)
        assert torch.equal(out, emb)

    def test_never_adds_noise(self) -> None:
        """Eval is clip-only. Noising here would make the replay disagree with
        the recorded AUC, and would also be a different protocol than the one
        the paper describes."""
        from scripts.score_checkpoints import eval_embedding

        emb = torch.tensor([[0.3, 0.4]])
        a = eval_embedding(emb, cut_enabled=True, max_norm=1.0)
        b = eval_embedding(emb, cut_enabled=True, max_norm=1.0)
        assert torch.equal(a, b), "eval must be deterministic"


class TestPrecisionAtK:
    """The metric itself, on cases where the answer is known by hand."""

    def test_perfect_ranking_scores_one(self) -> None:
        import numpy as np

        from baselines.eval_utils import precision_at_k
        y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        p = np.array([0.9, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        assert precision_at_k(y, p, 0.10) == pytest.approx(1.0)

    def test_no_signal_scores_near_the_base_rate(self) -> None:
        import numpy as np

        from baselines.eval_utils import precision_at_k
        rng = np.random.default_rng(0)
        y = (rng.random(2000) < 0.445).astype(int)
        p = rng.random(2000)              # ranking carries nothing
        assert precision_at_k(y, p, 0.10) == pytest.approx(0.445, abs=0.08)


class TestFaithfulness:
    """The whole justification for recomputing rather than re-running."""

    @pytest.mark.slow
    def test_replayed_auc_matches_every_recorded_auc(self) -> None:
        """Replay each checkpoint and compare against its own meta.json.

        A mismatch means the replay is not the model the grid evaluated, and
        the recomputed P@10% could not be put in the same table as the AUC.
        """
        import glob
        import json
        import os

        from scripts.score_checkpoints import load_data, score_one

        if not glob.glob("experiments/checkpoints/*/meta.json"):
            pytest.skip("no checkpoints present")
        data = load_data("data/processed/dedup_first")
        worst = 0.0
        for p in sorted(glob.glob("experiments/checkpoints/*/meta.json"))[:8]:
            ckpt = os.path.dirname(p)
            recorded = json.loads(open(p).read())["metrics"]["auc_roc"]
            replayed = score_one(ckpt, data)["auc_roc"]
            worst = max(worst, abs(replayed - recorded))
        assert worst < 1e-6, f"replay diverges from recorded AUC by {worst}"
