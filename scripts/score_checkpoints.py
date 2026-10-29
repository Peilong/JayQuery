#!/usr/bin/env python3
"""Recompute deployment utility metrics from saved checkpoints.

Precision within the top 10% of predicted risk is the deployment-facing
utility metric: an advisor can only contact a fixed number of students, so
ranking quality is what matters. The client's ``score_split`` handler does not
compute it, so it is replayed here from the saved models.

The replay reruns the live evaluation path, clipping eval embeddings when the
cut-layer channel was on and never noising them (``fl_client/common.py`` keeps
eval clip-only so the top model sees the training distribution). It also
re-derives AUC and refuses to write any result whose AUC disagrees with the
value the run recorded, which is what allows a replayed metric to be reported
alongside recorded ones.

Usage:
    python scripts/score_checkpoints.py
    python scripts/score_checkpoints.py --limit 4
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baselines.eval_utils import precision_at_k  # noqa: E402
from fl_server.top_model import TopModel  # noqa: E402
from privacy.noise_injection import clip_embedding_rows  # noqa: E402
from silo_a.bottom_model_a import BottomModelA  # noqa: E402
from silo_b.bottom_model_b import BottomModelB  # noqa: E402

#: Tolerance for the replay check. The forward pass is deterministic, so a
#: faithful replay should match to floating-point noise, not merely closely.
AUC_TOLERANCE = 1e-6


def eval_embedding(
    embedding: torch.Tensor, cut_enabled: bool, max_norm: float
) -> torch.Tensor:
    """Apply the live system's EVAL transform to a bottom-model output.

    Eval is clip-only: when the cut-layer channel is on the silo clips each
    row so the top model sees the training distribution, but adds no noise.
    Noising here would both diverge from the recorded metrics and describe a
    protocol the paper does not run.

    Args:
        embedding: Raw bottom-model output.
        cut_enabled: Whether that run used the cut-layer channel.
        max_norm: The per-row L2 clip the silo used.

    Returns:
        The embedding as the coordinator would have received it at eval.
    """
    if not cut_enabled:
        return embedding
    return clip_embedding_rows(embedding, max_norm)


def load_data(processed_dir: str | Path) -> dict:
    """Test-split features and labels for both silos."""
    processed_dir = Path(processed_dir)
    da = pickle.loads((processed_dir / "silo_a.pkl").read_bytes())
    db = pickle.loads((processed_dir / "silo_b.pkl").read_bytes())
    return {
        "Xa": torch.tensor(np.asarray(da["X_test"]), dtype=torch.float32),
        "Xb": torch.tensor(np.asarray(db["X_test"]), dtype=torch.float32),
        "y": np.asarray(da["y_test"]).astype(int),
    }


def score_one(ckpt: str | Path, data: dict, max_norm: float = 1.0) -> dict:
    """Replay one run's evaluation and return its utility metrics.

    Args:
        ckpt: Checkpoint directory holding the three model files and meta.json.
        data: Tensors from :func:`load_data`.
        max_norm: Per-row clip the silos used (not recorded in meta.json; the
            app-config default, which no sweep config overrode for this grid).

    Returns:
        Mapping with ``auc_roc`` and ``precision_at_10``.
    """
    from sklearn.metrics import roc_auc_score

    ckpt = Path(ckpt)
    meta = json.loads((ckpt / "meta.json").read_text())
    cut_enabled = "silo_a" in (meta.get("privacy_primitives") or {})

    sa = torch.load(ckpt / "silo_a.pt", map_location="cpu", weights_only=True)
    sb = torch.load(ckpt / "silo_b.pt", map_location="cpu", weights_only=True)
    tp = torch.load(ckpt / "top.pt", map_location="cpu", weights_only=True)
    dim = sa["encoder.7.weight"].shape[0]

    a = BottomModelA(input_dim=sa["encoder.0.weight"].shape[1], embedding_dim=dim,
                     hidden_dim=sa["encoder.0.weight"].shape[0])
    b = BottomModelB(input_dim=sb["layer1.0.weight"].shape[1], embedding_dim=dim,
                     hidden_dim=sb["layer1.0.weight"].shape[0])
    t = TopModel(embedding_dim=dim, hidden_dim=tp["net.0.weight"].shape[0])
    a.load_state_dict(sa)
    b.load_state_dict(sb)
    t.load_state_dict(tp)
    for m in (a, b, t):
        m.eval()

    with torch.no_grad():
        ea = eval_embedding(a(data["Xa"]), cut_enabled, max_norm)
        eb = eval_embedding(b(data["Xb"]), cut_enabled, max_norm)
        probs = torch.sigmoid(t(ea, eb).squeeze(1)).numpy()

    return {
        "auc_roc": float(roc_auc_score(data["y"], probs)),
        "precision_at_10": float(precision_at_k(data["y"], probs, 0.10)),
        "recorded_auc": float(meta["metrics"]["auc_roc"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoints", default="experiments/checkpoints")
    ap.add_argument("--processed-dir", default="data/processed/dedup_first")
    ap.add_argument("--out", default="experiments/deployment_metrics.json")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    data = load_data(args.processed_dir)
    ckpts = sorted(p.parent for p in Path(args.checkpoints).glob("*/meta.json"))
    if args.limit:
        ckpts = ckpts[: args.limit]

    rows, bad = [], []
    for ckpt in ckpts:
        r = score_one(ckpt, data)
        drift = abs(r["auc_roc"] - r["recorded_auc"])
        if drift > AUC_TOLERANCE:
            bad.append((ckpt.name, drift))
            continue
        rows.append({"run": ckpt.name, "precision_at_10": r["precision_at_10"],
                     "auc_roc": r["auc_roc"]})

    if bad:
        worst = max(d for _, d in bad)
        print(f"[score] REPLAY MISMATCH on {len(bad)}/{len(ckpts)} runs, "
              f"worst drift {worst:.2e}; refusing to write.", file=sys.stderr)
        for name, d in bad[:5]:
            print(f"  {name}: {d:.2e}", file=sys.stderr)
        raise SystemExit(1)

    Path(args.out).write_text(json.dumps({"rows": rows}, indent=1))
    print(f"[score] replayed {len(rows)} checkpoints, every AUC matched to "
          f"<{AUC_TOLERANCE:.0e}")
    print(f"[score] wrote {args.out}")


if __name__ == "__main__":
    main()
