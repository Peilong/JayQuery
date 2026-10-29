#!/usr/bin/env python3
"""Replay gradient-based label inference against every checkpoint.

The peer silo receives the gradient of the loss with respect to its own
embedding, and that gradient is computed from whatever labels the server
holds. Without the label channel those are the true labels; with it they are
the randomized ones. The attack should therefore recover the released labels
nearly perfectly in both cases, and the TRUE labels only at the mechanism's
keep rate.

Gradients are recomputed from the saved models rather than logged during
training: the server's backward is deterministic given the models, the batch
and the labels, so replaying it reproduces what Silo B received.

Two scores are reported per run. Accuracy against the true labels is what the
institution cares about. Accuracy against the released labels confirms the
attack itself works, so a low true-label score is attributable to the
mechanism rather than to a weak adversary.

Usage:
    python scripts/run_label_attack.py
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

from attack.label_inference_grad import infer_labels_from_gradients  # noqa: E402
from privacy.label_dp import randomized_response  # noqa: E402
from scripts.run_attack_grid import classify, load_models  # noqa: E402


def returned_gradient(
    bottom_a: torch.nn.Module,
    bottom_b: torch.nn.Module,
    top: torch.nn.Module,
    X_a: torch.Tensor,
    X_b: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Recompute the gradient the coordinator returns to Silo B.

    Mirrors the server's step: forward both bottoms, concatenate at the cut
    layer, take the loss against the labels the server holds, and read the
    gradient w.r.t. Silo B's embedding.
    """
    with torch.no_grad():
        emb_a = bottom_a(X_a)
        emb_b_raw = bottom_b(X_b)
    emb_a = emb_a.detach().requires_grad_(True)
    emb_b = emb_b_raw.detach().requires_grad_(True)
    loss = top.loss_fn(top(emb_a, emb_b).squeeze(1), labels.float())
    loss.backward()
    if emb_b.grad is None:
        raise RuntimeError("no gradient reached the Silo B embedding")
    return emb_b.grad.detach().clone()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoints", default="experiments/checkpoints")
    ap.add_argument("--processed-dir", default="data/processed/dedup_first")
    ap.add_argument("--out", default="experiments/label_attack_results.json")
    ap.add_argument("--split", default="train", choices=("train", "test"),
                    help="Split to attack; training rows are what the silo "
                         "actually receives gradients for.")
    args = ap.parse_args()

    d = Path(args.processed_dir)
    da = pickle.loads((d / "silo_a.pkl").read_bytes())
    db = pickle.loads((d / "silo_b.pkl").read_bytes())
    sfx = args.split
    X_a = torch.tensor(np.asarray(da[f"X_{sfx}"]), dtype=torch.float32)
    X_b = torch.tensor(np.asarray(db[f"X_{sfx}"]), dtype=torch.float32)
    y = np.asarray(da[f"y_{sfx}"]).astype(int)
    base_rate = float(y.mean())
    print(f"[label] {len(y)} {sfx} records | base rate {base_rate:.3f}")

    rows = []
    for ckpt in sorted(p.parent for p in Path(args.checkpoints).glob("*/meta.json")):
        meta = json.loads((ckpt / "meta.json").read_text())
        if meta["embedding_dim"] != 32:
            continue
        arm, eps = classify(meta)
        lp = meta.get("label_privacy") or {}

        # The server holds the released labels: true ones without the channel,
        # randomized ones with it, reproduced from the run's own seed.
        if lp:
            released = randomized_response(
                torch.tensor(y, dtype=torch.float32),
                epsilon=float(lp["epsilon"]), seed=int(lp["seed"]),
            ).numpy().astype(int)
        else:
            released = y

        a, b, t = load_models(ckpt)
        if lp:
            t.set_label_noise(float(lp["keep_prob"]))
        grad = returned_gradient(
            a, b, t, X_a, X_b, torch.tensor(released, dtype=torch.float32)
        )

        vs_true = infer_labels_from_gradients(grad, y, base_rate)
        vs_released = infer_labels_from_gradients(grad, released, float(released.mean()))
        rows.append({
            "run": ckpt.name, "arm": arm, "epsilon": eps,
            "stopped_early": meta["stopped_early"],
            "keep_prob": float(lp["keep_prob"]) if lp else None,
            "acc_true": vs_true.accuracy, "adv_true": vs_true.advantage,
            "baseline": vs_true.baseline,
            "acc_released": vs_released.accuracy,
            "agreement_released_true": float((released == y).mean()),
        })
        print(f"[label] {ckpt.name} {arm:10s} eps={str(eps):>4} "
              f"true={vs_true.accuracy:.3f} (adv {vs_true.advantage:+.3f}) "
              f"released={vs_released.accuracy:.3f}")

    Path(args.out).write_text(json.dumps({"split": sfx, "rows": rows}, indent=1))
    print(f"[label] wrote {args.out} ({len(rows)} runs)")


if __name__ == "__main__":
    main()
