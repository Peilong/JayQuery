#!/usr/bin/env python3
"""Do heuristic cut-layer defenses stop the same attacker that DP stops?

Section II claims that hashing or transforming the embedding bounds a
particular attack rather than a class of adversaries. This tests that claim
against the paper's own adversary instead of asserting it.

The comparison is deliberately narrow so only the defense varies. It takes the
NO-DP checkpoints, where nothing was added to the embedding during training,
applies each defense to what the coordinator receives, and runs the identical
learned inverse on each. The attacker's auxiliary data passes through the same
defense, which is what a real attacker collecting traffic would have, so this
measures the defense rather than a train/test mismatch.

The DP column is the cut-layer mechanism at the same operating point, so the
table reads as: here is what each defense costs the attacker, under one
attacker.

Usage:
    python scripts/run_defense_comparison.py
    python scripts/run_defense_comparison.py --attribute gender
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attack.defenses import DEFENSES  # noqa: E402
from attack.embedding_inversion import train_inverse_model  # noqa: E402
from privacy.noise_injection import (  # noqa: E402
    add_gaussian_noise,
    clip_embedding_rows,
)
from scripts.run_attack_grid import (  # noqa: E402
    ATTRIBUTES,
    CUT_MAX_NORM,
    classify,
    load_data,
    load_models,
)


def _score(inverse: torch.nn.Module, emb: torch.Tensor, y: torch.Tensor,
           n_classes: int) -> tuple[float, float]:
    """(accuracy, lift over the majority class) for a trained inverse."""
    with torch.no_grad():
        pred = inverse(emb).argmax(dim=1)
    acc = float((pred == y).float().mean())
    base = float(torch.bincount(y, minlength=n_classes).max()) / float(len(y))
    return acc, acc - base


def attack_defense(
    bottom_a: torch.nn.Module,
    data: dict,
    transform: Callable[[torch.Tensor], torch.Tensor],
    epochs: int,
    seed: int,
) -> tuple[float, float]:
    """Train and score the inverse against one defended channel."""
    with torch.no_grad():
        aux = transform(bottom_a(data["Xa_aux"]))
        tgt = transform(bottom_a(data["Xa_tgt"]))
    inverse = train_inverse_model(
        aux, data["imd_aux"], n_classes=data["n_classes"], epochs=epochs, seed=seed
    )
    return _score(inverse, tgt, data["imd_tgt"], data["n_classes"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoints", default="experiments/checkpoints")
    ap.add_argument("--processed-dir", default="data/processed/dedup_first")
    ap.add_argument("--attribute", default="imd_band", choices=sorted(ATTRIBUTES))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dp-sigma", type=float, default=3.0747,
                    help="Cut-layer noise multiplier to compare against "
                         "(default: the grid's realized sigma at eps=1).")
    ap.add_argument("--out", default="experiments/defense_comparison.json")
    args = ap.parse_args()

    data = load_data(args.processed_dir, seed=args.seed, attribute=args.attribute)

    # No-DP runs only: their embeddings were never noised during training, so
    # the defense under test is the only thing altering what is sent.
    ckpts = []
    for p in sorted(Path(args.checkpoints).glob("*/meta.json")):
        meta = json.loads(p.read_text())
        if classify(meta)[0] == "none" and meta["embedding_dim"] == 32:
            ckpts.append(p.parent)
    print(f"[defense] {len(ckpts)} no-DP checkpoints | attribute={args.attribute} "
          f"({data['n_classes']} classes)")

    def dp_transform(e: torch.Tensor) -> torch.Tensor:
        return add_gaussian_noise(
            clip_embedding_rows(e, CUT_MAX_NORM),
            noise_std=args.dp_sigma * CUT_MAX_NORM,
        )

    arms = dict(DEFENSES)
    arms[f"DP cut-layer (sigma={args.dp_sigma:g})"] = dp_transform

    rows = []
    for ckpt in ckpts:
        a, _b, _t = load_models(ckpt)
        for name, fn in arms.items():
            torch.manual_seed(args.seed)      # DP arm draws noise
            acc, lift = attack_defense(a, data, fn, args.epochs, args.seed)
            rows.append({"run": ckpt.name, "defense": name,
                         "accuracy": acc, "lift": lift})
        print(f"[defense] {ckpt.name} done")

    Path(args.out).write_text(json.dumps(
        {"attribute": args.attribute, "rows": rows}, indent=1))

    import statistics as st
    print(f"\n{'defense':34s} {'acc':>7} {'lift':>8}")
    print("-" * 51)
    for name in arms:
        v = [r for r in rows if r["defense"] == name]
        print(f"{name:34s} {st.mean(r['accuracy'] for r in v):7.4f} "
              f"{st.mean(r['lift'] for r in v):+8.4f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
