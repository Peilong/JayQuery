"""Measure per-student embedding L2 norms for each silo, to choose the DP sensitivity S.

Owner: Alex Pellet

Why this exists
---------------
Cut-layer Gaussian noise is calibrated to a *sensitivity* S — the assumed maximum
L2 norm of any single student's embedding row. The noise scales linearly with S
(sigma = S * sqrt(2*ln(1.25/delta)) / epsilon), so:
  - S too high  -> too much noise -> poor utility.
  - S too low   -> almost every row clipped hard -> embeddings squashed, signal lost.
  - S "right"   -> sits around the bulk of the row norms; only the outlier tail clips.

This script runs each bottom model on its training data (no noise) and reports the
distribution of per-row L2 norms, so S can be set from real numbers (near the median /
a middle percentile) rather than the current placeholder of 1.0. The two silos have
different bottom models, so they are measured separately and may want different S
(set per-silo via SiloSpec.cut_noise_config_fn).

IMPORTANT — weights matter
--------------------------
Embedding norms depend on the model weights. With no --checkpoint-*, this measures
FRESH, UNTRAINED weights, which is only a rough starting estimate: norms shift during
training. For the definitive numbers, first run a short no-DP VFL run, then pass the
saved bottom-model weights via --checkpoint-a / --checkpoint-b.

Usage
-----
    python scripts/measure_embedding_norms.py
    python scripts/measure_embedding_norms.py --checkpoint-a data/.../bottom_a.pt \
                                               --checkpoint-b data/.../bottom_b.pt
    python scripts/measure_embedding_norms.py --epsilon 1.0 --delta 1e-5
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

# Make the repo root importable when run as `python scripts/measure_embedding_norms.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _seed(seed: int) -> None:
    """Seed torch so a train-mode (dropout-on) pass is reproducible run to run."""
    torch.manual_seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


def _collect_X(train_loader: object) -> torch.Tensor:
    """Materialize the full training feature tensor from a silo's train DataLoader."""
    xs: list[torch.Tensor] = []
    for batch in train_loader:  # type: ignore[attr-defined]
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        xs.append(x)
    return torch.cat(xs).float()


def _maybe_load_weights(model: torch.nn.Module, ckpt_path: str | None, tag: str) -> str:
    """Load a state_dict into model if a path is given. Returns a short status string."""
    if not ckpt_path:
        return "FRESH (untrained) weights — rough estimate only"
    path = Path(ckpt_path)
    if not path.exists():
        raise FileNotFoundError(f"[{tag}] checkpoint not found: {path}")
    obj = torch.load(path, map_location="cpu", weights_only=False)
    # Accept either a raw state_dict or a bundle dict containing one.
    state = obj
    if isinstance(obj, dict) and "state_dict" in obj:
        state = obj["state_dict"]
    if isinstance(obj, dict) and tag in obj:  # e.g. bundle keyed by "bottom_model_a"
        state = obj[tag]
    model.load_state_dict(state)
    return f"loaded weights from {path}"


def _row_norms(model: torch.nn.Module, X: torch.Tensor, *, train_mode: bool) -> torch.Tensor:
    """Per-row L2 norm of the clean embedding. train_mode=True keeps dropout active,
    matching what generate_embeddings actually sends; eval mode is the stable baseline."""
    model.train() if train_mode else model.eval()
    with torch.no_grad():
        emb = model(X)
    return emb.norm(p=2, dim=1)


def _report(name: str, norms: torch.Tensor, *, epsilon: float, delta: float) -> float:
    """Print the percentile table and an S recommendation. Returns the median (default S)."""
    pctiles = [50, 75, 90, 95, 99]
    qs = {p: torch.quantile(norms, p / 100).item() for p in pctiles}
    median = qs[50]
    p75 = qs[75]
    noise_coef = math.sqrt(2 * math.log(1.25 / delta)) / epsilon  # sigma = S * noise_coef

    print(f"\n=== {name} — per-row embedding L2 norm ({norms.numel()} students) ===")
    print(f"  min   : {norms.min().item():.3f}")
    print(f"  mean  : {norms.mean().item():.3f}")
    for p in pctiles:
        print(f"  p{p:<4}: {qs[p]:.3f}")
    print(f"  max   : {norms.max().item():.3f}")
    frac_over_median = (norms > median).float().mean().item()
    print(
        f"  -> at S=median={median:.3f}: {frac_over_median * 100:.0f}% of rows clipped, "
        f"sigma={median * noise_coef:.3f}  (at eps={epsilon}, delta={delta})"
    )
    print(
        f"  -> at S=p75={p75:.3f}: "
        f"{(norms > p75).float().mean().item() * 100:.0f}% of rows clipped, "
        f"sigma={p75 * noise_coef:.3f}"
    )
    return median


def _dims_from_ckpt(path: str | None, first_key: str, embed_key: str | None) -> dict | None:
    """Read (input_dim, hidden_dim, embedding_dim) off a checkpoint's Linear weights.

    CLI defaults cannot be trusted here: a checkpoint trained with a non-default
    `hidden-dim-a/b` will not load into a default-constructed model, and the failure is a
    wall of shape-mismatch errors rather than anything actionable. Returns None when no
    checkpoint is given (fresh-weights mode).
    """
    if not path:
        return None
    state = torch.load(Path(path), map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    linears = [(k, v) for k, v in state.items()
               if k.endswith(".weight") and getattr(v, "ndim", 0) == 2]
    if not linears:
        raise ValueError(f"no Linear weights in {path}")
    first = dict(linears)[first_key] if first_key in dict(linears) else linears[0][1]
    emb = dict(linears)[embed_key] if (embed_key and embed_key in dict(linears)) \
        else linears[-1][1]
    return {
        "input_dim": int(first.shape[1]),
        "hidden_dim": int(first.shape[0]),
        "embedding_dim": int(emb.shape[0]),
    }


def _build_models(args: argparse.Namespace) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Construct BottomModelA / BottomModelB, sized from the checkpoints when given."""
    from silo_a.bottom_model_a import BottomModelA
    from silo_b.bottom_model_b import BottomModelB

    dims_a = _dims_from_ckpt(args.checkpoint_a, "encoder.0.weight", None)
    dims_b = _dims_from_ckpt(args.checkpoint_b, "layer1.0.weight", "embed.weight")

    model_a = BottomModelA(**(dims_a or {
        "input_dim": args.input_dim_a,
        "embedding_dim": args.embedding_dim,
        "dropout_rate": args.dropout_rate,
    }))
    model_b = BottomModelB(**(dims_b or {
        "input_dim": args.input_dim_b,
        "embedding_dim": args.embedding_dim,
    }))
    return model_a, model_b


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Config defaults mirror pyproject.toml [tool.flwr.app.config].
    parser.add_argument("--input-dim-a", type=int, default=18)
    parser.add_argument("--input-dim-b", type=int, default=26)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--dropout-rate", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-a", default=None, help="Optional trained Bottom Model A weights.")
    parser.add_argument("--checkpoint-b", default=None, help="Optional trained Bottom Model B weights.")
    parser.add_argument(
        "--processed-dir",
        default="data/processed",
        help="Silo data directory — point at a variant, e.g. data/processed/dedup_first.",
    )
    parser.add_argument("--epsilon", type=float, default=1.0, help="Epsilon for the illustrative sigma column.")
    parser.add_argument("--delta", type=float, default=1e-5, help="Delta for the illustrative sigma column.")
    args = parser.parse_args()

    _seed(args.seed)

    try:
        from silo_a.data_loader_a import load_silo_a
        from silo_b.data_loader_b import load_silo_b
        model_a, model_b = _build_models(args)
    except Exception as e:  # noqa: BLE001 — surface import/construction problems plainly
        sys.exit(
            f"Could not import silo models/loaders: {e!r}\n"
            "Run this from the repo root, after prepare_silos.py has produced the "
            "data/processed/*.pkl files, in the jayquery environment."
        )

    status_a = _maybe_load_weights(model_a, args.checkpoint_a, "bottom_model_a")
    status_b = _maybe_load_weights(model_b, args.checkpoint_b, "bottom_model_b")

    train_a, _, _ = load_silo_a(processed_dir=args.processed_dir, batch_size=args.batch_size)
    train_b, _, _ = load_silo_b(processed_dir=args.processed_dir, batch_size=args.batch_size)
    X_a = _collect_X(train_a)
    X_b = _collect_X(train_b)

    print("Weights:")
    print(f"  Silo A: {status_a}")
    print(f"  Silo B: {status_b}")
    if not (args.checkpoint_a or args.checkpoint_b):
        print(
            "\nNOTE: measuring UNTRAINED weights — treat as a rough starting point. "
            "Re-run with --checkpoint-a/--checkpoint-b from a no-DP run for the real numbers."
        )

    # train mode = the distribution actually sent (dropout on); eval = stable baseline.
    for silo, model, X in [("Silo A", model_a, X_a), ("Silo B", model_b, X_b)]:
        norms_train = _row_norms(model, X, train_mode=True)
        _report(f"{silo} [train mode / dropout ON — what gets sent]", norms_train, epsilon=args.epsilon, delta=args.delta)
        norms_eval = _row_norms(model, X, train_mode=False)
        _report(f"{silo} [eval mode / dropout OFF — baseline]", norms_eval, epsilon=args.epsilon, delta=args.delta)

    print(
        "\nHow to choose S  ** MEASURED 2026-07-30 — read this before using the numbers above **\n"
        "  - SMALL S WINS for the cut layer. Measured at eps=1.0, T=200, 3 seeds:\n"
        "        S=1.0 (all rows clipped)  -> AUC 0.8453  (sd 0.004)\n"
        "        S=median (5.6/4.0)        -> AUC 0.6022  (sd 0.066)\n"
        "        S=p75    (6.7/6.3)        -> AUC 0.5995  (sd 0.058)\n"
        "    Moving to the median costs -0.243 AUC and inflates variance ~16x.\n"
        "  - WHY the median heuristic FAILS here (it is gradient-clipping intuition applied\n"
        "    where it does not hold): the embedding magnitude SATURATES at its natural row\n"
        "    norm (~4.3) while noise keeps scaling linearly with S. So SNR = min(S, norm)/(sigma*S):\n"
        "        S <= min row norm -> SNR = 1/sigma           (flat, best)\n"
        "        S >  row norm     -> SNR = norm/(sigma*S)    (falls as S grows)\n"
        "    An oversized S is pure waste — the same failure mode as an oversized gradient\n"
        "    clip, in the opposite direction from what the percentiles suggest.\n"
        "  - The percentile table above is still useful for KNOWING where the norms sit;\n"
        "    just do not read it as 'set S here'.\n"
        "  - Set per-silo S via SiloSpec.cut_noise_config_fn (run-config keys\n"
        "    cut-noise-max-norm-a / -b) — the two silos differ.\n"
        "  - These are UNTRAINED norms unless you passed checkpoints; re-measure after a\n"
        "    no-DP run for the values you actually ship."
    )


if __name__ == "__main__":
    main()