"""Drive all three JayQuery privacy attacks against a trained checkpoint bundle.

Owner: Wyatt Wilfong

What this does
--------------
Given a trained VFL checkpoint (the ``experiments/checkpoints/<eps_tag>/`` bundle Alex's
run produces: ``silo_a.pt`` + ``silo_b.pt`` + ``top.pt``), this script mounts the
honest-but-curious server / man-in-the-middle threat model and runs the three attacks
that live in ``attack/``:

  1. Model inversion (iDLG)      — attack/inversion.py:run_inversion_attack
  2. Label inference             — attack/label_inference.py:infer_labels_from_gradient
  3. Gradient-norm monitoring    — attack/gradient_monitor.py:GradientNormMonitor

Self-contained by design
------------------------
The attack drivers here take a single pandas ``DataFrame`` of the aligned test students
(Silo A features, Silo B features, and the dropout label, one row per student) rather
than reaching into ``data/processed`` themselves. That keeps the module importable and
lets Alex feed his own evaluation frame straight in::

    from scripts.run_attack import run_all_attacks
    results = run_all_attacks(df, checkpoint_dir="experiments/checkpoints", eps_tag="eps_1.0")

Run as a CLI, it assembles that frame from ``load_silo_a`` / ``load_silo_b`` (falling back
to the synthetic generators when the processed pickles are absent) so it runs anywhere.

Model dimensions are inferred from the checkpoint tensors, not hard-coded, so this stays
correct across the Silo A (18) / Silo B (27) feature counts and any embedding-dim change.

Usage
-----
    python scripts/run_attack.py --checkpoint-dir experiments/checkpoints --eps-tag eps_1.0
    python scripts/run_attack.py --eps-tag eps_inf --n-students 100 --synthetic
    python scripts/run_attack.py --eps-tag eps_1.0 --results-csv experiments/attack_summary.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

# Make the repo root importable when run as `python scripts/run_attack.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attack.gradient_monitor import GradientNormMonitor  # noqa: E402
from attack.inversion import IMD_BAND_COL_IDX, run_inversion_attack  # noqa: E402
from attack.label_inference import infer_labels_from_gradient  # noqa: E402
from fl_server.top_model import TopModel  # noqa: E402
from silo_a.bottom_model_a import BottomModelA  # noqa: E402
from silo_b.bottom_model_b import BottomModelB  # noqa: E402

# DataFrame schema. Silo A features -> "a0".."a{n-1}", Silo B -> "b0".."b{m-1}", label -> "label".
A_PREFIX = "a"
B_PREFIX = "b"
LABEL_COL = "label"


def _seed(seed: int) -> None:
    """Seed torch (and numpy if present) so attack runs are reproducible.

    Args:
        seed: Base random seed.
    """
    torch.manual_seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


# --------------------------------------------------------------------------------------
# Checkpoint loading (dims inferred from the saved tensors, not hard-coded)
# --------------------------------------------------------------------------------------
def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    """Load a raw state_dict, tolerating either a bare dict or a bundle wrapping one.

    Args:
        path: Path to a .pt file.

    Returns:
        The state_dict mapping parameter names to tensors.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    return obj


def _first_linear_in_dim(state: dict[str, torch.Tensor], weight_key: str) -> int:
    """Return the input dimension of a Linear layer from its weight tensor.

    Args:
        state: A model state_dict.
        weight_key: Key of the Linear weight, shape (out_features, in_features).

    Returns:
        in_features for that layer.
    """
    return int(state[weight_key].shape[1])


def load_models(
    checkpoint_dir: str | Path, eps_tag: str
) -> tuple[BottomModelA, BottomModelB, TopModel]:
    """Load the bottom + top models for one privacy level, in eval mode.

    Dimensions are read off the checkpoint tensors so this stays correct regardless of
    the configured Silo A / Silo B feature counts or embedding dim.

    Args:
        checkpoint_dir: Root dir holding per-epsilon subdirs (e.g. ``experiments/checkpoints``).
        eps_tag: Subdir tag, ``"eps_inf"`` for the no-DP run or ``"eps_<epsilon>"`` for a DP run.

    Returns:
        Tuple of (BottomModelA, BottomModelB, TopModel), all in eval mode with loaded weights.
    """
    ckpt = Path(checkpoint_dir) / eps_tag
    sd_a = _load_state_dict(ckpt / "silo_a.pt")
    sd_b = _load_state_dict(ckpt / "silo_b.pt")
    sd_top = _load_state_dict(ckpt / "top.pt")

    # Silo A: encoder[0] is the first Linear; encoder[-1] emits the embedding.
    in_a = _first_linear_in_dim(sd_a, "encoder.0.weight")
    emb_dim = int(sd_a["encoder.6.weight"].shape[0])
    # Silo B: layer1[0] is the first Linear.
    in_b = _first_linear_in_dim(sd_b, "layer1.0.weight")

    model_a = BottomModelA(input_dim=in_a, embedding_dim=emb_dim)
    model_b = BottomModelB(input_dim=in_b, embedding_dim=emb_dim)
    top = TopModel(embedding_dim=emb_dim)

    model_a.load_state_dict(sd_a)
    model_b.load_state_dict(sd_b)
    top.load_state_dict(sd_top)
    for m in (model_a, model_b, top):
        m.eval()
    return model_a, model_b, top


def build_fresh_models(
    input_dim_a: int, input_dim_b: int, embedding_dim: int
) -> tuple[BottomModelA, BottomModelB, TopModel]:
    """Construct untrained models (for a smoke run when no checkpoint exists).

    Args:
        input_dim_a: Silo A feature count.
        input_dim_b: Silo B feature count.
        embedding_dim: Shared embedding dimension.

    Returns:
        Tuple of (BottomModelA, BottomModelB, TopModel) in eval mode with random weights.
    """
    model_a = BottomModelA(input_dim=input_dim_a, embedding_dim=embedding_dim)
    model_b = BottomModelB(input_dim=input_dim_b, embedding_dim=embedding_dim)
    top = TopModel(embedding_dim=embedding_dim)
    for m in (model_a, model_b, top):
        m.eval()
    return model_a, model_b, top


# --------------------------------------------------------------------------------------
# DataFrame assembly + tensor extraction
# --------------------------------------------------------------------------------------
def build_dataframe(
    processed_dir: str | Path = "data/processed",
    batch_size: int = 64,
    seed: int = 42,
    synthetic: bool = False,
) -> pd.DataFrame:
    """Assemble the aligned test-set DataFrame the attack drivers consume.

    Uses the real PSI-aligned test loaders when the processed pickles exist; otherwise
    (or with ``synthetic=True``) falls back to the synthetic generators so the script
    runs without OULAD. Silo A and Silo B rows are aligned by construction in both paths.

    Args:
        processed_dir: Directory containing silo_a.pkl / silo_b.pkl.
        batch_size: Loader batch size (only affects iteration, not row order).
        seed: Random seed for the synthetic fallback.
        synthetic: Force the synthetic generators even if real data is present.

    Returns:
        DataFrame with columns a0..a{n-1}, b0..b{m-1}, and "label", one row per student.
    """
    from silo_a.data_loader_a import load_silo_a, make_synthetic_loader_a
    from silo_b.data_loader_b import load_silo_b, make_synthetic_loader_b

    use_synthetic = synthetic
    if not synthetic:
        try:
            _, _, test_a = load_silo_a(processed_dir=processed_dir, batch_size=batch_size)
            _, _, test_b = load_silo_b(processed_dir=processed_dir, batch_size=batch_size)
        except FileNotFoundError as exc:
            print(f"[data] processed data unavailable ({exc}); using synthetic fallback.")
            use_synthetic = True
    if use_synthetic:
        _, _, test_a = make_synthetic_loader_a(batch_size=batch_size, seed=seed)
        _, _, test_b = make_synthetic_loader_b(batch_size=batch_size, seed=seed)

    xa = torch.cat([xb[0] for xb in test_a]).float()
    ya = torch.cat([xb[1] for xb in test_a]).float()
    xb_ = torch.cat([bb[0] for bb in test_b]).float()

    n = min(xa.shape[0], xb_.shape[0])
    xa, ya, xb_ = xa[:n], ya[:n], xb_[:n]

    cols = {f"{A_PREFIX}{i}": xa[:, i].numpy() for i in range(xa.shape[1])}
    cols.update({f"{B_PREFIX}{j}": xb_[:, j].numpy() for j in range(xb_.shape[1])})
    cols[LABEL_COL] = ya.numpy()
    return pd.DataFrame(cols)


def _tensors_from_df(df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split the attack DataFrame back into (X_a, X_b, y) float tensors.

    Args:
        df: DataFrame in the schema produced by build_dataframe.

    Returns:
        Tuple of (X_a [N, n_a], X_b [N, n_b], y [N]) as float tensors.
    """
    a_cols = sorted(
        (c for c in df.columns if c.startswith(A_PREFIX) and c[len(A_PREFIX):].isdigit()),
        key=lambda c: int(c[len(A_PREFIX):]),
    )
    b_cols = sorted(
        (c for c in df.columns if c.startswith(B_PREFIX) and c[len(B_PREFIX):].isdigit()),
        key=lambda c: int(c[len(B_PREFIX):]),
    )
    xa = torch.tensor(df[a_cols].to_numpy(), dtype=torch.float32)
    xb = torch.tensor(df[b_cols].to_numpy(), dtype=torch.float32)
    y = torch.tensor(df[LABEL_COL].to_numpy(), dtype=torch.float32)
    return xa, xb, y


# --------------------------------------------------------------------------------------
# Attack drivers
# --------------------------------------------------------------------------------------
def run_inversion(
    df: pd.DataFrame,
    model_a: BottomModelA,
    n_students: int = 100,
    num_iters: int = 200,
    lr: float = 0.1,
) -> dict:
    """Run the iDLG inversion attack over the first ``n_students`` rows.

    For each student the honest-but-curious server observes ``model_a(x)``; the attack
    tries to reconstruct the row and recover the true imd_band (feature 0).

    Args:
        df: Attack DataFrame (Silo A columns are used).
        model_a: Trained BottomModelA in eval mode.
        n_students: Number of students to attack.
        num_iters: L-BFGS iterations per attempt.
        lr: L-BFGS learning rate.

    Returns:
        Dict: {attack, n_students, asr, mean_final_loss}.
    """
    xa, _, _ = _tensors_from_df(df)
    n = min(n_students, xa.shape[0])
    successes = 0
    losses: list[float] = []
    for i in range(n):
        row = xa[i].unsqueeze(0)
        real_embedding = model_a(row)
        true_imd_band = int(row[0, IMD_BAND_COL_IDX].round().item())
        result = run_inversion_attack(
            model_a, real_embedding, true_imd_band, num_iters=num_iters, lr=lr
        )
        successes += int(result["success"])
        losses.append(float(result["final_loss"]))
    return {
        "attack": "inversion",
        "n_students": n,
        "asr": successes / n if n else 0.0,
        "mean_final_loss": sum(losses) / len(losses) if losses else 0.0,
    }


def run_label_inference(
    df: pd.DataFrame,
    model_a: BottomModelA,
    model_b: BottomModelB,
    top: TopModel,
) -> dict:
    """Run the gradient-based label-inference attack over the whole frame.

    Reproduces what an honest-but-curious server sees: it forwards both embeddings,
    computes the training loss, backprops to obtain ``grad_emb_b`` (the gradient it
    returns to Silo B), and asks the attack to recover the labels from that signal.

    Args:
        df: Attack DataFrame (all columns used).
        model_a: Trained BottomModelA in eval mode.
        model_b: Trained BottomModelB in eval mode.
        top: Trained TopModel in eval mode.

    Returns:
        Dict: {attack, n_students, accuracy}.
    """
    xa, xb, y = _tensors_from_df(df)

    emb_a = model_a(xa).detach()
    emb_b = model_b(xb).detach().requires_grad_(True)
    logits = top(emb_a, emb_b)
    loss = top.loss_fn(logits, y.unsqueeze(1))
    loss.backward()
    grad_emb_b = emb_b.grad.detach()
    predicted_probs = torch.sigmoid(logits.detach()).reshape(-1)

    inferred = infer_labels_from_gradient(grad_emb_b, top, predicted_probs)
    accuracy = (inferred == y.long()).float().mean().item()
    return {"attack": "label_inference", "n_students": int(y.shape[0]), "accuracy": accuracy}


def run_gradient_monitoring(
    df: pd.DataFrame,
    model_a: BottomModelA,
    model_b: BottomModelB,
    top: TopModel,
    batch_size: int = 64,
) -> dict:
    """Feed per-batch returned-gradient norms through the GradientNormMonitor defense.

    Each batch stands in for one training round: we compute the gradient the server
    would return and update the running-mean anomaly monitor with its norm.

    Args:
        df: Attack DataFrame (all columns used).
        model_a: Trained BottomModelA in eval mode.
        model_b: Trained BottomModelB in eval mode.
        top: Trained TopModel in eval mode.
        batch_size: Rows per simulated round.

    Returns:
        Dict: {attack, n_rounds, n_flagged, norms}.
    """
    xa, xb, y = _tensors_from_df(df)
    monitor = GradientNormMonitor()
    n = xa.shape[0]
    norms: list[float] = []
    flagged = 0
    for start in range(0, n, batch_size):
        sl = slice(start, start + batch_size)
        emb_a = model_a(xa[sl]).detach()
        emb_b = model_b(xb[sl]).detach().requires_grad_(True)
        logits = top(emb_a, emb_b)
        top.loss_fn(logits, y[sl].unsqueeze(1)).backward()
        norm, is_anomalous = monitor.update(emb_b.grad.detach())
        norms.append(norm)
        flagged += int(bool(is_anomalous))
    return {
        "attack": "gradient_monitor",
        "n_rounds": len(norms),
        "n_flagged": flagged,
        "norms": norms,
    }


def run_all_attacks(
    df: pd.DataFrame,
    checkpoint_dir: str | Path | None = None,
    eps_tag: str = "eps_inf",
    n_students: int = 100,
    models: tuple[BottomModelA, BottomModelB, TopModel] | None = None,
) -> dict:
    """Run all three attacks against one checkpoint and return their metrics.

    Each attack is isolated: a failure in one (e.g. an unimplemented scaffold) is
    reported and does not abort the others.

    Args:
        df: Attack DataFrame (schema from build_dataframe).
        checkpoint_dir: Root checkpoint dir; ignored if ``models`` is supplied.
        eps_tag: Which epsilon subdir to load.
        n_students: Students to attack for inversion.
        models: Optional pre-built (model_a, model_b, top) to bypass checkpoint loading.

    Returns:
        Dict keyed by attack name; each value is that attack's result dict or
        ``{"error": <message>}`` if it raised.
    """
    if models is None:
        if checkpoint_dir is None:
            raise ValueError("Provide either checkpoint_dir or models.")
        model_a, model_b, top = load_models(checkpoint_dir, eps_tag)
    else:
        model_a, model_b, top = models

    results: dict = {"eps_tag": eps_tag}
    for name, fn in (
        ("inversion", lambda: run_inversion(df, model_a, n_students=n_students)),
        ("label_inference", lambda: run_label_inference(df, model_a, model_b, top)),
        ("gradient_monitor", lambda: run_gradient_monitoring(df, model_a, model_b, top)),
    ):
        try:
            results[name] = fn()
        except Exception as exc:  # noqa: BLE001 — one broken attack must not sink the rest
            results[name] = {"attack": name, "error": f"{type(exc).__name__}: {exc}"}
    return results


# --------------------------------------------------------------------------------------
# Reporting + CLI
# --------------------------------------------------------------------------------------
def _print_report(results: dict) -> None:
    """Pretty-print the results dict from run_all_attacks.

    Args:
        results: The dict returned by run_all_attacks.
    """
    print(f"\n==== Attack report — {results.get('eps_tag', '?')} ====")

    inv = results.get("inversion", {})
    if "error" in inv:
        print(f"  [inversion]        FAILED: {inv['error']}")
    else:
        print(
            f"  [inversion]        ASR={inv['asr']:.3f} over {inv['n_students']} students "
            f"(mean final loss {inv['mean_final_loss']:.4g})"
        )

    lab = results.get("label_inference", {})
    if "error" in lab:
        print(f"  [label_inference]  FAILED: {lab['error']}")
    else:
        print(
            f"  [label_inference]  accuracy={lab['accuracy']:.3f} over {lab['n_students']} students"
        )

    gm = results.get("gradient_monitor", {})
    if "error" in gm:
        print(f"  [gradient_monitor] FAILED: {gm['error']}")
    else:
        print(f"  [gradient_monitor] flagged {gm['n_flagged']}/{gm['n_rounds']} rounds as anomalous")


def _write_csv(results: dict, path: str | Path) -> None:
    """Write a one-row-per-attack summary CSV of the scalar metrics.

    Args:
        results: The dict returned by run_all_attacks.
        path: Output CSV path.
    """
    rows = []
    for name in ("inversion", "label_inference", "gradient_monitor"):
        r = results.get(name, {})
        rows.append(
            {
                "eps_tag": results.get("eps_tag", ""),
                "attack": name,
                "asr": r.get("asr", ""),
                "accuracy": r.get("accuracy", ""),
                "n_flagged": r.get("n_flagged", ""),
                "n_rounds": r.get("n_rounds", ""),
                "n_students": r.get("n_students", ""),
                "error": r.get("error", ""),
            }
        )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nWrote summary -> {out}")


def main() -> None:
    """CLI entry point: assemble the frame, load the checkpoint, run + report attacks."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint-dir", default="experiments/checkpoints")
    parser.add_argument("--eps-tag", default="eps_inf", help='e.g. "eps_inf" or "eps_1.0".')
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--n-students", type=int, default=100, help="Students to attack (inversion).")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic data (no OULAD).")
    parser.add_argument(
        "--fresh-models",
        action="store_true",
        help="Skip the checkpoint and use untrained models (smoke test only).",
    )
    parser.add_argument("--results-csv", default=None, help="Optional summary CSV output path.")
    args = parser.parse_args()

    _seed(args.seed)

    df = build_dataframe(
        processed_dir=args.processed_dir,
        batch_size=args.batch_size,
        seed=args.seed,
        synthetic=args.synthetic,
    )
    print(f"[data] assembled {len(df)} students, {df.shape[1] - 1} features (+label).")

    models = None
    if args.fresh_models:
        xa, xb, _ = _tensors_from_df(df)
        models = build_fresh_models(xa.shape[1], xb.shape[1], embedding_dim=32)
        print("[models] using FRESH untrained weights (smoke test).")

    results = run_all_attacks(
        df,
        checkpoint_dir=args.checkpoint_dir,
        eps_tag=args.eps_tag,
        n_students=args.n_students,
        models=models,
    )
    _print_report(results)
    if args.results_csv:
        _write_csv(results, args.results_csv)


if __name__ == "__main__":
    main()
