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
import json
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
from privacy.noise_injection import add_gaussian_noise, clip_embedding_rows  # noqa: E402
from silo_a.bottom_model_a import BottomModelA  # noqa: E402
from silo_b.bottom_model_b import BottomModelB  # noqa: E402

# DataFrame schema. Silo A features -> "a0".."a{n-1}", Silo B -> "b0".."b{m-1}", label -> "label".
A_PREFIX = "a"
B_PREFIX = "b"
LABEL_COL = "label"

# Silo A's column layout, from scripts/prepare_silos.py:build_silo_a. Named here so the
# inversion attack can report WHICH attributes it recovers rather than only imd_band —
# "we recovered disability status and gender" is the privacy claim that actually matters
# for a FERPA-proxy dataset.
#
#   name -> (column index, number of ordinal levels)
SILO_A_CATEGORICALS: dict[str, tuple[int, int]] = {
    "imd_band": (0, 10),           # OrdinalEncoder over 10 deprivation bands
    "highest_education": (1, 5),   # mapped 0..4
    "age_band": (2, 3),            # OrdinalEncoder over 3 bands
    "gender": (3, 2),              # 1 = F
    "disability": (4, 2),          # 1 = Y
}
# studied_credits (index 5) is StandardScaler-scaled continuous; region is a drop-first
# one-hot block occupying the remaining columns.
SILO_A_CONTINUOUS = (5,)
SILO_A_REGION_BLOCK = slice(6, 18)


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


def _linear_weights(state: dict[str, torch.Tensor], prefix: str = "") -> list[tuple[str, tuple]]:
    """All 2-D (Linear) weight tensors under ``prefix``, in state_dict order."""
    return [
        (k, tuple(v.shape))
        for k, v in state.items()
        if k.startswith(prefix) and k.endswith(".weight") and v.ndim == 2
    ]


def _infer_dims(state: dict[str, torch.Tensor], prefix: str) -> tuple[int, int, int]:
    """Infer (input_dim, hidden_dim, output_dim) from a submodule's Linear layers.

    Reads the FIRST Linear's in/out features and the LAST Linear's out features, rather
    than indexing a fixed layer name. Hard-coded indices break silently whenever the
    architecture gains a layer: this previously read ``encoder.6.weight`` for the Silo A
    embedding when the emitting layer had become ``encoder.7``, so `load_models` raised
    KeyError on every current checkpoint.

    Args:
        state: A model state_dict.
        prefix: Submodule prefix to search (e.g. ``"encoder"``, ``""`` for the whole dict).

    Returns:
        Tuple of (in_features of the first Linear, out_features of the first Linear,
        out_features of the last Linear).

    Raises:
        ValueError: If no 2-D weight tensors are found under ``prefix``.
    """
    linears = _linear_weights(state, prefix)
    if not linears:
        raise ValueError(
            f"no Linear weights found under prefix {prefix!r}; "
            f"available keys: {sorted(state)[:8]}"
        )
    (_, first), (_, last) = linears[0], linears[-1]
    return int(first[1]), int(first[0]), int(last[0])


def load_models(
    checkpoint_dir: str | Path, eps_tag: str
) -> tuple[BottomModelA, BottomModelB, TopModel]:
    """Load the bottom + top models for one privacy level, in eval mode.

    Every dimension — input, hidden, and embedding — is inferred from the checkpoint
    tensors, so this stays correct across feature-count, hidden-width and embedding-dim
    changes. (It previously hard-coded a layer index for the Silo A embedding and did not
    infer hidden_dim at all, so it failed on any run whose widths differed from the model
    defaults.)

    Args:
        checkpoint_dir: Root dir holding per-run subdirs (e.g. ``experiments/checkpoints``).
        eps_tag: Subdir name — the run-tag used at training time
            (``"eps_inf"``, ``"eps_1.0"``, or e.g. ``"d3_dedup_first_all_dp_s42"``).

    Returns:
        Tuple of (BottomModelA, BottomModelB, TopModel), all in eval mode with loaded weights.
    """
    ckpt = Path(checkpoint_dir) / eps_tag
    sd_a = _load_state_dict(ckpt / "silo_a.pt")
    sd_b = _load_state_dict(ckpt / "silo_b.pt")
    sd_top = _load_state_dict(ckpt / "top.pt")

    in_a, hid_a, emb_a = _infer_dims(sd_a, "encoder")
    in_b, hid_b, emb_b = _infer_dims(sd_b, "layer1")
    # Silo B's embedding comes from its own `embed` Linear, not the layer1 stack.
    emb_b = int(sd_b["embed.weight"].shape[0]) if "embed.weight" in sd_b else emb_b
    if emb_a != emb_b:
        raise ValueError(f"embedding dim mismatch: silo_a={emb_a}, silo_b={emb_b}")
    # Top model: first Linear takes the concatenated pair, so hidden width is its output.
    _, hid_top, _ = _infer_dims(sd_top, "net")

    model_a = BottomModelA(input_dim=in_a, embedding_dim=emb_a, hidden_dim=hid_a)
    model_b = BottomModelB(input_dim=in_b, embedding_dim=emb_b, hidden_dim=hid_b)
    top = TopModel(embedding_dim=emb_a, hidden_dim=hid_top)

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
# The wire: what the adversary actually observes
# --------------------------------------------------------------------------------------
def load_mechanism(checkpoint_dir: str | Path, eps_tag: str) -> dict | None:
    """Read the DP mechanism a run actually used, from its own meta.json.

    The defense parameters MUST come from the run being attacked, never from config
    defaults — attacking a DP checkpoint with the wrong sigma measures nothing.

    Args:
        checkpoint_dir: Root dir holding per-run subdirs.
        eps_tag: The run tag / subdir name.

    Returns:
        The ``mechanism`` block from meta.json, or None if the run predates mechanism
        recording (in which case a wire-faithful attack is not possible).
    """
    meta_path = Path(checkpoint_dir) / eps_tag / "meta.json"
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text()).get("mechanism")


def wire_embedding(
    embedding: torch.Tensor,
    max_norm: float,
    noise_std: float,
    n_observations: int = 1,
) -> torch.Tensor:
    """Reproduce the embedding as it crosses the wire: row-clipped, then Gaussian-noised.

    This is the ONLY representation an honest-but-curious server or MITM ever sees.
    Attacking ``model_a(x)`` directly attacks a tensor that never leaves the silo, and so
    measures the effect of DP on the trained weights rather than the defense itself.
    Delegates to privacy.noise_injection so there is exactly one implementation of the
    mechanism (the same one fl_client/common.py:generate_embeddings calls).

    ``n_observations`` models the REPEATED-OBSERVATION attacker. A real adversary watches
    every round, and at sample rate q over T rounds a given student is transmitted about
    q*T times. Averaging k independent noisy views of the same embedding shrinks the noise
    by sqrt(k), so a single-observation attack understates leakage — which is exactly what
    the privacy accounting already assumes (composition over steps is why epsilon grows
    with T), but which the attack side was ignoring.

    Args:
        embedding: Clean embedding straight out of the bottom model.
        max_norm: The run's cut-layer clip S (its DP sensitivity).
        noise_std: The run's injected noise std, i.e. sigma * S.
        n_observations: Number of transmissions of this student the attacker averages.
            1 = the single-shot attacker.

    Returns:
        A new tensor: the mean of ``n_observations`` draws of
        ``clip_embedding_rows(embedding, S) + N(0, noise_std^2)``.

    Note:
        This treats the underlying embedding as fixed across observations. In training the
        model drifts between rounds, so the averaged view is an UPPER BOUND on the
        averaging attacker's advantage — a good approximation late in training, when
        weights change slowly, and optimistic for the attacker early on. Reported as a
        bound rather than a realized attack.
    """
    clipped = clip_embedding_rows(embedding, max_norm)
    if noise_std <= 0:
        return clipped
    k = max(1, int(n_observations))
    stacked = torch.stack([
        add_gaussian_noise(clipped, noise_std=noise_std) for _ in range(k)
    ])
    return stacked.mean(dim=0)


def _cut_params(mechanism: dict | None, silo: str) -> tuple[float, float]:
    """Extract (clip S, noise std) for one silo from a run's mechanism block.

    Args:
        mechanism: The ``mechanism`` block from meta.json, or None.
        silo: ``"silo_a"`` or ``"silo_b"``.

    Returns:
        Tuple of (max_norm, noise_std). Returns (inf, 0.0) — i.e. no defense — when the
        run had cut noise disabled or carries no mechanism record.
    """
    if not mechanism or not (mechanism.get("cut_noise") or {}).get("enabled"):
        return float("inf"), 0.0
    prims = (mechanism.get("per_silo") or {}).get(silo)
    if not prims:
        return float("inf"), 0.0
    return float(prims["max_norm"]), float(prims["noise_std_target"])


def _attribute_recovery(
    recon: torch.Tensor, truth: torch.Tensor
) -> dict[str, dict[str, float]]:
    """Per-attribute recovery rates for Silo A, each against its own majority floor.

    The inversion optimizer recovers the whole feature row; this reads out each attribute
    and scores it. Every rate is paired with the rate an attacker achieves by always
    guessing that attribute's most common value, because an accuracy without its floor is
    unreadable — 85% on a binary attribute that is 85% one-sided is zero information.

    Args:
        recon: Reconstructed rows, shape (n, n_features).
        truth: True rows, shape (n, n_features).

    Returns:
        {attribute: {acc, floor, lift, n}}. Includes the one-hot region block (scored by
        argmax) and a scale-free error for the continuous column.
    """
    out: dict[str, dict[str, float]] = {}
    n = truth.shape[0]
    if n == 0:
        return out

    for name, (idx, levels) in SILO_A_CATEGORICALS.items():
        true_col = truth[:, idx].round().clamp(0, levels - 1).long()
        pred_col = recon[:, idx].round().clamp(0, levels - 1).long()
        acc = float((pred_col == true_col).float().mean().item())
        modal = int(true_col.mode().values.item())
        floor = float((true_col == modal).float().mean().item())
        out[name] = {"acc": acc, "floor": floor, "lift": acc - floor, "n": n}

    # Region: drop-first one-hot, so "all zeros" is the dropped reference category.
    # argmax with a zero-vector check recovers that case correctly.
    true_block, pred_block = truth[:, SILO_A_REGION_BLOCK], recon[:, SILO_A_REGION_BLOCK]
    if true_block.shape[1]:
        true_region = torch.where(
            true_block.max(dim=1).values > 0.5, true_block.argmax(dim=1) + 1,
            torch.zeros(n, dtype=torch.long),
        )
        pred_region = torch.where(
            pred_block.max(dim=1).values > 0.5, pred_block.argmax(dim=1) + 1,
            torch.zeros(n, dtype=torch.long),
        )
        acc = float((pred_region == true_region).float().mean().item())
        modal = int(true_region.mode().values.item())
        floor = float((true_region == modal).float().mean().item())
        out["region"] = {"acc": acc, "floor": floor, "lift": acc - floor, "n": n}

    for idx in SILO_A_CONTINUOUS:
        # Normalised RMSE: 1.0 means the attack is no better than predicting the mean.
        err = float(((recon[:, idx] - truth[:, idx]) ** 2).mean().sqrt().item())
        scale = float(truth[:, idx].std().item()) or 1.0
        out["studied_credits"] = {"nrmse": err / scale, "n": n}
    return out


def _reconstruction_quality(recon: torch.Tensor, truth: torch.Tensor) -> dict[str, float]:
    """Scale-free reconstruction error over every feature, for either silo.

    Works where per-attribute decoding does not: Silo B is 26 continuous behavioural
    features with no categorical structure to score.

    Args:
        recon: Reconstructed rows, shape (n, n_features).
        truth: True rows, shape (n, n_features).

    Returns:
        {nrmse_mean, corr_mean, n_features}. nrmse_mean >= 1.0 means the attack does no
        better than predicting each feature's mean; corr_mean near 0 means no signal.
    """
    if truth.numel() == 0:
        return {}
    # Features that are (near-)constant in this sample carry nothing to recover, and
    # dividing by their ~0 spread sends nRMSE to ~1e7. Score only informative features and
    # report how many were used, rather than emitting an uninterpretable number.
    std = truth.std(dim=0)
    informative = std > 1e-6
    if not bool(informative.any()):
        return {"n_features": int(truth.shape[1]), "n_scored": 0}

    nrmse = ((recon - truth) ** 2).mean(dim=0).sqrt()[informative] / std[informative]
    corrs = []
    for j in range(truth.shape[1]):
        t, p = truth[:, j], recon[:, j]
        if t.std() < 1e-6 or p.std() < 1e-6:
            continue
        corrs.append(float(torch.corrcoef(torch.stack([t, p]))[0, 1].item()))
    return {
        "nrmse_mean": float(nrmse.mean().item()),
        "corr_mean": sum(corrs) / len(corrs) if corrs else 0.0,
        "n_features": int(truth.shape[1]),
        "n_scored": int(informative.sum().item()),
    }


def _wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation because attack success rates sit near 0 or 1,
    where the naive interval leaves the [0, 1] range.

    Args:
        successes: Number of successes.
        n: Number of trials.
        z: Normal quantile (1.96 = 95%).

    Returns:
        Tuple of (low, high).
    """
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5) / denom
    return max(0.0, center - half), min(1.0, center + half)


# --------------------------------------------------------------------------------------
# Attack drivers
# --------------------------------------------------------------------------------------
def run_inversion(
    df: pd.DataFrame,
    model: BottomModelA | BottomModelB,
    n_students: int = 200,
    num_iters: int = 200,
    lr: float = 0.1,
    max_norm: float = float("inf"),
    noise_std: float = 0.0,
    silo: str = "a",
    n_observations: int = 1,
) -> dict:
    """Run the iDLG inversion attack against the WIRE embedding.

    For each student the adversary observes what the silo actually transmits —
    ``wire_embedding(model(x), S, sigma*S)`` — and reconstructs the feature row.

    The attack recovers the WHOLE row, so the whole row is scored. For Silo A that means
    per-attribute recovery of gender, disability, age band, education, IMD band and region
    (see ``_attribute_recovery``); previously only imd_band was read out and the other 17
    columns were discarded, reporting 1/18 of what the attack actually achieves. Both
    silos additionally get a scale-free reconstruction error.

    Every rate is paired with its floor — the rate from always guessing the majority value
    — because an accuracy without its floor cannot be interpreted.

    Args:
        df: Attack DataFrame.
        model: Trained bottom model for the target silo, in eval mode.
        n_students: Number of students to attack. 200 keeps the SE near 0.035; at the
            previous default of 20 the SE is 0.112, which cannot resolve any DP effect.
        num_iters: L-BFGS iterations per attempt.
        lr: L-BFGS learning rate.
        max_norm: Cut-layer clip S from the attacked run (inf = no clipping).
        noise_std: Injected noise std from the attacked run (0 = no cut noise).
        silo: ``"a"`` or ``"b"`` — which silo's transmitted embedding to attack. Silo B
            holds 26 behavioural VLE features and was never attacked before.
        n_observations: Transmissions per student the attacker averages (see
            wire_embedding). 1 = single-shot; round(sample_rate * rounds) models an
            adversary that watched the whole run.

    Returns:
        Dict of attack metrics, CIs, anchors, per-attribute recovery (Silo A) and
        reconstruction quality.
    """
    xa, xb, _ = _tensors_from_df(df)
    features = xa if silo == "a" else xb
    n = min(n_students, features.shape[0])
    truths = features[:n, IMD_BAND_COL_IDX].round().int()
    successes = 0
    losses: list[float] = []
    recons: list[torch.Tensor] = []
    for i in range(n):
        row = features[i].unsqueeze(0)
        observed = wire_embedding(
            model(row).detach(), max_norm, noise_std, n_observations=n_observations
        )
        true_imd_band = int(truths[i].item())
        result = run_inversion_attack(
            model, observed, true_imd_band, num_iters=num_iters, lr=lr
        )
        successes += int(result["success"])
        losses.append(float(result["final_loss"]))
        recons.append(result["reconstruction"])

    recon = torch.stack(recons) if recons else torch.empty(0, features.shape[1])
    truth = features[:n]
    # Clamp the reconstruction into the feature ranges before scoring. This makes the
    # attacker STRONGER, not weaker: feature ranges are public schema (gender is binary,
    # imd_band is 0-9), so a real adversary would never report an out-of-range value.
    # Without it, unconstrained L-BFGS diverges on noised targets and the error metric
    # blows up to ~1e16, which is unusable rather than informative.
    if n:
        lo, hi = features.min(dim=0).values, features.max(dim=0).values
        recon = recon.clamp(min=lo, max=hi)
    quality = _reconstruction_quality(recon, truth)

    # Floor: the best an attacker does with no information at all.
    modal = int(truths.mode().values.item()) if n else 0
    asr_majority = float((truths == modal).float().mean().item()) if n else 0.0
    lo, hi = _wilson_ci(successes, n)
    out = {
        "attack": "inversion",
        "silo": silo,
        "n_students": n,
        "asr": successes / n if n else 0.0,
        "asr_ci_low": lo,
        "asr_ci_high": hi,
        "asr_majority": asr_majority,
        "asr_lift": (successes / n - asr_majority) if n else 0.0,
        "mean_final_loss": sum(losses) / len(losses) if losses else 0.0,
        "wire_max_norm": max_norm,
        "wire_noise_std": noise_std,
        "n_observations": n_observations,
        "effective_noise_std": noise_std / (max(1, n_observations) ** 0.5),
        **{f"recon_{k}": v for k, v in quality.items()},
    }
    if silo == "a":
        out["attributes"] = _attribute_recovery(recon, truth)
    return out


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
        Dict: {attack, n_students, accuracy, accuracy_ci_low/high, accuracy_majority, defended}.

    Note:
        This channel is defended by NEITHER mechanism. Cut-layer noise perturbs the
        embedding travelling silo -> server; server DP-SGD noises the top model's
        PARAMETER gradients, not the activation gradient dL/d(emb_b) sent back to the
        silo. That gradient's sign is proportional to (p - y), so it leaks the label
        regardless of epsilon. Reported as an explicitly undefended channel rather than
        silently attacked at reduced strength — it is the empirical counterpart of the
        scope decision "DP protects features, not labels" (docs/paper_topics.md section 7).
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
    correct = int((inferred == y.long()).sum().item())
    n = int(y.shape[0])
    lo, hi = _wilson_ci(correct, n)
    base = float(max(y.mean().item(), 1 - y.mean().item()))
    return {
        "attack": "label_inference",
        "n_students": n,
        "accuracy": correct / n if n else 0.0,
        "accuracy_ci_low": lo,
        "accuracy_ci_high": hi,
        "accuracy_majority": base,
        "defended": False,
    }


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
    n_students: int = 200,
    models: tuple[BottomModelA, BottomModelB, TopModel] | None = None,
    wire: bool = True,
    n_observations: int = 1,
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
        wire: Attack the transmitted (clipped + noised) embedding, using the defense
            parameters recorded by the run itself. Setting this False attacks the clean
            embedding, which measures only how DP changed the weights and will
            systematically UNDERSTATE the defense — kept for the ablation, not for
            headline numbers.

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

    mechanism = load_mechanism(checkpoint_dir, eps_tag) if checkpoint_dir else None
    norm_a, std_a = _cut_params(mechanism, "silo_a") if wire else (float("inf"), 0.0)
    norm_b, std_b = _cut_params(mechanism, "silo_b") if wire else (float("inf"), 0.0)

    results: dict = {
        "eps_tag": eps_tag,
        "wire": wire,
        "mechanism_available": mechanism is not None,
        "epsilon": _reported_epsilon(checkpoint_dir, eps_tag) if checkpoint_dir else None,
    }
    if wire and mechanism is None:
        print(
            f"[warn] {eps_tag}: no mechanism record — attacking the CLEAN embedding. "
            "This understates the defense; re-train so meta.json carries the mechanism block."
        )
    # The gradient-norm monitor is deliberately NOT run here. It is a defense-side anomaly
    # detector — it measures no leakage at all — and listing it beside two real attacks
    # implied it did. It stays available as run_gradient_monitoring() for a defenses
    # section, but it is not an attack result.
    for name, fn in (
        ("inversion_a", lambda: run_inversion(
            df, model_a, n_students=n_students, max_norm=norm_a, noise_std=std_a,
            silo="a", n_observations=n_observations)),
        ("inversion_b", lambda: run_inversion(
            df, model_b, n_students=n_students, max_norm=norm_b, noise_std=std_b,
            silo="b", n_observations=n_observations)),
        ("label_inference", lambda: run_label_inference(df, model_a, model_b, top)),
    ):
        try:
            results[name] = fn()
        except Exception as exc:  # noqa: BLE001 — one broken attack must not sink the rest
            results[name] = {"attack": name, "error": f"{type(exc).__name__}: {exc}"}
    return results


def _reported_epsilon(checkpoint_dir: str | Path, eps_tag: str) -> float | None:
    """The composed epsilon a run reported, for plotting ASR against.

    Args:
        checkpoint_dir: Root checkpoint dir.
        eps_tag: Run tag / subdir name.

    Returns:
        The run's combined epsilon, or None if unavailable.
    """
    meta_path = Path(checkpoint_dir) / eps_tag / "meta.json"
    if not meta_path.exists():
        return None
    spent = json.loads(meta_path.read_text()).get("epsilon_spent") or {}
    return spent.get("privacy/combined_epsilon")


def run_sweep(
    df: pd.DataFrame,
    checkpoint_dir: str | Path,
    eps_tags: list[str],
    n_students: int = 200,
    wire: bool = True,
    n_observations: int = 1,
    out_path: str | Path | None = None,
) -> list[dict]:
    """Attack several checkpoints and return one flat summary row per run.

    This is what produces the attack-success-vs-epsilon curve. Pair it with the no-DP
    checkpoint in the same call so the ceiling is measured, not assumed.

    Args:
        df: Attack DataFrame.
        checkpoint_dir: Root checkpoint dir.
        eps_tags: Run tags to attack, in plot order.
        n_students: Students per inversion run.
        wire: Attack the transmitted representation (see run_all_attacks).
        n_observations: Transmissions per student the attacker averages.
        out_path: If given, the CSV is rewritten after EVERY checkpoint rather than only at
            the end. A full sweep is ~2 hours (200 students x 2 silos x 4 checkpoints, each
            student a 200-step L-BFGS optimisation), so finishing-only writes meant a crash
            on the last checkpoint discarded every earlier result.

    Returns:
        One dict per tag, flattened for tabular reporting.
    """
    rows: list[dict] = []

    def _checkpoint() -> None:
        """Persist what has been collected so far."""
        if out_path is None or not rows:
            return
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(path, index=False)

    for tag in eps_tags:
        print(f"\n[sweep] attacking {tag} ...")
        try:
            res = run_all_attacks(
                df, checkpoint_dir=checkpoint_dir, eps_tag=tag,
                n_students=n_students, wire=wire, n_observations=n_observations,
            )
        except (FileNotFoundError, ValueError, KeyError) as exc:
            print(f"  [warn] {tag}: {type(exc).__name__}: {exc}")
            rows.append({"eps_tag": tag, "error": f"{type(exc).__name__}: {exc}"})
            _checkpoint()
            continue
        _print_report(res)
        inv_a = res.get("inversion_a", {})
        inv_b = res.get("inversion_b", {})
        lab = res.get("label_inference", {})
        row = {
            "eps_tag": tag,
            "epsilon": res.get("epsilon"),
            "wire": res.get("wire"),
            "mechanism_available": res.get("mechanism_available"),
            "inv_n": inv_a.get("n_students"),
            "wire_noise_std": inv_a.get("wire_noise_std"),
            "n_observations": inv_a.get("n_observations"),
            "effective_noise_std": inv_a.get("effective_noise_std"),
            "inv_asr": inv_a.get("asr"),
            "inv_asr_ci_low": inv_a.get("asr_ci_low"),
            "inv_asr_ci_high": inv_a.get("asr_ci_high"),
            "inv_asr_majority": inv_a.get("asr_majority"),
            "inv_asr_lift": inv_a.get("asr_lift"),
            "silo_a_nrmse": inv_a.get("recon_nrmse_mean"),
            "silo_a_corr": inv_a.get("recon_corr_mean"),
            "silo_b_nrmse": inv_b.get("recon_nrmse_mean"),
            "silo_b_corr": inv_b.get("recon_corr_mean"),
            "label_acc": lab.get("accuracy"),
            "label_acc_majority": lab.get("accuracy_majority"),
        }
        # Flatten the per-attribute recovery so the sweep CSV carries the sentence that
        # matters: which protected attributes came back, and by how much over their floor.
        for attr, vals in (inv_a.get("attributes") or {}).items():
            if "acc" in vals:
                row[f"attr_{attr}_acc"] = vals["acc"]
                row[f"attr_{attr}_floor"] = vals["floor"]
                row[f"attr_{attr}_lift"] = vals["lift"]
        rows.append(row)
        _checkpoint()
        if out_path is not None:
            print(f"  [sweep] {len(rows)}/{len(eps_tags)} checkpoints -> {out_path}")
    return rows


def _print_sweep(rows: list[dict]) -> None:
    """Print the sweep as a table with the interpretation anchors alongside.

    Args:
        rows: Output of run_sweep.
    """
    print("\n==== Attack vs epsilon ====")
    hdr = (
        f"{'run':<28} {'eps':>7} {'inv ASR':>9} {'95% CI':>17} "
        f"{'floor':>7} {'lift':>7} {'RMSE':>7} {'label':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if r.get("error"):
            print(f"{r['eps_tag']:<28} {'ERROR: ' + r['error'][:60]}")
            continue
        eps = f"{r['epsilon']:.3f}" if r.get("epsilon") else "-"
        ci = f"[{r['inv_asr_ci_low']:.3f},{r['inv_asr_ci_high']:.3f}]"
        print(
            f"{r['eps_tag']:<28} {eps:>7} {r['inv_asr']:>9.3f} {ci:>17} "
            f"{r['inv_asr_majority']:>7.3f} {r['inv_asr_lift']:>+7.3f} "
            f"{r['inv_recon_rmse']:>7.3f} {r['label_acc']:>7.3f}"
        )
    print("-" * len(hdr))
    print("  floor = always-guess-modal-imd_band; lift = ASR - floor (0 means no leakage")
    print("  beyond what an uninformed attacker gets). label = label inference accuracy,")
    print("  an UNDEFENDED channel by design — it should not fall with epsilon.")


# --------------------------------------------------------------------------------------
# Reporting + CLI
# --------------------------------------------------------------------------------------
def _print_report(results: dict) -> None:
    """Pretty-print the results dict from run_all_attacks.

    Args:
        results: The dict returned by run_all_attacks.
    """
    print(f"\n==== Attack report — {results.get('eps_tag', '?')} ====")

    wire = "WIRE (clipped+noised)" if results.get("wire") else "CLEAN embedding (ablation)"
    print(f"  target: {wire}")

    for key, silo in (("inversion_a", "Silo A"), ("inversion_b", "Silo B")):
        inv = results.get(key, {})
        if not inv:
            continue
        if "error" in inv:
            print(f"  [inversion {silo}]   FAILED: {inv['error']}")
            continue
        obs = inv.get("n_observations", 1)
        obs_note = (f" x{obs} observations -> effective "
                    f"{inv.get('effective_noise_std', 0):.4f}") if obs > 1 else ""
        print(
            f"  [inversion {silo}]   n={inv['n_students']} "
            f"noise_std={inv['wire_noise_std']:.4f}{obs_note} | reconstruction over "
            f"{inv.get('recon_n_features', 0)} features: "
            f"nRMSE={inv.get('recon_nrmse_mean', float('nan')):.3f} "
            f"corr={inv.get('recon_corr_mean', float('nan')):+.3f}"
        )
        for attr, vals in (inv.get("attributes") or {}).items():
            if "acc" in vals:
                verdict = "RECOVERED" if vals["lift"] > 0.05 else "no better than guessing"
                print(f"      {attr:<20} acc={vals['acc']:.3f} floor={vals['floor']:.3f} "
                      f"lift={vals['lift']:+.3f}  {verdict}")
            else:
                print(f"      {attr:<20} nRMSE={vals['nrmse']:.3f} "
                      "(>=1.0 = no better than the mean)")

    lab = results.get("label_inference", {})
    if "error" in lab:
        print(f"  [label_inference]  FAILED: {lab['error']}")
    elif lab:
        print(
            f"  [label_inference]  accuracy={lab['accuracy']:.3f} "
            f"[{lab['accuracy_ci_low']:.3f}, {lab['accuracy_ci_high']:.3f}] over "
            f"{lab['n_students']} students | floor={lab['accuracy_majority']:.3f}"
        )
        print("      NOT a defense evaluation — this channel is undefended by design. "
              "It quantifies a known scope limit.")


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
    parser.add_argument(
        "--n-students", type=int, default=200,
        help="Students to attack (inversion). 200 -> SE ~0.035; 20 -> SE 0.112 (useless).",
    )
    parser.add_argument(
        "--sweep", nargs="+", default=None,
        help="Attack several run tags and print the ASR-vs-epsilon table. Include the "
             "no-DP run so the ceiling is measured.",
    )
    parser.add_argument(
        "--observations", type=int, default=1,
        help="Transmissions per student the attacker averages. 1 = single-shot (weak). "
             "round(batch-sample-rate * num-server-rounds) — about 10 at q=0.05, T=200 — "
             "models an adversary that watched the whole run and averaged the noise down "
             "by sqrt(k).",
    )
    parser.add_argument(
        "--no-wire", action="store_true",
        help="Ablation: attack the CLEAN embedding instead of the transmitted one. "
             "Systematically understates the defense — never use for reported numbers.",
    )
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

    if args.sweep:
        out = Path(args.results_csv or "experiments/attack_sweep.csv")
        rows = run_sweep(
            df, args.checkpoint_dir, args.sweep,
            n_students=args.n_students, wire=not args.no_wire,
            n_observations=args.observations, out_path=out,
        )
        _print_sweep(rows)
        print(f"\nWrote sweep -> {out}")
        return

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
        wire=not args.no_wire,
        n_observations=args.observations,
    )
    _print_report(results)
    if args.results_csv:
        _write_csv(results, args.results_csv)


if __name__ == "__main__":
    main()
