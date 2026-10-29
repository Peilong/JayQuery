#!/usr/bin/env python3
"""Replay both adversaries against every checkpoint the grid produced.

The placement grid measures what each DP placement costs in utility; this
measures what each one buys, per adversary.

  1. Embedding inversion (:mod:`attack.embedding_inversion`): an
     honest-but-curious coordinator reads the embeddings Silo A sends, holds
     auxiliary records whose protected attribute it knows, and learns a
     mapping from embedding back to that attribute. Reported as lift over
     always guessing the majority class, since accuracy alone on a skewed
     attribute is not interpretable. ``--attribute`` selects the column; see
     :data:`ATTRIBUTES`.

  2. Loss-threshold membership inference
     (:mod:`attack.membership_inference`): whoever receives the released model
     asks whether a given student was in its training set. Reported as
     ROC-AUC, chance 0.5.

Three calibration details carry the result:

  * The attack uses each run's realized sigma from ``privacy_primitives``
    rather than the configured target, so a run that stopped early is sized to
    what it actually spent.
  * An undefended arm is attacked against a raw embedding, because the client
    clips and noises only when the cut-layer channel is on
    (``fl_client/common.py``: ``sent_emb = emb`` otherwise).
  * The auxiliary set is made disjoint from the target set in feature space.
    Silo A's columns are mostly categorical, so distinct students share
    byte-identical rows and an index split alone would score memorization as
    leakage.

Usage:
    python scripts/run_attack_grid.py
    python scripts/run_attack_grid.py --attribute gender --n-observations 64
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# Make the repo root importable when run as `python scripts/run_attack_grid.py`:
# sys.path[0] is then scripts/, not the root. Same bootstrap as run_attack.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attack.embedding_inversion import (  # noqa: E402
    InversionResult,
    run_learned_inversion,
    train_inverse_model,
)
from attack.membership_inference import (  # noqa: E402
    MembershipResult,
    run_loss_threshold_mia,
)
from fl_server.top_model import TopModel  # noqa: E402
from silo_a.bottom_model_a import BottomModelA  # noqa: E402
from silo_b.bottom_model_b import BottomModelB  # noqa: E402

#: Column 0 of Silo A is the ordinal-encoded IMD band; 10 deciles.
IMD_COL = 0
IMD_CLASSES = 10


@dataclass(frozen=True)
class Attribute:
    """A protected attribute the coordinator might try to recover.

    Attributes:
        extract: Feature matrix -> integer class per row.
        n_classes: Number of distinct values.
        label: Human-readable name for tables.
    """

    extract: Callable[[np.ndarray], np.ndarray]
    n_classes: int
    label: str


def _col(idx: int) -> Callable[[np.ndarray], np.ndarray]:
    """Read an ordinally-encoded column straight off the matrix."""
    return lambda X: X[:, idx].astype(np.int64)


def _region(X: np.ndarray) -> np.ndarray:
    """Recover region from its 12 drop-first one-hot columns.

    An all-zero row is not missing data: it is the reference level the
    encoder dropped, so it gets its own class. Folding it into class 0 would
    merge two different regions and understate the attack.
    """
    block = X[:, 6:18]
    hot = block.sum(axis=1) > 0
    return np.where(hot, block.argmax(axis=1), 12).astype(np.int64)


#: Silo A's layout is fixed by scripts/prepare_silos.py: 0 imd_band_enc,
#: 1 highest_education_enc, 2 age_band_enc, 3 gender_enc, 4 disability_enc,
#: 5 a scaled continuous feature, 6-17 region one-hots (drop-first).
ATTRIBUTES: dict[str, Attribute] = {
    "imd_band": Attribute(_col(0), 10, "IMD band"),
    "education": Attribute(_col(1), 5, "highest education"),
    "age_band": Attribute(_col(2), 3, "age band"),
    "gender": Attribute(_col(3), 2, "gender"),
    "disability": Attribute(_col(4), 2, "disability"),
    "region": Attribute(_region, 13, "region"),
}

#: Per-row L2 clip the silos used. Not recorded in meta.json, so it is taken
#: from the app config default, which no sweep config overrode for this grid
#: (verified: no `max-norm` key in any experiments/configs/*.yaml).
CUT_MAX_NORM = 1.0


def classify(meta: dict) -> tuple[str, float | None]:
    """Which placement a run used, and at what epsilon, from its metadata.

    Args:
        meta: A run's parsed meta.json.

    Returns:
        ``(arm, epsilon)`` where arm is one of none/server/label/cut-layer/all.
        Epsilon is None only for the no-DP arm.
    """
    prims = meta.get("privacy_primitives") or {}
    label = meta.get("label_privacy") or None
    cut = "silo_a" in prims
    srv = "server" in prims
    lab = bool(label)

    if cut and srv and lab:
        arm = "all"
    elif cut and not srv and not lab:
        arm = "cut-layer"
    elif srv and not cut and not lab:
        arm = "server"
    elif lab and not cut and not srv:
        arm = "label"
    elif not (cut or srv or lab):
        return "none", None
    else:
        arm = "mixed"

    # The Gaussian channels carry their own target; the label channel's budget
    # lives only in its own block.
    eps = (prims.get("silo_a") or prims.get("server") or {}).get("target_epsilon")
    if eps is None:
        eps = (label or {}).get("epsilon")
    return arm, eps


def cut_noise_for(meta: dict) -> tuple[float | None, int]:
    """The cut-layer channel's realized noise, or no-noise for an open channel.

    Returns:
        ``(sigma, steps)``. ``sigma`` is None when the cut-layer channel was
        off, which means the attacker saw a raw embedding.
    """
    prims = (meta.get("privacy_primitives") or {}).get("silo_a")
    if not prims:
        return None, 0
    return float(prims["sigma"]), int(prims["steps"])


def disjoint_auxiliary(train: np.ndarray, test: np.ndarray) -> np.ndarray:
    """Mask of train rows that do NOT appear byte-identically in ``test``.

    Args:
        train: Candidate auxiliary feature rows.
        test: Target feature rows.

    Returns:
        Boolean mask over ``train``.

    Raises:
        ValueError: If no auxiliary row survives, since attacking with an
            empty auxiliary set would otherwise fail deep inside training.
    """
    target_rows = {r.tobytes() for r in test}
    keep = np.array([r.tobytes() not in target_rows for r in train], dtype=bool)
    if not keep.any():
        raise ValueError(
            "no auxiliary rows left after removing every row that also appears "
            "in the target set; the attack cannot be scored on disjoint data"
        )
    return keep


def load_models(ckpt: str | Path) -> tuple[BottomModelA, BottomModelB, TopModel]:
    """Rebuild a run's three models, sizing them from the saved tensors.

    Args:
        ckpt: Checkpoint directory holding silo_a.pt, silo_b.pt and top.pt.

    Returns:
        ``(bottom_a, bottom_b, top)``, all in eval mode.
    """
    ckpt = Path(ckpt)
    sa = torch.load(ckpt / "silo_a.pt", map_location="cpu", weights_only=True)
    sb = torch.load(ckpt / "silo_b.pt", map_location="cpu", weights_only=True)
    tp = torch.load(ckpt / "top.pt", map_location="cpu", weights_only=True)

    dim = sa["encoder.7.weight"].shape[0]
    a = BottomModelA(
        input_dim=sa["encoder.0.weight"].shape[1],
        embedding_dim=dim,
        hidden_dim=sa["encoder.0.weight"].shape[0],
    )
    b = BottomModelB(
        input_dim=sb["layer1.0.weight"].shape[1],
        embedding_dim=dim,
        hidden_dim=sb["layer1.0.weight"].shape[0],
    )
    t = TopModel(embedding_dim=dim, hidden_dim=tp["net.0.weight"].shape[0])
    a.load_state_dict(sa)
    b.load_state_dict(sb)
    t.load_state_dict(tp)
    for m in (a, b, t):
        m.eval()
    return a, b, t


def _undefended_inversion(
    bottom_a: BottomModelA,
    X_aux: torch.Tensor,
    y_aux: torch.Tensor,
    X_tgt: torch.Tensor,
    y_tgt: torch.Tensor,
    n_classes: int,
    epochs: int,
    seed: int,
) -> InversionResult:
    """Inversion against a RAW embedding, for runs with the channel off.

    Mirrors run_learned_inversion except that no clip and no noise are
    applied, because the client applies none when the channel is off. Reuses
    the same inverse model and scoring so defended and undefended numbers
    remain comparable.
    """
    with torch.no_grad():
        emb_aux = bottom_a(X_aux)
        emb_tgt = bottom_a(X_tgt)
    inverse = train_inverse_model(
        emb_aux, y_aux, n_classes=n_classes, epochs=epochs, seed=seed
    )
    with torch.no_grad():
        predicted = inverse(emb_tgt).argmax(dim=1)
    accuracy = float((predicted == y_tgt).float().mean())
    counts = torch.bincount(y_tgt, minlength=n_classes)
    baseline = float(counts.max()) / float(len(y_tgt))
    return InversionResult(
        accuracy=accuracy,
        baseline=baseline,
        lift=accuracy - baseline,
        n_target=int(len(y_tgt)),
        sigma=0.0,
        n_observations=1,
    )


def attack_one(
    ckpt: Path,
    data: dict,
    epochs: int = 60,
    seed: int = 0,
    n_observations: int = 1,
) -> dict:
    """Run both adversaries against one checkpoint.

    Args:
        ckpt: The run's checkpoint directory.
        data: Prepared tensors from :func:`load_data`.
        epochs: Inverse-model training epochs.
        seed: Attack seed.
        n_observations: Noisy views the coordinator averages per record.

    Returns:
        One flat record: arm, epsilon, inversion and membership numbers.
    """
    meta = json.loads((ckpt / "meta.json").read_text())
    arm, eps = classify(meta)
    sigma, steps = cut_noise_for(meta)
    a, b, t = load_models(ckpt)

    if sigma is None:
        inv = _undefended_inversion(
            a, data["Xa_aux"], data["imd_aux"], data["Xa_tgt"], data["imd_tgt"],
            n_classes=data["n_classes"], epochs=epochs, seed=seed,
        )
    else:
        inv = run_learned_inversion(
            a, data["Xa_aux"], data["imd_aux"], data["Xa_tgt"], data["imd_tgt"],
            n_classes=data["n_classes"], max_norm=CUT_MAX_NORM, sigma=sigma,
            n_observations=n_observations, epochs=epochs, seed=seed,
        )

    mia: MembershipResult = run_loss_threshold_mia(
        t, a, b,
        members=(data["Xa_mem"], data["Xb_mem"], data["y_mem"]),
        nonmembers=(data["Xa_tgt"], data["Xb_tgt"], data["y_tgt_lab"]),
    )

    return {
        "run": ckpt.name,
        "arm": arm,
        "epsilon": eps,
        "embedding_dim": meta["embedding_dim"],
        "stopped_early": meta["stopped_early"],
        "test_auc": meta["metrics"]["auc_roc"],
        "cut_sigma": sigma,
        "cut_steps": steps,
        "inv_accuracy": inv.accuracy,
        "inv_baseline": inv.baseline,
        "inv_lift": inv.lift,
        # Recorded so runs at different denoising levels are distinguishable
        # on disk.
        "n_observations": inv.n_observations,
        "attribute": data["attribute"],
        "mia_auc": mia.auc,
        "mia_advantage": mia.advantage,
        "mia_accuracy": mia.accuracy,
        "n_target": inv.n_target,
        "n_members": mia.n_members,
    }


def load_data(processed_dir: str | Path, seed: int = 0,
              attribute: str = "imd_band") -> dict:
    """Assemble the attacker's auxiliary, target and membership splits.

    Target and non-member set is the held-out test split. Members are an
    equal-sized sample of the training split, so the membership AUC does not
    drift with the member/non-member ratio.
    """
    processed_dir = Path(processed_dir)
    da = pickle.loads((processed_dir / "silo_a.pkl").read_bytes())
    db = pickle.loads((processed_dir / "silo_b.pkl").read_bytes())

    spec = ATTRIBUTES[attribute]
    keep = disjoint_auxiliary(da["X_train"], da["X_test"])
    rng = np.random.default_rng(seed)
    n_test = len(da["X_test"])
    members = rng.choice(len(da["X_train"]), size=min(n_test, len(da["X_train"])),
                         replace=False)

    f32 = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32)  # noqa: E731
    return {
        "Xa_aux": f32(da["X_train"][keep]),
        "imd_aux": torch.tensor(spec.extract(da["X_train"])[keep], dtype=torch.long),
        "Xa_tgt": f32(da["X_test"]),
        "Xb_tgt": f32(db["X_test"]),
        "imd_tgt": torch.tensor(spec.extract(da["X_test"]), dtype=torch.long),
        "n_classes": spec.n_classes,
        "attribute": attribute,
        "y_tgt_lab": f32(da["y_test"]),
        "Xa_mem": f32(da["X_train"][members]),
        "Xb_mem": f32(db["X_train"][members]),
        "y_mem": f32(da["y_train"][members]),
        "n_aux": int(keep.sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoints", default="experiments/checkpoints")
    ap.add_argument("--processed-dir", default="data/processed/dedup_first")
    ap.add_argument("--out", default="experiments/attack_results.json")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-observations", type=int, default=1,
                    help="Noisy views the coordinator averages per record.")
    ap.add_argument("--attribute", default="imd_band", choices=sorted(ATTRIBUTES),
                    help="Protected attribute the coordinator tries to recover.")
    ap.add_argument("--limit", type=int, default=None, help="Attack only the first N.")
    args = ap.parse_args()

    data = load_data(args.processed_dir, seed=args.seed, attribute=args.attribute)
    ckpts = sorted(
        p.parent for p in Path(args.checkpoints).glob("*/meta.json")
        if (p.parent / "silo_a.pt").exists()
    )
    if args.limit:
        ckpts = ckpts[: args.limit]
    print(f"[attack] {len(ckpts)} checkpoints | attribute={args.attribute} "
          f"({data['n_classes']} classes) | aux={data['n_aux']} "
          f"target={len(data['imd_tgt'])} members={len(data['y_mem'])}")

    rows, failed = [], []
    for i, ckpt in enumerate(ckpts, 1):
        try:
            row = attack_one(ckpt, data, epochs=args.epochs, seed=args.seed,
                             n_observations=args.n_observations)
        except Exception as exc:  # noqa: BLE001 — one bad run must not stop the grid
            failed.append((ckpt.name, repr(exc)))
            print(f"[attack] {i}/{len(ckpts)} {ckpt.name} FAILED {exc}")
            continue
        rows.append(row)
        print(f"[attack] {i}/{len(ckpts)} {row['arm']:10s} eps={str(row['epsilon']):>4} "
              f"inv_lift={row['inv_lift']:+.4f} mia_auc={row['mia_auc']:.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows, "failed": failed}, indent=1))
    print(f"[attack] wrote {out} ({len(rows)} rows, {len(failed)} failed)")


if __name__ == "__main__":
    main()
