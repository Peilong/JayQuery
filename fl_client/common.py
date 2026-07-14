"""Generic VFL client handlers, parameterized by SiloSpec.

Owner: Alex Pellet

Generalized from the original client_a_new.py. Behavior is unchanged from
that file for Silo A; Silo B (and any future silo) gets the same logic via
its own SiloSpec.
"""

import io
import json
import time
from pathlib import Path

import torch
from flwr.app import Array, ArrayRecord, ConfigRecord, Context, Message, RecordDict

from fl_client.silo_spec import SiloSpec
from privacy.noise_injection import add_gaussian_noise, clip_embedding_rows
from privacy.privacy_accountant import PrivacyBudgetTracker, solve_noise_multiplier
from utils.wandb_utils import log_client_rounds, new_run_id


def _build_model(spec: SiloSpec, context: Context) -> torch.nn.Module:
    """Construct the silo's model, restoring weights from context.state if present."""
    model = spec.model_cls(**spec.model_kwargs_fn(context))
    if (rec := context.state.get("model")) is not None:
        model.load_state_dict(rec.to_torch_state_dict())
    return model


def _full_trainset(spec: SiloSpec, context: Context) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize the silo's entire training set as one (X, y) pair.

    Requires load_fn's train_loader to be deterministic (shuffle=False or a
    fixed sampler) — row order must match across rounds for apply_gradients
    to reproduce the embedding sent in generate_embeddings.
    """
    batch_size = context.run_config.get("batch-size", 64)
    train_loader, _, _ = spec.load_fn(batch_size=batch_size)
    xs, ys = [], []
    for x, y in train_loader:
        xs.append(x)
        ys.append(y)
    return torch.cat(xs).float(), torch.cat(ys).float()

def _eval_set(
    spec: SiloSpec, context: Context, split: str = "val"
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = context.run_config.get("batch-size", 64)
    _, val_loader, test_loader = spec.load_fn(batch_size=batch_size)
    loader = test_loader if split == "test" else val_loader
    xs, ys = [], []
    for x, y in loader:
        xs.append(x)
        ys.append(y)
    return torch.cat(xs).float(), torch.cat(ys).float()

# ── Configs ───────────────────────────────────────────────────

def _cut_noise_config(spec: SiloSpec, context: Context) -> dict:
    """Noise injection params, run_config defaults overridden by per-silo SiloSpec values."""
    rc = context.run_config
    cfg = {
        "enabled": bool(rc.get("cut-noise-enabled", False)),
        "target_epsilon": float(rc.get("cut-noise-target-epsilon", 1.0)),
        "delta": float(rc.get("cut-noise-delta", 1e-5)),
        "max_norm": float(rc.get("cut-noise-max-norm", 1.0)),
    }
    cfg.update(spec.cut_noise_config_fn(context))  # silo-specific overrides
    return cfg


def _batch_indices(n: int, cfg: ConfigRecord | None) -> tuple[torch.Tensor, float]:
    """Poisson-sample this round's rows from the server-sent seed + sample rate.

    Both silos receive the same seed and rate, so they select identical rows
    of the PSI-aligned training set. Returns (indices, sample_rate).
    """
    q = float(cfg["batch-sample-rate"]) if cfg is not None and "batch-sample-rate" in cfg else 1.0
    if q >= 1.0:
        return torch.arange(n), 1.0
    if cfg is None or "batch-seed" not in cfg:
        raise ValueError("batch-sample-rate < 1.0 requires a batch-seed in the round config")
    g = torch.Generator().manual_seed(int(cfg["batch-seed"]))
    idx = (torch.rand(n, generator=g) < q).nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        raise RuntimeError(f"Empty Poisson batch (q={q}, n={n}) — raise batch-sample-rate")
    return idx, q


def _load_tracker(context: Context) -> PrivacyBudgetTracker | None:
    rec = context.state.get("privacy")
    return PrivacyBudgetTracker.from_dict(dict(rec)) if rec is not None else None


def _save_tracker(context: Context, tracker: PrivacyBudgetTracker) -> None:
    context.state["privacy"] = ConfigRecord(tracker.to_dict())


def _save_checkpoint(spec: SiloSpec, model: torch.nn.Module, eps_tag: str) -> None:
    ckpt_dir = Path("experiments/checkpoints") / eps_tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / f"{spec.wandb_tag}.pt")

def _wandb_config(spec: SiloSpec, context: Context) -> dict:
    rc = context.run_config
    cfg = {
        "silo": spec.wandb_tag,
        "embedding-dim": rc.get("embedding-dim", 32),
        "learning-rate": rc.get("learning-rate", 1e-3),
        "batch-size": rc.get("batch-size", 64),
        "seed": rc.get("seed", 42),
    }
    cfg.update(spec.extra_wandb_config_fn(context))
    return cfg

# ── wandb tracking helpers ───────────────────────────────────────────────────


def _tracking_enabled(context: Context) -> bool:
    return bool(context.run_config.get("track-wandb", True))


def _run_id(context: Context) -> str:
    """Stable wandb run id for this node, persisted in context.state."""
    rec = context.state.get("wandb")
    if rec is not None and "run_id" in rec:
        return str(rec["run_id"])
    rid = new_run_id()
    context.state["wandb"] = ConfigRecord({"run_id": rid})
    return rid


def _round_number(context: Context, msg: Message, *, advance: bool) -> int:
    """Current FL round, used as the wandb logging step."""
    cfg = msg.content.get("config") if msg.content is not None else None
    if cfg is not None and "server-round" in cfg:
        r = int(cfg["server-round"])
    else:
        prev = int(context.state["round"]["v"]) if "round" in context.state else -1
        r = prev + 1 if advance else max(prev, 0)
    context.state["round"] = ConfigRecord({"v": r})
    return r


def _buffered_entries(context: Context) -> list[list]:
    rec = context.state.get("wandb_buffer")
    return json.loads(rec["v"]) if rec is not None else []


def _flush_wandb_buffer(spec: SiloSpec, context: Context) -> None:
    """Upload all buffered rounds in one init/log/finish cycle, then clear."""
    if not _tracking_enabled(context):
        return
    entries = _buffered_entries(context)
    if not entries:
        return
    rc = context.run_config
    rid = _run_id(context)
    log_client_rounds(
        run_id=rid,
        entries=entries,
        project=rc.get("wandb-project", "jayquery-vfl"),
        entity=rc.get("wandb-entity", "jayquery"),
        name=rc.get("wandb-run-name", "") or f"{spec.wandb_tag}-{rid[:8]}",
        group=rc.get("wandb-group", "") or None,
        tags=["vfl", spec.wandb_tag, "client"],
        mode=rc.get("wandb-mode", "online"),
        config=_wandb_config(spec, context),
    )
    context.state["wandb_buffer"] = ConfigRecord({"v": "[]"})


def _log(
    spec: SiloSpec, context: Context, metrics: dict, step: int, *, flush: bool = False
) -> None:
    """Buffer metrics in context.state; upload every wandb-log-interval rounds.

    Two entries accrue per round (generate_embeddings + apply_gradients);
    interval <= 1 uploads on every call. Pass flush=True on the last round.
    """
    if not _tracking_enabled(context):
        return
    entries = _buffered_entries(context)
    entries.append([step, metrics])
    context.state["wandb_buffer"] = ConfigRecord({"v": json.dumps(entries)})

    interval = int(context.run_config.get("wandb-log-interval", 10))
    if flush or interval <= 1 or len(entries) >= 2 * interval:
        _flush_wandb_buffer(spec, context)


# ── handlers ─────────────────────────────────────────────────────────────────


def generate_embeddings(spec: SiloSpec, msg: Message, context: Context) -> Message:
    """Round step 1: forward pass on this round's batch -> return (noised) embedding."""
    cfg = msg.content.get("config") if msg.content is not None else None
    model = _build_model(spec, context)
    model.train()

    X, y = _full_trainset(spec, context)
    idx, q = _batch_indices(len(X), cfg)

    noise = _cut_noise_config(spec, context)
    tracker = None
    if noise["enabled"]:
        tracker = _load_tracker(context)
        if tracker is None:
            sigma = solve_noise_multiplier(
                target_epsilon=noise["target_epsilon"],
                sample_rate=q,
                steps=int(context.run_config.get("num-server-rounds", 1)),
                delta=noise["delta"],
            )
            tracker = PrivacyBudgetTracker(
                target_epsilon=noise["target_epsilon"],
                sigma=sigma,
                sample_rate=q,
                delta=noise["delta"],
            )
        if tracker.would_exceed():
            _save_tracker(context, tracker)
            content = RecordDict({
                "config": ConfigRecord({"pos": spec.partition_id, "budget-exhausted": True}),
                "privacy": ConfigRecord(tracker.to_dict()),
            })
            return Message(content=content, reply_to=msg)

    # Snapshot RNG immediately before the forward so apply_gradients can
    # reproduce the same dropout mask.
    rng = torch.get_rng_state()
    t0 = time.perf_counter()
    embedding = model(X[idx])
    fwd_time = time.perf_counter() - t0

    buf = io.BytesIO()
    torch.save(rng, buf)
    context.state["rng"] = ConfigRecord({"serialized": buf.getvalue()})
    context.state["model"] = ArrayRecord(model.state_dict())
    context.state["batch_idx"] = ArrayRecord({"v": Array(idx.numpy())})

    emb = embedding.detach()
    # Clean embedding for the apply_gradients replay check; clip/noise the sent copy only.
    context.state["sent_embedding"] = ArrayRecord({"v": Array(emb.numpy())})

    sent_emb = emb
    if tracker is not None:
        sent_emb = add_gaussian_noise(
            clip_embedding_rows(emb, noise["max_norm"]),
            noise_std=tracker.sigma * noise["max_norm"],
        )
        tracker.step()
        _save_tracker(context, tracker)

    rnd = _round_number(context, msg, advance=True)
    with torch.no_grad():
        metrics = {
            f"{spec.wandb_tag}/round": rnd,
            f"{spec.wandb_tag}/forward_time_s": fwd_time,
            f"{spec.wandb_tag}/embedding_l2": float(emb.norm(p=2).item()),
            f"{spec.wandb_tag}/embedding_mean": float(emb.mean().item()),
            f"{spec.wandb_tag}/embedding_std": float(emb.std().item()),
            f"{spec.wandb_tag}/n_examples": int(emb.shape[0]),
        }
        if tracker is not None:
            metrics[f"{spec.wandb_tag}/epsilon"] = tracker.get_epsilon()
        _log(spec, context, metrics, step=rnd)

    arrays = {"embedding": Array(sent_emb.numpy())}
    if spec.sends_labels:
        arrays["labels"] = Array(y[idx].numpy())

    content = RecordDict({
        "arrays": ArrayRecord(arrays),
        "config": ConfigRecord({"pos": spec.partition_id, "n-total": int(len(X))}),
    })
    if tracker is not None:
        content["privacy"] = ConfigRecord(tracker.to_dict())
    return Message(content=content, reply_to=msg)

def generate_eval_embeddings(spec: SiloSpec, msg: Message, context: Context) -> Message:
    model = _build_model(spec, context)
    model.eval()

    cfg = msg.content.get("config")
    split = str(cfg["split"]) if cfg is not None and "split" in cfg else "val"
    X, y = _eval_set(spec, context, split)

    with torch.no_grad():
        embedding = model(X)

    emb = embedding.detach()
    noise = _cut_noise_config(spec, context)
    if noise["enabled"]:
        # Match the training distribution the top model sees (clip, no noise).
        emb = clip_embedding_rows(emb, noise["max_norm"])

    arrays = {"embedding": Array(emb.numpy())}
    if spec.sends_labels:
        arrays["labels"] = Array(y.numpy())

    content = RecordDict(
        {"arrays": ArrayRecord(arrays), "config": ConfigRecord({"pos": spec.partition_id})}
    )
    return Message(content=content, reply_to=msg)

def apply_gradients(spec: SiloSpec, msg: Message, context: Context) -> Message:
    """Round step 3: recompute forward on the same batch, inject server gradient, step."""
    cfg = msg.content.get("config") if msg.content is not None else None
    if cfg is not None and bool(cfg.get("stop-run", False)):
        # Server-side budget gate: checkpoint and stop without training.
        _save_checkpoint(spec, _build_model(spec, context), str(cfg["eps-tag"]))
        _flush_wandb_buffer(spec, context)
        return Message(content=RecordDict(), reply_to=msg)

    model = _build_model(spec, context)
    model.train()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=context.run_config.get("learning-rate", 1e-3)
    )
    if (rec := context.state.get("optimizer_state")) is not None:
        optimizer.load_state_dict(
            torch.load(io.BytesIO(rec["serialized"]), weights_only=False)
        )

    # Must remain before RNG restore
    X, _ = _full_trainset(spec, context)
    if (rec := context.state.get("batch_idx")) is not None:
        X = X[torch.from_numpy(rec["v"].numpy()).long()]

    if (rec := context.state.get("rng")) is not None:
        torch.set_rng_state(
            torch.load(io.BytesIO(rec["serialized"]), weights_only=False)
        )

    grad = torch.from_numpy(msg.content["gradients"]["local-gradients"].numpy())

    optimizer.zero_grad()
    embedding = model(X)  # same weights + same RNG -> reproduces the sent embedding

    if (rec := context.state.get("sent_embedding")) is not None:
        sent = torch.from_numpy(rec["v"].numpy())
        if not torch.allclose(embedding.detach(), sent):
            raise RuntimeError(
                f"[{spec.wandb_tag}] Recomputed embedding != sent embedding — "
                "RNG/data-order replay is broken."
            )

    noise = _cut_noise_config(spec, context)
    t0 = time.perf_counter()
    if noise["enabled"]:
        # The server's gradient is w.r.t. the clipped embedding — backprop through
        # the same clip so the bottom model sees the true Jacobian.
        clip_embedding_rows(embedding, noise["max_norm"]).backward(grad)
    else:
        embedding.backward(grad)
    optimizer.step()
    step_time = time.perf_counter() - t0

    context.state["model"] = ArrayRecord(model.state_dict())
    buf = io.BytesIO()
    torch.save(optimizer.state_dict(), buf)
    context.state["optimizer_state"] = ConfigRecord({"serialized": buf.getvalue()})

    final_round = (
        cfg is not None
        and "num-server-rounds" in cfg
        and int(cfg["server-round"]) >= int(cfg["num-server-rounds"]) - 1
    )
    if final_round:
        _save_checkpoint(spec, model, str(cfg["eps-tag"]))

    rnd = _round_number(context, msg, advance=False)
    with torch.no_grad():
        g = grad.detach()
        _log(
            spec,
            context,
            {
                f"{spec.wandb_tag}/grad_l2": float(g.norm(p=2).item()),
                f"{spec.wandb_tag}/grad_max_abs": float(g.abs().max().item()),
                f"{spec.wandb_tag}/step_time_s": step_time,
            },
            step=rnd,
            flush=final_round,
        )

    return Message(content=RecordDict(), reply_to=msg)
