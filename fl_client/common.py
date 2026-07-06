"""Generic VFL client handlers, parameterized by SiloSpec.

Owner: Alex Pellet

Generalized from the original client_a_new.py. Behavior is unchanged from
that file for Silo A; Silo B (and any future silo) gets the same logic via
its own SiloSpec.
"""

import io
import time

import torch
from flwr.app import Array, ArrayRecord, ConfigRecord, Context, Message, RecordDict

from fl_client.silo_spec import SiloSpec
from privacy.dp_wrapper import make_private
from privacy.noise_injection import add_dp_noise_to_embedding
from utils.wandb_utils import log_client_round, new_run_id


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

# ── Configs ───────────────────────────────────────────────────

def _dp_config(spec: SiloSpec, context: Context) -> dict:
    """DP params, run_config defaults overridden by per-silo SiloSpec values."""
    rc = context.run_config
    cfg = {
        "enabled": bool(rc.get("dp-enabled", False)),
        "epsilon": float(rc.get("dp-epsilon", 1.0)),
        "delta": float(rc.get("dp-delta", 1e-5)),
        "max_grad_norm": float(rc.get("dp-max-grad-norm", 1.0)),
        "num_epochs": int(rc.get("dp-epochs", rc.get("num-server-rounds", 10))),
    }
    cfg.update(spec.dp_config_fn(context))  # silo-specific overrides
    return cfg

def _cut_noise_config(spec: SiloSpec, context: Context) -> dict:
    """Noise injection params, run_config defaults overridden by per-silo SiloSpec values."""
    rc = context.run_config
    cfg = {
        "enabled": bool(rc.get("cut-noise-enabled", False)),
        "epsilon": float(rc.get("cut-noise-epsilon", 1.0)),
        "max_grad_norm": float(rc.get("cut-noise-max-grad-norm", 1.0)),
    }
    cfg.update(spec.cut_noise_config_fn(context))  # silo-specific overrides
    return cfg

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


def _log(spec: SiloSpec, context: Context, metrics: dict, step: int) -> None:
    """Resume this node's wandb run, log metrics at step, finish."""
    if not _tracking_enabled(context):
        return
    rc = context.run_config
    rid = _run_id(context)
    log_client_round(
        run_id=rid,
        metrics=metrics,
        step=step,
        project=rc.get("wandb-project", "jayquery-vfl"),
        entity=rc.get("wandb-entity", "jayquery"),
        name=rc.get("wandb-run-name", "") or f"{spec.wandb_tag}-{rid[:8]}",
        group=rc.get("wandb-group", "") or None,
        tags=["vfl", spec.wandb_tag, "client"],
        mode=rc.get("wandb-mode", "online"),
        config=_wandb_config(spec, context),
    )


# ── handlers ─────────────────────────────────────────────────────────────────


def generate_embeddings(spec: SiloSpec, msg: Message, context: Context) -> Message:
    """Round step 1: forward pass -> return embedding (+ labels if spec.sends_labels)."""
    model = _build_model(spec, context)
    model.train()

    X, y = _full_trainset(spec, context)

    # Snapshot RNG immediately before the forward so apply_gradients can
    # reproduce the same dropout mask.
    rng = torch.get_rng_state()
    t0 = time.perf_counter()
    embedding = model(X)
    fwd_time = time.perf_counter() - t0

    buf = io.BytesIO()
    torch.save(rng, buf)
    context.state["rng"] = ConfigRecord({"serialized": buf.getvalue()})
    context.state["model"] = ArrayRecord(model.state_dict())

    emb = embedding.detach()
    # Stored for the apply_gradients equivalence check below.
    context.state["sent_embedding"] = ArrayRecord({"v": Array(emb.numpy())})

    rnd = _round_number(context, msg, advance=True)
    with torch.no_grad():
        _log(
            spec,
            context,
            {
                f"{spec.wandb_tag}/round": rnd,
                f"{spec.wandb_tag}/forward_time_s": fwd_time,
                f"{spec.wandb_tag}/embedding_l2": float(emb.norm(p=2).item()),
                f"{spec.wandb_tag}/embedding_mean": float(emb.mean().item()),
                f"{spec.wandb_tag}/embedding_std": float(emb.std().item()),
                f"{spec.wandb_tag}/n_examples": int(emb.shape[0]),
            },
            step=rnd,
        )

    noise = _cut_noise_config(spec, context)
    if noise["enabled"]:
        emb = add_dp_noise_to_embedding(
            embedding = embedding,
            sensitivity = noise["max_grad_norm"],
            epsilon = noise["epsilon"]
        )

    arrays = {"embedding": Array(emb.numpy())}
    if spec.sends_labels:
        arrays["labels"] = Array(y.numpy())

    content = RecordDict(
        {"arrays": ArrayRecord(arrays), "config": ConfigRecord({"pos": spec.partition_id})}
    )
    return Message(content=content, reply_to=msg)


def apply_gradients(spec: SiloSpec, msg: Message, context: Context) -> Message:
    """Round step 3: recompute forward, inject server gradient, step optimizer."""
    model = _build_model(spec, context)
    model.train()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=context.run_config.get("learning-rate", 1e-3)
    )
    if (rec := context.state.get("optimizer_state")) is not None:
        optimizer.load_state_dict(
            torch.load(io.BytesIO(rec["serialized"]), weights_only=False)
        )

    dp = _dp_config(spec, context)
    if dp["enabled"]:
        # Not Implemented. Outlined in tasklist
        raise NotImplementedError(
            f"[{spec.wandb_tag}] DP-SGD not yet wired for the VFL upstream-gradient "
            "loop (concern list H1–H3). Run with dp-enabled=false until Phase B."
        )

    # Must remain before RNG restore
    X, _ = _full_trainset(spec, context)

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

    t0 = time.perf_counter()
    embedding.backward(grad)
    optimizer.step()
    step_time = time.perf_counter() - t0

    context.state["model"] = ArrayRecord(model.state_dict())
    buf = io.BytesIO()
    torch.save(optimizer.state_dict(), buf)
    context.state["optimizer_state"] = ConfigRecord({"serialized": buf.getvalue()})

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
        )

    return Message(content=RecordDict(), reply_to=msg)