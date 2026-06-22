"""Flower ClientApp for Silo A — Financial Aid / Demographics.

Owner: Alex Pellet

This is the Message-API counterpart to server_app.py. It is NOT a NumPyClient.
The ServerApp drives a custom VFL loop by sending two message types each round;
this file registers exactly one handler per type:

    server  "query.generate_embeddings"  ->  @app.query("generate_embeddings")
    server  "train.apply_gradients"       ->  @app.train("apply_gradients")

Key difference from a NumPyClient: handlers are STATELESS invocations. There is
no persistent client object, so BottomModelA's weights and the Adam optimizer
state do NOT live on `self` — they are serialized into `context.state` between
rounds and reloaded at the start of each handler.

wandb tracking (added):
    Because handlers are stateless, a Run object cannot be carried across rounds.
    We instead persist a stable wandb run id in context.state and, each round,
    *resume* that run (resume="allow"), log the client-observable signals, and
    finish. All wandb lifecycle is delegated to baselines.wandb_utils per the
    repo rule that model files never call wandb.init()/finish() directly. Every
    wandb call is best-effort: a logging failure NEVER breaks the FL round.

    Silo A cannot see the loss/AUC (those live on the Top Model server-side), so
    the client only logs what it actually holds: embedding statistics, the
    incoming dL/d_emb gradient norm (also useful for Wyatt's gradient_monitor),
    and per-step timing. Hyperparameters are logged as wandb config so a
    parameter sweep (see baselines.wandb_utils.build_flwr_sweep_fn) can correlate
    them with the server-side metric.

run_config keys (define these in pyproject.toml -> [tool.flwr.app.config],
hyphenated to match server_app.py which already reads "embedding-dim" etc.):
    embedding-dim, learning-rate, input-dim-a, dropout-rate, batch-size, seed
    track-wandb (bool, default True), wandb-project, wandb-entity, wandb-group,
    wandb-run-name, wandb-mode ("online" | "offline" | "disabled")
"""

import io
import time

import torch
from flwr.app import Array, ArrayRecord, ConfigRecord, Context, Message, RecordDict
from flwr.clientapp import ClientApp

from baselines.wandb_utils import log_client_round, new_run_id
from silo_a.bottom_model_a import BottomModelA
from silo_a.data_loader_a import load_silo_a
from utils.reproducibility import seed_everything  # noqa: F401  (kept for parity)

app = ClientApp()

SILO_A_POS = 0  # MUST match _SILO_A_POS in server_app.py


def _build_model(context: Context) -> BottomModelA:
    """Construct BottomModelA, restoring weights from context.state if present."""
    rc = context.run_config
    model = BottomModelA(
        input_dim=rc.get("input-dim-a", 18),
        embedding_dim=rc.get("embedding-dim", 32),
        dropout_rate=rc.get("dropout-rate", 0.3),
    )
    if (rec := context.state.get("model")) is not None:
        model.load_state_dict(rec.to_torch_state_dict())
    return model


def _full_trainset(batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize the ENTIRE Silo A training set as one (X, y) pair.

    VFL here is full-batch: one forward over all PSI-aligned rows per round, not
    a mini-batch loop. Row order must be identical every round so the recomputed
    forward in apply_gradients reproduces the embedding we sent.

    NOTE: this requires load_silo_a()'s train_loader to be deterministic
    (shuffle=False, or a fixed sampler). If it shuffles, the row order — and
    therefore the embedding — will differ between generate_embeddings and
    apply_gradients, and the injected gradient will be misaligned. Verify this
    in silo_a/data_loader_a.py.
    """
    train_loader, _, _ = load_silo_a(batch_size=batch_size)
    xs, ys = [], []
    for x, y in train_loader:
        xs.append(x)
        ys.append(y)
    return torch.cat(xs).float(), torch.cat(ys).float()


# ── wandb tracking helpers ───────────────────────────────────────────────────
# All of these are inexpensive and side-effect-free except _log(), which routes
# through baselines.wandb_utils.log_client_round() (best-effort, never raises).


def _tracking_enabled(context: Context) -> bool:
    return bool(context.run_config.get("track-wandb", True))


def _run_id(context: Context) -> str:
    """Return a stable wandb run id for this node, persisted in context.state.

    Generated once (on the first handler call for this node) and reused for the
    remainder of the run so every round resumes the SAME wandb run.
    """
    rec = context.state.get("wandb")
    if rec is not None and "run_id" in rec:
        return str(rec["run_id"])
    rid = new_run_id()
    context.state["wandb"] = ConfigRecord({"run_id": rid})
    return rid


def _round_number(context: Context, msg: Message, *, advance: bool) -> int:
    """Resolve the current FL round for use as the wandb logging step.

    Prefers a server-provided "server-round" in the message config; otherwise
    falls back to a counter persisted in context.state. generate_embeddings
    (round step 1) advances the counter; apply_gradients (round step 3) reads
    the same value so both log to one step.
    """
    cfg = msg.content.get("config") if msg.content is not None else None
    if cfg is not None and "server-round" in cfg:
        r = int(cfg["server-round"])
    else:
        prev = int(context.state["round"]["v"]) if "round" in context.state else -1
        r = prev + 1 if advance else max(prev, 0)
    context.state["round"] = ConfigRecord({"v": r})
    return r


def _wandb_config(context: Context) -> dict:
    rc = context.run_config
    return {
        "silo": "A",
        "embedding-dim": rc.get("embedding-dim", 32),
        "learning-rate": rc.get("learning-rate", 1e-3),
        "dropout-rate": rc.get("dropout-rate", 0.3),
        "input-dim-a": rc.get("input-dim-a", 18),
        "batch-size": rc.get("batch-size", 64),
        "seed": rc.get("seed", 42),
    }


def _log(context: Context, metrics: dict, step: int) -> None:
    """Resume this node's wandb run, log `metrics` at `step`, finish."""
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
        name=rc.get("wandb-run-name", "") or f"silo_a-{rid[:8]}",
        group=rc.get("wandb-group", "") or None,
        tags=["vfl", "silo_a", "client"],
        mode=rc.get("wandb-mode", "online"),
        config=_wandb_config(context),
    )


@app.query("generate_embeddings")
def generate_embeddings(msg: Message, context: Context) -> Message:
    """Round step 1: forward pass -> return embedding (+ labels, Silo A only)."""
    model = _build_model(context)
    model.train()

    X, y = _full_trainset(context.run_config.get("batch-size", 64))

    # Snapshot RNG *immediately before* the forward so apply_gradients can
    # reproduce the SAME dropout mask (BottomModelA has Dropout(0.3)).
    rng = torch.get_rng_state()
    t0 = time.perf_counter()
    embedding = model(X)
    fwd_time = time.perf_counter() - t0

    buf = io.BytesIO()
    torch.save(rng, buf)
    context.state["rng"] = ConfigRecord({"serialized": buf.getvalue()})
    context.state["model"] = ArrayRecord(model.state_dict())

    emb = embedding.detach()

    rnd = _round_number(context, msg, advance=True)
    with torch.no_grad():
        _log(
            context,
            {
                "client_a/round": rnd,
                "client_a/forward_time_s": fwd_time,
                "client_a/embedding_l2": float(emb.norm(p=2).item()),
                "client_a/embedding_mean": float(emb.mean().item()),
                "client_a/embedding_std": float(emb.std().item()),
                "client_a/n_examples": int(emb.shape[0]),
            },
            step=rnd,
        )

    arrays = ArrayRecord(
        {
            "embedding": Array(emb.numpy()),
            # Silo A is the active (label-holding) party; server reads labels
            # only from pos == _SILO_A_POS. Omit this line in client_b.py.
            "labels": Array(y.numpy()),
        }
    )
    content = RecordDict(
        {"arrays": arrays, "config": ConfigRecord({"pos": SILO_A_POS})}
    )
    return Message(content=content, reply_to=msg)


@app.train("apply_gradients")
def apply_gradients(msg: Message, context: Context) -> Message:
    """Round step 3: recompute forward, inject server gradient, step optimizer."""
    model = _build_model(context)
    model.train()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=context.run_config.get("learning-rate", 1e-3)
    )
    if (rec := context.state.get("optimizer_state")) is not None:
        # weights_only=False is explicit: this is our own trusted state, and on
        # torch 2.4.1 the implicit default already warns; on torch >=2.6 the
        # default flips to True and would reject the pickled optimizer state.
        optimizer.load_state_dict(
            torch.load(io.BytesIO(rec["serialized"]), weights_only=False)
        )

    # ── WYATT DP HOOK (Week 5) ───────────────────────────────────────────────
    # Opacus now wraps the optimizer HERE (there is no __init__ anymore).
    # Caveat: make_private expects a standard loss.backward() training loop, but
    # the embedding.backward(grad) below is a SplitNN upstream-gradient injection
    # — the integration risk noted in the risk register. Coordinate before Week 5.
    #   from privacy.dp_wrapper import make_private
    #   model, optimizer, _ = make_private(model=model, optimizer=optimizer, ...)
    # ─────────────────────────────────────────────────────────────────────────

    # Load the data BEFORE restoring the RNG. The forward's dropout mask is
    # driven by the GLOBAL torch RNG; if load_silo_a() consumes any RNG (e.g. a
    # shuffling sampler), doing it AFTER set_rng_state() would desync the mask
    # from generate_embeddings and the recomputed embedding would not match the
    # one we sent. This mirrors generate_embeddings' ordering: load -> snapshot
    # RNG -> forward.
    X, _ = _full_trainset(context.run_config.get("batch-size", 64))

    # Restore the exact RNG state from generate_embeddings so dropout matches.
    if (rec := context.state.get("rng")) is not None:
        torch.set_rng_state(
            torch.load(io.BytesIO(rec["serialized"]), weights_only=False)
        )

    grad = torch.from_numpy(msg.content["gradients"]["local-gradients"].numpy())

    optimizer.zero_grad()
    embedding = model(X)  # same weights + same RNG -> reproduces the sent embedding
    t0 = time.perf_counter()
    embedding.backward(grad)  # server's dL/d_emb flows into bottom-model weights
    optimizer.step()
    step_time = time.perf_counter() - t0

    # Persist updated weights + optimizer state for the next round.
    context.state["model"] = ArrayRecord(model.state_dict())
    buf = io.BytesIO()
    torch.save(optimizer.state_dict(), buf)
    context.state["optimizer_state"] = ConfigRecord({"serialized": buf.getvalue()})

    rnd = _round_number(context, msg, advance=False)
    with torch.no_grad():
        g = grad.detach()
        _log(
            context,
            {
                "client_a/grad_l2": float(g.norm(p=2).item()),
                "client_a/grad_max_abs": float(g.abs().max().item()),
                "client_a/step_time_s": step_time,
            },
            step=rnd,
        )

    return Message(content=RecordDict(), reply_to=msg)