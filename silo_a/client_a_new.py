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

run_config keys (define these in pyproject.toml -> [tool.flwr.app.config],
hyphenated to match server_app.py which already reads "embedding-dim" etc.):
    embedding-dim, learning-rate, input-dim-a, dropout-rate, batch-size, seed
"""

import io

import torch
from flwr.app import Array, ArrayRecord, ConfigRecord, Context, Message, RecordDict
from flwr.clientapp import ClientApp

from silo_a.bottom_model_a import BottomModelA
from silo_a.data_loader_a import load_silo_a
from utils.reproducibility import seed_everything

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
    """
    train_loader, _, _ = load_silo_a(batch_size=batch_size)
    xs, ys = [], []
    for x, y in train_loader:
        xs.append(x)
        ys.append(y)
    return torch.cat(xs).float(), torch.cat(ys).float()


@app.query("generate_embeddings")
def generate_embeddings(msg: Message, context: Context) -> Message:
    """Round step 1: forward pass -> return embedding (+ labels, Silo A only)."""
    model = _build_model(context)
    model.train()

    X, y = _full_trainset(context.run_config.get("batch-size", 64))

    # Snapshot RNG *immediately before* the forward so apply_gradients can
    # reproduce the SAME dropout mask (BottomModelA has Dropout(0.3)).
    rng = torch.get_rng_state()
    embedding = model(X)

    buf = io.BytesIO()
    torch.save(rng, buf)
    context.state["rng"] = ConfigRecord({"serialized": buf.getvalue()})
    context.state["model"] = ArrayRecord(model.state_dict())

    arrays = ArrayRecord(
        {
            "embedding": Array(embedding.detach().numpy()),
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
        optimizer.load_state_dict(torch.load(io.BytesIO(rec["serialized"])))

    # ── WYATT DP HOOK (Week 5) ───────────────────────────────────────────────
    # Opacus now wraps the optimizer HERE (there is no __init__ anymore).
    # Caveat: make_private expects a standard loss.backward() training loop, but
    # the embedding.backward(grad) below is a SplitNN upstream-gradient injection
    # — the integration risk noted in the risk register. Coordinate before Week 5.
    #   from privacy.dp_wrapper import make_private
    #   model, optimizer, _ = make_private(model=model, optimizer=optimizer, ...)
    # ─────────────────────────────────────────────────────────────────────────

    # Restore the exact RNG state from generate_embeddings so dropout matches.
    if (rec := context.state.get("rng")) is not None:
        torch.set_rng_state(torch.load(io.BytesIO(rec["serialized"])))

    X, _ = _full_trainset(context.run_config.get("batch-size", 64))
    optimizer.zero_grad()
    embedding = model(X)  # same weights + same RNG -> reproduces the sent embedding

    grad = torch.from_numpy(msg.content["gradients"]["local-gradients"].numpy())
    embedding.backward(grad)  # server's dL/d_emb flows into bottom-model weights
    optimizer.step()

    # Persist updated weights + optimizer state for the next round.
    context.state["model"] = ArrayRecord(model.state_dict())
    buf = io.BytesIO()
    torch.save(optimizer.state_dict(), buf)
    context.state["optimizer_state"] = ConfigRecord({"serialized": buf.getvalue()})

    return Message(content=RecordDict(), reply_to=msg)