"""Flower ClientApp dispatcher for JayQuery (Silo A / Silo B).

Owner: Alex Pellet

This file is intentionally a thin router. It owns NO model code, NO data
loading, and NO silo-specific logic — that all lives in the per-silo handler
modules (fl_client/silo_a/handlers.py, fl_client/silo_b/handlers.py), which are
independently ownable/editable without touching this file.

Why a dispatcher at all (and not two separate apps):
    Flower's deployment runtime delivers ONE app (one FAB) to every SuperNode in
    a run; a SuperNode is differentiated only by its --node-config, not by which
    code it loads. So "Silo A" and "Silo B" cannot be two genuinely separate
    Flower apps within one `flwr run` — they must be one app that branches on
    node_config["partition-id"]. This file is that branch point, kept as small
    as possible so the branch is the ONLY thing it does.

Partition mapping (set via --node-config or simulation config):
    partition-id 0  →  Silo A  →  fl_client.silo_a.handlers
    partition-id 1  →  Silo B  →  fl_client.silo_b.handlers

Each silo's handlers module must expose exactly:
    get_emb(msg: Message, context: Context) -> Message
    apply_grad(msg: Message, context: Context) -> Message

Adding a silo: drop in fl_client/silo_x/handlers.py with that same signature,
add one line to _HANDLERS below. Nothing else in this file changes.
"""

from flwr.app import Context, Message
from flwr.clientapp import ClientApp

from fl_client.silo_a import handlers as silo_a
from fl_client.silo_b import handlers as silo_b

app = ClientApp()

# partition-id -> handler module. Each module owns its own model + data logic.
_HANDLERS = {
    0: silo_a,
    1: silo_b,
}


def _handlers_for(context: Context):
    """Resolve the handler module for this node's partition-id.

    Raises:
        KeyError: if partition-id is missing or unmapped — surfaced as-is
            rather than silently defaulting, since running the wrong silo's
            logic on a node is a correctness bug, not a recoverable case.
    """
    partition_id: int = context.node_config["partition-id"]
    try:
        return _HANDLERS[partition_id]
    except KeyError:
        raise KeyError(
            f"No handler registered for partition-id={partition_id!r}. "
            f"Known partitions: {list(_HANDLERS)}. Check --node-config."
        ) from None


@app.query("generate_embeddings")
def get_emb(msg: Message, context: Context) -> Message:
    """Forward to this node's silo-specific embedding handler."""
    return _handlers_for(context).get_emb(msg, context)


@app.train("apply_gradients")
def apply_grad(msg: Message, context: Context) -> Message:
    """Forward to this node's silo-specific gradient-application handler."""
    return _handlers_for(context).apply_grad(msg, context)