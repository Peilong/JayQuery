"""Flower ClientApp shared by both JayQuery data silos.

Owner: Alex Pellet

Partition mapping (set via SuperNode or simulation config):
    partition-id 0  →  Silo A (BottomModelA, 12 demographic features)
    partition-id 1  →  Silo B (BottomModelB, 22 VLE/assessment features)

Each round has two message types:
    query.generate_embeddings  — forward pass, returns embedding (+ labels for silo A)
    train.apply_gradients      — backward pass, applies server gradient, updates model

Data loading priority:
    1. Real OULAD data from data/processed/silo_{a,b}.pkl (plug-and-play once prepared)
    2. Synthetic data (seeded, row-aligned across silos) used automatically as fallback
"""

import io
import warnings

import torch
from flwr.app import Array, ArrayRecord, ConfigRecord, Context, Message, RecordDict
from flwr.clientapp import ClientApp

from silo_a.bottom_model_a import BottomModelA
from silo_a.data_loader_a import load_silo_a, make_synthetic_loader_a
from silo_b.bottom_model_b import BottomModelB
from silo_b.data_loader_b import load_silo_b, make_synthetic_loader_b

app = ClientApp()

# Module-level cache: partition_id → (X_train, y_train) as tensors.
# Populated on first access; persists for the lifetime of the SuperNode process.
_data_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}


def _load_train_tensors(
    partition_id: int,
    processed_dir: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return all training features and labels as float32 tensors.

    Tries the real OULAD pickle first; falls back to synthetic data if it does
    not exist. Both silos use seed=42 so that labels are identical and rows are
    aligned across partitions.

    Args:
        partition_id: 0 for Silo A, 1 for Silo B.
        processed_dir: Path to the directory containing silo_*.pkl.

    Returns:
        Tuple of (X_train, y_train) float32 tensors.
    """
    if partition_id in _data_cache:
        return _data_cache[partition_id]

    if partition_id == 0:
        try:
            train_loader, _, _ = load_silo_a(processed_dir=processed_dir)
        except FileNotFoundError:
            warnings.warn(
                f"[SiloA] {processed_dir}/silo_a.pkl not found — using synthetic data. "
                "Run scripts/prepare_silos.py + psi_client to use real OULAD data."
            )
            train_loader, _, _ = make_synthetic_loader_a()
    else:
        try:
            train_loader, _, _ = load_silo_b(processed_dir=processed_dir)
        except FileNotFoundError:
            warnings.warn(
                f"[SiloB] {processed_dir}/silo_b.pkl not found — using synthetic data. "
                "Run scripts/prepare_silos.py to use real OULAD data."
            )
            train_loader, _, _ = make_synthetic_loader_b()

    X_all = torch.cat([x for x, _ in train_loader])
    y_all = torch.cat([y for _, y in train_loader])
    _data_cache[partition_id] = (X_all, y_all)
    return X_all, y_all


def _build_model(partition_id: int, context: Context) -> torch.nn.Module:
    """Instantiate the correct BottomModel for the given partition.

    Args:
        partition_id: 0 → BottomModelA, 1 → BottomModelB.
        context: Flower context providing run_config hyperparameters.

    Returns:
        Uninitialized (or state-loaded) BottomModel instance.
    """
    embedding_dim: int = context.run_config["embedding-dim"]
    if partition_id == 0:
        return BottomModelA(
            input_dim=context.run_config["input-dim-a"],
            embedding_dim=embedding_dim,
        )
    return BottomModelB(
        input_dim=context.run_config["input-dim-b"],
        embedding_dim=embedding_dim,
    )


@app.query("generate_embeddings")
def get_emb(msg: Message, context: Context) -> Message:
    """Forward pass: compute embedding over the full training set and return it.

    Silo A (partition-id 0) also includes the label vector so the server can
    compute BCEWithLogitsLoss without holding a separate copy of the labels.

    Args:
        msg: Incoming query message from the ServerApp.
        context: Flower context with node_config and run_config.

    Returns:
        Reply Message containing:
            arrays.embedding  — float32 array (N, embedding_dim)
            arrays.labels     — float32 array (N,) — Silo A only
            config.pos        — partition_id (int), used by server for routing
    """
    partition_id: int = context.node_config["partition-id"]
    processed_dir: str = context.run_config.get("processed-dir", "data/processed")

    X, y = _load_train_tensors(partition_id, processed_dir)

    model = _build_model(partition_id, context)
    if model_record := context.state.get("model", None):
        model.load_state_dict(model_record.to_torch_state_dict())
    model.eval()

    with torch.no_grad():
        embedding = model(X)

    arrays: dict[str, Array] = {"embedding": Array(embedding.numpy())}
    if partition_id == 0:
        arrays["labels"] = Array(y.numpy())

    content = RecordDict(
        {
            "arrays": ArrayRecord(arrays),
            "config": ConfigRecord({"pos": partition_id}),
        }
    )
    return Message(content=content, reply_to=msg)


@app.train("apply_gradients")
def apply_grad(msg: Message, context: Context) -> Message:
    """Backward pass: apply the server's gradient and update the bottom model.

    Re-runs the full forward pass (needed to build the computation graph) then
    calls embedding.backward(grad) followed by optimizer.step().

    Args:
        msg: Message from ServerApp containing gradients/local-gradients (N, emb_dim).
        context: Flower context with node_config and run_config.

    Returns:
        Empty reply Message (server does not use the reply payload).
    """
    partition_id: int = context.node_config["partition-id"]
    lr: float = context.run_config["learning-rate"]
    processed_dir: str = context.run_config.get("processed-dir", "data/processed")

    X, _ = _load_train_tensors(partition_id, processed_dir)

    model = _build_model(partition_id, context)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if model_record := context.state.get("model", None):
        model.load_state_dict(model_record.to_torch_state_dict())
    if opt_record := context.state.get("optimizer_state", None):
        buf = io.BytesIO(opt_record["serialized"])
        optimizer.load_state_dict(torch.load(buf))

    model.train()
    optimizer.zero_grad()
    embedding = model(X)

    grad = torch.from_numpy(msg.content["gradients"]["local-gradients"].numpy())
    embedding.backward(grad)
    optimizer.step()

    # Persist updated model and optimizer state for the next round
    context.state["model"] = ArrayRecord(model.state_dict())
    torch.save(optimizer.state_dict(), buf := io.BytesIO())
    context.state["optimizer_state"] = ConfigRecord({"serialized": buf.getvalue()})

    return Message(content=RecordDict(), reply_to=msg)
