"""Flower ServerApp for JayQuery Vertical Federated Learning.

Owner: Alex Pellet

Orchestrates the SplitNN VFL training loop each round:
    1. Query both SuperNodes for embeddings (Silo A also sends labels).
    2. Feed emb_a + emb_b into TopModel; compute BCE loss + backprop.
    3. Push per-silo gradients back to each SuperNode.

The TopModel and its optimizer live entirely on the server; bottom-model
weights stay on each SuperNode (persisted in context.state).

Usage (simulation mode — 2 virtual SuperNodes):
    flwr run . --run-config "num-server-rounds=10"

Usage (deployment mode — three separate processes):
    flwr superlink &
    flwr supernode --partition-id 0 &
    flwr supernode --partition-id 1 &
    flwr run .
    
TODO: 
- Implement Validation Rounds
"""

from logging import INFO

import torch
from flwr.app import Array, ArrayRecord, Context, Message, RecordDict
from flwr.common import log
from flwr.serverapp import Grid, ServerApp

from fl_server.top_model import TopModel

app = ServerApp()

_SILO_A_POS = 0
_SILO_B_POS = 1


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main VFL training loop.

    Args:
        grid: Flower Grid object for communicating with SuperNodes.
        context: Flower context providing run_config.
    """
    num_rounds: int = context.run_config["num-server-rounds"]
    embedding_dim: int = context.run_config["embedding-dim"]
    lr: float = context.run_config["learning-rate"]

    top_model = TopModel(embedding_dim=embedding_dim)
    optimizer = torch.optim.Adam(top_model.parameters(), lr=lr)
    top_model.train()

    losses: list[tuple[int, float]] = []

    for round_num in range(1, num_rounds + 1):
        log(INFO, "")
        log(INFO, "--- ServerApp Round %d / %d ---", round_num, num_rounds)

        node_ids = list(grid.get_node_ids())
        if len(node_ids) != 2:
            raise ValueError(
                f"Expected exactly 2 SuperNodes (Silo A + Silo B), got {len(node_ids)}. "
                "Run: flwr federation simulation-config --num-supernodes=2"
            )

        # 1. Collect embeddings (and labels from Silo A)
        emb_a, emb_b, labels, node_pos_map = _get_embeddings(
            grid=grid,
            node_ids=node_ids,
        )

        # 2. TopModel forward + backward
        optimizer.zero_grad()
        loss, grad_a, grad_b = top_model.compute_loss_and_gradients(emb_a, emb_b, labels)
        optimizer.step()

        log(INFO, "Round %d — loss: %.4f", round_num, float(loss))
        losses.append((round_num, float(loss)))

        # 3. Push per-silo gradients (fire-and-forget)
        _send_gradients(
            grid=grid,
            node_pos_map=node_pos_map,
            grad_a=grad_a,
            grad_b=grad_b,
        )

    log(INFO, "")
    log(INFO, "=== Training Complete ===")
    for r, loss_val in losses:
        log(INFO, "  Round %3d — loss: %.4f", r, loss_val)


def _get_embeddings(
    grid: Grid,
    node_ids: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[int, int]]:
    """Send generate_embeddings query to both SuperNodes and collect replies.

    Args:
        grid: Flower Grid.
        node_ids: List of active SuperNode IDs (exactly 2 expected).

    Returns:
        emb_a: Training embeddings from Silo A, shape (N, embedding_dim).
        emb_b: Training embeddings from Silo B, shape (N, embedding_dim).
        labels: Binary labels from Silo A, shape (N,).
        node_pos_map: Mapping from node_id → partition-id (pos).

    Raises:
        RuntimeError: If either silo does not return a valid reply.
    """
    messages = [
        Message(
            content=RecordDict(),
            message_type="query.generate_embeddings",
            dst_node_id=nid,
        )
        for nid in node_ids
    ]

    log(INFO, "Requesting embeddings from %d nodes...", len(messages))
    replies = grid.send_and_receive(messages)
    log(INFO, "\tReceived %d/%d results", len(replies), len(messages))

    emb_a = emb_b = labels = None
    node_pos_map: dict[int, int] = {}

    for reply in replies:
        if not reply.has_content():
            continue
        pos: int = reply.content["config"]["pos"]
        node_pos_map[reply.metadata.src_node_id] = pos
        emb = torch.from_numpy(reply.content["arrays"]["embedding"].numpy())

        if pos == _SILO_A_POS:
            emb_a = emb
            labels = torch.from_numpy(reply.content["arrays"]["labels"].numpy())
        elif pos == _SILO_B_POS:
            emb_b = emb

    if emb_a is None or emb_b is None or labels is None:
        raise RuntimeError(
            "Failed to collect embeddings from both silos. "
            f"Got positions: {list(node_pos_map.values())}"
        )

    return emb_a, emb_b, labels, node_pos_map


def _send_gradients(
    grid: Grid,
    node_pos_map: dict[int, int],
    grad_a: torch.Tensor,
    grad_b: torch.Tensor,
) -> None:
    """Push per-silo embedding gradients back to each SuperNode.

    Uses push_messages (no reply expected); each SuperNode's apply_gradients
    handler will receive the gradient and apply it in the next round.

    Args:
        grid: Flower Grid.
        node_pos_map: Mapping from node_id → partition-id.
        grad_a: Gradient w.r.t. Silo A embedding, shape (N, embedding_dim).
        grad_b: Gradient w.r.t. Silo B embedding, shape (N, embedding_dim).
    """
    pos_to_grad = {_SILO_A_POS: grad_a, _SILO_B_POS: grad_b}
    messages = []
    for node_id, pos in node_pos_map.items():
        arr_rec = ArrayRecord({"local-gradients": Array(pos_to_grad[pos].cpu().numpy())})
        messages.append(
            Message(
                content=RecordDict({"gradients": arr_rec}),
                message_type="train.apply_gradients",
                dst_node_id=node_id,
            )
        )

    log(INFO, "Sending gradients to %d nodes...", len(messages))
    grid.push_messages(messages)
