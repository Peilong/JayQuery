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
from flwr.app import Array, ArrayRecord, ConfigRecord, Context, Message, RecordDict
from flwr.common import log
from flwr.serverapp import Grid, ServerApp
from torch.utils.data import DataLoader, TensorDataset

import wandb

from pathlib import Path
import json

from fl_server.top_model import TopModel
from baselines.eval_utils import compute_metrics
from torch.utils.data import TensorDataset, DataLoader

from privacy.dp_wrapper import make_private

from privacy.privacy_accountant import PrivacyBudgetTracker

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
    
    track = bool(context.run_config.get("track-wandb", True))

    dp = _server_dp_config(context)
    train_model, train_optimizer, priv_engine = top_model, optimizer, None
    wrapped = False 

    if track:
        wandb.init(
            entity=context.run_config.get("wandb-entity", "jayquery"),
            project=context.run_config.get("wandb-project", "jayquery-vfl"),
            name=context.run_config.get("wandb-run-name", "") or "server",
            group=context.run_config.get("wandb-group", "") or None,
            tags=["vfl", "server"],
            config={
                "num-server-rounds": num_rounds,
                "learning-rate": lr,
                "embedding-dim": embedding_dim,
                "server-dp-enabled": dp["enabled"],
            },
            resume="never",
        )
        
    eps_tag = f"eps_{dp['epsilon']}" if dp["enabled"] else "eps_inf"

    dp = _server_dp_config(context)      # the Step 1 config helper
    train_model, train_optimizer, priv_engine = top_model, optimizer, None
    wrapped = False 
    privacy_tracker = PrivacyBudgetTracker()
    sample_rate = 1.0  # full-batch training on server, so sample_rate = 1.0

    for round_num in range(1, num_rounds + 1):
        log(INFO, "")
        log(INFO, "--- ServerApp Round %d / %d ---", round_num, num_rounds)

        node_ids = list(grid.get_node_ids())
        if len(node_ids) != 2:
            raise ValueError(
                f"Expected exactly 2 SuperNodes (Silo A + Silo B), got {len(node_ids)}. "
                "Run: flwr federation simulation-config --num-supernodes=2"
            )

        emb_a, emb_b, labels, node_pos_map = _get_embeddings(grid=grid, node_ids=node_ids, round_num=round_num)

        if dp["enabled"] and not wrapped:
            loader = _make_server_loader(emb_a, emb_b, labels)
            train_model, train_optimizer, _, priv_engine = make_private(
                model=top_model, optimizer=optimizer, data_loader=loader,
                target_epsilon=dp["epsilon"], target_delta=dp["delta"],
                max_grad_norm=dp["max_grad_norm"], num_epochs=dp["num_epochs"],
            )
            wrapped = True
            log(INFO, "DP wrap OK: N=%d, dp_optimizer=%s", len(loader.dataset), type(train_optimizer).__name__)
            
        emb_a = emb_a.detach().requires_grad_(True)
        emb_b = emb_b.detach().requires_grad_(True)
        if dp['enabled']:
            train_model.zero_grad(set_to_none=True)
        train_optimizer.zero_grad()
        logits = train_model(emb_a, emb_b).squeeze(1)
        loss = top_model.loss_fn(logits, labels.float())
        loss.backward()
        if emb_a.grad is None or emb_b.grad is None:
            raise RuntimeError(
                f"[server] input grads are None after backward "
                f"(train_model wrapped={type(train_model).__name__}). "
                "Opacus GradSampleModule is not exposing input gradients — "
                "escalate before relying on the DP path."
            )
        train_optimizer.step()
        grad_a, grad_b = emb_a.grad.clone(), emb_b.grad.clone()

        log(INFO, "Round %d — loss: %.4f", round_num, float(loss))
        losses.append((round_num, float(loss)))

        if track:
            wandb.log({"train/loss": float(loss)}, step=round_num)
            if round_num%10==0:
                eval_model = TopModel(embedding_dim=embedding_dim)
                src = train_model._module if dp["enabled"] else top_model
                eval_model.load_state_dict(src.state_dict())
                val_metrics = _run_eval(grid, node_ids, eval_model, split="val")
                wandb.log({"val/auc": float(val_metrics["auc_roc"])}, step=round_num)
    
        _send_gradients(
            grid=grid,
            node_pos_map=node_pos_map,
            grad_a=grad_a,
            grad_b=grad_b,
            round_num=round_num,
            num_rounds=num_rounds,
            eps_tag=eps_tag
        )

    ckpt_dir = Path("experiments/checkpoints") / eps_tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    top_sd = train_model._module.state_dict() if dp["enabled"] else top_model.state_dict()
    torch.save(top_sd, ckpt_dir / "top.pt")
    
    final_eval = TopModel(embedding_dim=embedding_dim)
    final_eval.load_state_dict((train_model._module if dp["enabled"] else top_model).state_dict())
    test_metrics = _run_eval(grid, node_ids, final_eval, split="test")
    
    meta = {
        "epsilon": dp["epsilon"] if dp["enabled"] else None,
        "delta": dp["delta"],
        "embedding_dim": embedding_dim,
        "num_rounds": num_rounds,
        "server_dp_enabled": dp["enabled"],
        "metrics": test_metrics,
    }
    (ckpt_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    log(INFO, "")
    log(INFO, "=== Training Complete ===")
    for r, loss_val in losses:
        log(INFO, "  Round %3d — loss: %.4f", r, loss_val)
    if track:
        wandb.finish()
        
    log(INFO, "Final test metrics: %s", test_metrics)

def _server_dp_config(context: Context) -> dict:
    """Server-side DP-SGD params from run_config (Step 1 keys in pyproject.toml).

    Unlike the client _cut_noise_config, there is no per-silo override — the server
    is a single node, so these come straight from run_config.
    """
    rc = context.run_config
    return {
        "enabled": bool(rc.get("server-dp-enabled", False)),
        "epsilon": float(rc.get("server-dp-epsilon", 1.0)),
        "delta": float(rc.get("server-dp-delta", 1e-5)),
        "max_grad_norm": float(rc.get("server-dp-max-grad-norm", 1.0)),
        "num_epochs": int(rc.get("server-dp-epochs", rc.get("num-server-rounds", 10))),
    }

def _make_server_loader(emb_a: torch.Tensor, emb_b: torch.Tensor, labels: torch.Tensor) -> DataLoader:
    """One full-batch DataLoader over [emb_a‖emb_b] and labels — gives Opacus the
    dataset size it needs. Full-batch => sample_rate = 1.0 (see M1)."""
    X = torch.cat([emb_a, emb_b], dim=1).detach()   # (N, 2*embedding_dim)
    y = labels.detach().float()                     # (N,)
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=len(dataset), shuffle=False)

def _get_embeddings(
    grid: Grid,
    node_ids: list[int],
    round_num: int,
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
            content=RecordDict({"config": ConfigRecord({"server-round": round_num})}),
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
    round_num: int,
    num_rounds: int,
    eps_tag: str,
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
                content=RecordDict({
                    "gradients": arr_rec,
                    "config": ConfigRecord({
                        "server-round": round_num,
                        "num-server-rounds": num_rounds,   # H6.2: lets client detect the last round
                        "eps-tag": eps_tag,                # H6.2: single source of truth for the dir name
                    }),
                }),
                message_type="train.apply_gradients",
                dst_node_id=node_id,
            )
        )

    log(INFO, "Sending gradients to %d nodes...", len(messages))
    grid.push_messages(messages)

def _run_eval(grid: Grid, node_ids: list[int], top_model: TopModel, split: str) -> dict:
    messages = [
        Message(
            content=RecordDict({"config": ConfigRecord({
                "split":split,
            })}),
            message_type="query.generate_eval_embeddings",
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

    top_model.eval()
    with torch.no_grad():
        logits = top_model(emb_a, emb_b).squeeze(1)    
    probs = torch.sigmoid(logits)
    
    return compute_metrics(labels.numpy(), probs.numpy())