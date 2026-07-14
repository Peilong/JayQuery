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

import json
from dataclasses import dataclass, field
from logging import INFO, WARNING
from pathlib import Path

import torch
from flwr.app import Array, ArrayRecord, ConfigRecord, Context, Message, RecordDict
from flwr.common import log
from flwr.serverapp import Grid, ServerApp

import wandb
from baselines.eval_utils import compute_metrics
from fl_server.top_model import TopModel
from privacy.dp_wrapper import make_private_with_noise
from privacy.privacy_accountant import (
    PrivacyBudgetTracker,
    combine_epsilon,
    solve_noise_multiplier,
)

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
    train_model, train_optimizer = top_model, optimizer
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

    sample_rate = float(context.run_config.get("batch-sample-rate", 1.0))
    seed = int(context.run_config.get("seed", 42))
    combined_target = float(context.run_config.get("combined-target-epsilon", 3.0))
    server_tracker: PrivacyBudgetTracker | None = None
    client_trackers: dict[int, PrivacyBudgetTracker] = {}
    stopped_early = False

    for round_num in range(1, num_rounds + 1):
        log(INFO, "")
        log(INFO, "--- ServerApp Round %d / %d ---", round_num, num_rounds)

        node_ids = list(grid.get_node_ids())
        if len(node_ids) != 2:
            raise ValueError(
                f"Expected exactly 2 SuperNodes (Silo A + Silo B), got {len(node_ids)}. "
                "Run: flwr federation simulation-config --num-supernodes=2"
            )

        rd = _get_embeddings(
            grid=grid, node_ids=node_ids, round_num=round_num,
            batch_seed=seed * 100_000 + round_num, sample_rate=sample_rate,
        )
        for pos, prims in rd.client_privacy.items():
            client_trackers[pos] = PrivacyBudgetTracker.from_dict(prims)

        if rd.exhausted:
            log(WARNING, "Silo(s) %s out of privacy budget — stopping run at round %d",
                rd.exhausted, round_num)
            _send_stop(grid, node_ids, eps_tag)
            stopped_early = True
            break

        emb_a, emb_b, labels, node_pos_map = rd.emb_a, rd.emb_b, rd.labels, rd.node_pos_map

        if dp["enabled"] and not wrapped:
            # Size the noise against the same accountant that enforces it (autodp),
            # over the actual run length: one optimizer.step per round.
            sigma = solve_noise_multiplier(
                target_epsilon=dp["epsilon"],
                sample_rate=sample_rate,
                steps=num_rounds,
                delta=dp["delta"],
            )
            expected_batch = max(1, round(sample_rate * rd.n_total)) if rd.n_total else len(labels)
            train_model, train_optimizer = make_private_with_noise(
                model=top_model, optimizer=optimizer,
                noise_multiplier=sigma, max_grad_norm=dp["max_grad_norm"],
                expected_batch_size=expected_batch,
            )
            wrapped = True
            log(INFO, "DP wrap OK: expected_batch=%d, sigma=%.4f, horizon=%d steps",
                expected_batch, sigma, num_rounds)
            server_tracker = PrivacyBudgetTracker(
                target_epsilon=dp["epsilon"],
                sigma=sigma,
                sample_rate=sample_rate,
                delta=dp["delta"],
            )

        if server_tracker is not None and server_tracker.would_exceed():
            log(WARNING, "Server budget would be exceeded — stopping run before round %d step",
                round_num)
            _send_stop(grid, node_ids, eps_tag)
            stopped_early = True
            break

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

        if server_tracker is not None:
            server_tracker.step()
        eps_metrics = _epsilon_metrics(server_tracker, client_trackers, dp["delta"])
        for k, v in eps_metrics.items():
            log(INFO, "Round %d — %s: %.4f", round_num, k, v)

        if track:
            wandb.log({"train/loss": float(loss), **eps_metrics}, step=round_num)
            if round_num%5==0:
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

        next_combined = _next_round_combined(server_tracker, client_trackers, dp["delta"])
        if next_combined is not None and next_combined > combined_target:
            log(WARNING,
                "Next round would push combined epsilon to %.4f > %.4f — stopping after round %d",
                next_combined, combined_target, round_num)
            _send_stop(grid, node_ids, eps_tag)
            stopped_early = True
            break

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
        "batch_sample_rate": sample_rate,
        "combined_target_epsilon": combined_target,
        "stopped_early": stopped_early,
        "epsilon_spent": _epsilon_metrics(server_tracker, client_trackers, dp["delta"]),
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
    }

def _pos_name(pos: int) -> str:
    return "silo_a" if pos == _SILO_A_POS else "silo_b"


def _epsilon_metrics(
    server_tracker: PrivacyBudgetTracker | None,
    client_trackers: dict[int, PrivacyBudgetTracker],
    delta: float,
) -> dict[str, float]:
    """Per-source cumulative epsilons plus the composed system total."""
    metrics: dict[str, float] = {}
    if server_tracker is not None:
        metrics["privacy/server_epsilon"] = server_tracker.get_epsilon()
    for pos, tracker in sorted(client_trackers.items()):
        metrics[f"privacy/{_pos_name(pos)}_epsilon"] = tracker.get_epsilon()
    all_trackers = list(client_trackers.values()) + (
        [server_tracker] if server_tracker is not None else []
    )
    if all_trackers:
        metrics["privacy/combined_epsilon"] = combine_epsilon(all_trackers, delta=delta)
    return metrics


def _next_round_combined(
    server_tracker: PrivacyBudgetTracker | None,
    client_trackers: dict[int, PrivacyBudgetTracker],
    delta: float,
) -> float | None:
    """Prospective combined epsilon if every active source spends one more step."""
    active = list(client_trackers.values()) + (
        [server_tracker] if server_tracker is not None else []
    )
    if not active:
        return None
    probes = []
    for tracker in active:
        probe = PrivacyBudgetTracker.from_dict(tracker.to_dict())
        probe.steps += 1
        probes.append(probe)
    return combine_epsilon(probes, delta=delta)


def _send_stop(grid: Grid, node_ids: list[int], eps_tag: str) -> None:
    """Tell every SuperNode the run is over (budget gate): checkpoint, no training."""
    messages = [
        Message(
            content=RecordDict(
                {"config": ConfigRecord({"stop-run": True, "eps-tag": eps_tag})}
            ),
            message_type="train.apply_gradients",
            dst_node_id=nid,
        )
        for nid in node_ids
    ]
    grid.push_messages(messages)


@dataclass
class _RoundData:
    """One round's collected client replies."""

    emb_a: torch.Tensor | None = None
    emb_b: torch.Tensor | None = None
    labels: torch.Tensor | None = None
    node_pos_map: dict[int, int] = field(default_factory=dict)
    n_total: int = 0
    client_privacy: dict[int, dict] = field(default_factory=dict)
    exhausted: list[int] = field(default_factory=list)


def _get_embeddings(
    grid: Grid,
    node_ids: list[int],
    round_num: int,
    batch_seed: int,
    sample_rate: float,
) -> _RoundData:
    """Query both SuperNodes for this round's batch embeddings.

    Sends the shared batch seed + sample rate so both silos Poisson-sample the
    same rows. Replies carry each silo's privacy-tracker primitives and may flag
    budget exhaustion instead of returning an embedding.

    Raises:
        RuntimeError: If a silo returns neither an embedding nor an exhausted flag.
    """
    messages = [
        Message(
            content=RecordDict({"config": ConfigRecord({
                "server-round": round_num,
                "batch-seed": batch_seed,
                "batch-sample-rate": sample_rate,
            })}),
            message_type="query.generate_embeddings",
            dst_node_id=nid,
        )
        for nid in node_ids
    ]

    log(INFO, "Requesting embeddings from %d nodes...", len(messages))
    replies = grid.send_and_receive(messages)
    log(INFO, "\tReceived %d/%d results", len(replies), len(messages))

    rd = _RoundData()
    for reply in replies:
        if not reply.has_content():
            continue
        cfg = reply.content["config"]
        pos: int = cfg["pos"]
        rd.node_pos_map[reply.metadata.src_node_id] = pos
        if "privacy" in reply.content:
            rd.client_privacy[pos] = dict(reply.content["privacy"])
        if bool(cfg.get("budget-exhausted", False)):
            rd.exhausted.append(pos)
            continue
        if "n-total" in cfg:
            rd.n_total = int(cfg["n-total"])
        emb = torch.from_numpy(reply.content["arrays"]["embedding"].numpy())
        if pos == _SILO_A_POS:
            rd.emb_a = emb
            rd.labels = torch.from_numpy(reply.content["arrays"]["labels"].numpy())
        elif pos == _SILO_B_POS:
            rd.emb_b = emb

    if not rd.exhausted and (rd.emb_a is None or rd.emb_b is None or rd.labels is None):
        raise RuntimeError(
            "Failed to collect embeddings from both silos. "
            f"Got positions: {list(rd.node_pos_map.values())}"
        )
    return rd


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
                        "num-server-rounds": num_rounds,  # lets client detect the last round
                        "eps-tag": eps_tag,               # single source of truth for the dir
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