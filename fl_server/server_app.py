"""Flower ServerApp for JayQuery Vertical Federated Learning.

Owner: Alex Pellet

Orchestrates the SplitNN VFL training loop each round:
    1. Query both SuperNodes for embeddings (Silo A also sends labels).
    2. Feed emb_a + emb_b into TopModel; compute BCE loss + backprop.
    3. Push per-silo gradients back to each SuperNode.

The TopModel and its optimizer live entirely on the server; bottom-model
weights stay on each SuperNode (persisted in context.state).

Early stopping + best-state checkpointing (added):
    The server tracks the best val/auc across eval rounds. On a new best it
    (a) snapshots the UNWRAPPED top-model state in memory and (b) flags the
    round's gradient message with `save-best` so each client snapshots its own
    bottom model (bottoms never leave the node — threat-model safe). Training
    stops after `early-stop-patience` eval events without improvement. At the
    end the server writes the best top model and the final test eval carries
    `restore-best`, so clients restore + checkpoint their best bottoms in the
    same round-trip and the reported test metric reflects the peak, not the
    collapsed last state.

Usage (simulation mode — 2 virtual SuperNodes):
    flwr run . --run-config "num-server-rounds=10"

Usage (deployment mode — three separate processes):
    flwr superlink &
    flwr supernode --partition-id 0 &
    flwr supernode --partition-id 1 &
    flwr run .
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
from utils.wandb_utils import METRICS_GLOSSARY_NOTES

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
    log(INFO, ("server run_config:", dict(context.run_config)))

    num_rounds: int = context.run_config["num-server-rounds"]
    embedding_dim: int = context.run_config["embedding-dim"]
    lr: float = context.run_config["learning-rate"]

    top_kwargs = {
        "embedding_dim": embedding_dim,
        "dropout_rate": float(context.run_config.get("dropout-top", 0.2)),
        "hidden_dim": int(context.run_config.get("hidden-dim-top", 64)),
    }
    top_model = TopModel(**top_kwargs)
    optimizer = torch.optim.Adam(top_model.parameters(), lr=lr)
    top_model.train()

    losses: list[tuple[int, float]] = []
    
    track = bool(context.run_config.get("track-wandb", True))
    
    pos_weight_cfg = float(context.run_config.get("pos-weight", -1.0))

    dp = _server_dp_config(context)
    train_model, train_optimizer = top_model, optimizer
    wrapped = False

    if track:
        tags = ["vfl","server"]
        sweep_tag = context.run_config.get("wandb-sweep-tag","")
        if sweep_tag:
            tags.append(sweep_tag)
        wandb.init(
            entity=context.run_config.get("wandb-entity", "jayquery"),
            project=context.run_config.get("wandb-project", "jayquery-vfl"),
            name=context.run_config.get("wandb-run-name", "") or "server",
            group=context.run_config.get("wandb-group", "") or None,
            tags=tags,
            notes=METRICS_GLOSSARY_NOTES,
            config={
                "num-server-rounds": num_rounds,
                "learning-rate": lr,
                "embedding-dim": embedding_dim,
                "dropout-top": top_kwargs["dropout_rate"],
                "hidden-dim-top": top_kwargs["hidden_dim"],
                "server-dp-enabled": dp["enabled"],
                "pos-weight": pos_weight_cfg,
            },
            resume="never",
        )

    # run-tag (e.g. a sweep trial id) isolates this run's checkpoints + meta.json.
    eps_tag = str(context.run_config.get("run-tag", "")) or (
        f"eps_{dp['epsilon']}" if dp["enabled"] else "eps_inf"
    )

    sample_rate = float(context.run_config.get("batch-sample-rate", 1.0))
    seed = int(context.run_config.get("seed", 42))
    combined_target = float(context.run_config.get("combined-target-epsilon", 3.0))
    server_tracker: PrivacyBudgetTracker | None = None
    client_trackers: dict[int, PrivacyBudgetTracker] = {}
    stopped_early = False

    eval_every = int(context.run_config.get("eval-every", 5))
    patience = int(context.run_config.get("early-stop-patience", 8))
    min_delta = float(context.run_config.get("early-stop-min-delta", 1e-4))
    best_val_auc = float("-inf")
    best_round = 0
    best_top_state: dict | None = None
    evals_since_improve = 0

    base_lr = lr
    warmup_rounds = int(context.run_config.get("lr-warmup-rounds", 0))
    warmup_start = float(context.run_config.get("lr-warmup-start-factor", 0.1))
    plateau_factor = float(context.run_config.get("lr-plateau-factor", 0.5))
    plateau_patience = int(context.run_config.get("lr-plateau-patience", 3))  # eval events
    lr_min = float(context.run_config.get("lr-min", 1e-6))
    grad_clip = float(context.run_config.get("grad-clip-norm", 0.0))  # 0 disables
    plateau_scale = 1.0
    current_lr = base_lr
    
    loss_ema = None
    loss_ema_alpha = float(context.run_config.get("loss-ema-alpha", 0.1))

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

        if round_num == 1:
            if pos_weight_cfg < 0:
                top_model.set_pos_weight(n_neg=rd.n_neg, n_pos=rd.n_pos)
            elif pos_weight_cfg == 0:
                top_model.set_pos_weight(n_neg=1, n_pos=1)   # weight = 1.0, i.e. off
            else:
                top_model.set_pos_weight(n_neg=int(pos_weight_cfg * 1000), n_pos=1000)
            log(INFO, "pos_weight = %.4f (mode=%s, n_neg=%d, n_pos=%d)",
                float(top_model.pos_weight), pos_weight_cfg, rd.n_neg, rd.n_pos)

        if dp["enabled"] and not wrapped:
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

        current_lr = max(lr_min, base_lr * _warmup_factor(round_num, warmup_rounds, warmup_start)
                         * plateau_scale)
        for g in train_optimizer.param_groups:
            g["lr"] = current_lr

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
            
        if grad_clip > 0 and not dp["enabled"]:
            torch.nn.utils.clip_grad_norm_(top_model.parameters(), grad_clip)
        train_optimizer.step()
        grad_a, grad_b = emb_a.grad.clone(), emb_b.grad.clone()

        log(INFO, "Round %d — loss: %.4f", round_num, float(loss))
        losses.append((round_num, float(loss)))
        loss_ema = float(loss) if loss_ema is None else (
            loss_ema_alpha * float(loss) + (1.0 - loss_ema_alpha) * loss_ema
        )

        # Train-batch prediction health, from the logits we already have.
        # pred_std ~ 0 or pred_pos_rate pinned at 0/1 = one-class collapse
        # (see docs/metrics_glossary.md) — visible EVERY round, unlike val/.
        with torch.no_grad():
            batch_probs = torch.sigmoid(logits.detach())
        train_health = {
            "train/pred_std": float(batch_probs.std()),
            "train/pred_pos_rate": float((batch_probs >= 0.5).float().mean()),
            "train/label_pos_rate": float(labels.float().mean()),
        }

        if server_tracker is not None:
            server_tracker.step()
        eps_metrics = _epsilon_metrics(server_tracker, client_trackers, dp["delta"])
        for k, v in eps_metrics.items():
            log(INFO, "Round %d — %s: %.4f", round_num, k, v)

        if track:
            wandb.log({"train/loss": float(loss), "train/loss_ema": loss_ema,
                       "train/lr": current_lr, **train_health, **eps_metrics}, step=round_num)

        save_best_now = False
        if round_num % eval_every == 0:
            src = train_model._module if dp["enabled"] else top_model
            eval_model = TopModel(**top_kwargs)
            eval_model.load_state_dict(src.state_dict())
            val_metrics = _run_eval(grid, node_ids, eval_model, split="val")
            val_auc = float(val_metrics["auc_roc"])
            if track:
                # Full suite (recall/precision/f1/brier/confusion counts +
                # pred_std/pred_collapsed health flags) as per-round curves,
                # so mis-prediction modes show up mid-run — not just at the
                # single final test point. "val/auc" key kept for continuity.
                wandb.log(
                    {"val/auc": val_auc,
                     **{f"val/{k}": float(v) for k, v in val_metrics.items()
                        if isinstance(v, int | float)}},
                    step=round_num,
                )
            log(INFO, "Round %d — val/auc: %.4f (best %.4f @ round %d)",
                round_num, val_auc, best_val_auc, best_round)
            if val_metrics["pred_collapsed"]:
                log(WARNING,
                    "Round %d — val predictions COLLAPSED to one class "
                    "(pred_std=%.4f, pred_pos_rate=%.4f, base_rate=%.4f)",
                    round_num, val_metrics["pred_std"], val_metrics["pred_pos_rate"],
                    val_metrics["base_rate"])

            if val_auc > best_val_auc + min_delta:
                best_val_auc, best_round = val_auc, round_num
                best_top_state = {
                    k: v.detach().cpu().clone() for k, v in src.state_dict().items()
                }
                evals_since_improve = 0
                save_best_now = True
            else:
                evals_since_improve += 1
                if (round_num > warmup_rounds and plateau_patience > 0
                        and evals_since_improve % plateau_patience == 0):
                    plateau_scale *= plateau_factor
                    log(INFO, "LR plateau reduce: scale=%.4g (next lr ~ %.2e)",
                        plateau_scale, max(lr_min, base_lr * plateau_scale))

        _send_gradients(
            grid=grid,
            node_pos_map=node_pos_map,
            grad_a=grad_a,
            grad_b=grad_b,
            round_num=round_num,
            num_rounds=num_rounds,
            eps_tag=eps_tag,
            save_best=save_best_now,
            lr=current_lr,
        )

        if evals_since_improve >= patience:
            log(INFO,
                "Early stop: no val/auc gain for %d evals (best %.4f @ round %d, now round %d)",
                patience, best_val_auc, best_round, round_num)
            stopped_early = True
            break

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

    if best_top_state is not None:
        top_sd = best_top_state
        log(INFO, "Checkpointing BEST top model: val/auc %.4f @ round %d", best_val_auc, best_round)
    else:
        top_sd = (train_model._module if dp["enabled"] else top_model).state_dict()
        log(WARNING, "No eval happened — checkpointing LAST top model instead of best.")
    torch.save(top_sd, ckpt_dir / "top.pt")

    final_eval = TopModel(**top_kwargs)
    final_eval.load_state_dict(top_sd)
    test_metrics = _run_eval(
        grid, node_ids, final_eval, split="test",
        extra_cfg={"restore-best": True, "eps-tag": eps_tag},
    )

    meta = {
        "epsilon": dp["epsilon"] if dp["enabled"] else None,
        "delta": dp["delta"],
        "embedding_dim": embedding_dim,
        "num_rounds": num_rounds,
        "server_dp_enabled": dp["enabled"],
        "batch_sample_rate": sample_rate,
        "combined_target_epsilon": combined_target,
        "stopped_early": stopped_early,
        "best_val_auc": None if best_val_auc == float("-inf") else best_val_auc,
        "best_round": best_round,
        "epsilon_spent": _epsilon_metrics(server_tracker, client_trackers, dp["delta"]),
        "metrics": test_metrics,
    }
    (ckpt_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    log(INFO, "")
    log(INFO, "=== Training Complete ===")
    for r, loss_val in losses:
        log(INFO, "  Round %3d — loss: %.4f", r, loss_val)
    if track:
        summary = {f"test/{k}": v for k, v in test_metrics.items() if isinstance(v, int | float)}
        if best_val_auc != float("-inf"):
            summary["val/auc_best"] = best_val_auc
        wandb.log(summary)
        wandb.finish()

    log(INFO, "Best val/auc: %.4f @ round %d", best_val_auc, best_round)
    log(INFO, "Final test metrics (best state): %s", test_metrics)

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

def _warmup_factor(round_num: int, warmup_rounds: int, start_factor: float) -> float:
    """Linear LR warmup multiplier: ramps start_factor -> 1.0 over warmup_rounds.

    Returns 1.0 once past warmup (or if warmup is disabled). Warming the LR from
    near-zero avoids the early wrong-direction steps that dig the U-shaped dip.
    """
    if warmup_rounds <= 0 or round_num >= warmup_rounds:
        return 1.0
    return start_factor + (1.0 - start_factor) * (round_num / warmup_rounds)


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
    n_neg: int = 0
    n_pos: int = 0
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
        if pos == _SILO_A_POS and "n-pos" in cfg:
            rd.n_pos, rd.n_neg = int(cfg["n-pos"]), int(cfg["n-neg"])
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
    save_best: bool = False,
    lr: float | None = None,
) -> None:
    """Push per-silo embedding gradients back to each SuperNode.

    Uses push_messages (no reply expected); each SuperNode's apply_gradients
    handler will receive the gradient and apply it in the next round.

    Args:
        grid: Flower Grid.
        node_pos_map: Mapping from node_id → partition-id.
        grad_a: Gradient w.r.t. Silo A embedding, shape (N, embedding_dim).
        grad_b: Gradient w.r.t. Silo B embedding, shape (N, embedding_dim).
        save_best: If True, the client should snapshot its CURRENT bottom model
            (the state this round's val/auc was measured on) as best-so-far,
            BEFORE applying the gradient in this message.
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
                        "num-server-rounds": num_rounds,
                        "eps-tag": eps_tag,
                        "save-best": save_best,
                        "lr": float(lr) if lr is not None else -1.0,
                    }),
                }),
                message_type="train.apply_gradients",
                dst_node_id=node_id,
            )
        )

    log(INFO, "Sending gradients to %d nodes...", len(messages))
    grid.push_messages(messages)

def _run_eval(
    grid: Grid,
    node_ids: list[int],
    top_model: TopModel,
    split: str,
    extra_cfg: dict | None = None,
) -> dict:
    """Distributed eval round: both silos forward `split` embeddings, server scores.

    Args:
        extra_cfg: Optional extra config merged into the query (e.g.
            {"restore-best": True, "eps-tag": ...} to have clients restore and
            checkpoint their best bottom model in this same round-trip).
    """
    cfg = {"split": split}
    if extra_cfg:
        cfg.update(extra_cfg)
    messages = [
        Message(
            content=RecordDict({"config": ConfigRecord(cfg)}),
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
    log(INFO, "probs: min=%.4f max=%.4f mean=%.4f | frac>0.5=%.4f | labels mean=%.4f",
        probs.min(), probs.max(), probs.mean(), (probs > 0.5).float().mean(), labels.float().mean())
    
    return compute_metrics(labels.numpy(), probs.numpy())