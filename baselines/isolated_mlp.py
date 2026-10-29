"""MLP baselines trained on isolated silo data.

Owner: Alex Pellet
Trains on a single silo's features without access to the other silo.
Establishes the lower bound — quantifies the benefit of VFL collaboration.
"""
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from baselines.eval_utils import compute_metrics, print_metrics_table
from baselines.training_plan import resolve_plan
from utils.reproducibility import seed_everything


class IsolatedMLP(nn.Module):
    """MLP trained on a single silo's features in isolation.

    Args:
        input_dim: Number of features for this silo (18 for Silo A, 27 for Silo B).
        dropout_rate: Dropout probability. Default 0.2.
    """

    def __init__(self, input_dim: int, dropout_rate: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: dict | None = None,
    run: Any | None = None,
) -> dict[str, float]:
    """Train IsolatedMLP on a single silo and return test metrics.

    Args:
        X_train/val/test: Feature arrays — shape (N, 18) for Silo A or (N, 27) for Silo B.
        y_train/val/test: Label arrays of shape (N,).
        config: Optional hyperparameter dict. Supports an additional key:
            silo_label (str): Display name for logging e.g. "Silo A" or "Silo B".
        run: Optional wandb run object for per-epoch logging.

    Returns:
        Test metrics dict from compute_metrics().
    """
    cfg = {
        "lr": 1e-3,
        "epochs": 50,
        "batch_size": 64,
        "dropout_rate": 0.2,
        "seed": 42,
        **(config or {}),
    }
    seed_everything(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IsolatedMLP(input_dim=X_train.shape[1], dropout_rate=cfg["dropout_rate"]).to(device)

    # Budget resolved centrally so this control matches the VFL run's
    # optimization effort. See baselines/training_plan.py.
    plan = resolve_plan(
        cfg,
        n_train=len(y_train),
        n_pos=int((y_train == 1).sum()),
        n_neg=int((y_train == 0).sum()),
    )
    pos_weight = (
        None if plan.pos_weight is None
        else torch.tensor([plan.pos_weight], device=device)
    )
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, mode="max")

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
    )
    # Sampled WITH replacement across steps, mirroring the VFL run's
    # per-round Poisson draw rather than an epoch sweep.
    train_loader = DataLoader(train_ds, batch_size=plan.batch_size, shuffle=True)

    best_val_auc, best_state = 0.0, None
    # One "epoch" here is an EVAL INTERVAL, not a pass over the data: the
    # budget is plan.steps optimizer steps, matching the VFL run's rounds.
    # Evaluating every eval_every steps mirrors the grid's eval-every=5 so
    # model selection gets a comparable number of candidate checkpoints.
    eval_every = max(1, int(cfg.get("eval_every", 5)))
    n_intervals = max(1, plan.steps // eval_every)
    step_iter = iter(train_loader)
    for epoch in range(n_intervals):
        model.train()
        for _ in range(eval_every):
            try:
                X_b, y_b = next(step_iter)
            except StopIteration:            # reshuffle and keep going
                step_iter = iter(train_loader)
                X_b, y_b = next(step_iter)
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(X_b).squeeze(1), y_b)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_prob = (
                torch.sigmoid(model(torch.tensor(X_val, dtype=torch.float32).to(device)))
                .squeeze(1)
                .cpu()
                .numpy()
            )
        val_metrics = compute_metrics(y_val, val_prob)
        scheduler.step(val_metrics["auc_roc"])

        if run is not None:
            from utils.wandb_utils import log_extra

            log_extra(
                {
                    "val/auc_roc": val_metrics["auc_roc"],
                    "val/f1": val_metrics["f1_binary"],
                    "val/precision": val_metrics["precision"],
                    "val/recall": val_metrics["recall"],
                    # Health: pred_std ~ 0 or pred_collapsed = 1 means the model
                    # is outputting one class — see docs/metrics_glossary.md.
                    "val/pred_std": val_metrics["pred_std"],
                    "val/pred_pos_rate": val_metrics["pred_pos_rate"],
                    "val/pred_collapsed": val_metrics["pred_collapsed"],
                },
                run=run,
                step=epoch,
            )

        if val_metrics["auc_roc"] > best_val_auc:
            best_val_auc = val_metrics["auc_roc"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Test evaluation
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_prob = (
            torch.sigmoid(model(torch.tensor(X_test, dtype=torch.float32).to(device)))
            .squeeze(1)
            .cpu()
            .numpy()
        )

    test_metrics = compute_metrics(y_test, test_prob)
    silo_label = config.get("silo_label", "Isolated Silo") if config else "Isolated Silo"
    print_metrics_table(test_metrics, f"{silo_label} MLP (test set)")
    return test_metrics
