"""Full MLP baseline trained on merged (centralized) data.

Owner: Alex Pellet
Architecture mirrors the TopModel but with raw 34-dim input.
This sets the performance ceiling for VFL comparisons.

Usage:
    python -m baselines.centralized_mlp \
        --config experiments/configs/exp01_centralized_baselines.yaml
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from baselines.eval_utils import compute_metrics, print_metrics_table
from utils.reproducibility import seed_everything


class CentralizedMLP(nn.Module):
    """Full MLP trained on all 45 features (Silo A + Silo B concatenated).

    Args:
        input_dim: Total number of features (18 + 27 = 45).
        dropout_rate: Dropout probability. Default 0.2.
    """

    def __init__(self, input_dim: int = 45, dropout_rate: float = 0.2) -> None:
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
    run = None,
) -> dict[str, float]:
    """Train CentralizedMLP and return test metrics.

    Args:
        X_train/val/test: Feature arrays of shape (N, 45).
        y_train/val/test: Label arrays of shape (N,).
        config: Optional hyperparameter dict (overrides defaults).

    Returns:
        Test metrics dict from compute_metrics().
    """
    cfg = {
        "lr": 1e-3, "epochs": 50, "batch_size": 64,
        "dropout_rate": 0.2, "seed": 42,
        **(config or {}),
    }
    seed_everything(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CentralizedMLP(input_dim=X_train.shape[1], dropout_rate=cfg["dropout_rate"]).to(device)

    pos_weight = torch.tensor([(y_train == 0).sum() / max((y_train == 1).sum(), 1)]).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, mode="max")

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                              torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)

    best_val_auc, best_state = 0.0, None
    for epoch in range(cfg["epochs"]):
        model.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(X_b).squeeze(1), y_b)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_prob = torch.sigmoid(
                model(torch.tensor(X_val, dtype=torch.float32).to(device))
            ).squeeze(1).cpu().numpy()
        val_metrics = compute_metrics(y_val, val_prob)
        scheduler.step(val_metrics["auc_roc"])
        
        if run is not None:
            from utils.wandb_utils import log_extra
            log_extra({
                "val/auc_roc": val_metrics["auc_roc"],
                "val/f1":      val_metrics["f1_binary"],
            }, run=run, step=epoch)
        
        if val_metrics["auc_roc"] > best_val_auc:
            best_val_auc = val_metrics["auc_roc"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Test evaluation
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_prob = torch.sigmoid(
            model(torch.tensor(X_test, dtype=torch.float32).to(device))
        ).squeeze(1).cpu().numpy()

    test_metrics = compute_metrics(y_test, test_prob)
    print_metrics_table(test_metrics, "Centralized MLP (test set)")
    return test_metrics
