"""Baseline Run — for testing and running the baseline models.

Owner: Alex Pellet
All baselines should be saved to the wandb entity 'jayquery' and project 'baseline'

Usage:
    python -m baselines.run_baselines \
        --config experiments/configs/exp01_centralized_baselines.yaml --run all
    python -m baselines.run_baselines \
        --config experiments/configs/exp01_centralized_baselines.yaml --run mlp
    python -m baselines.run_baselines \
        --run all
"""

import argparse

import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from baselines import centralized_lr, centralized_mlp, centralized_xgb, isolated_mlp
from silo_a.data_loader_a import load_silo_a
from silo_b.data_loader_b import load_silo_b
from utils.wandb_utils import run_with_tracking


def inject_run(run: any, train_kwargs: dict) -> None:
    train_kwargs["run"] = run


AVAILABLE_MODELS = {
    "xgboost": {
        "train_fn": centralized_xgb.train_and_evaluate,
        "tags": ["baseline", "xgboost"],
        "on_run_created": inject_run,
    },
    "mlp": {
        "train_fn": centralized_mlp.train_and_evaluate,
        "tags": ["baseline", "mlp"],
        "on_run_created": inject_run,
    },
    "logistic_regression": {
        "train_fn": centralized_lr.train_and_evaluate,
        "tags": ["baseline", "logistic_regression"],
        "on_run_created": None,
    },
    "isolated_silo_a": {
        "train_fn": isolated_mlp.train_and_evaluate,
        "tags": ["baseline", "isolated", "silo_a"],
        "on_run_created": inject_run,
        "data": "silo_a",
    },
    "isolated_silo_b": {
        "train_fn": isolated_mlp.train_and_evaluate,
        "tags": ["baseline", "isolated", "silo_b"],
        "on_run_created": inject_run,
        "data": "silo_b",
    },
}


def parse_args() -> None:
    parser = argparse.ArgumentParser(description="Run centralized baseline models.")
    parser.add_argument(
        "--config", type=str, default="experiments/configs/exp01_centralized_baselines.yaml"
    )
    parser.add_argument(
        "--run",
        type=str,
        default="all",
        choices=["all"] + list(AVAILABLE_MODELS.keys()),
    )
    return parser.parse_args()


def merge_silo_dataloaders(
    loader_a: DataLoader,
    loader_b: DataLoader,
) -> DataLoader:
    X_a = loader_a.dataset.X
    y_a = loader_a.dataset.y
    X_b = loader_b.dataset.X

    assert (
        X_a.shape[0] == X_b.shape[0]
    ), f"Row count mismatch: Silo A has {X_a.shape[0]}, Silo B has {X_b.shape[0]}"

    X_merged = torch.cat([X_a, X_b], dim=1)

    merged_dataset = TensorDataset(X_merged, y_a)
    return DataLoader(
        merged_dataset,
        batch_size=loader_a.batch_size,
        shuffle=False,
    )


def main() -> None:
    # Arg Parsing -------------------------
    args = parse_args()

    with open(args.config) as f:
        exp = yaml.safe_load(f)

    training = exp["training"]
    experiment = exp["experiment"]

    cfg = {
        "lr": float(training["learning_rate"]),
        "epochs": int(training["epochs"]),
        "batch_size": int(training["batch_size"]),
        "dropout_rate": float(training["dropout_rate"]),
        "seed": int(exp["data"]["seed"]),
    }

    to_run = AVAILABLE_MODELS if args.run == "all" else {args.run: AVAILABLE_MODELS[args.run]}

    # Data Prep ------------------------
    try:
        a_train_loader, a_val_loader, a_test_loader = load_silo_a(bypass_psi=True)
        b_train_loader, b_val_loader, b_test_loader = load_silo_b()
    except Exception as e:
        print(f"Exception: {e}")
        return
    else:
        print("Succeeded in loading data silos for baseline run.")

    train_dataloader = merge_silo_dataloaders(a_train_loader, b_train_loader)
    val_dataloader = merge_silo_dataloaders(a_val_loader, b_val_loader)
    test_dataloader = merge_silo_dataloaders(a_test_loader, b_test_loader)

    X_train = torch.cat([X_batch for X_batch, _ in train_dataloader], dim=0).numpy()
    y_train = torch.cat([y_batch for _, y_batch in train_dataloader], dim=0).numpy()

    X_val = torch.cat([X_batch for X_batch, _ in val_dataloader], dim=0).numpy()
    y_val = torch.cat([y_batch for _, y_batch in val_dataloader], dim=0).numpy()

    X_test = torch.cat([X_batch for X_batch, _ in test_dataloader], dim=0).numpy()
    y_test = torch.cat([y_batch for _, y_batch in test_dataloader], dim=0).numpy()

    # isolated datasets
    X_train_a = a_train_loader.dataset.X.numpy()
    y_train_a = a_train_loader.dataset.y.numpy()
    X_val_a = a_val_loader.dataset.X.numpy()
    y_val_a = a_val_loader.dataset.y.numpy()
    X_test_a = a_test_loader.dataset.X.numpy()
    y_test_a = a_test_loader.dataset.y.numpy()

    X_train_b = b_train_loader.dataset.X.numpy()
    X_val_b = b_val_loader.dataset.X.numpy()
    X_test_b = b_test_loader.dataset.X.numpy()

    data_map = {
        "merged": (X_train, y_train, X_val, y_val, X_test, y_test),
        "silo_a": (X_train_a, y_train_a, X_val_a, y_val_a, X_test_a, y_test_a),
        "silo_b": (X_train_b, y_train_a, X_val_b, y_val_a, X_test_b, y_test_a),
    }

    # Model Run -----------------------------
    for model_name, model_cfg in to_run.items():
        print(f"[Baselines] Running {model_name}...")
        data_key = model_cfg.get("data", "merged")
        X_tr, y_tr, X_v, y_v, X_te, y_te = data_map[data_key]
        run_with_tracking(
            train_fn=model_cfg["train_fn"],
            train_kwargs=dict(
                X_train=X_tr,
                y_train=y_tr,
                X_val=X_v,
                y_val=y_v,
                X_test=X_te,
                y_test=y_te,
                config={**cfg, "silo_label": data_key},
            ),
            project="baseline",
            name=f"{experiment['name']}-{model_name}",
            group=f"{experiment['name']}-{model_name}",
            tags=model_cfg["tags"] + [f"week{experiment['week']}"],
            on_run_created=model_cfg["on_run_created"],
        )


if __name__ == "__main__":
    main()
