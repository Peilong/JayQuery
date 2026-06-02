"""Baseline Run — for testing and running the baseline models.

Owner: Alex Pellet
All baselines should be saved to the wandb entity 'jayquery' and project 'baseline'
"""

from baselines import centralized_xgb, centralized_mlp
from utils.wandb_utils import run_with_tracking

from silo_a.data_loader_a import load_silo_a
from silo_b.data_loader_b import load_silo_b

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def merge_silo_dataloaders(
    loader_a: DataLoader,
    loader_b: DataLoader,
) -> DataLoader:
    X_a = loader_a.dataset.X
    y_a = loader_a.dataset.y
    X_b = loader_b.dataset.X

    assert X_a.shape[0] == X_b.shape[0], (
        f"Row count mismatch: Silo A has {X_a.shape[0]}, Silo B has {X_b.shape[0]}"
    )

    X_merged = torch.cat([X_a, X_b], dim=1)

    merged_dataset = TensorDataset(X_merged, y_a)
    return DataLoader(
        merged_dataset,
        batch_size=loader_a.batch_size,
        shuffle=False,
    )
    
def inject_run(run, train_kwargs):
    train_kwargs["run"] = run

def main():
    
    try:
        a_train_loader, a_val_loader, a_test_loader = load_silo_a(bypass_psi=True)
        b_train_loader, b_val_loader, b_test_loader = load_silo_b()
    except Exception as e:
        print(f"Exception: {e}")
        return
    else:
        print("Succeeded in loading data silos for baseline run.")

    train_dataloader = merge_silo_dataloaders(a_train_loader,b_train_loader)
    val_dataloader   = merge_silo_dataloaders(a_val_loader,b_val_loader)
    test_dataloader  = merge_silo_dataloaders(a_test_loader,b_test_loader)
    
    X_train = torch.cat([X_batch for X_batch, _ in train_dataloader], dim=0).numpy()
    y_train = torch.cat([y_batch for _, y_batch in train_dataloader], dim=0).numpy()
    
    X_val = torch.cat([X_batch for X_batch, _ in val_dataloader], dim=0).numpy()
    y_val = torch.cat([y_batch for _, y_batch in val_dataloader], dim=0).numpy()
    
    X_test = torch.cat([X_batch for X_batch, _ in test_dataloader], dim=0).numpy()
    y_test = torch.cat([y_batch for _, y_batch in test_dataloader], dim=0).numpy()
    
    #create a list of the baselines to run, and use run_with_tracking on each.
    #separate using functions for each model, and run them all in main.
    
    xgb_metrics = run_with_tracking(
        train_fn=centralized_xgb.train_and_evaluate,
        train_kwargs=dict(
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test,
            config={"max_depth": 7},
        ),
        project="baseline",
        group="xgb-centralized",
        name="xgb-centralized",
        tags=["baseline", "xgboost"],
        on_run_created=inject_run,
    )
    
    mlp_metrics = run_with_tracking(
        train_fn=centralized_mlp.train_and_evaluate,
        train_kwargs=dict(
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test,
        ),
        project="baseline",
        group="mlp-centralized",
        name="mlp-centralized",
        tags=["baseline", "mlp"],
        on_run_created=inject_run,
    )
    

if __name__ == "__main__":
    main()
