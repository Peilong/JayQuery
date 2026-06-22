"""XGBoost baseline trained on merged (centralized) data.

Owner: Alex Pellet
Typically the strongest centralized baseline on tabular data.
"""

from typing import Any

import numpy as np
import xgboost as xgb

from baselines.eval_utils import compute_metrics, print_metrics_table


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
    """Train XGBoost classifier and return test metrics.

    Args:
        X_train/val/test: Feature arrays of shape (N, # Features).
        y_train/val/test: Label arrays of shape (N,).
        config: Optional hyperparameter overrides.

    Returns:
        Test metrics dict from compute_metrics().
    """
    cfg = {
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42,
        **(config or {}),
    }

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    model = xgb.XGBClassifier(
        n_estimators=cfg["n_estimators"],
        max_depth=cfg["max_depth"],
        learning_rate=cfg["learning_rate"],
        subsample=cfg["subsample"],
        colsample_bytree=cfg["colsample_bytree"],
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="auc",
        random_state=cfg["seed"],
        verbosity=0,
        early_stopping_rounds=20,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    if run is not None:
        from utils.wandb_utils import log_extra

        for tree_idx, auc in enumerate(model.evals_result()["validation_0"]["auc"]):
            log_extra({"val/auc_per_tree": auc}, run=run, step=tree_idx)

    test_prob = model.predict_proba(X_test)[:, 1]
    test_metrics = compute_metrics(y_test, test_prob)
    print_metrics_table(test_metrics, "XGBoost Centralized (test set)")
    return test_metrics
