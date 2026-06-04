"""Logistic Regression baseline trained on centralized data.

Owner: Alex Pellet
Lower bound of performance.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from baselines.eval_utils import compute_metrics, print_metrics_table


def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: dict | None = None,
) -> dict[str, float]:
    """Train Logistic Regression classifier and return test metrics.

    Args:
        X_train/val/test: Feature arrays of shape (N, # Features).
        y_train/val/test: Label arrays of shape (N,).
        config: Optional hyperparameter overrides.
        run: A wandb run instance

    Returns:
        Test metrics dict from compute_metrics().
    """
    cfg = {
        "C": 1,
        "penalty": "l2",
        "solver": "lbfgs",
        "maxiter": 1000,
        "seed": 42,
        **(config or {}),
    }
    
    model = LogisticRegression(
        random_state=cfg["seed"],
        C=cfg["C"],
        penalty=cfg["penalty"],
        solver=cfg["solver"],
        max_iter=cfg["maxiter"],
        class_weight="balanced",
    )

    model.fit(
        X_train,
        y_train,
    )

    test_prob = model.predict_proba(X_test)[:, 1]
    test_metrics = compute_metrics(y_test, test_prob)
    print_metrics_table(test_metrics, "LogisticRegression Centralized (test set)")
    return test_metrics
