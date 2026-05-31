"""DataLoader for Silo B — VLE and assessment features, aligned to PSI order.

Owner: Alex Pellet
Mirrors data_loader_a.py — loads data/processed/silo_b.pkl.
"""

import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class SiloBDataset(Dataset):
    """PyTorch Dataset for Silo B VLE + assessment features."""

    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def load_silo_b(
    processed_dir: str | Path = "data/processed",
    batch_size: int = 64,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Load Silo B data and return train/val/test DataLoaders.

    Args:
        processed_dir: Directory containing silo_b.pkl.
        batch_size: Mini-batch size. Default 64.
        seed: Random seed. Default 42.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    processed_dir = Path(processed_dir)
    silo_path = processed_dir / "silo_b.pkl"

    if not silo_path.exists():
        raise FileNotFoundError(
            f"{silo_path} not found. Run: python scripts/prepare_silos.py"
        )

    with open(silo_path, "rb") as f:
        data: dict = pickle.load(f)

    train_ds = SiloBDataset(data["X_train"], data["y_train"])
    val_ds   = SiloBDataset(data["X_val"],   data["y_val"])
    test_ds  = SiloBDataset(data["X_test"],  data["y_test"])

    g = torch.Generator()
    g.manual_seed(seed)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False),
    )


def make_synthetic_loader_b(
    n_train: int = 800,
    n_val: int = 100,
    n_test: int = 100,
    batch_size: int = 64,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Generate synthetic Silo B data for development (no OULAD required).

    Uses seed+1 for features (distinct from Silo A) and seed+100 for labels
    (identical to make_synthetic_loader_a), ensuring row-level alignment.

    Args:
        n_train: Number of training samples.
        n_val: Number of validation samples.
        n_test: Number of test samples.
        batch_size: Mini-batch size.
        seed: Base random seed. Labels use seed+100 (shared with Silo A).

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    n_total = n_train + n_val + n_test
    X = np.random.default_rng(seed + 1).standard_normal((n_total, 22)).astype(np.float32)
    y = np.random.default_rng(seed + 100).binomial(1, 0.3, n_total).astype(np.float32)

    g = torch.Generator().manual_seed(seed)
    return (
        DataLoader(
            SiloBDataset(X[:n_train], y[:n_train]),
            batch_size=batch_size,
            shuffle=True,
            generator=g,
        ),
        DataLoader(
            SiloBDataset(X[n_train : n_train + n_val], y[n_train : n_train + n_val]),
            batch_size=batch_size,
            shuffle=False,
        ),
        DataLoader(
            SiloBDataset(X[n_train + n_val :], y[n_train + n_val :]),
            batch_size=batch_size,
            shuffle=False,
        ),
    )
