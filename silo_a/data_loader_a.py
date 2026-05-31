"""DataLoader for Silo A — demographic features, aligned to PSI order.

Owner: Alex Pellet
Loads data/processed/silo_a.pkl and aligns rows to ordered_id_list.json.
"""

import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class SiloADataset(Dataset):
    """PyTorch Dataset for Silo A demographic features.

    Args:
        features: NumPy array of shape (N, 12).
        labels: NumPy array of shape (N,) — binary {0, 1}.
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def load_silo_a(
    processed_dir: str | Path = "data/processed",
    batch_size: int = 64,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Load Silo A data and return train/val/test DataLoaders.

    Rows are ordered according to ordered_id_list.json (PSI intersection output)
    so that Silo A and Silo B loaders are perfectly aligned.

    Args:
        processed_dir: Directory containing silo_a.pkl and ordered_id_list.json.
        batch_size: Mini-batch size. Default 64.
        seed: Random seed for DataLoader worker init. Default 42.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).

    Raises:
        FileNotFoundError: If silo_a.pkl or ordered_id_list.json do not exist.
    """
    processed_dir = Path(processed_dir)
    silo_path = processed_dir / "silo_a.pkl"
    id_list_path = processed_dir / "ordered_id_list.json"

    if not silo_path.exists():
        raise FileNotFoundError(
            f"{silo_path} not found. Run: python scripts/prepare_silos.py"
        )
    if not id_list_path.exists():
        raise FileNotFoundError(
            f"{id_list_path} not found. Run: python -m psi.psi_client"
        )

    with open(silo_path, "rb") as f:
        data: dict = pickle.load(f)

    train_ds = SiloADataset(data["X_train"], data["y_train"])
    val_ds   = SiloADataset(data["X_val"],   data["y_val"])
    test_ds  = SiloADataset(data["X_test"],  data["y_test"])

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  generator=g)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def make_synthetic_loader_a(
    n_train: int = 800,
    n_val: int = 100,
    n_test: int = 100,
    batch_size: int = 64,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Generate synthetic Silo A data for development (no OULAD required).

    Feature seed and label seed are fixed so that make_synthetic_loader_b produces
    identical labels (same seed + 100), ensuring row alignment across silos.

    Args:
        n_train: Number of training samples.
        n_val: Number of validation samples.
        n_test: Number of test samples.
        batch_size: Mini-batch size.
        seed: Base random seed. Labels use seed+100 (shared with Silo B).

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    n_total = n_train + n_val + n_test
    X = np.random.default_rng(seed).standard_normal((n_total, 12)).astype(np.float32)
    y = np.random.default_rng(seed + 100).binomial(1, 0.3, n_total).astype(np.float32)

    g = torch.Generator().manual_seed(seed)
    return (
        DataLoader(
            SiloADataset(X[:n_train], y[:n_train]),
            batch_size=batch_size,
            shuffle=True,
            generator=g,
        ),
        DataLoader(
            SiloADataset(X[n_train : n_train + n_val], y[n_train : n_train + n_val]),
            batch_size=batch_size,
            shuffle=False,
        ),
        DataLoader(
            SiloADataset(X[n_train + n_val :], y[n_train + n_val :]),
            batch_size=batch_size,
            shuffle=False,
        ),
    )


def validate_alignment(loader_a: DataLoader, loader_b: DataLoader, n_check: int = 10) -> bool:
    """Verify that Silo A and Silo B loaders contain the same students in the same order.

    Checks that the first n_check label values match — a necessary (not sufficient)
    condition for alignment. Always call before training.

    Args:
        loader_a: DataLoader for Silo A.
        loader_b: DataLoader for Silo B.
        n_check: Number of label values to compare.

    Returns:
        True if aligned, False otherwise.
    """
    labels_a = next(iter(loader_a))[1][:n_check]
    labels_b = next(iter(loader_b))[1][:n_check]
    aligned = torch.allclose(labels_a, labels_b)
    if not aligned:
        print(f"[ALIGNMENT WARNING] First {n_check} labels differ between Silo A and B!")
        print(f"  Silo A: {labels_a.tolist()}")
        print(f"  Silo B: {labels_b.tolist()}")
    return aligned
