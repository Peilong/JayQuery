"""Flower Client for Silo B — Academic Affairs / VLE.

--MARKED FOR DELETION 23/06/2026--
-- This file is incorporated into the fl_client system, and serves no purpose outside of it. --

Owner: Alex Pellet
Mirrors client_a.py. TODO (Week 5, Alex): implement fit() and evaluate().
TODO (Week 6, Wyatt): insert dp_wrapper at the marked DP HOOK location.
"""

from collections import OrderedDict

import flwr as fl
import torch
from flwr.common import NDArrays, Scalar

from silo_b.bottom_model_b import BottomModelB
from silo_b.data_loader_b import load_silo_b
from utils.reproducibility import seed_everything

SILO_B_FEATURES = 26

class SiloBClient(fl.client.NumPyClient):
    """Flower client representing the Academic Affairs silo."""

    def __init__(self, config: dict, device: str = "cpu") -> None:
        seed_everything(config.get("seed", 42))
        self.device = torch.device(device)
        self.config = config

        self.model = BottomModelB(
            input_dim=config.get("input_dim_b", SILO_B_FEATURES),
            embedding_dim=config.get("embedding_dim", 32),
            dropout_rate=config.get("dropout_rate", 0.3),
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.get("learning_rate", 1e-3),
        )
        # ── WYATT DP HOOK (Week 6) ──────────────────────────────────────────────
        # Same pattern as client_a.py. Replace self.optimizer with Opacus version.
        # Reference: client_a.py lines 57-65 for the exact call pattern.
        # ───────────────────────────────────────────────────────────────────────

        self.train_loader, self.val_loader, self.test_loader = load_silo_b(
            batch_size=config.get("batch_size", 64),
        )

    def get_parameters(self, config: dict) -> NDArrays:
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters: NDArrays) -> None:
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: NDArrays, config: dict) -> tuple[NDArrays, int, dict[str, Scalar]]:
        """TODO (Alex, Week 5): Implement VFL fit round (mirrors client_a.py)."""
        self.set_parameters(parameters)
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters: NDArrays, config: dict) -> tuple[float, int, dict[str, Scalar]]:
        """TODO (Alex, Week 5): Implement evaluation."""
        self.set_parameters(parameters)
        return 0.0, len(self.val_loader.dataset), {"auc": 0.0}
