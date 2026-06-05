"""Flower Client for Silo A — Financial Aid / Demographics.

Owner: Alex Pellet
Handles the Silo A side of the VFL training loop:
  - Loads BottomModelA and Silo A DataLoader
  - On fit(): runs forward pass → sends embedding to server → receives gradient → updates model
  - TODO (Week 5, Alex): implement fit() and evaluate() fully
  - TODO (Week 5, Wyatt insert): dp_wrapper wraps self.optimizer at line marked below
"""

from collections import OrderedDict

import flwr as fl
import torch
from flwr.common import NDArrays, Scalar

from silo_a.bottom_model_a import BottomModelA
from silo_a.data_loader_a import load_silo_a
from utils.reproducibility import seed_everything


class SiloAClient(fl.client.NumPyClient):
    """Flower client representing the Financial Aid silo.

    Args:
        config: Dict with training hyperparameters from YAML config.
        device: torch device. Default cpu.
    """

    def __init__(self, config: dict, device: str = "cpu") -> None:
        seed_everything(config.get("seed", 42))
        self.device = torch.device(device)
        self.config = config

        # Model
        self.model = BottomModelA(
            input_dim=config.get("input_dim_a", 12),
            embedding_dim=config.get("embedding_dim", 32),
            dropout_rate=config.get("dropout_rate", 0.3),
        ).to(self.device)

        # Optimizer — NOTE: Wyatt wraps this with Opacus in Week 5
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.get("learning_rate", 1e-3),
        )
        # ── WYATT DP HOOK (Week 5) ──────────────────────────────────────────────
        # Replace self.optimizer with the Opacus-wrapped version here.
        # Example:
        #   from privacy.dp_wrapper import make_private
        #   self.model, self.optimizer, self.train_loader = make_private(
        #       model=self.model, optimizer=self.optimizer,
        #       data_loader=self.train_loader,
        #       target_epsilon=config["epsilon"], target_delta=config["delta"],
        #       max_grad_norm=config["max_grad_norm"], num_epochs=config["num_epochs"],
        #   )
        # ───────────────────────────────────────────────────────────────────────

        # DataLoaders
        self.train_loader, self.val_loader, self.test_loader = load_silo_a(
            batch_size=config.get("batch_size", 64),
        )

    def get_parameters(self, config: dict) -> NDArrays:
        """Return current model parameters as list of NumPy arrays."""
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Load parameters from server into model."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: NDArrays, config: dict) -> tuple[NDArrays, int, dict[str, Scalar]]:
        """TODO (Alex, Week 5): Implement one VFL training round.

        Steps:
        1. self.set_parameters(parameters)
        2. For each batch: compute embedding → send to server → receive gradient → backward
        3. Return updated parameters + metrics
        """
        self.set_parameters(parameters)
        # TODO: implement full VFL fit round
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters: NDArrays, config: dict) -> tuple[float, int, dict[str, Scalar]]:
        """TODO (Alex, Week 5): Evaluate on validation set."""
        self.set_parameters(parameters)
        # TODO: implement evaluation
        return 0.0, len(self.val_loader.dataset), {"auc": 0.0}
