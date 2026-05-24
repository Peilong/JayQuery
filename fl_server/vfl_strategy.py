"""
----- MARKED FOR DELETION 22-05-26 : AMXP
-- Replaced with the newer API of flwr
"""

"""Flower Strategy for Vertical Federated Learning.

Owner: Alex Pellet
Custom Strategy that orchestrates the VFL training loop:
  1. Both silo clients compute forward pass → send (embedding, label) to server.
  2. Server Top Model computes loss → backpropagates → returns (grad_emb_a, grad_emb_b).
  3. Both silo clients apply received gradients to their Bottom Models.
"""

from logging import WARNING
from typing import Optional

import flwr as fl
from flwr.common import FitRes, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy

# TODO (Alex, Week 5): Implement VFLStrategy class extending fl.server.strategy.Strategy
# See CLAUDE.md for the required interface and protocol_spec.md for message definitions.


class VFLStrategy(fl.server.strategy.FedAvg):
    """Placeholder VFL strategy — replace FedAvg parent with custom logic in Week 5.

    The full implementation should:
    - On fit_round: collect embeddings from BOTH silos simultaneously
    - Pass embeddings to TopModel.compute_loss_and_gradients()
    - Return gradient tensors to each silo as Parameters
    - Track val AUC and epsilon budget per round
    """

    def __init__(self, top_model_state_dict: Optional[dict] = None) -> None:
        super().__init__()
        # TODO (Alex, Week 4): Initialize TopModel here
        self.top_model_state = top_model_state_dict
        self.round_metrics: list[dict[str, float]] = []

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """TODO (Alex, Week 5): Implement VFL aggregation logic."""
        # Placeholder — calls FedAvg aggregation for scaffolding purposes
        return super().aggregate_fit(server_round, results, failures)
