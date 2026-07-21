"""Top Model — aggregates embeddings from both silos and produces dropout-risk prediction.

Owner: Alex Pellet
Architecture: Input(64) → Linear(64) → ReLU → Dropout(0.2) → Linear(32) → ReLU → Linear(1)
The loss gradient is sliced into (grad_emb_a, grad_emb_b) and returned to the silos.
"""

import torch
import torch.nn as nn


class TopModel(nn.Module):
    """Central aggregation model that takes concatenated embeddings and predicts dropout risk.

    Args:
        embedding_dim: Size of each silo's embedding vector. Default 32.
            The model input will be 2 * embedding_dim.
        dropout_rate: Dropout probability in the hidden layer. Default 0.2.
        hidden_dim: First hidden layer width; second layer is hidden_dim // 2. Default 64.
    """

    def __init__(
        self, embedding_dim: int = 32, dropout_rate: float = 0.2, hidden_dim: int = 64
    ) -> None:
        super().__init__()
        input_dim = embedding_dim * 2  # concatenate emb_a and emb_b

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            # NOTE: no Sigmoid — BCEWithLogitsLoss applies it internally
        )

        # Class-imbalance weight (set externally before training)
        self.register_buffer("pos_weight", torch.tensor(1.0))

    def forward(self, emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
        """Forward pass through the aggregation network.

        Args:
            emb_a: Embedding from Silo A, shape (batch_size, embedding_dim).
            emb_b: Embedding from Silo B, shape (batch_size, embedding_dim).

        Returns:
            Logit tensor of shape (batch_size, 1).
        """
        combined = torch.cat([emb_a, emb_b], dim=1)
        return self.net(combined)
    
    def loss_fn(
        self, 
        logits: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute BCE-with-logits loss using the current pos_weight.

        Exposed so the server loop can run the forward/backward itself (required
        for Opacus DP-SGD, which must own the backward)
        """
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.to(logits.device))
        return loss_fn(logits, labels.float())

    def set_pos_weight(self, n_neg: int, n_pos: int) -> None:
        """Set class imbalance weight. Call before training.

        Args:
            n_neg: Number of negative (not-at-risk) samples in training set.
            n_pos: Number of positive (at-risk) samples in training set.
        """
        weight = torch.tensor(n_neg / max(n_pos, 1), dtype=torch.float32)
        self.pos_weight = weight
