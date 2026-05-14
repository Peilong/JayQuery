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
    """

    def __init__(self, embedding_dim: int = 32, dropout_rate: float = 0.2) -> None:
        super().__init__()
        input_dim = embedding_dim * 2  # concatenate emb_a and emb_b

        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            # NOTE: no Sigmoid — BCEWithLogitsLoss applies it internally
        )

        # Class-imbalance weight (set externally before training)
        self.register_buffer("pos_weight", torch.tensor(1.0))
        self._loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

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

    def compute_loss_and_gradients(
        self,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run forward + backward and return loss + per-silo embedding gradients.

        This is the KEY interface called each Flower round. The returned gradients
        are sent back to silo clients so they can update their Bottom Models.

        Args:
            emb_a: Detached embedding from Silo A, shape (batch_size, embedding_dim).
                   Will have requires_grad=True set internally.
            emb_b: Detached embedding from Silo B, shape (batch_size, embedding_dim).
            labels: Binary float labels (0=not-at-risk, 1=at-risk), shape (batch_size,).

        Returns:
            Tuple of (loss_scalar, grad_emb_a, grad_emb_b) where gradients have the
            same shape as their respective embedding inputs.
        """
        # Enable gradient tracking on incoming embeddings
        emb_a = emb_a.detach().requires_grad_(True)
        emb_b = emb_b.detach().requires_grad_(True)

        # Forward pass
        logits = self.forward(emb_a, emb_b).squeeze(1)

        # Update pos_weight in loss (in case it changed externally)
        self._loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.to(logits.device))
        loss = self._loss_fn(logits, labels.float())

        # Backward pass — gradients accumulate on emb_a and emb_b
        loss.backward()

        assert emb_a.grad is not None, "grad_emb_a is None — check requires_grad"
        assert emb_b.grad is not None, "grad_emb_b is None — check requires_grad"

        return loss.detach(), emb_a.grad.clone(), emb_b.grad.clone()

    def set_pos_weight(self, n_neg: int, n_pos: int) -> None:
        """Set class imbalance weight. Call before training.

        Args:
            n_neg: Number of negative (not-at-risk) samples in training set.
            n_pos: Number of positive (at-risk) samples in training set.
        """
        weight = torch.tensor(n_neg / max(n_pos, 1), dtype=torch.float32)
        self.pos_weight = weight
        self._loss_fn = nn.BCEWithLogitsLoss(pos_weight=weight)
