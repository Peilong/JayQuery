"""Bottom Model A — processes demographic features from Silo A (Financial Aid).

Owner: Alex Pellet
Architecture: Input(12) → Linear(64) → BN → ReLU → Dropout(0.3) →
              Linear(64) → BN → ReLU → Linear(32)

Input features (12 total from studentInfo.csv):
  - imd_band: ordinal-encoded deprivation index (0-9)
  - highest_education: ordinal-encoded education level (0-4)
  - age_band: ordinal-encoded age group (0-2)
  - gender: binary (0=M, 1=F)
  - disability: binary (0=N, 1=Y)
  - region: one-hot encoded (8 UK regions)
  - studied_credits: normalized float
  Total: 1 + 1 + 1 + 1 + 1 + 8 - 1 (drop one hot ref) + 1 = 12
"""

import torch
import torch.nn as nn


class BottomModelA(nn.Module):
    """MLP with batch normalization for demographic (Silo A) features.

    Args:
        input_dim: Number of input features. Default 12.
        embedding_dim: Output embedding dimension. Default 32.
        dropout_rate: Dropout probability. Default 0.3.
    """

    def __init__(
        self,
        input_dim: int = 12,
        embedding_dim: int = 32,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim),
            # NOTE: NO activation here — raw embedding for cut layer
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass (alias for get_embedding).

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Embedding tensor of shape (batch_size, embedding_dim).
        """
        return self.encoder(x)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the cut-layer embedding for the given input batch.

        This is the value transmitted to the FL server each round.
        Gradients returned from the server are applied w.r.t. this output.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Embedding tensor of shape (batch_size, embedding_dim).
        """
        return self.forward(x)
