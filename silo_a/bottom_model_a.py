"""Bottom Model A — processes demographic features from Silo A (Financial Aid).

Owner: Alex Pellet
Architecture: Input(12) → Linear(64) → BN → ReLU → Dropout(0.3) →
              Linear(64) → BN → ReLU → Linear(32)

Input features (18 total from studentInfo.csv):
 - 0    | imd_band: Index of Multiple Deprivation band; higher = less deprived 
 - 1    | highest_education: 0-4, 0=No formal quals, 4=Post-grad 
 - 2    | age_band: 0-2, 0=0-35, 1=35-55, 2=55+
 - 3    | gender: 0/1, 0=M & 1=F 
 - 4    | disability: 0/1, 0=N & 1=Y 
 - 5    | studied_credits: Credits registered for in this module
 - 6-17 | region: OneHotEncoded 0/1, 13 UK regions & reference dropped
  Total: 1 + 1 + 1 + 1 + 1 + 1 + 13 - 1 (drop one hot ref) + 1 = 18
"""

import torch
import torch.nn as nn


class BottomModelA(nn.Module):
    """MLP with batch normalization for demographic (Silo A) features.

    Args:
        input_dim: Number of input features. Default 18.
        embedding_dim: Output embedding dimension. Default 32.
        dropout_rate: Dropout probability. Default 0.3.
    """

    def __init__(
        self,
        input_dim: int = 18,
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
