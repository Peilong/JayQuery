"""Bottom Model B — processes VLE interaction + assessment features from Silo B.

Owner: Alex Pellet
Architecture: Input(26) → Linear(128) → BN → ReLU →
              Linear(64) → BN → ReLU → Linear(64) [+skip projection] → Linear(32)

Input features (26 total):
> USS -> Unbounded Standard Scalar
 - 0    | num_assessments: Row count for id_student in studentAssessment (USS)
 - 1    | participation_rate: Sum of assessment weights taken / total class weight (0-1)
 - 2    | weighted_score_pct: Sum of weighted scores / total possible weighted score (0-1)
 - 3    | start_date: Day offset of first VLE interaction relative to module start (USS)
 - 4    | end_date: Day offset of last VLE interaction relative to module start (USS)
 - 5    | active_days: Count of distinct days with VLE activity (USS)
 - 6-25 | vle_click_[activity_type]: Sum of sum_click per activity_type (USS)
"""

import torch
import torch.nn as nn


class BottomModelB(nn.Module):
    """MLP with residual skip connection for VLE + academic features (Silo B).

    The skip connection helps preserve low-level VLE signals that might otherwise
    be washed out by the larger hidden layer.

    Args:
        input_dim: Number of input features. Default 22.
        embedding_dim: Output embedding dimension. Default 32.
        dropout_rate: Dropout probability. Default 0.3.
        use_cnn: If True, replace hidden MLP with 1D-Conv over week-binned VLE
                 time series (experimental, Week 6). Default False.
    """

    def __init__(
        self,
        input_dim: int = 26,
        embedding_dim: int = 32,
        dropout_rate: float = 0.3,
        use_cnn: bool = False,
    ) -> None:
        super().__init__()
        self.use_cnn = use_cnn

        if use_cnn:
            # TODO (Alex, Week 6): Implement 1D-CNN variant over week-binned VLE
            raise NotImplementedError("1D-CNN variant not yet implemented — set use_cnn=False")

        # Layer 1: expand
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        # Layer 2: compress
        self.layer2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        # Skip projection: 128 → 64 to match layer2 output
        self.skip_proj = nn.Linear(128, 64, bias=False)

        # Final embedding layer
        self.dropout = nn.Dropout(dropout_rate)
        self.embed = nn.Linear(64, embedding_dim)
        # NOTE: NO activation — raw embedding for cut layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual skip connection.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Embedding tensor of shape (batch_size, embedding_dim).
        """
        h1 = self.layer1(x)  # (B, 128)
        h2 = self.layer2(h1)  # (B, 64)
        h2 = h2 + self.skip_proj(h1)  # residual add
        h2 = self.dropout(h2)
        return self.embed(h2)  # (B, embedding_dim)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Compute cut-layer embedding (alias for forward).

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Embedding tensor of shape (batch_size, embedding_dim).
        """
        return self.forward(x)
