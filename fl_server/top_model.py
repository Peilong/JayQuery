"""Top Model — aggregates embeddings from both silos and produces dropout-risk prediction.

Owner: Alex Pellet
Architecture: Input(64) → Linear(64) → ReLU → Dropout(0.2) → Linear(32) → ReLU → Linear(1)
The loss gradient is sliced into (grad_emb_a, grad_emb_b) and returned to the silos.
"""

import torch
import torch.nn as nn

from privacy.label_dp import corrected_positive_prob


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

        # Label randomized-response keep probability. 1.0 = labels are clean,
        # so loss_fn stays plain BCE-with-logits. Set via set_label_noise()
        # when the label-DP channel is on. Deliberately a plain attribute, not
        # a buffer: it describes the TRAINING TARGETS, not the weights, and
        # must not ride along in a checkpoint's state_dict.
        self._keep_prob: float = 1.0

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
        """Compute the training loss using the current pos_weight.

        Exposed so the server loop can run the forward/backward itself (required
        for Opacus DP-SGD, which must own the backward).

        With the label-DP channel on, ``labels`` are randomized-response
        outputs rather than ground truth, so the likelihood is corrected for
        the known flip rate (see :func:`privacy.label_dp.corrected_positive_prob`).
        Training on noisy labels as if they were clean would calibrate the
        model to the corrupted base rate instead of the true one.

        Args:
            logits: Raw top-model outputs, shape (batch_size,).
            labels: Targets — true labels when label DP is off, noisy ones when on.

        Returns:
            Scalar loss.
        """
        if self._keep_prob >= 1.0:
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.to(logits.device))
            return loss_fn(logits, labels.float())

        # Weighted BCE against the noise-corrected probability. Written out
        # because F.binary_cross_entropy takes no pos_weight. The correction
        # also bounds prob into [1 - p, p], which keeps the logs finite at
        # saturated logits.
        prob = corrected_positive_prob(logits, self._keep_prob)
        weight = self.pos_weight.to(logits.device)
        target = labels.float()
        per_example = -(
            weight * target * torch.log(prob) + (1.0 - target) * torch.log1p(-prob)
        )
        return per_example.mean()

    def set_label_noise(self, keep_prob: float) -> None:
        """Tell the loss how often the training labels survived randomization.

        Args:
            keep_prob: The randomized-response keep probability p used by the
                label holder, in [0.5, 1]. 1.0 disables the correction.

        Raises:
            ValueError: If keep_prob is outside [0.5, 1].
        """
        if not 0.5 <= keep_prob <= 1.0:
            raise ValueError(f"keep_prob must be in [0.5, 1], got {keep_prob}")
        self._keep_prob = float(keep_prob)

    def set_pos_weight(self, n_neg: int, n_pos: int) -> None:
        """Set class imbalance weight. Call before training.

        Args:
            n_neg: Number of negative (not-at-risk) samples in training set.
            n_pos: Number of positive (at-risk) samples in training set.
        """
        weight = torch.tensor(n_neg / max(n_pos, 1), dtype=torch.float32)
        self.pos_weight = weight
