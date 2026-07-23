"""Gradient Norm Monitor — passive defense detecting anomalous gradient magnitudes.

Owner: Wyatt Wilfong
Status: SCAFFOLD — implement in Week 9.

Flags rounds where incoming gradient norms are >3x the running mean,
which can indicate an active gradient-based attack.
"""

import torch


class GradientNormMonitor:
    """Running-average gradient norm monitor with anomaly detection.

    Args:
        window: Number of recent rounds to include in the running average. Default 10.
        threshold_multiplier: Flag if norm > threshold_multiplier * running_mean. Default 3.0.
    """

    def __init__(self, window: int = 10, threshold_multiplier: float = 3.0) -> None:
        self.window = window
        self.threshold = threshold_multiplier
        self._history: list[float] = []

    def update(self, gradient: torch.Tensor) -> tuple[float, bool]:
        """Record a gradient tensor and check for anomaly.

        Args:
            gradient: Gradient tensor from one round, any shape.

        Returns:
            Tuple of (l2_norm, is_anomalous).

        TODO (Wyatt, Week 9):
            1. Compute L2 norm: norm = gradient.norm(p=2).item()
            2. Append to self._history; keep only last self.window values.
            3. If len(history) >= 3: running_mean = mean(history[:-1])
               is_anomalous = (norm > self.threshold * running_mean)
            4. Return (norm, is_anomalous).
        """
        norm = gradient.norm(p=2).item()
        self._history.append(norm)
        # Keep only the most recent `window` rounds in the running average.
        if len(self._history) > self.window:
            self._history = self._history[-self.window :]

        # Need at least two prior rounds before a running mean is meaningful; until
        # then nothing is flagged (avoids a false alarm on cold-start).
        is_anomalous = False
        if len(self._history) >= 3:
            prior = self._history[:-1]
            running_mean = sum(prior) / len(prior)
            is_anomalous = norm > self.threshold * running_mean
        return (norm, is_anomalous)

    @property
    def history(self) -> list[float]:
        return self._history.copy()
