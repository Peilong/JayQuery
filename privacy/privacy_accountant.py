"""Privacy Budget Accountant — tracks cumulative epsilon across training rounds.

Owner: Wyatt Wilfong
Status: SCAFFOLD — implement PrivacyBudgetTracker in Week 5.

References:
  - Mironov 2017 (CSF) — Rényi DP composition
"""

from privacy.dp_wrapper import PrivacyBudgetExceededError
from autodp import mechanism_zoo
from autodp import dp_acct
from autodp import transformer_zoo

from privacy.dp_wrapper import PrivacyBudgetExceededError


class PrivacyBudgetTracker:
    """Tracks per-round epsilon expenditure and raises an error when budget is exhausted.

    Usage:
        tracker = PrivacyBudgetTracker(
            target_epsilon=1.0, delta=1e-5,
            num_epochs=10, batch_size=64, dataset_size=5000
        )
        for round in range(num_rounds):
            tracker.step()
            current_eps = tracker.get_epsilon()
            print(f"Round {round}: epsilon = {current_eps:.4f}")

    Args:
        target_epsilon: Maximum allowable epsilon. Training stops when exceeded.
        delta: DP delta parameter (e.g., 1e-5).
        num_epochs: Total training epochs.
        batch_size: Mini-batch size.
        dataset_size: Total number of training samples.

    TODO (Wyatt, Week 5 — Step 4 in CLAUDE.md DP guide):
        Implement __init__, step(), and get_epsilon() using the pattern:
            current_eps = dp_optimizer.privacy_engine.get_epsilon(delta=self.delta)
        Store dp_optimizer reference or use autodp for standalone computation.
    """

    def __init__(
        self,
        target_epsilon: float,
        sigma: float = 1.0,
        sensitivity: float = 1.0,
        delta: float = 1e-5,
        num_epochs: int = 10,
        batch_size: int = 64,
        dataset_size: int = 5000,
    ) -> None:
        self.target_epsilon = target_epsilon
        self.delta = delta
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.dataset_size = dataset_size
        self._round = 0

        # TODO (Wyatt, Week 7): initialize autodp accountant.
        self.mechanism = mechanism_zoo.GaussianMechanism(sigma=sigma)
        self.accountant = dp_acct.DP_acct()

    def step(self) -> None:
        """Advance the accountant by one training round.

        Raises:
            PrivacyBudgetExceededError: If current epsilon > target_epsilon after this step.

        TODO (Wyatt, Week 5): implement step() — increment round counter and check budget.
        """
        self._round += 1
        self.accountant.update_DPlosses(eps = self.get_epsilon(), delta = self.delta)
        current = self.get_epsilon()
        if current > self.target_epsilon:
            raise PrivacyBudgetExceededError(
                f"Privacy budget exceeded: {current:.4f} > {self.target_epsilon}"
            )

    def get_epsilon(self) -> float:
        """Return current cumulative epsilon expenditure.

        Returns:
            Current epsilon as a float.

        TODO (Wyatt, Week 5): implement using autodp.
        """
        transformer = transformer_zoo.ComposeGaussian()
        self.mechanism = transformer.compose(mechanism_list = [self.mechanism], coeff_list = [self._round])
        return self.mechanism.get_eps(delta=self.delta)

    @property
    def rounds_completed(self) -> int:
        """Number of training rounds completed so far."""
        return self._round
