"""Label DP via randomized response — the THIRD privacy channel.

The cut-layer noise protects each silo's features from the coordinator, and
server DP-SGD protects students from whoever receives the released model.
Neither protects the *labels*, which only the label-holding silo possesses and
which it would otherwise ship to the coordinator in the clear.

Mechanism: each binary label is flipped once, right after PSI alignment, and
the noisy copy is reused for every training round. Returned gradients are
post-processing of those noisy labels, so the guarantee covers both the
coordinator and the peer silo.

NOTE ON rr_with_prior: the prior-aware variant below is implemented and
tested but is NOT wired into the system. LabelRRAccountant.randomize(), which
fl_client/common.py calls, uses blind randomized_response(). Every reported
result therefore comes from the blind mechanism, and the forward correction in
corrected_positive_prob() is exact for it. Wiring rr_with_prior in would make
that correction a misspecification for the argmax branch, where the released
label does not depend on the true one, so the loss would need a per-example
mixture before the variant could be used.

Three properties make this channel different from the other two, and all three
are load-bearing:

  1. **Pure epsilon-DP** (delta = 0), not (eps, delta)-Gaussian.
  2. **One-shot.** Randomizing once and reusing costs epsilon once, no matter
     how many rounds run. Re-randomizing each round would compose and multiply
     the real cost -- see Ghazi et al. 2021.
  3. **Label-adjacent** neighboring relation: two datasets differ in one
     student's label, not in whether that student is present. This is why the
     label epsilon must be reported beside -- never composed into -- the
     Gaussian channels' epsilon. ``combine_epsilon`` rejects it on purpose.

References:
  - Warner 1965 -- randomized response
  - Ghazi et al. 2021 (NeurIPS) -- Deep Learning with Label Differential Privacy
"""

from __future__ import annotations

import math

import torch

#: Clamp corrected probabilities away from the open interval's endpoints so a
#: downstream binary_cross_entropy cannot evaluate log(0).
_PROB_EPS = 1e-7


def keep_probability(epsilon: float) -> float:
    """Probability that randomized response leaves a label unchanged.

    The standard binary randomized-response rate ``e^eps / (1 + e^eps)``, which
    is exactly ``sigmoid(epsilon)``. At ``epsilon = 0`` it is 0.5, meaning the
    released label is independent of the true one.

    Args:
        epsilon: Label privacy budget. Must be >= 0.

    Returns:
        The keep probability, in [0.5, 1).

    Raises:
        ValueError: If epsilon is negative.
    """
    if epsilon < 0:
        raise ValueError(f"epsilon must be >= 0, got {epsilon}")
    return 1.0 / (1.0 + math.exp(-epsilon))


def randomized_response(
    labels: torch.Tensor, epsilon: float, seed: int
) -> torch.Tensor:
    """Flip each binary label independently, keeping it with probability p.

    Call this ONCE per run, after PSI alignment, and persist the result. The
    draw is seeded so a silo restarting mid-run reproduces the same noisy
    labels instead of spending the budget a second time.

    Args:
        labels: Binary label vector containing only 0.0 and 1.0.
        epsilon: Label privacy budget (see :func:`keep_probability`).
        seed: Seed for the flip draw. Same seed => same noisy labels.

    Returns:
        A NEW tensor of noisy labels, same shape and dtype. The input is not
        modified -- the label holder keeps the true labels for evaluation.

    Raises:
        ValueError: If epsilon is negative or the labels are not binary.
    """
    unique = torch.unique(labels)
    if not torch.all((unique == 0) | (unique == 1)):
        raise ValueError(
            f"randomized_response expects binary labels (0/1), got values {unique.tolist()}"
        )

    p = keep_probability(epsilon)
    generator = torch.Generator(device=labels.device).manual_seed(seed)
    keep = torch.rand(labels.shape, generator=generator, device=labels.device) < p
    flipped = 1.0 - labels
    return torch.where(keep, labels, flipped).to(labels.dtype)


def corrected_positive_prob(logits: torch.Tensor, keep_prob: float) -> torch.Tensor:
    """Push model logits through the known label-noise channel (forward correction).

    The model predicts ``P(y = 1 | x)``, but the training targets are the noisy
    labels, so the likelihood must describe what is actually observed::

        P(y~ = 1 | x) = p * sigmoid(z) + (1 - p) * (1 - sigmoid(z))
                      = (1 - p) + (2p - 1) * sigmoid(z)

    Train with ``binary_cross_entropy`` against this probability rather than
    ``binary_cross_entropy_with_logits`` against the raw logits. Fitting the
    noisy labels directly converges to the *corrupted* base rate, which at
    epsilon = 1 means a model calibrated to roughly 27% positives instead of
    the true 46%.

    Args:
        logits: Raw model outputs (pre-sigmoid), any shape.
        keep_prob: The p used when the labels were randomized.

    Returns:
        Probabilities of observing a noisy positive, same shape as ``logits``,
        clamped into (0, 1) for numerical safety.

    Raises:
        ValueError: If keep_prob is outside [0.5, 1].
    """
    if not 0.5 <= keep_prob <= 1.0:
        raise ValueError(f"keep_prob must be in [0.5, 1], got {keep_prob}")
    prob = (1.0 - keep_prob) + (2.0 * keep_prob - 1.0) * torch.sigmoid(logits)
    return prob.clamp(_PROB_EPS, 1.0 - _PROB_EPS)


def rr_with_prior(
    labels: torch.Tensor,
    prior: torch.Tensor,
    epsilon: float,
    seed: int,
) -> torch.Tensor:
    """RRWithPrior (Ghazi et al. 2021), specialized to binary labels for VFL.

    Plain randomized response spends its budget flipping labels blindly.
    RRWithPrior spends it only where the prior is uncertain. The general
    algorithm picks the subset size ``k`` maximizing

        w_k = e^eps / (e^eps + k - 1) * (sum of the top-k prior masses)

    and applies randomized response within that subset. For binary labels only
    k in {1, 2} exist, and the rule collapses to a comparison against the RR
    keep rate ``q = e^eps / (e^eps + 1)``:

      * ``p_top1 > q``  -> k* = 1: emit the prior's argmax. The output does not
        depend on the true label at all, so no budget is spent on this example
        and the released label is more accurate than a flip would be.
      * otherwise       -> k* = 2: plain randomized response.

    **VFL adaptation.** The label holder sees only its own 18 features, so a
    prior it computes alone is weak. The coordinator holds the joint model over
    both silos and already returns predicted probabilities to the holder for
    scoring, so those predictions serve as the prior at no extra communication.
    The guarantee survives because a stage's prior is produced by a model
    fitted on *other* students' already-randomized labels -- never on this
    student's label -- so each label is privatized exactly once and the stages
    compose in parallel rather than sequentially (LP-MST).

    Caveat worth stating in any write-up: on the k*=1 branch the released label
    is the model's own prediction, so a confidently wrong prior entrenches its
    error. The mechanism trades label noise for prior bias.

    Args:
        labels: True binary labels, shape (n,).
        prior: Per-example class probabilities, shape (n, 2), rows summing to 1.
        epsilon: Label privacy budget.
        seed: Seed for the one-time draw.

    Returns:
        A NEW tensor of released labels, same shape and dtype as ``labels``.

    Raises:
        ValueError: If ``prior`` is not (n, 2) matching ``labels``.
    """
    if prior.ndim != 2 or prior.shape[0] != labels.shape[0] or prior.shape[1] != 2:
        raise ValueError(
            f"prior must have shape ({labels.shape[0]}, 2) for binary labels, "
            f"got {tuple(prior.shape)}"
        )

    q = keep_probability(epsilon)
    top_prob, top_idx = prior.max(dim=1)

    # k* = 1 where the prior already beats the RR keep rate.
    use_argmax = top_prob > q

    generator = torch.Generator(device=labels.device).manual_seed(seed)
    keep = torch.rand(labels.shape, generator=generator, device=labels.device) < q
    rr_out = torch.where(keep, labels, 1.0 - labels)

    return torch.where(use_argmax, top_idx.to(labels.dtype), rr_out).to(labels.dtype)


class LabelRRAccountant:
    """One-shot pure-epsilon accountant for the label channel.

    Deliberately NOT a :class:`~privacy.privacy_accountant.PrivacyBudgetTracker`:
    it has no sigma, no sampling rate, and its epsilon does not grow with
    steps. ``combine_epsilon`` rejects it rather than silently folding a
    label-adjacent guarantee into an example-adjacent one.

    Args:
        epsilon: Label privacy budget. Must be > 0.
        seed: Seed used for the one-time flip draw, persisted for reproduction.
    """

    #: Reported in Table I so each channel's guarantee is unambiguous.
    neighboring_relation = "label-adjacent (one student's label differs)"
    mechanism = "randomized response"

    def __init__(self, epsilon: float, seed: int = 0) -> None:
        if epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}")
        self.epsilon = float(epsilon)
        self.seed = int(seed)
        self.delta = 0.0
        self.keep_prob = keep_probability(self.epsilon)
        #: Rounds that reused the noisy labels. Recorded for provenance only --
        #: it must never enter the epsilon calculation.
        self.steps = 0

    def step(self) -> None:
        """Record one more round of reuse. Does not spend any budget."""
        self.steps += 1

    def get_epsilon(self) -> float:
        """Cumulative label epsilon: constant, because the labels were drawn once."""
        return self.epsilon

    def randomize(self, labels: torch.Tensor) -> torch.Tensor:
        """Apply this accountant's mechanism to a true label vector.

        Args:
            labels: Binary label vector.

        Returns:
            A new noisy label tensor.
        """
        return randomized_response(labels, epsilon=self.epsilon, seed=self.seed)

    def to_dict(self) -> dict[str, float | int | str]:
        """Plain-primitive state for context.state / meta.json persistence."""
        return {
            "mechanism": self.mechanism,
            "neighboring_relation": self.neighboring_relation,
            "epsilon": self.epsilon,
            "delta": self.delta,
            "keep_prob": self.keep_prob,
            "seed": self.seed,
            "steps": self.steps,
        }

    @classmethod
    def from_dict(cls, state: dict[str, float | int | str]) -> LabelRRAccountant:
        """Rebuild an accountant from :meth:`to_dict` output."""
        acct = cls(epsilon=float(state["epsilon"]), seed=int(state["seed"]))
        acct.steps = int(state.get("steps", 0))
        return acct
