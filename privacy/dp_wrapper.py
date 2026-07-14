"""Opacus DP-SGD wrapper for the SERVER top model.

Owner: Wyatt Wilfong

The noise multiplier is chosen by the caller (via
``privacy.privacy_accountant.solve_noise_multiplier``) so that autodp — the
accountant that ENFORCES the budget — is also the one that sizes the noise.
No Opacus PrivacyEngine or DataLoader is involved: the server drives its own
training loop, so the model/optimizer are wrapped directly.

References:
  - Abadi et al. 2016 (CCS) — DP-SGD algorithm
"""

import warnings

import torch
import torch.nn as nn
from opacus.grad_sample import GradSampleModule
from opacus.optimizers import DPOptimizer

# Opacus GradSampleModule registers non-full backward hooks by design and never
# reads the grad_input the deprecation warning is about — safe to silence.
warnings.filterwarnings(
    "ignore", message="Using a non-full backward hook", category=FutureWarning
)


def make_private_with_noise(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    noise_multiplier: float,
    max_grad_norm: float,
    expected_batch_size: int,
) -> tuple[GradSampleModule, DPOptimizer]:
    """Wrap model + optimizer for DP-SGD with a caller-chosen noise multiplier.

    Use the returned pair in place of the originals; unwrap saved weights via
    ``dp_model._module.state_dict()``. Accounting is autodp's job — pass the same
    ``noise_multiplier`` and sample rate to ``PrivacyBudgetTracker``.

    Args:
        model: Top model (standard nn.Module, no custom autograd Functions).
        optimizer: Optimizer for the model.
        noise_multiplier: sigma = noise_std / max_grad_norm.
        max_grad_norm: Per-sample gradient clipping threshold.
        expected_batch_size: Expected rows per step (sample_rate * dataset size).

    Returns:
        Tuple of (dp_model, dp_optimizer).

    Raises:
        ValueError: If noise_multiplier, max_grad_norm, or expected_batch_size is <= 0.
    """
    if noise_multiplier <= 0:
        raise ValueError(f"noise_multiplier must be > 0, got {noise_multiplier}")
    if max_grad_norm <= 0:
        raise ValueError(f"max_grad_norm must be > 0, got {max_grad_norm}")
    if expected_batch_size <= 0:
        raise ValueError(f"expected_batch_size must be > 0, got {expected_batch_size}")

    dp_model = GradSampleModule(model)
    dp_optimizer = DPOptimizer(
        optimizer=optimizer,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        expected_batch_size=expected_batch_size,
    )
    return dp_model, dp_optimizer
