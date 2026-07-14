"""Opacus Differential Privacy wrapper for Bottom Model optimizers.

Owner: Wyatt Wilfong
Status: SCAFFOLD — implement make_private() in Week 4.

This module wraps a PyTorch model+optimizer pair with Opacus PrivacyEngine,
enabling DP-SGD training with per-sample gradient clipping and Gaussian noise.

References:
  - Abadi et al. 2016 (CCS) — DP-SGD algorithm
  - Opacus docs: https://opacus.ai/tutorials/
  - CLAUDE.md DP Wrapper Contract (Section "Key Interfaces")
"""


import torch
import torch.nn as nn
from opacus import PrivacyEngine
from opacus.grad_sample import GradSampleModule
from opacus.optimizers import DPOptimizer
from torch.utils.data import DataLoader


def make_private(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    target_epsilon: float,
    target_delta: float,
    max_grad_norm: float,
    num_epochs: int,
) -> tuple[GradSampleModule, DPOptimizer, DataLoader, PrivacyEngine]:
    """Wrap model, optimizer, and data_loader with Opacus PrivacyEngine.

    After calling this function, use the returned dp_model, dp_optimizer, and
    dp_loader IN PLACE of the originals. The dp_optimizer automatically adds
    calibrated Gaussian noise and clips per-sample gradients.

    Args:
        model: The Bottom Model to wrap (must be a standard nn.Module with no
               custom autograd functions — Opacus requirement).
        optimizer: The Adam/SGD optimizer for the model.
        data_loader: Training DataLoader (will be replaced with Opacus version
                     that uses UniformWithReplacementSampler).
        target_epsilon: Target privacy budget (e.g., 1.0). Lower = more private.
        target_delta: Probability of epsilon-DP failure (e.g., 1e-5).
        max_grad_norm: Per-sample gradient clipping threshold (e.g., 1.0).
        num_epochs: Total training epochs (needed to compute noise_multiplier).

    Returns:
        Tuple of (dp_model, dp_optimizer, dp_loader).

    Raises:
        ValueError: If target_epsilon <= 0 or target_delta <= 0.
        RuntimeError: If the model contains incompatible layers (e.g., BatchNorm
                      must be replaced with GroupNorm for Opacus compatibility).

    Example (from CLAUDE.md):
        dp_model, dp_optimizer, dp_loader = make_private(
            model=self.model, optimizer=self.optimizer,
            data_loader=self.train_loader,
            target_epsilon=1.0, target_delta=1e-5,
            max_grad_norm=1.0, num_epochs=10,
        )

    TODO (Wyatt, Week 4 — Steps 2–3 in CLAUDE.md DP guide):
        1. Validate inputs: raise ValueError if epsilon <= 0 or delta <= 0.
        2. Create PrivacyEngine instance.
        3. Call privacy_engine.make_private_with_epsilon() with all parameters.
        4. Return (dp_model, dp_optimizer, dp_loader).

        Exact Opacus call pattern (from CLAUDE.md):
            privacy_engine = PrivacyEngine()
            dp_model, dp_optimizer, dp_loader = privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=data_loader,
                target_epsilon=target_epsilon,
                target_delta=target_delta,
                epochs=num_epochs,
                max_grad_norm=max_grad_norm,
            )

    NOTE: If Opacus raises 'BatchNorm not supported', tell Alex — he will replace
          BatchNorm1d with GroupNorm in the Bottom Models.
    """
    if target_epsilon <= 0:
        raise ValueError(f"target_epsilon must be > 0, got {target_epsilon}")
    if target_delta <= 0:
        raise ValueError(f"target_delta must be > 0, got {target_delta}")
    
    priv_engine = PrivacyEngine()
    dp_model, dp_optimizer, dp_loader = priv_engine.make_private_with_epsilon(module = model, 
                                                                              optimizer = optimizer, 
                                                                              data_loader = data_loader, 
                                                                              target_epsilon = target_epsilon, 
                                                                              target_delta = target_delta, 
                                                                              epochs = num_epochs, 
                                                                              max_grad_norm = max_grad_norm)
    return dp_model, dp_optimizer, dp_loader, priv_engine


def get_current_epsilon(priv_engine: PrivacyEngine, delta: float = 1e-5) -> float:
    """Query the current epsilon expenditure from a DP-wrapped optimizer.

    Args:
        priv_engine: A PrivacyEngine instance.
        delta: The delta value used during training (should match target_delta).

    Returns:
        Current epsilon (float). Increases monotonically with training steps.
    """
    return priv_engine.get_epsilon(delta=delta)  # type: ignore[return-value]
