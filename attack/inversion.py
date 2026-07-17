"""Model Inversion Attack (iDLG / gradient-matching) against Silo A embeddings.

Owner: Wyatt Wilfong
Status: SCAFFOLD — implement in Week 8. Alex provides checkpoint loader utility.

Goal: Given only the embedding vector emitted by BottomModelA (the "cut layer output"),
      can an attacker reconstruct the original input features — specifically, imd_band?

Method (Zhao et al. 2020, iDLG):
  Initialize dummy_input ~ N(0, 1) with requires_grad=True.
  Optimize dummy_input to minimize: loss = ||bottom_model_a(dummy_input) - real_embedding||^2
  After convergence, read dummy_input[imd_band_col] and compare to true label.

References:
  - Zhao et al. 2020 (NeurIPS) — iDLG Algorithm 1
  - Jin et al. 2021 (NeurIPS) — CAFE attack (stronger VFL-specific variant)
  - CLAUDE.md "Implementing the Model Inversion Attack" step-by-step box
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from silo_a.bottom_model_a import BottomModelA

# Column index of imd_band in the 12-dim Silo A feature vector (see data_loader_a.py)
IMD_BAND_COL_IDX = 0
IMD_BAND_NUM_CLASSES = 10  # OULAD imd_band has 10 ordinal levels (0–9)


def load_checkpoint(checkpoint_path: str | Path) -> tuple[BottomModelA, dict]:
    """Load a BottomModelA checkpoint saved by Alex's training script.

    Args:
        checkpoint_path: Path to a .pt file like experiments/checkpoints/exp03_eps_1.0/model.pt

    Returns:
        Tuple of (loaded BottomModelA in eval mode, checkpoint metadata dict).
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model = BottomModelA(
        input_dim=ckpt.get("input_dim", 12),
        embedding_dim=ckpt.get("embedding_dim", 32),
    )
    model.load_state_dict(ckpt["bottom_model_a_state_dict"])
    model.eval()
    return model, ckpt


def run_inversion_attack(
    model: BottomModelA,
    real_embedding: torch.Tensor,
    true_imd_band: int,
    num_iters: int = 200,
    lr: float = 0.1,
) -> dict:
    """Run one iDLG attack attempt against a single real embedding.

    Args:
        model: BottomModelA in eval mode (from load_checkpoint).
        real_embedding: The target embedding to reconstruct from, shape (1, 32).
        true_imd_band: Ground-truth imd_band value (0–9) for success measurement.
        num_iters: Number of L-BFGS optimization iterations. Default 200.
        lr: L-BFGS learning rate. Default 0.1.

    Returns:
        Dict with keys: {success (bool), predicted_imd_band (int),
                         true_imd_band (int), final_loss (float), iters (int)}.

    TODO (Wyatt, Week 8 — Steps 3–5 in CLAUDE.md attack guide):
        Step 3: dummy_input = torch.randn(1, 12, requires_grad=True)
        Step 4: optimizer_attack = torch.optim.LBFGS([dummy_input], lr=lr)
                for i in range(num_iters):
                    def closure():
                        optimizer_attack.zero_grad()
                        dummy_emb = model(dummy_input)
                        loss = ((dummy_emb - real_embedding.detach()) ** 2).sum()
                        loss.backward()
                        return loss
                    optimizer_attack.step(closure)
        Step 5: predicted = dummy_input[0, IMD_BAND_COL_IDX].round().clamp(0, 9).int().item()
                success = (predicted == true_imd_band)
    """
    dummy_input = torch.randn(1, 12, requires_grad = True)
    optimizer_attack = torch.optim.LBFGS([dummy_input], lr = lr)
    for i in range(num_iters):
        def closure():
            optimizer_attack.zero_grad()
            dummy_emb = model(dummy_input)
            loss = ((dummy_emb - real_embedding.detach()) ** 2).sum()
            loss.backward()
            return loss
        end_loss = optimizer_attack.step(closure)
    predicted = dummy_input[0, IMD_BAND_COL_IDX].round().clamp(0, 9).int().item()
    success = (predicted == true_imd_band)

    return {"success" : success, "predicted_imd_band" : predicted, "true_imd_band" : true_imd_band, "final_loss" : end_loss.item(), "iters" : num_iters}



def evaluate_attack_across_epsilon(
    checkpoint_dir: str | Path,
    test_loader: DataLoader,
    epsilon_values: list[float],
    n_students: int = 100,
    output_csv: str | Path = "experiments/attack_results.csv",
) -> None:
    """Run the inversion attack for each epsilon checkpoint and record ASR.

    Args:
        checkpoint_dir: Root directory containing per-epsilon checkpoint subdirs.
        test_loader: DataLoader for Silo A test set (features + labels).
        epsilon_values: List of epsilon values to test (e.g., [0.1, 0.5, 1.0, 5.0, float('inf')]).
        n_students: Number of test students to attack per epsilon. Default 100.
        output_csv: Path to write results CSV.

    Output CSV columns: [epsilon, asr, n_students, mean_final_loss]

    TODO (Wyatt, Week 9): loop over epsilon_values, load checkpoint, run attack
                          on first n_students samples, record ASR in output_csv.
    """
    raise NotImplementedError("TODO (Wyatt, Week 9): implement evaluate_attack_across_epsilon()")
