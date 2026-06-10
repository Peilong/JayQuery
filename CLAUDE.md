# CLAUDE.md — JayQuery Agentic Development Guide

> This file is read by Claude Code on every task. Follow it exactly.

---

## Project Overview

**JayQuery** is a Vertical Federated Learning (VFL) system that trains a Student Success
(dropout-risk) prediction model across two simulated institutional data silos — without
ever centralizing raw student records. It uses the Open University Learning Analytics
Dataset (OULAD) as a proxy for FERPA-protected campus data.

**Stack:** Python 3.11 · PyTorch 2.2 · Flower (flwr) 1.8 · Opacus 1.4 · scikit-learn · XGBoost · wandb

---

## Repository Map

```
jayquery/
├── CLAUDE.md                   ← you are here
├── data/raw/                   ← OULAD CSVs (gitignored, never commit)
├── data/processed/             ← silo_a.pkl, silo_b.pkl, ordered_id_list.json
├── fl_server/                  ← Flower server, VFLStrategy, Top Model   [Alex]
├── silo_a/                     ← Flower client A, Bottom Model A, DataLoader A   [Alex]
├── silo_b/                     ← Flower client B, Bottom Model B, DataLoader B   [Alex]
├── psi/                        ← ECDH-based Private Set Intersection   [Wyatt]
├── privacy/                    ← Opacus DP wrapper, budget tracker, noise injection   [Wyatt]
├── attack/                     ← iDLG inversion attack, label inference, gradient monitor   [Wyatt]
├── baselines/                  ← centralized ML baselines, eval utilities   [Alex]
├── experiments/configs/        ← YAML experiment configs   [Alex]
├── experiments/checkpoints/    ← saved model checkpoints (gitignored)
├── notebooks/                  ← EDA (Alex), attack demo (Wyatt)
├── tests/                      ← pytest unit tests
├── docs/                       ← design docs, paper notes, weekly reports
├── proto/                      ← gRPC protobuf definitions
└── scripts/                    ← OULAD download, wandb setup, reproduce_all.sh
```

---

## Student Role Boundaries

| Module | Owner | Autonomy Level |
|--------|-------|----------------|
| `fl_server/`, `silo_a/`, `silo_b/`, `baselines/`, `notebooks/eda.ipynb` | **Alex Pellet** | High — make design decisions, document rationale in `docs/alex_paper_notes.md` |
| `psi/`, `privacy/`, `attack/`, `notebooks/attack_demo.ipynb` | **Wyatt Wilfong** | Guided — follow step-by-step scaffolds in `CLAUDE.md` and the research plan |
| `experiments/`, `tests/`, `docs/` | **Shared** | Alex owns experiment orchestration; Wyatt owns privacy docs |

When Claude Code works on a file, respect this ownership — do not refactor a module
that belongs to the other student without an explicit instruction.

---

## Coding Standards (Enforced by CI)

1. **Type hints on every function signature.** No bare `def foo(x):` — always `def foo(x: torch.Tensor) -> torch.Tensor:`.
2. **Google-style docstrings** on every public class and function.
3. **No magic numbers** — all hyperparameters go in a config dict or YAML, never hard-coded in model files.
4. **Seed everything** — use `utils/reproducibility.py:seed_everything(42)` at the top of every training script.
5. **Log every training run to wandb** — use `wandb.init(project="jayquery", tags=[...])`.
6. **Tests before merge** — every PR must have passing `pytest tests/` before review.
7. **No raw data in git** — `data/raw/` and `experiments/checkpoints/` are gitignored.

---

## Running the Project

### 1 — Environment
```bash
conda activate jayquery          # or: source .venv/bin/activate
python -c "import torch, flwr, opacus; print('OK')"
```

### 2 — Download OULAD
```bash
bash scripts/download_oulad.sh   # places 7 CSVs into data/raw/
```

### 3 — Prepare silos (Alex)
```bash
python scripts/prepare_silos.py  # outputs data/processed/silo_{a,b}.pkl + ordered_id_list.json
```

### 4 — Run PSI (Wyatt)
```bash
python -m psi.psi_server &       # start server in background
python -m psi.psi_client         # run PSI, writes data/processed/ordered_id_list.json
```

### 5 — Train VFL (no DP)
```bash
python -m fl_server.server --config experiments/configs/exp02_vfl_no_dp.yaml
```

### 6 — Train VFL + DP
```bash
python -m fl_server.server --config experiments/configs/exp03_dp_epsilon_grid.yaml --epsilon 1.0
```

### 7 — Run all experiments
```bash
bash scripts/reproduce_all.sh
```

### 8 — Run tests
```bash
pytest tests/ -v --cov=. --cov-report=term-missing
```

---

## Key Interfaces (Do Not Break These)

### PSI Contract (`psi/psi_client.py`)
```python
def run_psi(local_ids: list[str], server_host: str, server_port: int) -> list[str]:
    """Returns sorted list of IDs in the intersection. Writes ordered_id_list.json."""
```

### DP Wrapper Contract (`privacy/dp_wrapper.py`)
```python
def make_private(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    target_epsilon: float,
    target_delta: float,
    max_grad_norm: float,
    num_epochs: int,
) -> tuple[GradSampleModule, DPOptimizer, DataLoader, PrivacyEngine]:
    """Returns (dp_model, dp_optimizer, dp_loader, priv_engine). Raises ValueError if epsilon <= 0."""
```

### Embedding Interface (cut layer)
```python
# Bottom models MUST expose:
def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
    """Returns (batch_size, 32) embedding tensor. No activation on output."""

# Top model MUST expose:
def compute_loss_and_gradients(
    self, emb_a: torch.Tensor, emb_b: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (loss_scalar, grad_emb_a, grad_emb_b)."""
```

---

## Experiment Naming Convention

All wandb runs must be tagged:
```python
wandb.init(
    project="jayquery",
    name=f"exp{EXP_ID:02d}_{description}_eps{epsilon}",
    tags=["week{N}", "alex" or "wyatt", f"eps_{epsilon}"],
    config={...all hyperparams...}
)
```

Checkpoint files: `experiments/checkpoints/exp{ID:02d}_eps{epsilon}/model.pt`

---

## Wyatt — Step-by-Step Task Queue

When Claude Code helps Wyatt, always:
1. Work on one TODO at a time — never implement multiple functions in one step.
2. Run the relevant pytest after each function is implemented.
3. Print "STEP N COMPLETE — run: pytest tests/test_[module].py::[test_name]" after each step.
4. Never modify files in `fl_server/`, `silo_a/`, `silo_b/`, or `baselines/`.
5. If a test fails, diagnose and fix before moving to the next step.

---

## Common Pitfalls

- **Opacus + custom backward**: Bottom models must use only standard `nn.Module` ops. No custom autograd functions.
- **DataLoader alignment**: Always load silos using `ordered_id_list.json` to guarantee matching student rows.
- **Gradient shapes**: `grad_emb_a` and `grad_emb_b` are both `(batch_size, 32)` — slice the 64-dim top-model input gradient.
- **DP budget**: Never exceed `target_epsilon`. The `PrivacyBudgetTracker` raises `PrivacyBudgetExceededError` — catch it gracefully.
- **wandb offline**: If no internet, set `WANDB_MODE=offline` in `.env` and sync later with `wandb sync`.
