# CLAUDE.md — JayQuery Agentic Development Guide

> This file is read by Claude Code on every task. Follow it exactly.

---

## Project Overview

**JayQuery** is a Vertical Federated Learning (VFL) system that trains a Student Success
(dropout-risk) prediction model across two simulated institutional data silos — without
ever centralizing raw student records. It uses the Open University Learning Analytics
Dataset (OULAD) as a proxy for FERPA-protected campus data.

**Stack (see `requirements.txt` for exact pins):** Python 3.11 · PyTorch 2.4.1 · Flower (flwr[simulation]) 1.28.0 · Opacus 1.5.2 · autodp · scikit-learn 1.5.2 · XGBoost 2.1.1 · wandb 0.27.0

> **Architecture note:** The DP design was revised — see **DP Architecture (READ FIRST)** below and
> `project_plan_new.md` (authoritative). Short version: client privacy is **cut-layer Gaussian noise**,
> the server's top model uses **Opacus DP-SGD**, there is **no client-side Opacus**, and budget is tracked
> with **autodp**, not a persisted `PrivacyEngine`.

> **Dependency warning (tested, act before Wyatt starts privacy work):**
> - `autodp==0.2.3` (currently commented in `requirements.txt`) is **broken** — no wheel, sdist fails to
>   build. Use **`autodp==0.2.3.1`**.
> - autodp otherwise pulls a modern `scipy` that requires numpy≥2.0, conflicting with `numpy==1.26.4`.
>   Add a pin: **`scipy==1.13.1`**. Verified working stack: `numpy 1.26.4 + scipy 1.13.1 + autodp 0.2.3.1`.

---

## DP Architecture (READ FIRST)

The original single-Opacus-wrapper-on-the-bottom-models plan was **dropped** — it is incompatible with
the VFL loop, where clients update via upstream-gradient injection (`embedding.backward(grad)`), not a
local `loss.backward()`, so Opacus has no local loss to hook on the clients.

Current design:
1. **Clients → cut-layer Gaussian noise.** Each silo adds calibrated Gaussian noise to its embedding
   *before send*, in `fl_client/common.py:generate_embeddings`, gated on a config flag. Implemented in
   `privacy/noise_injection.py`. This is the client-side DP mechanism.
2. **Server → DP-SGD.** The top model has a real `loss.backward()`, so Opacus DP-SGD works there
   unmodified. `privacy/dp_wrapper.py:make_private_with_noise` is applied in `fl_server/`. Firm deliverable.
3. **No client-side Opacus.** Bottom models are plain `nn.Module`s trained via injected gradients.
4. **Accounting → autodp.** Cumulative epsilon is composed with autodp over persisted *primitives*
   (`sigma`, `sensitivity`, `steps`). **Never persist a live `PrivacyEngine`** — `context.state` holds
   structured records only, and the engine's hooks do not survive serialization.
5. **Combined epsilon.** The system spends budget on both mechanisms; report the **composed** total.

**Threat model:** honest-but-curious server / man-in-the-middle on the inter-silo channel. Protected
surface = embeddings and gradients on the wire + the aggregating server. Local silo weights are assumed
protected by institutional trust boundaries (an attacker who can read them can read the raw data).

---

## Repository Map

```
jayquery/
├── CLAUDE.md                   ← you are here
├── README.md                   ← project entry point
├── project_plan_new.md         ← authoritative architecture & goals (supersedes DOCX where they differ)
├── pyproject.toml              ← flwr app config, run-config defaults, ruff/mypy/pytest
├── run_deployment.py           ← launches SuperLink + 2 SuperNodes + submits the run
├── data/raw/                   ← OULAD CSVs (gitignored, never commit)
├── data/processed/             ← silo_{a,b}.pkl, scaler_{a,b}.pkl, ordered_id_list.json (gitignored)
├── fl_server/                  ← ServerApp loop, Top Model, server-side DP-SGD wiring   [Alex]
│   ├── server_app.py           ← @app.main VFL round loop (persistent in-memory)
│   └── top_model.py            ← TopModel + compute_loss_and_gradients
│                                 (vfl_strategy.py is dead — remove; server_app.py is the live path)
├── fl_client/                  ← client dispatcher + generic handlers   [Alex]
│   ├── client_app.py           ← thin dispatcher: partition-id → silo handlers
│   ├── common.py               ← generic generate_embeddings / apply_gradients (+ cut-layer noise hook)
│   ├── silo_spec.py            ← SiloSpec: per-silo config consumed by common.py
│   ├── silo_a/handlers.py      ← Silo A SiloSpec + get_emb/apply_grad wrappers
│   └── silo_b/handlers.py      ← Silo B SiloSpec + get_emb/apply_grad wrappers
├── silo_a/                     ← Bottom Model A, DataLoader A   [Alex]
├── silo_b/                     ← Bottom Model B, DataLoader B   [Alex]
├── psi/                        ← ECDH-based Private Set Intersection   [Wyatt]
├── privacy/                    ← cut-layer noise, autodp budget tracker, server DP-SGD wrapper   [Wyatt]
│   ├── noise_injection.py      ← cut-layer Gaussian noise (client DP mechanism)
│   ├── privacy_accountant.py   ← PrivacyBudgetTracker (autodp-based)
│   └── dp_wrapper.py           ← make_private_with_noise (Opacus) — applied to the SERVER top model
├── attack/                     ← iDLG inversion attack, label inference, gradient monitor   [Wyatt]
├── baselines/                  ← centralized ML baselines, eval utilities   [Alex]
├── utils/                      ← reproducibility seed, wandb helpers, config loader   [Shared]
├── experiments/configs/        ← YAML experiment configs   [Alex]
├── experiments/checkpoints/    ← saved model checkpoints (gitignored)
├── notebooks/                  ← EDA notebook (Alex), testing notebook (Wyatt)
├── tests/                      ← pytest unit tests (test_loaders, test_models, test_dp, test_psi)
├── tests/fixtures/             ← synthetic OULAD CSVs committed for CI
├── docs/                       ← design docs, paper notes, weekly reports, original DOCX
├── proto/                      ← gRPC protobuf definitions
└── scripts/                    ← download_oulad.sh, prepare_silos.py, reproduce_all.sh
```

---

## Student Role Boundaries

| Module | Owner | Autonomy Level |
|--------|-------|----------------|
| `fl_server/`, `fl_client/`, `silo_a/`, `silo_b/`, `baselines/`, EDA notebook | **Alex Pellet** | High — make design decisions, document rationale in `docs/alex_paper_notes.md` |
| `psi/`, `privacy/`, `attack/`, testing notebook | **Wyatt Wilfong** | Guided — follow step-by-step scaffolds in the week plan and research plan |
| `utils/`, `experiments/`, `tests/`, `docs/` | **Shared** | Alex owns experiment orchestration; Wyatt owns privacy docs |

Respect this ownership — do not refactor a module that belongs to the other student without an explicit
instruction. **One cross-owned seam:** the server DP-SGD wrapper — Alex wires the structural hook in
`fl_server/`; Wyatt owns the mechanism and its accounting in `privacy/`.

---

## Coding Standards (Enforced by CI)

1. **Type hints on every function signature.** No bare `def foo(x):`. (ruff `ANN` is enabled.)
2. **Google-style docstrings** on every public class and function.
3. **No magic numbers** — hyperparameters go in `pyproject.toml` run-config or a YAML, never hard-coded in model files.
4. **Seed everything** — use `utils/reproducibility.py:seed_everything(42)` at the top of every training script.
5. **Log every training run to wandb** (see naming convention below).
6. **Tests before merge** — every PR must have passing `pytest tests/` before review.
7. **No raw data in git** — `data/raw/`, `data/processed/`, and `experiments/checkpoints/` are gitignored.

Tooling: `ruff` (line-length 100, rules E/F/I/UP/ANN), `mypy` (py311, ignore-missing-imports), `pytest`
with markers `slow`, `psi`, `dp`, `model`, `integration`.

---

## Running the Project

### 1 — Environment setup (first time only)
```bash
git clone https://github.com/Peilong/JayQuery.git
cd JayQuery
conda create -n jayquery python=3.11 && conda activate jayquery   # conda optional
pip install -r requirements.txt
pip install -e .
cp .env.example .env             # fill in WANDB_API_KEY
```

### 2 — Verify environment
```bash
python -c "import torch, flwr, opacus; print('OK')"
# once autodp is uncommented (0.2.3.1) + scipy 1.13.1 pinned:
python -c "import autodp; from autodp.mechanism_zoo import GaussianMechanism; print('autodp OK')"
```

### 3 — Download OULAD
```bash
bash scripts/download_oulad.sh   # curls + extracts 7 CSVs into data/raw/
```

### 4 — Run PSI (Wyatt) → writes data/processed/ordered_id_list.json
```bash
python scripts/extract_ids_for_psi.py
python -m psi.psi_server &
python -m psi.psi_client
```

### 5 — Prepare silos (Alex)
```bash
python scripts/prepare_silos.py                 # default module CCC_2014J, seed 42
# outputs silo_{a,b}.pkl + scaler_{a,b}.pkl into data/processed/
```

### 6 — Run VFL training (deployment: SuperLink + 2 SuperNodes)
```bash
python run_deployment.py --rounds 50
# run_deployment.py starts flower-superlink + one flower-supernode per silo
# (partition-id 0 = Silo A, 1 = Silo B), registers the connection in ~/.flwr/config.toml,
# then submits `flwr run`. This is the supported run path — prefer it over a bare `flwr run .`,
# which currently has no federation configured in pyproject.toml.
```
DP behavior is controlled by the `[tool.flwr.app.config]` keys in `pyproject.toml` (see below).

### 7 — Baselines (Alex)
```bash
python -m baselines.run_baselines --config experiments/configs/exp01_centralized_baselines.yaml --run all
```

### 8 — All experiments / tests
```bash
bash scripts/reproduce_all.sh
pytest tests/ -v --cov=. --cov-report=term-missing
```

---

## Run-Config (pyproject.toml `[tool.flwr.app.config]`)

Current keys (defaults):
```
num-server-rounds        = 50
learning-rate            = 1e-2
embedding-dim            = 32
input-dim-a              = 18     # 6 encoded + 12 region one-hots (drop-first) — matches prepare_silos.py
input-dim-b              = 27     # 7 assessment/VLE stats + 20 vle_clicks_* — matches prepare_silos.py
processed-dir            = "data/processed"
seed                     = 42     # also seeds the per-round batch selection
batch-sample-rate        = 0.1    # Poisson sampling rate q; 1.0 = full-batch rounds
server-dp-enabled        = true   # server DP-SGD (Opacus mechanics, autodp accounting)
server-dp-epsilon        = 1.0
server-dp-delta          = 1e-5
server-dp-max-grad-norm  = 1.0
cut-noise-enabled        = true   # client cut-layer noise
cut-noise-target-epsilon = 1.0
cut-noise-delta          = 1e-5
cut-noise-max-norm       = 1.0    # per-row embedding L2 clip = the mechanism's sensitivity
combined-target-epsilon  = 3.0    # backstop on the composed (all-sources) epsilon
track-wandb              = true
wandb-log-interval       = 10     # client wandb flush cadence (rounds); <=1 = every round
```

Mini-batching: the server sends a per-round `batch-seed` + `batch-sample-rate`; both silos
Poisson-sample the **same** rows of the PSI-aligned training set from that seed. Both accountants
carry the same `sample_rate`, so subsampling amplification keeps the composed epsilon low. Noise for
both mechanisms is sized with `solve_noise_multiplier` over the full `num-server-rounds` horizon.
Budget gating: each source checks `would_exceed()` before spending, and the server additionally
gates on the prospective **combined** epsilon (`combined-target-epsilon`) at the end of each round;
if ANY gate trips, the server stops the whole run and pushes a `stop-run` message so clients
checkpoint. The clip is part of the training graph: `apply_gradients` backprops through
`clip_embedding_rows`, and eval embeddings are clipped (not noised) so the top model sees the
training distribution at eval. Client wandb metrics are buffered in `context.state` and uploaded
in one resume-by-id cycle every `wandb-log-interval` rounds (plus a forced flush on the final
round and on `stop-run`).

---

## Key Interfaces (Do Not Break These)

### Cut-layer noise (`privacy/noise_injection.py`) — CLIENT DP mechanism
```python
def clip_embedding_rows(embedding: torch.Tensor, max_norm: float) -> torch.Tensor:
    """Row-wise L2 clip enforcing the DP sensitivity bound. Returns a NEW tensor."""

def add_gaussian_noise(embedding: torch.Tensor, noise_std: float) -> torch.Tensor:
    """Gaussian noise at fixed std = sigma * max_norm, where sigma comes from
    solve_noise_multiplier over the run horizon. Returns a NEW tensor. Applied
    (after clipping) in generate_embeddings to the SENT embedding only."""
```

### Privacy accounting (`privacy/privacy_accountant.py`)
```python
class PrivacyBudgetTracker:
    """autodp-based cumulative-epsilon accountant. Holds plain primitives (sigma, sample_rate,
    steps, delta, target_epsilon); does NOT hold a live PrivacyEngine. sigma is the normalized
    noise multiplier (noise_std / sensitivity — caller does the division; the server passes the
    DP optimizer's noise_multiplier). sample_rate < 1.0 applies Poisson-subsampling amplification
    (mini-batch); 1.0 uses exact Gaussian composition. step() advances a round; raises
    PrivacyBudgetExceededError (defined HERE, not dp_wrapper) past target_epsilon;
    would_exceed(n=1) gates BEFORE spending. to_dict()/from_dict() persist plain
    primitives for context.state."""

def solve_noise_multiplier(target_epsilon, sample_rate, steps, delta=1e-5, ...) -> float:
    """Least sigma whose autodp-composed epsilon over `steps` stays within budget."""

def combine_epsilon(trackers: Sequence[PrivacyBudgetTracker], delta: float) -> float:
    """The system's true guarantee: composed epsilon across ALL sources
    (silo A + silo B cut noise + server DP-SGD). This is the reported total."""
```

### Server DP-SGD wrapper (`privacy/dp_wrapper.py`) — SERVER top model only
```python
def make_private_with_noise(
    model: nn.Module, optimizer: torch.optim.Optimizer,
    noise_multiplier: float, max_grad_norm: float, expected_batch_size: int,
) -> tuple[GradSampleModule, DPOptimizer]:
    """Loader-free Opacus wrap with a caller-chosen sigma (from solve_noise_multiplier),
    so autodp both sizes AND enforces the budget. NOT used on the clients — bottom
    models use cut-layer noise instead. Unwrap weights via dp_model._module."""
```

### Embedding / cut layer
```python
def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
    """Bottom model: returns (batch_size, 32) embedding. No activation on output."""

def compute_loss_and_gradients(
    self, emb_a: torch.Tensor, emb_b: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Top model: returns (loss_scalar, grad_emb_a, grad_emb_b)."""
```

---

## wandb Convention

Entity (team) is always **`jayquery`**. Project varies by what's tracked:
- **`baselines`** — centralized/isolated-silo baseline models.
- **`jayquery-vfl`** — the SplitNN VFL runs (the client handlers already default to this).

```python
wandb.init(
    entity="jayquery",
    project="jayquery-vfl",   # or "baselines"
    name=f"exp{EXP_ID:02d}_{description}_eps{epsilon}",
    tags=["week{N}", "alex" or "wyatt", f"eps_{epsilon}"],
    config={...all hyperparams...},
)
```
Checkpoints: `experiments/checkpoints/exp{ID:02d}_eps{epsilon}/model.pt`, bundling both bottom models +
the top model + epsilon/delta + cut-layer noise params. Unwrap the server DP-SGD model
(`._module.state_dict()`) so saved weights are clean.

---

## Wyatt — Step-by-Step Task Queue

When Claude Code helps Wyatt:
1. One TODO at a time — never implement multiple functions in one step.
2. Run the relevant pytest after each function.
3. Print "STEP N COMPLETE — run: pytest tests/test_[module].py::[test_name]" after each step.
4. Never modify `fl_server/`, `silo_a/`, `silo_b/`, or `baselines/`. (The server DP-SGD *wiring* is Alex's;
   Wyatt supplies the mechanism/accounting from `privacy/`.)
5. If a test fails, diagnose and fix before the next step.

---

## Common Pitfalls

- **Why no Opacus on the clients:** the client update is `embedding.backward(grad)` (injected upstream
  gradient), not `loss.backward()`, so Opacus's per-sample hooks have nothing to attach to. Clients use
  cut-layer noise; Opacus lives on the **server** top model, which has a real loss.
- **Never persist a `PrivacyEngine`:** persist accountant *primitives* (plain numbers) and recompute
  epsilon via autodp. `context.state` holds structured records, not live objects with hooks.
- **autodp install:** use `0.2.3.1` (not `0.2.3`) and pin `scipy==1.13.1` alongside `numpy==1.26.4`.
- **DataLoader alignment:** always load silos using `ordered_id_list.json` to guarantee matching rows.
- **Deterministic client forward:** `generate_embeddings` snapshots RNG so `apply_gradients` can replay
  the same forward. Apply cut-layer noise to the *sent* embedding only — keep the stored clean embedding
  for the replay check.
- **Gradient shapes:** `grad_emb_a` and `grad_emb_b` are both `(batch_size, 32)`, sliced from the 64-dim top-model input gradient.
- **DP budget:** `PrivacyBudgetTracker` raises `PrivacyBudgetExceededError` — catch it gracefully.
- **Combined epsilon:** client cut-layer + server DP-SGD both spend budget — report the composed total.
- **wandb offline:** set `WANDB_MODE=offline` in `.env` and `wandb sync` later.