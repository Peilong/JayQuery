# JayQuery 🎓🔒

**Privacy-Preserving Student Success Prediction with Vertical Federated Learning**

> SCARP 2026 · Elizabethtown College · Dr. Peilong Li (mentor)  
> Alex Pellet (System Architect & ML Engineer) · Wyatt Wilfong (Privacy & Security Lead)

---

## What Is JayQuery?

Higher-ed institutions are *data-rich but insight-poor*: FERPA regulations prevent the
Registrar, Financial Aid, and the LMS from merging their student records into a single
model. JayQuery solves this with **Vertical Federated Learning (VFL)** — two data silos
collaborate to train a dropout-risk predictor *without ever sharing raw records*.

### System Overview

```
  ┌─────────────────────┐          ┌──────────────────────┐
  │   Silo A            │   PSI    │   Silo B             │
  │  (Financial Aid /   │◄────────►│  (Academic Affairs / │
  │   Demographics)     │ handshake│   VLE + Assessments) │
  │                     │          │                      │
  │  Bottom Model A     │          │  Bottom Model B      │
  │  → 32-dim embedding │          │  → 32-dim embedding  │
  │  + cut-layer noise  │          │  + cut-layer noise   │
  └────────┬────────────┘          └────────┬─────────────┘
           │  noised emb_a                   │  noised emb_b
           │         ┌──────────────┐        │
           └────────►│  FL Server   │◄───────┘
                     │  Top Model   │
                     │  [emb_a‖emb_b]→P(dropout)
                     │  DP-SGD (Opacus)
                     └──────────────┘
```

**Privacy design (revised).** Privacy is enforced in two places: each silo adds **cut-layer
Gaussian noise** to its embedding before it leaves the node, and the server's top model is
trained with **Opacus DP-SGD**. Client-side Opacus was dropped — it's incompatible with the
VFL loop, where clients update via injected gradients rather than a local loss. See
[`project_plan_new.md`](project_plan_new.md) for the full architecture and threat model.

**Key technologies:**
| Component | Tech |
|-----------|------|
| Federated framework | Flower (flwr) 1.28.0 |
| Privacy-preserving record alignment | ECDH-based Private Set Intersection (PSI) |
| Client privacy | Cut-layer Gaussian noise (Gaussian mechanism) |
| Server privacy | Differential Privacy via Opacus 1.5.2 (DP-SGD on the top model) |
| Privacy accounting | autodp composition (cumulative epsilon) |
| Red Team evaluation | Model Inversion Attack (iDLG / gradient-matching) |
| Dataset | [OULAD](https://analyse.kmi.open.ac.uk/open-dataset) |

---

## Quick Start

```bash
# 1. Clone and set up environment
git clone https://github.com/Peilong/JayQuery.git
cd JayQuery
conda create -n jayquery python=3.11 && conda activate jayquery   # conda optional
pip install -r requirements.txt
pip install -e .

# 2. Configure environment variables
cp .env.example .env        # fill in WANDB_API_KEY

# 3. Download OULAD dataset
bash scripts/download_oulad.sh

# 4. Run PSI protocol (terminal 1: server, terminal 2: client)
python scripts/extract_ids_for_psi.py
python -m psi.psi_server &
python -m psi.psi_client

# 5. Prepare data silos
python scripts/prepare_silos.py     # default module CCC_2014J

# 6. Train VFL (SuperLink + 2 SuperNodes, submits the run)
python run_deployment.py --rounds 50

# 7. Run all experiments (reproduces paper results)
bash scripts/reproduce_all.sh

# 8. Run tests
pytest tests/ -v
```

> DP behavior is set via the `[tool.flwr.app.config]` keys in `pyproject.toml`. Those key
> names are still being finalized as the DP layer is reworked (cut-layer noise on clients +
> DP-SGD on the server) — check `pyproject.toml` for the current set.

---

## Project Structure

```
jayquery/
├── CLAUDE.md               # Agentic dev instructions for Claude Code
├── project_plan_new.md     # Authoritative architecture & goals
├── pyproject.toml          # flwr app config + tooling
├── run_deployment.py       # Launches SuperLink + 2 SuperNodes + submits the run
├── fl_server/              # Flower server, Top Model, server-side DP-SGD
├── fl_client/              # Client dispatcher + generic handlers (client_app, common, silo_spec, silo_*/handlers)
├── silo_a/                 # Silo A: Bottom Model A + DataLoader A
├── silo_b/                 # Silo B: Bottom Model B + DataLoader B
├── psi/                    # ECDH-based Private Set Intersection
├── privacy/                # Cut-layer noise, autodp budget accountant, server DP-SGD wrapper
├── attack/                 # Red Team: iDLG inversion, label inference, gradient monitor
├── baselines/              # Centralized baselines (LogReg, MLP, XGBoost) + eval utils
├── experiments/configs/    # YAML configs for all experiments
├── notebooks/              # EDA + attack demo
├── tests/                  # pytest unit tests
├── docs/                   # Design docs, paper notes, weekly reports
└── scripts/                # download_oulad.sh, prepare_silos.py, reproduce_all.sh
```

---

## Research Plan

The authoritative description of the system architecture, privacy design, and goals is
[`project_plan_new.md`](project_plan_new.md). The original 10-week plan (reading lists,
software specs, week-by-week deliverables, symposium outline) is in
`docs/JayQuery_Research_Plan.docx` — still the record for reading assignments, roles, and
schedule, but superseded on architecture wherever the two differ.

---

## Experiments

| # | Experiment | Owner |
|---|-----------|-------|
| 1 | Centralized baselines (LogReg, MLP, XGBoost, isolated silos) | Alex |
| 2 | VFL no-DP (upper bound) | Alex |
| 3 | Privacy epsilon grid {0.1, 0.5, 1.0, 2.0, 5.0, 10.0} (cut-layer noise + server DP-SGD) | Both |
| 4 | Gaussian vs. Laplace cut-layer noise at epsilon=1.0 | Alex+Wyatt |
| 5 | Clipping norm sensitivity (server DP-SGD) | Wyatt |
| 6 | PSI overhead benchmark | Wyatt |
| 7 | Silo contribution ablation | Alex |
| 8 | Embedding dimension sweep | Alex |
| 9 | iDLG Model Inversion Attack vs. epsilon | Wyatt |
| 10 | Label inference attack | Wyatt |
| 11 | Temporal sufficiency (earliest intervention week) | Alex |
| 12 | Privacy-Utility Tradeoff curve | Both |

Tracked in wandb under entity **`jayquery`** — project `baselines` for baseline runs,
`jayquery-vfl` for the VFL/SplitNN runs.

---

## Contributing

- **Daily commits** required (see `CLAUDE.md` for format)
- **Branch naming:** `feat/[module]`, `fix/[issue]`, `exp/[exp-id]`
- **PR template:** `.github/PULL_REQUEST_TEMPLATE.md`
- **CI:** GitHub Actions runs `pytest` + `ruff` on every push

---

## License

MIT · See `LICENSE` for details.  
Dataset: OULAD is released under CC BY 4.0 by the Open University.