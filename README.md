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
  ┌─────────────────────┐          ┌─────────────────────┐
  │   Silo A            │   PSI    │   Silo B            │
  │  (Financial Aid /   │◄────────►│  (Academic Affairs / │
  │   Demographics)     │ handshake│   VLE + Assessments) │
  │                     │          │                     │
  │  Bottom Model A     │          │  Bottom Model B     │
  │  (32-dim embedding) │          │  (32-dim embedding) │
  └────────┬────────────┘          └────────┬────────────┘
           │  emb_a                          │  emb_b
           │         ┌──────────────┐        │
           └────────►│  FL Server   │◄───────┘
                     │  Top Model   │
                     │  [emb_a‖emb_b]→P(dropout)
                     │  + DP Noise  │
                     └──────────────┘
```

**Key technologies:**
| Component | Tech |
|-----------|------|
| Federated framework | Flower (flwr) 1.8 |
| Privacy-preserving record alignment | ECDH-based Private Set Intersection (PSI) |
| Privacy guarantee | Differential Privacy via Opacus 1.4 (DP-SGD) |
| Red Team evaluation | Model Inversion Attack (iDLG / gradient-matching) |
| Dataset | [OULAD](https://analyse.kmi.open.ac.uk/open-dataset) |

---

## Quick Start

```bash
# 1. Clone and set up environment
git clone https://github.com/Peilong/JayQuery.git
cd JayQuery
conda create -n jayquery python=3.11 && conda activate jayquery #conda is used here, but not neccessary
pip install -r requirements.txt

# 2. Copy and configure environment variables
cp .env.example .env        # fill in WANDB_API_KEY

# 3. Download OULAD dataset
bash scripts/download_oulad.sh

# 4. Extract ids for PSI
python scripts/extract_ids_for_psi.py

# 5. Run PSI protocol (terminal 1: server, terminal 2: client)
python -m psi.psi_server &
python -m psi.psi_client

# 6. Prepare data silos
python scripts/prepare_silos.py

# 7. Train VFL (no DP)
python -m fl_server.server --config experiments/configs/exp02_vfl_no_dp.yaml

# 8. Run all experiments (reproduces paper results)
bash scripts/reproduce_all.sh

# 9. Run tests
pytest tests/ -v
```

---

## Project Structure

```
jayquery/
├── CLAUDE.md               # Agentic dev instructions for Claude Code
├── fl_server/              # Flower server, Top Model, VFL Strategy
├── silo_a/                 # Silo A: demographics client + Bottom Model A
├── silo_b/                 # Silo B: VLE+assessments client + Bottom Model B
├── psi/                    # ECDH-based Private Set Intersection
├── privacy/                # Opacus DP wrapper, budget accountant, noise injection
├── attack/                 # Red Team: iDLG inversion, label inference, gradient monitor
├── baselines/              # Centralized baselines (LogReg, MLP, XGBoost) + eval utils
├── experiments/configs/    # YAML configs for all 12 experiments
├── notebooks/              # EDA + attack demo
├── tests/                  # pytest unit tests
├── docs/                   # Design docs, paper notes, weekly reports
└── scripts/                # Data download, silo prep, full reproduction
```

---

## Research Plan

The full 10-week plan (reading lists, software specs, model architectures, and
week-by-week deliverables) is in `docs/JayQuery_Research_Plan.docx`.

---

## Experiments

| # | Experiment | Owner |
|---|-----------|-------|
| 1 | Centralized baselines (LogReg, MLP, XGBoost, isolated silos) | Alex |
| 2 | VFL no-DP (upper bound) | Alex |
| 3 | DP-SGD epsilon grid {0.1, 0.5, 1.0, 2.0, 5.0, 10.0} | Both |
| 4 | Gaussian vs. Laplace noise at epsilon=1.0 | Alex+Wyatt |
| 5 | Clipping norm sensitivity | Wyatt |
| 6 | PSI overhead benchmark | Wyatt |
| 7 | Silo contribution ablation | Alex |
| 8 | Embedding dimension sweep | Alex |
| 9 | iDLG Model Inversion Attack vs. epsilon | Wyatt |
| 10 | Label inference attack | Wyatt |
| 11 | Temporal sufficiency (earliest intervention week) | Alex |
| 12 | Privacy-Utility Tradeoff curve | Both |

Track all runs at: https://wandb.ai/etown-cs/jayquery

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
