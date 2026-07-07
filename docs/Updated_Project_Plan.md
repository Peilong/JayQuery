# JayQuery — Project Plan (New)

_Privacy-Preserving Student Success Prediction with Vertical Federated Learning_ SCARP 2026 · Elizabethtown College · Dr. Peilong Li (mentor) Alex Pellet (System Architect & ML Engineer) · Wyatt Wilfong (Privacy & Security Lead)

---

## 0. Purpose & Status of This Document

This is the **current source of truth for JayQuery's architecture and structure.** It supersedes the technical/architecture description in the original `docs/JayQuery_Research_Plan.docx` wherever the two disagree — most importantly on the differential-privacy design (§1).

It does **not** replace the DOCX wholesale. The DOCX remains the record for administrative and background material the refactor didn't touch: reading lists, role-mentoring detail, the week-by-week SCARP schedule, the symposium outline, and the risk register. When the two conflict on _how the system works_, this file wins; for _reading assignments, roles, and schedule_, the DOCX still applies.

Week-scoped planning lives in separate living documents, not duplicated here:

- `JayQuery_DP_Concern_List.md` — prioritized action items (H/M/L) with owners.
- `JayQuery_Week_Plan.md` — the current week's day-by-day.
- `JayQuery_Architecture_Decision_Client_State.md` — the stateless-client decision record.

---

## 1. Architecture Decision — DP Redesign (READ FIRST)

The original plan specified a single Opacus DP-SGD wrapper applied to the bottom-model optimizers. **That approach was dropped.** It is incompatible with the SplitNN/VFL loop: the clients update their bottom models via upstream-gradient injection (`embedding.backward(grad)` using the gradient the server returns), not via a local `loss.backward()`. Opacus's per-sample machinery hooks the loss-to-parameter backward path, so on the clients there is no local loss for it to attach to.

**The redesigned privacy architecture:**

1. **Client-side privacy = cut-layer Gaussian noise.** Each silo adds calibrated Gaussian noise to its embedding _before_ it leaves the node. This is a differential-privacy (Gaussian) mechanism in its own right, so the privacy-first commitment is preserved. It protects the actual exposed surface — the embedding that crosses the wire.
2. **Server-side privacy = DP-SGD on the top model.** The server's top model is the one node with a genuine local loss (`loss.backward()` on the BCE objective), so Opacus DP-SGD works there unmodified. The server aggregates both silos and is the honest-but-curious party in our threat model, so hardening its protections carries outsized weight — a firm part of the design, not an add-on.
3. **No client-side Opacus.** The bottom models are plain `nn.Module`s trained via injected gradients; they are never wrapped by Opacus.

**Threat model.** The attacker is an honest-but-curious server and/or a man-in-the-middle on the inter-silo channel. The protected surface is what crosses the wire (embeddings, gradients) plus the aggregating server. We do **not** attempt to defend a silo's local model weights from an attacker who has already compromised that institution — such an attacker could read the raw data directly, so protecting the local model buys nothing. This scoping is what makes cut-layer noise (guarding the wire) plus server DP-SGD (guarding the aggregator) the right pair.

**Total privacy spend.** The system expends budget in two places — client cut-layer noise and server DP-SGD. The system's true guarantee is the **composition of both** and must be reported as a combined figure, not one mechanism in isolation.

---

## 2. System Architecture

```
  ┌─────────────────────┐          ┌──────────────────────┐
  │   Silo A            │   PSI    │   Silo B             │
  │  (Financial Aid /   │◄────────►│  (Academic Affairs / │
  │   Demographics)     │ handshake│   VLE + Assessments) │
  │  Bottom Model A     │          │  Bottom Model B      │
  │  → 32-dim embedding │          │  → 32-dim embedding  │
  │  + cut-layer noise  │          │  + cut-layer noise   │
  └────────┬────────────┘          └────────┬─────────────┘
           │  noised emb_a                   │  noised emb_b
           │         ┌─────────────────┐     │
           └────────►│   FL Server     │◄────┘
                     │   Top Model     │
                     │   [emb_a‖emb_b] → P(dropout)
                     │   DP-SGD (Opacus)
                     └────────┬────────┘
                              │  grad_emb_a, grad_emb_b
                              ▼  (returned to each silo)
```

**Round loop (SplitNN VFL):**

1. **PSI** aligns the two silos' student rows once, up front (ECDH-based), so embeddings correspond to the same students without sharing IDs in the clear. The intersection order is written to `data/processed/ordered_id_list.json` and consumed by `prepare_silos.py`.
2. Each silo runs its bottom model forward, adds cut-layer Gaussian noise, and sends the noised embedding (Silo A also sends labels) to the server.
3. The server concatenates `[emb_a ‖ emb_b]`, runs the DP-SGD-wrapped top model forward + backward on the BCE loss, and returns the per-silo embedding gradients.
4. Each silo applies its returned gradient to its bottom model (upstream-gradient injection) and steps its optimizer.

**Client execution model.** Clients are **stateless** — Flower invokes their handlers per message, and durable state lives in `context.state`. This matches real cross-silo deployment (nodes crash/restart/rejoin, so state must be reconstructable) and keeps simulation and deployment on one code path. Expensive-to-recompute artifacts (the materialized training set) use a rebuildable cache. Rationale is recorded in `JayQuery_Architecture_Decision_Client_State.md`. The **server** is persistent in-memory (its `@app.main` loop runs continuously), which is why its DP-SGD-wrapped top model needs no round-to-round serialization.

**Deployment target.** The project runs as a real Flower deployment (SuperLink + two SuperNodes) via `run_deployment.py`, not simulation-only — this is a deliberate proof-of-concept that the stateless design holds under genuine multi-process deployment.

---

## 3. Privacy Design

|Layer|Mechanism|Owner|Location|
|---|---|---|---|
|Client (each silo)|Cut-layer Gaussian noise on the embedding|Wyatt|`privacy/noise_injection.py`; injected in `fl_client/common.py:generate_embeddings` before send|
|Server (top model)|Opacus DP-SGD|Wyatt (mechanism) / Alex (wiring)|`privacy/dp_wrapper.py:make_private`, applied in `fl_server/`|
|Accounting|autodp composition over persisted primitives|Wyatt|`privacy/privacy_accountant.py:PrivacyBudgetTracker`|

**Accounting approach.** Privacy budget is tracked with **autodp** (a standalone library that composes per-round privacy spend into a cumulative epsilon), **not** by persisting a live Opacus `PrivacyEngine`. `context.state` holds only structured records, and a `PrivacyEngine` is a live object with hooks that does not survive serialization. Instead we persist plain primitives — noise `sigma`, `sensitivity`, and a cumulative `steps` counter — and recompute cumulative epsilon each round via autodp's `GaussianMechanism` + `Composition`. The client (Gaussian) accountant has no sampling term; the server (DP-SGD) accountant carries a `sample_rate` (stated as `1.0` while the server trains full-batch, with minibatching as a later tightening).

> **Dependency note (tested):** the pinned `autodp==0.2.3` in `requirements.txt` is broken (no wheel, sdist fails to build). Use **`autodp==0.2.3.1`**, and pin **`scipy==1.13.1`** — autodp otherwise pulls a scipy that requires numpy≥2.0 and conflicts with the project's `numpy==1.26.4`. The stack `numpy 1.26.4 + scipy 1.13.1 + autodp 0.2.3.1` was verified to compose Gaussian epsilon correctly and monotonically.

---

## 4. Project Structure

```
jayquery/
├── CLAUDE.md                  ← agentic dev guide (architecture, contracts, conventions)
├── README.md                  ← entry point: what it is, setup, how to run
├── project_plan_new.md        ← authoritative architecture & goals
├── pyproject.toml             ← flwr app config, run-config defaults, tooling
├── run_deployment.py          ← launches SuperLink + 2 SuperNodes + submits the run
├── data/raw/                  ← OULAD CSVs (gitignored)
├── data/processed/            ← silo_{a,b}.pkl, scaler_{a,b}.pkl, ordered_id_list.json (gitignored)
├── fl_server/                 ← ServerApp loop, Top Model, server-side DP-SGD wiring   [Alex]
│   ├── server_app.py          ← @app.main VFL round loop (persistent in-memory)
│   └── top_model.py           ← TopModel + compute_loss_and_gradients
├── fl_client/                 ← client dispatcher + generic handlers                    [Alex]
│   ├── client_app.py          ← thin dispatcher: partition-id → silo handlers
│   ├── common.py              ← generic generate_embeddings / apply_gradients (+ cut-layer noise hook)
│   ├── silo_spec.py           ← SiloSpec: per-silo config consumed by common.py
│   ├── silo_a/handlers.py     ← Silo A SiloSpec + handler wrappers
│   └── silo_b/handlers.py     ← Silo B SiloSpec + handler wrappers
├── silo_a/                    ← Bottom Model A, DataLoader A                             [Alex]
├── silo_b/                    ← Bottom Model B, DataLoader B                             [Alex]
├── psi/                       ← ECDH-based Private Set Intersection                      [Wyatt]
├── privacy/                   ← cut-layer noise, autodp budget tracker, DP-SGD wrapper  [Wyatt]
├── attack/                    ← iDLG inversion, label inference, gradient monitor        [Wyatt]
├── baselines/                 ← centralized ML baselines + eval utilities               [Alex]
├── utils/                     ← reproducibility seed, wandb helpers, config loader       [Shared]
├── experiments/              ← YAML configs, sweeps, checkpoints (gitignored)           [Alex]
├── notebooks/                 ← EDA (Alex), attack demo (Wyatt)
├── tests/                     ← pytest unit tests
├── docs/                      ← design docs, paper notes, weekly reports, original DOCX
└── scripts/                   ← download_oulad.sh, prepare_silos.py, reproduce_all.sh
```

**Ownership.** Alex owns system architecture, all models, data engineering, the server/client structure, and experiment orchestration. Wyatt owns the privacy mechanisms (cut-layer noise, accounting), the DP-SGD mechanism/budget for the server wrapper, and the Red Team attacks. The server DP-SGD wrapper is the one cross-owned piece: Alex wires the structural hook in `fl_server/`; Wyatt owns the mechanism and its accounting in `privacy/`.

---

## 5. Final Goals

**Near-term (system functional target).** DP trains end-to-end, is observable, and produces artifacts the Red Team can attack:

- The VFL loop runs with cut-layer noise on the clients and DP-SGD on the server.
- Cumulative privacy budget is tracked correctly (autodp) and the combined client+server epsilon is reported.
- The model is evaluated (AUC / F1 / Accuracy / Brier) via a proper VFL eval round, and training loss + validation metrics are logged to wandb.
- Trained models are exported to disk per epsilon config, in a bundle the attack code can load.

**Empirical (research contribution).**

- **Privacy-utility tradeoff:** sweep epsilon across a grid and produce the epsilon-vs-AUC curve; identify a recommended operating point.
- **Utility:** quantify how VFL + DP compares to centralized and isolated-silo baselines.
- **Red Team:** run model-inversion (and label-inference) attacks across the epsilon grid and show attack success rate falling as privacy tightens.
- **Temporal sufficiency:** identify the earliest week of VLE data that supports actionable prediction.

**Deliverable.** A reproducible codebase (one-command reproduction), a written paper covering methods/results/privacy analysis, and a SCARP symposium presentation — as scheduled in the original DOCX.

---

## 6. Reference Map

| Need                                                     | Document                                         |
| -------------------------------------------------------- | ------------------------------------------------ |
| How the system works (architecture, privacy, structure)  | **this file**                                    |
| Coding conventions, module contracts, agentic-dev rules  | `CLAUDE.md`                                      |
| What it is + how to install and run                      | `README.md`                                      |
| Current prioritized action items with owners             | `JayQuery_DP_Concern_List.md`                    |
| This week's day-by-day                                   | `JayQuery_Week_Plan.md`                          |
| Why clients are stateless                                | `JayQuery_Architecture_Decision_Client_State.md` |
| Reading lists, roles, schedule, symposium, risk register | `docs/JayQuery_Research_Plan.docx` (original)    |