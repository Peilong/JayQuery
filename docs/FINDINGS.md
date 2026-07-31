# JayQuery — Findings & Results Record

> **What this is.** A single place for (a) every bug and finding from the 2026-07-27/28 work,
> (b) the run-results table, (c) the justification for using simulation runs as real results.
>
> **Companion documents.** `docs/action_plan.md` = the decision log and open tasks.
> `docs/DATA_DICTIONARY.md` = the data contract. `docs/dp_explained_for_alex.md` = DP mechanics.
> This file is the *evidence*; those are the *decisions*, the *contract*, and the *background*.

---

## ⚠️ Read this first: every number before 2026-07-28 is void

Two pre-existing bugs (**§H1 double-scaling**, **§H2 scaling blow-up**) corrupted every result
collected before 2026-07-28, including all four CSVs in `docs/wandb_results/`. They are not
"slightly off" — H1 alone gave train and evaluation *different feature distributions*, with
train σ up to **50×** narrower than val.

Do not cite pre-2026-07-28 numbers. Do not compare against them. The re-baseline (§J) and
everything after supersedes them.

---

## 1. Bugs found and fixed

| # | Bug | Impact | Status |
|---|---|---|---|
| **H1** | `fit_and_scale` scaled **train rows twice** — wrote scaled values into train, then transformed the whole array again | Train/eval distribution mismatch up to **50×**; hit 1/18 Silo A and **22/26 Silo B** features. Training features near-constant, eval features 10× wider. Prime suspect for the flat-at-ln(2) loss | ✅ fixed |
| **H2** | Sparse click-rate columns exploded under `StandardScaler` (near-zero training σ) | Scaled σ up to **720**, max \|x\| **84,391** | ✅ fixed — `log1p` on non-negative columns first |
| **N** | `seed_everything` **never called** in `fl_client/` or `fl_server/` — model init unseeded | Identical config + seed differed by **0.0046 AUC**; violated CLAUDE.md standard 4 | ✅ fixed — variance ↓ **8×** to 0.00058 |
| **L2** | Client replayed dropout via **process-global RNG snapshot** | Worked in deployment; under Ray **every `apply_gradients` failed**, so **bottom models never trained** while the top model kept learning. Invisible — gradients are pushed fire-and-forget | ✅ fixed — seeded `fork_rng` |
| **C** | Threshold tuner used **F1**; at base rate 0.558 the trivial all-positive classifier scores F1 = 0.717 | Tuner picked "flag everyone": `tp=129 fp=102 tn=0 fn=0`, precision == base rate, zero skill | ✅ fixed — default now **Youden's J**, structural non-degeneracy guard on every objective, `balanced_accuracy`+`mcc` added, AUC/AP primary |
| **F** | `Withdrawn` label included students who **unregistered before the course started** | 8.5% of CCC_2014J labelled at-risk with **66.7% having zero VLE clicks** — trivially separable, inflating AUC | ✅ fixed — excluded; task restated |
| **E-a** | Multi-module dedup rule choice | 33% of repeat students have **mixed outcomes**; rule moves at-risk rate by **4.06 pts** | ✅ `first` primary, `best`/`worst` as bounds |
| **I** | `pytest` overwrote the **real PSI output** with 200 synthetic IDs | Any `pytest tests/` run silently destroyed `ordered_id_list.json` | ✅ fixed — autouse fixture redirects to `tmp_path` |
| **A3** | `wandb-mode` read **only by the client** | Setting offline left the **server online** — a half-offline run | ✅ fixed |
| **O-A** | `flwr-simulation` workers **leak ~800 MB each**, never reaped | OOM at ~15–18 consecutive runs; failures appear as *missing `meta.json`*, not errors | ✅ mitigated in `run_matrix.py` |
| **O-B** | `flwr run --stream` **returns before `meta.json` is written** | Cleanup immediately after kills the worker mid-write | ✅ wait-then-reap |
| **O-C** | `pkill -f flwr-simulation` matches the **caller's own command line** | A batch script naming the process kills **itself** | ✅ `pkill -x` only |

### Corrections I made to my own earlier claims

Recorded because they changed conclusions, not just wording:

1. **"`all_dp` turned over at T=150"** — wrong. T=200 recovered to the sweep's *highest* value;
   T=150 was a **5σ outlier**. The genuine turnover is at T=400.
2. **"A shared horizon is fine because nothing turned over"** — wrong premise. `all_dp` *does*
   turn over. The conclusion survives for a different reason (see §K3).
3. **"Sim and deployment are statistically equivalent"** — asserted **before** testing. The
   first paired run exceeded the significance bar; only the 3-seed §L5 test established it.
4. **"Failures come from SuperLink state accumulation"** — wrong. The cause is a per-run
   **memory leak** (O-A).

---

## 2. Results table — every run to date

All on `dedup_first` (25,515 students; test n = 2,552) unless noted.
Significance bar for pairwise AUC at this test size: **0.025**.

### 2a. §J re-baseline — T=50, deployment (post-scaling-fix, pre-seeding-fix)

| config | AUC | AP | r@p70 | ε | pred_std |
|---|---|---|---|---|---|
| no_dp | 0.8224 | 0.8189 | 0.716 | — | 0.238 |
| server_only | 0.8009 | 0.7764 | 0.633 | 1.000 | 0.112 |
| cut_only | 0.7781 | 0.7304 | 0.621 | 1.375 | 0.016 |
| all_dp | 0.7689 | 0.7268 | 0.639 | 1.674 | 0.006 |

⚠️ **Superseded by §K**: T=50 is severely under-trained. Retained only to show the pre/post
scaling-fix improvement (e.g. no_dp 0.723 → 0.822).

### 2b. §K horizon sweep — simulation

| T | no_dp | server_only | all_dp |
|---|---|---|---|
| 50 | 0.8224* | 0.7858 | 0.7464 |
| 100 | — | 0.8708 | 0.8197 |
| 150 | — | 0.8878 | 0.7440 ⚠️ 5σ outlier |
| 200 | 0.9064 | **0.8997** | 0.8445 / 0.8442 / 0.8197 |
| 400 | 0.9119 / 0.9131 | — | 0.7800 (peaked @ r220) |
| 600 | 0.9164 | — | — |

\* deployment. **Chosen horizon: T = 200** (§K3).

### 2c. §L5 cross-mode equivalence — no_dp, T=200

| seed | simulation | deployment | diff |
|---|---|---|---|
| 42 | 0.904543 | 0.904760 | −0.000218 |
| 43 | 0.901381 | 0.901621 | −0.000240 |
| 44 | 0.905149 | 0.905458 | −0.000309 |
| **mean** | **0.903691** | **0.903946** | **−0.000255** |

### 2d. Variance by config (the seeding budget driver)

| config | σ (AUC) | source |
|---|---|---|
| same-run reproducibility floor | 0.00058 | §N, post-fix |
| cross-mode (sim vs deployment) | 0.00026 | §L5 |
| no_dp seed variance | ~0.002 | §L5, n=3 |
| **all_dp seed variance** | **~0.014** | §K, n=3 at T=200 — **heavy-tailed** |

**Implication for §D3:** at σ≈0.014 the SE over *n* seeds is 0.014/√n — n=3 → ±0.016,
n=5 → ±0.012. Target effects are ~0.02, so **5 seeds is the floor for DP arms**; `no_dp`
needs far fewer. And "two seeds agreed" does **not** certify stability: at T=200 two seeds
matched to 0.0003 while a third sat 0.025 away, and the T=150 point was 5σ out.

### 2e. §D3 FULL MATRIX — 60 runs (3 variants x 4 DP modes x 5 seeds), T=200, simulation

**0 failures, 0 `pred_collapsed`, MCC 0.436-0.706 across all 60 runs.**

Absolute AUC (mean of 5 seeds):

| arm | dedup_first | dedup_best | dedup_worst |
|---|---|---|---|
| no_dp | 0.9023 | 0.9204 | 0.9131 |
| server_only | 0.9001 | 0.9176 | 0.9109 |
| cut_only | 0.8461 | 0.8548 | 0.8589 |
| all_dp | 0.8227 | 0.8356 | 0.8277 |

Gap vs the no-DP ceiling (Welch, n=5/arm; * = significant at 95%):

| arm | dedup_first | dedup_best | dedup_worst |
|---|---|---|---|
| **server_only** | −0.0022 [−0.0053, +0.0009] | −0.0027 [−0.0073, +0.0019] | −0.0021 [−0.0051, +0.0008] |
| cut_only | −0.0562 [−0.0616, −0.0508]* | −0.0656 [−0.0761, −0.0551]* | −0.0541 [−0.0606, −0.0477]* |
| all_dp | −0.0797 [−0.1033, −0.0560]* | −0.0847 [−0.1007, −0.0688]* | −0.0853 [−0.1122, −0.0585]* |

**The sensitivity analysis passes.** The dedup rule moves the label rate by **4.06 points**
(0.4935 → 0.5340) and shifts absolute AUC by up to 0.018 — yet the measured cost of server-side
DP moves by **less than 0.001** (−0.0022 / −0.0027 / −0.0021), and its CI contains zero in all
three. Every conclusion is invariant to the cohort-construction choice that was flagged as the
largest methodological risk in §E-a.

That invariance is the point: the *absolute* level depends on how you define the label, but the
*DP effect* does not. Conclusions rest on the effect, not the level.

### 2f. Headline scientific findings (now 5-seed, CI-backed)

1. **Server-side DP-SGD at ε=1.0 has NO measurable utility cost.** ΔAUC −0.0022, 95% CI
   **[−0.0053, +0.0009]** — contains zero, confirmed on all three dedup variants with 5 seeds
   each. At T=50 the same gap measured 0.037, so **an under-trained model makes DP look far
   more expensive than it is** — a confound worth flagging in the DP-utility literature.
2. **Cut-layer noise at the SAME nominal ε=1.0 costs 25× more** (−0.0562, significant).
   Cut noise corrupts the model's *inputs*; server DP-SGD only perturbs *gradients*.
   **Where you inject noise matters far more than how much.**
3. **DP crushes confidence separation, not ranking.** `pred_std` 0.365 (no_dp) → 0.006
   (all_dp) while AUC stays high. Deploy as a **ranking**, never as calibrated probabilities.
4. **§A2's premise inverts at this data scale.** σ ∝ √T argues for short horizons, but that
   assumes a converged model. Here extra steps buy far more than the extra noise costs.

---

### 2g. Cut-layer clip (S) calibration — the median heuristic is WRONG here

Cut-noise only, eps=1.0, T=200, `dedup_first`, 3 seeds:

| setting | (S_a, S_b) | mean AUC | sd |
|---|---|---|---|
| **S=1.0** (the shipped "placeholder") | (1.0, 1.0) | **0.8453** | 0.0042 |
| S=median | (5.6, 4.0) | 0.6022 | 0.0664 |
| S=p75 | (6.7, 6.3) | 0.5995 | 0.0582 |

**Moving to the calibrated median costs −0.2431 AUC and inflates variance ~16×.**

**Why the intuition fails.** The natural expectation — shared with gradient clipping — is that
clipping far below the data's scale destroys signal, so S should sit near the median. For the
cut layer that is backwards, because **signal saturates but noise does not**:

    SNR = min(S, row_norm) / (sigma * S)
      S <= min row norm  ->  SNR = 1/sigma          (flat — best, and where S=1.0 sits)
      S >  row_norm      ->  SNR = row_norm/(sigma*S)  (falls as S grows)

The embedding's magnitude is capped by the model at ~4.3, while the injected noise scales
linearly with S. At S=1.0 every row clips to length 1.0 and SNR is 1/sigma; at S=5.6 most rows
pass through unclipped at ~4.3 but carry 5.6x the noise, giving 0.77/sigma.

**Consequences:**
- `cut-noise-max-norm = 1.0`, flagged in its own docstring as an uncalibrated placeholder, is
  near-optimal — right by accident.
- `scripts/measure_embedding_norms.py` previously advised *"start S near the median (or p75)"*.
  That guidance is now corrected in the script with the measured table.
- Worth a sentence in the paper: **percentile-based clip selection is a gradient-clipping
  heuristic that does not transfer to input-side (cut-layer) noise.**

---

### 2h. ε SWEEP — the privacy–utility curve (35 runs, 5 seeds, S=1.0, T=200)

wandb group **`eps-sweep`**. Cut-layer noise only; server DP off, so this isolates the
input-side mechanism.

| ε | AUC | sd | AP | r@p70 | MCC | stopped early | composed ε |
|---|---|---|---|---|---|---|---|
| 0.1 | 0.5862 | 0.082 | 0.528 | 0.047 | 0.077 | **5/5** | 0.111 |
| 0.25 | 0.6464 | 0.105 | 0.577 | 0.121 | 0.196 | **4/5** | 0.297 |
| 0.5 | 0.7324 | 0.130 | 0.685 | 0.465 | 0.417 | 1/5 | 0.680 |
| 1 | 0.8463 | 0.005 | 0.816 | 0.827 | 0.599 | 0/5 | 1.435 |
| **2** | **0.8609** | 0.005 | 0.842 | 0.839 | 0.611 | 0/5 | 2.844 |
| 4 | 0.7805 | 0.016 | 0.745 | 0.623 | 0.434 | **5/5** | 2.981 |
| off | 0.9027 | 0.002 | 0.906 | 0.887 | 0.639 | 0/5 | — |

Cost vs the no-noise ceiling (Welch, n=5): ε=0.1 **−0.3165**, ε=0.25 −0.2563, ε=0.5 −0.1703,
ε=1 −0.0564, ε=2 −0.0418, ε=4 −0.1222. All significant.

**ε=1 reproduces §D3's `cut_only` (−0.0564 vs −0.0562)** — independent replication via a
different job list.

#### ⚠️ Two truncation confounds — do not read the curve naively

**1. ε=4 is INVALID as a noise measurement.** It scores *worse* than ε=2, which is backwards.
Cause: `combined-target-epsilon = 3.0` caps the **composed** budget across both silos. At ε=4
the composed spend reaches 2.981 and **all 5 runs stop early at round ~50** of 200. Those runs
are **under-trained, not over-noised**. ε=2 sits at composed 2.844 — close to the cap, though
it still ran full length.
→ **Either raise `combined-target-epsilon` and re-run the high end, or report only up to ε=2
and state the cap.** As it stands the ε=4 point must not be plotted as a noise effect.

**2. The low end is truncated legitimately.** At ε=0.1 all 5 runs stop at round ~21; at ε=0.25,
4/5 stop by ~60. Here the **per-mechanism budget genuinely exhausts** — the accountant working
as designed. This *is* part of the privacy cost, but it means low-ε points reflect
"noise + fewer rounds", not noise alone. State it rather than let a reader assume 200 rounds
throughout.

#### What the curve genuinely shows

1. **Variance explodes as ε falls**: sd 0.002 (off) → 0.005 (ε=1–2) → **0.130** (ε=0.5). At
   tight budgets the model is not merely worse, it is **unpredictable** — arguably the stronger
   deployment argument, since an institution retraining could get materially different results.
2. **Deployment utility collapses far faster than AUC suggests.** `recall@precision-70`:
   0.887 → 0.827 (ε=1) → 0.465 (ε=0.5) → 0.121 (ε=0.25) → **0.047** (ε=0.1). At ε=0.5 you catch
   under half of at-risk students at acceptable precision; at ε=0.25 the system is useless while
   AUC still reads 0.65. **Reporting only AUC would substantially overstate low-ε viability.**
3. **The usable knee is ε≈1.** Below it, utility and stability both degrade sharply; above it,
   gains are small and the composed-budget cap intervenes.

---

## 3. Why simulation results are acceptable (for non-headline numbers)

**Short version:** the two run modes were measured to produce the same answer to within
**0.000255 AUC** — *smaller than the same-mode reproducibility floor*. Simulation is ~10×
faster, so it is used for sweeps, seed studies and the matrix. Headline paper numbers still
come from deployment, for provenance rather than accuracy reasons.

### The two modes

| | Simulation | Deployment |
|---|---|---|
| Runtime | one local SuperLink, virtual SuperNodes (Ray) | real SuperLink + 2 SuperNode processes |
| Speed | ~0.75 s/round | ~8–10 s/round |
| Invocation | `flwr run . local --federation-config "num-supernodes=2"` | `scripts/run_deployment.py` |

They execute **identical ServerApp and ClientApp code** — the same handlers, the same messages,
the same model math. Only the transport and process topology differ.

### The evidence

Paired on seed, `no_dp`, T=200, 3 seeds (§L5, §2c above): **mean difference −0.000255**.

| quantity | value | ratio to the difference |
|---|---|---|
| significance bar (n=2552 test rows) | 0.025000 | **98×** |
| same-mode reproducibility floor (§N) | 0.000576 | **2.3×** |
| DP seed variance (§K) | 0.014 | **55×** |

**The cross-mode difference is smaller than the floor of re-running the same mode twice.** Two
deployment runs of the same config differ *more* from each other than a simulation run differs
from a deployment run.

### The statistical caveat, stated plainly

The paired t-statistic is **−9.3**, and the offset is consistent in sign across all three seeds.
A naive reading calls that "highly significant." **That reading would be wrong.** With n=3 and
sd 5e-5 the test is detecting a systematic ~2.6e-4 bias — almost certainly floating-point and
threading differences between runtimes — that sits **98× below the smallest difference the test
set can resolve**. Statistical significance is not practical significance. The t-value is
reported rather than dropped because hiding it would be the dishonest move.

### The privacy guarantee is unaffected either way

ε is computed **analytically** from `sigma`, `steps` and `sample_rate` — never from realized
noise. Same σ, same clip, same ε in both modes. The choice of runtime cannot change a privacy
claim.

### Rules of use

1. ✅ Simulation for sweeps, seed studies, the §D3 matrix, and attack development.
2. ✅ Deployment for the **headline** paper numbers — the honest "real federation" story, and
   the configuration a reader would reproduce.
3. ⚠️ **Never mix modes inside one comparison table.** Not because they disagree, but because
   provenance should be legible without a footnote.
4. ⚠️ Establishing equivalence required **fixing L2 first** — before the seeded-`fork_rng` fix,
   simulation silently never trained the bottom models. Equivalence is a property of the
   *current* code, not of simulation in general. Re-verify (`run_matrix.py --jobs l5`) after
   any change to the client message/state path.

---

## 4. Reproducing everything

```bash
python scripts/prepare_variants.py                  # 3 dataset variants + manifests
python scripts/run_matrix.py --jobs l5               # cross-mode equivalence (§L5)
python scripts/run_matrix.py --jobs d3 --seeds 42 43 44 45 46   # the reported matrix
pytest tests/ -q                                     # 194 tests
```

`run_matrix.py` encodes the §O fixes (reap, wait-then-reap, per-run logs, retry, resume,
memory logging). Do not hand-roll a batch loop — every failure mode in §1 O-A/B/C came from
exactly that.

---

## 5. Metric contract (as of 2026-07-28)

**Primary (threshold-free, immune to the §C failure):** `auc_roc`, `avg_precision` — now
reported with **bootstrap 95% CIs** (`*_ci_low`, `*_ci_high`, `*_se`) quantifying test-set
sampling uncertainty from a single run.

**Deployment view:** `recall_at_precision_{60,70,80}` — "at precision p, what share of real
dropouts do we catch?" Immune to threshold degeneracy.

**Threshold-based (only after `tune_threshold`, never at a naive 0.5):** `recall`, `precision`,
`f1_binary`, plus **`balanced_accuracy`** and **`mcc`**, which are exactly 0.5 / 0.0 for *both*
zero-skill classifiers and therefore cannot be gamed the way `accuracy` and `f1_binary` were.

**Threshold selection:** `tune_threshold` defaults to **Youden's J**, fitted on validation and
applied to test. Every objective is additionally restricted to non-degenerate thresholds, so
the protection is structural rather than a property of the objective chosen.

Verified against a replica of the §J failure (majority-positive, `pred_std` 0.0028):

| | threshold | pred_pos_rate | precision | mcc |
|---|---|---|---|---|
| unguarded F1 (the old default) | 0.4451 | **1.000** | 0.5584 == base rate | **0.000** |
| guarded Youden (current) | 0.4568 | 0.069 | 0.8750 | 0.174 |

**Health flags:** `pred_collapsed` fires only when AUC ≤ 0.55 (no ranking signal — the true
failure). `one_class_at_threshold` and `pred_low_variance` are non-fatal diagnostics; a DP model
routinely trips them while ranking perfectly well.
