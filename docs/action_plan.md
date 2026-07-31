# JayQuery — Agreed Action Plan

> Living document. Decisions we've agreed to act on, plus open questions still
> needing a call. Add new agreed actions here as they're settled.
>
> Context/background for these items: [dp_explained_for_alex.md](dp_explained_for_alex.md)
> (esp. §9 on the threshold artifact). Last updated 2026-07-27.

---

## Status legend

| Mark | Meaning |
|---|---|
| ✅ AGREED | Decided; safe to implement |
| 🟡 OPEN | Needs a decision before acting |
| ⛔ REJECTED | Considered and ruled out (kept so we don't revisit) |

---

## ✅ AGREED — A. Run-speed work

**Problem:** ~20 min per run (135 rounds ≈ 9 s/round). Actual compute is sub-millisecond
(1,153-param top model, ~92 rows/batch) — essentially all wall-clock is overhead. This blocks
the multi-seed × multi-epsilon grid we need.

**Measured, NOT the bottleneck:** `_full_trainset()` reloads the pickle + rebuilds DataLoaders
twice per round per client, but that benchmarks at **9.4 ms (silo A) / 9.6 ms (silo B)** →
only ~6 s across a 150-round run. Fix for tidiness, not speed.

**Actual bottlenecks:** (1) deployment-mode gRPC round-trips — `scripts/run_deployment.py`
spawns real `flower-superlink` + 2 `flower-supernode` processes, ≥2 message round-trips per
round (+1 on eval rounds) ≈ **300 round-trips/run**, with the ClientApp re-instantiated per
message; (2) wandb `init`/`finish` cycles in `log_client_rounds()` — ~12 network handshakes
per run at 5–10 s each ≈ **1–2 min**.

| # | Action | Expected gain | Effort |
|---|---|---|---|
| A1 | Add a `[tool.flwr.federations]` **simulation federation** to `pyproject.toml`; use it for sweeps/exploration | **5–20×** | ~10 lines |
| A2 | `num-server-rounds` 150 → 50 (best val/auc lands at round ~35) | **~3×**, and *improves DP* (σ ∝ √T) | config |
| A3 | `wandb-mode = "offline"` while iterating; `wandb sync` afterwards | 1–2 min/run | config |
| A4 | `eval-every` 5 → 10 | ~27 → 14 extra round-trips | config |
| A5 | Cache the dataset at module level (kill the per-round reload) | ~6 s/run | small patch |

Combined, A1+A2 plausibly take a 20-min run to **under a minute** — which is what makes a
5-seed × 5-epsilon grid (25 runs) feasible.

### Outcomes (2026-07-28)

| # | Status | Notes |
|---|---|---|
| A1 | ✅ **done** | ~10× achieved, but not via `pyproject.toml` — see §L. Exposed a real bug (§L2). |
| A2 | ⏸️ **deferred** | The round count is exactly what §K is reviewing; left at the current default rather than baking in a value already flagged as stale. |
| A3 | ✅ **done** | Also fixed a gap: `wandb-mode` was read **only by the client** (`fl_client/common.py`), so setting it offline left the SERVER logging online — a half-offline run. `fl_server/server_app.py` now passes `mode=` too. Default left at `"online"`; flip it per-run via `--run-config "wandb-mode='offline'"`, then `wandb sync`. |
| A4 | ❌ **not applied — recommend skipping** | See below. |
| A5 | ✅ **done** | Split cache in `fl_client/common.py`, keyed on `(load_fn, tag, processed-dir, batch-size, seed, split)`. Cold 71.5 ms → warm 0.0004 ms. **~13.6 s saved per 50-round run** (2 train materializations/round × 2 silos), ~27% of a simulation run's wall clock — far more than the ~6 s originally estimated, because that estimate was measured on the old 1,843-row dataset and the cohort is now 20,411 rows. |

**Why A4 (`eval-every` 5 → 10) should NOT be applied:**

1. **Its rationale has evaporated.** A4 was justified by deployment round-trip cost (~27 → 14
   round-trips). With A1 done, a simulation round is ~0.75 s, so eval round-trips are no
   longer a meaningful share of runtime.
2. **It actively harms §K.** Every §J run peaked at the final round; §K has to locate where
   the model *actually* converges. Halving eval frequency halves the resolution of
   `best_round` — 5 sample points across 50 rounds instead of 10.
3. **It silently changes early stopping.** Patience is counted in **eval events**, not rounds
   (`early-stop-patience = 20`). At `eval-every = 5` that is 100 rounds of patience; at 10 it
   becomes **200** — so early stopping would effectively never fire inside a 50–150 round run.

Net: a small speed gain in the mode we no longer use, paid for with worse convergence data and
altered early-stopping semantics, right when §K needs that data. **Left at 5.**

**A5 cache-key bug caught by the suite:** the first version keyed on `wandb_tag`, which two
specs can share while loading different data. `tests/test_fl_client.py` builds exactly that
situation, and the full suite caught it (passed in isolation, failed together). `load_fn` is
now part of the key.

### A1 caveats — simulation vs deployment equivalence

**Statistically equivalent, NOT bit-identical.** Same ClientApp code, handlers, messages, and
model math. But `add_gaussian_noise` draws from the global torch RNG, whose stream depends on
process lifecycle (persistent supernode process vs Ray-backed actors). Expect **a different
draw from the same distribution — equivalent to changing the seed.**

**The privacy guarantee is unaffected:** ε is computed analytically from
`sigma`/`steps`/`sample_rate`, never from the realized noise. Same σ, same clip, same ε.

Rules of use:
- Fine for sweeps and seed studies (sampling a distribution anyway).
- **Never mix simulation and deployment results in the same comparison table.**
- Validate once with a paired config (same settings both modes; metrics should land within
  each other's CI).
- Report headline/paper numbers from **one** mode consistently. Keep deployment mode for the
  final headline runs — it's the honest "real federation" story.

**Verify before trusting (real risks, not hypotheticals):**
1. **Working directory** — Ray actors may resolve relative paths differently. Confirm
   `processed-dir = "data/processed"` and `experiments/checkpoints/` land where expected.
2. **partition-id mapping** — deployment passes `--node-config partition-id=0/1` explicitly;
   simulation auto-assigns `0..N-1`. If this flipped, Silo A and Silo B would silently swap.
   Assert the mapping on the first round.

---

## ✅ AGREED — B. Privacy cost of hyperparameter tuning

**The issue.** Reported **ε = 1.68 is the guarantee for a *single training run*.** It does not
cover running many configurations, reading validation AUC, and choosing parameters based on
what we saw. Those choices leak information about the data, and standard composition (what
autodp does over the mechanisms) does not capture it. Established result — see
**Papernot & Steinke, "Hyperparameter Tuning with Renyi Differential Privacy" (ICLR 2022)**;
also Liu & Talwar (2019).

**Direction matters for the attack stage:**
- Wyatt tuning *attack* hyperparameters against the data is the **attacker's** side — it does
  not inflate our defender-side ε claim. Attackers aren't budget-constrained. Fine.
- **The dangerous loop** is tuning *defense* parameters (σ, clip norms, model size) based on
  attack results. That's a feedback channel from the data into the released model, and it is
  not in our accounting.

| # | Action |
|---|---|
| B1 | Report ε explicitly as a **per-run training guarantee**; state in Limitations that hyperparameter search is not accounted for |
| B2 | **Freeze the defense config before attack evaluation.** Tune → freeze → attack. Breaks the feedback loop cleanly and is easy to document |
| B3 | Do exploratory tuning on a **different module** than the one we report on (nearly free if the multi-module question below resolves yes) |
| B4 | Cite Papernot & Steinke in Limitations |

---

## ✅ AGREED — C. Fix the threshold/metric reporting bug

**Confirmed bug (introduced by the F1-tuning default).** On the 7/27 all-DP run the tuner
picked a threshold 2.42 std *below* the prediction mean, producing `tp=129 fp=102 tn=0 fn=0` —
**flag every student**. `precision == base_rate` exactly and `f1_binary == 2b/(1+b) == 0.7167`
exactly: the algebraic signature of the all-positive classifier.

**The apparent "improvement" over earlier runs was not real.** On this test split (129 pos /
102 neg) the two degenerate classifiers score:

| metric | ALL-NEG | ALL-POS | |
|---|---|---|---|
| accuracy | 0.4416 | 0.5584 | ← looked like a gain |
| recall | 0.0000 | 1.0000 | ← looked like a gain |
| f1_binary | 0.0000 | 0.7167 | ← looked like a gain |
| f1_macro | 0.3063 | 0.3583 | |
| **balanced_accuracy** | **0.5000** | **0.5000** | *both chance* |
| **MCC** | **0.0000** | **0.0000** | *both zero skill* |
| **cohen_kappa** | **0.0000** | **0.0000** | *both zero skill* |

Earlier DP runs matched ALL-NEG; the 7/27 run matched ALL-POS to four decimals. It was a move
between two zero-skill classifiers, while AUC actually fell 0.767 → 0.682.

**Why F1 was the wrong objective here:** base rate is **0.558 — positives are the majority**,
so trivial all-positive already scores F1 = 0.717. There is almost no headroom, and F1
optimization actively *pulls toward* the degenerate solution. F1 suits **rare**-positive
problems (fraud, screening at 1–5% prevalence), not this one.

**The bug is confined to the reporting layer** — training, the DP mechanisms, and ε were never
touched. The AUC history is uncontaminated.

| # | Action |
|---|---|
| C1 | ✅ **DONE** — default is now `youden`; plus a structural non-degeneracy guard applied to **every** objective (`_non_degenerate`), so the protection does not depend on which objective is chosen |
| C2 | ✅ **DONE** — `balanced_accuracy` and `mcc` added; both are exactly 0.5 / 0.0 for all-positive AND all-negative |
| C3 | Keep **AUC + AP as the headline** (threshold-free; unaffected by this bug) |
| C4 | Use **`recall_at_precision`** as the deployment view — immune to degeneracy. Current: precision ≥0.70 → recall **0.651** |
| C5 | Prefer tuning the threshold by **intervention capacity** ("advisors can contact 30%" → 70th percentile) over any metric-maximizing objective |

---

## ✅ AGREED — D. Cheap variance reduction (no new data required)

Two distinct noise sources need different tools: **(a) test-set sampling noise** (231 test
rows) and **(b) training stochasticity** (init, Poisson batching, DP noise draws).

| # | Action | Cost | Fixes |
|---|---|---|---|
| D1 | ✅ **DONE** — `bootstrap_ci()`; wired into the server's final test eval, emitting `auc_roc_ci_low/high/se` + same for AP | ~free, 1 run | (a) |
| D2 | **Paired comparison** across conditions on the *same* test rows — far more powerful than the unpaired ±0.092 bar currently quoted | free | (a) |
| D3 | **3–5 seeds** per config, report mean ± CI | 3–5× runtime | (b) |

⚠️ **Do NOT upsample / SMOTE.** Beyond the usual risk of manufacturing trends, duplicated
records **break the DP guarantee**: DP assumes one record per individual, so a duplicated
student pushes true sensitivity above `C` and makes the reported ε invalid. Bootstrap (D1) is
a *measurement* technique on predictions — it never touches training or creates rows.

**Current uncertainty for reference** (Hanley–McNeil, 129 pos / 102 neg): SE ≈ 0.034,
pairwise significance bar ≈ **0.092**. Every pairwise difference among the four runs falls
under it — no ordering is currently defensible.

| Run | AUC | 95% CI | AP | pred_std | ε |
|---|---|---|---|---|---|
| NO_DP | 0.723 | [0.659, 0.787] | 0.732 | 0.0343 | — |
| CUT_ONLY | 0.699 | [0.632, 0.766] | 0.718 | 0.0050 | 1.33 |
| SERVER_ONLY | 0.767 | [0.707, 0.827] | 0.762 | 0.0086 | 0.87 |
| ALL_DP (7/27) | 0.682 | [0.614, 0.750] | 0.717 | 0.0028 | 1.68 |

The one trend that *is* trustworthy because it's mechanistic rather than statistical:
**`pred_std` compression tracks noise monotonically** — 0.034 (none) → 0.009/0.005 (one
source) → 0.0028 (both). DP is provably active; it crushes confidence separation, not ranking.

---

## ✅ AGREED — F. Label validity: exclude pre-start withdrawals

**Found while investigating the dedup question (§E) — but this applies to the CURRENT
single-module data and is independent of the multi-module decision.**

`data/raw/studentRegistration.csv` carries `date_unregistration`. Cross-referencing it against
`final_result == "Withdrawn"`:

- **30.4% of ALL withdrawals in OULAD unregistered before the course started** (day ≤ 0);
  44% within two weeks; median unregistration day = 27.
- **In CCC_2014J: 213 students (8.5% of the cohort) never started**, yet are labelled
  `at_risk = 1`.

VLE engagement by outcome group (CCC_2014J):

| group | n | median clicks | % zero clicks |
|---|---|---|---|
| pass/distinction | 1015 | 1469 | 0.1% |
| fail | 406 | 510 | 2.0% |
| withdrew during course | 864 | 225 | 5.2% |
| **never started** | **213** | **0** | **66.7%** |

**Consequence:** ~8.5% of the dataset is a trivially separable freebie — zero clicks, labelled
positive. **Reported AUC is likely inflated by easy wins on students who never showed up.**

**Bonus implication:** because every model (DP and non-DP alike) classifies these correctly,
they compress the measured gap between conditions. Removing them may make the DP effect
*more* visible, not less.

| # | Action |
|---|---|
| F1 | Exclude withdrawals with `date_unregistration <= 0` from the cohort |
| F2 | Restate the prediction task as **"dropout among students who began the course."** Deployment framing supports this: an early-warning system cannot intervene on someone who unregistered before day 1 |
| F3 | Expect AUC to **drop**; that is the honest number on the task that actually matters. Re-baseline all four existing runs after this change |
| F4 | Consider a sensitivity cut at day ≤ 14 as well (44% of withdrawals) and report whether conclusions hold |

---

## ✅ AGREED — E. Multi-module data with three dedup variants

**DECIDED 2026-07-27 (Alex).** Go multi-module, generating **three dataset variants** by dedup
rule (`first` / `best` / `worst`), with §F's pre-start-withdrawal exclusion applied throughout.
`first` is the primary result; `best`/`worst` are reported as sensitivity bounds.

**Hard requirement from Alex:** the data-prep stage must be **cheap to re-run end-to-end**.
Any future change to a prep decision (a new dedup rule, a different withdrawal cutoff, a new
module filter) must be a parameter change plus one command — never a code edit scattered
across files. Build for repetition from the start.

**Consequence — the run matrix grows, so speed work is now a prerequisite, not an optimization:**

| Axis | Values | n |
|---|---|---|
| Dedup variant | first, best, worst | 3 |
| DP mode | no_dp, cut_only, server_only, all_dp | 4 |
| Seeds | (per §D3) | 3–5 |
| **Total runs** | | **36–60** |

At the current ~20 min/run that is **12–20 hours**. With §A1+A2 (~1 min/run) it is **~1 hour**.
§A is therefore promoted to a blocking dependency for §E.

### Settled design decisions (Alex, 2026-07-27)

| Decision | Choice | Rationale |
|---|---|---|
| **PSI per variant** | **Bypass during prep; PSI verifies separately.** `prepare_silos.py` computes the A∩B intersection directly and writes a per-variant `ordered_id_list.json`. The PSI protocol is run separately, once, to verify it yields the same intersection. | Each variant has a different student set, so one shared list cannot work. Coupling a multi-process protocol run into every regeneration would break the one-command requirement. **Decouples from `psi/` rather than modifying it — Wyatt's module is untouched.** |
| **Module scope** | **All 22 module-presentations.** | 28,785 unique students → ~25,000 after dedup (~11× current). Maximises the CI tightening and the DP batch-size win. Heterogeneity (at-risk 0.274–0.658) must be handled by stratifying / one-hot encoding module. |
| **§F label fix** | **Baked into all variants; cutoff is a flag** (`--withdrawal-cutoff-day`, default 0). | Keeps the matrix at 3 variants. §F4's day ≤ 14 sensitivity cut becomes a parameter change, not a code edit. |

### Regenerating the variants (implemented)

One command rebuilds everything — `scripts/prepare_variants.py` shells out to
`prepare_silos.py` once per variant:

```bash
python scripts/prepare_variants.py                        # all 3 -> data/processed/dedup_{first,best,worst}
python scripts/prepare_variants.py --rules first          # primary only
python scripts/prepare_variants.py --withdrawal-cutoff-day 14 --tag cutoff14   # §F4 sweep
python scripts/prepare_variants.py --dry-run              # print commands only
```

Behaviour: skips variants that already have a `manifest.json` (`--force` rebuilds);
**refuses to write the canonical `dedup_*` directories with non-default parameters unless
`--tag` is given**, so a sweep cannot silently overwrite the reported datasets; continues past
a failing variant and exits non-zero listing which failed; prints a comparison table from the
manifests. Full rebuild of all three ≈ **1m32s**.

Point a run at a variant with the `processed-dir` run-config key:
`flwr run . --run-config "processed-dir='data/processed/dedup_first'"`.

Current output (all 22 presentations, cutoff day 0):

| rule | cohort | intersect | train/val/test | label | label_test | dimA/dimB |
|---|---|---|---|---|---|---|
| first | 26,365 | 25,515 | 20411/2552/2552 | 0.4633 | 0.4455 | 18/26 |
| best | 26,365 | 25,520 | 20416/2552/2552 | 0.4470 | 0.4287 | 18/26 |
| worst | 26,365 | 25,512 | 20408/2552/2552 | 0.4841 | 0.4675 | 18/26 |

Label-rate spread across rules: **0.0372**. Feature dimensions are unchanged from the
single-module pipeline (`input-dim-a` 18, `input-dim-b` 26), so `pyproject.toml` needs no edit.

### Design constraints for the refactor

1. **Variant-addressable outputs.** Each variant writes a self-contained dataset; the existing
   `processed-dir` run-config key already lets the FL app point at one, so no client/server
   changes should be needed to select a variant.
2. **One command regenerates everything.** A single entry point produces all variants.
3. **Parameterized, not hard-coded.** Dedup rule, withdrawal cutoff day, and module set are
   all flags with defaults — §F4's "day ≤ 14 sensitivity cut" must need no code change.
4. **Manifest per variant.** Each variant records the exact parameters that produced it
   (rule, cutoff, modules, seed, row counts, label rate) so any result traces back to its data.
5. **Verifiable without an FL run.** Row counts, label rates, and dedup correctness must be
   checkable directly from the generated artifacts.

---

## ✅ DONE — H. Two feature-scaling bugs (found 2026-07-27, fixed)

Both were **pre-existing** and affected every result collected so far. Found while
validating the §E variants.

### H1. Train rows were scaled TWICE (the serious one)

`fit_and_scale` did:

```python
X[train_idx, c] = scaler.fit_transform(X[train_idx, c])   # writes SCALED into train
X[:, c]         = scaler.transform(X[:, c])               # transforms EVERYTHING again
```

The second line re-scaled the already-scaled training rows, so **train and val/test came
from different distributions**:

| | train std | val std | mismatch |
|---|---|---|---|
| Silo A `studied_credits` | 0.026 | 0.988 | **38×** |
| Silo B col 5 | 0.021 | 1.034 | **50×** |

Affected **1 of 18** Silo A features and **22 of 26** Silo B features. Training features were
compressed to near-constant (little signal to learn from), then evaluation presented features
an order of magnitude wider — far outside the training distribution.

**Fix:** `fit()` on train rows, `transform()` every row exactly once.

### H2. Sparse click-rate columns exploded under standardization

`StandardScaler` divides by the training std. With 22 presentations × 20 global activity types,
activity types used by only a few modules are ~all-zero for most students → near-zero training
std → the few active students get divided by it:

| | cols std>5 | cols std>50 | max std | max abs value |
|---|---|---|---|---|
| single-module (legacy) | 1 | 0 | 20.5 | 175 |
| multi-module, before fix | 6 | 2 | **720** | **84,391** |
| **after fix** | **0** | **0** | **1.00** | **116** |

**Fix:** `log1p` on structurally non-negative columns before standardizing (applied to 21 of 22
scaled columns; `start_date` is excluded automatically because it ranges negative, −25 to 238).
The mask is stored on the scaler as `log1p_mask_` and counted in `manifest.json`
(`silo_b_log1p_cols`).

### Measured effect

A 30-round no-DP in-process run on `dedup_first` **after** the fixes:

| | before (150-round runs) | after (30 rounds) |
|---|---|---|
| train loss | stuck ~0.693 = ln(2) (chance) | 0.694 → **0.613**, descending |
| val AUC | 0.753–0.814 best, after 150 rounds | **0.789 after 30** |
| pred_std | 0.0028–0.034 | **0.096** |

**Implication:** H1 is a strong candidate for the flat-loss / miscalibration behaviour diagnosed
earlier this session. **Every number collected before 2026-07-27 is suspect** and should be
re-baselined once the §D3 matrix runs — including the four runs in `docs/wandb_results/`.

---

## ✅ DONE — I. pytest silently destroyed the real PSI output

`tests/test_psi.py` calls `psi_client.run_psi()`, which unconditionally writes to the
module-level `OUTPUT_PATH = data/processed/ordered_id_list.json`. So **any `pytest tests/` run**
replaced the real 2,305-student intersection with the ~200 synthetic IDs from the test — with no
test failure to signal it.

Blast radius was limited because `silo_{a,b}.pkl` bake `intersection_ids` in, and the refactored
`prepare_silos.py` writes its own per-variant list rather than reading this one. The danger was
the pre-refactor path, which read it and would have silently produced a 200-row dataset.

- **Restored** from `silo_a.pkl`'s baked-in ids, cross-checked to equal
  `silo_a_ids ∩ silo_b_ids` and to be in lexicographic PSI order (2,305 ids, 23,249 bytes —
  byte-identical size to the original).
- **Fixed** with an autouse fixture in `tests/test_psi.py` redirecting `OUTPUT_PATH` to
  `tmp_path`. Verified by md5: the file is now unchanged across a full `pytest tests/` run.

---

## ✅ DONE — J. Post-fix re-baseline (2026-07-27)

Replaces the four **void** pre-fix runs in `docs/wandb_results/`. All on
`data/processed/dedup_first`, 50 rounds, wandb group `rebaseline-postfix`,
checkpoints in `experiments/checkpoints/rb_*`.

| config | AUC | AP | r@p60 | r@p70 | r@p80 | ε | best val | pred_std |
|---|---|---|---|---|---|---|---|---|
| no_dp | **0.8224** | 0.8189 | 0.859 | 0.716 | 0.589 | — | 0.8337 | 0.2377 |
| server_only | 0.8009 | 0.7764 | 0.870 | 0.633 | 0.455 | 1.000 | 0.8133 | 0.1120 |
| cut_only | 0.7781 | 0.7304 | 0.821 | 0.621 | 0.367 | 1.375 | 0.7920 | 0.0159 |
| all_dp | 0.7689 | 0.7268 | 0.799 | 0.639 | 0.444 | 1.674 | 0.7796 | 0.0061 |

### Before vs after (same four conditions)

| config | AUC pre-fix (void) | AUC post-fix | pred_std pre → post |
|---|---|---|---|
| no_dp | 0.723 | **0.8224** | 0.034 → **0.238** |
| server_only | 0.767 | 0.8009 | 0.009 → 0.112 |
| cut_only | 0.699 | 0.7781 | 0.005 → 0.016 |
| all_dp | 0.682 | 0.7689 | 0.003 → 0.006 |

### What changed qualitatively

1. **The ordering is now coherent.** no_dp > server_only > cut_only > all_dp. Pre-fix,
   `server_only` (0.767) nonsensically beat the no-DP ceiling (0.723).
2. **Differences are now measurable.** Test set 231 → **2,552** rows drops the Hanley–McNeil
   pairwise significance bar from **0.092 → 0.025**. Against the ceiling:
   `cut_only` −0.0443 **significant**, `all_dp` −0.0535 **significant**,
   `server_only` −0.0215 *within noise*.
3. **Nothing collapses.** `pred_collapsed = 0` AND `one_class_at_threshold = 0` for all four.
   Pre-fix every DP run tripped `one_class_at_threshold`.
4. **The §C threshold work is earning its keep.** `all_dp` recall is **0.015 at the naive 0.5
   cutoff** but **0.770 at the val-tuned threshold** — the model ranks fine and only the cutoff
   was wrong, exactly the failure §C was built to fix. F1 degeneracy is also gone: base rate is
   now 0.4455 (minority-positive), so the all-positive baseline F1 is 0.616 and all four
   configs beat it (0.690–0.711) — genuine skill, not a degenerate classifier.

### Headline finding: server DP-SGD is cheaper than cut-layer noise

`server_only` costs **−0.021 AUC (within noise) at ε = 1.0**, while `cut_only` costs
**−0.044 (significant) at ε = 1.375** — less utility lost *and* less budget spent. Cut-layer
noise corrupts the model's *inputs*; server DP-SGD only perturbs gradients. Needs multi-seed
(§D3) before publishing, but it is a real architectural result.

### ⚠️ Caveat — all four runs were TRUNCATED, not converged

Every config hit `best_val` at **round 50 (the final round) with `early_stop = False`**. The old
runs peaked ~round 35 on 1,843 rows; with 20,411 rows the model is still climbing at the cap.
**These numbers are a floor.** The §A2 argument for fewer rounds (σ ∝ √T) was derived on the
small dataset and needs revisiting before §D3 fixes a round count.

---

## ✅ DONE — L. Simulation mode (§A1) — enabled, with an open equivalence question

### L1. The planned approach did not exist in this flwr version

`[tool.flwr.federations]` in `pyproject.toml` is **not parsed by flwr 1.28** (verified: the
config parser contains no reference to it). That flow was removed. In 1.28 simulation is
launched through a `~/.flwr/config.toml` SuperLink entry whose `address = ":local:"` magic
value makes the CLI lazily start `flower-superlink --insecure --simulation`. **No
`pyproject.toml` change was needed.** Working invocation:

```bash
flwr run . local --federation-config "num-supernodes=2" --run-config "..."
```

`num-supernodes=2` is required (default 10; the ServerApp asserts exactly 2). The field is
bare `num-supernodes` — `options.num-supernodes` is rejected.

### L2. Simulation exposed a real bug: bottom models silently never trained

Simulation initially failed the replay guard in `fl_client/common.py` **every round**:

```
RuntimeError: [silo_a] Recomputed embedding != sent embedding — RNG/data-order replay is broken.
```

The guard raises *before* `optimizer.step()`, and gradients are pushed fire-and-forget, so
the server never saw the failure: **only the top model learned** while both bottom models sat
frozen. Deployment was unaffected (0 occurrences across all four §J runs).

**Root cause.** `generate_embeddings` snapshotted *process-global* torch RNG state and
`apply_gradients` restored it to reproduce the dropout mask. That assumption holds for a
persistent SuperNode process but not under Ray's actor pool.

**Fix.** Both forwards now run inside `torch.random.fork_rng()` re-seeded from a per-round
integer (`_forward_seed`, derived from run seed + `server-round`, persisted in
`context.state`). Process- and scheduler-independent, and it removes the fragile
serialize/restore of RNG blobs. **Do not reintroduce global-RNG snapshotting.**

### L3. Speed — the §A1 payoff is real

| mode | per round | 20 rounds |
|---|---|---|
| deployment | ~8 s | 3m26s |
| **simulation** | **~0.75 s** | **~15 s** |

**~10×**, at the top of the estimated 5–20× range. This is what makes a 36–60 run matrix
practical (hours → minutes).

### L5. ✅ RESOLVED (2026-07-28) — the runtimes ARE equivalent

Paired on seed, `no_dp`, T=200, `dedup_first`, run via `scripts/run_matrix.py --jobs l5`:

| seed | sim AUC | deployment AUC | diff |
|---|---|---|---|
| 42 | 0.904543 | 0.904760 | −0.000218 |
| 43 | 0.901381 | 0.901621 | −0.000240 |
| 44 | 0.905149 | 0.905458 | −0.000309 |
| **mean** | **0.903691** | **0.903946** | **−0.000255** |
| sd | 0.002023 | 0.002044 | |

**Paired difference: −0.000255** (sd 0.000048, n=3), 95% CI [−0.000332, −0.000179].
Average precision agrees too: −0.000181.

**Scale check — this is the whole point:**

| quantity | value | vs the diff |
|---|---|---|
| significance bar (n=2552) | 0.025000 | **98× larger** |
| same-mode reproducibility floor (§N) | 0.000576 | **2.3× larger** |
| DP seed variance (all_dp σ, §K) | 0.014 | **55× larger** |

The cross-mode difference is **smaller than the floor of same-mode repeatability**. Simulation
and deployment are computing the same thing.

**Statistical honesty:** the paired t is −9.3, i.e. the tiny offset is *consistent in sign*
across all three seeds and would be "significant" in a null-hypothesis sense. That is expected
and irrelevant — with n=3 and sd 5e-5 the test is detecting a systematic 2.6e-4 bias (almost
certainly FP/threading differences between the runtimes). **Statistical significance is not
practical significance here**: the effect is 98× below the smallest difference the test set
can resolve. Reporting it as "a real difference" would be a misuse of the test.

### Consequences

- **§D3 may run in simulation.** ~3 h for 60 runs instead of ~25 h+ in deployment. This is what
  L5 existed to decide.
- The §L4 rule ("never report a simulation number as a deployment number") is **lifted** for
  `no_dp`-style comparisons at this scale. Keep reporting the *headline* paper numbers from
  deployment for the honest "real federation" story, but sweeps, seed studies and the matrix
  can use simulation without a utility caveat.
- Not re-tested per DP mode. The DP configs share the identical code path and differ only by
  noise draws — already characterised in §K — so the runtime-equivalence result carries.

---

### L4. ⚠️ SUPERSEDED — the original (underpowered) equivalence attempt

One paired run (same config, same seed, 20 rounds, `dedup_first`):

| mode | AUC | AP | best_val |
|---|---|---|---|
| simulation | 0.7064 | 0.6691 | 0.7335 |
| deployment | 0.7378 | 0.7109 | 0.7541 |
| **\|diff\|** | **0.0314** | 0.0418 | 0.0206 |

ΔAUC **0.0314 exceeds** the n=2552 significance bar of **0.0250**, so equivalence is *not*
demonstrated. Equally, a systematic difference is *not* demonstrated: that bar quantifies
**test-set sampling** uncertainty, while two runs differing only in RNG stream differ by
**training stochasticity** — a separate, typically larger source (the §D (a)/(b) distinction).
Both runs were also truncated at 20 rounds, training's noisiest phase.

| # | Action |
|---|---|
| L5 | ✅ DONE — 3 seeds/mode at T=200; paired diff −0.000255, below the same-mode floor |
| L6 | ✅ N/A — no practically meaningful gap survived |
| L7 | ✅ RESOLVED — see §L5. Simulation is cleared for sweeps/matrix; headline paper numbers still come from deployment |

Verified alongside: partition mapping is correct in simulation (`silo_a.pt` first layer is
18-dim, `silo_b.pt` 26-dim — a swap would fail loudly on shape mismatch), and relative paths
like `processed-dir` resolve correctly under Ray. Both were flagged as risks; both are clear.

---

## 🔬 K PROBE RESULTS (2026-07-28) — nothing has converged

Nine simulation runs, `dedup_first`, early stopping disabled (`patience=99999`) so curves are
unbiased. **K1** = no-DP long run. **K2** = DP horizon sweep — for DP the curve is
horizon-*dependent* (σ is calibrated over `num-server-rounds`, so changing T changes the noise
on every step), so each T is its own run at the same ε budget.

### Test AUC by horizon

| T | no_dp | server_only | all_dp |
|---|---|---|---|
| 50 | — | 0.7858 | 0.7464 |
| 100 | — | 0.8708 | 0.8197 |
| 150 | — | 0.8878 | **0.7440** ← outlier |
| 200 | **0.9064** | 0.8997 | 0.8445 |

### Findings

1. **Every one of the 9 runs is truncated** — `best_round == num_rounds`, `early_stop=False`
   in all cases. **Nothing has converged, even at 200 rounds.**
2. **§J was severely under-trained, not slightly.** no_dp at 50 rounds = 0.8224; at 200 =
   **0.9064**. Every §J number understates its config by a wide margin.
3. **§A2's premise is inverted at this data size.** σ ∝ √T made long horizons look expensive,
   but that reasoning assumes a converged model. Here extra steps buy far more than the extra
   noise costs: `server_only` rises monotonically 0.7858 → 0.8997 across 50 → 200.
   **No turnover is visible in either sweep.**
4. **Server DP is ~free at an adequate horizon.** At matched T=200, `server_only` trails
   `no_dp` by only **0.0066** — well inside the 0.025 bar. At T=50 that gap was 0.037. The
   measured cost of DP *shrinks as the horizon grows*.
5. **Cut-layer noise remains the expensive mechanism.** `all_dp` at T=200 trails by
   **0.0619 — significant**. Consistent with §J.
6. **DP runs have large run-to-run variance.** The `all_dp` T=150 point (0.7440) sits ~0.075
   *below* both its neighbours and is an **outlier, not a turnover** — T=200 recovers to the
   sweep's highest value. With n=1 per point, single-seed horizon selection is unreliable for
   the DP configs. (This corrects an in-flight reading that treated T=150 as a genuine
   turnover; T=200 refuted it.)

### Consequences for K3 — superseded by the K2 probe below

The first reading ("a shared horizon is fine *because* nothing turned over") was **wrong**.
The K2 probe found a genuine turnover for `all_dp`. The shared-horizon conclusion survives,
but for a different reason — see below.

---

## ✅ K2 PROBE + K3 DECISION (2026-07-28)

Extended horizons with seeds. `no_dp` bounds the ceiling; `all_dp` is the config most likely
to turn over first. Seeds concentrated on the DP arms where the variance is.

### Complete horizon data (test AUC)

| T | no_dp | server_only | all_dp |
|---|---|---|---|
| 50 | 0.8224 | 0.7858 | 0.7464 |
| 100 | — | 0.8708 | 0.8197 |
| 150 | — | 0.8878 | 0.7440 *(5σ outlier)* |
| 200 | 0.9064 | 0.8997 | **0.8445 / 0.8442 / 0.8197** (μ 0.836, σ 0.014) |
| 400 | 0.9119 / 0.9131 | — | **0.7800** ← peaked internally @ r220 |
| 600 | 0.9164 | — | — |

### Findings

1. **`all_dp` genuinely turns over.** At T=400 it scores 0.7800 and **peaks internally at
   round 220** — the first DP run to peak before its own horizon. That is ~4σ below the T=200
   mean of 0.836. Mechanistically coherent: σ is calibrated over the full horizon, so a longer
   T raises the noise on *every* step — the model peaks both **earlier and lower**.
2. **`no_dp` plateaus around T≈400.** `nodp_T400_s43` peaked internally at r370 and
   `nodp_T600_s42` at r590. Going 200 → 600 buys only **+0.010 total**, entirely inside the
   0.025 bar.
3. **Variance is heavily config-dependent and heavy-tailed.**
   - `no_dp`: σ ≈ **0.001** (0.9119 vs 0.9131 at T=400).
   - `all_dp`: σ ≈ **0.014** at T=200 (n=3), i.e. **~14×** no-DP — and the T=150 point sits
     ~5σ out, so the distribution has a heavy tail rather than a wide body. Two close seeds do
     **not** certify stability.
4. **§D3 seed count follows from this.** At σ≈0.014 the SE of a mean over *n* seeds is
   0.014/√n: n=3 → ±0.016, n=5 → ±0.012 at 95%. The effects being resolved are ~0.02, so
   **3 seeds is marginal and 5 is the safer floor for the DP arms**; `no_dp` needs far fewer.

### K3 DECISION: shared horizon **T = 200**

Defensible for all three configs, but *not* for the reason first given:

- **`all_dp`**: T=200 is at/near its optimum — T=400 is clearly worse.
- **`no_dp`**: T=200 gives 0.9064 vs a best of 0.9164 at T=600 — a loss of 0.010, **inside the
  noise bar**, for one third the compute.
- **`server_only`**: 0.8997 at T=200, already within 0.017 of the no-DP ceiling's best — also
  inside the bar.

So the optima *do* diverge (contradicting the first reading), but they diverge into a region
where a single T=200 costs every config less than the noise floor. **Per-config horizons are
not justified** — they would complicate the comparison for sub-noise gains.

**Outstanding:** `server_only` was never probed beyond T=200, so its own turnover point is
unmeasured. It is bracketed by the other two (server DP is strictly less noisy than all_dp,
strictly more than no_dp), so T=200 is safe — but one confirmatory run at T=400 would close it.

---

## ✅ DONE — N. Model init was never seeded (reproducibility bug)

**Found while designing §L5.** `seed_everything` was **never called anywhere in `fl_client/`
or `fl_server/`** — the only `manual_seed` calls were DataLoader generators, the batch sampler,
and the §L2 forward seed, all *after* construction. Bottom and top model weights initialized
from whatever the ambient process RNG happened to be.

This violated the project's own standard, **CLAUDE.md coding standard 4**: *"Seed everything —
use `utils/reproducibility.py:seed_everything(42)` at the top of every training script."* The
utility existed and was simply never wired in.

**Measured:** identical config, identical `seed=42`, 20 rounds, no DP, run twice →
test AUC **0.721735** vs **0.726311**, a spread of **0.004576**.

### Fix — deliberately different on server vs client

| | Approach | Why |
|---|---|---|
| **Server** (`fl_server/server_app.py`) | `seed_everything(seed)` once, before `TopModel(...)` | Runs once per run, so a single global seed makes the whole run reproducible while the DP noise sequence still *varies across rounds*. |
| **Client** (`fl_client/common.py:_build_model`) | `fork_rng` + `manual_seed(seed*7919 + partition_id)` scoped to construction only | Seeding globally here would reset the RNG **on every message**, so cut-layer noise would draw the *same values every round*. Scoping makes init deterministic without touching the noise stream. |

### Result: 8× variance reduction, not bit-exact

Three runs, same config, same seed, after the fix: 0.7371907350 / 0.7370123473 / 0.7375879119
→ **spread 0.000576** (was 0.004576).

Residual ~6e-4 is most likely floating-point non-determinism from multithreaded CPU reductions,
not a remaining seeding gap. Acceptable: it is **~40× below the 0.025 significance bar** and
**~23× below the DP seed variance (σ≈0.014)**, so it is now a negligible noise floor rather
than a confound.

**Note for §K/§J:** the σ figures measured there (no_dp ≈0.001, all_dp ≈0.014) were taken with
unseeded init, so they conflate init variance with seed variance. The DP figure is dominated by
noise draws and stands; the no_dp figure was largely *this bug*.

---

## ⚠️ OPERATIONAL — the real cause of every "intermittent" simulation failure

Runs kept "completing" with **no `meta.json`** — the K2 probe (2 lost runs), and repeated
reproducibility attempts. **Root-caused 2026-07-28.** The earlier "SuperLink state
accumulation" hypothesis was **wrong**. There are three separate causes:

### O-A. `flwr-simulation` workers leak, one per run (~800 MB each)

The real killer. Each `flwr run . local` spawns a `flwr-simulation` worker that is **never
reaped**. After ~25 runs in one session:

```
30 flwr/ray processes · 25 live flwr-simulation @ ~800 MB · 11 GB / 15 GB RAM used · swap 4.0/4.0 GB FULL
ray.exceptions.OutOfMemoryError: Task was killed due to the node running low on memory.
ServerApp thread raised an exception: Failed to collect embeddings from both silos. Got positions: [1]
```

Killing them recovered **11 GB → 1.5 GB used**. On a 15 GB machine this OOMs at roughly
**15–18 consecutive simulation runs** — a 36–60 run matrix cannot complete without reaping.
(Reaped workers leave zombies with `rss=0KB`; those are harmless.)

### O-B. `flwr run --stream` returns BEFORE `meta.json` is written

Cleaning up immediately after the command returns kills the worker **mid-write**, destroying
the run's output. Must wait for the artifact, then reap.

### O-C. `pkill -f "flwr-simulation"` kills the calling shell

`-f` matches the *full command line*, and a batch script that mentions the process name matches
itself. This silently killed my own runner mid-loop (exit 144) and produced several of the
"missing meta.json" results I was chasing. **Use `pkill -x flwr-simulation`** (exact process
name) — never `-f`.

### Implemented: `scripts/run_matrix.py`

**§L5, §D3 and §M share one runner** — the machinery (run → wait → verify → reap → retry →
collect) is identical and only the job list differs, so maintaining three copies of this
fragile process handling would be the bug source, not the fix.

```bash
python scripts/run_matrix.py --jobs l5                    # sim-vs-deployment equivalence
python scripts/run_matrix.py --jobs d3 --seeds 42 43 44 45 46 --variants dedup_first dedup_best dedup_worst
python scripts/run_matrix.py --jobs d3 --dry-run          # print the plan, run nothing
python scripts/run_matrix.py --jobs l5 --resume           # skip jobs that already succeeded
```

Encodes every §O fix: reaps with `pkill -x` after each run (O-A/O-C), waits for `meta.json`
before reaping (O-B), writes a per-run log under `experiments/matrix_logs/` instead of
`/dev/null` (O-D), retries failed jobs, logs available memory per run, checkpoints results
after every job so a crash loses nothing, and `--resume` skips completed work.

The summary automatically flags the three failure modes that matter: **failed** jobs, jobs
**truncated at the horizon** (§K5: `best_round == num_rounds` with no early stop), and any
run tripping **`pred_collapsed`**.

### Required runner pattern (what the script implements)

```bash
run_sim () {
  tag=$1
  flwr run . local --federation-config "num-supernodes=2" --run-config "... run-tag='$tag'" --stream
  for i in $(seq 1 30); do [ -f experiments/checkpoints/$tag/meta.json ] && break; sleep 2; done  # O-B
  pkill -x flwr-simulation || true                                                               # O-A, O-C
  sleep 2
}
```

| # | Action |
|---|---|
| O1 | Runner **must reap `flwr-simulation` with `pkill -x`** after every run, or OOM by run ~15 |
| O2 | Runner must **wait for `meta.json`, then verify it exists, and retry on failure** — never record a silent hole |
| O3 | **Never redirect run output to `/dev/null`** in batch scripts — doing so in the K2 probe destroyed the evidence and forced a reproduction run |
| O4 | Log free memory per run so an approaching OOM is visible before it bites |

---

## ⚠️ Q. Replay-guard desync in SIMULATION only (found 2026-07-30)

**Deployment mode is clean. Simulation intermittently skips bottom-model updates.**

### What happens

`apply_gradients` must reproduce the exact embedding `generate_embeddings` sent, or the
injected upstream gradient lands on a different computation graph. A guard checks this. In
simulation it **fails intermittently**; when it does, that silo **silently skips its weight
update** for that round — gradients are pushed fire-and-forget, so the server never learns of
it, `meta.json` is still written, and `run_matrix.py` scores the run as clean.

### Measured — this is NOT a tolerance issue

| | max abs difference |
|---|---|
| when the guard passes | **0.000e+00** (bit-identical) |
| when it fails | **1.658e-02** (Silo A) / 4.841e-02 (Silo B) |

~1000x too large for floating point. Loosening the tolerance would be **wrong** — the two
computations genuinely disagree, and tolerating that means applying a gradient to a different
graph than the one that produced the embedding.

### Mechanism

- **Deterministic magnitude**, identical across independent runs (always `1.658e-02`) — rules
  out a race condition.
- **Coincides with new Ray actor PIDs.** Passing rounds run on the original workers; the
  failure appears on **freshly-spawned** workers, which then handle all later rounds. The
  *first round handled by a new worker* is the one that fails.
- **Batch is correct, weights are not.** Row counts match exactly (1012 sent / 1012 recomputed),
  so the right students are loaded — but a ~0.017 embedding difference is about the size of
  **one gradient update**, consistent with the new worker rebuilding from a model state one
  step out of sync.

### Deployment vs simulation — controlled comparison

| mode | runs | rounds | failures |
|---|---|---|---|
| **deployment** | 4 | ~614 | **0** |
| simulation | 5 | ~442 | 8, 8, 8, 4, 0 |

The decisive pair is §L5: **identical config, identical seeds, identical code** — simulation 8
failures/run, deployment 0 across three 200-round runs. Deployment gives each silo a persistent
SuperNode process, so there is no actor pool and no mid-run worker spawning.

### Impact on existing results

~8 skips per 400 gradient applications (**~2%**), concentrated early. Models still train fully
(val/auc 0.575 -> 0.913) and all 60 §D3 runs were internally consistent with 0 collapses.

**But the arms were not equally affected** — cut-noise-OFF runs (`no_dp`, `server_only`) had
skips; cut-noise-ON runs (`cut_only`, `all_dp`) had none. That biases slightly **against the
no-DP ceiling**, i.e. it would make the measured DP cost an **under**estimate. Since
`server_only` measured -0.0022 with a CI containing zero, this is the direction that matters
for the headline claim.

| # | Action | Status |
|---|---|---|
| Q1 | Deployment mode eliminates the issue | ✅ **verified** — 0 skips in ~614 rounds vs 8/run in the paired sim arm |
| Q2 | Count + log skipped updates; surface in `meta.json` | ✅ **DONE** — hard failure replaced by a counted, logged skip; `replay_skips` + `replay_skips_total` in every `meta.json`. Verified live. Tolerance deliberately NOT loosened: the mismatch is ~1.7e-2 vs 0.0 when healthy, so tolerating it would apply a gradient to a different graph |
| Q3 | Re-run §D3 | 📋 **DEFERRED (Alex, 2026-07-30) — re-run in SIMULATION, not deployment.** Skips are now reported, so a sim re-run declares its own bound; 60 deployment runs would cost ~34 h vs ~3 h. **See `docs/HANDOFF.md` §5 — do not forget this** |
| Q4 | §L5 equivalence still holds | ✅ measured with this present in the sim arm; modes still agreed to 0.000255 |

**Bias direction (why Q3 matters):** the arms were **not equally affected** — cut-noise-OFF
runs (`no_dp`, `server_only`) skipped updates while cut-noise-ON runs did not. That handicaps
the **no-DP ceiling**, so the true DP cost may be marginally **larger** than §D3 reports. With
`server_only` at −0.0022 and a CI containing zero, this is the direction that could matter for
the headline claim.

---

## ⚠️ R. Privacy-verification findings (2026-07-30)

Found while building the audit pipeline (`scripts/verify_privacy.py`,
`docs/privacy_presentation_plan.md`). Three are holes in the guarantee, one is tooling.

### R1. The per-silo accountant nondeterministically UNDER-counts by one step

Two runs, identical config, 6 rounds, simulation:

| run | server ε (steps) | silo ε (steps) |
|---|---|---|
| `audit_smoke` | 0.9988 (6) | **0.9684 (5)** |
| `audit_smoke3` | 0.9988 (6) | 0.9988 (6) |

The silo trackers reported 5 steps for a 6-round run. **Under-reporting ε is the one
direction a privacy claim cannot tolerate.** Simulation-only in signature (like §Q) but
`replay_skips_total` was 0, so it is *not* the replay-skip mechanism. Root cause not yet
found. **Mitigated, not fixed:** `verify_privacy.py` V5 now FAILs any run whose accountants
did not charge the rounds actually run, so it can never pass silently again. Re-check
whether it reproduces in deployment mode before the write-up.

### R2. Validation/test embeddings are released with NO formal protection

`fl_client/common.py:generate_eval_embeddings` clips but does **not** noise, and takes no
accountant step. At `eval-every = 5`, T = 200 that is ~40 unnoised releases of 2,552
validation students, plus the final test release. Not covered by the reported ε and not
previously written down anywhere.

Two defensible positions (pick one and state it): **(a) scope it** — "ε is a guarantee over
the *training* set; val/test embeddings are released clipped but unnoised" — or **(b) close
it** by noising eval too, which composes in *parallel* (val/test rows are disjoint from
train) and so yields a **second, separate** ε over the eval population rather than raising
the training ε. (a) is the call for this timeline; (b) is future work. Leaving it unstated
is not an option.

### R3. The attacks were not attacking the defense

`attack/` and `scripts/run_attack.py` contained **zero** references to noise, clipping,
sigma or max_norm. `run_inversion` attacked `model_a(row)` — the clean embedding — while
the wire carries `add_gaussian_noise(clip_embedding_rows(emb, S), sigma*S)`. Any ASR-vs-ε
curve built that way measures only how DP changed the *weights*, would read as near-flat,
and would **understate the defense**.

✅ **Fixed** in `scripts/run_attack.py` (no change to `attack/` — `run_inversion_attack`
already takes the target embedding as an argument, so the wire reconstruction lives in
`scripts/`). Defense parameters are read from the attacked run's own `meta.json`, never
from config defaults. Added: Wilson CIs, n=200 default (n=20 gives SE 0.112), a measured
always-guess-modal **floor**, ASR **lift** over that floor, and reconstruction RMSE.

First wire-faithful numbers (toy 3/6-round checkpoints, ε≈1):

| run | inversion ASR | floor | lift | label inference | floor |
|---|---|---|---|---|---|
| `audit_smoke2` | 0.100 | 0.150 | −0.050 | 0.998 | 0.554 |
| `audit_smoke3` | 0.050 | 0.150 | −0.100 | 0.998 | 0.554 |

Inversion against the noised wire does **worse than an uninformed attacker**; label
inference — an undefended channel by construction — leaks almost perfectly. The anchors are
what make this readable: without the floor, "ASR = 0.10" looks like 10% of students
compromised.

### R4. 176 existing checkpoints are unauditable

`meta.json` recorded no mechanism primitives (no sigma, no S, no cut-noise target, no
server `noise_multiplier`), so no existing run's ε can be recomputed from its artifact.
✅ **Fixed** — `meta.json` now carries a `mechanism` block plus a measured audit
(`row_norm_max_sent`, `noise_std_realized`) proving the mechanism ran as specified. Runs
predating it are reported as SKIP by the audit. **This is the argument for `--fresh`.**

### R6. ⚠️ Baselines were computed on the WRONG dataset — and now question the architecture

**The bug.** `baselines/run_baselines.py:main` parsed `data.processed_dir` from the YAML and
then called `load_silo_a()` / `load_silo_b()` **with no argument**, so both fell back to the
default `data/processed` — the legacy single-module pickles (1,843 train rows vs 20,411).
Every baseline number was computed on 11× less data than the VFL runs it exists to be
compared against. The config key that would have fixed it was silently ignored.

✅ **Fixed** — `processed_dir` is honoured, `--processed-dir` overrides it, the YAML default
is now `data/processed/dedup_first`, and results are written to `experiments/baselines.json`
(they previously went to wandb only, so `make_report.py` could never publish them).

**What the corrected baselines show** (`dedup_first`, single seed):

| model | data | AUC |
|---|---|---|
| xgboost | merged | **0.9382** |
| isolated_silo_b | silo B only | **0.9219** |
| mlp | merged | 0.9158 |
| logistic_regression | merged | 0.8704 |
| isolated_silo_a | silo A only | 0.6210 |
| *(VFL no_dp, T=200)* | *federated* | *0.9023* |

### ✅ FRAMING DECISION (Alex, 2026-07-30)

**This is a feasibility paper, not an accuracy-gain paper.** The claim is *"an architecture
of this shape is implementable in practice, and you lose little utility by doing so"* — not
*"VFL improves prediction on OULAD"*. Under that framing, silo B outperforming the federated
model is a **property of this dataset**, stated plainly, not a threat to the contribution.
OULAD is a FERPA proxy; a deployment with a genuine vertical split would have different
per-silo value.

**What the decision changes:** the *cost of federation* stops being incidental and becomes
the **headline utility claim**. It therefore has to be measured properly — see R6a/R6b,
which are promoted rather than dropped.

**Two consequences to handle:**

1. **Silo B alone (0.9219) beats the federated model (0.9023).** If one institution can do
   better on its own data than by federating, the architecture's premise is undercut.
   Silo A contributes 0.6210 alone and appears to *subtract* value: merged MLP (0.9158) is
   also below silo B alone.
2. **Cost of federation vs the architecture-matched centralized control:** 0.9158 − 0.9023
   = **0.0135**, which is a legitimate and reportable number.

**Before concluding anything, note the confound:** the VFL runs are horizon-truncated (4/5
`no_dp` runs still improving at T=200) and use Poisson batches at q=0.05, while the
baselines train to convergence. **This is not a like-for-like training budget**, so the
comparison is suggestive, not conclusive.

| # | Action |
|---|---|
| R6a | Re-run baselines at multiple seeds — currently n=1, so no CI |
| R6b | Establish a like-for-like budget before quoting the federation cost |
| R6c | Investigate why Silo A's 18 features do not help the merged model — feature quality, scaling, or a genuine finding about demographic vs behavioural predictors |
| R6d | Decide the paper's framing if silo B alone wins: this may become a *finding* ("behavioural data dominates; demographics add little") rather than a problem |

Auto-detected by `scripts/make_report.py:known_issues` on every report build.

### R7. Attack evaluation rebuilt (2026-07-30)

Four changes, prompted by Alex's review of what the three attacks actually measured.

**1. Inversion reported 1/18 of what it recovers.** `run_inversion_attack` optimizes a full
feature row and then read out only column 0 (`imd_band`), discarding the rest. The full
`reconstruction` is now returned and scored, giving per-attribute recovery for
**gender, disability, age band, highest education, IMD band and region** — the attributes
that actually matter for a FERPA proxy. Every rate is paired with its always-guess-majority
floor; an accuracy without its floor is unreadable (disability scores 0.425 against a
**0.900** floor, i.e. far worse than guessing).

**2. Silo B was structurally un-attackable.** `run_inversion_attack` read
`model.encoder[0].in_features`, hard-coding Silo A's attribute name, so it raised
`AttributeError` on `BottomModelB` (whose stack is `layer1`). Now walks the module tree for
the first `Linear`. Silo B's 26 behavioural VLE features are attacked for the first time.

**3. The attacker was artificially single-shot.** A real MITM watches every round; at
`batch-sample-rate` q over T rounds a student is transmitted ~q·T times (≈10 at q=0.05,
T=200), and averaging k noisy views shrinks the noise by √k. `wire_embedding` now takes
`n_observations`, exposed as `--observations`. **This is the honest version of the attack**:
composition over steps is exactly why ε grows with T, so the accounting already assumes this
adversary while the attack ignored it. Documented as an upper bound — it treats the
embedding as stationary, which holds late in training and is optimistic for the attacker
early on.

**4. Gradient monitor removed from attack results; label inference reframed.**
The monitor is a defense-side anomaly detector that measures no leakage — listing it beside
two real attacks implied otherwise. It remains available as `run_gradient_monitoring()` for
a "defenses considered" note. Label inference is kept but explicitly marked **not a defense
evaluation**: the returned activation gradient is noised by neither mechanism (server DP-SGD
perturbs *parameter* gradients), so its ~0.998 accuracy against a 0.554 floor quantifies a
known scope limit rather than testing a defense.

**Also fixed:** unconstrained L-BFGS diverges on noised targets, producing nRMSE ~1e16. The
reconstruction is now clamped to the public feature ranges — which makes the attacker
*stronger*, since a real adversary knows gender is binary — and near-constant features are
excluded from the error average instead of dividing by ~0.

### R8. `stopped_early` conflated FOUR unrelated causes (2026-07-31)

**The bug.** `stopped_early` in `meta.json` is set by four different code paths — silo
budget exhausted, server budget exceeded, combined-epsilon cap, **and ordinary validation
early stopping** — and every downstream report read it as "ran out of privacy budget".

**How it was caught.** The clip-calibration arms `Smed`/`Sp75` were flagged as 3/3
budget-truncated. They stopped at rounds 10–35 with composed epsilon **1.046–1.165 against
a 3.0 cap** — nowhere near any budget. They stopped because validation AUC stopped
improving, which is ordinary convergence behaviour, not a privacy effect. The published
caveat on those arms was therefore wrong.

✅ **Fixed** — `meta.json` now records `stop_reason` (`silo_budget` / `server_budget` /
`combined_cap` / `val_early_stop` / `null`). `results_lib.Group` exposes
`n_budget_stopped`, `n_val_stopped` and `n_stop_unknown` separately, and runs predating the
field are reported as **unknown** rather than silently assumed to be budget stops.

**Consequence for §2g:** the clip-calibration result (S=1.0 beats S=median by 0.243 AUC)
still holds directionally — but the mechanism is *"large S makes the model stop learning
early"*, not *"large S exhausts the privacy budget"*. Restate it that way.

### R9. Other pre-fresh-run fixes (2026-07-31)

| fix | why |
|---|---|
| `combined-target-epsilon` 3.0 → **10.0** | At 3.0 it bound during the epsilon sweep: eps=4 hit composed 2.981 and all 5 seeds stopped at round ~50, so that point measured under-training, not noise. It is a guardrail, not a scientific parameter; the composed epsilon actually spent is still recorded and reported |
| `results/tables/` wiped before each build | Table names carry a positional prefix, so adding an experiment renumbers them (`t3_x` → `t4_x`) and the old file survived beside the new one — two tables, same content, one stale |
| **Cost of federation generated, not hand-typed** | The feasibility framing's headline claim was computed in a throwaway one-liner outside the repo. `analyze.py:_compare_federation_cost` now emits it into `claims.csv` with a Welch CI, carrying the budget-mismatch caveat on every row |
| `reproduce_all.sh --attack-obs` (default 10) | The attack stage was running the artificially weak single-shot attacker; 10 ≈ `batch-sample-rate x rounds` (§R7) |

**Generated federation cost** (centralized minus VFL; positive = what federating costs):

| baseline | ΔAUC | 95% CI |
|---|---|---|
| xgboost | +0.0359 | [+0.0333, +0.0385] |
| isolated_silo_b | +0.0207 | [+0.0179, +0.0236] |
| **mlp (architecture-matched)** | **+0.0128** | **[+0.0102, +0.0154]** |
| logistic_regression | −0.0319 | [−0.0345, −0.0293] |
| isolated_silo_a | −0.2821 | [−0.2847, −0.2795] |

All are **upper bounds** until §R6b matches the training budgets.

### R5. Two of the five headline `all_dp` runs stopped early

Surfaced by the generated report's `early` column: `dedup_first / all_dp` shows **2/5** runs
stopped by the budget gate, i.e. truncated at ~round 50 rather than 200. The §2h ε-sweep
truncation confound is therefore **also present inside the headline §D3 matrix** and is not
currently stated in `FINDINGS.md` §2e. Check before quoting the all_dp figure.

---

## 📋 TRACKED — K. Round-count revisit ⏳ REVIEW AT END

**Owner: Alex. Deferred by decision (2026-07-27): revisit once the §D3 matrix is configured,
then review again before the write-up.** Not blocking Step 2.

**The problem.** All four §J re-baseline runs hit `best_val` at **round 50 — the final round —
with `early_stop = False`**. They were still improving at the cap, so every §J number is a
**floor, not a converged result**.

**Why the current value is stale.** §A2 set 150 → 50 rounds based on the old runs peaking
around round 35. That was measured on **1,843 training rows**. The cohort is now **20,411 rows**
(11×), with Poisson batches of ~1,020 instead of ~92, so convergence behaviour is entirely
different and the old peak-at-35 observation no longer applies.

**The tension to resolve.** Two forces pull opposite ways and must be balanced *per config*:
- **More rounds → better utility** for the no-DP ceiling (still climbing at 50).
- **More rounds → worse utility under DP**, since σ scales ~√T: `solve_noise_multiplier` sizes
  noise over the full `num-server-rounds` horizon, so a longer horizon buys more noise for
  every step. The DP configs may genuinely want a *shorter* horizon than the ceiling.

A single round count for all configs may therefore be the wrong shape — but using different
horizons per config makes the comparison harder to defend. **This is the open question.**

| # | Action |
|---|---|
| K1 | Convergence probe: one `no_dp` run at 150 rounds with early stopping active — find where it actually peaks on 20,411 rows (~15 min) |
| K2 | Same probe for `server_only`, to see whether the DP optimum sits earlier than the ceiling's |
| K3 | Decide: single shared horizon (simpler to defend) vs per-config horizon (better utility), and record the rationale |
| K4 | Re-check the §6a issue in `docs/dp_explained_for_alex.md` — reported ε must match the horizon actually run, not the one σ was calibrated for |
| K5 | **REVIEW AT END:** confirm the final matrix ran at the chosen horizon and that no config was silently truncated (`best_round == num_rounds` and `early_stop == False` is the tell) |

---

## 🎯 ATTACK-HANDOFF CHECKLIST (2026-07-28) ⏳ DO LAST

*What must be true before the build is handed to attack evaluation. **None of these gate any
other work** — corrected 2026-07-28 (Alex): the attacks sit on top of the pipeline and inject
into what is already built, so this whole section is the LAST thing to do, not a prerequisite.*

**Dependency direction (why these are genuinely last, not merely deferrable):**
- **P3 is an OUTPUT of §D3**, not an input to it — §D3 produces the checkpoint set.
- **P2 can only follow tuning** — you freeze the config that tuning selected (§B2:
  tune → freeze → attack), so it sits after §D3 by definition.
- **P1 is isolated tooling** — it touches no training, metric, or data code. Fixing it *now*,
  against checkpoints whose architecture could still shift, risks re-breaking it; fixing it
  once against frozen checkpoints is more robust.

### 🔧 HANDOFF ITEMS — 3, all deferred to the end

#### P1. `scripts/run_attack.py` cannot load ANY current checkpoint

Verified by running it against a real post-fix checkpoint. **Two independent breaks:**

```
KeyError: 'encoder.6.weight'          # run_attack.py:137
```

1. **Hard-coded layer index.** It reads `sd_a["encoder.6.weight"]` for the embedding dim, but
   the current `BottomModelA` emits from **`encoder.7`**. Note the module docstring already
   claims *"dimensions are inferred from the checkpoint tensors, not hard-coded"* — that
   claim is false for this line.
2. **`hidden_dim` is never inferred.** `BottomModelB(input_dim=…, embedding_dim=…)` takes the
   **default `hidden_dim=128`**, but our runs use **96** (`layer1.0.weight` is `(96, 26)`), so
   `load_state_dict` fails on shape mismatch even after break 1 is fixed.

**Fix:** infer embedding dim from the *last* Linear in the encoder (not a fixed index) and
infer `hidden_dim` from the first Linear's output dim — i.e. actually do what the docstring
says. Small, self-contained.

**Good news:** everything *else* in the attack path works. The data assembly loaded our
multi-module variant correctly (`assembled 2552 students, 44 features` = 18 + 26), `--processed-dir`
already accepts variant directories, and all **15 attack tests pass**.

#### P2. The defense config is not frozen (§B2)

§B2 requires **tune → freeze → attack** so that defense tuning cannot become a feedback channel
from the data into the released model. Settled so far: horizon **T=200** (§K3), primary variant
**`dedup_first`** (§E-a). **Not deliberately chosen:** `server-dp-epsilon`, `cut-noise-target-epsilon`,
`cut-noise-max-norm`, `server-dp-max-grad-norm` — these are still at defaults nobody selected on
evidence. Freezing means writing them down and not touching them again.

#### P3. No canonical checkpoint set at the frozen config

Attacks need a trained bundle **per DP mode** at the frozen config. Existing checkpoints are
probe artifacts at assorted horizons and seeds — not a coherent set. §D3 produces this.

### 🟢 NOT BLOCKING — needed for the paper, not for attacks

These affect *reported utility*, not model weights, and attacks read weights/embeddings/gradients:

| # | Item | Status | Why not blocking |
|---|---|---|---|
| C1 | Threshold objective still defaults to `f1` | ❌ not done | Threshold affects reported metrics only. **Note:** at the new base rate (~0.46, minority-positive) the all-positive F1 baseline is 0.616 and current runs score 0.69–0.71, so it is not *currently* degenerating — but it is still the wrong objective |
| C2 | `balanced_accuracy` / MCC not added | ❌ not done | Degeneracy-proof diagnostics; reporting only |
| D1 | Bootstrap CIs | ❌ not done | Reporting only |
| G | PSI verification | 📋 tracked | Validates an architecture claim |
| M | Full-check sweep | 📋 tracked | Final combined-correctness pass |

**C1/C2/D1 were in the step-3 slot of the execution order and were skipped** when work pivoted
to §K and §L. They are still owed for the write-up.

### Handoff sequence (after all other work completes)

1. **Freeze the config (P2)** — record the DP parameters §D3 selected; commit them, then stop
   touching them. This is what breaks the tuning→attack feedback loop (§B2).
2. **Fix P1** (~30 min) — infer both dims from the checkpoint; re-run the attack smoke test
   against the frozen bundles.
3. **Hand over**: frozen config + checkpoint bundles (P3, produced by §D3) +
   `run_attack.py --eps-tag <tag>`.

**No discovery risk remains here.** The smoke test already exercised the whole attack path:
data assembly loads the multi-module variants correctly (44 features = 18 + 26),
`--processed-dir` accepts variant directories, and all 15 attack tests pass. Only the loader's
hard-coded layer indices are stale — a contained fix, not an integration unknown.

---

## ⛔ DISSOLVED — M. Full-check sweep (superseded, 2026-07-28)

**Decision (Alex): dissolve §M and defer its remaining value to a future full parameter sweep
tracked in wandb.**

**Why it is no longer needed as specified.** §M was written to verify that the accumulated
refactors work *in combination*, not just in isolation. **§D3 already did that**: 60 runs over
the complete `variant × DP-mode × seed` cross-product, **0 failures, 0 `pred_collapsed`**, MCC
0.436–0.706, dimensions stable, ε within budget on every run. That is the combined-correctness
sweep §M existed to perform.

**What §D3 did NOT cover, and where it now lives:**

| Original | Status |
|---|---|
| M1 every run writes `meta.json` | ✅ done — §D3, 60/60 |
| M2 no `pred_collapsed`; `one_class_at_threshold` explainable | ✅ done — §D3, 0 collapses |
| M3 dimensions + ε within budget | ✅ done — §D3 |
| M4 no silent truncation | ✅ measured — 12/20 flagged at T=200, quantified as ~+0.010–0.014 AUC forgone (§K5); applies to all arms equally so between-arm comparisons are unaffected |
| M5 diff the three dedup variants | ✅ done — §D3 sensitivity analysis passes (DP effect moves <0.001 across rules) |
| M6 reproducibility record | ✅ done — `docs/FINDINGS.md` §2e |
| **deployment-mode spot check** | 📋 → deferred to the wandb parameter sweep |
| **broader parameter cross-product** (ε grid, clip norms, model size, horizon) | 📋 → **this is the real remaining work**, see below |

### ✅ DONE — clip calibration + ε sweep (2026-07-30)

**Clip calibration** (`experiments/clip_calibration.json`, 9 runs): **S=1.0 wins** —
0.8453 vs 0.6022 (median) vs 0.5995 (p75). Moving to the median costs **−0.243 AUC** and
inflates variance ~16×. The percentile heuristic is gradient-clipping intuition that does not
transfer to input-side noise: the embedding magnitude saturates at its natural row norm (~4.3)
while noise scales linearly with S, so SNR = min(S, norm)/(σ·S). `measure_embedding_norms.py`
guidance corrected; `run_matrix.py` defaults set to 1.0. Full write-up: `FINDINGS.md` §2g.

**ε sweep** (`experiments/eps_sweep.json`, 35 runs, wandb group `eps-sweep`, synced): the
privacy–utility curve. Usable knee at **ε≈1**. Two truncation confounds documented in
`FINDINGS.md` §2h — **ε=4 is invalid** (capped by `combined-target-epsilon = 3.0`, stopped at
round ~50) and low-ε points are budget-exhausted. `recall@precision-70` collapses far faster
than AUC (0.887 → 0.465 at ε=0.5 → 0.047 at ε=0.1).

**→ Outstanding: re-run ε > 2 with `combined-target-epsilon` raised.** The curve above ε=2 is
currently unusable.

### 📋 DEFERRED — remaining parameter sweep, tracked in wandb

Not a correctness sweep — a **parameter study**. §D3 fixed ε=1.0 per mechanism, T=200, and the
current clip norms; none of those were swept. Candidate axes:

- ~~`cut-noise-target-epsilon` grid~~ ✅ **DONE** — see above. **Still open:** the
  `server-dp-epsilon` grid (this sweep covered the cut layer only), and ε > 2 with a raised
  `combined-target-epsilon`
- ~~`cut-noise-max-norm`~~ ✅ **DONE** — calibrated 2026-07-30; S=1.0 confirmed optimal
- `hidden-dim-top` — the √d argument says model size is a strong DP lever
- horizon, now that §K has mapped its shape
- deployment-mode spot check folded in as one arm

Run through `scripts/run_matrix.py` (add a job list) with `track-wandb=true`, so the sweep is
browsable rather than living in JSON files.

---

## ✅ DONE — G. PSI verification against the generated variants (2026-07-28)

**Owner: Alex** (2026-07-27 — Wyatt is focused on the attack work, so all remaining
refinements move to Alex's list; this is a deliberate departure from the CLAUDE.md
ownership table, where `psi/` is Wyatt's).

**Why this exists.** The §E decision decoupled PSI from data prep: `prepare_silos.py`
computes the A∩B intersection directly and writes a per-variant `ordered_id_list.json`, so
regeneration stays a single command. The PSI protocol is therefore **no longer on the data-prep
critical path** — but the architecture still claims PSI is what aligns the silos, so that claim
needs to be demonstrated at least once.

| # | Action | Status |
|---|---|---|
| G1 | Run the real ECDH protocol against a variant's student set | ✅ **all three variants** via `scripts/verify_psi.py --all` |
| G2 | Assert the protocol reproduces the directly-computed intersection | ✅ **set-identical on all three**; ordering differs by convention — see below |
| G3 | Record as evidence PSI alignment is real, not assumed | ✅ this section + `docs/FINDINGS.md` |
| G4 | CI check pinning G2 | 📋 still open — would need a small fixture cohort |

### Result

| variant | Silo A ids | Silo B ids | pipeline ∩ | PSI ∩ | same set | time |
|---|---|---|---|---|---|---|
| dedup_first | 26,365 | 25,515 | 25,515 | 25,515 | ✅ | 4.6 s |
| dedup_best | 26,365 | 25,520 | 25,520 | 25,520 | ✅ | 4.5 s |
| dedup_worst | 26,365 | 25,512 | 25,512 | 25,512 | ✅ | 4.5 s |

**The architecture's alignment claim is verified against the data actually used.**

### ⚠️ Ordering convention differs (known, documented, NOT a bug)

`prepare_silos.py` sorts the intersection **numerically** (ints); `psi_client.run_psi` sorts
**lexicographically** (strings). Same students, different row order.

Both are internally consistent — **both** silos read the same file — but the two lists are
**NOT interchangeable**: the row order determines which students land in train/val/test, so
substituting one for the other silently changes the dataset. **The pipeline's numeric ordering
is canonical**; if PSI output is ever adopted directly, regenerate both silos from it.

Aligning them was considered and rejected: it would change every variant's split and invalidate
all 60 §D3 runs, for a cosmetic convention. The verification therefore asserts **set equality**
— which is what PSI must actually prove (the correct intersection is computable without either
party revealing its ID list) — and reports ordering as a documented difference.

### ⚠️ `run_psi` writes to a hard-coded global path — third encounter

`psi_client.run_psi()` returns the intersection **and** writes it to a module-level
`OUTPUT_PATH` pointing at the real `data/processed/ordered_id_list.json`. This has now
destroyed that file twice (once via `pytest` — §I; once via the first run of `verify_psi.py`).
Both times the fix was to redirect `OUTPUT_PATH` at the call site.

**A durable fix would be an `output_path` parameter on `run_psi`** so the write is opt-in rather
than an invisible side effect of a function that already returns its result. Two workarounds for
the same hazard is a signal the API is wrong, not the callers. `psi/` is Wyatt's module —
flagging rather than changing.

**Not blocking** the run matrix — it validates a claim rather than producing data. Schedule it
before the write-up, not before §D3.

---

## 🟡 SUPERSEDED — E (original open question, kept for context)

**Why it's attractive:** currently training on `CCC_2014J` only (1,843 train / 231 val /
231 test). OULAD holds **28,785 unique students across 22 module-presentations**. More data
would give ~3.7× tighter CIs, a *free DP win* (at fixed sampling rate q, larger N ⇒ larger
batch B ⇒ noise-per-example shrinks as 1/B while ε is unchanged), and would upgrade the
finding from "DP unreachable at single-course size" to a **scaling study**: at what cohort
size does DP become viable?

**Why it is NOT a config flag — the blockers we found:**

| Finding | Value |
|---|---|
| Students in >1 module-presentation | **3,538 (12.3%)**, up to 5 enrolments |
| Rows attributable to repeat students | **22.5%** |
| **CCC_2014J students who also appear elsewhere** | **1,432 / 2,498 = 57.3%** |
| At-risk rate range across module-presentations | **0.274 → 0.658** (2.4× spread) |

1. **DP correctness (most serious).** Pooling naively means 57% of the current cohort gains
   duplicate records. DP assumes one record per individual — repeat students would push true
   sensitivity above `C` and **invalidate the reported ε**. Same failure mode as upsampling.
2. **Train/test leakage.** The same student appearing on both sides inflates metrics.
3. **Heterogeneity.** A 2.4× dropout-rate spread across modules confounds any pooled result.
4. **Pipeline scope.** `scripts/prepare_silos.py` is parameterized on a single
   `module = "CCC_2014J"` throughout; PSI (`ordered_id_list.json`) would need re-running on
   the larger ID set.

**Mitigations if we proceed:** dedup to one enrolment per `id_student` (still ~25,000 unique
students ≈ 11× current data, with the one-person-one-record invariant intact — this is the
clean fix for blockers 1 and 2); stratify or one-hot encode by module (blocker 3); re-run PSI.
Encouraging sign: `prepare_silos.py:250` already keeps a **global activity-type column list
"consistent across modules"** — the pipeline was built anticipating this.

### E-a. Blocker 5 — which duplicate record do we drop? (raised by Alex)

Dropping a *completion* row and keeping a *withdrawal* row for the same person misreads a
scheduling conflict / non-serious withdrawal as a genuine dropout signal. Quantified:

- **33.0% of repeat students (1,168 of 3,538) have MIXED outcomes** — passed one module,
  withdrew from another. For these students the label *is* an artifact of the rule chosen.
- Consistently not-at-risk: 21.6%. Consistently at-risk: 45.4%.

The rule choice moves the label materially — **a 4.06-point spread, larger than most DP
effects we are trying to measure**:

| Dedup rule | n | at-risk rate |
|---|---|---|
| keep BEST (any pass) | 28,785 | 0.4935 |
| **keep FIRST (chronological)** | 28,785 | **0.5131** |
| keep LAST (chronological) | 28,785 | 0.5147 |
| keep WORST (any withdrawal) | 28,785 | 0.5340 |
| *(no dedup, all rows)* | 32,593 | 0.5280 |

FIRST and LAST land within 0.16 pt of each other and mid-range; BEST/WORST are the extremes —
evidence that chronological rules are label-neutral while best/worst are label-selecting.

**Options:**

| # | Rule | Verdict |
|---|---|---|
| S1 | **Chronological-first** (keep earliest enrolment) | ✅ **Recommended primary.** Mimics deployment (predict on a student's first course, no look-ahead), label-neutral, mid-range |
| S2 | Chronological-last | ⚠️ Look-ahead bias; its near-identity to S1 is useful corroboration |
| S3 | Best / worst | ⛔ Not as primary (this is exactly the objection). ✅ **But report both as sensitivity bounds** |
| S4 | Random | ⚠️ Unbiased in expectation; adds variance, feels arbitrary |
| S5 | Single-enrolment students only (drop all 3,538 repeaters) | ⚠️ 25,247 remain (~11× current), k=1 by construction, no rule needed — but repeaters skew at-risk (45.4% consistently), so this introduces selection bias. Note `num_of_prev_attempts` is already a feature |
| S6 | User-level DP + `GroupKFold` by `id_student` (no dedup) | ⚠️ Most principled, but needs per-student gradient clipping — significant work |

**Recommendation: S1 as primary + S3 as a reported sensitivity analysis.** Running the
pipeline under first/best/worst and showing the conclusion holds across all three converts
this objection from a weakness into a rigor argument in the paper.

**Decision needed:** is this in scope for the remaining timeline, or do we stay single-module
and report the negative result as-is? (Note §F is worth doing either way.)

---

## Suggested order of execution

**Updated 2026-07-27 after the §E decision.** §E is agreed, so the data-prep refactor leads and
§A becomes a blocking prerequisite for the run matrix.

| Step | Work | Why here |
|---|---|---|
| 1 | **§E + §F data-prep refactor** — parameterized prep, 3 variants, pre-start exclusion, manifests | Defines what every later run trains on; verifiable without an FL run |
| 2 | **§A1 + A2** (simulation federation, fewer rounds) | Blocking prerequisite: 36–60 runs is 12–20 h without it, ~1 h with it |
| 3 | **§C1 + C2 + D1** (threshold objective, degeneracy-proof metrics, bootstrap CIs) | ✅ **DONE 2026-07-28** |
| 4 | **§D1** (bootstrap CIs) | ~free, honest error bars immediately |
| 5 | ✅ **§K1–K3 DONE** — horizon settled at **T=200** (shared) | See K2 probe + K3 decision |
| 6 | **§D3** run matrix (3 variants × 4 DP modes × seeds) | Needs 1–5 in place |
| 7 | **§B2** freeze defense config → hand to Wyatt for attacks | Breaks the tuning feedback loop |
| 8 | ~~**§M** full-check sweep~~ | ⛔ **DISSOLVED** — subsumed by §D3; parameter-sweep remnant deferred to a wandb-tracked study |
| 9 | **§P1–P3** attack handoff (freeze config, fix attack loader, hand over) | ⏳ **TRUE LAST** — attacks sit on top; nothing else depends on these. P1 ✅ done |

**⚠️ Also outstanding — see `docs/HANDOFF.md`:**
- **§Q3 §D3 re-run in simulation** (deferred 2026-07-30) — the current §D3 numbers predate the
  replay-skip fix and are biased slightly against the no-DP ceiling.
- **Attack ASR at n≈200** — n=20 gives SE 0.112, which cannot resolve DP differences.
| — | **§G** PSI verification | Not blocking; validates a claim. Do before write-up |

**⏳ END-OF-PROCESS REVIEW LIST** — check these before the write-up, not before the next step:
- **§K5** — confirm no config was silently truncated (`best_round == num_rounds` with
  `early_stop == False`).
- **§L5–L7** — cross-mode equivalence settled, and no simulation number reported as a
  deployment number.
- **§G** — PSI protocol reproduces the directly-computed intersection.
- **§H** — every pre-2026-07-27 number is void; make sure none survived into the paper.
- **§B1/B4** — ε stated as a per-run guarantee, hyperparameter-tuning cost acknowledged.

**Ownership note (2026-07-27):** Wyatt is focused on the attack work, so every remaining
refinement in this document — including `psi/` verification (§G), normally his per CLAUDE.md —
is on **Alex's** list.

Execution protocol (agreed with Alex): work step by step; verify the code after each step
before moving on; re-check that the next step still makes sense; stop and report before
starting each step; ask before any mid-step change of plan or specific design decision.
