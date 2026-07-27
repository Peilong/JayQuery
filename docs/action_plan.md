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
| C1 | Change the default threshold objective off `f1` → **Youden's J** (cannot degenerate to all-positive), plus a non-degeneracy guard |
| C2 | Add **balanced_accuracy** and **MCC** to `compute_metrics` so degeneracy can never again masquerade as improvement |
| C3 | Keep **AUC + AP as the headline** (threshold-free; unaffected by this bug) |
| C4 | Use **`recall_at_precision`** as the deployment view — immune to degeneracy. Current: precision ≥0.70 → recall **0.651** |
| C5 | Prefer tuning the threshold by **intervention capacity** ("advisors can contact 30%" → 70th percentile) over any metric-maximizing objective |

---

## ✅ AGREED — D. Cheap variance reduction (no new data required)

Two distinct noise sources need different tools: **(a) test-set sampling noise** (231 test
rows) and **(b) training stochasticity** (init, Poisson batching, DP noise draws).

| # | Action | Cost | Fixes |
|---|---|---|---|
| D1 | **Bootstrap CIs** on test predictions (resample 231 rows 10k×, percentile CI) | ~free, 1 run | (a) |
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

## 📋 TRACKED — G. PSI verification against the generated variants

**Owner: Alex** (2026-07-27 — Wyatt is focused on the attack work, so all remaining
refinements move to Alex's list; this is a deliberate departure from the CLAUDE.md
ownership table, where `psi/` is Wyatt's).

**Why this exists.** The §E decision decoupled PSI from data prep: `prepare_silos.py`
computes the A∩B intersection directly and writes a per-variant `ordered_id_list.json`, so
regeneration stays a single command. The PSI protocol is therefore **no longer on the data-prep
critical path** — but the architecture still claims PSI is what aligns the silos, so that claim
needs to be demonstrated at least once.

| # | Action |
|---|---|
| G1 | Run the PSI protocol (`scripts/extract_ids_for_psi.py` → `psi.psi_server` / `psi.psi_client`) against **one** variant's student set |
| G2 | Assert the protocol's `ordered_id_list.json` matches the directly-computed one **exactly** (same IDs, same order) |
| G3 | Record the result in the paper as the evidence that PSI alignment is real, not assumed |
| G4 | Consider a CI check pinning G2 for a small fixture cohort, so the two paths cannot silently diverge |

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
| 3 | **§C1 + C2** (threshold objective, degeneracy-proof metrics) | Quick; stops bad numbers reaching the paper |
| 4 | **§D1** (bootstrap CIs) | ~free, honest error bars immediately |
| 5 | **§D3** run matrix (3 variants × 4 DP modes × seeds) | Needs 1–4 in place |
| 6 | **§B2** freeze defense config → hand to Wyatt for attacks | Breaks the tuning feedback loop |
| — | **§G** PSI verification | Not blocking; validates a claim. Do before write-up |

**Ownership note (2026-07-27):** Wyatt is focused on the attack work, so every remaining
refinement in this document — including `psi/` verification (§G), normally his per CLAUDE.md —
is on **Alex's** list.

Execution protocol (agreed with Alex): work step by step; verify the code after each step
before moving on; re-check that the next step still makes sense; stop and report before
starting each step; ask before any mid-step change of plan or specific design decision.
