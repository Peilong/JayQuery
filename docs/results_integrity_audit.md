# JayQuery — Results-Integrity & Completeness Audit

**Date:** 2026-07-20
**Scope:** Full top-to-bottom read of the codebase with two questions in mind:
(1) what is still hanging / non-functional / unimplemented, and (2) what could be
silently *falsifying or biasing* the results we intend to write a paper about.
**Method:** static read of every module + inspection of on-disk artifacts
(`data/processed/*.pkl`, `experiments/checkpoints/*/meta.json`). No code was changed.

> **Read this first.** The single most important finding is not a missing feature —
> it is that the one real end-to-end DP run we have on disk produced a **degenerate
> classifier** (predicts one class for every student). Several of the other findings
> below explain *why* the numbers we do have cannot yet be trusted as evidence of
> "architecture effectiveness." Treat the current results as **not paper-ready** until
> the Tier-1 items are resolved.

---

## Tier 1 — Infections that can falsify or bias the results

### 1. The real DP-VFL run collapses to a trivial classifier
**Evidence:** `experiments/checkpoints/kzsvzk2e/meta.json` — a genuine 150-round run,
`server_dp_enabled=true`, target ε=1.0, combined ε=1.82:

```
best_val_auc = 0.638
test  auc_roc = 0.597     (barely above chance)
test  precision = recall = f1_binary = 0.0
confusion: tn=102, fp=0, fn=129, tp=0   →  predicts "negative" for everyone
```

At the configured privacy level the system is **not learning a usable model** — it
degenerates to majority-class prediction. Any claim of "effectiveness of the
architecture under DP" is currently contradicted by our own artifact. This must be
understood and fixed (or honestly reported as a negative result with analysis) before
the paper. Likely contributors: noise level vs. tiny effective signal, learning-rate
schedule, `pos_weight` interacting with DP noise, and items 2/6 below.

### 2. SMOTE is applied per-silo → cross-silo row misalignment in the training set
**Location:** `scripts/prepare_silos.py:433-435`.

```python
smote = SMOTE(random_state=args.seed)
X_a_train, y_train = smote.fit_resample(X_a[idx_train], y_a[idx_train])
X_b_train, _       = smote.fit_resample(X_b[idx_train], y_a[idx_train])
```

SMOTE synthesizes minority rows by interpolating between nearest neighbors **in each
silo's own feature space**. Silo A and Silo B have different feature spaces, so the
synthetic row *i* in Silo A and synthetic row *i* in Silo B are interpolations of
**different pairs of students**. The original rows stay aligned (they come first and
identical), but every *synthetic* row is a Frankenstein pairing: `emb_a[i]` and
`emb_b[i]` no longer describe the same person, while `y_train[i]` is treated as their
shared label.

Consequences:
- The VFL top model trains on incoherent `(emb_a, emb_b)` pairs for the whole synthetic
  fraction of the training set — it is being asked to fuse signals that don't belong
  together. This directly depresses achievable VFL utility (feeds item 1).
- The **centralized "merged" baseline is contaminated the same way**:
  `baselines/run_baselines.py:merge_silo_dataloaders` concatenates the *separately
  SMOTE'd* `X_a_train`/`X_b_train`. So the "upper bound" we compare VFL against is
  itself trained on partly-fictional joint rows — it is not a valid ceiling.

Mitigating facts (good): val/test are **not** SMOTE'd (`X_full[idx_val/idx_test]`), so
evaluation rows are correctly aligned; and current data uses a 50/50 balanced train,
meaning the synthetic fraction is large (~the full minority-to-majority gap), not
negligible. **Fix direction:** SMOTE once in the *joint* concatenated feature space and
split the result back into A/B columns, or drop SMOTE in favor of the loss `pos_weight`
that the models already carry.

### 3. The inversion attack never exercises the privacy mechanism
**Location:** `attack/inversion.py:147` inside `evaluate_attack_across_epsilon`.

```python
real_embedding = model(row.unsqueeze(0))   # clean, un-clipped, un-noised
```

The client DP mechanism *is* `clip_embedding_rows` + `add_gaussian_noise` on the **sent**
embedding (`fl_client/common.py:284-289`). The attack targets `model(row)` — the clean
embedding — and never applies clip/noise. The `epsilon` argument only selects which
*checkpoint* to load; the noise defense that ε is supposed to represent is absent from
the attack entirely.

So an "ASR vs. ε" curve produced by this code measures *how training at different ε
reshaped the weights*, **not** the privacy defense. This misattributes the core
privacy–utility tradeoff the red-team experiment exists to demonstrate. The attacker's
observation must be the *same noised embedding the honest-but-curious server sees.*

### 4. Classification metrics at a fixed 0.5 threshold are unreliable
**Location:** `baselines/eval_utils.py:40` (`y_pred = (y_pred_prob >= 0.5)`), used for
both baselines and VFL eval.

With `pos_weight` rebalancing and DP noise, the model's calibrated operating point is not
0.5. Item 1's run shows `f1=precision=recall=0` (everything predicted negative) while AUC
is ~0.6 — the threshold metrics are collapsing independently of ranking quality. For the
paper, rely on **threshold-independent** metrics (AUC-ROC, average precision) or tune the
threshold on the validation set; do not quote the 0.5-threshold F1/precision/recall as-is.

### 5. Default data prep leaks the label — verify every result used de-leaked data
**Location:** `scripts/prepare_silos.py` (`--cutoff_week` default `None`).

With no cutoff, Silo B aggregates VLE/assessment features over the **full ~39-week
course**, which encodes the outcome (a withdrawn student shows zero activity for the rest
of the term). The script itself warns about this.

- **Good:** the live `data/processed/silo_b.pkl` was built with `cutoff_week=5`
  (verified from the pickle's `cutoff_week` provenance field), so the *current* data is
  de-leaked.
- **Risk:** `scripts/reproduce_all.sh` and `scripts/run_deployment.py`'s documented prep
  step both call `python scripts/prepare_silos.py` **with no `--cutoff_week`** — anyone
  reproducing "from scratch" regenerates *leaky* data and gets inflated numbers.
- **Stale artifact:** `docs/baseline_results.md` still reports the pre-fix leaky numbers
  (XGBoost merged **AUC 0.997**, labeled "Week 3"). These must not be cited; they are the
  data-leakage results the project already identified and moved away from (commit
  `f90d881`, "Baselines had leaked data").

### 6. The client-side DP guarantee is unsound with BatchNorm in the bottom models
**Location:** `silo_a/bottom_model_a.py` (`BatchNorm1d` ×2) and
`silo_b/bottom_model_b.py` (`BatchNorm1d` ×2).

The Gaussian-mechanism accounting for the cut-layer noise assumes the sent embedding is a
**per-row** function whose sensitivity is bounded by the row-wise L2 clip
(`clip_embedding_rows`). BatchNorm couples rows within a batch: changing one student's
features shifts the batch mean/variance and therefore perturbs **every** row's embedding.
On neighboring datasets (one student added/removed) the released batch of embeddings can
change by far more than the single-row clip bound, so the stated per-step ε for the client
mechanism is **not rigorously valid**. This is a real DP-correctness issue for a *privacy*
paper, not just a utility detail.
*(The server DP-SGD path is unaffected on this point — the top model has no BatchNorm — but
note Opacus also generally disallows BatchNorm for the same reason.)*

### 7. Server DP-SGD does not privatize the gradients returned to the clients
**Location:** `fl_server/server_app.py:213-227`. DP-SGD (Opacus) clips and noises the
**top-model parameter** gradients only. The quantities sent *back down the wire*,
`grad_a`/`grad_b` (`emb_a.grad`, `emb_b.grad`), are the **raw** input-space gradients — no
clipping, no noise, not tracked by any accountant.

That backward gradient is exactly the surface the label-inference attack targets
(`attack/label_inference.py`) and it crosses the protected inter-silo channel in the
threat model. So (a) the reported **combined epsilon does not account for a released
quantity**, and (b) the system offers no formal protection on the gradient return path.
This should be stated explicitly in the threat-model section or closed with a mechanism.

---

## Tier 2 — Non-functional / unimplemented (blocks the paper's experiments)

| Item | Location | State | Impact |
|---|---|---|---|
| Inversion attack loader is incompatible with real checkpoints | `attack/inversion.py:33-49,132` | **Broken** | Attack pipeline cannot run end-to-end (details below) |
| Label-inference attack | `attack/label_inference.py:40` | `raise NotImplementedError` | Threat-model deliverable missing |
| Gradient-norm monitor (defense) | `attack/gradient_monitor.py:42` | `raise NotImplementedError` | Defense deliverable missing |
| `reproduce_all.sh` Exps 2–5 | `scripts/reproduce_all.sh:30,42,48,54` | All **stubbed** | No-DP baseline, ε-grid, attacks, temporal sweep not reproducible via the entry point |
| No-DP "upper bound" VFL run | `experiments/checkpoints/eps_inf/meta.json` | `num_rounds=1` smoke test | The real no-DP reference run does not exist yet |
| `exp02/exp03` YAML configs | `experiments/configs/` | Stale, **not wired** | Old dims (12/22), module BBB, single-privacy block; nothing consumes them |
| BottomModelB CNN variant | `silo_b/bottom_model_b.py:49` | `NotImplementedError` | Optional; only if `use_cnn=True` |

**Detail — why the attack pipeline is dead on arrival:** `load_checkpoint` expects
`ckpt["bottom_model_a_state_dict"]` with `input_dim=12, embedding_dim=32`, loaded from a
file named `model.pt`. The checkpoints the trainer actually writes
(`fl_client/common.py:_save_checkpoint`) are **bare `state_dict` OrderedDicts** (verified:
top-level keys are `encoder.0.weight`, …, no wrapper key), the first Linear is
`(32, 18)` → **input_dim=18**, embedding_dim was **64** in the real runs, and the files
are named **`silo_a.pt` / `silo_b.pt` / `top.pt`**, not `model.pt`. Every one of these
mismatches raises before the attack even starts. `run_inversion_attack` itself is written,
but it has never been exercised against a real artifact.

**Also verify (not read line-by-line in this pass):** `psi/` (`ecdh_utils.py`,
`psi_client.py`, `psi_server.py`) still carry `Status: SCAFFOLD — implement … Week 3`
headers and TODO markers. `data/processed/ordered_id_list.json` exists, so PSI apparently
ran at least once, but the stale headers mean correctness of the intersection should be
confirmed — the entire cross-silo alignment invariant depends on it.

---

## Tier 3 — Correctness / hygiene (lower severity, worth noting)

- **BatchNorm running stats are double-updated per round.** `generate_embeddings` and
  `apply_gradients` each run a `train()`-mode forward on the same batch
  (`fl_client/common.py:271` and `:392`), so BN running mean/var advance twice per round.
  Eval uses those running stats, so they track ~2× the intended momentum. Minor bias, but
  it means train- and eval-time normalization drift apart faster than nominal.
- **Opacus `expected_batch_size` is fixed while the actual Poisson batch fluctuates.**
  `server_app.py:180` sets it once from `sample_rate * n_total`; the realized batch size
  varies each round. The DPOptimizer divides the summed noised gradient by this constant,
  so per-step effective gradient magnitude wobbles with the actual/expected ratio.
  Accounting is fine (it uses the nominal `sample_rate`), but optimization is slightly
  noisier than intended — a possible contributor to item 1.
- **Tiny single-seed test set.** Test n≈231 (verified). AUC/AP estimates have wide
  confidence intervals; the paper should report multiple seeds and CIs rather than point
  estimates from one run.
- **`eps_inf` naming vs. reality.** The no-DP checkpoint dir exists but holds a 1-round
  run; don't let its presence imply a completed no-DP baseline.

---

## Recommended pre-paper checklist (priority order)

1. **Fix SMOTE alignment** (item 2): resample in the joint feature space, or drop SMOTE
   for `pos_weight`. Re-run baselines *and* VFL afterward — this touches every number.
2. **Diagnose the DP collapse** (item 1): sweep noise/LR/`pos_weight`; confirm the model
   can beat majority-class at the target ε, or write it up as an analyzed negative result.
3. **Rebuild the attack pipeline** (items 3, Tier-2): load bare `state_dict`s with correct
   dims/filenames, and — critically — attack the *clipped+noised* embedding so ε actually
   enters the experiment.
4. **Report threshold-independent metrics** (item 4); tune the decision threshold on val
   if a point classifier is needed.
5. **Lock the de-leaked data path** (item 5): make `--cutoff_week` mandatory (or defaulted)
   in the prep/reproduce scripts; delete/annotate `docs/baseline_results.md`'s leaky table.
6. **State the DP scope honestly** (items 6, 7): either close the BatchNorm-sensitivity and
   gradient-return gaps, or scope the formal guarantee precisely in the threat model.
7. **Produce the missing runs**: a real multi-round no-DP upper bound, the full ε-grid, and
   implement label-inference + gradient monitor if the paper claims them.
8. **Verify PSI correctness** and refresh its scaffold docstrings.
