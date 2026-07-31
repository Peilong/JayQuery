# Paper Topics — Findings, Trade-offs, and Claims

> **Purpose.** A writing aid. Every finding that could matter to the paper, as bullets you can
> scan while drafting. Grouped by where it would land in a write-up.
>
> **Evidence lives in** `docs/FINDINGS.md` (numbers, tables) and `docs/action_plan.md`
> (decisions, rationale). This file is the *index of what there is to say*.
>
> **Confidence tags:** 🟢 = multi-seed, CI-backed · 🟡 = measured, single-seed or n<5 ·
> 🔵 = mechanistic/definitional (no seed noise) · ⚪ = observation, not yet quantified

---

## 1. Headline claims (the paper's spine)

- 🟢 **Server-side DP-SGD at ε=1.0 costs no measurable utility.** ΔAUC vs no-DP =
  **−0.0022, 95% CI [−0.0053, +0.0009]** — the interval contains zero, with 5 seeds/arm.
  **Replicated on all three dedup variants** (−0.0022 / −0.0027 / −0.0021, every CI containing
  zero). 60 runs total, 0 failures, 0 collapses.
- 🟢 **Cut-layer noise at the same nominal ε=1.0 costs 25× more.** ΔAUC = **−0.0562**,
  CI [−0.0616, −0.0508], t=24. Both mechanisms claim ε=1.0; their utility costs are not
  remotely comparable.
- 🔵 **Why the asymmetry:** cut-layer noise perturbs the model's *inputs* (the embedding
  crossing the wire); server DP-SGD perturbs only *gradients*. Corrupting inputs damages every
  downstream computation; corrupting gradients is partially averaged away over rounds.
- 🟢 **Combined DP (both mechanisms) costs −0.0797**, CI [−0.1033, −0.0560], ε_combined=1.75 —
  roughly additive with cut-noise dominating, plus a large variance penalty (below).
- 🟢 **DP degrades *calibration*, not *ranking*.** `pred_std` falls 0.365 → 0.006 (no_dp →
  all_dp) while AUC stays 0.90 → 0.82. **Deploy as a ranking (top-K), never as calibrated
  probabilities.**

### Framing suggestion
The interesting result is not "DP costs X utility" but **"where you inject noise matters far
more than how much you inject."** Two mechanisms, identical ε, 25× different cost.

---

## 2. The privacy–utility trade-off (core experimental section)

- 🟢 **Trade-off is mechanism-dependent, not budget-dependent.** At fixed ε=1.0 per mechanism,
  cost ranges from ~0 (server) to −0.056 (cut-layer).
- 🟡 **DP's measured cost shrinks as the training horizon grows.** server_only vs ceiling:
  −0.037 at T=50 → −0.0066 at T=200 → −0.0022 (5-seed). An under-trained model makes DP look
  far more expensive than it is.
  - ⚠️ **Methodological warning worth stating:** papers reporting DP cost at a fixed, short
    horizon may systematically *overstate* it. Horizon is a confound in the DP-utility
    literature if not controlled.
- 🔵 **The σ ∝ √T intuition inverts below convergence.** Standard reasoning says a longer
  horizon buys more noise, so keep T short. That assumes a *converged* model. Below
  convergence, extra steps buy more signal than the extra noise costs — measured: server_only
  rises monotonically 0.786 → 0.900 across T=50→200 with **no turnover**.
- 🟡 **But it does eventually turn over — for the noisiest config.** `all_dp` peaks near T=200
  and *declines* by T=400 (0.845 → 0.780, peaking internally at round 220). The turnover point
  is **mechanism-dependent**: no_dp plateaus ~T=400, all_dp turns at ~T=200–400.
- 🟡 **Operating-point view for deployment.** recall @ precision 0.70: 0.882 (no_dp) → 0.877
  (server) → 0.813 (cut) → 0.769 (all_dp). The practical question — "what share of dropouts do
  we catch at an acceptable false-alarm rate" — degrades far more gently than raw AUC suggests.

---

## 2c. The privacy–utility curve (the core figure)

Cut-layer noise, 5 seeds/point, S=1.0, T=200, wandb group `eps-sweep`.

- 🟢 **The usable knee is ε≈1.** AUC: 0.903 (off) → 0.861 (ε=2) → 0.846 (ε=1) → 0.732 (ε=0.5)
  → 0.646 (ε=0.25) → 0.586 (ε=0.1).
- 🟢 **Deployment utility collapses far faster than AUC.** recall@precision-70:
  0.887 → 0.827 (ε=1) → **0.465** (ε=0.5) → 0.121 (ε=0.25) → **0.047** (ε=0.1). At ε=0.5 you
  catch under half of at-risk students at acceptable precision; at ε=0.25 the system is useless
  while AUC still reads 0.65.
  - ⚠️ **Reporting only AUC would overstate low-ε viability.** Worth stating as a methods point:
    threshold-free ranking metrics mask operational collapse.
- 🟢 **Variance explodes as ε falls**: sd 0.002 (off) → 0.005 (ε=1–2) → **0.130** (ε=0.5). At
  tight budgets the model is **unpredictable**, not merely worse — an institution retraining on
  the same data could get materially different results. Arguably the stronger deployment
  argument against aggressive budgets.
- 🟢 **ε=1 replicates §D3's `cut_only`** (−0.0564 vs −0.0562) via an independent job list.
- ⚠️ **Two truncation confounds — see `FINDINGS.md` §2h before plotting:**
  - **ε=4 is invalid**: `combined-target-epsilon = 3.0` capped the composed budget, so all 5
    runs stopped at round ~50 of 200. Under-trained, not over-noised. Do not plot as a noise
    effect; re-run with a raised cap or report only to ε=2.
  - **Low-ε points are budget-exhausted**: ε=0.1 stops at round ~21. Legitimate (the accountant
    working), but those points reflect "noise + fewer rounds", not noise alone.

---

## 3. Variance, and why it is a finding not a nuisance

- 🟢 **DP training variance is 10× the no-DP variance.** Seed-to-seed σ(AUC): no_dp 0.0021,
  server_only 0.0021, cut_only 0.0048, **all_dp 0.0229**.
- 🟢 **Variance, not mean utility, is the strongest argument against stacking both mechanisms.**
  all_dp's seed range spans **0.792–0.850** (0.058). Two runs of the same config can differ by
  more than the entire measured cost of server-side DP.
- 🟢 **all_dp variance is heavy-tailed, not merely wide.** At T=200, two seeds agreed to 0.0003
  while a third sat 0.025 away; a T=150 probe landed **5σ** out. Two agreeing seeds do **not**
  certify stability.
- 🟢 **Training variance dominates test-sampling variance ~3×.** For all_dp: bootstrap SE
  ≈0.008 (test resampling) vs seed σ 0.0229 (training). **More seeds beat a bigger test set.**
- ⚪ **Deployment implication:** a high-variance DP configuration is a reproducibility hazard for
  an institution — retraining on the same data could move who gets flagged. Worth a sentence in
  a deployment/ethics discussion.

---

## 4. Label validity and cohort construction (methods section — and a contribution)

- 🔵 **"Withdrawn" is not the same as "dropout risk."** **30.4% of all OULAD withdrawals
  unregister before the course starts.** They never began.
- 🔵 **Those non-starters are trivially separable: 66.7% have zero VLE clicks** vs 2.4% for
  everyone else. Keeping them inflates AUC with unearnable wins — the model detects
  non-enrolment, not risk.
- 🔵 **Restated task:** *"dropout among students who began the course."* Justified on deployment
  grounds — an early-warning system cannot intervene on someone gone before day 1.
- ⚪ **Possible contribution:** this is a reusable critique of OULAD-based dropout work. Papers
  using `final_result == Withdrawn` without checking `date_unregistration` may be reporting
  inflated numbers. Worth checking the literature before claiming novelty.
- 🔵 **Repeat enrolment breaks the DP unit of analysis.** 12.3% of OULAD students appear in >1
  module-presentation; **33% of those have mixed outcomes** (passed one, withdrew another).
  DP assumes one record per individual — *k* records make true sensitivity *k·C* and
  **invalidate the reported ε**.
- 🔵 **Deduplication is therefore a privacy requirement, not a data-cleaning preference.** Same
  argument forbids SMOTE/upsampling in any DP pipeline — a point worth making explicitly.
- 🟡 **Dedup rule moves the label rate by 4.06 points** (0.4935 "best" → 0.5340 "worst"), larger
  than most DP effects being measured. Handled by reporting `first` (label-neutral: selects on
  time, not outcome) with `best`/`worst` as sensitivity bounds.
- 🟢 **The sensitivity analysis PASSES — and this is a result worth reporting, not a footnote.**
  Across the three rules, absolute AUC shifts by up to 0.018 while the **measured DP cost moves
  by <0.001** (server_only: −0.0022 / −0.0027 / −0.0021). The label definition changes the
  *level*; it does not change the *effect*. State it as evidence the conclusions are not an
  artifact of a cohort-construction choice.

---

## 5. Scale effects (a scaling-study angle)

- 🔵 **Cohort size drives everything.** Single module = 2,305 students (231 test) → 22
  module-presentations = 25,515 (2,552 test).
- 🟢 **The significance bar fell from 0.092 → 0.025** (Hanley–McNeil, ~11× more test data).
  Differences that were unmeasurable became measurable — the DP-vs-ceiling comparisons in §1
  are simply *impossible* at single-course scale.
- 🔵 **More data is the one lever that improves privacy AND utility simultaneously.** At fixed
  sampling rate *q*, larger *N* ⇒ larger batch *B* ⇒ noise-per-example falls as 1/B, **while ε
  is unchanged**. Every other lever trades one against the other.
- ⚪ **Suggested framing:** "meaningful DP is unreachable at single-course cohort sizes" is a
  defensible finding *and* motivates the multi-course design. The scaling story is a stronger
  paper than a single-cohort result.

---

## 6. Metric methodology (worth a subsection — it is a genuine trap)

- 🔵 **F1 is the wrong objective when positives are the majority.** At base rate 0.558 the
  trivial all-positive classifier scores **F1 = 0.717**. Maximizing F1 selected "flag everyone":
  precision collapsed to the base rate, MCC = 0.000, zero skill.
- 🔵 **Accuracy and F1 rate the two zero-skill classifiers very differently** (all-negative:
  acc 0.442 / F1 0.000; all-positive: acc 0.558 / F1 0.717) — which is exactly how a degenerate
  model masquerades as an improvement. **Balanced accuracy = 0.5 and MCC = 0.0 for both.**
- 🔵 **A fixed 0.5 threshold is meaningless under DP.** DP shrinks logits toward zero, so a
  correctly-ranking model can put every probability below 0.5 and flag nobody. Observed:
  recall **0.015 at 0.5** vs **0.770 at a validation-tuned threshold**, same model.
- 🔵 **Recommended metric contract:** AUC/AP primary (threshold-free) · `recall@precision` for
  the deployment view · balanced accuracy + MCC for any threshold-based claim · thresholds
  tuned on validation, never fixed at 0.5.
- ⚪ **Reusable methods point:** DP-utility papers reporting accuracy/F1 at a fixed 0.5 threshold
  may be reporting threshold artifacts rather than model degradation. We observed exactly this
  and initially misdiagnosed it as model collapse.

---

## 6b. Clip calibration — a transferable methods point

- 🟢 **For cut-layer noise, a SMALL clip wins.** S=1.0 -> AUC 0.8453; S=median (5.6/4.0) ->
  0.6022; S=p75 -> 0.5995. Moving to the median costs **−0.243 AUC** and inflates seed variance
  **~16x** (sd 0.004 -> 0.066).
- 🔵 **Why: signal saturates, noise does not.** SNR = min(S, row_norm)/(sigma·S). Below the
  minimum row norm SNR is flat at 1/sigma; above it, SNR falls as row_norm/(sigma·S). The
  embedding magnitude is capped by the model (~4.3) while noise scales linearly with S.
- ⚪ **Transferable claim:** percentile-based clip selection is a **gradient-clipping** heuristic.
  It does **not** transfer to input-side (cut-layer) noise, where the quantity being clipped has
  a bounded natural scale. Our own tooling recommended the median and was wrong by 0.24 AUC.
- 🔵 Reinforces the §1 asymmetry: the two DP mechanisms differ not only in cost but in **how
  their hyperparameters must be tuned**. Treating them as interchangeable "epsilon knobs" is
  the mistake.

---

## 7. Privacy accounting — honesty items for Limitations

- 🔵 **Reported ε is a per-run training guarantee.** It does **not** cover hyperparameter search.
  Choosing configs by reading validation AUC leaks information that standard composition does
  not capture. Cite **Papernot & Steinke, ICLR 2022**; also Liu & Talwar 2019.
- 🔵 **Mitigation used:** tune → **freeze** → attack. The defense config is frozen before attack
  evaluation, so attack results cannot feed back into defense tuning.
- 🔵 **Direction matters:** the attacker tuning against the data does *not* inflate the
  defender's ε (attackers are not budget-constrained). The dangerous loop is tuning *defense*
  parameters on attack outcomes.
- 🔵 **ε is analytic, not empirical.** Computed from σ/steps/sample-rate via autodp — never from
  realized noise. Consequence: the runtime (simulation vs deployment) cannot affect a privacy
  claim.
- ⚠️ **Horizon/ε honesty:** σ is calibrated over `num-server-rounds`. If early stopping fires
  sooner, the ε *reported* is not the ε *spent*. Verify the reported horizon matches the one
  actually run.

---

## 8. Systems / reproducibility (methods appendix, or a short "engineering" note)

- 🟢 **Simulation and deployment runtimes are equivalent.** Paired on seed, ΔAUC =
  **−0.000255** — *smaller than the same-mode reproducibility floor* (0.00058). Justifies using
  the fast runtime for sweeps.
- 🔵 **A worked example of statistical vs practical significance:** that difference has t=−9.3
  and a consistent sign, so it is "significant" — while sitting **98× below** the smallest
  difference the test set can resolve. Nice teaching example if the paper has room.
- ⚪ **Silent-failure catalogue** (candidate for a reproducibility appendix — all cost real time):
  - Train rows scaled **twice** → train/eval distribution mismatch up to **50×**, model stuck
    at chance. Invisible in loss curves; looked like "DP destroys the model."
  - Client dropout replayed via **process-global RNG** → worked in one runtime, silently
    **never trained the bottom models** in the other. Gradients are fire-and-forget, so the
    server saw nothing wrong.
  - Model init **never seeded** → identical configs differed by 0.0046 AUC, contaminating every
    variance estimate.
  - `pytest` **overwrote real PSI output** with test fixtures.
  - Simulation workers **leak ~800 MB each** → OOM at ~15–18 consecutive runs, surfacing as a
    *missing output file* rather than an error.
- ⚪ **Common thread worth stating:** every one of these failed *silently* — producing plausible
  numbers rather than errors. Argues for artifact-level assertions (health flags, invariant
  checks, manifests) over trusting that a run "completed."

---

## 9. Threats to validity (be explicit — reviewers will look)

- **Horizon truncation.** T=200 is a deliberate under-train; no_dp plateaus ~T=400. Costs
  ~+0.010–0.014 AUC, applied roughly equally across arms, so *between-arm comparisons are
  unaffected*. State the bound rather than hide it.
- **Single dataset.** OULAD as a FERPA proxy; no real institutional validation.
- **Simulated silos.** The vertical split is synthetic — a real deployment's feature partition
  would differ.
- **DP parameters not exhaustively swept.** ε=1.0 per mechanism was chosen, not optimized;
  clip norms partially tuned (server `max_grad_norm` set from measured gradient norms).
- **Hyperparameter-tuning privacy cost unaccounted** (§7).
- **Attack results pending** — the privacy side of the trade-off is not yet measured.

---

## 10. Open questions / future work

- Does **attack success collapse in the same ε region where utility does**? The cleanest version
  of the paper's story — and the reason server-vs-cut asymmetry matters: if cut-layer noise
  costs 25× more utility, does it buy proportionally more attack resistance? **If not, the
  architecture should drop it.**
- **Per-mechanism ε allocation.** Given the asymmetry, spending budget on server DP-SGD and
  little/none on cut-layer noise may dominate — but only if the threat model permits it
  (cut-layer noise protects the *wire*, server DP protects the *aggregate*).
- **Where is the cohort-size threshold** at which DP becomes viable? We have 2 points
  (2.3k, 25k); a curve would be a contribution.
- **User-level vs example-level DP** for repeat enrollees, instead of dedup.
- **Calibration recovery under DP** — temperature/Platt scaling on a held-out split, to restore
  usable probabilities rather than ranking-only output.

---

## 11. Numbers worth memorizing while drafting

| Quantity | Value |
|---|---|
| Cohort (multi-module, deduped, exclusion applied) | 25,515 students; test n = 2,552 |
| No-DP ceiling (T=200, n=5) | AUC **0.9023** ± 0.0021 |
| Server DP-SGD (ε=1.0) | AUC **0.9001**; Δ −0.0022 **[−0.0053, +0.0009]** |
| Cut-layer noise (ε=1.0) | AUC **0.8461**; Δ −0.0562 [−0.0616, −0.0508] |
| Both (ε_combined 1.75) | AUC **0.8227**; Δ −0.0797 [−0.1033, −0.0560] |
| Significance bar (test n=2552) | 0.025 |
| Seed σ: no_dp / server / cut / all | 0.0021 / 0.0021 / 0.0048 / **0.0229** |
| Pre-start withdrawals (all OULAD) | **30.4%** of withdrawals; 66.7% have zero clicks |
| Students in >1 presentation | **12.3%**; 33% of those mixed-outcome |
| Dedup rule label-rate spread | **4.06 points** |
