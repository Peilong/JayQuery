# Presenting the Privacy Guarantee — structure, figures, and what must be fixed first

> **Purpose.** How to present "the privacy guarantee changes with parameters" in the paper,
> given what is currently *proven*, what is *measured*, and what is *unprotected*.
>
> Companions: `FINDINGS.md` (evidence) · `action_plan.md` (decisions) · `paper_topics.md`
> (claim index). Written 2026-07-30.

---

## 0. Why it currently feels shaky — the honest diagnosis

Three distinct problems, only one of which is a presentation problem:

1. **ε is an input, not an output.** Every experiment *sets* `cut-noise-target-epsilon`, solves
   σ from it, and then reads utility. Plotting "ε vs utility" is therefore a plot of a knob
   against a consequence — nothing in it *verifies* that the mechanism delivers the claimed ε.
   Nothing is wrong with the accounting; the issue is that no artifact demonstrates it.
2. **There is no empirical privacy axis.** The attack numbers cannot currently move with ε
   (§3 below — the attacks never see the DP'd wire), so the privacy side of the
   "privacy–utility trade-off" is presently unmeasured. A trade-off plot with one measured axis
   is an ablation, not a trade-off.
3. **The composed ε mixes two adversaries.** Cut-layer noise defends the wire against the
   honest-but-curious server/MITM. Server DP-SGD defends the *released top model* against
   whoever receives it — an adversary the server itself is not. Adding them into one
   `combined_epsilon` is arithmetically conservative but semantically muddled, which is exactly
   what makes a reviewer (and you) distrust the number.

Fixing (1) and (3) is presentation + a cheap verification script. Fixing (2) is real work.

---

## 1. The structure to use: three layers, stated separately

Do not present one ε. Present a **ladder**, and be explicit about which rung each claim sits on.

| Layer | Claim type | Status | Where it lands |
|---|---|---|---|
| **L1 — Proven** | Analytic DP: parameters → ε, given the mechanism and the unit of privacy | ✅ solid, needs verification artifacts | Methods + Fig B, Table 1 |
| **L2 — Measured** | Empirical leakage: attack success vs ε | ❌ not yet valid (§3) | Results + Fig C |
| **L3 — Uncovered** | What ε explicitly does *not* protect | ⚠️ partly undocumented (§4) | Threat model + Limitations |

The paper's credibility comes from L3 being explicit. Reviewers distrust a guarantee that
claims everything; they trust a fenced one.

---

## 2. L1 — the parameter → guarantee map (this is your strongest, most under-used result)

### 2.1 Which knobs actually move ε

Verified against `privacy/privacy_accountant.py` and `fl_client/common.py:337-395`:

| Parameter | Enters ε? | Effect | Evidence you already have |
|---|---|---|---|
| σ (noise multiplier) | **Yes** | ε ↓ as σ ↑ | analytic |
| T (`num-server-rounds`) | **Yes** | at fixed ε, σ ∝ ~√T | §K horizon sweep |
| q (`batch-sample-rate`) | **Yes** | subsampling amplification; ε ↓ as q ↓ | analytic, unswept |
| δ | **Yes** | fixed at 1e-5 throughout | — |
| **S (`cut-noise-max-norm-a/b`)** | **NO — cancels exactly** | pure utility knob | **§2g: +0.243 AUC at identical ε** |
| **C (`server-dp-max-grad-norm`)** | **NO — cancels exactly** | pure utility knob | unswept |
| N (cohort size) | No (at fixed q) | larger batch ⇒ less noise per example | §5 scaling argument |

The cancellation is structural, not approximate: `solve_noise_multiplier` never receives S, and
`noise_std = sigma * max_norm` while the accountant is handed the *normalized* `sigma`. So S
divides out of the guarantee entirely.

### 2.2 The framing this unlocks — "the ε-isocline"

**This is the sharpest privacy statement in the project and it is currently filed as a
calibration footnote.**

> At a fixed, identical formal guarantee of ε = 1.0, measured utility ranges from
> **AUC 0.60 to 0.90** depending on *where* the noise is injected (cut layer vs server) and on
> clip parameters that **do not appear in ε at all**.

Concretely, every one of these is ε = 1.0:

| config | ε | AUC | source |
|---|---|---|---|
| server DP-SGD only | 1.0 | 0.9001 | §D3 |
| cut noise, S = 1.0 | 1.0 | 0.8461 | §D3 / §2h |
| cut noise, S = median | 1.0 | 0.6022 | §2g |
| cut noise, S = p75 | 1.0 | 0.5995 | §2g |

**The consequence to state:** ε is not a sufficient description of a DP deployment. Two systems
reporting ε = 1.0 can differ by 0.30 AUC. Papers that report only ε are under-specifying their
mechanism. That is a methods contribution, and it converts your "shaky-feeling" ε into the
paper's point rather than its weakness.

### 2.3 Verification artifacts that make L1 non-circular

Cheap, and they are what turns "we assert ε" into "we verified ε". Each should emit a
pass/fail row in the reproduce report (§6):

| # | Check | Why it matters |
|---|---|---|
| V1 | **Accountant cross-check** — recompute ε from the same (σ, q, T, δ) with an independent library (Opacus RDP / `dp_accounting` PLD) and report all three | Proves ε is not an artifact of one library's bound |
| V2 | **Sensitivity enforced** — max L2 row norm of the *sent* embedding ≤ S, over a full run | The DP proof assumes it; measure it |
| V3 | **Noise realized** — measured std of (sent − clipped) / (σ·S) ≈ 1.0 | Proves the mechanism ran with the σ the accountant charged for |
| V4 | **Unit of privacy** — one row per `id_student`; no student in two splits | dedup is a *privacy* requirement, not cleaning (§E-a) |
| V5 | **Horizon honesty** — steps actually taken == steps σ was calibrated over, else ε recomputed on actual steps | §7 of `paper_topics.md`; matters for every early-stopped low-ε run |
| V6 | **Budget gate fired** — runs that hit the cap stopped, and the recorded composed ε ≤ target | Already implemented; just needs reporting |

V1–V3 do not exist yet. V5 partially exists (`epsilon_spent` is recomputed from actual steps,
so this is likely already honest — confirm and *say so*, it is a point in your favour).

### 2.4 What `meta.json` must record (currently missing)

For a privacy claim to be reproducible, the run artifact must carry the mechanism primitives.
Today `meta.json` records `epsilon` (null when server DP is off), `epsilon_spent`, `delta`,
`batch_sample_rate`, and nothing else about the mechanism. **Missing:** `cut_noise_enabled`,
`cut_noise_target_epsilon`, `cut_noise_max_norm_a/b`, the solved σ per silo, the server's
`noise_multiplier` and `max_grad_norm`, and the step counts each accountant actually charged.
Without these, no reader — including us — can recompute ε from an artifact.

---

## 3. L2 — the empirical axis, and why it is currently invalid

**`attack/` contains zero references to noise, clipping, σ, or max_norm** (verified by grep
across `attack/*.py` and `scripts/run_attack.py`). The consequences:

- `run_inversion` attacks `model_a(row)` — the **clean** embedding. The wire carries
  `add_gaussian_noise(clip_embedding_rows(emb, S), σ·S)`.
- `run_label_inference` backprops a **clean** loss to get `grad_emb_b`. The wire carries the
  DP-SGD-noised, clipped gradient.

So today's attack numbers measure only how DP changed the trained *weights* — not the defense.
An ASR-vs-ε curve built this way would be near-flat and would **understate the defense**, which
is the worst possible direction: it would read as "cut-layer noise costs 25× the utility and
buys nothing," and that conclusion would be an artifact of the harness.

**Requirements before any attack number goes in the paper:**

1. Attacks consume the **wire representation**, reconstructed from the run's own recorded
   (S, σ) — not from re-derived defaults.
2. **n ≥ 200** with Wilson/bootstrap CIs. At n = 20 the SE is 0.112; nothing at ε-scale
   resolves against that.
3. **Both anchors on every plot**: the random-guess floor (what does ASR mean for a 12-band
   `imd_band`? 1/12? majority class?) and the no-DP ceiling. Without them the y-axis is
   uninterpretable.
4. **Same x-axis as the utility curve**, so Fig A and Fig C stack.

**If the result is "attack success does not fall with ε":** that is publishable and it is the
paper's most interesting possible finding — *cut-layer noise costs 25× the utility of server
DP-SGD and buys no measurable attack resistance, therefore drop it from the architecture.*
State it as a finding, not a failure. But it must be measured against the real wire, or it is
just a harness bug.

---

## 4. L3 — what ε does not cover (write this table into the paper)

| Surface | Covered by reported ε? | Note |
|---|---|---|
| Training-set embeddings on the wire | ✅ cut-layer ε per silo | the mechanism |
| Top-model parameters as released | ✅ server DP-SGD ε | different adversary — see §0.3 |
| **Val/test embeddings on the wire** | ❌ **NO** | `generate_eval_embeddings` clips but **does not noise**, and takes no accountant step. At `eval-every = 5`, T = 200, that is ~40 unnoised releases of 2,552 val students, plus the final test release |
| Labels | ❌ by construction | server computes the loss; moving it client-side does not fix it (gradient ∝ p−y). Cite Li et al. 2022 |
| Local silo weights | ❌ out of threat model | institutional trust boundary |
| Hyperparameter search | ❌ | per-run guarantee only; cite Papernot & Steinke ICLR 2022 |
| The 95+ published runs on the same cohort | ❌ | ε is per run, not per project |

**The val/test hole is the one to deal with deliberately.** Two defensible positions — pick one
and say it out loud:

- **(a) Scope it.** "The reported ε is a guarantee over the *training* set. Validation and test
  embeddings are released clipped but unnoised; those 5,104 students receive no formal
  protection." Honest, costs nothing, and is *true of most VFL papers* — which is itself worth
  a sentence.
- **(b) Close it.** Noise the eval embeddings too. Because val/test rows are disjoint from
  train, this composes in **parallel**, so it is a *second, separate* ε over the eval
  population — not an increase to the training ε. Costs some eval utility and a second
  accountant.

(a) is the right call for this timeline; (b) is a strong "future work" line. What is not
defensible is leaving it unstated.

---

## 5. The figure set

| Fig | Content | Status |
|---|---|---|
| **A — privacy–utility curve** | AUC **and** recall@p70 vs ε, cut-layer; annotate the two truncation confounds (ε=4 invalid, low-ε budget-exhausted) | ✅ data exists (§2h); **re-run ε>2 with the cap raised** |
| **B — the ε-isocline** | Utility spread at *identical* ε=1.0 across mechanism placement and clip S. The §2.2 table as a dot plot | ✅ data exists (§2g + §D3) |
| **C — attack ASR vs ε** | Same x-axis as A, with random-guess floor and no-DP ceiling, n≥200, CIs | ❌ blocked on §3 |
| **D — variance vs ε** | sd(AUC) across seeds vs ε: 0.002 → 0.005 → 0.130 | ✅ data exists (§2h) |
| **T1 — parameter → ε map** | §2.1 table: which knobs move the guarantee, which are free | ✅ analytic |
| **T2 — coverage** | §4 table: what ε does and does not protect | ✅ writing only |
| **T3 — verification** | §2.3 V1–V6 pass/fail, generated by the reproduce pipeline | ❌ to build |

**Fig D is underrated.** "At tight budgets the model is not merely worse, it is unpredictable"
(sd 0.130 at ε=0.5) is a deployment argument that does not depend on the attack results
landing, and no one else reports it.

---

## 6. Ordering — what to do before writing

1. **Record mechanism primitives in `meta.json`** (§2.4). Everything downstream needs them, and
   without them no existing checkpoint can be audited.
2. **Build the verification script** (V1–V6, §2.3) → T3. Half a day; converts the guarantee from
   asserted to verified, and is independent of the attack work.
3. **Re-run ε > 2** with `combined-target-epsilon` raised → completes Fig A.
4. **Write L1 + L3 now** (§2.2 isocline framing + §4 coverage table). Neither depends on the
   attacks.
5. **Then the attack work** (§3): wire-faithful attacks, n≥200, anchors → Fig C.

Steps 1–4 give a complete, defensible privacy section on their own — with Fig C as the
upgrade rather than the dependency. If the attacks slip, the paper still has a privacy story.
