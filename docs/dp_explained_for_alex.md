# DP & Model Collapse — Explained, With Code Locations

A plain-language walkthrough of the chat response you got about DP wrecking your model,
mapped to the exact variables and files in JayQuery. Read top to bottom; each section
answers "what is this thing, where is it in the code, how do I tune it, why does it matter."

> **UPDATE (2026-07-23) — the "model is useless" premise was wrong; §9 has the resolution.**
> Two measurements overturned the early hypotheses in this doc, in a *good* way:
> 1. The per-sample grad-norm logging (§1) showed median norm ≈ **4.0** with **100% of
>    samples clipped** at `C=1.0` — the *opposite* of §1's guess that norms were far below
>    `C`. So C-tuning is **not** a free SNR win (when everything is clipped, `C` cancels);
>    it only reduces clipping bias. The §3 √d argument still stands.
> 2. The three diagnostic runs (no-DP / cut-noise / server-DP) showed the DP models were
>    **not** collapsing at all — they held **AUC 0.70–0.77** (server-DP was the *best*
>    ranker). The apparent "everyone classified negative" was a **fixed-0.5-threshold
>    artifact**, not a dead model. See **§9** for the full analysis and the code fixes.

> **Quick orientation.** DP-SGD does two things to every gradient step: (1) **clip** each
> per-sample gradient to a max length `C`, then (2) **add Gaussian noise** whose size is
> `sigma * C`. Clipping bounds how much one student can influence the model (the privacy
> "sensitivity"); noise is what actually buys the privacy. The whole debate below is about
> the *ratio* between your real training signal and that injected noise — the
> **signal-to-noise ratio (SNR)**. When SNR drops below ~1, the model is mostly following
> noise, and that's your "useless model."

---

## 1. `max_grad_norm` (the "C" in the chat) — the clipping threshold

**What it is.** The maximum L2 length any single sample's gradient is allowed to have.
Opacus clips every per-sample gradient down to this length before summing them. It is the
DP *sensitivity* — the cap on how much one student can move the weights.

**Where it lives in code:**
- Config knob: `server-dp-max-grad-norm = 1.0` in [pyproject.toml:98](../pyproject.toml).
- Read into the `dp` dict: [`_server_dp_config`](../fl_server/server_app.py) →
  `"max_grad_norm"` at [fl_server/server_app.py:400](../fl_server/server_app.py).
- Passed to Opacus here: [fl_server/server_app.py:198](../fl_server/server_app.py)
  (`max_grad_norm=dp["max_grad_norm"]`), which forwards to the `DPOptimizer` in
  [`make_private_with_noise`](../privacy/dp_wrapper.py) at
  [privacy/dp_wrapper.py:66](../privacy/dp_wrapper.py).

**Why your instinct ("smaller C = cleaner signal") is only half right.** Opacus scales the
*noise* by `C` too. Concretely it does: clip each gradient to length ≤ `C`, sum them, add
noise `N(0, sigma²·C²)`, divide by batch size. Both the signal and the noise are
proportional to `C`, so:

- **If your true gradient norms are ≥ C** (clipping actually bites): shrinking `C` shrinks
  signal and noise *together*. It changes clipping **bias**, not SNR. No free lunch.
- **If your true gradient norms are well below C** (clipping rarely triggers): the signal
  stops growing once you raise `C` past the real norms, but the noise keeps growing with
  `C`. Here **SNR ∝ 1/C**, and an oversized `C` is pure wasted noise.

The chat's bet — and it's a good one for a small top model 150 steps into a shrinking loss —
is that you're in the second case: your real per-sample gradient norms are maybe 0.05–0.2,
but `C = 1.0`, so you're injecting perhaps **10× more noise than the mechanism needs**.

**How to tune it (the one genuinely free knob).** Measure your real per-sample gradient
norms, then set `C` near their median. Opacus stashes per-sample gradients on each parameter
as `p.grad_sample` once the model is wrapped. Add this logging right after `loss.backward()`
and before `train_optimizer.step()` — i.e. between
[fl_server/server_app.py:230](../fl_server/server_app.py) and
[fl_server/server_app.py:241](../fl_server/server_app.py):

```python
# Gated on dp["enabled"]: grad_sample only exists when train_model is the
# Opacus GradSampleModule. With DP off, no param has grad_sample, the list is
# empty, and torch.stack([]) raises — so this block must not run.
if dp["enabled"]:
    with torch.no_grad():
        norms = torch.stack([
            p.grad_sample.reshape(p.grad_sample.shape[0], -1).norm(2, dim=1)
            for p in train_model.parameters() if hasattr(p, "grad_sample")
        ]).norm(2, dim=0)
    log(INFO, "per-sample grad norm: median=%.4f p90=%.4f frac_clipped=%.3f",
        norms.median(), norms.quantile(0.9), (norms > dp["max_grad_norm"]).float().mean())
```

This must sit *after* `loss.backward()` (so `grad_sample` is populated) and *before*
`train_optimizer.step()` (so you read the true per-sample norms, before Opacus clips them).
If `frac_clipped` is tiny (say < 0.1) and the median is ~0.1, set
`server-dp-max-grad-norm` to roughly that median. That alone can cut your injected noise
several-fold with zero privacy cost.

---

## 2. `sigma` — the noise multiplier (the star of the show)

**What it is.** `sigma` is the *normalized* amount of noise: the Gaussian noise standard
deviation divided by the sensitivity. In DP-SGD the actual noise added has std
`sigma * C`. Bigger `sigma` = more noise = **stronger privacy** (smaller epsilon) but
**worse utility**. It is the single number that trades privacy against model quality.

**Where `sigma` is in the code.** It is *not* a config knob you set directly — it's
*solved for* from your target epsilon. The chain:

1. You set the target: `server-dp-epsilon = 1.0` in
   [pyproject.toml:96](../pyproject.toml).
2. The server solves the smallest `sigma` that hits that epsilon over the run:
   `sigma = solve_noise_multiplier(...)` at
   [fl_server/server_app.py:189](../fl_server/server_app.py). That function lives in
   [privacy/privacy_accountant.py:158](../privacy/privacy_accountant.py) and bisects for
   the least noise that still keeps composed epsilon ≤ target.
3. That `sigma` is handed to Opacus as `noise_multiplier=sigma` at
   [fl_server/server_app.py:198](../fl_server/server_app.py) →
   [privacy/dp_wrapper.py:64](../privacy/dp_wrapper.py) (`DPOptimizer(noise_multiplier=...)`).
4. The *same* `sigma` is stored in the accountant that enforces the budget:
   `PrivacyBudgetTracker(sigma=sigma, ...)` at
   [fl_server/server_app.py:204](../fl_server/server_app.py). The tracker field is
   `self.sigma` — [privacy/privacy_accountant.py:56](../privacy/privacy_accountant.py).

The client cut-layer noise mechanism has its own parallel `sigma`, solved the same way at
[fl_client/common.py:265](../fl_client/common.py) and applied as
`noise_std = tracker.sigma * noise["max_norm"]` at
[fl_client/common.py:306](../fl_client/common.py). Same idea, different surface (the
embedding instead of the top-model gradient).

**Why you can't just "turn sigma down."** You don't set it — epsilon sets it. To get less
noise you must either (a) ask for a bigger epsilon, (b) reduce the number of steps it has to
cover, or (c) keep more subsampling amplification. See §4.

**Why noise ∝ sigma destroys the model.** The averaged *signal* (the real gradient
direction) has length at most `C`. The *noise* is a vector drawn in every parameter
dimension. Which brings us to the real problem:

---

## 3. Why ε = 1.0 is essentially unreachable at your scale (the √d problem)

**The core fact.** Isotropic Gaussian noise is added independently to *every* parameter, so
the *length* of the noise vector grows as **√(number of parameters)**. Your signal length is
capped at `C`. So SNR ≈ `C / (sigma · √d)`. When `d` is large, you lose.

**Your numbers, from the code:**
- Top model architecture: `Input(64) → Linear(hidden) → ReLU → Dropout → Linear(hidden/2)
  → ReLU → Linear(1)` — [fl_server/top_model.py:28](../fl_server/top_model.py).
- `hidden-dim-top = 128` in [pyproject.toml:87](../pyproject.toml), and
  `embedding-dim = 32` so the input is `2 × 32 = 64`.
- Parameter count at hidden=128: `(64·128+128) + (128·64+64) + (64·1+1) = 8320 + 8256 + 65
  = 16,641` params → √d ≈ **129**.

So the noise vector is ~129× longer than a unit direction, while your signal is capped at
`C = 1`. Even with a modest `sigma`, the update you actually apply is dominated by noise.
The chat's table put your live config (ε=1, q=0.05, 150 steps) at **SNR ≈ 0.13** — i.e. the
update is ~87% noise. That's not a bug in your wiring; it's the mechanism behaving exactly as
designed at this scale. **A tiny cohort + a 16k-parameter model + ε=1 cannot coexist.**

One counterintuitive note from the chat: going to **full batch** (`batch-sample-rate = 1.0`)
barely helps at ε=1. You'd think a 20× bigger batch averages away noise — but losing
subsampling amplification forces `sigma` way up (roughly 2.7 → 28.6 in their table), which
almost exactly cancels the bigger batch. Don't spend a day on that lever.

---

## 4. Your actual levers, ranked by effect

| Lever | Config key | File | Effect on SNR |
|---|---|---|---|
| **Model size** (strongest, cheapest) | `hidden-dim-top` | [pyproject.toml:87](../pyproject.toml) | 128→32 params 16.6k→1.2k, ~**4× SNR** |
| Target epsilon | `server-dp-epsilon` | [pyproject.toml:96](../pyproject.toml) | ε 1→10 roughly 7× SNR (at hidden=32, full batch) |
| Fewer steps | `num-server-rounds` | [pyproject.toml:80](../pyproject.toml) | sigma ∝ ~√T, so 150→60 steps ≈ 1.9× |
| Clipping threshold | `server-dp-max-grad-norm` | [pyproject.toml:98](../pyproject.toml) | Only if C is oversized (§1) |
| Batch / subsampling | `batch-sample-rate` | [pyproject.toml:94](../pyproject.toml) | Weak at ε=1 (amplification cancels) |

**Model size is the big one.** The top model only has to classify a 64-dim concatenation —
it doesn't need 128 hidden units. Dropping `hidden-dim-top` from 128 to 32 cuts parameters
~14×, shrinks √d ~3.8×, and buys roughly 4× SNR for very little capacity. This is the first
thing to try.

---

## 5. The A/B/C diagnostic — is it the clipping or the noise?

The chat suggests one run that separates the two DP effects, to prove noise (not clipping) is
what's killing you:

| Run | DP | C | sigma | Isolates |
|---|---|---|---|---|
| A | off | — | — | The ceiling (no-DP quality) |
| B | on | 1.0 | **0.0** | Clipping bias *alone* (no noise) |
| C | on | 1.0 | solved | Clipping + noise (real DP) |

If **B ≈ A**, clipping is harmless and noise is the entire story (the expected result). If
**B ≪ A**, your `C` is mis-set and §1's measurement is the fix.

**What you'd need to add.** Right now there is no way to force `sigma = 0` while keeping
clipping on — `sigma` is always solved from epsilon at
[fl_server/server_app.py:189](../fl_server/server_app.py). Add an override, e.g. a
`server-dp-noise-multiplier` config key that, when ≥ 0, skips `solve_noise_multiplier` and
uses the given value directly. **Run C's ε is valid; Run B has no privacy guarantee** (σ=0
means zero privacy) — label it clearly and never report it as a DP result.

---

## 6. Two implementation notes from the chat

### 6a. Your sigma is calibrated for 150 steps but you stop around ~55–70

`solve_noise_multiplier` is called with `steps=num_rounds` (=150) at
[fl_server/server_app.py:191–194](../fl_server/server_app.py). But early stopping
(`early-stop-patience = 20`, [pyproject.toml:114](../pyproject.toml)) fires much sooner. So
you size `sigma` to survive 150 noisy steps, then only take ~60 — paying ~1.6× more noise
than needed, **and** the epsilon you *report* (budgeted for 150) is not the epsilon you
actually *spent* (~60 steps). Since sigma ∝ ~√T, calibrating to a realistic ~60-step horizon
is close to free utility. The clean fix is to pass a realistic horizon into
`solve_noise_multiplier` and enforce that cap.

### 6b. "Double clipping" — already handled in your code ✅

The chat warned that `grad-clip-norm = 2` ([pyproject.toml:121](../pyproject.toml)) running
alongside Opacus's `max_grad_norm = 1.0` could clip an already-noised gradient. **Your code
already guards against this** — no action needed:

- On the **server**, your own clip only runs when DP is *off*:
  `if grad_clip > 0 and not dp["enabled"]:` at
  [fl_server/server_app.py:239](../fl_server/server_app.py). When DP is on, Opacus owns the
  clipping and your `clip_grad_norm_` is skipped. Correct.
- On the **clients**, `grad-clip-norm` clips the *bottom* models
  ([fl_client/common.py:435–437](../fl_client/common.py)) — which is exactly where the chat
  says it *should* apply (bottoms use cut-layer noise, not Opacus, so plain clipping there is
  fine).

So this one's a false alarm for your current wiring — but worth understanding so you don't
accidentally remove the `not dp["enabled"]` guard.

---

## 7. What to actually do (in order)

1. **Log per-sample grad norms** (§1 snippet); set `server-dp-max-grad-norm` to the median.
   One run.
2. **Run the A/B/C diagnostic** (§5) to confirm noise is the culprit — needs the
   `server-dp-noise-multiplier` override.
3. **Set `hidden-dim-top = 32`** and calibrate sigma to a realistic ~60-step horizon (§6a).
4. **Re-run your epsilon grid** — and *expect* ε = 0.1 and ε = 0.5 to sit at chance.

That last point is the real reframe: a utility curve that's **flat at chance below ε ≈ 2**
and **rises toward the no-DP ceiling above it** *is* your privacy–utility tradeoff. "Useful
DP guarantees are unreachable at single-course cohort sizes with this architecture" is a
legitimate, publishable finding for a 10-week project — and it hands Wyatt the cleanest
version of the story: attack success should collapse in exactly the ε region where utility
does.

---

### One-line glossary

- **C / `max_grad_norm`** — max length of one sample's gradient (the sensitivity).
- **sigma / noise multiplier** — noise std ÷ C; solved from epsilon, not set by hand.
- **epsilon (ε)** — the privacy budget; smaller = more private = more noise = worse utility.
- **SNR** — signal length ÷ noise length; below ~1 the model follows noise.
- **√d** — noise length grows with the square root of the parameter count; why big models hurt under DP.

---

## 9. The runs don't collapse — they miscalibrate (analysis + the fix)

### 9a. What the three diagnostic runs actually showed

| Run | base_rate | pred_mean | pred_std | AUC | **AP** | recall @0.5 | combined ε |
|---|---|---|---|---|---|---|---|
| NO_DP | 0.558 | 0.445 | 0.034 | 0.723 | 0.732 | 0.093 | — |
| CUT_NOISE | 0.558 | 0.448 | 0.005 | 0.699 | 0.718 | 0.000 | 1.33 |
| SERVER_DP | 0.558 | 0.465 | 0.009 | 0.767 | **0.762** | 0.000 | 0.87 |

The number that settles the "is it worthless?" question is **average precision (AP)**. A
random ranker scores `AP = base_rate = 0.558`. All three runs sit at **AP ≈ 0.72–0.76** —
a real lift, i.e. dropouts really are concentrated at the top of the ranking. The DP models
**can** find dropouts.

So why does `recall @0.5 = 0`? Because **`pred_mean` (0.44–0.47) is below both 0.5 and the
true dropout rate (0.558)**. DP-SGD's clip+noise shrink the logits toward zero, so the whole
probability cloud is pushed under 0.5 and the fixed cutoff flags nobody. That is
**miscalibration**, not a dead model — and it's exactly what the old `pred_collapsed` flag
was mislabeling as collapse.

### 9b. The four fixes (all landed)

1. **AUC/AP is the primary utility metric** (threshold-free). It shows DP preserves ranking.
2. **`pred_collapsed` redefined to be AUC-based** — [baselines/eval_utils.py](../baselines/eval_utils.py):
   `collapsed = AUC <= AUC_COLLAPSE_FLOOR (0.55)`. The old fixed-0.5 one-class condition is
   kept as the *non-fatal* diagnostics `one_class_at_threshold` and `pred_low_variance`.
3. **Validation-tuned decision threshold** — `tune_threshold()` in
   [baselines/eval_utils.py](../baselines/eval_utils.py); the server tunes on val then scores
   test at that threshold, keeping the 0.5 result alongside for an honest before/after
   ([fl_server/server_app.py](../fl_server/server_app.py), final-eval block; controlled by the
   `threshold-objective` run-config key, default `"f1"`).
4. **Deployment operating points** — `recall_at_precision()` reports "if advisors accept
   flagged students at ≥ precision p, what fraction of real dropouts do we catch?" logged as
   `recall_at_precision_{60,70,80}`.

### 9c. The honest deployment caveat (the *real* cost of DP here)

DP doesn't destroy the ranking — it destroys **confidence separation**. `pred_std` collapses
from 0.034 (no-DP) to **0.005–0.009**, i.e. every student's probability is crammed into a
~0.44–0.47 band. Consequences to state plainly in the paper:

- **Deploy as a ranking, not as probabilities.** Flag the top-K% by risk score for
  intervention. Do *not* surface the raw number as "your dropout risk is 45%" — under DP
  they're all ~45% and meaningless as calibrated probabilities.
- **The tuned threshold is fragile.** Because everyone is packed together, a small
  calibration drift moves many students across the cutoff. A real deployment would add a
  recalibration step (temperature / Platt scaling) fit on a held-out split.
- **Pick the operating point from intervention capacity, never 0.5.** "Advisors can contact
  30% of students" → threshold at the 70th percentile of scores. "Must catch 80% of
  dropouts" → set for recall 0.8 and accept the resulting precision/workload. `recall@precision`
  is the curve that quantifies this.

### 9d. Two cautions on the numbers

- **AUC gaps are within noise.** Test set is ~231 rows; SERVER_DP (0.767) "beating" NO_DP
  (0.723) is sampling variance. Report DP *preserves* AUC — run 3–5 seeds before claiming any
  ordering.
- **This retro-explains §1.** The flat median-4.0 gradients weren't a dead model; the model
  was ranking fine the whole time (AUC 0.77). Same root cause as the "collapse": reading a
  fixed-threshold metric instead of AUC.
