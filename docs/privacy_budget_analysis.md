## Part 1 — What this file is *supposed* to do (the mental model)

Forget the code for a second. Here's the whole idea in five steps.

**1. Every noisy step leaks a little.**
Each time the server takes a DP-SGD step, it (a) clips each per-sample gradient to a max norm `C`, then (b) adds Gaussian noise before applying the update. That noise is what buys privacy. But the step still reveals *something* about the training data. Epsilon (ε) is the number that measures "how much has leaked so far." Higher ε = weaker privacy.

**2. One step = one "Gaussian mechanism."**
A single noisy step is fully described by one number: the **noise multiplier**

```
σ = (std-dev of the noise added) / (sensitivity)
```

where "sensitivity" = how much one student's data can move the thing you're noising. In DP-SGD, clipping *is* what sets sensitivity: it's `max_grad_norm`. So Opacus's `noise_multiplier` is already σ — the division is done for you. (For Wyatt's client-side cut-layer noise, it is *not* done for you; you divide the noise std by the embedding sensitivity yourself. Same formula, different plumbing. This is the one place people mix things up.)

**3. Many steps compose.**
Doing T steps costs more than one step. "Composition" is the math for turning `T` steps of cost σ each into one total ε. Naively you'd add them up; the whole point of RDP / autodp is that it composes them *tightly* (roughly √T growth instead of T growth), so you can afford far more rounds at the same budget.

**4. Epsilon is always recomputed, never accumulated.**
This is the single most important design rule and the one the current file violates. The accountant should be a **pure function**:

```
ε = f(σ, sample_rate, steps, δ)
```

You hold four primitives, and any time someone asks "what's my ε?", you recompute from scratch. You never take last round's ε and add to it. Two reasons: (a) adding cumulative numbers to cumulative numbers double-counts, and (b) the project plan (§3) requires that state survive serialization — plain floats and an int survive; a live accountant object with hooks does not.

**5. Compare against the budget.**
`target_epsilon` is the ceiling. After each round you recompute ε; if it's over the ceiling, you raise `PrivacyBudgetExceededError` and stop training. That's the whole contract.

So the finished class is really:

| Method | Job |
|---|---|
| `__init__` | Store σ, δ, sample_rate, target_ε, steps=0. Nothing else. |
| `step()` | `steps += 1`, recompute ε, raise if over budget. |
| `get_epsilon()` | Build a fresh mechanism from σ, compose it `steps` times, return ε at δ. **No mutation.** |
| `state_dict()` / `from_dict()` | Dump/restore `{σ, sensitivity, sample_rate, steps}` so the tracker can be rebuilt from `context.state`. |

---

## Part 2 — Where the file actually is

The skeleton is right and you clearly understand the *shape* of the autodp API. But there are three correctness bugs, and the middle one is severe.

### 🔴 Critical — `get_epsilon()` mutates `self.mechanism`, so ε explodes

```python
self.mechanism = transformer.compose(mechanism_list=[self.mechanism], coeff_list=[self._round])
```

You compose the mechanism and then **overwrite the base with the composed result**. So the next call composes the *already-composed* thing again. Worse, `step()` calls `get_epsilon()` twice (once inside `update_DPlosses`, once for `current`). Trace it:

| Round | Mechanism after `step()` | Should be |
|---|---|---|
| 1 | M | M |
| 2 | M⁴ | M² |
| 3 | M³⁶ | M³ |
| 4 | M⁵⁷⁶ | M⁴ |

Your ε will look plausible for one round and then rocket to garbage. **Fix:** keep an immutable base mechanism, compose into a local variable, return its ε. Also short-circuit `steps == 0 → return 0.0`.

### 🔴 The `dp_acct.DP_acct()` accountant is redundant and double-counts

`update_DPlosses(eps=self.get_epsilon(), ...)` feeds it your *cumulative* ε as if it were *this round's* ε. It's a second accounting path that's being fed the wrong units, running in parallel with the one that actually produces your return value. It contributes nothing. Delete it — compose-over-the-base-mechanism is the whole accountant.

### 🟠 `sensitivity` is accepted and then never used

It's in the signature, never stored, never applied. Right now you're passing raw `sigma` straight into `GaussianMechanism`, which is only correct if sensitivity happens to be 1.0. Decide explicitly:
- **Server / DP-SGD:** `sigma = opacus_noise_multiplier` directly (sensitivity is already baked in by `max_grad_norm`). Passing `sensitivity` again would double-divide.
- **Client / cut-layer:** `sigma = noise_std / sensitivity`.

Cleanest resolution: make the tracker take **only a noise multiplier**, and let the caller do the division. Then the same class serves both surfaces without an ambiguity trap.

### 🟠 No subsampling term — DP-SGD isn't a plain Gaussian

Per the plan, the server accountant needs a `sample_rate`. It's `1.0` today (full-batch), which is why plain `GaussianMechanism` happens to give the right answer *right now* — but the moment you minibatch, this silently under-reports ε. Take `sample_rate` in `__init__` and either use `mechanism_zoo.SubsampledGaussianMechanism`, or keep `GaussianMechanism` with an explicit assertion that `sample_rate == 1.0` so it fails loudly rather than lying.

Also: `ComposeGaussian` is the *exact* composition for homogeneous Gaussians — correct choice for the plain case, but it expects each mechanism to expose a `sigma` param. Feeding it an already-composed mechanism (see bug 1) is exactly what it isn't built for. For subsampled Gaussians you'd switch to `transformer_zoo.AmplificationBySampling` + generic `Composition`.

### 🟡 Smaller things

- Duplicate import of `PrivacyBudgetExceededError` (lines 8 and 11).
- `num_epochs`, `batch_size`, `dataset_size` are stored but unused — vestigial from the old Opacus-on-the-clients design. Either delete them, or use them to *derive* `sample_rate = batch_size / dataset_size`, which is the honest version.
- No `state_dict()` / `from_dict()`. The server is persistent so you can get away with this for now, but the plan says accounting persists as plain primitives, and Wyatt's client-side accountant will need it. Build it now, it's four lines.
- No combined client+server reporting. Plan §1: *"the system's true guarantee is the composition of both."* Two `PrivacyBudgetTracker` instances is fine, but you need a `combine_epsilon(client_tracker, server_tracker, delta)` that composes their mechanisms into one reported number. Right now nothing does this.
- Budget check happens *after* the spend. Consider a `would_exceed(n=1)` so training can stop *before* over-spending rather than after.
- Env: pinned `autodp==0.2.3` is broken → `autodp==0.2.3.1` + `scipy==1.13.1`.

---

## The corrected core, so you can see the shape

```python
def get_epsilon(self) -> float:
    if self._round == 0:
        return 0.0
    base = mechanism_zoo.GaussianMechanism(sigma=self.noise_multiplier)  # fresh each call
    composed = transformer_zoo.ComposeGaussian().compose(
        mechanism_list=[base], coeff_list=[self._round]
    )
    return composed.get_eps(delta=self.delta)          # nothing on self was mutated

def step(self) -> None:
    self._round += 1
    current = self.get_epsilon()                        # called exactly once
    if current > self.target_epsilon:
        raise PrivacyBudgetExceededError(
            f"Privacy budget exceeded: {current:.4f} > {self.target_epsilon}"
        )
```

**Sanity check to write before you trust any of it:** with a fixed σ, ε must be monotonically increasing and grow *sub-linearly* in `steps`. If `ε(100)` is more than ~100× `ε(1)`, composition is broken. That single assertion in `tests/test_dp.py` would have caught the mutation bug immediately.

Want me to write out the full corrected `privacy_accountant.py` with the `sample_rate`, `state_dict()`, and the combined client+server reporting function?