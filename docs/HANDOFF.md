# JayQuery — Session Handoff

> **Read this first in a new chat.** Current state, what's running, what's next, and the
> traps that cost real time. Written 2026-07-30.
>
> **Companion docs:** `docs/action_plan.md` (decisions + open tasks, the authoritative log) ·
> `docs/FINDINGS.md` (evidence + results tables) · `docs/paper_topics.md` (writing aid) ·
> `docs/privacy_presentation_plan.md` (how to present the guarantee; what is proven vs
> measured vs uncovered) · `docs/DATA_DICTIONARY.md` (data contract) ·
> `docs/dp_explained_for_alex.md` (DP background).
>
> ## ➜ Results live in `results/` — start at `results/README.md`
>
> ```bash
> bash scripts/reproduce_all.sh                  # reuse artifacts, rebuild results/ (~10s)
> bash scripts/reproduce_all.sh --fresh --wandb  # re-run everything, then publish (~6h)
> bash scripts/reproduce_all.sh --stages publish # just rebuild results/
> ```
>
> `results/` is **gitignored and fully regenerable** — a build output, not a source of
> truth. It holds the headline claims (`claims.csv`, with Welch deltas + 95% CIs), the
> figures, every table as csv/tex/md, the privacy audit, and `provenance.json`.
> Its README opens with the standing open issues, **re-detected from the artifacts on every
> build** so they cannot be forgotten. Supporting scripts: `analyze.py` (claims),
> `figures.py` (plots), `verify_privacy.py` (V1–V6 audit), `results_lib.py` (shared
> loading/stats). See `action_plan.md` §R.

---

## 1. Where the project stands

**The pipeline works and the headline results are in.** 60-run matrix complete, PSI verified,
cross-mode equivalence established, metrics hardened. Twelve bugs found and fixed — several of
which silently corrupted earlier results (see `FINDINGS.md` §1).

### Headline results (§D3, 5 seeds/arm, T=200, `dedup_first`, simulation)

| arm | AUC | Δ vs no-DP ceiling | verdict |
|---|---|---|---|
| no_dp | 0.9023 | — | ceiling |
| **server_only** (ε=1.0) | 0.9001 | **−0.0022 [−0.0053, +0.0009]** | **no measurable cost** |
| cut_only (ε=1.0) | 0.8461 | −0.0562 [−0.0616, −0.0508] | significant |
| all_dp (ε=1.75) | 0.8227 | −0.0797 [−0.1033, −0.0560] | significant |

Replicated on all three dedup variants; the DP effect moves <0.001 across them.

**The paper's spine:** *where* you inject noise matters far more than *how much*. Two
mechanisms, identical nominal ε=1.0, 25× different utility cost.

---

## 2. COMPLETED THIS SESSION

**✅ ε sweep — DONE.** 35/35 runs, 0 failures. `experiments/eps_sweep.json`, wandb group
**`eps-sweep`** (synced). Full table + analysis in `FINDINGS.md` §2h.

**Headline:** the usable knee is **ε≈1**. Two truncation confounds must be understood before
plotting — **ε=4 is invalid** (hit the `combined-target-epsilon = 3.0` cap and stopped at round
~50; under-trained, not over-noised) and **low-ε points are budget-exhausted** (ε=0.1 stops at
round ~21 — legitimate, but they reflect noise *plus* fewer rounds).

**Most quotable result:** `recall@precision-70` collapses 0.887 → 0.465 (ε=0.5) → 0.047 (ε=0.1)
while AUC still reads 0.73 / 0.59. Reporting AUC alone would badly overstate low-ε viability.

### ✅ Clip calibration — DONE, and it inverted the expectation

| setting | (S_a, S_b) | mean AUC | sd |
|---|---|---|---|
| **S=1.0** (shipped placeholder) | (1.0, 1.0) | **0.8453** | 0.0042 |
| S=median | (5.6, 4.0) | 0.6022 | 0.0664 |
| S=p75 | (6.7, 6.3) | 0.5995 | 0.0582 |

**Small S wins.** Moving to the median costs **−0.243 AUC** and inflates variance ~16×. Reason:
the embedding magnitude saturates at its natural row norm (~4.3) while noise scales linearly
with S, so SNR = min(S, norm)/(σ·S) is flat below the min norm and falls above it. The
percentile heuristic is gradient-clipping intuition that does **not** transfer to input-side
noise. `measure_embedding_norms.py` has been corrected; `run_matrix.py` defaults are now 1.0.

---

## 3. NEXT TASK — the ε sweep (ready to run)

The privacy–utility curve. **This is the paper's core figure.**

```bash
# LAUNCHED 2026-07-30 with the calibrated clip. To resume/re-run:
python scripts/run_matrix.py --jobs eps --variants dedup_first \
    --seeds 42 43 44 45 46 --rounds 200 --mode sim \
    --wandb-group "eps-sweep" --max-norm-a 1.0 --max-norm-b 1.0 \
    --out experiments/eps_sweep.json --resume
```

- Sweeps `cut-noise-target-epsilon` over **{off, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0}**
- 7 points × 5 seeds = **35 runs ≈ 2h** (3 seeds ≈ 1.2h; cut-noise σ≈0.005 so 3 may suffice)
- Logs to wandb **offline** under `--wandb-group`; `wandb sync` afterwards
- `--max-norm-a/-b` MUST come from the clip calibration — sweeping ε against a mis-set clip
  maps the wrong curve

**Why calibration came first, and what it showed:** `cut-noise-max-norm` shipped at **1.0**, a
documented placeholder, while measured embedding row norms are ~4.3 (Silo A) / ~4.0 (Silo B) —
so 100% of rows clip. The expectation was that this destroyed signal and that raising S would
be a free win. **Measurement showed the opposite** (see §2): S=1.0 beats the median by
**+0.243 AUC**.

The reason is that **signal saturates while noise does not**:

    SNR = min(S, row_norm) / (sigma * S)
      S <= min row norm ->  1/sigma            (flat — best)
      S >  row_norm     ->  row_norm/(sigma*S) (falls as S grows)

The embedding's magnitude is capped by the model (~4.3); the noise scales linearly with S. So an
oversized S is pure waste. Percentile-based clip selection is a **gradient-clipping** heuristic
that does not transfer to input-side noise.

---

## 4. Open tasks, in order

| # | Task | Notes |
|---|---|---|
| 0 | **Re-run everything with `--fresh`** | `bash scripts/reproduce_all.sh --fresh --wandb` — all 176 existing checkpoints predate mechanism recording and are **unauditable** (§R4). Nothing else in this list is worth doing against unauditable artifacts |
| 0b | **Root-cause §R1** (accountant under-counts a step) | Now caught by `verify_privacy.py` V5, but under-reporting ε is the direction that matters. Check whether it reproduces in deployment mode |
| 0c | **Decide §R2** — scope or close the val/test coverage hole | One sentence in the paper either way; leaving it unstated is not an option |
| 1 | ~~Clip calibration~~ | ✅ **DONE** — S=1.0 wins by 0.243 AUC; see §2 |
| 2 | ~~ε sweep~~ | ✅ **DONE** — 35/35, synced, written up in `FINDINGS.md` §2h |
| 2b | **Re-run the high-ε end** with `combined-target-epsilon` raised | ε=4 was capped at composed 2.981 and truncated; the curve above ε=2 is currently unusable |
| 3 | **Re-run §D3 in simulation** (deferred by decision) | see §5 — skips are now counted, so the bound is reportable |
| 4 | Attack ASR at n≈200 | current n=20 gives SE 0.112; cannot resolve DP differences |
| 5 | **§P2** freeze the defense config | after tuning settles; breaks the tune→attack feedback loop |
| 6 | **§P3** hand off to attacks | §D3 produces the checkpoints |
| 7 | Deferred wandb parameter sweep | ε grid, clip norms, `hidden-dim-top`, deployment arm |

**Attack handoff is LAST by decision** — attacks sit on top of the pipeline and gate nothing.
`scripts/run_attack.py` loader is already fixed (§P1 done) and verified working.

---

## 5. ⚠️ §D3 re-run — DEFERRED, do not forget

**Decision 2026-07-30 (Alex): skip for now, re-run in SIMULATION later.**

**Why it needs re-running:** the §D3 numbers were produced before the replay-skip fix. In
simulation, Ray spawns new actors mid-run and the first round on a fresh actor silently skipped
that silo's weight update (~2% of updates, early rounds). Crucially the arms were **not equally
affected** — cut-noise-OFF runs (`no_dp`, `server_only`) had skips, cut-noise-ON runs had none.
That biases **against the no-DP ceiling**, so the true DP cost may be marginally **larger** than
reported. With `server_only` at −0.0022 (CI containing zero), that is the direction that matters.

**Why simulation is still the right choice:** skips are now **counted and reported** in
`meta.json` (`replay_skips`, `replay_skips_total`), so a re-run declares its own bound instead
of hiding it. Deployment is provably clean (0 skips in ~614 rounds) but 60 runs would take
~34 hours vs ~3 in simulation.

```bash
python scripts/run_matrix.py --jobs d3 --variants dedup_first dedup_best dedup_worst \
    --seeds 42 43 44 45 46 --rounds 200 --mode sim --out experiments/d3_rerun.json
```

Then confirm `replay_skips_total` per run and state the bound in the paper.

---

## 6. TRAPS — these cost hours, do not rediscover them

### Running batches
- **ALWAYS use `scripts/run_matrix.py`.** Never hand-roll a batch loop. It encodes four
  failure modes (§O) that each produced *silently missing runs*.
- **`pkill -f <name>` matches the calling shell's own command line** and kills your runner.
  Use `pkill -x`. I hit this **three times** in one session.
- **`flwr run --stream` returns BEFORE `meta.json` is written.** Wait for the artifact, then reap.
- **`flwr-simulation` workers leak ~800 MB each.** OOM at ~15–18 consecutive runs, surfacing as
  a *missing output file*, not an error. Reap after every run.
- **Python block-buffers stdout to a file.** A redirected log stays empty until exit — monitor
  the results JSON (checkpointed after every job), never the log.

### Config
- **New run-config keys MUST be added to `pyproject.toml` `[tool.flwr.app.config]` first**, or
  flwr rejects the override with *"Key 'x' is not present in the main dictionary"*.
- **`[tool.flwr.federations]` is NOT parsed by flwr 1.28.** Simulation runs via the stock
  `~/.flwr/config.toml` `local` entry: `flwr run . local --federation-config "num-supernodes=2"`.
  The field is bare `num-supernodes` — `options.num-supernodes` is rejected.

### Data / correctness
- **`psi_client.run_psi()` writes to a hard-coded global path.** It has destroyed the real
  `ordered_id_list.json` twice (once via `pytest`, once via `verify_psi.py`). Both fixed by
  redirecting `OUTPUT_PATH` at the call site — a durable fix would be an `output_path`
  parameter. **`psi/` is Wyatt's module.**
- **Never reintroduce SMOTE/upsampling.** Duplicated records break the per-record DP
  sensitivity bound and invalidate the reported ε. Same reason dedup is mandatory.
- **Do not reintroduce global-RNG snapshotting** in the client forward replay (§L2).
- **Every number before 2026-07-28 is void** (two scaling bugs). Do not cite the CSVs in
  `docs/wandb_results/`.

### Environment
- `micromamba run -n jayquery python ...` — bare `python` is not on PATH.
- Test suite: **201 passing**. Run before and after any change.

---

## 7. Things flagged but NOT done

| Item | Status |
|---|---|
| `--insecure` (no TLS) in deployment | Config issue, not research. MITM protection is TLS's job, not DP's |
| Label leakage | **Scope decision**: DP protects *features*, not labels. Server-side loss means the server holds labels by construction. Moving loss client-side does NOT fix it — the returned gradient is ∝ (p−y). Cite Li et al. 2022 |
| `measure_embedding_norms.py` | Fixed (dim inference + `--processed-dir`) |
| PSI ordering convention | Pipeline sorts numerically, PSI lexicographically. Same students, different row order — **not interchangeable**. Pipeline ordering is canonical |
| §K5 truncation | 12/20 runs peak at T=200. Costs ~+0.010–0.014 AUC, applies to all arms equally, so comparisons unaffected. State as a known bound |
