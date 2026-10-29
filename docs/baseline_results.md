# Baseline Results — ICC 2027 cohort

Regenerated 2026-10-28. **Supersedes the Week 3 table**, which reported
XGBoost merged at AUC 0.9974 and logistic regression at 0.9784. Those numbers
came from the deprecated `--features v0` / `--cutoff_week` path, which leaks
post-cutoff information (CLAUDE.md, "Common Pitfalls"), and they cited
`wandb.ai/jayquery/baseline`, an entity this project's API key cannot reach,
so nothing was ever written there. Both problems are fixed below.

## Provenance

| | |
|---|---|
| cohort | `data/processed/dedup_first` — all 22 module-presentations, `--dedup-rule first` |
| students | 25,515 (train/val/test = 20,411 / 2,552 / 2,552) |
| at-risk rate | 0.445 |
| features | Silo A 18 (demographics), Silo B 26 (`rate_v1` engagement) |
| config | `experiments/configs/exp01_centralized_baselines.yaml` |
| wandb | `lip-etown-research/baseline` |
| seed | 42 |

## Two kinds of baseline, and why the distinction matters

**Ceiling baselines** are tuned to their best and take no step budget. They
answer "what is extractable from these features at all".

**Matched controls** get the *same optimization budget as the VFL run*: 200
gradient steps, batch 1,021 rows (`batch_fraction` 0.05 of the training set,
mirroring the grid's Poisson `batch-sample-rate`), and **no class weighting**,
because the VFL grid runs `pos-weight = 0`. They answer "what does this
architecture achieve for the same training effort".

That distinction was not enforced until now. `gradient_steps`,
`batch_fraction`, `class_weighting`, `grad_clip_norm` and `warmup_steps` were
passed by `run_baselines.py` and read by neither MLP, which fell back to
`epochs=50` at `batch_size=64` with a hardcoded `pos_weight`: **15,950
gradient steps with class balancing, against the federated model's 200
without**. `baselines/training_plan.py` now resolves the budget for both MLPs,
covered by `tests/test_training_plan.py` so a key cannot go dead again.

## Results (test split)

| Model | Kind | AUC-ROC | Avg Prec | Accuracy | F1 (macro) | Precision | Recall | Brier |
|---|---|---|---|---|---|---|---|---|
| XGBoost (merged) | ceiling | **0.9389** | — | 0.8691 | 0.8668 | 0.8721 | 0.8276 | 0.0960 |
| **JayQuery VFL, no DP** | system | **0.9008** | 0.9033 | 0.8202 | 0.8184 | 0.7923 | 0.8096 | 0.1216 |
| Logistic regression (merged) | ceiling | 0.8704 | — | 0.7790 | 0.7756 | 0.7607 | 0.7353 | 0.1435 |
| Isolated Silo B (engagement) | matched | 0.8511 | — | 0.7727 | 0.7633 | 0.8077 | 0.6429 | 0.1524 |
| Centralized MLP (merged) | matched | 0.8487 | — | 0.7672 | 0.7596 | 0.7831 | 0.6605 | 0.1536 |
| Isolated Silo A (demographics) | matched | 0.6118 | — | 0.5921 | 0.5217 | 0.6101 | 0.2339 | 0.2389 |

VFL figures are the mean over 4 no-DP seeds; AUC sd is 0.0030. Every other row
is a single seed-42 run.

## Reading these numbers

**The federated model sits between the ceiling and the floor.** XGBoost on the
merged features reaches 0.9389, the split VFL model 0.9008, and one office
alone with demographics 0.6118. The collaboration is worth roughly 29 AUC
points over the registrar's own data.

**At an equal optimization budget the VFL model is competitive with
centralized training**, beating the matched centralized MLP (0.8487) and
matched isolated Silo B (0.8511). This is NOT evidence that federating beats
centralizing: the VFL stack has **24,049 parameters against the centralized
MLP's 4,993**, 4.8x more, at the same step count. The honest claim is parity
at equal effort, not superiority, and any write-up should give both numbers.

| | parameters |
|---|---|
| VFL: bottom A + bottom B + top | 24,049 = 7,712 + 13,712 + 2,625 |
| centralized MLP (44 features) | 4,993 |
| isolated Silo B (26 features) | 3,841 |
| isolated Silo A (18 features) | 3,329 |

**Engagement dominates demographics.** Silo B alone (0.8511) matches the
merged MLP (0.8487) to within 0.002 at the same budget, so Silo A's 18
demographic features add little *accuracy*. They are nonetheless the features
carrying the FERPA exposure the paper is about, which is why the registrar
silo is the inversion target in C2 rather than the engagement silo.

**Gradient-boosted trees still beat the neural stack** by 3.8 AUC points, as
they usually do on tabular data. The paper's contribution is where to place DP
noise in a split architecture, not that a neural VFL model is the best
available classifier for OULAD, and claiming otherwise would be easy to
refute.

## Reproducing

```bash
python -m baselines.run_baselines \
    --config experiments/configs/exp01_centralized_baselines.yaml --run all
```

Takes about two minutes. Change the budget through the config's `training`
block; those keys are live now.
