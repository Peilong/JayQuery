# Metrics Glossary — what every wandb value means and when to worry

Owner: Alex Pellet

Every tracked run (baselines project + jayquery-vfl project) carries a compact version
of this guide in its wandb run **notes** (Overview tab). This file is the full reference.
All metrics come from one place: `baselines/eval_utils.py:compute_metrics()`.

The task: binary dropout-risk prediction on OULAD. Positive class = at-risk/dropout.
The classes are **imbalanced** (`base_rate` ≈ the positive fraction of the split), which is
exactly why accuracy alone is misleading and why the health checks below exist.

---

## 1. Health checks — read these first

These exist to answer one question instantly: *did the model actually learn, or did it
collapse to predicting a single class?* (One-class prediction has been a recurring
failure mode in past runs — loss and AUC alone can hide it.)

| Metric | Healthy | Failure signature |
|---|---|---|
| `pred_collapsed` | **0.0** | **1.0 = FAILED.** The single go/no-go flag. |
| `pred_std` | ≥ 0.05 (typically 0.15–0.30) | ~0 → the model outputs a near-constant probability for every student. |
| `pred_pos_rate` | near `base_rate` | 0.0 or 1.0 → literally one class predicted. Far from `base_rate` → threshold/calibration problem. |
| `pred_mean` | near `base_rate` | Drifting toward 0 or 1 → logits saturating, collapse in progress. |
| `base_rate` | (context, not a score) | The actual positive fraction. Reference point for the two rows above. |

`pred_collapsed` is defined in `eval_utils.py` as:
`pred_std < PRED_STD_FLOOR (0.05)` **or** minority predicted class < `PRED_MINORITY_FRAC_FLOOR (0.01)`.

**Why "std"?** A constant prediction has zero standard deviation. Any model that is
actually discriminating between students must spread its probabilities out. So a
single glance at `pred_std` — is it above 0.05? — tells you whether the run produced
a real classifier before you look at any quality metric.

### Collapse signatures in the *quality* metrics
- `recall = 1.0` **and** `precision ≈ base_rate` → predicting **everyone** positive.
- `recall = 0.0` (`tp = 0`) → predicting **everyone** negative.
- `accuracy ≈ 1 − base_rate` with bad f1 → the majority-class trap.
- AUC can still look mediocre-but-plausible (~0.5–0.6) during collapse — that is why
  AUC-only monitoring missed this before.

---

## 2. Quality metrics

| Metric | Meaning | Rules of thumb here |
|---|---|---|
| `auc_roc` | Probability a random positive is ranked above a random negative. Threshold-free. | 0.5 = coin flip. < 0.55 = model learned ~nothing. Centralized baselines set the ceiling; VFL should land within a few points of them. |
| `avg_precision` | Area under precision-recall curve. More honest than AUC under imbalance. | Compare to `base_rate` — that is the chance-level value, **not** 0.5. |
| `accuracy` | Fraction of correct threshold-0.5 calls. | Only meaningful next to `base_rate`; `accuracy ≈ 1 − base_rate` is the do-nothing score. |
| `f1_binary` | Harmonic mean of precision & recall for the positive class. | 0 when either precision or recall is 0 — collapses show up as ~0. |
| `f1_macro` | Mean of per-class F1s, classes weighted equally. | Punishes one-class prediction hard; a collapsed model scores ≈ 0.4–0.45 even with high accuracy. |
| `precision` | Of students flagged at-risk, how many truly are. | Low precision = advisor time wasted on false alarms. |
| `recall` | Of truly at-risk students, how many we caught. | The intervention-value metric — a missed dropout (fn) is the costly error. |
| `brier_score` | Mean squared error of the probabilities (calibration). Lower is better. | Beat `base_rate·(1−base_rate)` — the score of always predicting the base rate. |
| `tn/fp/fn/tp` | Confusion counts at threshold 0.5. | Watch `fn` (missed dropouts) and `tp` — `tp = 0` is collapse. Logged as per-round curves in VFL eval. |

---

## 3. Where each metric appears

**Baselines** (`wandb: jayquery/baseline`)
- `val/*` per epoch (MLPs): auc, f1, precision, recall + `pred_std`, `pred_pos_rate`, `pred_collapsed`.
- `test/*` summary (all models): the full suite above + confusion-matrix table.
- Quick pass/fail read on a finished run: `test/pred_collapsed == 0`,
  `test/pred_std ≥ 0.05`, `test/auc_roc` comfortably above 0.5.

**VFL deployment / sweeps** (`wandb: jayquery/jayquery-vfl`)
- Every round: `train/loss`, `train/loss_ema`, `train/lr`, `train/pred_std`,
  `train/pred_pos_rate`, `train/label_pos_rate`, `privacy/*` epsilons.
  `train/pred_pos_rate` pinned at 0 or 1 while loss still falls = the top model is
  riding class imbalance, not learning.
- Every eval round (`eval-every`): the **full** `val/*` suite as curves — recall,
  precision, f1, brier, confusion counts, and the health block. You no longer have to
  infer behavior from `val/auc` alone, and you get the metric trajectories rather than
  one final test point.
- End of run: `test/*` summary from the best-val-AUC checkpoint, plus `val/auc_best`.

**Privacy** (`privacy/*`): per-source cumulative epsilons + `privacy/combined_epsilon`
(the composed, reported guarantee). Rising monotonically each round is correct behavior;
the run stops itself before `combined-target-epsilon` is exceeded.
