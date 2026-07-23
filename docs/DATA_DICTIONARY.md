# Data Dictionary

> Owner: Alex Pellet
> Feature version: **rate_v1** (2026-07-23) — exposure-normalized Silo B features.
> Supersedes the v0 full-course totals and the `--cutoff_week` censored variants,
> both of which remain available as comparison modes (see [Legacy modes](#legacy-modes-comparison-only)).

---

## Why the rate_v1 restructure

### The leak in the v0 features

Silo B's original features were **totals aggregated over the full ~39-week course**
(`sum_click`, `num_assessments`, `active_days`, per-activity click sums) plus the raw
**`end_date`** of last VLE interaction. For a student who withdrew in week 4, every one of
these encodes the outcome rather than predicting it: activity goes silent after withdrawal,
so low totals *are* the label, and `end_date` is effectively the withdrawal date itself.
Models trained on v0 features score deceptively well by reading the outcome back out of
the features (see `docs/results_integrity_audit.md`).

### Why not just keep the cutoff-week fix

The `--cutoff_week K` censoring (restrict all Silo B records to days ≤ 7K) is an honest
"knowable at week K" simulation, but it has real costs:

- It **discards most of the course** — a week-5 cutoff uses ~13% of the term's data.
- **K is arbitrary**, forcing a sweep (weeks 3/5/8/10) and multiple processed datasets.
- Small K starves the model of signal; large K lets the leak creep back in.

### The rate_v1 idea: normalize by exposure, drop duration

Instead of truncating the data, rate_v1 uses the **full course** but removes *duration of
presence* from the feature set. Every engagement quantity is expressed **relative to the
student's own active window**:

- Click totals become **clicks per active week** — intensity, not longevity.
- Submission counts become **ratios over what was due while the student was active** —
  compliance, not exposure.
- Score sums are re-denominated to the **weight due during the active window** — quality
  of work done, not fraction of the whole course completed.
- `end_date` is **dropped outright** — no transformation launders a withdrawal date.
- `active_days` becomes a **regularity ratio** (active days per week of window) — cadence,
  not a total.

The active-window length itself (`weeks_active`) is used only as a denominator and is
**never emitted as a feature**; nothing in the emitted set is monotone in how long the
student stayed.

### What this does and does not claim

- **Residual signal is intentional.** A student's *per-week* intensity may still taper
  before withdrawal. That is legitimate early-warning signal — the kind a deployed
  early-alert system keys on — not a mechanical encoding of the label. What rate_v1
  removes is the duration/totals channel, where the outcome was readable directly.
- **Deployment semantics now match training.** For a live mid-course student, every
  rate_v1 feature is computable over their "so-far" window. v0 totals were only
  computable in hindsight.
- **Verification, not assumption.** The restructure is validated empirically: per-feature
  single-variable AUC vs. label (new set must not approach `end_date`/`sum_click` v0
  levels) and a three-way model ablation — v0 full-course vs. v0 cutoff-5 vs. rate_v1.
  If rate_v1 lands near v0 full-course, that is a red flag to investigate, not a win.

---

## Derived quantities (computed per student, NOT emitted as features)

| Quantity | Definition |
|---|---|
| `T_end` | Course end: `module_presentation_length` from `courses.csv` (269 days for CCC_2014J). |
| `active_start` | min(first VLE interaction date, first `date_submitted`). Days from course start; can be negative (pre-module engagement). |
| `active_end` | max(last VLE interaction date, last `date_submitted`), capped at `T_end`. |
| `weeks_active` | max((`active_end` − `active_start`) / 7, 1.0). Float; the min-1 clamp handles single-day students. |
| Due set `D` | Module assessments with a non-NaN due date ≤ `active_end`. Exams (NaN due date) are always excluded, as in v0/cutoff modes. |
| Submitted set `S` | Student's submissions of assessments in `D`, with `date_submitted` ≤ `T_end`. |

Definitional details (deliberate, flag if you disagree):

1. **Post-course submissions are dropped** (`date_submitted` > `T_end`) — records after the
   presentation ends add skew, not signal. Requires loading `courses.csv` (new).
2. **`D` runs from course start, not `active_start`.** Assessments due before a late
   starter's first activity still count in the denominator: missing them is real,
   knowable-at-the-time risk signal, and `start_date` (kept as a feature) contextualizes
   it. Excluding them would let a week-10 starter score perfect compliance having missed
   a third of the course.
3. **Early submissions of assessments due after `active_end` are excluded from both `S`
   and `D`** (rare: student submits ahead of a due date, then departs). Keeps every ratio
   ≤ 1 by construction.
4. **`active_end` includes submission dates** so a student whose last act is an assessment
   submission is credited as active through it — numerators and denominators stay
   consistent.
5. **Banked submissions (`is_banked = 1`) are excluded from the active-window
   computation** but still count in `S` (and score numerators) when their assessment is
   in `D`. OULAD encodes banked results with the sentinel `date_submitted = -1` (all 137
   banked rows in CCC_2014J) — it is not a real activity timestamp, and letting it set
   `active_start = -1` would silently inflate `weeks_active` and dilute every rate for
   any student carrying a banked result. Side effect: a banked-only student with no
   activity in this presentation has no window, hence `D` = ∅ and all-zero features —
   consistent with the "no evidence of engagement" 0-fill semantics.
6. **Empty due set (`D` = ∅) → all three assessment ratios are 0.** This is a real
   population, not a corner case: **215 of 2305 students (~9%)** in CCC_2014J have
   `active_end` before the first due date (day 18). Compliance is undefined for them;
   0 is the "no evidence" value consistent with the missing-value strategy, and the
   model must lean on the VLE-rate features for these students. The divide-by-zero
   guard therefore covers both `D` = ∅ and zero total weight in `D`.

---

## Silo A Features (18 columns — studentInfo.csv) — unchanged in rate_v1

| Column | Raw Name | Encoding | Range | Notes |
|--------|----------|----------|-------|-------|
| 0 | imd_band | Ordinal (0–9) | 0–9 | Index of Multiple Deprivation band; higher = less deprived. |
| 1 | highest_education | Ordinal (0–4) | 0–4 | 0 = No formal quals … 4 = Post-grad. |
| 2 | age_band | Ordinal (0–2) | 0–2 | 0 = 0–35, 1 = 35–55, 2 = 55+. |
| 3 | gender | Binary | 0/1 | 0 = M, 1 = F. |
| 4 | disability | Binary | 0/1 | 0 = N, 1 = Y. |
| 5 | studied_credits | Standardized float | z-score | Credits registered for in this module. |
| 6–17 | region | One-hot (12 cols, drop-first) | 0/1 | 13 UK regions, reference level dropped. |

**Label:** `final_result` → binary: {Withdrawn, Fail} = 1 (at-risk), {Pass, Distinction} = 0.

## Silo B Features (26 columns — studentVle.csv + vle.csv + assessments.csv + studentAssessment.csv + courses.csv)

Column order groups the four unscaled ratios first so the StandardScaler slice
(columns 4–25) is contiguous.

| Column | Feature | Construction | Scaling | Notes |
|--------|---------|--------------|---------|-------|
| 0 | weighted_score_pct | Σ over `S` of (score × weight / 100) ÷ Σ over `D` of weight | None | ∈ [0, 1]. Quality of work relative to what was due while active. Replaces v0's whole-course (/200) denominator. |
| 1 | participation_rate | Σ over `S` of weight ÷ Σ over `D` of weight | None | ∈ [0, 1]. Weight-based compliance. Guard: `D` = ∅ or zero total weight in `D` → 0 (see definitional detail 6). |
| 2 | submission_ratio | \|`S`\| ÷ \|`D`\| | None | ∈ [0, 1]. Count-based compliance; the only feature that sees **zero-weight formative** assessments (invisible to weight-based ratios; some OULAD modules have them). Replaces v0 `num_assessments` raw count. |
| 3 | regularity | active_days ÷ (7 × `weeks_active`) | None | ≈ (0, 1]; boundary effects can nudge slightly above 1. `active_days` = distinct days with ≥1 VLE click. Replaces v0 raw `active_days` total. |
| 4 | start_date | `active_start` (day offset from module start) | StandardScaler | Negative = pre-module engagement. Kept raw: engagement latency is knowable and predictive, not outcome-derived. (Now min of VLE/submission activity; v0 was VLE-only.) |
| 5 | clicks_per_active_week | total `sum_click` ÷ `weeks_active` | StandardScaler | Intensity, not longevity. Replaces v0 raw `sum_click` total. |
| 6–25 | vle_clicks_[activity_type]_per_week | per-activity-type `sum_click` sum ÷ `weeks_active` | StandardScaler | 20 columns from the global `vle.csv` activity-type list (absent types 0-filled). Replace v0 raw per-type totals. |

**Dropped from v0 with no replacement:** `end_date` — it is effectively the withdrawal
date and the single largest leak; no transformation of it is admissible.

---

## Legacy modes (comparison only)

Kept solely for the leak-quantification ablation. **Do not quote their numbers as
baselines.**

| Mode | Invocation | Silo B layout |
|---|---|---|
| v0 full-course | `prepare_silos.py` legacy path | 27 cols: [weighted_score_pct, participation_rate, num_assessments, start_date, end_date, active_days, sum_click, vle_clicks_* ×20] — raw totals, whole-course denominators. Leaks the label. |
| v0 cutoff-K | `prepare_silos.py --cutoff_week K` | Same 27 cols, all records censored to days ≤ 7K. Honest "knowable at week K" but discards most of the course. Sweep dirs: `data/processed/week{03,05,08,10}`. |

`input-dim-b` is **26 for rate_v1** and **27 for legacy modes** — keep
`pyproject.toml` in sync with whichever dataset is in `data/processed/`.

---

## Missing Value Strategy

| Field | Strategy |
|-------|----------|
| imd_band (Silo A) | Mode imputation within region group. |
| All Silo B fields | 0-fill after the outer VLE↔assessment join: no record = no activity. Students with clicks but no submissions get 0 compliance ratios; students absent from both sources get all-zero rows. |

## Train/Val/Test Split

- Seed: 42; split 80% train / 10% val / 10% test, stratified on label.
- Counts (CCC_2014J, PSI intersection = 2305): train = 1843, val = 231, test = 231.
- Class imbalance handled by **weighting in the top-model loss** (server side) — SMOTE
  was removed from silo prep (it fabricated cross-silo-inconsistent synthetic rows).
- StandardScaler fit on training rows only; transform applied to all rows. Split
  happens **before** scaling.

## Provenance

Each `silo_{a,b}.pkl` records its lineage: `feature_version` (`"rate_v1"` or absent for
legacy) and `cutoff_week` (None unless a legacy cutoff run). Check these before comparing
runs across datasets.
