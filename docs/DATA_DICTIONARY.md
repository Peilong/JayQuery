# Data Dictionary

> Owner: Alex Pellet

## Silo A Features (18 columns - studentInfo.csv)

| Column | Raw Name | Encoding | Range | Notes |
|--------|----------|----------|-------|-------|
| 0 | imd_band | Ordinal (0-9) | 0-9 | Index of Multiple Deprivation band; higher = less deprived |
| 1 | highest_education | Ordinal (0-4) | 0-4 | 0=No formal quals, 4=Post-grad |
| 2 | age_band | Ordinal (0-2) | 0-2 | 0=0-35, 1=35-55, 2=55+ |
| 3 | gender | Binary | 0/1 | 0=M, 1=F |
| 4 | disability | Binary | 0/1 | 0=N, 1=Y |
| 5 | studied_credits | Standardized float | z-score | Credits registered for in this module |
| 6-17 | region | One-hot (12 cols, drop-first) | 0/1 | 13 UK regions, reference dropped |

**Label:** `success` → binary: {Withdrawn, Fail} = 1 (at-risk), {Pass, Distinction} = 0

## Silo B Features (26 columns - vle.csv + studentVle.csv + assessments.csv + studentAssessment.csv)

| Column | Feature | Construction | Scaling | Notes |
|--------|---------|--------------|---------|-------|
| 0 | num_assessments | Row count for id_student in studentAssessment | StandardScaler | Unbounded count; varies with module length |
| 1 | participation_rate | Sum of assessment weights taken / total class weight | None | By definition 0-1 |
| 2 | weighted_score_pct | Sum of weighted scores / total possible weighted score | None | By definition 0-1 |
| 3 | start_date | Day offset of first VLE interaction relative to module start (e.g. -10, 0, 4) | StandardScaler | Negative values indicate pre-module engagement; captures engagement latency |
| 4 | end_date | Day offset of last VLE interaction relative to module start (e.g. 140, 196, 230) | StandardScaler | On a much larger scale than start_date; scaling prevents it from dominating |
| 5 | active_days | Count of distinct days with VLE activity | StandardScaler | Unbounded count; correlated with but distinct from end_date − start_date span |
| 6-25 | vle_click_[activity_type] | Sum of sum_click per activity_type | StandardScaler | Counts are right-skewed and unbounded; scaling prevents high-click types from dominating |

## Missing Value Strategy

| Field | Strategy |
|-------|----------|
| imd_band | Mode imputation within region group |
| All Silo B fields | 0-fill (no VLE activity = 0 clicks) |

## Train/Val/Test Split

- Seed: 42
- Split: 80% train / 10% val / 10% test
- Stratified on label to preserve class balance
- SMOTE applied to training set only (fit separately on Silo A and Silo B to avoid cross-silo leakage)
- StandardScaler fit on training set only; transform applied to val and test

> Fill in actual counts after running prepare_silos.py: train=2374, val=250, test=250