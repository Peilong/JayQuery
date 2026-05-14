# Data Dictionary

> Owner: Alex Pellet | Status: TEMPLATE — fill in during Week 2

## Silo A Features (12 columns — studentInfo.csv)

| Column | Raw Name | Encoding | Range | Notes |
|--------|----------|----------|-------|-------|
| 0 | imd_band | Ordinal (0–9) | 0–9 | Index of Multiple Deprivation band; higher = less deprived |
| 1 | highest_education | Ordinal (0–4) | 0–4 | 0=No formal quals, 4=Post-grad |
| 2 | age_band | Ordinal (0–2) | 0–2 | 0=0-35, 1=35-55, 2=55+ |
| 3 | gender | Binary | 0/1 | 0=M, 1=F |
| 4 | disability | Binary | 0/1 | 0=N, 1=Y |
| 5–11 | region | One-hot (7 cols, drop-first) | 0/1 | 8 UK regions, reference dropped |
| 12–12 | studied_credits | Standardized float | z-score | Credits registered for in this module |

**Label:** `final_result` → binary: {Withdrawn, Fail} = 1 (at-risk), {Pass, Distinction} = 0

## Silo B Features (22 columns — studentVle.csv + studentAssessment.csv)

| Column | Feature | Construction | Notes |
|--------|---------|-------------|-------|
| 0–14 | vle_click_[activity_type] | Sum of sum_click per activity_type | TODO (Alex, Week 2): list 15 activity types |
| 15 | weighted_avg_score | Mean of score weighted by assessment weight | Fill NaN with 0 |
| 16 | num_assessments_submitted | Count of submitted assessments | |
| 17 | days_active | Count of distinct days with VLE activity | |
| 18 | total_activity_types_used | Count of distinct activity_types used | |
| 19 | weeks_with_any_activity | Count of distinct weeks with VLE activity | |
| 20 | max_single_week_clicks | Max sum_click in any single week | |
| 21 | early_submission_ratio | Fraction of assessments submitted before due date | |

## Missing Value Strategy

| Field | Strategy |
|-------|---------|
| imd_band | Mode imputation within region group |
| All Silo B fields | 0-fill (no activity = 0 clicks) |
| studied_credits | Median imputation |

## Train/Val/Test Split

- Seed: 42
- Split: 80% train / 10% val / 10% test
- Stratified on label to preserve class balance
- SMOTE applied to training set ONLY (Silo A and Silo B separately)

> Fill in actual counts after running prepare_silos.py: train=___, val=___, test=___
