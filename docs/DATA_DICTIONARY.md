# Data Dictionary

> Owner: Alex Pellet | Status: TEMPLATE — fill in during Week 2

## Silo A Features (18 columns — studentInfo.csv)

| Column | Raw Name | Encoding | Range | Notes |
|--------|----------|----------|-------|-------|
| 0 | imd_band | Ordinal (0–9) | 0–9 | Index of Multiple Deprivation band; higher = less deprived |
| 1 | highest_education | Ordinal (0–4) | 0–4 | 0=No formal quals, 4=Post-grad |
| 2 | age_band | Ordinal (0–2) | 0–2 | 0=0-35, 1=35-55, 2=55+ |
| 3 | gender | Binary | 0/1 | 0=M, 1=F |
| 4 | disability | Binary | 0/1 | 0=N, 1=Y |
| 5 | studied_credits | Standardized float | z-score | Credits registered for in this module |
| 6–17 | region | One-hot (12 cols, drop-first) | 0/1 | 13 UK regions, reference dropped |


**Label:** `success` → binary: {Withdrawn, Fail} = 1 (at-risk), {Pass, Distinction} = 0

## Silo B Features (26 columns — studentVle.csv + studentAssessment.csv)

| Column | Feature | Construction | Notes |
|--------|---------|-------------|-------|
| 0 | num_assessments | Count of submitted assessments | Row Count for id_student in studentAssessment table |
| 1 | participation_rate | Sum of weights for all student assessments taken divided by the total weight for the class | Sum of weights for students/200, value from 0-1 |
| 2 | weighted_score_pct | The weighted score of all assignments against the total possible weighted score | Sum of weighted score for students/200, value from 0-1 |
| 3 | start_date | First day of activity recorded with VLE | |
| 4 | end_date | Last day of activity recorded with VLE | |
| 5 | active_days | Count of distinct days with VLE activity | |
| 6-25 | vle_click_[activity_type] | Sum of sum_click per activity_type | |

## Missing Value Strategy

| Field | Strategy |
|-------|---------|
| imd_band | Mode imputation within region group |
| All Silo B fields | 0-fill (no activity = 0 clicks) |

## Train/Val/Test Split

- Seed: 42
- Split: 80% train / 10% val / 10% test
- Stratified on label to preserve class balance
- SMOTE applied to training set ONLY (Silo A and Silo B separately)

> Fill in actual counts after running prepare_silos.py: train=___, val=___, test=___
