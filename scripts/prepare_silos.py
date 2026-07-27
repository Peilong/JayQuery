"""Prepare Silo A and Silo B data files from raw OULAD CSVs.

Owner: Alex Pellet
Run AFTER downloading OULAD (scripts/download_oulad.sh).

The A-B intersection is computed here and written per variant; the PSI protocol is
run separately to VERIFY it, not as a prerequisite (docs/action_plan.md §E).

Output (per --out_dir, i.e. per variant):
  silo_a.pkl            — {X_train, y_train, X_val, y_val, X_test, y_test, provenance}
  silo_b.pkl            — same structure
  scaler_a.pkl          — fitted StandardScaler for Silo A (studied_credits)
  scaler_b.pkl          — fitted StandardScaler for Silo B
  ordered_id_list.json  — A-B intersection order for this variant
  manifest.json         — every parameter + row count that produced this variant

Silo B feature definitions (rate_v1 and the legacy v0 modes), their rationale, and
every edge-case rule live in docs/DATA_DICTIONARY.md.

CURRENT vs DEPRECATED
---------------------
CURRENT (use these):
  - build_cohort()             — module filter + pre-start-withdrawal exclusion + dedup
  - prepare_silo_b_rate_v1()   — exposure-normalized Silo B features
  - SILO_B_LAYOUTS["rate_v1"]

[DEPRECATED] (retained only to reproduce historical numbers — do NOT use for new work):
  - prepare_silo_b_data()      — legacy v0 features; full-course totals LEAK the label
  - SILO_B_LAYOUTS["v0"]
  - the --features v0 and --cutoff_week CLI flags
  Deprecated paths predate the cohort refactor: they filter by a single --module
  themselves and do NOT honour the dedup / pre-start-withdrawal rules
  (docs/action_plan.md §E-a, §F), so their labels are not comparable to current runs.

CLI (cohort flags are NEW — see docs/action_plan.md §E/§F):
  --modules                 comma-separated codes, or 'all'. REPLACES the old --module.
  --dedup-rule              first (PRIMARY) | best | worst — one enrolment per student.
  --withdrawal-cutoff-day   drop withdrawals unregistering on/before this day (default 0;
                            'none' disables). Non-starters are not dropout risk.
  --out_dir                 per-variant output directory.

Usage:
  # single module (legacy default cohort), current features
  python scripts/prepare_silos.py

  # the three agreed variants, all 22 module-presentations
  python scripts/prepare_silos.py --modules all --dedup-rule first \
      --out_dir data/processed/dedup_first
  python scripts/prepare_silos.py --modules all --dedup-rule best \
      --out_dir data/processed/dedup_best
  python scripts/prepare_silos.py --modules all --dedup-rule worst \
      --out_dir data/processed/dedup_worst

  # §F4 sensitivity cut — a flag change, no code edit
  python scripts/prepare_silos.py --modules all --withdrawal-cutoff-day 14 \
      --out_dir data/processed/cutoff14

  # [DEPRECATED] legacy v0 paths — leaky, comparison-only, ignore the cohort flags
  python scripts/prepare_silos.py --features v0
  python scripts/prepare_silos.py --cutoff_week 5
"""

import argparse
import json
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from utils.reproducibility import seed_everything

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

# Per-mode Silo B stat columns + index where the StandardScaler slice starts;
# vle_clicks_* columns always follow the stats and are always scaled (docs/DATA_DICTIONARY.md).
SILO_B_LAYOUTS: dict[str, tuple[list[str], int]] = {
    # CURRENT — the only layout that should be used for reported results.
    "rate_v1": (
        ["weighted_score_pct", "participation_rate", "submission_ratio",
         "regularity", "start_date", "clicks_per_active_week"],
        4,
    ),
    # [DEPRECATED] Legacy v0 layout — full-course totals LEAK the label.
    # Retained for reproducing old comparison numbers only. Do not use for new work.
    "v0": (
        ["weighted_score_pct", "participation_rate", "num_assessments",
         "start_date", "end_date", "active_days", "sum_click"],
        2,
    ),
}

# ── Cohort construction (CURRENT pipeline) ────────────────────────────────────

DEDUP_RULES: tuple[str, ...] = ("first", "best", "worst")
"""Rules for students enrolled in more than one module-presentation.

12.3% of OULAD students appear in >1 module-presentation, and **33% of those
repeat students have MIXED outcomes** (passed one module, withdrew from another),
so the rule materially moves the label — the at-risk rate spans 0.4935 ("best")
to 0.5340 ("worst"), a 4.06-point spread. See docs/action_plan.md §E-a.

- ``first``  — earliest enrolment chronologically. **PRIMARY.** Label-neutral
  (selects on time, not outcome) and mirrors deployment: predict on a student's
  first course with no look-ahead.
- ``best`` / ``worst`` — keep the student's best/worst outcome. Label-SELECTING:
  report as sensitivity bounds only, never as the headline result.
"""

_PRESENTATION_TERM_ORDER = {"B": 0, "J": 1}
"""OULAD presentation suffix -> within-year order (B = February, J = October)."""


def _add_presentation_order(df: pd.DataFrame) -> pd.DataFrame:
    """Attach integer year/term columns so presentations can be sorted chronologically.

    Args:
        df: Frame with a ``code_presentation`` column (e.g. "2014J").

    Returns:
        A copy with ``_pres_year`` and ``_pres_term`` columns added.

    Raises:
        ValueError: If any ``code_presentation`` has an unrecognised term suffix.
    """
    out = df.copy()
    out["_pres_year"] = out["code_presentation"].str[:4].astype(int)
    out["_pres_term"] = out["code_presentation"].str[4].map(_PRESENTATION_TERM_ORDER)
    if out["_pres_term"].isna().any():
        bad = sorted(out.loc[out["_pres_term"].isna(), "code_presentation"].unique())
        raise ValueError(f"Unrecognised code_presentation term suffix(es): {bad}")
    return out


def build_cohort(
    student_info: pd.DataFrame,
    student_registration: pd.DataFrame,
    modules: list[str] | None = None,
    dedup_rule: str = "first",
    withdrawal_cutoff_day: int | None = 0,
) -> tuple[pd.DataFrame, dict]:
    """Build the deduplicated, label-corrected student cohort.

    Three things happen here, in order, and each is a decision recorded in
    docs/action_plan.md:

    1. **Module filter** — restrict to ``modules``, or keep all 22 module-presentations.
    2. **Pre-start withdrawal exclusion** (§F) — drop rows whose ``final_result`` is
       "Withdrawn" AND whose ``date_unregistration <= withdrawal_cutoff_day``. 30.4%
       of all OULAD withdrawals unregister before the course starts; they are
       non-starters, not dropout risk, and are trivially separable (66.7% have zero
       VLE clicks), which inflates AUC with easy wins.
    3. **Deduplication** (§E-a) — one row per ``id_student`` under ``dedup_rule``.
       This enforces the one-record-per-person invariant that the DP guarantee
       depends on: a student contributing k records would push true sensitivity to
       k*C and invalidate the reported epsilon.

    Args:
        student_info: Raw studentInfo DataFrame.
        student_registration: Raw studentRegistration DataFrame (supplies
            ``date_unregistration``).
        modules: Module-presentation codes to keep, e.g. ["CCC_2014J", "FFF_2013B"].
            None (default) keeps every module-presentation.
        dedup_rule: One of :data:`DEDUP_RULES`. "first" is the primary rule.
        withdrawal_cutoff_day: Withdrawals unregistering on or before this course day
            are dropped. Default 0 (before/at course start). None disables the
            exclusion entirely.

    Returns:
        Tuple of ``(cohort, stats)``. ``cohort`` is one row per student with the raw
        studentInfo columns plus a binary ``label`` (1 = Withdrawn or Fail), and is
        guaranteed unique on ``id_student``. ``stats`` is a manifest-ready dict of
        the parameters used and the row counts at each stage.

    Raises:
        ValueError: If ``dedup_rule`` is unknown, or ``modules`` names a
            module-presentation absent from the data.
    """
    if dedup_rule not in DEDUP_RULES:
        raise ValueError(f"dedup_rule must be one of {DEDUP_RULES}, got {dedup_rule!r}")

    df = student_info.copy()
    n_raw = len(df)
    df["label"] = df["final_result"].isin(["Withdrawn", "Fail"]).astype(int)
    df["module_presentation"] = df["code_module"] + "_" + df["code_presentation"]

    if modules is not None:
        unknown = sorted(set(modules) - set(df["module_presentation"].unique()))
        if unknown:
            raise ValueError(f"Unknown module-presentation(s): {unknown}")
        df = df[df["module_presentation"].isin(modules)]
    n_after_modules = len(df)

    # Step 2 — drop pre-start withdrawals (non-starters, not dropout risk).
    n_excluded_prestart = 0
    if withdrawal_cutoff_day is not None:
        unreg = student_registration[
            ["id_student", "code_module", "code_presentation", "date_unregistration"]
        ]
        df = df.merge(unreg, on=["id_student", "code_module", "code_presentation"], how="left")
        is_prestart = (df["final_result"] == "Withdrawn") & (
            df["date_unregistration"] <= withdrawal_cutoff_day
        )
        n_excluded_prestart = int(is_prestart.sum())
        df = df[~is_prestart].drop(columns=["date_unregistration"])
    n_before_dedup = len(df)

    # Step 3 — one row per student. Chronological order is the tie-break for every
    # rule, so "best"/"worst" differ from "first" only where outcomes actually differ.
    df = _add_presentation_order(df)
    sort_cols = ["id_student", "_pres_year", "_pres_term"]
    ascending: list[bool] = [True, True, True]
    if dedup_rule == "best":  # prefer label 0
        sort_cols.insert(1, "label")
        ascending.insert(1, True)
    elif dedup_rule == "worst":  # prefer label 1
        sort_cols.insert(1, "label")
        ascending.insert(1, False)

    cohort = (
        df.sort_values(sort_cols, ascending=ascending, kind="mergesort")
        .groupby("id_student", as_index=False, sort=False)
        .head(1)
        .drop(columns=["_pres_year", "_pres_term"])
        .reset_index(drop=True)
    )

    if cohort["id_student"].duplicated().any():
        raise ValueError("Dedup failed: id_student is not unique — DP sensitivity would break.")

    stats = {
        "dedup_rule": dedup_rule,
        "withdrawal_cutoff_day": withdrawal_cutoff_day,
        "modules": "all" if modules is None else sorted(modules),
        "n_module_presentations": int(cohort["module_presentation"].nunique()),
        "n_raw_rows": n_raw,
        "n_after_module_filter": n_after_modules,
        "n_excluded_prestart_withdrawals": n_excluded_prestart,
        "n_before_dedup": n_before_dedup,
        "n_dropped_by_dedup": n_before_dedup - len(cohort),
        "n_students": len(cohort),
        "label_rate": round(float(cohort["label"].mean()), 6),
    }
    return cohort, stats


def prepare_silo_a_data(cohort: pd.DataFrame) -> pd.DataFrame:
    """Encode and impute Silo A features from a :func:`build_cohort` cohort.

    Cohort selection (module filter, pre-start-withdrawal exclusion, dedup) has
    already happened in :func:`build_cohort` — this function only encodes. Module
    identity is deliberately NOT encoded as a feature: heterogeneity across
    module-presentations is handled by stratifying the split instead, which keeps
    ``input-dim-a`` stable at 18 and avoids inflating the DP parameter count
    (noise scales with sqrt(d) — see docs/dp_explained_for_alex.md §3).

    Scaling of studied_credits is deferred to main() and applied post-split
    to avoid data leakage into the val/test sets.

    Args:
        cohort: Output of :func:`build_cohort` — one row per student, with a
            binary ``label`` column already present.

    Returns:
        DataFrame indexed by id_student with encoded features and the label column.
    """
    df = cohort.copy()
    df = df.drop(
        columns=[
            c
            for c in ("code_module", "code_presentation", "final_result", "module_presentation")
            if c in df.columns
        ]
    )

    # Impute missing imd_band with the mode within each region group. Falls back to
    # the global mode if a region has no observed value at all (possible once the
    # cohort spans 22 module-presentations); no-op when every region has one.
    global_mode = df["imd_band"].mode()
    fallback = global_mode.iloc[0] if not global_mode.empty else None
    df["imd_band"] = df.groupby("region")["imd_band"].transform(
        lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else fallback)
    )

    # Ordinal encoding
    enc = OrdinalEncoder()
    df["imd_band_enc"] = enc.fit_transform(df[["imd_band"]])
    df["highest_education_enc"] = df["highest_education"].map(
        {
            "No Formal quals": 0,
            "Lower Than A Level": 1,
            "A Level or Equivalent": 2,
            "HE Qualification": 3,
            "Post Graduate Qualification": 4,
        }
    )
    df["age_band_enc"] = enc.fit_transform(df[["age_band"]])
    df.drop(columns=["imd_band", "highest_education", "age_band"], inplace=True)

    # Binary encoding
    df["gender_enc"] = (df["gender"] == "F").astype(int)
    df["disability_enc"] = (df["disability"] == "Y").astype(int)
    df.drop(columns=["gender", "disability"], inplace=True)

    # One-hot encoding for region (drop-first to avoid multicollinearity)
    ohe = OneHotEncoder(drop="first", sparse_output=False)
    region_encoded = pd.DataFrame(
        ohe.fit_transform(df[["region"]]),
        columns=ohe.get_feature_names_out(["region"]),
    )
    df = pd.concat([df.reset_index(drop=True), region_encoded.reset_index(drop=True)], axis=1)
    df.drop(columns=["region"], inplace=True)

    return df.set_index("id_student")


def prepare_silo_b_rate_v1(
    cohort: pd.DataFrame,
    student_vle: pd.DataFrame,
    vle: pd.DataFrame,
    assessments: pd.DataFrame,
    student_assessment: pd.DataFrame,
    courses: pd.DataFrame,
) -> pd.DataFrame:
    """Engineer exposure-normalized (rate_v1) Silo B features across the cohort.

    Features are computed **per module-presentation** and concatenated, because the
    assessment schedule and course length differ per presentation. Each student is
    scored only on the enrolment :func:`build_cohort` selected for them, so the
    result stays one row per ``id_student``.

    All feature definitions, window rules, and guards: docs/DATA_DICTIONARY.md.

    Args:
        cohort: Output of :func:`build_cohort`; supplies each student's chosen
            ``module_presentation``.
        student_vle: Raw studentVle DataFrame.
        vle: Raw vle DataFrame.
        assessments: Raw assessments DataFrame.
        student_assessment: Raw studentAssessment DataFrame.
        courses: Raw courses DataFrame (provides module_presentation_length).

    Returns:
        DataFrame with one row per student: id_student + the rate_v1 feature columns.

    Raises:
        ValueError: If the concatenated result is not unique on ``id_student``.
    """
    frames = [
        _rate_v1_for_presentation(
            module=mp,
            student_ids=set(grp["id_student"]),
            student_vle=student_vle,
            vle=vle,
            assessments=assessments,
            student_assessment=student_assessment,
            courses=courses,
        )
        for mp, grp in cohort.groupby("module_presentation", sort=True)
    ]
    out = pd.concat(frames, ignore_index=True)
    if out["id_student"].duplicated().any():
        raise ValueError(
            "Silo B features are not unique per id_student — a student was scored on "
            "more than one enrolment."
        )
    return out


def _rate_v1_for_presentation(
    module: str,
    student_ids: set,
    student_vle: pd.DataFrame,
    vle: pd.DataFrame,
    assessments: pd.DataFrame,
    student_assessment: pd.DataFrame,
    courses: pd.DataFrame,
) -> pd.DataFrame:
    """rate_v1 features for ONE module-presentation, restricted to ``student_ids``.

    This is the original single-module rate_v1 implementation, unchanged except for
    the ``student_ids`` restriction; :func:`prepare_silo_b_rate_v1` loops it over the
    cohort's presentations.

    Args:
        module: Module presentation code, e.g. "CCC_2014J".
        student_ids: Students whose chosen enrolment is this module-presentation.
        student_vle: Raw studentVle DataFrame.
        vle: Raw vle DataFrame.
        assessments: Raw assessments DataFrame.
        student_assessment: Raw studentAssessment DataFrame.
        courses: Raw courses DataFrame (provides module_presentation_length).

    Returns:
        DataFrame with one row per student: id_student + the rate_v1 feature columns.
    """
    m_code, p_code = module.split("_")
    t_end = float(
        courses.query("code_module == @m_code and code_presentation == @p_code")[
            "module_presentation_length"
        ].iloc[0]
    )

    all_activity_types = vle["activity_type"].unique()

    merged = (
        student_vle.merge(
            vle.drop(columns=["week_from", "week_to", "code_module", "code_presentation"]),
            on="id_site",
            how="left",
        )
        .query("code_module == @m_code and code_presentation == @p_code")
        .drop(columns=["code_module", "code_presentation"])
    )
    merged = merged[merged["id_student"].isin(student_ids)]

    mod_assess = assessments.query("code_module == @m_code and code_presentation == @p_code")[
        ["id_assessment", "date", "weight"]
    ]
    assess = student_assessment.merge(mod_assess, on="id_assessment", how="inner")
    assess = assess[assess["id_student"].isin(student_ids)]
    # Drop post-course submissions
    assess = assess[assess["date_submitted"] <= t_end]

    # Active window from VLE dates + non-banked submission dates (banked date is a sentinel)
    vle_win = merged.groupby("id_student")["date"].agg(["min", "max"])
    sub_win = (
        assess[assess["is_banked"] == 0]
        .groupby("id_student")["date_submitted"]
        .agg(["min", "max"])
    )
    win = vle_win.join(sub_win, how="outer", lsuffix="_vle", rsuffix="_sub")
    win["start_date"] = win[["min_vle", "min_sub"]].min(axis=1)
    active_end = win[["max_vle", "max_sub"]].max(axis=1).clip(upper=t_end)
    weeks_active = ((active_end - win["start_date"]) / 7.0).clip(lower=1.0)

    # Per-student due set D: dated assessments due by active_end
    dated = mod_assess.dropna(subset=["date"]).sort_values("date")
    n_due = np.searchsorted(dated["date"].values, active_end.values, side="right")
    weight_due = np.concatenate([[0.0], np.cumsum(dated["weight"].values)])[n_due]

    # Submitted set S: submissions of assessments in D
    sub = assess.merge(active_end.rename("active_end"), left_on="id_student", right_index=True)
    sub = sub[sub["date"].notna() & (sub["date"] <= sub["active_end"])]
    sub_agg = (
        sub.assign(weighted_score=lambda x: (x["score"] * x["weight"]) / 100)
        .groupby("id_student")
        .agg(
            weighted_score=("weighted_score", "sum"),
            weight_sub=("weight", "sum"),
            n_sub=("id_assessment", "nunique"),
        )
    )

    feats = win[["start_date"]].join(sub_agg, how="left")
    feats[["weighted_score", "weight_sub", "n_sub"]] = feats[
        ["weighted_score", "weight_sub", "n_sub"]
    ].fillna(0.0)
    # Empty/zero-weight D: numerators are provably 0, safe denominator keeps ratios at 0
    weight_due_safe = np.where(weight_due > 0, weight_due, 1.0)
    n_due_safe = np.where(n_due > 0, n_due, 1)
    feats["weighted_score_pct"] = feats["weighted_score"] / weight_due_safe
    feats["participation_rate"] = feats["weight_sub"] / weight_due_safe
    feats["submission_ratio"] = feats["n_sub"] / n_due_safe

    vle_stats = merged.groupby("id_student").agg(
        active_days=("date", "nunique"), sum_click=("sum_click", "sum")
    )
    feats = feats.join(vle_stats, how="left")
    feats["regularity"] = feats["active_days"].fillna(0) / (7.0 * weeks_active)
    feats["clicks_per_active_week"] = feats["sum_click"].fillna(0) / weeks_active

    activity_rates = (
        merged.groupby(["id_student", "activity_type"])["sum_click"]
        .sum()
        .unstack(fill_value=0)
        .reindex(columns=all_activity_types, fill_value=0)
        .reindex(win.index, fill_value=0)
        .div(weeks_active, axis=0)
        .rename(columns=lambda a: f"vle_clicks_{a}_per_week")
    )
    feats = feats.join(activity_rates, how="left")

    stat_cols, _ = SILO_B_LAYOUTS["rate_v1"]
    return feats[stat_cols + list(activity_rates.columns)].reset_index()


def prepare_silo_b_data(
    module: str,
    student_vle: pd.DataFrame,
    vle: pd.DataFrame,
    assessments: pd.DataFrame,
    student_assessment: pd.DataFrame,
    cutoff_week: int | None = None,
) -> pd.DataFrame:
    """[DEPRECATED] Engineer legacy v0 Silo B features (full-course totals, censored).

    .. deprecated::
        **Do not use for new work or reported results.** The full-course totals LEAK
        the label (they encode course completion, which is the thing being predicted).
        Retained only to reproduce historical comparison numbers.
        **Use :func:`prepare_silo_b_rate_v1` instead** — see docs/DATA_DICTIONARY.md
        (Legacy modes).

        This function also predates the cohort refactor: it takes a single ``module``
        string and does its own filtering, so it is NOT wired to :func:`build_cohort`
        and does not honour the dedup / pre-start-withdrawal rules in
        docs/action_plan.md §E-a and §F.

    Args:
        module: Module presentation code, e.g. "CCC_2014J".
        student_vle: Raw studentVle DataFrame.
        vle: Raw vle DataFrame.
        assessments: Raw assessments DataFrame.
        student_assessment: Raw studentAssessment DataFrame.
        cutoff_week: If set, only records dated on or before this course week are used.

    Returns:
        DataFrame with one row per student containing assessment and VLE features.
    """
    m_code, p_code = module.split("_")
    cutoff_day = None if cutoff_week is None else cutoff_week * 7

    # Global activity-type list keeps the column set consistent across modules
    all_activity_types = vle["activity_type"].unique()
    all_vle_cols = [f"vle_clicks_{a}" for a in all_activity_types]

    merged = (
        student_vle.merge(
            vle.drop(columns=["week_from", "week_to", "code_module", "code_presentation"]),
            on="id_site",
            how="left",
        )
        .query("code_module == @m_code and code_presentation == @p_code")
        .drop(columns=["code_module", "code_presentation"])
    )

    # Censor VLE records to the cutoff (covers both the pivot and student_stats below)
    if cutoff_day is not None:
        merged = merged[merged["date"] <= cutoff_day]

    activity_clicks = (
        merged.groupby(["id_student", "activity_type"], as_index=False)
        .agg(sum_click=("sum_click", "sum"))
        .pivot_table(index="id_student", columns="activity_type", values="sum_click", fill_value=0)
        .rename(columns=lambda x: f"vle_clicks_{x}")
        .reindex(columns=all_vle_cols, fill_value=0)
        .reset_index()
    )

    student_stats = merged.groupby("id_student", as_index=False).agg(
        start_date=("date", "min"),
        end_date=("date", "max"),
        active_days=("date", "nunique"),
        sum_click=("sum_click", "sum"),
    )

    vle_df = student_stats.merge(activity_clicks, on="id_student", how="left")

    # Censor assessments on DUE date; NaN-dated items (final exam) excluded under a cutoff
    assess = student_assessment.merge(assessments, on="id_assessment", how="left").query(
        "code_module == @m_code and code_presentation == @p_code"
    )
    if cutoff_day is not None:
        assess = assess[assess["date"].notna() & (assess["date"] <= cutoff_day)]

    # Denominator = total weight available by the cutoff (constant within a module)
    mod_assess = assessments.query("code_module == @m_code and code_presentation == @p_code")
    due = mod_assess
    if cutoff_day is not None:
        due = mod_assess[mod_assess["date"].notna() & (mod_assess["date"] <= cutoff_day)]
    total_weight_available = float(due["weight"].sum()) or 1.0

    assessment_df = (
        assess.drop(columns=["is_banked", "code_module", "code_presentation", "date"])
        .assign(weighted_score=lambda x: (x["score"] * x["weight"]) / 100)
        .groupby("id_student", as_index=False)
        .agg(
            weighted_score=("weighted_score", "sum"),
            total_weight=("weight", "sum"),
            num_assessments=("id_assessment", "nunique"),
        )
        .assign(
            participation_rate=lambda x: x["total_weight"] / total_weight_available,
            weighted_score_pct=lambda x: x["weighted_score"] / total_weight_available,
        )
        .drop(columns=["total_weight", "weighted_score"])
    )

    # Outer join keeps VLE-only students; NaNs are 0-filled in build_silo_b()
    return assessment_df.merge(vle_df, on="id_student", how="outer")


def build_silo_a(
    prepared_silo_a: pd.DataFrame,
    intersection_ids: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Align Silo A to PSI intersection order and extract feature and label arrays.

    Args:
        prepared_silo_a: Output of prepare_silo_a_data(), indexed by id_student.
        intersection_ids: Ordered student IDs from PSI output.

    Returns:
        Tuple of (X, y) float32 arrays aligned to intersection_ids order.
    """
    int_ids = [int(i) for i in intersection_ids]
    df = prepared_silo_a.reindex(int_ids)

    missing = df["label"].isna().sum()
    if missing > len(int_ids) * 0.01:
        raise ValueError(
            f"build_silo_a: {missing}/{len(int_ids)} intersection IDs not found in "
            "Silo A DataFrame. The intersection is derived from the cohort, so a "
            "mismatch means ordered_id_list.json came from a DIFFERENT variant — "
            "check that --modules / --dedup-rule / --withdrawal-cutoff-day match the "
            "ones recorded in this directory's manifest.json."
        )

    feature_cols = [
        "imd_band_enc",
        "highest_education_enc",
        "age_band_enc",
        "gender_enc",
        "disability_enc",
        "studied_credits",
    ]
    region_cols = [col for col in df.columns if col.startswith("region_")]

    X = df[feature_cols + region_cols].fillna(0).values.astype(np.float32)
    y = df["label"].fillna(0).values.astype(np.float32)
    return X, y


def build_silo_b(prepared_silo_b: pd.DataFrame, intersection_ids: list[str]) -> np.ndarray:
    """Align Silo B to PSI intersection order and extract feature array.

    Args:
        prepared_silo_b: Output of prepare_silo_b_rate_v1() or prepare_silo_b_data().
        intersection_ids: Ordered student IDs from PSI output.

    Returns:
        Float32 feature array aligned to intersection_ids order.
    """
    intersection_ids = [int(i) for i in intersection_ids]
    df = (
        prepared_silo_b.dropna(subset=["id_student"])
        .assign(id_student=lambda x: x["id_student"].astype(int))
        .set_index("id_student")
        .reindex(intersection_ids)
    )

    # Layout is detected from the columns the prep function emitted
    layout = "rate_v1" if "submission_ratio" in df.columns else "v0"
    stat_cols, _ = SILO_B_LAYOUTS[layout]
    vle_cols = [col for col in df.columns if col.startswith("vle_clicks_")]

    return df[stat_cols + vle_cols].fillna(0).values.astype(np.float32)


def fit_and_scale(
    X_a: np.ndarray,
    X_b: np.ndarray,
    train_idx: np.ndarray,
    b_scale_start: int,
    log1p_b: bool = True,
) -> tuple[np.ndarray, np.ndarray, StandardScaler, StandardScaler]:
    """Fit StandardScalers on training rows only, then transform all rows once.

    Silo A: scales the studied_credits column (index 5).
    Silo B: optionally log1p-compresses the non-negative click-rate columns, then
    scales columns [b_scale_start:] — the scaled stats plus all vle_clicks_* columns,
    which are contiguous through the end in every layout.

    Args:
        X_a: Full Silo A feature array (all students, pre-split).
        X_b: Full Silo B feature array (all students, pre-split).
        train_idx: Indices of training-set rows.
        b_scale_start: First Silo B column index to scale (from SILO_B_LAYOUTS).
        log1p_b: Apply log1p to structurally non-negative Silo B columns before
            standardizing. Prevents sparse activity-type columns from exploding
            (see the comment in the body). The chosen mask is stored on the
            returned scaler as ``log1p_mask_``.

    Returns:
        Tuple of (X_a_scaled, X_b_scaled, scaler_a, scaler_b).
    """
    # Silo A: studied_credits is at index 5 (after 5 encoded cols, before region one-hots).
    # Fit on train rows, then transform every row EXACTLY ONCE. (Previously this
    # wrote scaled values into the train rows and then transformed the whole array
    # again, double-scaling train and leaving it ~50x narrower than val/test.)
    scaler_a = StandardScaler()
    X_a = X_a.copy()
    scaler_a.fit(X_a[train_idx, 5:6])
    X_a[:, 5:6] = scaler_a.transform(X_a[:, 5:6])

    X_b = X_b.copy()
    scale_slice = slice(b_scale_start, X_b.shape[1])
    block = X_b[:, scale_slice].copy()

    # Compress the heavy right tail of the click-rate columns BEFORE standardizing.
    # Sparse activity types (used by only a few of the 22 presentations) are ~all-zero
    # for most students, so their training std is near zero and the handful of active
    # students get divided by it — blowing scaled values up to ~1e5. log1p is applied
    # only to columns that are structurally non-negative in TRAIN (start_date ranges
    # negative, so it is excluded automatically).
    if log1p_b:
        log1p_mask = block[train_idx].min(axis=0) >= 0
        if log1p_mask.any():
            block[:, log1p_mask] = np.log1p(np.clip(block[:, log1p_mask], 0.0, None))
    else:
        log1p_mask = np.zeros(block.shape[1], dtype=bool)

    scaler_b = StandardScaler()
    scaler_b.fit(block[train_idx])
    X_b[:, scale_slice] = scaler_b.transform(block)
    # Travels with the joblib dump so the transform can be reproduced from the artifact.
    scaler_b.log1p_mask_ = log1p_mask

    return X_a, X_b, scaler_a, scaler_b


def _cutoff_day_arg(value: str) -> int | None:
    """Parse --withdrawal-cutoff-day: an int, or "none" to disable the exclusion."""
    return None if value.strip().lower() == "none" else int(value)


def _stratify_key(cohort: pd.DataFrame, intersection_ids: list[int]) -> np.ndarray:
    """Module-presentation x label strata, aligned to ``intersection_ids`` order.

    Stratifying on the pair keeps both the 22-presentation mix and the class balance
    intact in every split — the agreed alternative to encoding module as a feature
    (docs/action_plan.md §E).

    Args:
        cohort: Output of :func:`build_cohort`.
        intersection_ids: Ordered student IDs the split will index.

    Returns:
        Array of "<module_presentation>|<label>" strings, one per intersection id.
    """
    lookup = cohort.set_index("id_student")
    sub = lookup.reindex(intersection_ids)
    return (sub["module_presentation"].astype(str) + "|" + sub["label"].astype(str)).values


def _split_indices(
    idx: np.ndarray, strat: np.ndarray, y: np.ndarray, val_size: float, test_size: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train/val/test split, falling back to label-only if strata are too thin.

    ``train_test_split`` requires >= 2 members per stratum; a small cohort (or a rare
    presentation) can violate that on the module x label key, so we degrade to
    label-only rather than fail.

    Returns:
        Tuple of (idx_train, idx_val, idx_test).
    """

    def _try(key: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tv, test = train_test_split(idx, test_size=test_size, random_state=seed, stratify=key)
        train, val = train_test_split(
            tv, test_size=val_size / (1 - test_size), random_state=seed, stratify=key[tv]
        )
        return train, val, test

    try:
        return _try(strat)
    except ValueError as exc:
        print(
            f"[Silo Prep] WARNING: module x label stratification failed ({exc}); "
            "falling back to label-only stratification."
        )
        return _try(y)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare Silo A/B datasets from raw OULAD CSVs. "
        "Cohort selection (modules, dedup rule, withdrawal cutoff) is parameterized "
        "so a variant can be regenerated with a flag change — see docs/action_plan.md §E."
    )
    parser.add_argument(
        "--modules",
        default="CCC_2014J",
        help="Comma-separated module-presentation codes (e.g. 'CCC_2014J,FFF_2013B'), "
        "or 'all' for every presentation in the data. REPLACES the old --module flag.",
    )
    parser.add_argument(
        "--dedup-rule",
        choices=DEDUP_RULES,
        default="first",
        help="How to pick one enrolment for students appearing in several presentations. "
        "'first' (chronological) is the PRIMARY rule; 'best'/'worst' select on outcome "
        "and are for sensitivity bounds only (docs/action_plan.md §E-a).",
    )
    parser.add_argument(
        "--withdrawal-cutoff-day",
        type=_cutoff_day_arg,
        default=0,
        help="Drop withdrawals that unregistered on/before this course day — they are "
        "non-starters, not dropout risk (docs/action_plan.md §F). Default 0 "
        "(before/at course start). Pass 'none' to keep them.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument(
        "--features",
        choices=["rate_v1", "v0"],
        default="rate_v1",
        help="Silo B feature set (see docs/DATA_DICTIONARY.md). "
        "[DEPRECATED] v0 leaks the label — comparison-only, never for reported results.",
    )
    parser.add_argument(
        "--cutoff_week",
        type=int,
        default=None,
        help="[DEPRECATED] Legacy v0 temporal censoring: restrict Silo B records to this "
        "course week. Implies --features v0.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=PROCESSED_DIR,
        help="Output directory for this variant's silo pkls, scalers, ordered_id_list.json "
        "and manifest.json (e.g. data/processed/dedup_first). Point the FL app at it via "
        "the processed-dir run-config key.",
    )
    args = parser.parse_args()

    # Censoring only exists in the v0 pipeline
    if args.cutoff_week is not None:
        args.features = "v0"

    seed_everything(args.seed)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[Silo Prep] Loading raw OULAD CSVs ...")
    student_info = pd.read_csv(RAW_DIR / "studentInfo.csv")
    student_reg  = pd.read_csv(RAW_DIR / "studentRegistration.csv")
    student_vle  = pd.read_csv(RAW_DIR / "studentVle.csv")
    vle          = pd.read_csv(RAW_DIR / "vle.csv")
    student_ass  = pd.read_csv(RAW_DIR / "studentAssessment.csv")
    assessments  = pd.read_csv(RAW_DIR / "assessments.csv")
    courses      = pd.read_csv(RAW_DIR / "courses.csv")

    modules = None if args.modules.strip().lower() == "all" else [
        m.strip() for m in args.modules.split(",") if m.strip()
    ]
    cohort, cohort_stats = build_cohort(
        student_info,
        student_reg,
        modules=modules,
        dedup_rule=args.dedup_rule,
        withdrawal_cutoff_day=args.withdrawal_cutoff_day,
    )
    print(
        f"[Silo Prep] Cohort: {cohort_stats['n_students']} students across "
        f"{cohort_stats['n_module_presentations']} presentation(s) "
        f"[rule={args.dedup_rule}, cutoff={args.withdrawal_cutoff_day}] — "
        f"excluded {cohort_stats['n_excluded_prestart_withdrawals']} pre-start withdrawal(s), "
        f"dropped {cohort_stats['n_dropped_by_dedup']} duplicate enrolment(s); "
        f"label rate {cohort_stats['label_rate']:.4f}"
    )

    silo_a_prep = prepare_silo_a_data(cohort)
    if args.features == "rate_v1":
        print("[Silo Prep] Features: rate_v1 (exposure-normalized, full course).")
        silo_b_prep = prepare_silo_b_rate_v1(
            cohort, student_vle, vle, assessments, student_ass, courses
        )
    else:
        # [DEPRECATED] v0 paths predate the cohort refactor: they filter by a single
        # module themselves and ignore the dedup / withdrawal rules above.
        legacy_module = (modules or [cohort_stats["modules"]])[0]
        if args.cutoff_week is not None:
            print(f"[Silo Prep] [DEPRECATED] legacy v0, censored to weeks <= {args.cutoff_week} "
                  f"(days <= {args.cutoff_week * 7}). Comparison-only.")
            silo_b_prep = prepare_silo_b_data(
                legacy_module, student_vle, vle, assessments, student_ass,
                cutoff_week=args.cutoff_week,
            )
        else:
            print(
                "[Silo Prep] WARNING: [DEPRECATED] legacy v0 full-course features LEAK the "
                "label. Comparison-only — do not quote these numbers as a baseline."
            )
            silo_b_prep = prepare_silo_b_data(
                legacy_module, student_vle, vle, assessments, student_ass
            )
        print("[Silo Prep] WARNING: v0 ignores --dedup-rule and --withdrawal-cutoff-day; "
              "its labels are NOT comparable to rate_v1 runs.")

    # Intersection computed directly (docs/action_plan.md §E: PSI is verified separately,
    # not run per variant). Written per variant so each dataset is self-contained.
    ids_b = set(silo_b_prep["id_student"].dropna().astype(int))
    intersection_ids = sorted(set(silo_a_prep.index.astype(int)) & ids_b)
    if not intersection_ids:
        raise ValueError("Empty A-B intersection — check the module filter and raw CSVs.")
    (out_dir / "ordered_id_list.json").write_text(json.dumps(intersection_ids))
    print(
        f"[Silo Prep] Intersection: {len(intersection_ids)} students "
        f"({len(silo_a_prep) - len(intersection_ids)} in A had no Silo B records) "
        f"-> {out_dir / 'ordered_id_list.json'}"
    )

    X_a, y_a = build_silo_a(silo_a_prep, intersection_ids)
    X_b = build_silo_b(silo_b_prep, intersection_ids)

    # Split first, then scale — scaler is fit on training rows only.
    # Stratified on module-presentation x label (docs/action_plan.md §E).
    idx = np.arange(len(intersection_ids))
    strat = _stratify_key(cohort, intersection_ids)
    idx_train, idx_val, idx_test = _split_indices(
        idx, strat, y_a, args.val_size, args.test_size, args.seed
    )

    _, b_scale_start = SILO_B_LAYOUTS[args.features]
    X_a, X_b, scaler_a, scaler_b = fit_and_scale(X_a, X_b, idx_train, b_scale_start)

    joblib.dump(scaler_a, out_dir / "scaler_a.pkl")
    joblib.dump(scaler_b, out_dir / "scaler_b.pkl")
    print(f"[Silo Prep] Scalers saved to {out_dir}/scaler_a.pkl and scaler_b.pkl")

    X_a_train, X_b_train = X_a[idx_train], X_b[idx_train]
    y_train = y_a[idx_train]

    for name, X_train, X_full in [("silo_a", X_a_train, X_a), ("silo_b", X_b_train, X_b)]:
        data = {
            "idx_train": idx_train,
            "idx_val":   idx_val,
            "idx_test":  idx_test,
            "X_train": X_train,
            "y_train": y_train, #TODO: Should be removed in silo b
            "X_val":   X_full[idx_val],
            "y_val":   y_a[idx_val],
            "X_test":  X_full[idx_test],
            "y_test":  y_a[idx_test],
            "intersection_ids": intersection_ids,
            # Provenance for cross-run comparisons (docs/DATA_DICTIONARY.md)
            "feature_version": args.features,
            "cutoff_week": args.cutoff_week,
            # Cohort provenance — which variant produced this file
            "dedup_rule": args.dedup_rule,
            "withdrawal_cutoff_day": args.withdrawal_cutoff_day,
            "modules": cohort_stats["modules"],
        }
        out_path = out_dir / f"{name}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(data, f)
        print(
            f"[Silo Prep] Saved {out_path} — train={len(y_train)}, "
            f"val={len(y_a[idx_val])}, test={len(y_a[idx_test])}"
        )

    # Manifest: every parameter that produced this variant, so any result traces
    # back to its data (docs/action_plan.md §E design constraint 4).
    manifest = {
        **cohort_stats,
        "feature_version": args.features,
        "cutoff_week": args.cutoff_week,
        "seed": args.seed,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "stratified_on": "module_presentation x label",
        "silo_b_log1p_cols": int(getattr(scaler_b, "log1p_mask_", np.zeros(0)).sum()),
        "n_intersection": len(intersection_ids),
        "n_train": int(len(idx_train)),
        "n_val": int(len(idx_val)),
        "n_test": int(len(idx_test)),
        "label_rate_train": round(float(y_a[idx_train].mean()), 6),
        "label_rate_val": round(float(y_a[idx_val].mean()), 6),
        "label_rate_test": round(float(y_a[idx_test].mean()), 6),
        "input_dim_a": int(X_a.shape[1]),
        "input_dim_b": int(X_b.shape[1]),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[Silo Prep] Manifest -> {out_dir / 'manifest.json'}")
    print(
        f"[Silo Prep] input-dim-a={manifest['input_dim_a']} "
        f"input-dim-b={manifest['input_dim_b']} "
        "(must match pyproject.toml [tool.flwr.app.config])"
    )


if __name__ == "__main__":
    main()
