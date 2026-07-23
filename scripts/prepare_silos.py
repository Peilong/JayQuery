"""Prepare Silo A and Silo B data files from raw OULAD CSVs.

Owner: Alex Pellet
Run AFTER:
    - Downloading OULAD (scripts/download_oulad.sh)
    - Running PSI (python -m psi.psi_client) to get ordered_id_list.json.

Output:
  data/processed/silo_a.pkl      — {X_train, y_train, X_val, y_val, X_test, y_test}
  data/processed/silo_b.pkl      — same structure
  data/processed/scaler_a.pkl    — fitted StandardScaler for Silo A (studied_credits)
  data/processed/scaler_b.pkl    — fitted StandardScaler for Silo B

Silo B feature definitions (rate_v1 and the legacy v0 modes), their rationale, and
every edge-case rule live in docs/DATA_DICTIONARY.md.

Usage:
  python scripts/prepare_silos.py                       # rate_v1 features (default)
  python scripts/prepare_silos.py --features v0         # legacy full-course totals (leaky)
  python scripts/prepare_silos.py --cutoff_week 5       # legacy v0, censored to week 5
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
    "rate_v1": (
        ["weighted_score_pct", "participation_rate", "submission_ratio",
         "regularity", "start_date", "clicks_per_active_week"],
        4,
    ),
    "v0": (
        ["weighted_score_pct", "participation_rate", "num_assessments",
         "start_date", "end_date", "active_days", "sum_click"],
        2,
    ),
}


def prepare_silo_a_data(module: str, student_info: pd.DataFrame) -> pd.DataFrame:
    """Filter, encode, and impute Silo A features from studentInfo.csv.

    Scaling of studied_credits is deferred to main() and applied post-split
    to avoid data leakage into the val/test sets.

    Args:
        module: Module presentation code, e.g. "CCC_2014J".
        student_info: Raw studentInfo DataFrame.

    Returns:
        DataFrame indexed by id_student with encoded features and a binary label column.
    """

    df = student_info.copy()
    df["label"] = df["final_result"].isin(["Withdrawn", "Fail"]).astype(int)
    m_code, p_code = module.split("_")
    df = df.query("code_module == @m_code and code_presentation == @p_code").drop(
        columns=["code_module", "code_presentation", "final_result"]
    )

    # Impute missing imd_band with the mode within each region group
    df["imd_band"] = df.groupby("region")["imd_band"].transform(lambda x: x.fillna(x.mode()[0]))

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
    module: str,
    student_vle: pd.DataFrame,
    vle: pd.DataFrame,
    assessments: pd.DataFrame,
    student_assessment: pd.DataFrame,
    courses: pd.DataFrame,
) -> pd.DataFrame:
    """Engineer exposure-normalized (rate_v1) Silo B features.

    All feature definitions, window rules, and guards: docs/DATA_DICTIONARY.md.

    Args:
        module: Module presentation code, e.g. "CCC_2014J".
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

    mod_assess = assessments.query("code_module == @m_code and code_presentation == @p_code")[
        ["id_assessment", "date", "weight"]
    ]
    assess = student_assessment.merge(mod_assess, on="id_assessment", how="inner")
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
    """Engineer legacy v0 Silo B features (full-course totals, optionally censored).

    Comparison-only mode — the full-course totals leak the label; see
    docs/DATA_DICTIONARY.md (Legacy modes). rate_v1 is the default pipeline.

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
            "Silo A DataFrame. Check that ordered_id_list.json was produced for the "
            "same module as prepare_silos.py --module."
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
) -> tuple[np.ndarray, np.ndarray, StandardScaler, StandardScaler]:
    """Fit StandardScalers on training rows only, then transform all rows.

    Silo A: scales the studied_credits column (index 5).
    Silo B: scales columns [b_scale_start:] — the scaled stats plus all vle_clicks_*
    columns, which are contiguous through the end in every layout.

    Args:
        X_a: Full Silo A feature array (all students, pre-split).
        X_b: Full Silo B feature array (all students, pre-split).
        train_idx: Indices of training-set rows.
        b_scale_start: First Silo B column index to scale (from SILO_B_LAYOUTS).

    Returns:
        Tuple of (X_a_scaled, X_b_scaled, scaler_a, scaler_b).
    """
    # Silo A: studied_credits is at index 5 (after 5 encoded cols, before region one-hots)
    scaler_a = StandardScaler()
    X_a = X_a.copy()
    X_a[train_idx, 5:6] = scaler_a.fit_transform(X_a[train_idx, 5:6])
    X_a[:, 5:6] = scaler_a.transform(X_a[:, 5:6])

    scale_slice = slice(b_scale_start, X_b.shape[1])
    scaler_b = StandardScaler()
    X_b = X_b.copy()
    X_b[train_idx, scale_slice] = scaler_b.fit_transform(X_b[train_idx, scale_slice])
    X_b[:, scale_slice] = scaler_b.transform(X_b[:, scale_slice])

    return X_a, X_b, scaler_a, scaler_b


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", default="CCC_2014J", help="OULAD module presentation code")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument(
        "--features",
        choices=["rate_v1", "v0"],
        default="rate_v1",
        help="Silo B feature set (see docs/DATA_DICTIONARY.md). v0 is comparison-only.",
    )
    parser.add_argument(
        "--cutoff_week",
        type=int,
        default=None,
        help="Legacy v0 temporal censoring: restrict Silo B records to this course week. "
        "Implies --features v0.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=PROCESSED_DIR,
        help="Output directory for silo pkls/scalers (e.g. data/processed/week05 for "
        "a cutoff-week sweep). ordered_id_list.json is still read from data/processed "
        "and copied alongside the outputs.",
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
    student_vle  = pd.read_csv(RAW_DIR / "studentVle.csv")
    vle          = pd.read_csv(RAW_DIR / "vle.csv")
    student_ass  = pd.read_csv(RAW_DIR / "studentAssessment.csv")
    assessments  = pd.read_csv(RAW_DIR / "assessments.csv")
    courses      = pd.read_csv(RAW_DIR / "courses.csv")

    silo_a_prep = prepare_silo_a_data(args.module, student_info)
    if args.features == "rate_v1":
        print("[Silo Prep] Features: rate_v1 (exposure-normalized, full course).")
        silo_b_prep = prepare_silo_b_rate_v1(
            args.module, student_vle, vle, assessments, student_ass, courses
        )
    elif args.cutoff_week is not None:
        print(f"[Silo Prep] Features: legacy v0, censored to weeks <= {args.cutoff_week} "
              f"(days <= {args.cutoff_week * 7}). Comparison-only.")
        silo_b_prep = prepare_silo_b_data(
            args.module, student_vle, vle, assessments, student_ass, cutoff_week=args.cutoff_week
        )
    else:
        print(
            "[Silo Prep] WARNING: legacy v0 full-course features LEAK the label. "
            "Comparison-only — do not quote these numbers as a baseline."
        )
        silo_b_prep = prepare_silo_b_data(
            args.module, student_vle, vle, assessments, student_ass
        )
    print(f"[Silo Prep] Module {args.module}: {len(silo_a_prep)} students")

    # Load PSI intersection order (written by psi_client.py)
    id_list_path = PROCESSED_DIR / "ordered_id_list.json"
    if id_list_path.exists():
        with open(id_list_path) as f:
            intersection_ids = json.load(f)
        print(f"[Silo Prep] Using PSI intersection: {len(intersection_ids)} students")
        if out_dir != PROCESSED_DIR:
            (out_dir / "ordered_id_list.json").write_text(id_list_path.read_text())
    else:
        intersection_ids = silo_a_prep.index.tolist()
        print("[Silo Prep] WARNING: ordered_id_list.json not found — using all IDs (no PSI)")

    X_a, y_a = build_silo_a(silo_a_prep, intersection_ids)
    X_b = build_silo_b(silo_b_prep, intersection_ids)

    # Split first, then scale — scaler is fit on training rows only
    idx = np.arange(len(intersection_ids))
    idx_tv, idx_test = train_test_split(
        idx, test_size=args.test_size, random_state=args.seed, stratify=y_a
    )
    idx_train, idx_val = train_test_split(
        idx_tv,
        test_size=args.val_size / (1 - args.test_size),
        random_state=args.seed,
        stratify=y_a[idx_tv],
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
        }
        out_path = out_dir / f"{name}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(data, f)
        print(
            f"[Silo Prep] Saved {out_path} — train={len(y_train)}, "
            f"val={len(y_a[idx_val])}, test={len(y_a[idx_test])}"
        )


if __name__ == "__main__":
    main()
