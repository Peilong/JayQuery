"""Prepare Silo A and Silo B data files from raw OULAD CSVs.

Owner: Alex Pellet
Run AFTER:
    - Downloading OULAD (scripts/download_oulad.sh)
    - Running PSI (python -m psi.psi_client) to get ordered_id_list.json.

Output:
  data/processed/silo_a.pkl      — {X_train, y_train, X_val, y_val, X_test, y_test}
  data/processed/silo_b.pkl      — same structure
  data/processed/scaler_a.pkl    — fitted StandardScaler for Silo A (studied_credits)
  data/processed/scaler_b.pkl    — fitted StandardScaler for Silo B (counts + VLE clicks)

Usage:
  python scripts/prepare_silos.py
  python scripts/prepare_silos.py --module AAA_2013J --seed 99
"""

import argparse
import json
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from utils.reproducibility import seed_everything

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

# Silo B columns that require standard scaling (unbounded counts / wide-range integers).
# participation_rate and weighted_score_pct are already 0-1 and excluded.
SILO_B_SCALE_COLS = ["num_assessments", "start_date", "end_date", "active_days", "sum_click"]


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


def prepare_silo_b_data(
    module: str,
    student_vle: pd.DataFrame,
    vle: pd.DataFrame,
    assessments: pd.DataFrame,
    student_assessment: pd.DataFrame,
) -> pd.DataFrame:
    """Filter, engineer, and merge Silo B features from studentVle and studentAssessment CSVs.

    Scaling of count/date columns is deferred to main() and applied post-split
    to avoid data leakage into the val/test sets.

    Args:
        module: Module presentation code, e.g. "CCC_2014J".
        student_vle: Raw studentVle DataFrame.
        vle: Raw vle DataFrame.
        assessments: Raw assessments DataFrame.
        student_assessment: Raw studentAssessment DataFrame.

    Returns:
        DataFrame with one row per student containing assessment and VLE features.
    """
    m_code, p_code = module.split("_")

    # Derive VLE click columns from all activity types across all modules so the
    # feature set is consistent regardless of which module is being processed.
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

    assessment_df = (
        student_assessment.merge(assessments, on="id_assessment", how="left")
        .query("code_module == @m_code and code_presentation == @p_code")
        .drop(columns=["is_banked", "code_module", "code_presentation", "date"])
        .assign(weighted_score=lambda x: (x["score"] * x["weight"]) / 100)
        .groupby("id_student", as_index=False)
        .agg(
            weighted_score=("weighted_score", "sum"),
            total_weight=("weight", "sum"),
            num_assessments=("id_assessment", "nunique"),
        )
        .assign(
            participation_rate=lambda x: x["total_weight"] / 200,
            weighted_score_pct=lambda x: x["weighted_score"] / 200,
        )
        .drop(columns=["total_weight", "weighted_score"])
    )

    return assessment_df.merge(vle_df, on="id_student", how="left")


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
    df = prepared_silo_a.reindex(intersection_ids)

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
        prepared_silo_b: Output of prepare_silo_b_data().
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

    vle_cols = [col for col in df.columns if col.startswith("vle_clicks_")]
    feature_cols = [
        "weighted_score_pct",
        "participation_rate",
        "num_assessments",
        "start_date",
        "end_date",
        "active_days",
        "sum_click",
        *vle_cols,
    ]

    return df[feature_cols].fillna(0).values.astype(np.float32)


def fit_and_scale(
    X_a: np.ndarray,
    X_b: np.ndarray,
    train_idx: np.ndarray,
    n_vle_cols: int,
) -> tuple[np.ndarray, np.ndarray, StandardScaler, StandardScaler]:
    """Fit StandardScalers on training rows only, then transform all rows.

    Silo A: scales the studied_credits column (index 5).
    Silo B: scales SILO_B_SCALE_COLS (indices 2-6) plus all vle_clicks_* columns.
    Scaler objects are returned for persistence to disk.

    Args:
        X_a: Full Silo A feature array (all students, pre-split).
        X_b: Full Silo B feature array (all students, pre-split).
        train_idx: Indices of training-set rows.
        n_vle_cols: Number of vle_clicks_* columns, used to compute the Silo B scale slice.

    Returns:
        Tuple of (X_a_scaled, X_b_scaled, scaler_a, scaler_b).
    """
    # Silo A: studied_credits is at index 5 (after 5 encoded cols, before region one-hots)
    scaler_a = StandardScaler()
    X_a = X_a.copy()
    X_a[train_idx, 5:6] = scaler_a.fit_transform(X_a[train_idx, 5:6])
    X_a[:, 5:6] = scaler_a.transform(X_a[:, 5:6])

    # Silo B: indices 2-6 are the count/date cols; remainder of scalable cols are vle_clicks_*
    # Layout: [weighted_score_pct, participation_rate, num_assessments, start_date,
    #          end_date, active_days, sum_click, *vle_clicks_]
    n_scale = len(SILO_B_SCALE_COLS) + n_vle_cols  # cols 2 onwards
    scale_slice = slice(2, 2 + n_scale)
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
    args = parser.parse_args()

    seed_everything(args.seed)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("[Silo Prep] Loading raw OULAD CSVs ...")
    student_info = pd.read_csv(RAW_DIR / "studentInfo.csv")
    student_vle  = pd.read_csv(RAW_DIR / "studentVle.csv")
    vle          = pd.read_csv(RAW_DIR / "vle.csv")
    student_ass  = pd.read_csv(RAW_DIR / "studentAssessment.csv")
    assessments  = pd.read_csv(RAW_DIR / "assessments.csv")

    silo_a_prep = prepare_silo_a_data(args.module, student_info)
    silo_b_prep = prepare_silo_b_data(args.module, student_vle, vle, student_ass, assessments)
    print(f"[Silo Prep] Module {args.module}: {len(silo_a_prep)} students")

    # Load PSI intersection order (written by psi_client.py)
    id_list_path = PROCESSED_DIR / "ordered_id_list.json"
    if id_list_path.exists():
        with open(id_list_path) as f:
            intersection_ids = json.load(f)
        print(f"[Silo Prep] Using PSI intersection: {len(intersection_ids)} students")
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

    n_vle_cols = sum(1 for col in silo_b_prep.columns if col.startswith("vle_clicks_"))
    X_a, X_b, scaler_a, scaler_b = fit_and_scale(X_a, X_b, idx_train, n_vle_cols)

    joblib.dump(scaler_a, PROCESSED_DIR / "scaler_a.pkl")
    joblib.dump(scaler_b, PROCESSED_DIR / "scaler_b.pkl")
    print("[Silo Prep] Scalers saved to data/processed/scaler_a.pkl and scaler_b.pkl")

    # Apply SMOTE to training set only
    smote = SMOTE(random_state=args.seed)
    X_a_train, y_train = smote.fit_resample(X_a[idx_train], y_a[idx_train])
    X_b_train, _       = smote.fit_resample(X_b[idx_train], y_a[idx_train])

    for name, X_train, X_full in [("silo_a", X_a_train, X_a), ("silo_b", X_b_train, X_b)]:
        data = {
            "X_train": X_train,
            "y_train": y_train,
            "X_val":   X_full[idx_val],
            "y_val":   y_a[idx_val],
            "X_test":  X_full[idx_test],
            "y_test":  y_a[idx_test],
            "intersection_ids": intersection_ids,
        }
        out_path = PROCESSED_DIR / f"{name}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(data, f)
        print(f"[Silo Prep] Saved {out_path} — train={len(y_train)},\
 val={len(y_a[idx_val])}, test={len(y_a[idx_test])}")


if __name__ == "__main__":
    main()