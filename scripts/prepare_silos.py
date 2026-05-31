"""Prepare Silo A and Silo B data files from raw OULAD CSVs.

Owner: Alex Pellet
Run AFTER downloading OULAD (scripts/download_oulad.sh) and AFTER running PSI
(python -m psi.psi_client) to get ordered_id_list.json.

Output:
  data/processed/silo_a.pkl  — {X_train, y_train, X_val, y_val, X_test, y_test}
  data/processed/silo_b.pkl  — same structure

Usage:
  python scripts/prepare_silos.py
  python scripts/prepare_silos.py --module AAA_2013J --seed 99
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder

from utils.reproducibility import seed_everything

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

def prepare_silo_a_data(
    module: str,
    student_info: pd.DataFrame
) -> pd.DataFrame:
    """Prepares the data from the raw csv file for usage in Silo A. 
    Encodes & Normalizes the features.

    Args:
        student_info: Raw studentInfo DataFrame.

    Returns:
        Dataframe of the prepared data
    """
    df = student_info.copy()
    df["label"] = df["final_result"].isin(["Withdrawn", "Fail"]).astype(int)
    m_code, p_code = module.split("_")
    df = df.query("code_module == @m_code and code_presentation == @p_code").drop(columns=['code_module', 'code_presentation','final_result'])
    
    #impute null imd_band with mode of the region
    df['imd_band'] = df.groupby('region')['imd_band'].transform(lambda x: x.fillna(x.mode()[0]))    
    
    #Ordinal encoding
    enc = OrdinalEncoder()
    df['imd_band_enc'] = enc.fit_transform(df[['imd_band']])
    df['highest_education_enc'] = df['highest_education'].map({'No Formal quals':0,'Lower Than A Level':1,'A Level or Equivalent':2,'HE Qualification':3,'Post Graduate Qualification':4})
    df['age_band_enc'] = enc.fit_transform(df[['age_band']])
    df.drop(axis=1,inplace=True,labels=['imd_band','highest_education','age_band'])
    
    #Binary Encoding
    df["gender_enc"] = (df["gender"] == "F").astype(int)
    df["disability_enc"] = (df["disability"] == "Y").astype(int)
    df.drop(axis=1,inplace=True,labels=['gender','disability'])
    
    #One Hot Encoding
    ohe = OneHotEncoder(drop='first',sparse_output=False)
    region_encoded = ohe.fit_transform(df[['region']])
    region_encoded = pd.DataFrame(region_encoded, columns=ohe.get_feature_names_out(['region']))
    df = pd.concat([df.reset_index(drop=True),region_encoded.reset_index(drop=True)],axis=1)
    df.drop(axis=1,inplace=True,labels=['region'])
    
    #Normalization
    scaler = StandardScaler()
    df['studied_credits_norm'] = scaler.fit_transform(df[['studied_credits']].fillna(0))
    df.drop(axis=1,inplace=True,labels=['studied_credits'])
    return df.set_index("id_student")

def prepare_silo_b_data(
    module: str,
    student_vle: pd.DataFrame,
    vle:pd.DataFrame,
    assessments: pd.DataFrame,
    student_assessment: pd.DataFrame
) -> pd.DataFrame:
    """Prepares the data from the raw csv file for usage in Silo B. 
    Does Cleaning, Feature Engineering, and Merging of datasets.

    Args:
        module: the argument from the combination of code_module and code_presentation
        student_vle: Raw studentVle DataFrame.
        vle: Raw vle DataFrame
        assessments: Raw assessments DataFrame
        student_assessment: Raw studentAssessment DataFrame

    Returns:
        Dataframe of the prepared data
    """
    m_code, p_code = module.split("_")

    # Get the full list of activity types across ALL modules
    all_activity_types = vle['activity_type'].unique()
    all_vle_cols = [f'vle_clicks_{a}' for a in all_activity_types]

    merged = (
        student_vle
        .merge(vle.drop(columns=['week_from', 'week_to','code_module','code_presentation']), on='id_site', how='left')
        .query("code_module == @m_code and code_presentation == @p_code")
        .drop(columns=['code_module', 'code_presentation'])
    )

    activity_clicks = (
        merged
        .groupby(['id_student', 'activity_type'], as_index=False)
        .agg(sum_click=('sum_click', 'sum'))
        .pivot_table(index='id_student', columns='activity_type', values='sum_click', fill_value=0)
        .rename(columns=lambda x: f'vle_clicks_{x}')
        .reindex(columns=all_vle_cols, fill_value=0)  # enforce all columns
        .reset_index()
    )

    student_stats = (
        merged
        .groupby('id_student', as_index=False)
        .agg(
            start_date=('date', 'min'),
            end_date=('date', 'max'),
            active_days=('date', 'nunique'),
            sum_click=('sum_click', 'sum')
        )
    )

    vle_df = student_stats.merge(activity_clicks, on='id_student', how='left')
    
    assessment_df = (
        student_assessment
        .merge(assessments, on='id_assessment', how='left')
        .query("code_module == @m_code and code_presentation == @p_code")
        .drop(columns=['is_banked', 'code_module', 'code_presentation', 'date'])
        .assign(weighted_score=lambda x: (x['score'] * x['weight']) / 100)
        .groupby('id_student', as_index=False)
        .agg(
            weighted_score=('weighted_score', 'sum'),
            total_weight=('weight', 'sum'),
            num_assessments=('id_assessment', 'nunique')
        )
        .assign(
            participation_rate=lambda x: x['total_weight'] / 200, #200 is the maximum of the total_weight column
            weighted_score_pct=lambda x: x['weighted_score'] / 200 #200 is the theoretical maximum of the weight_score column
        )
        .drop(columns=['total_weight','weighted_score'])
    )
    
    silo_b = assessment_df.merge(vle_df, on='id_student', how='left')
    
    return silo_b

def build_silo_a(
    prepared_silo_a: pd.DataFrame,
    intersection_ids: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Build Silo A feature matrix from prepared_silo_a DataFrame created in prepare_silo_a_data().

    Args:
        prepared_silo_a: The prepared Silo A DataFrame.
        intersection_ids: Ordered list of student IDs from PSI output.

    Returns:
        Tuple of (X, y) arrays aligned to intersection_ids order.
    """
    df = prepared_silo_a.reindex(intersection_ids)

    feature_cols = [
        "imd_band_enc", "highest_education_enc", "age_band_enc",
        "gender_enc", "disability_enc", "studied_credits_norm"
    ]
    region_cols = [col for col in df.columns if col.startswith("region_")]

    X = df[feature_cols + region_cols].fillna(0).values.astype(np.float32)
    y = df["label"].fillna(0).values.astype(np.float32)
    return X, y

def build_silo_b(
    prepared_silo_b: pd.DataFrame,
    intersection_ids: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build Silo B feature matrix from prepared_silo_b DataFrame created in prepare_silo_b_data().

    Args:
        prepared_silo_b: The prepared Silo B DataFrame.
        intersection_ids: Ordered list of student IDs from PSI output.

    Returns:
        X an array of the Silo B features aligned to intersection_ids order.
    """
    intersection_ids = [int(i) for i in intersection_ids]
    df = (
        prepared_silo_b
        .dropna(subset=["id_student"])
        .assign(id_student=lambda x: x["id_student"].astype(int))
        .set_index("id_student")
        .reindex(intersection_ids)
    )

    vle_cols = [col for col in df.columns if col.startswith("vle_clicks_")]
    feature_cols = [
        "weighted_score_pct", "participation_rate", "num_assessments",
        "start_date", "end_date", "active_days", "sum_click",
        *vle_cols,
    ]

    X = df[feature_cols].fillna(0).values.astype(np.float32)
    return X


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", default="BBB_2013J", help="OULAD module presentation code")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--test_size", type=float, default=0.1)
    args = parser.parse_args()

    seed_everything(args.seed)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Load OULAD CSVs
    print("[Silo Prep] Loading raw OULAD CSVs ...")
    student_info = pd.read_csv(RAW_DIR / "studentInfo.csv")
    student_vle  = pd.read_csv(RAW_DIR / "studentVle.csv")
    vle          = pd.read_csv(RAW_DIR / "vle.csv")
    student_ass  = pd.read_csv(RAW_DIR / "studentAssessment.csv")
    assessments  = pd.read_csv(RAW_DIR / "assessments.csv")

    # Prepare Data for each silo before creation.
    silo_a_prep = prepare_silo_a_data(args.module, student_info)
    silo_b_prep = prepare_silo_b_data(args.module, student_vle, vle, student_ass, assessments)
    print(f"[Silo Prep] Module {args.module}: {len(student_info)} students")

    # Load PSI intersection order (written by psi_client.py)
    id_list_path = PROCESSED_DIR / "ordered_id_list.json"
    if id_list_path.exists():
        with open(id_list_path) as f:
            intersection_ids = json.load(f)
        print(f"[Silo Prep] Using PSI intersection: {len(intersection_ids)} students")
    else:
        # Fallback: use all student IDs (run PSI first in production)
        intersection_ids = silo_a_prep.index.tolist()
        print("[Silo Prep] WARNING: ordered_id_list.json not found — using all IDs (no PSI)")

    # Build feature matrices
    X_a, y_a = build_silo_a(silo_a_prep, intersection_ids)
    X_b      = build_silo_b(silo_b_prep, intersection_ids)

    # Train/val/test split (same indices for both silos)
    idx = np.arange(len(intersection_ids))
    idx_tv, idx_test = train_test_split(idx, test_size=args.test_size, random_state=args.seed, stratify=y_a)
    idx_train, idx_val = train_test_split(idx_tv, test_size=args.val_size/(1-args.test_size),
                                          random_state=args.seed, stratify=y_a[idx_tv])

    # Apply SMOTE to training set only (Silo A)
    print(y_a[:5])
    print(np.unique(y_a))
    smote = SMOTE(random_state=args.seed)
    X_a_train_res, y_a_train_res = smote.fit_resample(X_a[idx_train], y_a[idx_train])
    X_b_train_res, _             = smote.fit_resample(X_b[idx_train], y_a[idx_train])

    # Save pickles
    for name, X_train, y_train, X_idx in [
        ("silo_a", X_a_train_res, y_a_train_res, X_a),
        ("silo_b", X_b_train_res, y_a_train_res, X_b),
    ]:
        data = {
            "X_train": X_train, "y_train": y_train,
            "X_val":   X_a[idx_val] if name == "silo_a" else X_b[idx_val],
            "y_val":   y_a[idx_val],
            "X_test":  X_a[idx_test] if name == "silo_a" else X_b[idx_test],
            "y_test":  y_a[idx_test],
            "intersection_ids": intersection_ids,
        }
        out_path = PROCESSED_DIR / f"{name}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(data, f)
        print(f"[Silo Prep] Saved {out_path} — train={len(y_train)}, val={len(y_a[idx_val])}, test={len(y_a[idx_test])}")


if __name__ == "__main__":
    main()
