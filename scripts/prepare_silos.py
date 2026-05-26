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
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from utils.reproducibility import seed_everything

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

def prepare_silo_a_data(
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
    df['cohort'] = df['code_module'].astype(str)+'_'+df['code_presentation'].astype(str)
    df.drop(aixs=1,inplace=True,labels=['code_module','code_presentation','final_result'])
    
    #Ordinal encoding
    enc = OrdinalEncoder()
    df['imd_band_enc'] = enc.fit_transform(df[['imd_band']])
    df['highest_education_enc'] = df['highest_education'].map({'No Formal quals':0,'Lower Than A Level':1,'A Level or Equivalent':2,'HE Qualification':3,'Post Graduate Qualification':4})
    df['age_band_enc'] = enc.fit_transform(df['age_band'])
    df.drop(axis=1,inplace=True,labels=['imd_band','highest_education','age_band'])
    
    #Binary Encoding
    df["gender_enc"] = (df["gender"] == "F").astype(int)
    df["disability_enc"] = (df["disability"] == "Y").astype(int)
    
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
    return df

def build_silo_a(
    student_info: pd.DataFrame,
    intersection_ids: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Build Silo A feature matrix from studentInfo.csv (demographics).

    Args:
        student_info: Raw studentInfo DataFrame.
        intersection_ids: Ordered list of student IDs from PSI output.

    Returns:
        Tuple of (X, y) arrays aligned to intersection_ids order.
    """
    
    """
    Moved to new method.
    df = student_info[student_info["id_student"].astype(str).isin(set(intersection_ids))].copy()
    df["id_student"] = df["id_student"].astype(str)
    df = df.set_index("id_student").reindex(intersection_ids)

    # Binary label: Withdrawn or Fail = 1 (at-risk), Pass or Distinction = 0
    df["label"] = df["final_result"].isin(["Withdrawn", "Fail"]).astype(int)

    # ── Features ──────────────────────────────────────────────────────────────
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    df["imd_band_enc"]           = enc.fit_transform(df[["imd_band"]])
    df["highest_education_enc"]  = enc.fit_transform(df[["highest_education"]])
    df["age_band_enc"]           = enc.fit_transform(df[["age_band"]])
    df["gender_enc"]             = (df["gender"] == "F").astype(float)
    df["disability_enc"]         = (df["disability"] == "Y").astype(float)

    # Region one-hot (drop one for reference)
    region_dummies = pd.get_dummies(df["region"], prefix="region", drop_first=True)

    # Normalize studied_credits
    scaler = StandardScaler()
    df["studied_credits_norm"] = scaler.fit_transform(df[["studied_credits"]].fillna(0))"""



    feature_cols = (
        ["imd_band_enc", "highest_education_enc", "age_band_enc",
         "gender_enc", "disability_enc", "studied_credits_norm"]
    )
    X_base = df[feature_cols].fillna(0).values
    X_region = region_dummies.reindex(intersection_ids).fillna(0).values
    X = np.hstack([X_base, X_region]).astype(np.float32)
    y = df["label"].fillna(0).values.astype(np.float32)
    return X, y


def build_silo_b(
    student_vle: pd.DataFrame,
    student_assessment: pd.DataFrame,
    assessments: pd.DataFrame,
    intersection_ids: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build Silo B feature matrix from VLE + assessment data.

    Returns: (X, y, label_array) where label_array mirrors silo_a labels.
    """
    # TODO (Alex, Week 2): implement full Silo B feature engineering
    # Placeholder — returns zeros until Week 2 implementation
    n = len(intersection_ids)
    X_placeholder = np.zeros((n, 22), dtype=np.float32)
    y_placeholder = np.zeros(n, dtype=np.float32)
    return X_placeholder, y_placeholder


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
    student_ass  = pd.read_csv(RAW_DIR / "studentAssessment.csv")
    assessments  = pd.read_csv(RAW_DIR / "assessments.csv")

    # Prepare Data for each silo before creation.
    prepare_silo_a_data()
    prepare_silo_b_data()

    # Filter to target module
    module_code, pres_code = args.module.split("_")
    student_info = student_info[
        (student_info["code_module"] == module_code) &
        (student_info["code_presentation"] == pres_code)
    ].copy()
    print(f"[Silo Prep] Module {args.module}: {len(student_info)} students")

    # Load PSI intersection order (written by psi_client.py)
    id_list_path = PROCESSED_DIR / "ordered_id_list.json"
    if id_list_path.exists():
        with open(id_list_path) as f:
            intersection_ids = json.load(f)
        print(f"[Silo Prep] Using PSI intersection: {len(intersection_ids)} students")
    else:
        # Fallback: use all student IDs (run PSI first in production)
        intersection_ids = student_info["id_student"].astype(str).tolist()
        print("[Silo Prep] WARNING: ordered_id_list.json not found — using all IDs (no PSI)")

    # Build feature matrices
    X_a, y_a = build_silo_a(student_info, intersection_ids)
    X_b, y_b = build_silo_b(student_vle, student_ass, assessments, intersection_ids)

    # Train/val/test split (same indices for both silos)
    idx = np.arange(len(intersection_ids))
    idx_tv, idx_test = train_test_split(idx, test_size=args.test_size, random_state=args.seed, stratify=y_a)
    idx_train, idx_val = train_test_split(idx_tv, test_size=args.val_size/(1-args.test_size),
                                          random_state=args.seed, stratify=y_a[idx_tv])

    # Apply SMOTE to training set only (Silo A)
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
