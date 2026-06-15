"""Extract raw student ID lists from OULAD CSVs for PSI protocol.

Owner: Alex Pellet
Purpose: Breaks the circular dependency between prepare_silos.py and psi_client/server.
         PSI only needs student IDs — not processed features. This script reads the
         raw CSVs and writes two plain ID files that the PSI protocol consumes.

Correct 3-step workflow
-----------------------
  Step 1  python scripts/extract_ids_for_psi.py      <- THIS SCRIPT
           reads raw CSVs → writes data/processed/silo_a_ids.json
                                    data/processed/silo_b_ids.json

  Step 2  python -m psi.psi_server
          python -m psi.psi_client
           reads silo_a_ids.json + silo_b_ids.json
           → writes data/processed/ordered_id_list.json

  Step 3  python scripts/prepare_silos.py
           reads raw CSVs + ordered_id_list.json
           → writes data/processed/silo_a.pkl + silo_b.pkl

Usage:
  python scripts/extract_ids_for_psi.py
  python scripts/extract_ids_for_psi.py --module AAA_2013J
"""

import argparse
import json
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

SILO_A_IDS_PATH = PROCESSED_DIR / "silo_a_ids.json"
SILO_B_IDS_PATH = PROCESSED_DIR / "silo_b_ids.json"


def extract_silo_a_ids(module: str, student_info: pd.DataFrame) -> list[str]:
    """Return student IDs present in studentInfo.csv for the given module.

    Args:
        module: Module presentation code, e.g. "CCC_2014J".
        student_info: Raw studentInfo DataFrame.

    Returns:
        Sorted list of string student IDs.
    """
    m_code, p_code = module.split("_")
    ids = (
        student_info
        .query("code_module == @m_code and code_presentation == @p_code")
        ["id_student"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    return sorted(str(i) for i in ids)


def extract_silo_b_ids(
    module: str,
    student_vle: pd.DataFrame,
    student_assessment: pd.DataFrame,
    assessments: pd.DataFrame,
) -> list[str]:
    """Return student IDs present in Silo B sources for the given module.

    A student is included if they appear in either studentVle OR studentAssessment
    for this module — consistent with how prepare_silo_b_data() builds features.

    Args:
        module: Module presentation code, e.g. "CCC_2014J".
        student_vle: Raw studentVle DataFrame.
        student_assessment: Raw studentAssessment DataFrame.
        assessments: Raw assessments DataFrame (needed to filter by module).

    Returns:
        Sorted list of string student IDs.
    """
    m_code, p_code = module.split("_")

    vle_ids = (
        student_vle
        .query("code_module == @m_code and code_presentation == @p_code")
        ["id_student"]
        .dropna()
        .astype(int)
        .unique()
    )

    module_assessment_ids = (
        assessments
        .query("code_module == @m_code and code_presentation == @p_code")
        ["id_assessment"]
        .unique()
    )
    assessment_ids = (
        student_assessment
        [student_assessment["id_assessment"].isin(module_assessment_ids)]
        ["id_student"]
        .dropna()
        .astype(int)
        .unique()
    )

    combined = set(vle_ids) | set(assessment_ids)
    return sorted(str(i) for i in combined)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract raw student ID lists for PSI (run before psi_client/server)."
    )
    parser.add_argument(
        "--module",
        default="CCC_2014J",
        help="OULAD module presentation code (default: CCC_2014J)",
    )
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("[ID Extract] Loading raw OULAD CSVs ...")
    student_info = pd.read_csv(RAW_DIR / "studentInfo.csv")
    student_vle = pd.read_csv(RAW_DIR / "studentVle.csv")
    student_assessment = pd.read_csv(RAW_DIR / "studentAssessment.csv")
    assessments = pd.read_csv(RAW_DIR / "assessments.csv")

    silo_a_ids = extract_silo_a_ids(args.module, student_info)
    silo_b_ids = extract_silo_b_ids(args.module, student_vle, student_assessment, assessments)

    SILO_A_IDS_PATH.write_text(json.dumps(silo_a_ids))
    SILO_B_IDS_PATH.write_text(json.dumps(silo_b_ids))

    overlap = len(set(silo_a_ids) & set(silo_b_ids))
    print(f"[ID Extract] Module {args.module}")
    print(f"[ID Extract]   Silo A IDs : {len(silo_a_ids):,}  → {SILO_A_IDS_PATH}")
    print(f"[ID Extract]   Silo B IDs : {len(silo_b_ids):,}  → {SILO_B_IDS_PATH}")
    print(f"[ID Extract]   Expected PSI intersection ≈ {overlap:,} students")
    print()
    print("[ID Extract] Next steps:")
    print("  1. python -m psi.psi_server          (terminal 1)")
    print("  2. python -m psi.psi_client          (terminal 2)")
    print("  3. python scripts/prepare_silos.py")


if __name__ == "__main__":
    main()