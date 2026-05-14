#!/usr/bin/env bash
# Download the Open University Learning Analytics Dataset (OULAD)
# Source: https://analyse.kmi.open.ac.uk/open_dataset
# License: CC BY 4.0

set -euo pipefail

RAW_DIR="${OULAD_RAW_DIR:-data/raw}"
mkdir -p "$RAW_DIR"

BASE_URL="https://analyse.kmi.open.ac.uk/open_dataset/download"

FILES=(
  "studentInfo.csv"
  "studentRegistration.csv"
  "studentAssessment.csv"
  "studentVle.csv"
  "assessments.csv"
  "vle.csv"
  "courses.csv"
)

echo "[OULAD] Downloading dataset to $RAW_DIR ..."
for f in "${FILES[@]}"; do
  if [ -f "$RAW_DIR/$f" ]; then
    echo "  [SKIP] $f already exists"
  else
    echo "  [DOWNLOAD] $f"
    curl -L -o "$RAW_DIR/$f" "$BASE_URL/$f" --progress-bar
  fi
done

echo "[OULAD] Done. Files in $RAW_DIR:"
ls -lh "$RAW_DIR"
