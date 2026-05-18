#!/usr/bin/env bash
# Download the Open University Learning Analytics Dataset (OULAD)
# Source: https://analyse.kmi.open.ac.uk/open_dataset
# License: CC BY 4.0

set -euo pipefail

RAW_DIR="${OULAD_RAW_DIR:-data/raw}"
mkdir -p "$RAW_DIR"

ZIP_URL="https://analyse.kmi.open.ac.uk/open-dataset/download"
ZIP_FILE="$RAW_DIR/oulad.zip"

# One file we expect after extraction
CHECK_FILE="$RAW_DIR/studentInfo.csv"

if [ -f "$CHECK_FILE" ]; then
  echo "[OULAD] Dataset already extracted in $RAW_DIR"
  exit 0
fi

echo "[OULAD] Downloading dataset archive..."
curl -L "$ZIP_URL" -o "$ZIP_FILE" --progress-bar

echo "[OULAD] Extracting archive..."
unzip -o "$ZIP_FILE" -d "$RAW_DIR"

# Optional cleanup
rm -f "$ZIP_FILE"

echo "[OULAD] Done. Files in $RAW_DIR:"
ls -lh "$RAW_DIR"