#!/usr/bin/env bash
# Download the Open University Learning Analytics Dataset (OULAD)
# Source: https://analyse.kmi.open.ac.uk/open_dataset
# License: CC BY 4.0

set -euo pipefail

RAW_DIR="${OULAD_RAW_DIR:-data/raw}"
mkdir -p "$RAW_DIR"

# The OU's own links are dead as of Oct 2026: analyse.kmi.open.ac.uk answers
# the old path with 200 + an HTML landing page, and the CDN link the current
# OU Analyse page offers (schools.stem.open.ac.uk/.../anonymisedData.zip) is a
# hard 404. figshare is the archival host behind the dataset's Scientific Data
# paper (DOI 10.6084/m9.figshare.5081998), so it is the citable source, not a
# third-party mirror. Override with OULAD_ZIP_URL if the OU restores theirs.
ZIP_URL="${OULAD_ZIP_URL:-https://ndownloader.figshare.com/files/8606371}"
# md5 published alongside the figshare record; skipped when OULAD_ZIP_URL is set.
ZIP_MD5="7412686fd77cf0e0ee1e8c3e9b354308"
ZIP_FILE="$RAW_DIR/oulad.zip"

# One file we expect after extraction
CHECK_FILE="$RAW_DIR/studentInfo.csv"

if [ -f "$CHECK_FILE" ]; then
  echo "[OULAD] Dataset already extracted in $RAW_DIR"
  exit 0
fi

echo "[OULAD] Downloading dataset archive..."
curl -fL "$ZIP_URL" -o "$ZIP_FILE" --progress-bar

# The OU has moved this file before, and the site answers a dead path with a
# 200 + HTML landing page rather than a 404. Without this check that HTML got
# written to oulad.zip and the failure only surfaced later as a confusing
# unzip error, leaving a bogus archive behind.
if ! unzip -tq "$ZIP_FILE" >/dev/null 2>&1; then
  echo "[OULAD] ERROR: $ZIP_URL did not return a zip archive." >&2
  echo "[OULAD] Got: $(file -b "$ZIP_FILE")" >&2
  echo "[OULAD] The dataset has moved. Find the current link at" >&2
  echo "[OULAD]   https://analyse.kmi.open.ac.uk/open_dataset" >&2
  echo "[OULAD] and set OULAD_ZIP_URL, or place the CSVs in $RAW_DIR by hand." >&2
  rm -f "$ZIP_FILE"
  exit 1
fi

if [ -z "${OULAD_ZIP_URL:-}" ]; then
  if command -v md5 >/dev/null 2>&1; then
    GOT="$(md5 -q "$ZIP_FILE")"
  else
    GOT="$(md5sum "$ZIP_FILE" | cut -d' ' -f1)"
  fi
  if [ "$GOT" != "$ZIP_MD5" ]; then
    echo "[OULAD] ERROR: checksum mismatch." >&2
    echo "[OULAD]   expected $ZIP_MD5" >&2
    echo "[OULAD]   got      $GOT" >&2
    rm -f "$ZIP_FILE"
    exit 1
  fi
  echo "[OULAD] Checksum OK ($ZIP_MD5)"
fi

echo "[OULAD] Extracting archive..."
unzip -o "$ZIP_FILE" -d "$RAW_DIR"

# Optional cleanup
rm -f "$ZIP_FILE"

echo "[OULAD] Done. Files in $RAW_DIR:"
ls -lh "$RAW_DIR"