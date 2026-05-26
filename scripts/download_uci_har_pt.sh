#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR="${1:-data/uci_har_pt}"
ARCHIVE_PATH="${TARGET_DIR}/hapt_dataset.zip"
export TARGET_DIR

mkdir -p "${TARGET_DIR}"

if [ -f "${TARGET_DIR}/RawData/labels.txt" ] || [ -f "${TARGET_DIR}/HAPT Data Set/RawData/labels.txt" ]; then
  echo "UCI HAPT dataset already present at ${TARGET_DIR}"
  exit 0
fi

URL_CANDIDATES=(
  "https://archive.ics.uci.edu/static/public/341/smartphone+based+recognition+of+human+activities+and+postural+transitions.zip"
  "https://archive.ics.uci.edu/ml/machine-learning-databases/00341/HAPT%20Data%20Set.zip"
)

downloaded=0
for url in "${URL_CANDIDATES[@]}"; do
  echo "Trying ${url}"
  if curl -fL "${url}" -o "${ARCHIVE_PATH}"; then
    downloaded=1
    break
  fi
done

if [ "${downloaded}" -ne 1 ]; then
  echo "Failed to download the UCI HAPT dataset from the known URLs." >&2
  exit 1
fi

python3 - <<'PY'
import os
from pathlib import Path
from zipfile import ZipFile

target_dir = Path(os.environ["TARGET_DIR"]).resolve()
archive_path = target_dir / "hapt_dataset.zip"
with ZipFile(archive_path) as zf:
    zf.extractall(target_dir)

if not ((target_dir / "RawData" / "labels.txt").exists() or (target_dir / "HAPT Data Set" / "RawData" / "labels.txt").exists()):
    raise SystemExit("Extracted archive does not contain the expected RawData/labels.txt file.")

print(f"Extracted UCI HAPT dataset to {target_dir}")
PY
