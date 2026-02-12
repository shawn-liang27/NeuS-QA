#!/usr/bin/env bash

set -euo pipefail

# hf download lmms-lab/Video-MME  --repo-type dataset
# unzip subtitle.zip -d "/usr/homes/sgl57/.data/Video-MME"
# for f in videos_chunked_*.zip; do unzip "$f" -d "/usr/homes/sgl57/.data/Video-MME"; done

source .venv/bin/activate
DATA_DIR="/usr/homes/sgl57/.data/Video-MME"           
OUT_DIR="${DATA_DIR}/burn-subtitles-full-benchmark"
CATEGORIES=("Temporal Perception" "Spatial Perception" "Attribute Perception" "Action Recognition" "Object Recognition" "OCR Problems" "Temporal Reasoning" "Spatial Reasoning" "Object Reasoning" "Information Synopsis")
CAT_STR=$(IFS='_'; echo "${CATEGORIES[*]}")

echo "Starting burning subtitles..."
mkdir -p "$OUT_DIR"

python scripts/dataset_preprocessing/video_mme/prepare_video_mme.py --data_dir "${DATA_DIR}"

python scripts/dataset_preprocessing/video_mme/burn_subtitles.py --data_dir "${DATA_DIR}" --out_dir "${OUT_DIR}"  --categories "${CATEGORIES[@]}" > "${OUT_DIR}/burn_subtitles.log" 2>&1 

echo "burn subtitle completed"