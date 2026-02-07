#!/usr/bin/env bash
set -euo pipefail

cd /home/sgl57/ECSE_gxd234_1/NeuS-QA/
source .venv/bin/activate

DATA_DIR="${HF_HOME}/hub/datasets--lmms-lab--Video-MME/snapshots/ead1408f75b618502df9a1d8e0950166bf0a2a0b"           
OUT_DIR="${HF_HOME}/hub/datasets--lmms-lab--Video-MME/burn-subtitles" 
echo "Starting burning subtitles..."

mkdir -p "$OUT_DIR"

python scripts/dataset_preprocessing/video_mme/burn_subtitles.py --data_dir "${DATA_DIR}" --out_dir "${OUT_DIR}" > "${OUT_DIR}/burn_subtitles.log" 2>&1 

echo "burn subtitle completed"