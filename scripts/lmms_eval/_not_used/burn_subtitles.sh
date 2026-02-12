#!/usr/bin/env bash
set -euo pipefail

JOB_ID=$(date +%Y-%m-%d_%H-%M-%S)
JOB_DIR="/usr/homes/sgl57/NeuS-VLM/NeuS-QA"

cd $JOB_DIR
source ./activate_storm.sh

source .venv/bin/activate

DATA_DIR="/usr/homes/sgl57/.data/LongVideoBench"              
OUT_DIR="/usr/homes/sgl57/.data/LongVideoBench/burn-subtitles" 
CATEGORIES=("T3E" "E3E" "T3O" "O3O") # "T3E", "E3E", "T3O", "O3O"
echo "Starting burning subtitles..."

LOG_DIR="${JOB_DIR}/logs/burn_subtitles"
mkdir -p "$OUT_DIR"
mkdir -p "$LOG_DIR"

python scripts/burn_subtitles.py --data_dir "${DATA_DIR}" --out_dir "${OUT_DIR}" --categories "${CATEGORIES[@]}" --mix true > "${LOG_DIR}/longvideobench_${JOB_ID}.log" 2>&1 

echo "burn subtitle completed"