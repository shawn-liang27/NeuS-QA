#!/bin/bash
set -euo pipefail

JOB_DIR="$HOME/NeuS-VLM/NeuS-QA"
JOB_ID=$(date +%Y%m%d_%H%M%S)

export HF_HOME="$HOME/.cache/huggingface"
export DECORD_EOF_RETRY_MAX=40960
# Variables
DATA_DIR="/usr/homes/sgl57/.data/LongVideoBench"
# BURNED_DIR="/usr/homes/sgl57/.data/LongVideoBench/burn-subtitles/T3E_E3E_T3O_O3O_mix"
BURNED_DIR="/usr/homes/sgl57/.data/LongVideoBench/burn-subtitles/T3E_E3E_T3O_O3O_mix_2026_01_14_21_55"
MODEL="InternVL2-8B"
MAX_TOKEN_LEN=65000

CATEGORIES=("T3E" "E3E" "T3O" "O3O") # "T3E", "E3E", "T3O", "O3O"
CAT_STR=$(IFS='_'; echo "${CATEGORIES[*]}")
OUT_DIR="$JOB_DIR/experiment_results/debug_nsvs/"${MODEL//\//_}"/clip_adaptive_sampling/${JOB_ID}"

mkdir -p "$OUT_DIR"

LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

echo ">>> Starting Parallel Job: $JOB_ID"
echo ">>> Logs will be saved to: $LOG_DIR"

source ./activate_storm.sh
source .venv/bin/activate
set -a
source .ENV
set +a
# =========================================================
# CONFIGURATION
# =========================================================
TOTAL_SPLITS=306  # Set this to your number of GPUs
CURRENT_SPLIT=$1
GPU=$2

# =========================================================
# FUNCTION: Worker Logic (Runs in Parallel)
# =========================================================
launch_worker() {
    local MODEL=$1
    local SPLIT_ID=$2
    local GPU_ID=$3
    # Unique log files for this worker
    local WORKER_LOG="${LOG_DIR}/worker_${SPLIT_ID}"

    echo ">>> [Worker $SPLIT_ID] Server Ready. Running Python script..."
    local START_TIME=$(date +%s)

    python -u scripts/lmms_eval/nsvs_only.py \
        --vlm_model_name "${MODEL}" \
        --port_number "${GPU_ID}" \
        --data_dir "${DATA_DIR}" \
        --burned_dir "${BURNED_DIR}" \
        --output_dir "${OUT_DIR}/split_${SPLIT_ID}" \
        --current_split "${SPLIT_ID}" \
        --total_splits "${TOTAL_SPLITS}" \
        --categories "${CATEGORIES[@]}" \
        --measure_metrics > "${WORKER_LOG}_eval.out" 2>&1

    local PY_EXIT=$?
    
    # 3. Capture End Time & Calculate Duration
    local END_TIME=$(date +%s)
    local DURATION=$(( END_TIME - START_TIME ))

    # 4. Calculate Hours, Minutes, Seconds for readability
    local HOURS=$(( DURATION / 3600 ))
    local MINS=$(( (DURATION % 3600) / 60 ))
    local SECS=$(( DURATION % 60 ))

    # 5. Log the time
    echo ">>> [Worker $SPLIT_ID] Finished in ${HOURS}h ${MINS}m ${SECS}s."
    
    echo ">>> [Worker $SPLIT_ID] Finished (Exit Code: $PY_EXIT) in ${HOURS}h ${MINS}m ${SECS}s. Stopping server..." >> "${WORKER_LOG}_eval.out"
}

trap 'echo ">>> Killing all workers..."; kill $(jobs -p); exit' SIGINT SIGTERM

launch_worker $MODEL $CURRENT_SPLIT $GPU &

# =========================================================
# WAIT
# =========================================================
echo ">>> All workers launched. Waiting for completion..."
wait
echo ">>> All jobs finished."