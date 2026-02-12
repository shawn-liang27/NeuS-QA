#!/bin/bash
set -euo pipefail

# =========================================================
# Variables
MODEL=$1
TOTAL_SPLITS=4  # Set this to your number of GPUs
MAX_NUM_FRAMES=$2
CURRENT_SPLIT=$3
GPU=$4
NGRES=$5
GPU_USAGE=$6
CATEGORIES=("T3E" "E3E" "T3O" "O3O") # "T3E", "E3E", "T3O", "O3O"
# =========================================================
# configurations

JOB_DIR="$HOME/NeuS-VLM/NeuS-QA"
JOB_ID=$(date +%Y%m%d_%H%M%S)

export HF_HOME="$HOME/.cache/huggingface"


CAT_STR=$(IFS='_'; echo "${CATEGORIES[*]}")

NSVS_VQA_DIR="$JOB_DIR/experiment_results/nsvs_vqa/"${MODEL//\//_}"/nsvs_vqa_${CAT_STR}_${JOB_ID}"

# EXPERIMENT_DIR="/usr/homes/sgl57/NeuS-VLM/NeuS-QA/experiment_results/nsvs/OpenGVLab_InternVL2_5-8B/nsvs_qa_E3E_T3O_O3O_20251226_000802"
# EXPERIMENT_DIR="/usr/homes/sgl57/NeuS-VLM/NeuS-QA/experiment_results/nsvs/InternVL2-8B/nsvs_qa_T3E_E3E_T3O_O3O_20260112_173959"
EXPERIMENT_DIR="/usr/homes/sgl57/NeuS-VLM/NeuS-QA/experiment_results/nsvs/InternVL2-8B/nsvs_qa_T3E_E3E_T3O_O3O_20260114_230246"

MAX_TOKEN_LEN=60000

mkdir -p "$NSVS_VQA_DIR"

LOG_DIR="$NSVS_VQA_DIR/logs"
mkdir -p "$LOG_DIR"

echo ">>> Starting Parallel Job: $JOB_ID"
echo ">>> Logs will be saved to: $LOG_DIR"

source ./activate_storm.sh
source .env/bin/activate

# =========================================================
# FUNCTION: Worker Logic (Runs in Parallel)
# =========================================================

launch_worker() {
    local MODEL=$1
    local SPLIT_ID=$2
    local GPU_ID=$3
    local NGRES=$4
    # Unique log files for this worker
    local WORKER_LOG="${LOG_DIR}/worker_${SPLIT_ID}"
    local PROCESSOR_ARGS='{"max_dynamic_patch": 12}'

    while true; do
        # 1. Ask the OS for a random FREE ephemeral port (High range guaranteed)
        local PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
        
        echo ">>> [Worker $SPLIT_ID] Attempting to start on GPU $GPU_ID | Port $PORT"

        # 2. Start vLLM Server
        export VLLM_PORT=$PORT
        # Overwrite the log for this attempt
        ./scripts/vllm_serve.sh "$MODEL" "$MAX_TOKEN_LEN" "$GPU_ID" "$PORT" "$GPU_USAGE" "$PROCESSOR_ARGS" "$NGRES"> "${WORKER_LOG}_vllm.log" 2>&1 &
        local SERVER_PID=$!

        # vLLM takes a few seconds to import torch. wait 5s to catch early crashes.
        sleep 5


        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo ">>> [Worker $SPLIT_ID]  Crash detected on Port $PORT. Retrying with new port..."
            # (Optional) Cat the end of the log to see why
            tail -n 3 "${WORKER_LOG}_vllm.log"
            continue  # Loop back to top, get NEW port, try again
        fi

        # 4. Deep Health Check (Wait for model load)
        echo ">>> [Worker $SPLIT_ID] PID $SERVER_PID survived startup. Waiting for model load..."
        local MAX_RETRIES=60
        local count=0
        local READY=0
        
        while [ $count -lt $MAX_RETRIES ]; do
            if curl -s "http://localhost:$PORT/health" > /dev/null; then
                READY=1
                break
            fi
            
            # If it dies LATE (e.g. OOM), we must catch it
            if ! kill -0 $SERVER_PID 2>/dev/null; then
                echo ">>> [Worker $SPLIT_ID] vLLM died during model load. Retrying..."
                break # Break inner loop, continue outer loop
            fi
            
            sleep 10
            count=$((count+1))
        done

        # If we are ready, break the Retry Loop and move to evaluation
        if [ $READY -eq 1 ]; then
            echo ">>> [Worker $SPLIT_ID] Server Ready on Port $PORT!"
            break
        else
            # If we timed out or died late, kill (just in case) and retry
            kill $SERVER_PID 2>/dev/null
            echo ">>> [Worker $SPLIT_ID] Restarting worker sequence..."
        fi
    done

    worker_cleanup() {
            if [ -n "$SERVER_PID" ]; then
                echo ">>> Worker [Split $SPLIT_ID] cleaning up Server PID $SERVER_PID"
                kill "$SERVER_PID" 2>/dev/null
            fi
        }
    trap worker_cleanup EXIT

    # 3. Run Evaluation
    echo ">>> [Worker $SPLIT_ID] Server Ready. Running Python script..."
    local START_TIME=$(date +%s)

    python -u scripts/lmms_eval/nsvs_repeat_vqa.py \
        --vlm_model_name "${MODEL}" \
        --port_number "${PORT}" \
        --out_dir "${NSVS_VQA_DIR}/split_${SPLIT_ID}" \
        --experiment_dir "${EXPERIMENT_DIR}/split_${SPLIT_ID}" \
        --current_split "${SPLIT_ID}" \
        --max_num_frames "${MAX_NUM_FRAMES}" \
        > "${WORKER_LOG}_eval.out" 2>&1

    local PY_EXIT=$?
    
    local END_TIME=$(date +%s)
    local DURATION=$(( END_TIME - START_TIME ))
    
    local HOURS=$(( DURATION / 3600 ))
    local MINS=$(( (DURATION % 3600) / 60 ))
    local SECS=$(( DURATION % 60 ))

    # 5. Log the time
    echo ">>> [Worker $SPLIT_ID] Finished in ${HOURS}h ${MINS}m ${SECS}s."

    # 4. Cleanup
    echo ">>> [Worker $SPLIT_ID] Finished (Exit Code: $PY_EXIT) in ${HOURS}h ${MINS}m ${SECS}s. Stopping server..." >> "${WORKER_LOG}_eval.out"

}

# =========================================================
# MAIN LOOP: Spawn Workers
# =========================================================


# for (( i=1; i<=TOTAL_SPLITS; i++ ))
# do
#     # Calculate GPU ID (0-based) from Split ID (1-based)
#     GPU_ID=$(($GPU_START + (i-1)))
    
#     # Launch function in background
#     launch_worker $MODEL $i $GPU_ID &
    
#     # Small sleep to prevent all 4 servers from spiking CPU/Disk at the exact same millisecond
#     sleep 5
# done
# =========================================================
# WAIT
# =========================================================
launch_worker $MODEL $CURRENT_SPLIT $GPU $NGRES &

echo ">>> All workers launched. Waiting for completion..."
wait

python scripts/combine_result.py "${NSVS_VQA_DIR}" --prefix "repeat_vqa_run_"${MODEL//\//_}"_${JOB_ID}"

echo ">>> All jobs finished."