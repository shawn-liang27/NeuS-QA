#!/bin/bash


set -euo pipefail

JOB_DIR="$HOME/NeuS-VLM/NeuS-QA"
JOB_ID=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/experiments/nsvs_qa_${JOB_ID}"
mkdir -p "$LOG_DIR"

echo ">>> Starting Parallel Job: $JOB_ID"
echo ">>> Logs will be saved to: $LOG_DIR"

source ./activate_storm.sh
source .venv/bin/activate
set -a
source .ENV
set +a


export HF_HOME="$HOME/.cache/huggingface"
# Use $HOME or relative paths instead of /scratch...
DATA_DIR="/usr/homes/sgl57/.data/LongVideoBench"
BURNED_DIR="/usr/homes/sgl57/.data/LongVideoBench/burn-subtitles/T3E_E3E_T3O_O3O_mix"
OUT_DIR="$JOB_DIR/experiment_result/nsvs_qa_${JOB_ID}"
mkdir -p "$OUT_DIR"


export VLLM_PORT=35426


# ==============================================================================
# 1. Start vLLM Server in Background
# ==============================================================================
echo ">>> Starting vLLM server on port $VLLM_PORT..."

if curl -s "http://localhost:$VLLM_PORT/health" > /dev/null; then
    echo ">>> Port $VLLM_PORT is already in use"
    return 1
fi

# launch the server script in the background using '&'

./scripts/vllm_serve.sh "6" "${VLLM_PORT}" "0.2" > $LOG_DIR/vllm_server.log 2>&1 &

# Capture the Process ID (PID) of the server to kill later
SERVER_PID=$!

# ==============================================================================
# 2. Wait for Server to be Ready
# ==============================================================================
echo ">>> Waiting for vLLM to load model..."

# Loop until the server responds to a health check or timeout
MAX_RETRIES=60 # Wait up to 10-20 minutes (depending on sleep time)
count=0
sleep 10

while true; do
    # Attempt to connect to the vLLM health endpoint
    # If curl returns HTTP 200, the server is ready.
    if curl -s "http://localhost:$VLLM_PORT/health" > /dev/null; then
        echo ">>> vLLM server is READY!"
        break
    fi
    
    # Also check if the process died
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo ">>> Error: vLLM server process died unexpectedly. Check logs/vllm_server_$JOB_ID.log"
        exit 1
    fi

    sleep 10
    count=$((count+1))
    if [ $count -ge $MAX_RETRIES ]; then
        echo ">>> Timeout waiting for vLLM server."
        kill $SERVER_PID
        exit 1
    fi
    echo "    ... loading ($count/$MAX_RETRIES)"
done

# ==============================================================================
# 3. Run Evaluation
# ==============================================================================
echo ">>> Starting Evaluation..."

MODEL="OpenGVLab/InternVL2_5-8B"
EXAMPLE_VID_PATH="/usr/homes/sgl57/NeuS-VLM/NeuS-QA/mH9LdC7IFH8.mp4"

python evaluate.py --vlm_model_name "${MODEL}" --port_number $VLLM_PORT --output_dir "${OUT_DIR}" --example_vid_path "${EXAMPLE_VID_PATH}"

EXIT_CODE=$?

# ==============================================================================
# 4. Cleanup
# ==============================================================================
echo ">>> Evaluation finished with code $EXIT_CODE. Stopping server..."
kill $SERVER_PID

exit $EXIT_CODE