#!/bin/bash
#SBATCH -J neus_qa_eval
#SBATCH --output=logs/nsvs_qa_%j/eval_%j.out
#SBATCH --error=logs/nsvs_qa_%j/eval_%j.err
#SBATCH -p gpu
#SBATCH -C gpu4090
#SBATCH -A gxd234_1
#SBATCH -N 1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16              
#SBATCH --gres=gpu:1
#SBATCH --mem=64G                        
#SBATCH --time=04:00:00

# ==============================================================================
# Configuration
# ==============================================================================
# Set paths
cd "$SLURM_SUBMIT_DIR"

# set openai api key here
export OPENAI_API_KEY=""
export VLLM_PORT=8006
export HF_HOME="/scratch/pioneer/users/sgl57/huggingface"

module load CUDA/12.8.0
module load CMake/3.27.6-GCCcore-13.2.0
module load Boost/1.83.0-GCC-13.2.0
module load GMP/6.3.0-GCCcore-13.2.0
module load FFmpeg/6.0-GCCcore-13.2.0

# source project env
source .venv/bin/activate

# set
source activate_storm_env.sh

# ==============================================================================
# 1. Start vLLM Server in Background
# ==============================================================================
echo ">>> Starting vLLM server on port $VLLM_PORT..."

# launch the server script in the background using '&'

./scripts/vllm_serve.sh > logs/nsvs_qa_$SLURM_JOB_ID/vllm_server_$SLURM_JOB_ID.log 2>&1 &

# Capture the Process ID (PID) of the server to kill later
SERVER_PID=$!

# ==============================================================================
# 2. Wait for Server to be Ready
# ==============================================================================
echo ">>> Waiting for vLLM to load model..."

# Loop until the server responds to a health check or timeout
MAX_RETRIES=60 # Wait up to 10-20 minutes (depending on sleep time)
count=0
while true; do
    # Attempt to connect to the vLLM health endpoint
    # If curl returns HTTP 200, the server is ready.
    if curl -s "http://localhost:$VLLM_PORT/health" > /dev/null; then
        echo ">>> vLLM server is READY!"
        break
    fi
    
    # Also check if the process died
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo ">>> Error: vLLM server process died unexpectedly. Check logs/vllm_server_$SLURM_JOB_ID.log"
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
OUT_DIR="/home/sgl57/ECSE_gxd234_1/Neus_VLM/NeuS-QA/experiment_result"
EXAMPLE_VID_PATH="/scratch/pioneer/users/sgl57/LongVideoBench/burn-subtitles/mH9LdC7IFH8.mp4"

python evaluate.py --vlm_model_name "${MODEL}" --port_number "6" --output_dir "${OUT_DIR}" --example_vid_path "${EXAMPLE_VID_PATH}"

EXIT_CODE=$?

# ==============================================================================
# 4. Cleanup
# ==============================================================================
echo ">>> Evaluation finished with code $EXIT_CODE. Stopping server..."
kill $SERVER_PID

exit $EXIT_CODE