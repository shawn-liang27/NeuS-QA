#!/bin/bash
#SBATCH -J neus_qa_eval
#SBATCH --output=logs/naive_neus_video_mme%j/eval_%j.out
#SBATCH --error=experiment/naive_neus_video_mme%j/eval_%j.err
#SBATCH -p gpu
#SBATCH -C gpu4090
#SBATCH -A gxd234_1
#SBATCH -N 1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16              
#SBATCH --gres=gpu:1
#SBATCH --mem=64G                        
#SBATCH --time=2-00:00:00
set -euo pipefail
# ==============================================================================
# Configuration
# ==============================================================================

module load CUDA/12.8.0
module load CMake/3.27.6-GCCcore-13.2.0
module load Boost/1.83.0-GCC-13.2.0
module load GMP/6.3.0-GCCcore-13.2.0
module load FriBidi/1.0.13-GCCcore-13.2.0
module load HarfBuzz/8.2.2-GCCcore-13.2.0

hf_cache_dir=/scratch/pioneer/users/sgl57/huggingface
mkdir -p $hf_cache_dir
export HF_HOME=$hf_cache_dir
export PATH="$HOME/bin:$PATH"
export PKG_CONFIG_PATH="$HOME/ffmpeg_build/lib/pkgconfig:$PKG_CONFIG_PATH"
# Library Paths (Crucial for running the binary if any parts are dynamic)
export LD_LIBRARY_PATH="$HOME/ffmpeg_build/lib:$LD_LIBRARY_PATH"
# Fontconfig Path (HPC specific: ensures libass can find fonts)
export FONTCONFIG_PATH="/etc/fonts"
# ===========================================


# source project env
source .venv/bin/activate
source activate_storm_env.sh

JOB_ID=$(date +%Y%m%d_%H%M%S)
# Variables
# DATA_DIR="/usr/homes/sgl57/.data/LongVideoBench"
# BURNED_DIR="/usr/homes/sgl57/.data/LongVideoBench/burn-subtitles/T3E_E3E_T3O_O3O_mix_2026_01_14_21_55"

DATA_DIR="/scratch/pioneer/users/sgl57/huggingface/hub/datasets/video_mme"
BURNED_DIR="/scratch/pioneer/users/sgl57/huggingface/hub/datasets/video_mme/burn-subtitles"
MODEL="InternVL2-8B"

# CATEGORIES=("T3E" "E3E" "T3O" "O3O") # "T3E", "E3E", "T3O", "O3O"
CAT_STR=$(IFS='_'; echo "${CATEGORIES[*]}")
OUT_DIR="${SLURM_SUBMIT_DIR}/experiment_results/nsvs/naive_neus_baseline/videomme_"${MODEL//\//_}"_${JOB_ID}"

mkdir -p "$OUT_DIR"

LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

echo ">>> Starting Parallel Job: $JOB_ID"
echo ">>> Logs will be saved to: $LOG_DIR"

source ./activate_storm_env.sh
source .venv/bin/activate
set -a
source .ENV
set +a
# ==============================================================================
# 1. Start vLLM Server in Background
# ==============================================================================
TOTAL_SPLITS=8  # Set this to your number of GPUs
GPU_START=$1
FRAME_WINDOW=$2
export CUDA_VISIBLE_DEVICE=0,1,2,3,4,5,6,7

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
        --frame_window ${FRAME_WINDOW} \
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


# =========================================================
# MAIN LOOP: Spawn Workers
# =========================================================

# export CUDA_LAUNCH_BLOCKING=1
# export PYTHONUNBUFFERED=1
# export TORCH_SHOW_CPP_STACKTRACES=1
# export TORCH_DISABLE_ADDR2LINE=1

trap 'echo ">>> Killing all workers..."; kill $(jobs -p); exit' SIGINT SIGTERM

for (( i=1; i<=TOTAL_SPLITS; i++ ))
do
    # Calculate GPU ID (0-based) from Split ID (1-based)
    GPU_ID=$(($GPU_START + (i-1))
    
    # Launch function in background
    launch_worker $MODEL $i $GPU_ID &
    
    # Small sleep to prevent all 4 servers from spiking CPU/Disk at the exact same millisecond
    sleep 5
done

# =========================================================
# WAIT
# =========================================================
echo ">>> All workers launched. Waiting for completion..."
wait
echo ">>> All jobs finished."