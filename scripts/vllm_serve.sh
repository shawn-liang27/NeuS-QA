#!/bin/bash
MODEL=$1
MAX_LEN=$2
# MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
# export NCCL_P2P_DISABLE=1
# export CUDA_VISIBLE_DEVICES=$3
export VLLM_USE_V1=1
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NUM_GPUS=8 # Assumes CUDA_VISIBLE_DEVICES is set
PORT=$4
GPU_UTIL_SIZE=$5
PROCESSOR_ARGS=$6
source .env/bin/activate
exec vllm serve $MODEL \
    --tensor-parallel-size $NUM_GPUS \
    --port $PORT \
    --trust-remote-code \
    --gpu-memory-utilization $GPU_UTIL_SIZE \
    --mm-processor-kwargs "${PROCESSOR_ARGS}" \
    --mm-processor-cache-gb 0
    

