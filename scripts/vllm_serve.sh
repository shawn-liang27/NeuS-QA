#!/bin/bash

MODEL="OpenGVLab/InternVL2_5-8B"
# MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
# export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES="0"
MAX_LEN=40000
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NUM_GPUS=1 # Assumes CUDA_VISIBLE_DEVICES is set
PORT=8006
source .venv/bin/activate
vllm serve $MODEL \
    --tensor-parallel-size $NUM_GPUS \
    --port $PORT \
    --trust-remote-code \
    --max-model-len $MAX_LEN \
    --gpu-memory-utilization 0.95 \
    

