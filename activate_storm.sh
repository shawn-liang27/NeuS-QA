#!/bin/bash
set -euo pipefail

export CUDA_HOME=/usr/lib/nvidia-cuda-toolkit

# 2. Add the library path so the linker finds the 12.8 libs (not the 13.0 system default)
export PATH=$CUDA_HOME/bin:${PATH:-}
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}
export LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib:${LIBRARY_PATH:-}

export PATH="/usr/homes/sgl57/NeuS-VLM/NeuS-QA/vendors/install/bin:$PATH"
export LD_LIBRARY_PATH="/usr/homes/sgl57/NeuS-VLM/NeuS-QA/vendors/install/lib:/usr/homes/sgl57/NeuS-VLM/NeuS-QA/vendors/install/lib64:${LD_LIBRARY_PATH:-}"
export CPATH="/usr/homes/sgl57/NeuS-VLM/NeuS-QA/vendors/install/include:${CPATH:-}"
export CMAKE_PREFIX_PATH="/usr/homes/sgl57/NeuS-VLM/NeuS-QA/vendors/install:${CMAKE_PREFIX_PATH:-}"
export STORM_DIR_HINT="/usr/homes/sgl57/NeuS-VLM/NeuS-QA/vendors/install"
export CARL_DIR_HINT="/usr/homes/sgl57/NeuS-VLM/NeuS-QA/vendors/install"