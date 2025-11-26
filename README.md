# NeuS-QA

[Original Repo](https://github.com/UTAustin-SwarmLab/NeuS-QA)

# Set up

```bash
git clone
cd NeuS-QA
chmod +x build_dependency.sh

# Build carl-storm and storm stable
./build_dependency.sh

# create env
uv sync
```

## Trouble setting up env in HPC env

Need to load modules

```bash
module load CMake/3.27.6-GCCcore-13.2.0
module load Boost/1.83.0-GCC-13.2.0
module load GMP/6.3.0-GCCcore-13.2.0
module load FFmpeg/6.0-GCCcore-13.2.0
```

before `uv sync`, need to manually install stormpy in the uv .venv before doing uv sync, because HPC has no CLN and stormpy doesn't know that storm is already installed without CLN due to build isolation, and is attempting to reinstall, causing errors.

```bash

source .venv/bin/activate

.venv/bin/python pip install scikit-build-core cmake ninja pathspec pyproject_metadata pybind11

.venv/bin/python -m pip install stormpy --no-build-isolation --config-settings=cmake.args="-DSTORM_USE_CLN_EA=OFF;-DSTORM_USE_CLN_RF=OFF;-DCMAKE_PREFIX_PATH=$LOCAL_INSTALL_DIR;-Dstorm_DIR=$LOCAL_INSTALL_DIR/lib/cmake/storm-Dpybind11_DIR=$PYBIND_CMAKE_DIR;"

uv sync
```


# burn in substitles for LongVideoBench

in terminal, run

```bash
hf download longvideobench/LongVideoBench --repo-type dataset --local-dir "your_data_dir"

cd "your_data_dir"
cat videos.tar.part.* | tar -xvf - -C .
tar -xvf subtitles.tar
```

This will download LongVideoBench Datasets to "your_data_dir" and then unzip all videos and subtitles

and after you'll have the LongVideoBench dataset to burn subtitles

run

```bash
DATA_DIR="your_longvideobench_data_dir"
OUT_DIR="your_output_dir"

python scripts/burn_subtitles.py --data_dir "${DATA_DIR}" --out_dir "${OUT_DIR}"
```

This requires ffmpeg that has flag --enable-libass

\* Takes about 10-12 hours to burn-in, but lots of the videos reported errors in my last run

# scripts/vllm_serve.sh

NeuS-QA uses **vllm** package to manage (download and host) specified VLM model.

vllm hosts the model locally, then it can be used like OpenAI client structure to run queries

change MODEL variable in `vllm_serve.sh` script to specify VLM in use

# evaluate.py

This is the evaluation script to test out the NeuS-QA code, however, it is currently hardcoded to run only one exmaple video, only have video queries and answers for this particular video. Haven't looked into how to retrieve correct QA question and answer for any video in LongVideoBench

**The only example video to run:**   
**mH9LdC7IFH8.mp4**

## How to Run

Two steps, first is to initiate vllm serve to download and host the specified VLM model locally at https:localhost:8000 so that NeuS and VQA process can use the VLM model. Once the connection is established, run evaluate.py



1. ./scripts/vllm_serve.sh
2. python evaluate.py

```bash
# run with NeuS-QA as pwd
export OPENAI_API_KEY="your_api_key"

source .venv/bin/activate

# set up HPC env for carl-storm
source activate_storm_env.sh

# make sure model hosted matches with the evaluation model
./scripts/vllm_serve.sh &

python evaluate.py --vlm_model_name "${MODEL}" --port_number "6" --output_dir "${OUT_DIR}" --example_vid_path "${EXAMPLE_VID_PATH}"
```
