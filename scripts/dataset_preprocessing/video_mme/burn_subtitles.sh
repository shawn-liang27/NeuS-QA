#!/usr/bin/env bash
#SBATCH --gxd234_1
#SBATCH --job-name=video-mme-burn
#SBATCH --output=logs/burn_%j.out
#SBATCH --error=logs/burn_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10          # Matches the 10 parallel workers in your script
#SBATCH --mem=32G                   # Video processing is memory-intensive
#SBATCH --time=04:00:00             # Estimated time for ~900 videos

set -euo pipefail

# hf download lmms-lab/Video-MME  --repo-type dataset
# unzip subtitle.zip -d "/usr/homes/sgl57/.data/Video-MME"
# for f in videos_chunked_*.zip; do unzip "$f" -d "/usr/homes/sgl57/.data/Video-MME"; done

source .venv/bin/activate
DATA_DIR="/usr/homes/sgl57/.data/Video-MME"           
OUT_DIR="${DATA_DIR}/burn-subtitles" 

echo "Starting burning subtitles..."
mkdir -p "$OUT_DIR"

python scripts/dataset_preprocessing/video_mme/prepare_video_mme.py --data_dir "${DATA_DIR}"

python scripts/dataset_preprocessing/video_mme/burn_subtitles.py --data_dir "${DATA_DIR}" --out_dir "${OUT_DIR}" > "${OUT_DIR}/burn_subtitles.log" 2>&1 

echo "burn subtitle completed"