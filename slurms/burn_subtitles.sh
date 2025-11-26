#!/usr/bin/env bash
#SBATCH -J longvideobench_burn_subtitles
#SBATCH -o longvideobench_burn_subtitles.o%j
#SBATCH -A gxd234_1
#SBATCH --output=longvideobench_burn_subtitles%j.out
#SBATCH --error=longvideobench_burn_subtitles%j.err
#SBATCH -p batch
#SBATCH --time=2-00:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
set -euo pipefail

module load CUDA/12.8.0
module load FFmpeg/6.0-GCCcore-13.2.0
module load CMake/3.29.3-GCCcore-13.3.0
module load Miniconda3/23.10.0-1
source "/usr/local/easybuild_allnodes/software/Miniconda3/23.10.0-1/etc/profile.d/conda.sh" 
conda activate "$HOME/.conda/envs/neus_qa"
export PATH="$CONDA_PREFIX/bin:$PATH"

echo "Starting job on $(hostname)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"

DATA_DIR="/scratch/pioneer/users/sgl57/LongVideoBench"
OUT_DIR="/sratch/pioneer/users/sgl57/LongVideoBench/burn-subtitles"


echo "starting burning subtitles onto LongVideoBench"

python scripts/burn_subtitles.py --data_dir "${DATA_DIR}" --out_dir "${OUT_DIR}"

echo "burn substitle completed"
