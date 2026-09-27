#!/bin/bash
#SBATCH --job-name=celeba_ddim_rdr
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=/cwork/yx306/RDR/out/celeba-ddim-rdr-%j.out
#SBATCH --error=/cwork/yx306/RDR/out/celeba-ddim-rdr-%j.err
#SBATCH --partition=mastatlab
#SBATCH --account=mastatlab

set -euo pipefail

cd /hpc/home/yx306/RDR

CELEBA_DDIM_TRAIN_OUTPUT_DIR="${CELEBA_DDIM_TRAIN_OUTPUT_DIR:-/cwork/yx306/RDR/celeba-ddim-rdr}"
CELEBA_JOB_OUTPUT_ROOT="${CELEBA_JOB_OUTPUT_ROOT:-/cwork/yx306/RDR/out}"

mkdir -p "${CELEBA_DDIM_TRAIN_OUTPUT_DIR}" "${CELEBA_JOB_OUTPUT_ROOT}"

export MPLBACKEND=Agg
export MPLCONFIGDIR="/tmp/matplotlib-celeba-ddim-${SLURM_JOB_ID:-0}"
mkdir -p "${MPLCONFIGDIR}"

export CELEBA_DDIM_TRAIN_OUTPUT_DIR
export CELEBA_REQUIRE_CUDA="${CELEBA_REQUIRE_CUDA:-1}"
export CELEBA_NUM_WORKERS="${CELEBA_NUM_WORKERS:-4}"

"${PYTHON:-python3}" experiments/CelebA_ddim_train_slurm.py
