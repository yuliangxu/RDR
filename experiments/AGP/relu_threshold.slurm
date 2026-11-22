#!/bin/bash
#SBATCH --job-name=agp-threshold
#SBATCH --partition=mastatlab,gpu-common
#SBATCH --account=mastatlab
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --array=0-19
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the RDR repository root}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
/usr/bin/python3 -B -m experiments.AGP.relu_threshold_experiment fit \
    --output "${1:?Pass the prepared run directory}" --index "${SLURM_ARRAY_TASK_ID:?}" --device cuda
