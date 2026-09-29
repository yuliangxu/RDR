#!/bin/bash
#SBATCH --job-name=mnist-selected-comparison
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=mastatlab
#SBATCH --account=mastatlab
set -euo pipefail
mnist_comparison_root="${1:?Expected prepared output directory}"
export PYTHONPATH= PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export MPLBACKEND=Agg CUBLAS_WORKSPACE_CONFIG=:4096:8
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mnist-comparison-mpl-${SLURM_JOB_ID:-local}"
cd "$mnist_comparison_root"
"${RDR_PYTHON:-/usr/bin/python3}" -B "$mnist_comparison_root/source/experiments/MNIST/comparison.py" run --output "$mnist_comparison_root" --device cuda
