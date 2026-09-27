#!/bin/bash
#SBATCH --job-name=mnist-history-check
#SBATCH --time=00:30:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=mastatlab
#SBATCH --account=mastatlab
#SBATCH --output=/cwork/yx306/RDR/MNIST_jrssb_final/historical-%j.out
#SBATCH --error=/cwork/yx306/RDR/MNIST_jrssb_final/historical-%j.err
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="/tmp/mnist-history-mpl-${SLURM_JOB_ID}"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mnist_package=/cwork/yx306/RDR/MNIST_jrssb_final
cd "$mnist_package"
python3 workflow.py train --root "$mnist_package" --output "$mnist_package/training_historical_repeat" --experiment all --numerics historical
