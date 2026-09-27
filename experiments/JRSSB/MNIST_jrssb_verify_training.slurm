#!/bin/bash
#SBATCH --job-name=mnist-final-reproduce
#SBATCH --time=01:00:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=mastatlab
#SBATCH --account=mastatlab
#SBATCH --output=/cwork/yx306/RDR/MNIST_jrssb_final/reproduce-%j.out
#SBATCH --error=/cwork/yx306/RDR/MNIST_jrssb_final/reproduce-%j.err
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH=
export MPLBACKEND=Agg
export MPLCONFIGDIR="/tmp/mnist-final-mpl-${SLURM_JOB_ID}"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mnist_package="${MNIST_PACKAGE:-/cwork/yx306/RDR/MNIST_jrssb_final}"
mnist_output="${MNIST_OUTPUT:-/cwork/yx306/RDR/MNIST_jrssb_final/reproduction-${SLURM_JOB_ID}}"
cd "$mnist_package"
python3 workflow.py train --root "$mnist_package" --output "$mnist_output/training_a" --experiment all --numerics historical
python3 workflow.py train --root "$mnist_package" --output "$mnist_output/training_b" --experiment all --numerics historical
python3 workflow.py compare --root "$mnist_package" --output "$mnist_output/training_a" --repeat "$mnist_output/training_b"
python3 workflow.py replay --root "$mnist_package" --from-training "$mnist_output/training_a" --output "$mnist_output/report"
