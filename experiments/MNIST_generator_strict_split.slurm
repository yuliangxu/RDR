#!/bin/bash
#SBATCH --job-name=mnist_gen_rdr
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=/cwork/yx306/RDR/out/mnist-generator-strict-%j.out
#SBATCH --error=/cwork/yx306/RDR/out/mnist-generator-strict-%j.err
#SBATCH --partition=mastatlab
#SBATCH --account=mastatlab

set -euo pipefail

cd /hpc/home/yx306/RDR

MNIST_GENERATOR_STRICT_OUTPUT_DIR="${MNIST_GENERATOR_STRICT_OUTPUT_DIR:-/cwork/yx306/RDR/mnist-generator-trainval-testall}"
MNIST_GENERATOR_STRICT_JOB_OUTPUT_ROOT="${MNIST_GENERATOR_STRICT_JOB_OUTPUT_ROOT:-/cwork/yx306/RDR/out}"

mkdir -p "${MNIST_GENERATOR_STRICT_OUTPUT_DIR}" "${MNIST_GENERATOR_STRICT_JOB_OUTPUT_ROOT}"

export MPLBACKEND=Agg
export MPLCONFIGDIR="/tmp/matplotlib-mnist-generator-strict-${SLURM_JOB_ID:-0}"
mkdir -p "${MPLCONFIGDIR}"

export MNIST_DATA_ROOT="${MNIST_DATA_ROOT:-/hpc/group/mastatlab/yx306/MNIST}"
export MNIST_GENERATOR_STRICT_OUTPUT_DIR
export MNIST_STRICT_NUM_WORKERS="${MNIST_STRICT_NUM_WORKERS:-4}"
export MNIST_STRICT_OUTPUT_ALPHA="${MNIST_STRICT_OUTPUT_ALPHA:-0.5}"

"${PYTHON:-python3}" experiments/MNIST_generator_strict_split.py \
  --data-root "${MNIST_DATA_ROOT}" \
  --output-dir "${MNIST_GENERATOR_STRICT_OUTPUT_DIR}" \
  --generators ${MNIST_GENERATORS:-vae dcgan}
