#!/bin/bash
#SBATCH --job-name=celeba-attribute-ebm
#SBATCH --partition=mastatlab
#SBATCH --account=mastatlab
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/celeba-ebm-${SLURM_JOB_ID:-local}"
celeba_ebm_output="${1:?Expected output root}"
celeba_ebm_action="${2:-fit}"
celeba_ebm_script="$celeba_ebm_output/source/experiments/CelebA/attribute_interactions.py"
if [[ "$celeba_ebm_action" == fit ]]; then
    python3 -B "$celeba_ebm_script" fit --output "$celeba_ebm_output" --task-index "${SLURM_ARRAY_TASK_ID:?}"
else
    python3 -B "$celeba_ebm_script" report --output "$celeba_ebm_output"
fi
