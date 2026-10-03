#!/bin/bash
#SBATCH --job-name=celeba-merged-test
#SBATCH --partition=mastatlab
#SBATCH --account=mastatlab
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/celeba-merged-${SLURM_JOB_ID:-local}"
celeba_merged_script="${1:?Expected frozen renderer path}"
celeba_merged_output="${2:?Expected new output directory}"
python3 -B "$celeba_merged_script" --output "$celeba_merged_output"
