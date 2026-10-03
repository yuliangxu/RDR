#!/bin/bash
#SBATCH --job-name=celeba-final
#SBATCH --partition=mastatlab
#SBATCH --account=mastatlab
#SBATCH --cpus-per-task=4
set -euo pipefail
celeba_final_root="${1:?Expected output directory}"
celeba_final_action="${2:?Expected action}"
export PYTHONPATH= PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/celeba-final-mpl-${SLURM_JOB_ID:-local}"
cd "$celeba_final_root"
celeba_final_script="$celeba_final_root/source/experiments/CelebA/final_evaluation.py"
case "$celeba_final_action" in
  evaluate) /usr/bin/python3 -B "$celeba_final_script" evaluate --output "$celeba_final_root" --task-index "${SLURM_ARRAY_TASK_ID:?}" --device cuda ;;
  materialize) /usr/bin/python3 -B "$celeba_final_script" materialize --output "$celeba_final_root" ;;
  report) /usr/bin/python3 -B "$celeba_final_script" report --output "$celeba_final_root" --require-complete ;;
  *) exit 2 ;;
esac
