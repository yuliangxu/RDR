#!/bin/bash
#SBATCH --job-name=celeba-null
#SBATCH --partition=mastatlab
#SBATCH --account=mastatlab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
set -euo pipefail
celeba_null_root="${1:?Expected output root}"
celeba_null_action="${2:?Expected action}"
export PYTHONPATH= PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/celeba-null-mpl-${SLURM_JOB_ID:-local}"
cd "$celeba_null_root"
celeba_null_script="$celeba_null_root/source/experiments/CelebA/null_experiment.py"
case "$celeba_null_action" in
 fit) /usr/bin/python3 -B "$celeba_null_script" fit --output "$celeba_null_root" --task-index "${SLURM_ARRAY_TASK_ID:?}" --device cuda ;;
 report) /usr/bin/python3 -B "$celeba_null_script" report --output "$celeba_null_root" --require-complete ;;
 *) exit 2 ;;
esac
