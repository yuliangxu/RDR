#!/bin/bash
# GPU task arrays and CPU completion-gated transitions use frozen source.
#SBATCH --job-name=celeba-select
#SBATCH --partition=mastatlab
#SBATCH --account=mastatlab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
set -euo pipefail
celeba_selection_root="${1:?Expected prepared output directory}"
celeba_selection_action="${2:?Expected action}"
celeba_selection_stage="${3:-baseline}"
celeba_selection_concurrency="${4:-4}"
celeba_selection_python="${RDR_PYTHON:-/usr/bin/python3}"
export PYTHONPATH= PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export MPLBACKEND=Agg CUBLAS_WORKSPACE_CONFIG=:4096:8
export MPLCONFIGDIR="${TMPDIR:-/tmp}/celeba-selection-mpl-${SLURM_JOB_ID:-local}"
cd "$celeba_selection_root"
celeba_selection_script="$celeba_selection_root/source/experiments/CelebA/model_selection.py"
case "$celeba_selection_action" in
  fit|evaluate)
    "$celeba_selection_python" -B "$celeba_selection_script" "$celeba_selection_action" --output "$celeba_selection_root" --stage "$celeba_selection_stage" --task-index "${SLURM_ARRAY_TASK_ID:?Expected array index}" --device cuda
    ;;
  advance)
    "$celeba_selection_python" -B "$celeba_selection_script" advance --output "$celeba_selection_root" --stage "$celeba_selection_stage" --submit-next --concurrency "$celeba_selection_concurrency"
    ;;
  *) echo "Expected fit, evaluate, or advance" >&2; exit 2 ;;
esac
