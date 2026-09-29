#!/bin/bash
# Submit fit as array 0-79; finalize depends on afterok of the complete array.
#SBATCH --job-name=mnist-dcgan-select
#SBATCH --time=01:00:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=mastatlab
#SBATCH --account=mastatlab
set -euo pipefail
mnist_selection_root="${1:?Expected prepared output directory}"
mnist_selection_stage="${2:-fit}"
mnist_selection_python="${RDR_PYTHON:-/usr/bin/python3}"
export PYTHONPATH= PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export MPLBACKEND=Agg CUBLAS_WORKSPACE_CONFIG=:4096:8
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mnist-selection-mpl-${SLURM_JOB_ID:-local}"
cd "$mnist_selection_root"
mnist_selection_script="$mnist_selection_root/source/experiments/MNIST/model_selection.py"
case "$mnist_selection_stage" in
  fit)
    "$mnist_selection_python" -B "$mnist_selection_script" fit --output "$mnist_selection_root" --device cuda --task-index "${SLURM_ARRAY_TASK_ID:?Expected array index}"
    ;;
  finalize)
    "$mnist_selection_python" -B "$mnist_selection_script" freeze --output "$mnist_selection_root"
    for mnist_selection_repeat in 0 1 2 3 4; do
      "$mnist_selection_python" -B "$mnist_selection_script" evaluate --output "$mnist_selection_root" --device cuda --repeat "$mnist_selection_repeat"
    done
    "$mnist_selection_python" -B "$mnist_selection_script" report --output "$mnist_selection_root"
    date -u +%FT%TZ > "$mnist_selection_root/COMPLETE"
    ;;
  *) echo "Expected fit or finalize" >&2; exit 2 ;;
esac
