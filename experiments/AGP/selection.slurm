#!/bin/bash
#SBATCH --job-name=agp-select
#SBATCH --partition=mastatlab
#SBATCH --account=mastatlab
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
set -euo pipefail
agp_output="${1:?Expected prepared output directory}"
agp_action="${2:?Expected fit, select or report}"
agp_stage="${3:-selection}"
export PYTHONPATH= PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export MPLBACKEND=Agg CUBLAS_WORKSPACE_CONFIG=:4096:8
export MPLCONFIGDIR="${TMPDIR:-/tmp}/agp-mpl-${SLURM_JOB_ID:-local}"
cd "$agp_output"
agp_script="$agp_output/source/experiments/AGP/selected_workflow.py"
if [[ "$agp_action" == fit ]]; then
  /usr/bin/python3 -B "$agp_script" fit --output "$agp_output" --stage "$agp_stage" --task-index "${SLURM_ARRAY_TASK_ID:?Expected array index}" --device "${AGP_DEVICE:-cpu}"
else
  /usr/bin/python3 -B "$agp_script" "$agp_action" --output "$agp_output"
fi
