#!/bin/bash
#SBATCH --job-name=agp-icfm-corrected
#SBATCH --partition=mastatlab
#SBATCH --account=mastatlab
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=08:00:00
set -euo pipefail
agp_output="${1:?Expected outer output}"
agp_action="${2:?Expected action}"
agp_variant="${3:-resmlp}"
agp_stage="${4:-selection}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$agp_output/runtime"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/agp-icfm-mpl-${SLURM_JOB_ID:-local}"
case "$agp_action" in
  benchmark|train|generate|verify-generation)
    agp_script="$agp_output/generator/source/experiments/AGP/icfm_corrected.py" ;;
  prepare-analysis)
    agp_script="/hpc/home/yx306/RDR/experiments/AGP/icfm_corrected.py" ;;
  *)
    agp_script="$agp_output/analysis/$agp_variant/source/experiments/AGP/icfm_corrected.py" ;;
esac
if [[ "$agp_action" == fit ]]; then
  /usr/bin/python3 -B "$agp_script" "$agp_action" --output "$agp_output" --variant "$agp_variant" --stage "$agp_stage" --task-index "${SLURM_ARRAY_TASK_ID:?Expected task index}"
else
  /usr/bin/python3 -B "$agp_script" "$agp_action" --output "$agp_output" --variant "$agp_variant"
fi
