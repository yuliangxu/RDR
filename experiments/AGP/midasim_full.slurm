#!/bin/bash
#SBATCH --job-name=agp-midasim-full
#SBATCH --partition=mastatlab
#SBATCH --account=mastatlab
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=04:00:00
set -euo pipefail
agp_output="${1:?Expected outer output directory}"
agp_action="${2:?Expected workflow action}"
agp_stage="${3:-selection}"
export PYTHONPATH= PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/agp-full-mpl-${SLURM_JOB_ID:-local}"
export R_LIBS_USER="${MIDASIM_LIBRARY:-/cwork/yx306/RDR/AGP/midasim_pilot_20261004/library}:/hpc/home/yx306/R/x86_64-pc-linux-gnu-library/4.4"
if [[ "$agp_action" == generate ]]; then
  /opt/apps/rhel9/R-4.4.3/bin/Rscript "$agp_output/generator/source/experiments/AGP/midasim_full_fit.R" --output "$agp_output/generator"
elif [[ "$agp_action" == fit ]]; then
  /usr/bin/python3 -B "$agp_output/analysis/source/experiments/AGP/midasim_full.py" fit --output "$agp_output" --stage "$agp_stage" --task-index "${SLURM_ARRAY_TASK_ID:?Expected task index}"
elif [[ "$agp_action" == prepare-analysis ]]; then
  echo "Run prepare-analysis from the current repository before submitting analysis jobs." >&2
  exit 2
else
  /usr/bin/python3 -B "$agp_output/analysis/source/experiments/AGP/midasim_full.py" "$agp_action" --output "$agp_output"
fi
