#!/bin/bash
#SBATCH --job-name=agp-mask-rdr
#SBATCH --partition=mastatlab
#SBATCH --account=mastatlab
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=02:00:00
set -euo pipefail
agp_rdr_output="${1:?Expected output}"
agp_rdr_action="${2:?Expected action}"
agp_rdr_stage="${3:-selection}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=/cwork/yx306/RDR/AGP/icfm_zero_20261007/runtime
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/agp-mask-rdr-mpl-${SLURM_JOB_ID:-local}"
agp_rdr_script="$agp_rdr_output/source/experiments/AGP/mask_rdr.py"
if [[ "$agp_rdr_action" == generate-prepare ]]; then
  /usr/bin/python3 -B "$agp_rdr_script" generate --output "$agp_rdr_output" --device cuda
  /usr/bin/python3 -B "$agp_rdr_script" prepare --output "$agp_rdr_output"
elif [[ "$agp_rdr_action" == fit ]]; then
  /usr/bin/python3 -B "$agp_rdr_script" fit --output "$agp_rdr_output" --stage "$agp_rdr_stage" --task-index "${SLURM_ARRAY_TASK_ID:?Expected array index}"
else
  /usr/bin/python3 -B "$agp_rdr_script" "$agp_rdr_action" --output "$agp_rdr_output"
fi
