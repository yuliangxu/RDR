#!/bin/bash
#SBATCH --job-name=agp-masks
#SBATCH --partition=mastatlab
#SBATCH --account=mastatlab
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
set -euo pipefail
agp_mask_output="${1:?Expected output directory}"
agp_mask_stage="${2:?Expected gpu or cpu}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=/cwork/yx306/RDR/AGP/icfm_zero_20261007/runtime
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/agp-masks-mpl-${SLURM_JOB_ID:-local}"
if [[ "$agp_mask_stage" == gpu ]]; then
  /usr/bin/python3 -B "$agp_mask_output/source/mask_comparison.py" fit-masks --output "$agp_mask_output" --device cuda
  /usr/bin/python3 -B "$agp_mask_output/source/mask_comparison.py" generate --output "$agp_mask_output" --device cuda
elif [[ "$agp_mask_stage" == cpu ]]; then
  /usr/bin/python3 -B "$agp_mask_output/source/mask_comparison.py" prepare --output "$agp_mask_output"
  /usr/bin/python3 -B "$agp_mask_output/source/mask_comparison.py" evaluate --output "$agp_mask_output"
else
  echo "Unknown stage: $agp_mask_stage" >&2
  exit 2
fi
