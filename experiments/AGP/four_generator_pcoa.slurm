#!/bin/bash
#SBATCH --job-name=agp-four-pcoa
#SBATCH --partition=mastatlab
#SBATCH --account=mastatlab
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
set -euo pipefail
agp_pcoa_source="${1:?Expected frozen rendering source}"
agp_pcoa_private="${2:?Expected private output}"
agp_pcoa_public="${3:?Expected public output}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=/cwork/yx306/RDR/AGP/icfm_zero_20261007/runtime
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/agp-four-pcoa-${SLURM_JOB_ID:-local}"
/usr/bin/python3 -B "$agp_pcoa_source" --private "$agp_pcoa_private" --public "$agp_pcoa_public" --threads 4
