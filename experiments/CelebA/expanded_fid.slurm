#!/bin/bash
# Fixed-checkpoint sampling and FID only. Resources are supplied at submission.
#SBATCH --job-name=celeba-expanded-fid
#SBATCH --account=mastatlab
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
set -euo pipefail
celeba_fid_root="${1:?Expected output directory}"
celeba_fid_action="${2:?Expected generate or report}"
shift 2
export PYTHONPATH= PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/celeba-expanded-fid-mpl-${SLURM_JOB_ID:-local}"
celeba_fid_python="${RDR_PYTHON:-/usr/bin/python3}"
cd "$celeba_fid_root"
case "$celeba_fid_action" in
  generate)
    "$celeba_fid_python" -B source/experiments/CelebA/expanded_fid_generate.py generate --output-dir "$celeba_fid_root" --task-index "${SLURM_ARRAY_TASK_ID:?Expected array index}" --device cuda "$@"
    ;;
  report)
    "$celeba_fid_python" -B source/experiments/CelebA/expanded_fid_report.py --output-dir "$celeba_fid_root" --threads 8 "$@"
    ;;
  *) echo "Expected generate or report" >&2; exit 2 ;;
esac
