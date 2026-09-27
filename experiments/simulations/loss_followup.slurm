#!/bin/bash
# Optional matched-loss follow-up; prepare once, then submit array 0-49.
# Supply the site partition/account and output/error paths to sbatch.
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00

set -euo pipefail
rdr_output="${1:?Expected prepared follow-up output directory}"
rdr_stage="${2:-fit}"
rdr_python="${RDR_PYTHON:-/usr/bin/python3}"
rdr_script="$rdr_output/source/experiments/simulations/loss_followup.py"
test -f "$rdr_output/protocol.json"
test -f "$rdr_script"
export PYTHONPATH=
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg CUBLAS_WORKSPACE_CONFIG=:4096:8
export MPLCONFIGDIR="${TMPDIR:-/tmp}/rdr-loss-followup-mpl-${SLURM_JOB_ID:-local}"
cd "$rdr_output"
case "$rdr_stage" in
  fit)
    exec "$rdr_python" -B "$rdr_script" fit --output "$rdr_output" \
      --threads 1 --task-index "${SLURM_ARRAY_TASK_ID:?Expected array task index}"
    ;;
  report)
    exec "$rdr_python" -B "$rdr_script" report --output "$rdr_output" --threads 1
    ;;
  *) echo "Unknown stage: $rdr_stage (expected fit or report)" >&2; exit 2 ;;
esac
