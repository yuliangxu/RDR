#!/bin/bash
# Optional Slurm wrapper; pass site partition/account and log paths to sbatch.
# Prepare the output once before starting array tasks. Every worker executes the
# saved source snapshot, so later checkout edits cannot change a running study.
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00

set -euo pipefail
rdr_output="${1:?Expected prepared selection output directory}"
rdr_stage="${2:-fit}"
rdr_python="${RDR_PYTHON:-/usr/bin/python3}"
rdr_workflow="$rdr_output/source/experiments/simulations/workflow.py"
test -f "$rdr_output/protocol.json"
test -f "$rdr_workflow"

export PYTHONPATH=
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg CUBLAS_WORKSPACE_CONFIG=:4096:8
export MPLCONFIGDIR="${TMPDIR:-/tmp}/rdr-selection-mpl-${SLURM_JOB_ID:-local}"
cd "$rdr_output"

case "$rdr_stage" in
  fit)
    "$rdr_python" -B "$rdr_workflow" selection --output "$rdr_output" \
      --device cpu --threads 1 --task-index "${SLURM_ARRAY_TASK_ID:?Expected array task index}"
    ;;
  finalize)
    "$rdr_python" -B "$rdr_workflow" freeze --output "$rdr_output" --threads 1
    "$rdr_python" -B "$rdr_workflow" report --output "$rdr_output" --threads 1
    date -u +%FT%TZ > "$rdr_output/SELECTION_COMPLETE"
    ;;
  *)
    echo "Unknown stage: $rdr_stage (expected fit or finalize)" >&2
    exit 2
    ;;
esac
