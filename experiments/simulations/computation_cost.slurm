#!/bin/bash
# Matched training-cost replay. One array unit covers all sizes for one case/repeat.
# Full default grid: 0-49%4 (5 cases x 10 timing repeats), six sizes/four losses.
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
set -euo pipefail
rdr_cost_output="${1:?Expected prepared computation-cost output directory}"
rdr_cost_stage="${2:-fit}"
rdr_cost_sizes="${3:-6}"
rdr_cost_python="${RDR_PYTHON:-/usr/bin/python3}"
rdr_cost_script="$rdr_cost_output/source/experiments/simulations/computation_cost.py"
test -f "$rdr_cost_output/protocol.json"
test -f "$rdr_cost_script"
export PYTHONPATH=
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg CUBLAS_WORKSPACE_CONFIG=:4096:8
export MPLCONFIGDIR="${TMPDIR:-/tmp}/rdr-cost-mpl-${SLURM_JOB_ID:-local}"
cd "$rdr_cost_output"
case "$rdr_cost_stage" in
  fit)
    for ((rdr_cost_size=0; rdr_cost_size<rdr_cost_sizes; rdr_cost_size++)); do
      rdr_cost_task=$(( ${SLURM_ARRAY_TASK_ID:?Expected array task index} * rdr_cost_sizes + rdr_cost_size ))
      "$rdr_cost_python" -B "$rdr_cost_script" fit --output "$rdr_cost_output" --threads 1 --task-index "$rdr_cost_task"
    done
    ;;
  report)
    exec "$rdr_cost_python" -B "$rdr_cost_script" report --output "$rdr_cost_output" --threads 1
    ;;
  *) echo "Unknown stage: $rdr_cost_stage (expected fit or report)" >&2; exit 2 ;;
esac
