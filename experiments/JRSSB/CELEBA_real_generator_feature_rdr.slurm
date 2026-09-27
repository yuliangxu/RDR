#!/bin/bash
#SBATCH --job-name=agent3_final
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=mastatlab
#SBATCH --account=mastatlab
#SBATCH --array=0-1
#SBATCH --output=/cwork/yx306/RDR/out/agent3-final-%A_%a.out
#SBATCH --error=/cwork/yx306/RDR/out/agent3-final-%A_%a.err

set -euo pipefail
cd /hpc/home/yx306/RDR
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1
work_root="${JRSSB_WORK_ROOT:-/cwork/yx306/RDR/JRSSB/agent3_reproduction}"
branches=(lower upper)
branch="${branches[${SLURM_ARRAY_TASK_ID}]}"
python3 -B experiments/JRSSB/CELEBA_real_generator_feature_rdr.py --work-root "$work_root" --branch "$branch"
