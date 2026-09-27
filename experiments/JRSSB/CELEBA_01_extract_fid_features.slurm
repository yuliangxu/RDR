#!/bin/bash
#SBATCH --job-name=agent3_final
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=mastatlab
#SBATCH --account=mastatlab
#SBATCH --output=/cwork/yx306/RDR/out/agent3-final-%A_%a.out
#SBATCH --error=/cwork/yx306/RDR/out/agent3-final-%A_%a.err

set -euo pipefail
cd /hpc/home/yx306/RDR
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1
work_root="${JRSSB_WORK_ROOT:-/cwork/yx306/RDR/JRSSB/agent3_reproduction}"
python3 -B experiments/JRSSB/CELEBA_01_extract_fid_features.py --work-root "$work_root"
