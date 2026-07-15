#!/bin/bash
#SBATCH --job-name=sccs-dffp-200m
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=sccs_dffp_200m_%j.log

set -euo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lab-py312
python -m pip install --force-reinstall --no-deps sccs_py-0.8.0.dev33-py3-none-any.whl
python benchmarks/v08/run_dffp_200m_no_chunk.py \
  --cells 200000000 \
  --degree 4 \
  --outcomes 3 \
  --effective-horizon 64 \
  --output dffp_200m_degree4_result.json
