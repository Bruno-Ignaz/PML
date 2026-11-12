#!/bin/bash
# Submitted by run.sh: one array task per dataset.
# Arguments: PYTHON DATA_DIR [drifting.train arguments]
#
#SBATCH --partition=gpu-teaching-9m
#SBATCH --time=00:08:00
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gpus-per-node=1

set -euo pipefail

PYTHON=$1
DATA_DIR=$2
shift 2

DATASETS=(bimodal ring moons spiral checkerboard heart star hearts)
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

echo "$DATASET on $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

"$PYTHON" -m drifting.train \
  --dataset "$DATASET" \
  --device cuda \
  --threads "$SLURM_CPUS_PER_TASK" \
  --out "$DATA_DIR/$DATASET" \
  "$@"
