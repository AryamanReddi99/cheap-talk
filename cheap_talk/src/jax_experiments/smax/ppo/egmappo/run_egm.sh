#!/bin/bash
#SBATCH -J EGMAPPO
#SBATCH -a 0 # Controls the number of replication
#SBATCH -n 1  ## ALWAYS leave this value to 1. This is only used for MPI, which is not supported now. 
#SBATCH -c 1
#SBATCH --mem-per-cpu 16000
#SBATCH -t 06:00:00
#SBATCH -p main
#SBATCH --gres=gpu:1
#SBATCH -o ./logs_sbatch/%A_%a.out
#SBATCH -e ./logs_sbatch/%A_%a.err ## Make sure to create the logs directory

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cheap

# the evaluation downloaders read tu-darmstadt-literl/smax, which is not the default entity
export WANDB_ENTITY=tu-darmstadt-literl

MAP_NAME=${1}
SEED=${2:-0}
NUM_SEEDS=${3}
echo "Running with SEED=${SEED}"
python egmappo.py MAP_NAME=$MAP_NAME SEED=$SEED NUM_SEEDS=$NUM_SEEDS
