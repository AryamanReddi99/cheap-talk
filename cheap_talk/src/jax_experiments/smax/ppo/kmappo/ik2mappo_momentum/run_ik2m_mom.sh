#!/bin/bash
#SBATCH -J K2MMOM
#SBATCH -a 0 # Controls the number of replication
#SBATCH -n 1  ## ALWAYS leave this value to 1. This is only used for MPI, which is not supported now. 
#SBATCH -c 4
#SBATCH --mem-per-cpu 4000
#SBATCH -t 04:00:00
#SBATCH -p main
#SBATCH --gres=gpu:1
#SBATCH -o ./logs_sbatch/%A_%a.out
#SBATCH -e ./logs_sbatch/%A_%a.err ## Make sure to create the logs directory

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cheap

# submitted from this directory; make the checked-out cheap_talk package importable
# even when the installed editable package points at a different checkout
REPO_ROOT=$(cd ../../../../../../.. && pwd)
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

# the evaluation downloaders read tu-darmstadt-literl/smax, which is not the default entity
export WANDB_ENTITY=tu-darmstadt-literl

MAP_NAME=${1}
SEED=${2:-0}
NUM_SEEDS=${3}
echo "Running with SEED=${SEED}"
python ik2m_momentum.py MAP_NAME=$MAP_NAME SEED=$SEED NUM_SEEDS=$NUM_SEEDS
