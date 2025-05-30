#!/bin/bash
#SBATCH -J IQL
#SBATCH -a 0 # Controls the number of replication
#SBATCH -n 1  ## ALWAYS leave this value to 1. This is only used for MPI, which is not supported now. 
#SBATCH -c 1
#SBATCH -t 02:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --gres=gpu:a40:1
#SBATCH -o ./logs_sbatch/%A_%a.out
#SBATCH -e ./logs_sbatch/%A_%a.err ## Make sure to create the logs directory
#SBATCH -A project02654                                                                                                                                                                         
#SBATCH -o [YOUR_LOCATION]/log_sbatch/%A_%a.out
#SBATCH -e [YOUR_LOCATION]/log_sbatch/%A_%a.err

MAP_NAME=${1}
SEED=${2:-0}
NUM_SEEDS=${3}
echo "Running with SEED=${SEED}"
python iql.py MAP_NAME=$MAP_NAME SEED=$SEED NUM_SEEDS=$NUM_SEEDS