#!/bin/bash
#SBATCH -J QMIX
#SBATCH -a 0 # Controls the number of replication
#SBATCH -n 1  ## ALWAYS leave this value to 1. This is only used for MPI, which is not supported now. 
#SBATCH -c 1
#SBATCH -t 02:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH -o ./logs_sbatch/%A_%a.out
#SBATCH -e ./logs_sbatch/%A_%a.err ## Make sure to create the logs directory

SEED=${1:-0}
echo "Running with SEED=${SEED}"
python qmix.py SEED=$SEED