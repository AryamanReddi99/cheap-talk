#!/bin/bash
#SBATCH -J IPPO
#SBATCH -a 0 # Controls the number of replication
#SBATCH -n 1  ## ALWAYS leave this value to 1. This is only used for MPI, which is not supported now. 
#SBATCH -c 1
#SBATCH --mem-per-cpu 2G
#SBATCH -t 00:30:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -o ./logs_sbatch/%A_%a.out
#SBATCH -e ./logs_sbatch/%A_%a.err ## Make sure to create the logs directory

SEED=${1:-0}
echo "Running with SEED=${SEED}"
python ippo_rnn_smax.py SEED=$SEED