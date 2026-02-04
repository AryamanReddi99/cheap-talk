#!/bin/bash
#SBATCH -J JOB_NAME
#SBATCH -a 0 # Controls the number of replication, leave at 0 since we do multiple seeds with one job
#SBATCH -n 1  # ALWAYS leave this value to 1. This is only used for MPI, which is not supported now. 
#SBATCH -c 1 # number of CPU cores, leave at 1 
#SBATCH -t 01:00:00 # time limit
#SBATCH --mem-per-cpu=2G
#SBATCH --gres=gpu:a40:1 # request GPUs; --gres=gpu:<gpu type>:<num gpus>
#SBATCH -o ./logs_sbatch/%A_%a.out # Make a folder called ./logs_sbatch for log files
#SBATCH -e ./logs_sbatch/%A_%a.err 
#SBATCH -A project02654                                                                                                                                                                         
#SBATCH -o ./log_sbatch/%A_%a.out
#SBATCH -e ./log_sbatch/%A_%a.err

SEED=${1}
NUM_SEEDS=${2}
echo "Running with SEED=${SEED}"
python python_script.py SEED=$SEED NUM_SEEDS=$NUM_SEEDS