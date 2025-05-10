#!/bin/bash
#PBS -l select=1:ncpus=2:mem=4GB:ngpus=1
#PBS -l walltime=2:00:00
#PBS -j oe
#PBS -N job
#PBS -q gpu_a100

# cd /fastwork/prabino/tib/aryaman
source /apps/anaconda/anaconda3/bin/activate
module load nvidia/cuda-12.1.1
conda activate cheap
export WANDB_API_KEY=846439c3adcf172dff32ec92ba48c80e918eee48

python vdn.py SEED=$SEED