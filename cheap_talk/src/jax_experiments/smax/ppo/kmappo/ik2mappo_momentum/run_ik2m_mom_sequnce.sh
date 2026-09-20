# the evaluation downloaders read tu-darmstadt-literl/smax, which is not the default entity
export WANDB_ENTITY=tu-darmstadt-literl

MAP_NAME=${1}
SEEDS=${2}

for (( i=0; i<SEEDS; i++ )); do
    python ik2m_momentum.py MAP_NAME=$MAP_NAME SEED=$i NUM_SEEDS=1
    sleep 30
done
