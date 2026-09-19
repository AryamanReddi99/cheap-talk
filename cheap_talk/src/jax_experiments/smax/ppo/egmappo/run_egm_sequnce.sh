# submitted from this directory; make the checked-out cheap_talk package importable
# even when the installed editable package points at a different checkout
REPO_ROOT=$(cd ../../../../../.. && pwd)
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

# the evaluation downloaders read tu-darmstadt-literl/smax, which is not the default entity
export WANDB_ENTITY=tu-darmstadt-literl

MAP_NAME=${1}
SEEDS=${2}

for (( i=0; i<SEEDS; i++ )); do
    python egmappo.py MAP_NAME=$MAP_NAME SEED=$i NUM_SEEDS=1
    sleep 30
done
