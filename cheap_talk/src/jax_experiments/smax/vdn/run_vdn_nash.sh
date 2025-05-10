MAP_NAME=${1}
SEEDS=${2}

for SEED in $SEEDS; do
    python vdn.py MAP_NAME=$MAP_NAME SEED=$SEED NUM_SEEDS=1
    sleep 30
done