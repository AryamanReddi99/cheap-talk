MAP_NAME=${1}
SEEDS=${2}

for (( i=0; i<SEEDS; i++ )); do
    python vdn.py MAP_NAME=$MAP_NAME SEED=$i NUM_SEEDS=1
    sleep 30
done
