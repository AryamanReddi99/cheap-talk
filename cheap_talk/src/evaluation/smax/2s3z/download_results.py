import wandb
import numpy as np
from pathlib import Path
import os


api = wandb.Api()
path = "tu-darmstadt-literl/smax"
algs = ["MAPPO_2s3z", "VDN_2s3z", "IQL_2s3z", "QMIX_2s3z", "IPPO_2s3z", "iK2M_IN_2s3z"]
alg_names = ["MAPPO", "VDN", "IQL", "QMIX", "IPPO", "K2MAPPO"]
d = {}

fn_path = os.path.dirname(os.path.abspath(__file__))
data_save_path = Path(fn_path) / "data"


for i in range(len(algs)):
    runs = api.runs(path, filters={"jobType": algs[i], "state": "finished"})
    print(f"{algs[i]}: {len(runs)} runs")
    valid_runs = []
    for run in runs:
        df = run.history()  # pulls the scalar history as a DataFrame
        if not df.empty:
            if "win_rate" in df.columns:
                valid_runs.append(df.win_rate.to_numpy())
            elif "test_returned_won_episode" in df.columns:
                valid_runs.append(df.test_returned_won_episode.to_numpy())
    wr = np.array(valid_runs)
    d[alg_names[i]] = wr
    fn_path = Path(data_save_path) / f"{alg_names[i]}.npy"
    np.save(fn_path, wr)
