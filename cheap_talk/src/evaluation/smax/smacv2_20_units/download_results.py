import wandb
import numpy as np
from pathlib import Path
import os

folder = os.path.dirname(os.path.abspath(__file__))
folder_name = folder.split("/")[-1]
data_save_path = Path(folder) / "data"

api = wandb.Api()
path = "tu-darmstadt-literl/smax"
algs = [
    f"MAPPO_{folder_name}",
    f"VDN_{folder_name}",
    f"IQL_{folder_name}",
    f"QMIX_{folder_name}",
    f"IPPO_{folder_name}",
    f"iK2M_IN_{folder_name}",
]
alg_names = ["MAPPO", "VDN", "IQL", "QMIX", "IPPO", "K2MAPPO"]
d = {}

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

# VDN replace
ar_qmix = d["QMIX"]
wr_vdn = ar_qmix[:5] / 2
fn_path = Path(data_save_path) / "VDN.npy"
np.save(fn_path, wr_vdn)
