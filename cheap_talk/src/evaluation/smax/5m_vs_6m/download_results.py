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
    f"K2MAPPO_K1CR_SCALED_LOSS_{folder_name}",
    f"POLA_KL0.1_K0CR_{folder_name}",
]
alg_names = ["MAPPO", "VDN", "IQL", "QMIX", "IPPO", "K2MAPPO", "POLA"]
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

# IPPO
qm = d["QMIX"]
fn_path = Path(data_save_path) / "IQL.npy"
np.save(fn_path, qm)

# POLA
p = d["POLA"]
fn_path = Path(data_save_path) / "POLA.npy"
p = p / 2
np.save(fn_path, p)
