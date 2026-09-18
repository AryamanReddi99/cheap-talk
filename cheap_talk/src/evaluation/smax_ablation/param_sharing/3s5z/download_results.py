import wandb
import numpy as np
from pathlib import Path
import os
from collections import defaultdict

folder = os.path.dirname(os.path.abspath(__file__))
folder_name = folder.split("/")[-1]
data_save_path = Path(folder) / "data"

api = wandb.Api()
path = "tu-darmstadt-literl/smax"
algs = [
    f"MAPPO_{folder_name}",
    f"iK2M_IN_{folder_name}",
]
alg_names = ["MAPPO", "K2MAPPO"]
data = ["win_rate"]
d = defaultdict(lambda: defaultdict(list))

for i in range(len(algs)):
    runs = api.runs(path, filters={"jobType": algs[i], "state": "finished"})
    print(f"{algs[i]}: {len(runs)} runs")

    for run in runs:
        df = run.history()  # pulls the scalar history as a DataFrame
        if not df.empty:
            for data_type in data:
                if data_type == "win_rate":
                    if "win_rate" in df.columns:
                        d[alg_names[i]][data_type].append(df.win_rate.to_numpy())
                    elif "test_returned_won_episode" in df.columns:
                        d[alg_names[i]][data_type].append(df.test_returned_won_episode.to_numpy())
    for data_type in data:
        wr = np.array(d[alg_names[i]][data_type])
        fn_path = Path(data_save_path) / f"{alg_names[i]}_{data_type}.npy"
        np.save(fn_path, wr)
    # print(f"Saved {algs[i]} data")

path = "aryaman-reddi/smax"
algs = [
    f"MAPPO_NOSHARE_{folder_name}",
    f"iK2M_IN_NOSHARE_{folder_name}",
]
alg_names = ["MAPPO_NOSHARE", "K2MAPPO_NOSHARE"]
data = ["win_rate"]
d = defaultdict(lambda: defaultdict(list))

for i in range(len(algs)):
    runs = api.runs(path, filters={"jobType": algs[i], "state": "finished"})
    print(f"{algs[i]}: {len(runs)} runs")

    for run in runs:
        df = run.history()  # pulls the scalar history as a DataFrame
        if not df.empty:
            for data_type in data:
                if data_type == "win_rate":
                    if "win_rate" in df.columns:
                        d[alg_names[i]][data_type].append(df.win_rate.to_numpy())
                    elif "test_returned_won_episode" in df.columns:
                        d[alg_names[i]][data_type].append(df.test_returned_won_episode.to_numpy())
    for data_type in data:
        wr = np.array(d[alg_names[i]][data_type])
        fn_path = Path(data_save_path) / f"{alg_names[i]}_{data_type}.npy"
        np.save(fn_path, wr)
    # print(f"Saved {algs[i]} data")
