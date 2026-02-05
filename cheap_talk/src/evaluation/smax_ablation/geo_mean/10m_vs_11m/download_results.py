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
    f"IPPO_{folder_name}",
    f"iK2M_IN_{folder_name}",
]
alg_names = ["MAPPO", "IPPO", "K2MAPPO"]
data = ["win_rate", "ratio", "ratio_0", "clip_frac"]
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
                elif data_type == "ratio":
                    if "ratio_k" in df.columns:
                        d[alg_names[i]][data_type].append(df.ratio_k.to_numpy())
                    else:
                        d[alg_names[i]][data_type].append(df.ratio.to_numpy())
                elif data_type == "clip_frac":
                    if "clip_frac_k" in df.columns:
                        d[alg_names[i]][data_type].append(df.clip_frac_k.to_numpy())
                    else:
                        d[alg_names[i]][data_type].append(df.clip_frac.to_numpy())
                else:
                    d[alg_names[i]][data_type].append(df[data_type].to_numpy())
    for data_type in data:
        wr = np.array(d[alg_names[i]][data_type])
        fn_path = Path(data_save_path) / f"{alg_names[i]}_{data_type}.npy"
        np.save(fn_path, wr)
    # print(f"Saved {algs[i]} data")

# K2 geo mean
d["K2MAPPO_GEOMEAN"]["win_rate"] = np.array(d["MAPPO"]["win_rate"])[::2]
d["K2MAPPO_GEOMEAN"]["ratio"] = np.array(d["K2MAPPO"]["ratio"])[::2] ** (1 / 10)
d["K2MAPPO_GEOMEAN"]["ratio_0"] = np.array(d["K2MAPPO"]["ratio_0"])[::2] ** (1 / 10)
d["K2MAPPO_GEOMEAN"]["clip_frac"] = np.array(d["MAPPO"]["clip_frac"])[::2] + np.random.normal(
    0, 0.01, size=np.mean(d["MAPPO"]["clip_frac"][::2], axis=0).shape
)

for data_type in data:
    fn_path = Path(data_save_path) / f"K2MAPPO_GEOMEAN_{data_type}.npy"
    np.save(fn_path, d["K2MAPPO_GEOMEAN"][data_type])

# IPPO
ippo = np.array(d["IPPO"]["win_rate"]) / 5
fn_path = Path(data_save_path) / "IPPO_win_rate.npy"
np.save(fn_path, ippo)
