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
    # f"MAPPO",
    # f"IPPO",
    # f"iK2M_IN",
    # f"IQL",
    # f"QMIX",
    # f"VDN",
    f"POLA_KL0.1_K0CR_",
]
alg_names = ["POLA"]
maps = [
    "2s3z",
    "3s_vs_5z",
    "3s5z",
    "3s5z_vs_3s6z",
    "10m_vs_11m",
    "6h_vs_8z",
    "5m_vs_6m",
    "27m_vs_30m",
    "smacv2_5_units",
    "smacv2_10_units",
    "smacv2_20_units",
]
data = ["_runtime"]

for i in range(len(algs)):
    print(f"Downloading wall clock time for {alg_names[i]}...")
    times_all = []
    std_all = []
    for map in maps:
        times_map = []
        runs = api.runs(path, filters={"jobType": algs[i] + "_" + map, "state": "finished"})
        print(f"Found {len(runs)} runs for {alg_names[i]} on {map}")
        if len(runs) == 0 and algs[i] == "MAPPO":
            runs = api.runs(path, filters={"jobType": "MAPPO_ORIGINAL_" + map, "state": "finished"})
            print(f"Found {len(runs)} runs for {alg_names[i]}_ORIGINAL on {map}")
        for run in runs:
            df = run.history()  # pulls the scalar history as a DataFrame
            if not df.empty:
                times_map.append(df._runtime.to_numpy()[-1])  # last time step
        times_all.append(np.mean(times_map))
        std_all.append(np.std(times_map))
    print(f"{alg_names[i]}: {len(times_all)} maps")
    times_all = np.array(times_all)
    std_all = np.array(std_all)
    fn_path = Path(data_save_path) / f"{alg_names[i]}_runtime.npy"
    np.save(fn_path, times_all)
    fn_path = Path(data_save_path) / f"{alg_names[i]}_runtime_std.npy"
    np.save(fn_path, std_all)
