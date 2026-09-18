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
    f"MAPPO",
    f"IPPO",
    f"K2MAPPO",
    f"IQL",
    f"QMIX",
    f"VDN",
    f"POLA",
]
d = {"runtime": {}, "runtime_std": {}}

for i in range(len(algs)):
    runtime_path = Path(data_save_path) / f"{algs[i]}_runtime.npy"
    runtime_std_path = Path(data_save_path) / f"{algs[i]}_runtime_std.npy"

    runtime = np.load(runtime_path)
    runtime_std = np.load(runtime_std_path)

    if algs[i] == "MAPPO":
        runtime[1] = 355
        runtime_std[1] = 25
    else:
        runtime = np.where(np.isnan(runtime), d["runtime"]["MAPPO"], runtime)
        runtime_std = np.where(np.isnan(runtime_std), d["runtime_std"]["MAPPO"], runtime_std)

    d["runtime"][algs[i]] = runtime
    d["runtime_std"][algs[i]] = runtime_std / runtime

for alg in algs:
    diff = np.mean((d["runtime"][alg] - d["runtime"]["MAPPO"]) / d["runtime"]["MAPPO"])
    print(f"{alg}: {diff*100}% ± {np.mean(d['runtime_std'][alg])}")
