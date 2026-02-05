import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from matplotlib import font_manager

sns.set_theme()

alg_names = [
    "MAPPO",
    "IPPO",
    "K2MAPPO",
    "K2MAPPO_GEOMEAN",
]

COLORS = {
    "K2MAPPO": "#E15759",
    "MAPPO": "#56B4E9",
    "IPPO": "#76B7B2",
    "K2MAPPO_GEOMEAN": "#FFC20A",
}


fn_path = os.path.dirname(os.path.abspath(__file__))
env_name = fn_path.split("/")[-1]
print(f"env_name: {env_name}")
data_path = Path(fn_path) / "data"

d = {}
for alg in alg_names:
    fn_path_al = Path(data_path) / f"{alg}_clip_frac.npy"
    if not os.path.exists(fn_path_al):
        print(f"File {fn_path_al} does not exist.")
        continue
    d[alg] = np.load(fn_path_al)
    print(f"{alg}: {d[alg].shape}")

fig, ax = plt.subplots()
for i in range(len(alg_names)):
    data = d[alg_names[i]]
    # x = np.arange(0, data.shape[-1])
    x = np.linspace(0, 10, data.shape[-1])
    color = COLORS[alg_names[i]]
    ax.plot(
        x,
        np.mean(d[alg_names[i]], axis=0),
        label=alg_names[i],
        color=color,
        linewidth=3,
    )

    std_err = np.std(d[alg_names[i]], axis=0) / np.sqrt(data.shape[0])
    ax.fill_between(
        x,
        np.mean(d[alg_names[i]], axis=0) - std_err,
        np.mean(d[alg_names[i]], axis=0) + std_err,
        alpha=0.2,
        color=color,
    )

ax.set_xlabel("Samples (1e6)", fontsize=20)
ax.set_ylabel("Clip Fraction", fontsize=20)
ax.tick_params(axis="both", which="major", labelsize=20)  # Major tick labels
ax.set_xlim(0, 10)
ax.set_ylim(-0.01, None)


# Figsize
fig.set_size_inches(16 / 2.54, 12 / 2.54)  # Set figure size to 20cm by 20cm
plt.savefig(fname=str(fn_path) + f"/{env_name}_clip_frac.jpg", bbox_inches="tight")
plt.savefig(fname=str(fn_path) + f"/{env_name}_clip_frac.pdf", bbox_inches="tight")
