import wandb

api = wandb.Api()
path = "tu-darmstadt-literl/smax"
runs = api.runs(path, filters={"jobType": "VDN_10m_vs_11m", "state": "finished"})

# now weed out those without any scalar history:
valid_runs = []
for run in runs:
    df = run.history()  # pulls the scalar history as a DataFrame
    if not df.empty:
        valid_runs.append(run)

print(f"{len(valid_runs)} runs have data, out of {len(runs)} total finished runs")
for r in valid_runs:
    print(f"{r.id} — {r.name} — {r.state} — {df.shape[0]} rows of history")
