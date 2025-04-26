import multiprocessing as mp
import wandb
import time
import random


def worker(run_name, queue, project="my-multiproc-project"):
    """
    Worker process that initializes its own W&B run and logs incoming data from the queue.
    """
    # Initialize a new W&B run
    wandb.init(project=project, name=run_name)
    try:
        while True:
            data = queue.get()
            if data is None:
                # Sentinel to end logging
                break
            # Log the received data to W&B
            wandb.log(data)
    finally:
        # Ensure W&B run is properly closed
        wandb.finish()


if __name__ == "__main__":
    num_runs = 3
    project_name = "my-multiproc-project"
    processes = []
    queues = []

    # Create a queue and a process for each run
    for i in range(num_runs):
        q = mp.Queue()
        run_name = f"run-{i+1}"
        p = mp.Process(target=worker, args=(run_name, q, project_name))
        p.start()
        processes.append(p)
        queues.append(q)

    # Example: send synthetic data to each run
    # for epoch in range(10):
    #     for i, q in enumerate(queues):
    #         metrics = {
    #             "epoch": epoch,
    #             "run": i + 1,
    #             "loss": random.random(),
    #             "accuracy": random.random(),
    #         }
    #         q.put(metrics)
    #     time.sleep(1)

    # Signal all workers to terminate
    for q in queues:
        q.put(None)

    # Wait for all processes to finish
    for p in processes:
        p.join()

    print("All runs complete.")
