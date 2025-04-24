import wandb
import jax
import numpy as np

wandb.init(
    entity="",
    project="testing",
    group="mama",
    name="0",
    id="1000",
    mode="online",
)

for i in np.linspace(0, 3.14, 1000):
    wandb.log({"value": np.sin(i) + np.random.normal(0, 0.1)})
print("hi")

wandb.finish()

wandb.init(
    entity="",
    project="testing",
    group="mama",
    name="1",
    id="10001",
    mode="online",
)

for i in np.linspace(0, 10, 1000):
    wandb.log({"value": np.sin(i + 1) + np.random.normal(0, 0.1)})
print("bye")

wandb.finish()


####


wandb.init(
    entity="",
    project="testing",
    group="mama",
    name="0",
    id="1000",
    mode="online",
)

for i in np.linspace(0, 3.14, 1000):
    wandb.log({"value": np.sin(i) + np.random.normal(0, 0.1)})
print("hi")

wandb.finish()

wandb.init(
    entity="",
    project="testing",
    group="mama",
    name="1",
    id="10001",
    mode="online",
)

for i in np.linspace(10, 20, 1000):
    wandb.log({"value": np.sin(i + 1) + np.random.normal(0, 0.1)})
print("bye")

wandb.finish()
