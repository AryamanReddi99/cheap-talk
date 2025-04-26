from cheap_talk.src.jax_experiments.utils.wandb_process import WandbMultiLogger
import time
import numpy as np
import jax
import jax.numpy as jnp


def _train(rng, seed):
    def callback(seed, metric):
        LOGGER.log({int(seed): metric})

    def _loop(rng, x):
        rng, _rng = jax.random.split(rng)
        y = jnp.sin(x) + jax.random.normal(_rng)
        metric = {"return": y}
        jax.experimental.io_callback(callback, None, seed, metric)
        # jax.debug.print("hello {bar}", bar=type(y))
        return rng, y

    xs = jnp.linspace(0, jnp.pi, 100)
    final_rng, ys = jax.lax.scan(_loop, rng, xs)
    return ys


num_seeds = 2
global LOGGER
LOGGER = WandbMultiLogger(
    project="testing_momo",
    group="group_name_8",
    config={"var": 0},
    mode="online",
    num_seeds=num_seeds,
)
time.sleep(5)

rng = jax.random.PRNGKey(0)
rng_seeds = jax.random.split(rng, num_seeds)
train_jit = jax.jit(_train)
out = jax.vmap(train_jit)(rng_seeds, jnp.arange(num_seeds))
LOGGER.finish()
print("finished")
