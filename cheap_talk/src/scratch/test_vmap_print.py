import jax
import jax.numpy as jnp


def printer(rng, run_idx):
    s = str(run_idx)
    jax.debug.print("hello {bar}", bar=run_idx.astype(int))
    return run_idx


num_seeds = 10
rng = jax.random.PRNGKey(0)
rng_seeds = jax.random.split(rng, num_seeds)
jit_printer = jax.jit(printer)
out = jax.vmap(jit_printer)(rng_seeds, jnp.arange(num_seeds))
print(out)
