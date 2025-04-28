import jax
import jax.numpy as jnp


rng = jax.random.PRNGKey(0)
rng_2 = jax.random.PRNGKey(1)

rng_old = rng

rngs = jax.random.split(rng, 2)

print(rng == rng_old)


def f(rng, x):
    return jax.random.uniform(rng)


f_jit = jax.jit(f)
print(f_jit(rng, 0))

f_jit_vmap = jax.vmap(f_jit, in_axes=(None, 0))
print(f_jit_vmap(rng, jnp.array([0, 1])))
