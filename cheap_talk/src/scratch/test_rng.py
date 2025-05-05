import jax
import jax.numpy as jnp

rng, rng_debug = jax.random.split(rng, 2)
a = jax.random.normal(rng_debug)
jax.debug.print("0 {}", a)

rng = jax.random.PRNGKey(0)
rngs = jax.random.split(rng, 2)

print("default")
a = jax.random.normal(rngs[0])
print(a)

b = jax.random.normal(rngs[1])
print(b)

print("jit")
a = jax.jit(jax.random.normal)(rngs[0])
print(a)

b = jax.jit(jax.random.normal)(rngs[1])
print(b)


def _f(rng):
    a = jax.random.normal(rng)
    return a


print("jit a vmap")
fv = jax.jit(jax.vmap(_f))(rngs)
print(fv)

print("vmap a jit")
fjv = jax.vmap(jax.jit(_f))(rngs)
print(fjv)
