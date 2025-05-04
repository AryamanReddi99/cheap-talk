import jax
import jax.numpy as jnp

xs = jnp.array([1, 2, 3])


def overall():
    def _f(x, idx):
        return x, x[idx] * 2

    _, y = jax.lax.scan(_f, xs, jnp.arange(len(xs)))

    return y


overall_jit = jax.jit(overall)
y = overall_jit()
print(y)
