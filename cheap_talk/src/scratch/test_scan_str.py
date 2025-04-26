import jax
from jax import lax
import jax.numpy as jnp


arr = jnp.array([0, 1, 2])


def outer_scan(carry, inc):
    def step_fn(carry, unuse):
        d = {"a": 1, "b": 2, "c": 3}
        k = list(d.keys())
        carry += d[k[inc]]
        # carry += inc
        return carry, None

    carry, _ = lax.scan(step_fn, carry, None, 10)
    return carry, _


do = {"a": 1, "b": 2, "c": 3}
init = 0
inc = jnp.arange(len(do.keys()))
final_carry, outputs = lax.scan(outer_scan, init, inc)

print("Final sum:", final_carry)  # sum of all values
print("Outputs:", outputs)  # a tuple of (keys, vals), stacked
