import jax
from jax import lax
import jax.numpy as jnp


def outer_scan(carry, inc):
    def step_fn(carry, unuse):
        arr = jnp.ones(5)
        carry += arr[inc.astype(int)]
        return carry, None

    carry, _ = lax.scan(step_fn, carry, None, 10)
    return carry, _


init = 0
inc = jnp.ones(5)
final_carry, outputs = lax.scan(outer_scan, init, inc)

print("Final sum:", final_carry)  # sum of all values
print("Outputs:", outputs)  # a tuple of (keys, vals), stacked
