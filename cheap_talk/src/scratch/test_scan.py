import jax
import jax.numpy as jnp
from typing import NamedTuple


class Data(NamedTuple):
    x: jnp.ndarray


var = 0
var2 = 1
increment = jnp.ones(100)


def scan_fn(x, inc):
    var, var2 = x
    var += inc
    var2 += inc
    return (var, var2), (var, var2)


final_vars, incremented_vars = jax.lax.scan(scan_fn, (var, var2), increment)

print(final_vars, incremented_vars)
print(type(final_vars), type(incremented_vars))
