import jax
import jax.numpy as jnp
from typing import NamedTuple


class Data(NamedTuple):
    x: int


a = Data(x=0)
b = Data(x=1)
c = jnp.stack([a, b])
print(c)
