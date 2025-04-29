from flax import struct
import jax.numpy as jnp
from flax.training.train_state import TrainState


@struct.dataclass
class Test:
    a: jnp.ndarray
    b: jnp.ndarray


t = Test(a=jnp.array([1, 2, 3]))
b = t.a
t = t.replace(a=b * 2)

print(t.a)
print(b)
