import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state
import numpy as np


# 1. Dummy Neural Network
class DummyModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(1)(x)  # Simple linear layer


# 2. Create dummy train state
def create_train_state(rng):
    model = DummyModel()
    params = model.init(rng, jnp.ones((1, 3)))  # Randomly initialize params
    tx = optax.adam(learning_rate=0.1)  # Adam optimizer
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


# 3. Create a dummy gradient
def dummy_gradient_like(params):
    return jax.tree_util.tree_map(lambda p: jnp.ones_like(p), params)


# ---- Run the test ----
rng = jax.random.PRNGKey(0)
state = create_train_state(rng)

# Create a fake gradient
grads = dummy_gradient_like(state.params)

# Update the state
new_state = state.apply_gradients(grads=grads)

# ---- Check immutability ----
print("state is new_state? ->", state is new_state)  # Expect False
print(
    "state.params is new_state.params? ->", state.params is new_state.params
)  # Expect False

# Optional: Show parameter update
flat_params = jax.tree_util.tree_flatten(state.params)
flat_new_params = jax.tree_util.tree_flatten(new_state.params)
print("\nOld param sample:", flat_params)
print("New param sample:", flat_new_params)

print(state.opt_state[0].count)
print(new_state.opt_state[0].count)
