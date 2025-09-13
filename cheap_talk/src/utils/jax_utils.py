import jax
import jax.numpy as jnp

jprint = lambda *args: [jax.debug.print("{var}", var=arg) for arg in args]


def pytree_norm(pytree):
    """
    Computes the L2 norm of a pytree
    """
    squares = jax.tree_util.tree_map(lambda x: jnp.sum(x**2), pytree)
    total_square = jax.tree.reduce(lambda leaf_1, leaf_2: leaf_1 + leaf_2, squares)
    return jnp.sqrt(total_square)


def pytree_diff_norm(pytree1, pytree2):
    """
    Computes the L2 norm of the difference between two pytrees
    """
    square_diff = jax.tree_util.tree_map(
        lambda x, y: jnp.sum((x - y) ** 2), pytree1, pytree2
    )
    total_square_diff = jax.tree_util.tree_reduce(
        lambda leaf_1, leaf_2: leaf_1 + leaf_2, square_diff
    )
    return jnp.sqrt(total_square_diff)
