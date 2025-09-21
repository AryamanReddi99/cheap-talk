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


def pytree_cosine_similarity(pytree1, pytree2):
    """
    Computes the cosine similarity between two pytrees.

    Cosine similarity is defined as: cos(θ) = (A · B) / (||A|| * ||B||)

    Args:
        pytree1: First pytree
        pytree2: Second pytree (must have same structure as pytree1)

    Returns:
        Cosine similarity value between -1 and 1
    """
    # Compute dot product between the two pytrees
    dot_products = jax.tree_util.tree_map(lambda x, y: jnp.sum(x * y), pytree1, pytree2)
    total_dot_product = jax.tree_util.tree_reduce(
        lambda leaf_1, leaf_2: leaf_1 + leaf_2, dot_products
    )

    # Compute norms of both pytrees
    norm1 = pytree_norm(pytree1)
    norm2 = pytree_norm(pytree2)

    # Avoid division by zero
    norm_product = norm1 * norm2
    return jnp.where(norm_product == 0, 0.0, total_dot_product / norm_product)
