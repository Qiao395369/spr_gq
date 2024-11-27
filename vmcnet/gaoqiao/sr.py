"""
    Second-order optimization algorithm using stochastic reconfiguration.
    The design of API signatures is in parallel with the package `optax`.
"""
import jax
from jax.flatten_util import ravel_pytree

# Make arbitrary function `f` callable with pytree as arguments.
tree_fn = lambda f: lambda *args: jax.tree_util.tree_map(f, *args)

block_ravel_pytree = lambda block_fn: lambda pytree: jax.tree_util.tree_map(
    lambda x: ravel_pytree(x)[0], pytree, is_leaf=block_fn
)
block_unravel_pytree = lambda block_fn: lambda pytree: jax.tree_util.tree_map(
    lambda x: ravel_pytree(x)[1], pytree, is_leaf=block_fn
)

