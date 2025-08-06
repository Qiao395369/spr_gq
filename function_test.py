from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar, Union
import jax.numpy as jnp
import jax
import functools
PyTree = Any

def is_tuple_of_arrays(x: PyTree) -> bool:
    """Returns True if x is a tuple of Array objects."""
    return isinstance(x, tuple) and all(isinstance(x_i, jnp.ndarray) for x_i in x)


# 定义矩阵
m1 = jnp.array(
        [
            [
                [1, 2],
                [3, 4],
            ],  # det = -2
            [
                [1, 2],
                [4, 3],
            ],  # det = -5
        ]
    )
m2 = jnp.array(
        [
            [
                [1, 1, 1],
                [2, 2, 2],
                [3, 3, 3],
            ],  # det = 0
            [
                [1, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
            ],  # det = -1
        ]
    )

matrix_pytree = {0: m1, 1: (m2, m1)}
slogdets = jax.tree_map(jnp.linalg.slogdet, matrix_pytree)  #matrix_pytree中的每一个(...,n,n)数组被一个二元元祖替代，元祖的每个元素是(...,)维的数组
print("slogdets:",slogdets)   #{0:(sign1,logdet1),1:((sign2,logdet2),(sign1,logdet1))}
slogdet_leaves, _ = jax.tree_util.tree_flatten(slogdets, is_tuple_of_arrays)
print("slogdet_leaves:",slogdet_leaves)   #[(sign1,logdet1),(sign2,logdet2),(sign1,logdet1)]
sign_prod, log_prod = functools.reduce(
        lambda a, b: (a[0] * b[0], a[1] + b[1]), slogdet_leaves
    )
print("sign_prod:",sign_prod)   #sign1*sign2*sign1
print("log_prod:",log_prod)   #logdet1+logdet2+logdet1



