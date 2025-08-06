from config import * 
from src.utils import logdet_matmul
from src import utils

def test_sublist_with_same_order():
    assert utils._sublist_with_same_order(("W",), ("W", "T", "B")) == True
    assert utils._sublist_with_same_order(("W", "T", "B"), ("W", "T", "B")) == True
    assert utils._sublist_with_same_order(("T",), ("W", "T", "B")) == True
    assert utils._sublist_with_same_order(("T", "W"), ("W", "T", "B")) == False

def test_vmap():
    arg_axes_name = {
        "xe": ("W", "T", "B"), 
        "xp": ("W",), 
        "twist": ("T",), 
    }
    out_axes_name = ("W", "T", "B")

    def f(xe, params, xp, twist):
        pass
    vmap_in_axes = utils.vmap(f, arg_axes_name, out_axes_name, return_in_axes=True)
    assert vmap_in_axes == (
        (0, None, 0, None), 
        (0, None, None, 0), 
        (0, None, None, None), 
    )

    def f(xp, xe, q, kappa, G, L):
        pass
    vmap_in_axes = utils.vmap(f, arg_axes_name, out_axes_name, return_in_axes=True)
    assert vmap_in_axes == (
        (0, 0, None, None, None, None), 
        (None, 0, None, None, None, None), 
        (None, 0, None, None, None, None), 
    )

def test_logdet_matmul():
    
    key = jax.random.PRNGKey(42)
    K = 8 
    N = 20 

    A = jax.random.normal(key, (K, N, N)) + 1J*jax.random.normal(key, (K, N, N))
    logw = jax.random.normal(key, (K, ))

    phase, logabsdet = logdet_matmul([A], logw)

    res = 0.0 
    for k in range(K):
        res += jnp.exp( logw[k]) * jnp.linalg.det(A[k])

    print (res, phase*jnp.exp(logabsdet))
    assert ( jnp.abs(res == phase * jnp.exp(logabsdet)) < 1e-10)
