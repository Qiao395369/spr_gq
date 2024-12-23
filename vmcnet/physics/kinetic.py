"""Kinetic energy terms."""
from typing import Callable

import jax
# import fwdlap
# import jax.numpy as jnp
import vmcnet.physics as physics
from vmcnet.utils.typing import Array, P, ModelApply


def create_laplacian_kinetic_energy(
    log_psi_apply: Callable[[P, Array], Array],
) -> ModelApply[P]:
    """Create the local kinetic energy fn (params, x) -> -0.5 (nabla^2 psi(x) / psi(x)).

    Args:
        log_psi_apply (Callable): a function which computes log|psi(x)| for single
            inputs x. It is okay for it to produce batch outputs on batches of x as long
            as it produces a single number for single x. Has the signature
            (params, single_x_in) -> log|psi(single_x_in)|

    Returns:
        Callable: function which computes the local kinetic energy for continuous
        problems (as opposed to discrete/lattice problems), i.e. -0.5 nabla^2 psi / psi.
        Evaluates on only a single configuration so must be externally vmapped
        to be applied to a batch of walkers.
    """
    grad_log_psi_apply = jax.grad(log_psi_apply, argnums=2)

    def kinetic_energy_fn(params: P, ion_pos:Array, x: Array) -> Array:
        return -0.5 * physics.core.laplacian_psi_over_psi(grad_log_psi_apply, params, ion_pos,x)

    return kinetic_energy_fn

# def create_laplacian_kinetic_energy_gq(
#     log_psi_apply: Callable[[P, Array], Array],
# ):
#     kw = {"inner_size": 2}
#     grad_log_psi_apply = jax.grad(log_psi_apply, argnums=1)
#     logpsi_laplacian = make_laplacian_real(log_psi_apply,argnums=1, method="fwdlap", **kw)

#     def kinetic_energy_fn(params: P, x: Array) -> Array:
#         grad = grad_log_psi_apply(params, x)
#         laplacian = logpsi_laplacian(params, x)
#         kinetic = -0.5 * (laplacian + (grad**2).sum(axis=(-2, -1))) # (W, B)
#         return kinetic

#     return kinetic_energy_fn


# def make_laplacian_real(f, argnums=0, **kwargs):
#     """
#     Given a REAL-VALUED scalar function `f`, return the laplacian w.r.t.
#     a positional argument specified by the integer `argnums`.
#     """
#     if not isinstance(argnums, int):
#         raise ValueError("argnums should be an integer.")

#     def f_laplacian(*args):
#         x = args[argnums]
#         shape, size = x.shape, x.size
#         x_flatten = x.reshape(-1)
#         eye = jnp.eye(size, dtype=x.dtype)

#         inner_size = kwargs.get("inner_size", None)
#         f_argx = lambda x_flat: f(*[
#             x_flat.reshape(shape) if i == argnums else arg
#             for i, arg in enumerate(args)
#         ])
#         zero = fwdlap.Zero.from_value(x_flatten)
        
#         if inner_size is None:
#             _, _, laplacian = fwdlap.lap(f_argx, (x_flatten,), (eye,), (zero,))
#         else:
#             eye = eye.reshape(size // inner_size, inner_size, size)
#             _, f_lap_pe = fwdlap.lap_partial(f_argx, (x_flatten,), (eye[0],), (zero,))
            
#             def loop_fn(i, val):
#                 jac, lap = f_lap_pe((eye[i],), (zero,))
#                 return val + lap
            
#             laplacian = jax.lax.fori_loop(0, size // inner_size, loop_fn, 0.0)

#         return laplacian

#     return f_laplacian
