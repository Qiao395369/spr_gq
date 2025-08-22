"""Kinetic energy terms."""

from typing import Callable
import jax.numpy as jnp

from vmcnet.utils.typing import Array, P, ModelApply
from vmcnet.physics.fwdlap import zero_tangent_from_primal, lap  # type: ignore


def create_laplacian_kinetic_energy_new(
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

    def kinetic_energy_fn(params: P, atoms_positions:Array,x: Array) -> Array:
        """Compute -1/2 * (nabla^2 psi) / psi at x given a function which evaluates psi'(x)/psi.

        This function uses the identity

            (nabla^2 psi) / psi = nabla^2 log|psi|) + || nabla log|psi| ||^2

        to avoid leaving the log domain during the computation.

        This function should be vmapped in order to be applied to batches of inputs, as it
        completely flattens x in order to take second derivatives w.r.t. each component.
        """
        x_shape = x.shape
        flat_x = jnp.reshape(x, (-1,))
        n = flat_x.shape[0]
        eye = jnp.eye(n)

        def flattened_log_psi(flat_x_in):
            """Flattened input to flattened output version of log_psi."""
            return log_psi_apply(params,atoms_positions, jnp.reshape(flat_x_in, x_shape))

        zero = zero_tangent_from_primal(flat_x)

        _, grads, laps = lap(flattened_log_psi, (flat_x,), (eye,), (zero,))
        laplacian_psi_over_psi = jnp.sum(grads**2) + laps

        return -0.5 * laplacian_psi_over_psi

    return kinetic_energy_fn

import jax
def create_laplacian_kinetic_energy_old(
    log_psi_apply: Callable[[P, Array], Array],
) -> ModelApply[P]:
    grad_log_psi_apply = jax.grad(log_psi_apply, argnums=2)

    def kinetic_energy_fn(params: P, atoms_positions:Array, positions: Array) -> Array:
        return -0.5 * laplacian_psi_over_psi(grad_log_psi_apply, params,atoms_positions, positions)

    return kinetic_energy_fn

from typing import Callable, Optional
def laplacian_psi_over_psi(
    grad_log_psi_apply: ModelApply,
    params: P,
    ion_pos: Array,
    x: Array,
    nparticles: Optional[int] = None,
    particle_perm: Optional[Array] = None,
) -> Array:

    x_shape = x.shape
    flat_x = jnp.reshape(x, (-1,))
    n = flat_x.shape[0]
    identity_mat = jnp.eye(n)

    def flattened_grad_log_psi_of_flat_x(flat_x_in):
        """Flattened input to flattened output version of grad_log_psi."""
        grad_log_psi_out = grad_log_psi_apply(params,ion_pos, jnp.reshape(flat_x_in, x_shape))
        return jnp.reshape(grad_log_psi_out, (-1,))

    length = n
    multiplier = 1.0
    vecs = identity_mat

    if nparticles is not None and particle_perm is not None:
        length = 3 * nparticles
        multiplier = n / (3 * nparticles)

        d = x_shape[-1]
        triple_perm = jnp.stack([particle_perm * d + i for i in range(d)]).T.reshape(
            (n,)
        )
        vecs = identity_mat[triple_perm, :]

    def step_fn(carry, unused):
        del unused
        i = carry[0]
        primals, tangents = jax.jvp(
            flattened_grad_log_psi_of_flat_x, (flat_x,), (vecs[i],)
        )
        return (
            i + 1,
            carry[1]
            + jnp.square(jnp.dot(primals, vecs[i]))
            + jnp.dot(tangents, vecs[i]),
        ), None

    out, _ = jax.lax.scan(step_fn, (0, jnp.array(0.0)), xs=None, length=length)
    return out[1] * multiplier
