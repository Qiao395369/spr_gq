import typing
from functools import partial
from typing import Protocol

import jax
import jax.numpy as jnp
import kfac_jax



# __all__ = ["ClipMaskFunction", "make_local_energy_fn", "make_loss"]




def flat_ansatz_call(
    ansatz, params, atoms_position, elec_position
) -> jax.Array:
    """Call the function with batch dimensions flattened."""
    n_elec_batch = elec_position.shape[1]
    elecs_flat = elec_position.reshape(-1, *elec_position.shape[2:])
    atoms_flat = jnp.repeat(atoms_position[:, None, ...], repeats=n_elec_batch, axis=1).reshape(-1, *atoms_position.shape[1:])


    return jax.vmap(ansatz, (None, 0, 0))(params, atoms_flat, elecs_flat)


def regular_ansatz_call(
    ansatz, params, atoms_position, elec_position
) -> jax.Array:
    """Call the function in the normal way with double vmap."""
    return jax.vmap(jax.vmap(ansatz, (None, None, 0)), (None, 0, 0))(params, atoms_position, elec_position)


def ansatz_jvp(
    ansatz,
    params,
    dparams,
    atoms_position,
    elec_position,
) -> tuple[jax.Array, jax.Array]:
    """Evaluate the wave function and gradient for a batch of electron configurations.

    Expects a molecule batch dimension and an electron batch dimension.

    Parameters:
    ------------
    params: Parameters of the wave function
    dparams: Derivative of the parameters of the wave function
    elecs: electron configurations [mol_batch, mol_electrons, n_elec, 3]
    inputs: dictionary containing MolecularConfigurations [mol_batch, n_nuc, 3],
        and potentially other WF inputs
    """

    def _call(params):
        return jax.vmap(jax.vmap(ansatz, (None, None, 0)), (None, 0, 0))(params, atoms_position, elec_position)

    return typing.cast(tuple[jax.Array, jax.Array], jax.jvp(_call, (params,), (dparams,)))


def determinant_jvp(
    ansatz,
    params,
    dparams,
    atoms_position,
    elec_position,
) -> tuple[jax.Array, jax.Array]:
    """Evaluate the relative contribution of each determinant and its gradient.

    Expects a molecule batch dimension and an electron batch dimension.

    Parameters:
    ------------
    params: Parameters of the wave function
    dparams: Derivative of the parameters of the wave function
    elecs: electron configurations [mol_batch, mol_electrons, n_elec, 3]
    inputs: dictionary containing MolecularConfigurations [mol_batch, n_nuc, 3],
        and potentially other WF inputs
    """

    def _call(params):
        det_dist = jax.vmap(
            jax.vmap(partial(ansatz, return_det_dist=True), (None, None, 0)), (None, 0, 0)
        )(params, atoms_position, elec_position)[1]
        return det_dist.mean(-1)

    return typing.cast(tuple[jax.Array, jax.Array], jax.jvp(_call, (params,), (dparams,)))




def make_loss(
    ansatz,
    energy_and_statistics_fn,
    repeat_single_mol: bool,
    pmap_axis_name: str,
    ansatz_call_fn=regular_ansatz_call,
    det_dist_weight: float = 1.0,
):
    def energy_loss_tangent_w_penalty(
        log_psi_tangent: jax.Array,
        # det_dist_tangent: jax.Array,
        local_energies: jax.Array,
        energy_per_w: jax.Array,
    ) -> jax.Array:
        if repeat_single_mol:
            energy_per_w = jnp.mean(energy_per_w, axis=0, keepdims=True)
            energy_per_w = jax.lax.pmean(energy_per_w, axis_name=pmap_axis_name)

        # [mol_batch, elec_batch]
        centered_energy = local_energies - energy_per_w

        loss_tangent = jnp.nanmean(centered_energy * log_psi_tangent)

        # Weight the det penalty relative to the local energy stddev, to give correct magnitude
        # TODO: fix for non-uniform weights
        # pre_conditioner_det = jnp.sqrt( (centered_energy ** 2).sum(-1)/ (local_energies.shape[-1] - 1) )

        # Now make mean over molecules
        # loss_tangent -= det_dist_weight * (pre_conditioner_det * det_dist_tangent.mean(-1)).mean()

        return loss_tangent

    @jax.custom_jvp
    def loss(
        params,
        rng,  # Accept rng as input
        data,
    ):
        del rng
        energy_per_w, E_loc, stats = energy_and_statistics_fn(params, data["atoms_position"], data["walker_data"]["elec_position"])
        return jnp.nanmean(E_loc)[0], stats

    @loss.defjvp
    def loss_jvp(primals, tangents):
        params, rng, data = primals
        atoms_position, elec_position = data["atoms_position"], data["walker_data"]["elec_position"]
        energy_per_w, local_energies, stats = energy_and_statistics_fn(params, atoms_position, elec_position)

        dparams, _ , _= tangents
        _, log_psi_tangent = ansatz_jvp(ansatz, params, dparams, atoms_position, elec_position)
        # _, det_dist_tangent = determinant_jvp(ansatz, params, dparams, atoms_position, elec_position)

        log_psi = ansatz_call_fn(ansatz, params, atoms_position, elec_position)
        # register log density for kfac
        kfac_jax.register_normal_predictive_distribution(log_psi[:, None])

        loss_tangent = energy_loss_tangent_w_penalty(
            log_psi_tangent, 
            # det_dist_tangent, 
            local_energies, 
            energy_per_w
        )
        loss = (jnp.nanmean(local_energies), stats)
        stats_tangent_zero = jax.tree_map(lambda x: jnp.zeros_like(x), stats)

        return loss, (loss_tangent, stats_tangent_zero)

    return loss
