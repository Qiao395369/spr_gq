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
    repeat_single_mol: bool,
    pmap_axis_name: str,
    ansatz_call_fn=regular_ansatz_call,
):
    @jax.custom_jvp
    def loss(
        params,
        batch,
    ):
        local_energies, energy_per_w, data = batch
        log_psi = ansatz_call_fn(ansatz, params, data["atoms_position"], data["walker_data"]["elec_position"])
        # register log density for kfac
        kfac_jax.register_normal_predictive_distribution(log_psi[:, None])
        return jnp.nanmean(local_energies)

    @loss.defjvp
    def loss_jvp(primals, tangents):
        params, (local_energies, energy_per_w, data) = primals
        dparams, _ = tangents
        _, log_psi_tangent = ansatz_jvp(ansatz, params, dparams, data["atoms_position"], data["walker_data"]["elec_position"])

        log_psi = ansatz_call_fn(ansatz, params, data["atoms_position"], data["walker_data"]["elec_position"])
        # register log density for kfac
        kfac_jax.register_normal_predictive_distribution(log_psi[:, None])

        if repeat_single_mol:
            energy_per_w = jnp.mean(energy_per_w, axis=0, keepdims=True)
            energy_per_w = jax.lax.pmean(energy_per_w, axis_name=pmap_axis_name)

        # [mol_batch, elec_batch]
        centered_energy = local_energies - energy_per_w

        loss_tangent = jnp.nanmean(centered_energy * log_psi_tangent)
        loss = jnp.nanmean(local_energies)

        return loss, loss_tangent

    return loss

def make_value_and_grad(
    ansatz,
    repeat_single_mol: bool,
    pmap_axis_name: str,
    ansatz_call_fn,
    apply_pmap: bool,
):
    def value_and_grad(params, batch):
        local_energies, energy_per_w, data = batch
        atoms = data["atoms_position"]
        elec  = data["walker_data"]["elec_position"]

        # 前向：log_psi，并注册给 KFAC
        def f(p):
            y = ansatz_call_fn(ansatz, p, atoms, elec)   # shape [W*B]
            kfac_jax.register_normal_predictive_distribution(y[:, None])
            return y

        log_psi, vjp_fun = jax.vjp(f, params)           # 得到 VJP 闭包

        if repeat_single_mol:
            energy_per_w = jnp.mean(energy_per_w, axis=0, keepdims=True)
            if apply_pmap:
                energy_per_w = jax.lax.pmean(energy_per_w, axis_name=pmap_axis_name)

        centered = local_energies - energy_per_w         # (W,B)

        weights = centered.reshape(-1)
        is_finite = jnp.isfinite(weights)
        n_local = jnp.sum(is_finite)
        n = jnp.maximum(1, n_local)
        weights = jnp.where(is_finite, weights / n, 0.0).astype(log_psi.dtype)

        grad_params = vjp_fun(weights)[0]                # PyTree，与 params 同结构
        loss_val = jnp.nanmean(local_energies)           # 标量

        return loss_val, grad_params

    return value_and_grad
