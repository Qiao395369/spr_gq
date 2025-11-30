"""SPRING implementation, see https://doi.org/10.1016/j.jcp.2024.113351."""

from typing import Callable, Dict
import jax
import jax.flatten_util
import jax.numpy as jnp
from ml_collections import ConfigDict
import chex
import optax

from vmcnet.utils.typing import Array, D, ModelApply, P, S, Tuple
from vmcnet.utils.pytree_helpers import (
    multiply_tree_by_scalar,
    tree_inner_product,
    tree_reduce_l1,
)
from vmcnet.utils.distribute import pmean_if_pmap
from vmcnet.utils.typing import UpdateDataFn, GetPositionFromData, LearningRateSchedule

from .update_param_fns import (
    UpdateParamFn,
    make_traced_fn_with_single_metrics,
    update_metrics_with_noclip,
)
from .optax_utils import initialize_optax_optimizer
import psutil
import logging

def construct_spring_update_param_fn(
    energy_and_statistics_fn,
    optimizer_apply: Callable[[P, P, S, D, Dict[str, Array]], Tuple[P, S]],
    get_position_fn: GetPositionFromData[D],
    update_data_fn: UpdateDataFn[D, P],
    apply_pmap: bool = True,
    record_param_l1_norm: bool = False,
) -> UpdateParamFn[P, D, S]:
    """Create the `update_param_fn` based on the gradient of the total energy."""

    def update_param_fn(key, params, optimizer_state, data):
        position = get_position_fn(data)
        atoms_position = data["atoms_position"]
        energy_per_w, E_loc, stats = energy_and_statistics_fn(params, atoms_position, position)
        params, optimizer_state = optimizer_apply(
            energy_per_w,
            E_loc,
            params,
            optimizer_state,
            data,
        )
        energy = jnp.mean(energy_per_w)
        data = update_data_fn(data, params)
        metrics = {"energy": energy, "variance": stats["variance"],
                   "multi_energy":stats["multi_energy"],
                   }
        metrics = update_metrics_with_noclip(
            stats["energy_noclip"],
            stats["variance_noclip"],
            metrics,
        )
        if record_param_l1_norm:
            metrics.update({"param_l1_norm": tree_reduce_l1(params)})
        return params, data, optimizer_state, metrics, key

    traced_fn = make_traced_fn_with_single_metrics(update_param_fn, apply_pmap)

    return traced_fn


def initialize_spring(
    log_psi_apply: ModelApply[P],
    energy_and_statistics_fn,
    params: P,
    get_position_fn: GetPositionFromData[D],
    update_data_fn: UpdateDataFn[D, P],
    learning_rate_schedule: LearningRateSchedule,
    optimizer_config: ConfigDict,
    record_param_l1_norm: bool = False,
    apply_pmap: bool = True,
) -> Tuple[UpdateParamFn[P, D, optax.OptState], optax.OptState]:
    """Get an update param function and initial state for SPRING."""
    spring_step = get_spring_step(
        log_psi_apply,
        optimizer_config.damping,
        optimizer_config.mu,
    )

    descent_optimizer = optax.sgd(
        learning_rate=learning_rate_schedule, momentum=0, nesterov=False
    )

    def prev_update(optimizer_state):
        return optimizer_state[0].trace

    def optimizer_apply(energy, local_energies, params, optimizer_state, data):
        positions = get_position_fn(data)

        centered_local_energies = local_energies - energy
        grad = spring_step(
            centered_local_energies,
            params,
            prev_update(optimizer_state),
            data["atoms_position"],
            positions,
        )
        updates, optimizer_state = descent_optimizer.update(
            grad, optimizer_state, params
        )
        if optimizer_config.constrain_norm:
            updates = constrain_norm(
                updates,
                optimizer_config.norm_constraint,
            )
        params = optax.apply_updates(params, updates)
        return params, optimizer_state

    update_param_fn = construct_spring_update_param_fn(
        energy_and_statistics_fn,
        optimizer_apply,
        get_position_fn=get_position_fn,
        update_data_fn=update_data_fn,
        record_param_l1_norm=record_param_l1_norm,
        apply_pmap=apply_pmap,
    )
    optimizer_state = initialize_optax_optimizer(
        descent_optimizer, params, apply_pmap=apply_pmap
    )

    return update_param_fn, optimizer_state


def get_spring_step(
    log_psi_apply: ModelApply[P],
    damping: chex.Scalar = 0.001,
    mu: chex.Scalar = 0.99,
):

    def raveled_log_psi_grad(params: P,ion_pos: Array, positions: Array) -> Array:
        log_grads = jax.grad(log_psi_apply)(params,ion_pos, positions)
        return jax.flatten_util.ravel_pytree(log_grads)[0]

    batch_raveled_log_psi_grad = jax.vmap(jax.vmap(raveled_log_psi_grad, in_axes=(None, None,0)),in_axes=(None,0,0))

    def spring_update_fn(
        centered_energies: P,
        params: P,
        prev_grad,
        atoms_positions: Array,
        positions: Array,
    ) -> Tuple[Array, P]:
        nchains = positions.shape[1]*positions.shape[0]
        prev_grad, unravel_fn = jax.flatten_util.ravel_pytree(prev_grad)
        prev_grad_decayed = mu * prev_grad  #(nparams,)
        log_psi_grads_pre = batch_raveled_log_psi_grad(params,atoms_positions, positions) 
        W,B,nparams=log_psi_grads_pre.shape
        log_psi_grads=log_psi_grads_pre.reshape((W*B,nparams)) /jnp.sqrt(nchains)  #(W*B,nparams)
        Ohat = log_psi_grads - jnp.mean(log_psi_grads, axis=0, keepdims=True)  #(W*B,nparams)
        T = Ohat @ Ohat.T  #(W*B,W*B)
        ones = jnp.ones((nchains, 1)) #(W*B,1)
        T_reg = T + ones @ ones.T / nchains + damping * jnp.eye(nchains)  #(W*B,W*B)
        epsilon_bar = centered_energies.reshape((-1,)) / jnp.sqrt(nchains) #(W*B,)
        epsion_tilde = epsilon_bar - Ohat @ prev_grad_decayed   #(W*B,)
        dtheta_residual = Ohat.T @ jax.scipy.linalg.solve(T_reg, epsion_tilde, assume_a="pos") #(nparams,)
        # print(f"dtheta_residual:{dtheta_residual.shape}")   #(nparams,)
        # print(f"prev_grad_decayed:{prev_grad_decayed.shape}")   #(nparams,)
        SR_G = dtheta_residual + prev_grad_decayed

        return unravel_fn(SR_G)

    return spring_update_fn

def constrain_norm(
    grad: P,
    norm_constraint: chex.Numeric = 0.001,
) -> P:
    """Euclidean norm constraint."""
    sq_norm_scaled_grads = tree_inner_product(grad, grad)

    # Sync the norms here, see:
    # https://github.com/deepmind/deepmind-research/blob/30799687edb1abca4953aec507be87ebe63e432d/kfac_ferminet_alpha/optimizer.py#L585
    sq_norm_scaled_grads = pmean_if_pmap(sq_norm_scaled_grads)

    norm_scale_factor = jnp.sqrt(norm_constraint / sq_norm_scaled_grads)
    coefficient = jnp.minimum(norm_scale_factor, 1)
    constrained_grads = multiply_tree_by_scalar(grad, coefficient)

    return constrained_grads