"""SPRING implementation, see https://doi.org/10.1016/j.jcp.2024.113351."""

from typing import Callable, Dict, Tuple, Any, TypeAlias, Callable, NamedTuple, Protocol
import jax
import jax.flatten_util
import jax.numpy as jnp
import neural_tangents as nt  # type: ignore
from ml_collections import ConfigDict
import chex
import optax
from functools import partial

from vmcnet.utils.typing import Array, D, ModelApply, P, S
from vmcnet.utils.pytree_helpers import (
    multiply_tree_by_scalar,
    tree_inner_product,
    tree_reduce_l1,
)
from vmcnet.utils.distribute import pmean_if_pmap
from vmcnet.utils.typing import UpdateDataFn, GetPositionFromData, LearningRateSchedule
from vmcnet.utils.distribute import PMAP_AXIS_NAME
from .update_param_fns import (
    UpdateParamFn,
    make_traced_fn_with_single_metrics,
    update_metrics_with_noclip,
)
from .optax_utils import initialize_optax_optimizer

OptimizerState: TypeAlias = Dict
WavefunctionParams: TypeAlias = Dict
class OptInitFunction(Protocol):
    """Protocol for optimizer initialization functions."""

    def __call__(
        self,
        params: P,
    ) -> OptimizerState:
        ...


class OptStepFunction(Protocol):
    """Protocol for optimizer step functions."""

    def __call__(
        self,
        params: P,
        data: D,
        opt_state: OptimizerState,
    ) -> Tuple[P,D, OptimizerState, Dict[str, Any]]:
        ...

class Optimizer(NamedTuple):
    """Optimizer interface with init and step function."""

    init: OptInitFunction
    step: OptStepFunction

def spring_wrapper(spring_opt, log_psi_apply, update_data_fn, energy_and_statistics_fn) -> Optimizer:
    """Wrap the spring optimizer to make it compatible with the optimizer interface."""

    @partial(jax.pmap, axis_name=PMAP_AXIS_NAME)
    def init(
        params: P,
    ) -> OptimizerState:
        return spring_opt.init(params)

    @partial(jax.vmap, in_axes=(None, 0, 0))
    @partial(jax.vmap, in_axes=(None, None, 0))
    def raveled_log_psi_grad(params: P,ion_pos: Array, positions: Array) -> Array:
        log_grads = jax.grad(log_psi_apply)(params,ion_pos, positions)
        return jax.flatten_util.ravel_pytree(log_grads)[0]
    

    @partial(jax.pmap, axis_name=PMAP_AXIS_NAME)
    def step(
        key,
        params: P,
        opt_state: OptimizerState,
        data:D ,
    ) -> tuple[P,D, OptimizerState, Dict]:
        log_psi_grads = raveled_log_psi_grad(params, data["atoms_position"], data["walker_data"]["elec_position"])
        energy_per_w, E_loc, stats = energy_and_statistics_fn(params, data["atoms_position"], data["walker_data"]["elec_position"])
        updates, E_mean, opt_state = spring_opt.update(log_psi_grads, E_loc, energy_per_w, opt_state)
        gradient = opt_state["prev_grad"]
        param_norm, update_norm, grad_norm = map(tree_norm, [params, updates, gradient])
        params = apply_updates(params, updates)
        data = update_data_fn(data, params)

        metrics = {
                    "energy": E_mean, "variance": stats["variance"],
                    "kinetic": stats["kinetic"],
                    "ei_potential": stats["ei_potential"],
                    "ee_potential": stats["ee_potential"],
                    "ii_potential": stats["ii_potential"],
                    "multi_energy": stats["multi_energy"],
                    "opt_param_norm": param_norm,
                    "opt_grad_norm": grad_norm,
                    "opt_update_norm": update_norm,
                    "energy_noclip": stats["energy_noclip"],
                    "variance_noclip": stats["variance_noclip"],
            }
        # metrics = jax.tree_map(lambda x: jnp.asarray(x, dtype=jnp.float32), metrics)
        return params,data, opt_state, metrics, key

    return Optimizer(init=init, step=step)

class Spring:
    """Implements the SPRING optimizer from arXiv:2401.10190."""

    def __init__(
        self,
        mu: float,
        norm_constraint: float,
        learning_rate_schedule: Callable[[int], float],
        damping_schedule: Callable[[int], float],
        repeat_single_mol: bool = False,
    ):
        self.mu = mu
        self.norm_constraint = norm_constraint
        self.lr_schedule = learning_rate_schedule
        self.dp_schedule = damping_schedule
        self.repeat_single_mol = repeat_single_mol

    def init(self, params: P):
        opt_state = {
            "prev_grad": jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), params),
            "step": jnp.asarray(0, dtype=jnp.int32),
        }
        return opt_state
    
    def get_grad(
        self,
        log_psi_grads,
        E_loc: P,
        E_mean_per_mol,
        opt_state,
    ) -> Tuple[Array, P]:
        walker_batch_this_process, electron_batch_size = E_loc.shape
        prev_grad, unravel_fn = jax.flatten_util.ravel_pytree(opt_state["prev_grad"])
        
        Ohat = (log_psi_grads - jnp.mean(log_psi_grads, axis=-2, keepdims=True)) / jnp.sqrt(electron_batch_size)  #(W,B,np)-(W,1,np)/sqrt(B)->(W,B,np)
        T = jnp.einsum("mjk, mlk  -> mjl", Ohat, Ohat)  #(W,B,np),(W,B,np)->(W,B,B)
        ones = jnp.ones_like(T) / electron_batch_size
        T_reg = T + ones + self.dp_schedule(opt_state["step"]) * jnp.eye(electron_batch_size)[None,:,:] #(W,B,B)
        # E_mean_per_mol = jnp.mean(E_loc, axis=-1, keepdims=True)  #(W,B)->(W,1)
        E_mean = jnp.mean(E_mean_per_mol, keepdims=True)  #(1,1)
        E_mean = jax.lax.pmean(E_mean, axis_name=PMAP_AXIS_NAME)  #(1,1)
        if self.repeat_single_mol:
            E_mean_per_mol = E_mean
        epsilon_bar = (E_loc - E_mean_per_mol) / jnp.sqrt(electron_batch_size)
        epsilon_tilde = epsilon_bar - jnp.einsum("mjk, k -> mj", Ohat, self.mu * prev_grad)  #(W,B,np),(np)->(W,B)
        epsilon_projected = jax.scipy.linalg.solve(T_reg, epsilon_tilde[..., None])[..., 0]  #(W,B,B)(W,B,1)-->(W,B,1)-->(W,B)
        dtheta_residual = jax.lax.pmean(jnp.einsum("mjk,mj->k", Ohat, epsilon_projected) ,axis_name=PMAP_AXIS_NAME)/walker_batch_this_process
        grad = dtheta_residual + self.mu * prev_grad
        scaled_grad = self.apply_norm_constraint(grad)
        return unravel_fn(grad), unravel_fn(scaled_grad), jnp.squeeze(E_mean)

    def apply_norm_constraint(self, grad: WavefunctionParams) -> WavefunctionParams:
        """Scales update to have L2 norm <= norm_constraint."""
        sq_norm_grads = jnp.sum(grad * grad)
        eps=1e-12
        coefficient = jnp.minimum(1, jnp.sqrt(self.norm_constraint / (sq_norm_grads + eps)))
        return grad * coefficient
\
    def update(
        self, grad_psi, E_loc, energy_per_w, opt_state: OptimizerState
    ) -> tuple[WavefunctionParams, OptimizerState]:

        grad, scaled_grad, E_mean = self.get_grad(grad_psi, E_loc, energy_per_w, opt_state)
        update = jax.tree_util.tree_map(lambda x: -self.lr_schedule(opt_state["step"]) * x, scaled_grad)

        return update, E_mean, {
            "prev_grad": grad,
            "step": opt_state["step"] + 1,  # How is the step handled with other optimizers?
        }


def tree_norm(x, sq=False):
    sq_norm = jax.tree_util.tree_reduce(lambda c, x: c + jnp.sum(x**2), x, jnp.zeros(()))
    return sq_norm if sq else jnp.sqrt(sq_norm)

def apply_updates(params: WavefunctionParams, updates: WavefunctionParams) -> WavefunctionParams:
    """Apply updates to wave function parameters."""

    return jax.tree_util.tree_map(
        lambda p, u: jnp.asarray(p + u).astype(jnp.asarray(p).dtype), params, updates
    )
