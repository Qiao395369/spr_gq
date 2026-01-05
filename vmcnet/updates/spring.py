"""SPRING implementation, see https://doi.org/10.1016/j.jcp.2024.113351."""

from typing import Callable, Dict, Tuple, Any, TypeAlias, Callable, NamedTuple, Protocol
import jax
import jax.flatten_util
import jax.numpy as jnp
from ml_collections import ConfigDict
import chex
import optax
from functools import partial
import logging
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

    def init(
        params: P,
    ) -> OptimizerState:
        return spring_opt.init(params)

    @partial(jax.vmap, in_axes=(None, 0, 0))
    @partial(jax.vmap, in_axes=(None, None, 0))
    def raveled_log_psi_grad(params: P,ion_pos: Array, positions: Array) -> Array:
        log_grads = jax.grad(log_psi_apply)(params,ion_pos, positions)
        return jax.flatten_util.ravel_pytree(log_grads)[0]

    def step(
        key,
        params: P,
        opt_state: OptimizerState,
        data:D ,
    ) -> tuple[P,D, OptimizerState, Dict]:
        position = data["walker_data"]["elec_position"]
        atoms_position = data["atoms_position"]
        log_psi_grads = raveled_log_psi_grad(params, atoms_position, position)
        energy_per_w, E_loc, stats = energy_and_statistics_fn(params, atoms_position, position)
        updates, E_mean, opt_state = spring_opt.update(log_psi_grads, E_loc, energy_per_w, opt_state)
        gradient = opt_state["prev_grad"]
        param_norm, update_norm, grad_norm = map(tree_norm, [params, updates, gradient])
        params = apply_updates(params, updates)
        data = update_data_fn(data, params)

        metrics = {
                    "energy": E_mean, "variance": stats["variance"],
                    "variance_noclip": stats["variance_noclip"],
                    "multi_variance": stats["multi_variance"],
                    "energy_noclip": stats["energy_noclip"],
                    "multi_energy": stats["multi_energy"],
                    "opt_param_norm": param_norm,
                    "opt_grad_norm": grad_norm,
                    "opt_update_norm": update_norm,

            }
        return params,data, opt_state, metrics, key

    return Optimizer(init=init, step=step)

class Spring:

    def __init__(
        self,
        spr_type: str,
        mu: float,
        norm_constraint: float,
        learning_rate_schedule: Callable[[int], float],
        damping_schedule: Callable[[int], float],
        repeat_single_mol: bool = False,
    ):
        self.spr_type = spr_type
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
    
    def get_grad_1(
        self,
        log_psi_grads,
        E_loc: P,
        E_mean_per_mol,
        opt_state,
    ) -> Tuple[Array, P]:
        prev_grad, unravel_fn = jax.flatten_util.ravel_pytree(opt_state["prev_grad"])
        prev_grad_decayed = self.mu * prev_grad  #(nparams,)
        walker_batch_this_process, electron_batch_size = E_loc.shape
        Ohat = (log_psi_grads - jnp.mean(log_psi_grads, axis=-2, keepdims=True)) / jnp.sqrt(electron_batch_size)  #(W,B,np)-(W,1,np)/sqrt(B)->(W,B,np)
        T = jnp.einsum("mjk,  mlk -> mjl", Ohat, Ohat)  #(W,B,np),(W,B,np)->(W,B,B)
        ones = jnp.ones_like(T) / electron_batch_size
        T_reg = T + ones + self.dp_schedule(opt_state["step"]) * jnp.eye(electron_batch_size) #(W,B,B)
        E_mean = jnp.mean(E_mean_per_mol, keepdims=True)  #(1,1)
        E_mean = pmean_if_pmap(E_mean)  #(1,1)
        if self.repeat_single_mol:
            E_mean_per_mol = E_mean
        epsilon_bar = (E_loc - E_mean_per_mol) / jnp.sqrt(electron_batch_size)
        epsilon_tilde = epsilon_bar - jnp.einsum("mjk, k -> mj", Ohat, prev_grad_decayed)  #(W,B,np),(np)->(W,B)
        epsilon_projected = jax.scipy.linalg.solve(T_reg, epsilon_tilde[..., None])[..., 0]  #(W,B,B)(W,B,1)-->(W,B,1)-->(W,B)
        dtheta_residual = jnp.einsum("mjk, mj -> k", Ohat, epsilon_projected)/walker_batch_this_process
        dtheta_residual = pmean_if_pmap(dtheta_residual)
        grad = dtheta_residual + prev_grad_decayed
        # scaled_grad = self.apply_norm_constraint(grad)
        return unravel_fn(grad), None, jnp.squeeze(E_mean)

    def get_grad_2(
        self,
        log_psi_grads,
        E_loc: P,
        E_mean_per_mol,
        opt_state,
    ) -> Tuple[Array, P]:
        prev_grad, unravel_fn = jax.flatten_util.ravel_pytree(opt_state["prev_grad"])
        prev_grad_decayed = self.mu * prev_grad  #(nparams,)
        W,B,nparams=log_psi_grads.shape
        nchains = W*B
        log_psi_grads=log_psi_grads.reshape((W*B,nparams)) /jnp.sqrt(nchains)  #(W*B,nparams)
        Ohat = log_psi_grads - jnp.mean(log_psi_grads, axis=0, keepdims=True)  #(W*B,nparams)
        T = Ohat @ Ohat.T  #(W*B,W*B)
        ones = jnp.ones((nchains, 1)) #(W*B,1)
        T_reg = T + ones @ ones.T / nchains + self.dp_schedule(opt_state["step"]) * jnp.eye(nchains)  #(W*B,W*B)
        E_mean = jnp.mean(E_mean_per_mol, keepdims=True)  #(1,1)
        E_mean = pmean_if_pmap(E_mean)  #(1,1)
        if self.repeat_single_mol:
            E_mean_per_mol = E_mean
        centered_energies = (E_loc - E_mean_per_mol)
        epsilon_bar = centered_energies.reshape((-1,)) / jnp.sqrt(nchains) #(W*B,)
        epsion_tilde = epsilon_bar - Ohat @ prev_grad_decayed   #(W*B,)
        dtheta_residual = Ohat.T @ jax.scipy.linalg.solve(T_reg, epsion_tilde, assume_a="pos") #(nparams,)
        dtheta_residual = pmean_if_pmap(dtheta_residual)
        SR_G = dtheta_residual + prev_grad_decayed
        # scaled_grad = self.apply_norm_constraint(SR_G)
        return unravel_fn(SR_G),  None, jnp.squeeze(E_mean)
    
    def get_grad_3(
            self,
            prev_grad,  #(np,)
            log_psi_grads,  #(B,np)
            E_loc: P,    #(B,)
            E_mean_per_mol,   #(1,)
            opt_state,
        ) -> Tuple[Array, P]:
        prev_grad_decayed = self.mu * prev_grad  #(nparams,)
        # logging.info("log_psi_grads shape: {}".format(log_psi_grads.shape))
        B,nparams=log_psi_grads.shape
        nchains = B
        Ohat = (log_psi_grads - jnp.mean(log_psi_grads, axis=-2, keepdims=True)) / jnp.sqrt(nchains)
        T = Ohat @ Ohat.T
        ones = jnp.ones_like(T) / nchains
        T_reg = T + ones + self.dp_schedule(opt_state["step"]) * jnp.eye(nchains)
        E_mean = jnp.mean(E_mean_per_mol, keepdims=True)  #(1,)
        E_mean = pmean_if_pmap(E_mean)  #(1,)
        if self.repeat_single_mol:
            E_mean_per_mol = E_mean
        epsilon_bar = (E_loc - E_mean_per_mol) / jnp.sqrt(nchains)
        epsilon_tilde = epsilon_bar - Ohat @ prev_grad_decayed
        epsilon_projected = jax.scipy.linalg.solve(T_reg, epsilon_tilde, assume_a="pos")
        dtheta_residual = Ohat.T @ epsilon_projected
        dtheta_residual = pmean_if_pmap(dtheta_residual)
        grad = dtheta_residual + prev_grad_decayed
        return grad,jnp.squeeze(E_mean)

    def apply_norm_constraint(self, grad: WavefunctionParams) -> WavefunctionParams:
        """Scales update to have L2 norm <= norm_constraint."""
        sq_norm_grads = jnp.sum(grad * grad)
        eps=1e-12
        coefficient = jnp.minimum(1, jnp.sqrt(self.norm_constraint / (sq_norm_grads + eps)))
        return grad * coefficient

    

    def update(
        self, grad_psi, E_loc, energy_per_w, opt_state: OptimizerState
    ) -> tuple[WavefunctionParams, OptimizerState]:
        if self.spr_type=="1":
            get_grad = self.get_grad_1
        elif self.spr_type=="2":
            get_grad = self.get_grad_2
        elif self.spr_type=="3":
            def get_grad(
                log_psi_grads,
                E_loc: P,
                E_mean_per_mol,
                opt_state,
            ) -> Tuple[Array, P]:
                prev_grad, unravel_fn = jax.flatten_util.ravel_pytree(opt_state["prev_grad"])
                grad, E_mean = jax.vmap(self.get_grad_3,in_axes=(None, 0, 0, 0, None))(prev_grad, log_psi_grads, E_loc, E_mean_per_mol, opt_state)
                grad = jnp.mean(grad, axis=0)
                E_mean = jnp.mean(E_mean, axis=0)
                return unravel_fn(grad), None, E_mean
        else:
            raise ValueError("Invalid SPRING type")
        # logging.info("grad_psi shape: {}".format(grad_psi.shape))
        grad, _, E_mean = get_grad(grad_psi, E_loc, energy_per_w, opt_state)
        update = jax.tree_util.tree_map(lambda x: -self.lr_schedule(opt_state["step"]) * x, grad)
        update = constrain_norm(update, self.norm_constraint)

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



from jax import lax

def check_nan(name, x):
    isnan = jnp.isnan(x)
    isinf = jnp.isinf(x)
    bad   = jnp.any(isnan | isinf)

    frac_nan = jnp.mean(isnan.astype(jnp.float32))
    min_val = jnp.nanmin(x)
    max_val = jnp.nanmax(x)

    dev_id = lax.axis_index(PMAP_AXIS_NAME)

    jax.debug.print(
            "[dev {}] {}: bad={} frac_nan={} min={} max={}",
            dev_id,
            name,
            bad,
            frac_nan,
            min_val,
            max_val,
        )

    return x


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