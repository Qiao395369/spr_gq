"""Get update functions from ConfigDicts."""

from typing import Tuple, TypeAlias, Dict, Callable, Optional, Any, Union

import jax
import kfac_jax
import optax
from ml_collections import ConfigDict
import chex
import jax.numpy as jnp

import vmcnet.mcmc.position_amplitude_core as pacore
import vmcnet.physics as physics
import vmcnet.utils as utils
from vmcnet.updates.spring import Optimizer
from vmcnet.utils.pytree_helpers import (
    tree_reduce_l1,
)

import vmcnet.utils.curvature_tags_and_blocks as curvature_tags_and_blocks

from vmcnet.utils.typing import (
    Array,
    Callable,
    D,
    GetPositionFromData,
    LearningRateSchedule,
    # OptimizerState,
    P,
    PRNGKey,
    PyTree,
    UpdateDataFn,
)

from .update_param_fns import UpdateParamFn, update_metrics_with_noclip
OptimizerState: TypeAlias = Dict
WavefunctionParams: TypeAlias = Dict

def _get_traced_compute_param_norm(
    apply_pmap: bool = True,
) -> Callable[[PyTree], Array]:
    if not apply_pmap:
        return jax.jit(tree_reduce_l1)

    return utils.distribute.pmap(tree_reduce_l1)


    

def kfac_wrapper(
    kfac_opt, 
    energy_and_statistics_fn,
    update_data_fn: UpdateDataFn[D, P],
)-> Optimizer:
    """Wrap a KFAC optimizer to make it compatible with the optimizer interface."""

    momentum = jnp.asarray(0.0)

    if kfac_opt.multi_device:
        momentum = utils.distribute.replicate_all_local_devices(momentum)
        update_data_fn = utils.distribute.pmap(update_data_fn)
        energy_and_statistics_fn = utils.distribute.pmap(energy_and_statistics_fn)


    def init(
        rng,
        params: P,
        data,
    ) -> OptimizerState:
        energy_per_w, E_loc, stats = energy_and_statistics_fn(params, data["atoms_position"], data["walker_data"]["elec_position"])
        batch = (E_loc, energy_per_w, data)
        return kfac_opt.init(params=params, batch=batch , rng=rng)

    def step(
        key: PRNGKey,
        params: WavefunctionParams,
        opt_state: OptimizerState,
        data,
    ) -> tuple[P,D, OptimizerState, Dict]:
        key, subkey = utils.distribute.split_or_psplit_key(key, kfac_opt.multi_device)
        energy_per_w, E_loc, stats = energy_and_statistics_fn(params, data["atoms_position"], data["walker_data"]["elec_position"])
        batch = (E_loc, energy_per_w, data)

        params, opt_state, opt_stats = kfac_opt.step(
            params=params,
            state=opt_state,
            rng=subkey,
            batch=batch,
            momentum=momentum,
        )
        data = update_data_fn(data, params)
        metrics = {
                    "energy": opt_stats["loss"], "variance": stats["variance"],
                    "kinetic": stats["kinetic"],
                    "ei_potential": stats["ei_potential"],
                    "ee_potential": stats["ee_potential"],
                    "ii_potential": stats["ii_potential"],
                    "multi_energy": stats["multi_energy"],
                    "opt_param_norm": opt_stats["param_norm"],
                    "opt_grad_norm": opt_stats["grad_norm"],
                    "opt_update_norm": opt_stats["update_norm"],
                    "energy_noclip": stats["energy_noclip"],
                    "variance_noclip": stats["variance_noclip"],
            }
        return params, data, opt_state, metrics, key

    return Optimizer(init=init, step=step)



