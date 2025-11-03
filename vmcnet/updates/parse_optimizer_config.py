"""Get update functions from ConfigDicts."""

from typing import Optional, Tuple
from ml_collections import ConfigDict

import vmcnet.physics as physics
from vmcnet.utils.typing import (
    ClippingFn,
    D,
    GetPositionFromData,
    LearningRateSchedule,
    LocalEnergyApply,
    ModelApply,
    OptimizerState,
    P,
    PRNGKey,
    UpdateDataFn,
)
import vmcnet.utils as utils
from .update_param_fns import UpdateParamFn
from .optax_utils import (
    initialize_adam,
    initialize_sgd,
)
from .spring import spring_wrapper, Spring
from .kfac import initialize_kfac
from .gauss_newton import initialize_gauss_newton
from vmcnet.updates.loss import flat_ansatz_call, make_loss, make_value_and_grad
import jax, kfac_jax
from functools import partial
from vmcnet.updates.kfac_multi import kfac_wrapper
from vmcnet.updates.kfacext import make_graph_patterns

def _get_learning_rate_schedule(
    optimizer_config: ConfigDict,
) -> LearningRateSchedule:
    if optimizer_config.schedule_type == "constant":

        def learning_rate_schedule(t):
            return optimizer_config.learning_rate

    elif optimizer_config.schedule_type == "inverse_time":

        def learning_rate_schedule(t):
            return optimizer_config.learning_rate / (
                1.0 + optimizer_config.learning_decay_rate * t
            )

    else:
        raise ValueError(
            "Learning rate schedule type not supported; {} was requested".format(
                optimizer_config.schedule_type
            )
        )

    return learning_rate_schedule

def _get_damping_rate_schedule(
    vmc_config: ConfigDict,
) -> LearningRateSchedule:
    optimizer_config = vmc_config.optimizer[vmc_config.optimizer_type]
    if optimizer_config.damp_schedule_type == "constant":

        return lambda t : optimizer_config.damping

    elif optimizer_config.damp_schedule_type == "inverse_time":

        return _get_InverseSchedule(
            optimizer_config.damping, vmc_config.nepochs // 100, optimizer_config.damping / 1000
            )

    else:
        raise ValueError(
            "damping rate schedule type not supported; {} was requested".format(
                optimizer_config.damp_schedule_type
            )
        )


def _get_InverseSchedule(init_value, decay_rate, offset=0.0):
        return lambda n: (init_value - offset) / (1 + n / decay_rate) + offset

def initialize_optimizer(
    log_psi_apply_novmap: ModelApply[P],
    kinetic_fn,ei_potential_fn,ee_potential_fn,ii_potential_fn,
    det_fn_novmap,
    clipping_fn: Optional[ClippingFn],
    vmc_config: ConfigDict,
    params: P,
    data: D,
    get_position_fn: GetPositionFromData[D],
    update_data_fn: UpdateDataFn[D, P],
    key: PRNGKey,
    apply_pmap: bool = True,
) -> Tuple[UpdateParamFn[P, D, OptimizerState], OptimizerState, PRNGKey]:
    """Get an update function and initialize optimizer state from the vmc configuration."""
    learning_rate_schedule = _get_learning_rate_schedule(
        vmc_config.optimizer[vmc_config.optimizer_type]
    )
    optimizer_config=vmc_config.optimizer[vmc_config.optimizer_type]

    if vmc_config.optimizer_type == "kfac":
        energy_data_val_and_grad = physics.core.create_value_and_grad_energy_fn(
            log_psi_apply_novmap,
            kinetic_fn,ei_potential_fn,ee_potential_fn,ii_potential_fn,
            vmc_config.nchains,
            clipping_fn,
            nan_safe=vmc_config.nan_safe,
        )
        return initialize_kfac(
            params,
            data,
            get_position_fn,
            update_data_fn,
            energy_data_val_and_grad,
            key,
            learning_rate_schedule,
            vmc_config.optimizer.kfac,
            vmc_config.record_param_l1_norm,
            apply_pmap=apply_pmap,
        )
    elif vmc_config.optimizer_type == "kfac_multi":
        opt_kwargs={}
        opt_kwargs["norm_constraint"] = optimizer_config.norm_constraint
        opt_kwargs["learning_rate_schedule"] = learning_rate_schedule
        opt_kwargs["damping_schedule"] = lambda n: optimizer_config.damping

        kfac_defaults = {
            "l2_reg": optimizer_config.l2_reg,
            "value_func_has_aux": False,
            "value_func_has_rng": False,
            "auto_register_kwargs": {"graph_patterns": make_graph_patterns()},
            # "use_automatic_registration": True,      # 保持自动注册
            # "register_only_generic": True,           # ★ 仅注册 generic，不跑复杂模式匹配
            # "auto_register_kwargs": None, 
            "include_norms_in_stats": True,
            "estimation_mode": optimizer_config.estimation_mode,
            "num_burnin_steps": 0,
            "min_damping": optimizer_config.min_damping,
            "inverse_update_period": optimizer_config.inverse_update_period,
            "pmap_axis_name": utils.distribute.PMAP_AXIS_NAME,
            # KFAC will be flatbatched to combine leading two dims
            "batch_size_extractor": (lambda batch, *_: (int(batch[-1]["walker_data"]["elec_position"].shape[0])
                                                                  * int(batch[-1]["walker_data"]["elec_position"].shape[1])
                                                                )
            ),
            "multi_device": apply_pmap,
        }
        energy_and_statistics_fn = physics.core.create_energy_and_statistics_fn(
            kinetic_fn,ei_potential_fn,ee_potential_fn,ii_potential_fn, vmc_config.debug, clipping_fn, vmc_config.nan_safe
        )
        loss_fn = make_value_and_grad(
            log_psi_apply_novmap,
            det_fn_novmap,
            vmc_config.repeat_single_mol,
            utils.distribute.PMAP_AXIS_NAME,
            flat_ansatz_call,
            vmc_config.det_penalty_weight,
            apply_pmap,
        )
        # value_and_grad_fn = jax.value_and_grad(loss_fn)

        opt = kfac_wrapper(
            kfac_jax.Optimizer(value_and_grad_func=loss_fn, **{**kfac_defaults, **opt_kwargs}),
            energy_and_statistics_fn,
            update_data_fn,
        )
        key, subkey = utils.distribute.split_or_psplit_key(key, apply_pmap)

        optimizer_state = opt.init(subkey,params,data)
        update_param_fn = opt.step
        return update_param_fn, optimizer_state, key

    elif vmc_config.optimizer_type == "sgd":
        energy_data_val_and_grad = physics.core.create_value_and_grad_energy_fn(
            log_psi_apply_novmap,
            kinetic_fn,ei_potential_fn,ee_potential_fn,ii_potential_fn,
            vmc_config.nchains,
            clipping_fn,
            nan_safe=vmc_config.nan_safe,
        )
        (
            update_param_fn,
            optimizer_state,
        ) = initialize_sgd(
            params,
            get_position_fn,
            update_data_fn,
            energy_data_val_and_grad,
            learning_rate_schedule,
            vmc_config.optimizer.sgd,
            vmc_config.record_param_l1_norm,
            apply_pmap=apply_pmap,
        )
        return update_param_fn, optimizer_state, key
    
    elif vmc_config.optimizer_type == "adam":
        energy_data_val_and_grad = physics.core.create_value_and_grad_energy_fn(
            log_psi_apply_novmap,
            kinetic_fn,ei_potential_fn,ee_potential_fn,ii_potential_fn,
            vmc_config.nchains,
            clipping_fn,
            nan_safe=vmc_config.nan_safe,
        )
        (
            update_param_fn,
            optimizer_state,
        ) = initialize_adam(
            params,
            get_position_fn,
            update_data_fn,
            energy_data_val_and_grad,
            learning_rate_schedule,
            vmc_config.optimizer.adam,
            vmc_config.record_param_l1_norm,
            apply_pmap=apply_pmap,
        )
        return update_param_fn, optimizer_state, key

    elif vmc_config.optimizer_type == "spring":
        damping_rate_schedule = _get_damping_rate_schedule(vmc_config)
        energy_and_statistics_fn = physics.core.create_energy_and_statistics_fn(
            kinetic_fn,ei_potential_fn,ee_potential_fn,ii_potential_fn, vmc_config.debug, clipping_fn, vmc_config.nan_safe
        )
        opt_kwargs = {}
        opt_kwargs["mu"] = optimizer_config.mu
        opt_kwargs["norm_constraint"] = optimizer_config.norm_constraint
        opt_kwargs["learning_rate_schedule"] = learning_rate_schedule
        opt_kwargs["damping_schedule"] = damping_rate_schedule
        opt_kwargs["repeat_single_mol"] = vmc_config.repeat_single_mol
        opt = spring_wrapper(Spring(**opt_kwargs), log_psi_apply_novmap, update_data_fn, energy_and_statistics_fn)
        if apply_pmap:
            update_param_fn = jax.pmap(opt.step, axis_name=utils.distribute.PMAP_AXIS_NAME)
            init_fn = jax.pmap(opt.init, axis_name=utils.distribute.PMAP_AXIS_NAME)
        else:
            update_param_fn = jax.jit(opt.step)
            init_fn = jax.jit(opt.init)
        optimizer_state = init_fn(params)
        return update_param_fn, optimizer_state, key
    
    elif vmc_config.optimizer_type == "gauss_newton":
        energy_and_statistics_fn = physics.core.create_energy_and_statistics_fn(
            kinetic_fn,ei_potential_fn,ee_potential_fn,ii_potential_fn, vmc_config.nchains, clipping_fn, vmc_config.nan_safe
        )
        (
            update_param_fn,
            optimizer_state,
        ) = initialize_gauss_newton(
            kinetic_fn,ei_potential_fn,ee_potential_fn,ii_potential_fn,
            log_psi_apply_novmap,
            energy_and_statistics_fn,
            params,
            get_position_fn,
            update_data_fn,
            learning_rate_schedule,
            vmc_config.optimizer.gauss_newton,
            vmc_config.record_param_l1_norm,
            apply_pmap=apply_pmap,
        )
        return update_param_fn, optimizer_state, key
    else:
        raise ValueError(
            "Requested optimizer type not supported; {} was requested".format(
                vmc_config.optimizer_type
            )
        )
