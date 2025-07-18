"""Entry points for running standard jobs."""
import argparse
import datetime
import functools
import logging
import os
import subprocess
from typing import Any, Optional, Tuple

import chex
import flax
import jax
import jax.numpy as jnp
import numpy as np
from absl import flags
from ml_collections import ConfigDict

from vmcnet.utils.distribute import distribute_vmc_state_from_checkpoint
import vmcnet.mcmc as mcmc
import vmcnet.mcmc.dynamic_width_position_amplitude as dwpa
import vmcnet.mcmc.position_amplitude_core as pacore
import vmcnet.models as models
import vmcnet.physics as physics
import vmcnet.train as train
import vmcnet.updates as updates
import vmcnet.utils as utils
from vmcnet.utils.typing import (
    Array,
    P,
    ClippingFn,
    PRNGKey,
    D,
    S,
    GetPositionFromData,
    GetAmplitudeFromData,
    LocalEnergyApply,
    ModelApply,
    OptimizerState,
)

FLAGS = flags.FLAGS


def _get_logdir_and_save_config(extra_config: ConfigDict, config: ConfigDict,type:str) -> str:
    if type=="infer":
        if config.subfolder_name != train.default_config.NO_NAME:
            config.logdir = os.path.join(config.logdir, config.subfolder_name)
        if config.save_to_current_datetime_subfolder:
            config.logdir = os.path.join(config.logdir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        config.logdir = utils.io.add_suffix_for_uniqueness(config.logdir)
    elif type=="reload":
        if extra_config.same_logdir:
            config.logdir = extra_config.logdir
        else:
            if config.subfolder_name != train.default_config.NO_NAME:
                config.logdir = os.path.join(config.logdir, config.subfolder_name)
            if config.save_to_current_datetime_subfolder:
                config.logdir = os.path.join(config.logdir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            config.logdir = utils.io.add_suffix_for_uniqueness(config.logdir)
    elif type=="RFM":
        if extra_config.same_logdir:
            config.logdir = extra_config.logdir
        else:
            if config.subfolder_name != train.default_config.NO_NAME:
                config.logdir = os.path.join(config.logdir, config.subfolder_name)
            if config.save_to_current_datetime_subfolder:
                config.logdir = os.path.join(config.logdir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            config.logdir = utils.io.add_suffix_for_uniqueness(config.logdir)
    utils.io.save_config_dict_to_json(config, config.logdir, "config")
    utils.io.save_config_dict_to_json(extra_config, config.logdir, type+"_config")
    # logging.info("%s configuration: \n%s", (name,reload_config))
    # logging.info("Running with configuration: \n%s", config)
    logging.info("%s configuration   : %s", type, config.logdir+"/"+type+"_config.json")
    logging.info("Run with configuration: %s", config.logdir+"/"+"config.json")
    return config.logdir


def _save_git_hash(logdir):
    # if logdir is None:
    #     return

    # git_hash = (
    #     subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    # )
    # git_file = os.path.join(logdir, "git_hash.txt")
    # writer = open(git_file, "wt")
    # writer.write(git_hash)
    # writer.close()
    return 




def _get_dtype(config: ConfigDict):
    if config.dtype == "float32":
        jax.config.update("jax_enable_x64", False)
        return jnp.float32
    elif config.dtype == "float64":
        jax.config.update("jax_enable_x64", True)
        return jnp.float64

    raise ValueError(
        "dtype other than float32, float64 not supported; {} was requested".format(
            config.dtype
        )
    )


def _get_electron_ion_config_as_arrays(
    config: ConfigDict, dtype=jnp.float32
) -> Tuple[Array, Array, Array]:
    ion_pos = jnp.array(config.ion_pos, dtype=dtype)
    if len(ion_pos.shape)==2:
        ion_pos=ion_pos[None,:]
    assert len(ion_pos.shape)==3
    ion_charges = jnp.array(config.ion_charges, dtype=dtype)
    single_nspins=jnp.array(config.single_nspins,dtype=int)
    nelec = jnp.array(config.nspins)
    nspins=config.nspins
    print("ion_pos:",ion_pos)
    return ion_pos, ion_charges, nelec ,nspins,single_nspins


def _get_and_init_model(
    model_config: ConfigDict,
    ion_pos: Array,
    ion_charges: Array,
    nelec: Array,
    init_pos: Array,
    key: PRNGKey,
    dtype=jnp.float32,
    apply_pmap: bool = True,
) -> Tuple[ModelApply[flax.core.FrozenDict], Any, PRNGKey]:
    slog_psi = models.construct.get_model_from_config(
        model_config, nelec, ion_pos, ion_charges, dtype=dtype
    )
    key, subkey = jax.random.split(key)
    params = slog_psi.init(subkey, init_pos[0:1])
    if apply_pmap:
        params = utils.distribute.replicate_all_local_devices(params)
    log_psi_apply = models.construct.slog_psi_to_log_psi_apply(slog_psi.apply)
    return log_psi_apply, params, key

def _get_gaoqiao_model(
        config_gq,
        wfn_type,
        nelec,
        charges,
        nspins,
        key,
        apply_pmap,
        do_RFM,
        params_fixed,
):
    if (do_RFM and params_fixed is None )or(not do_RFM and params_fixed is not None) :
        raise ValueError("params_fixed and params_RFM should be both None or both not None")
    import vmcnet.gaoqiao.build as gaoqiaobuild
    import vmcnet.gaoqiao.param_blocks as param_blocks
    from vmcnet.gaoqiao.sr import block_ravel_pytree
    key, subkey = jax.random.split(key)
    # charges=jnp.asarray([7.,7.])
    if wfn_type == "gaoqiao":
        layer_update_scheme = {
                                "update_alpha": config_gq.wfn_layer_update_alpha,
                                "do_resd_dt": config_gq.wfn_layer_do_resd_dt,
                              }    
        if config_gq.wfn_layer_resd_dt_shift is not None:
            layer_update_scheme["resd_dt_shift"] = config_gq.wfn_layer_resd_dt_shift
        if config_gq.wfn_layer_resd_dt_scale is not None:
            layer_update_scheme["resd_dt_scale"] = config_gq.wfn_layer_resd_dt_scale
        if config_gq.do_attn:
            attn_params = {
                'qkdim' : config_gq.attn_nchnl, 
                'nhead' : config_gq.attn_nhead, 
                'do_gate' : config_gq.attn_do_gate,
                'do_lnorm' : config_gq.attn_do_lnorm,
            }
        else: 
            attn_params = None
        if config_gq.do_h1_attn:
            h1_attn_params = {
                'qkdim' : config_gq.h1_attn_nchnl,
                'nhead' : config_gq.h1_attn_nhead,
                'do_gate' : config_gq.h1_attn_do_gate,
                'do_lnorm' : config_gq.h1_attn_do_lnorm,
                'resd_mode' : config_gq.h1_resd_mode,
            }
        else:
            h1_attn_params = None
        if config_gq.do_trimul:
            trimul_params = {
                "nchnl" : config_gq.trimul_nchnl, 
                "mode" : config_gq.trimul_mode,
            }
        else:
            trimul_params = None
        if config_gq.det_mode == "gemi":
            gemi_params = {
                "odim" : config_gq.gemi_odim, 
                "init_style": config_gq.gemi_init_style, 
                "diag_shift": config_gq.gemi_diag_shift,
                "weight_dim": config_gq.gemi_weight_dim,
                "hiddens": config_gq.gemi_hiddens if config_gq.gemi_hiddens is not None else [],
            }
        else:
            gemi_params = None

        feat_params = {
            "do_act": config_gq.feat_do_act,
            "act_func": config_gq.feat_act_func,
            "numb_divid": config_gq.feat_numb_divid,
            "scale": config_gq.scale if config_gq.scale != 1.0 else []
        }
        # trimul_params = None
        # gemi_params = None
        # feat_params = None
        params_pre,params_RFM, network_wfn_pre = gaoqiaobuild.build_network(           #orbitals
            n=nelec,  #电子个数
            charges=charges,  #i.e. charges=jnp.asarray([7.,7.])
            nspins=nspins,   #i.e. (7,7)
            key=key, 
            ndet=config_gq.ndet,  
            depth=config_gq.wfn_depth, 
            h1=config_gq.h1, 
            h2=config_gq.h2, 
            nh=config_gq.nh,
            do_complex=config_gq.do_complex,
            gq_type=config_gq.type,
            ef_construct_features_type=config_gq.ef_construct_features_type,
            envelope_type=config_gq.envelope_type,
            ef=config_gq.ef, 
            layer_update_scheme=layer_update_scheme,
            attn=attn_params, 
            h1_attn=h1_attn_params,
            trimul=trimul_params,
            feat_params=feat_params,
            det_mode=config_gq.det_mode, 
            gemi_params=gemi_params,
            RFM_layer=config_gq.RFM_layer,
        )        

    elif wfn_type == "gq_ferminet":
        from vmcnet.gaoqiao.fermi_ferminet import fermi_networks
        from vmcnet.gaoqiao.fermi_ferminet import fermi_envelopes
        import vmcnet.gaoqiao.fermi_ferminet.fermi_system as fermi_system
        envelope = fermi_envelopes.make_isotropic_envelope()
        feature_layer = fermi_networks.make_ferminet_features(
            natoms=charges.shape[0],
            nspins=nspins,
            ndim=3,
            rescale_inputs=False,#If true, rescale the inputs so they grow as log(|r|)
        )
        network = fermi_networks.make_fermi_net(
            nspins=nspins,
            charges=charges,
            ndim=3,
            determinants=config_gq.ndet,
            states=0,
            envelope=envelope,
            feature_layer=feature_layer,
            jastrow='default',
            bias_orbitals=False,
            full_det=True,
            rescale_inputs=False,
            complex_output=config_gq.do_complex,
            hidden_dims=tuple([(config_gq.h1,config_gq.h2) for _ in range(config_gq.wfn_depth)])
        )
        key, subkey = jax.random.split(key)
        params_pre ,params_RFM= network.init(subkey)
        spins_psi=None
        network_wfn_pre = lambda params_pre,params_RFM,xe,xp:network.apply(params_pre,params_RFM,xe,spins=spins_psi,atoms=xp,charges=charges)
    else:
        raise ValueError(f"Unknown electron wavefunction type: {wfn_type}")
    
    if not do_RFM:
        params_RFM['w']=None
        network_wfn=lambda params,xe,xp:network_wfn_pre(params,params_RFM,xe,xp)
        params=params_pre
    else:
        network_wfn=lambda params,xe,xp:network_wfn_pre(params_fixed,params,xe,xp)
        params=params_RFM

    def block_fn(block):
        if not isinstance(block, dict):
            return False
        return set() < set(block.keys()) <= {"w", "b"}

    print("params.shape:\n", jax.tree_util.tree_map(lambda x: x.shape, params))
    print("params.block.shape:\n", jax.tree_util.tree_map(lambda x: x.shape, block_ravel_pytree(block_fn)(params)))
    raveled_params, _ = jax.flatten_util.ravel_pytree(params)
    print("#parameters in the wavefunction model: %d" % raveled_params.size)

    if apply_pmap:
            params = utils.distribute.replicate_all_local_devices(params)

    @jax.jit
    def log_psi_apply_novmap(params,xp,xe):
        phase, logabsdet = network_wfn(params,xe,xp) #xe(ne,3),xp(na,3)
        return logabsdet

    # def log_psi_apply(params,xp,xe):
    #     if len(xe.shape)==4:
    #         return jax.vmap(jax.vmap(log_psi_apply_novmap,in_axes=(None,None,0)),in_axes=(None,0,0))(params,xp,xe)
    #     elif len(xe.shape)==2:
    #         return log_psi_apply_novmap(params,xp,xe)
    #     else:
    #         raise ValueError(f"wrong len(xe.shape): {len(xe.shape)} , xe.shape: {xe.shape}")
    @jax.jit
    def log_psi_apply(params, xp, xe):
        # if not isinstance(xe, jnp.ndarray):
        #     raise TypeError(f"xe must be a JAX array, got {type(xe)}")
        # def vmap_fn(params, xp, xe):
        return jax.vmap(jax.vmap(log_psi_apply_novmap, in_axes=(None, None, 0)), in_axes=(None, 0, 0))(params, xp, xe)
        # return jax.lax.cond(
        #     len(xe.shape) == 4,
        #     vmap_fn,
        #     log_psi_apply_novmap,
        #     params, xp, xe
        # )
        
    return log_psi_apply,log_psi_apply_novmap, params, key


# TODO: figure out how to merge this and other distributing logic with the current
# vmcnet/utils/distribute.py as well as vmcnet/mcmc
# TODO: make this flexible w.r.t. the type of data, not just use dwpa
# TODO: Here and elsewhere, fix the type hinting for model.apply and the local energy,
# which are more accurately described as Callables with signature
# (params, potentially-multiple-args-not-necessarily-arrays...) -> array
#
# The easiest, but somewhat inaccurate solution might be to just do
# Callable[[P, Union[Array, SLArray]], Array]
#
# The ideal would probably be something like Callable[[P, ...], Array], but this
# is not allowed (probably for good reason)
#
# The correct solution is probably something like this involving Protocols (PEP 544):
#
#     class ModelApply(Protocol[P]):
#         def __call__(params: P, *args) -> Array:
#             ...
#
# which creates a Generic class called ModelApply with only the first argument typed
def _make_initial_distributed_data(
    distributed_log_psi_apply: ModelApply[P],
    run_config: ConfigDict,
    init_pos: Array,
    params: P,
    dtype=jnp.float32,
) -> dwpa.DWPAData:
    # Need to use distributed_log_psi_apply here, in the case where there is not enough
    # memory to form the initial amplitudes on a single device
    sharded_init_pos = utils.distribute.default_distribute_data(init_pos)
    sharded_amplitudes = distributed_log_psi_apply(params, sharded_init_pos)
    move_metadata = utils.distribute.replicate_all_local_devices(
        dwpa.MoveMetadata(
            std_move=run_config.std_move,
            move_acceptance_sum=dtype(0.0),
            moves_since_update=0,
        )
    )
    return pacore.make_position_amplitude_data(
        sharded_init_pos, sharded_amplitudes, move_metadata
    )


def _make_initial_single_device_data(
    log_psi_apply: ModelApply[P],
    run_config: ConfigDict,
    ion_pos: Array,
    init_pos: Array,
    params: P,
    dtype=jnp.float32,
) -> dwpa.DWPAData:
    amplitudes = log_psi_apply(params, ion_pos, init_pos)
    return dwpa.make_dynamic_width_position_amplitude_data(
        ion_pos,
        init_pos,
        amplitudes,
        std_move=run_config.std_move,
        move_acceptance_sum=dtype(0.0),
        moves_since_update=0,
    )


def _make_initial_data(
    log_psi_apply: ModelApply[P],
    run_config: ConfigDict,
    ion_pos: Array,
    init_pos: Array,
    params: P,
    dtype=jnp.float32,
    apply_pmap: bool = True,
) -> dwpa.DWPAData:
    if apply_pmap:
        return _make_initial_distributed_data(
            utils.distribute.pmap(log_psi_apply), run_config, init_pos, params, dtype
        )
    else:
        return _make_initial_single_device_data(
            log_psi_apply, run_config, ion_pos, init_pos, params, dtype
        )


# TODO: add threshold_adjust_std_move options to configs
# TODO: add more options than just dwpa
# TODO: remove dependence on exact field names
def _get_mcmc_fns(
    run_config: ConfigDict, log_psi_apply: ModelApply[P], apply_pmap: bool = True
) -> Tuple[
    mcmc.metropolis.BurningStep[P, dwpa.DWPAData],
    mcmc.metropolis.WalkerFn[P, dwpa.DWPAData],
]:
    metrop_step_fn = dwpa.make_dynamic_pos_amp_gaussian_step(
        log_psi_apply,
        run_config.nmoves_per_width_update,
        dwpa.make_threshold_adjust_std_move(0.5, 0.05, 0.1),
    )
    burning_step = mcmc.metropolis.make_jitted_burning_step(
        metrop_step_fn, apply_pmap=apply_pmap
    )
    walker_fn = mcmc.metropolis.make_jitted_walker_fn(
        run_config.nsteps_per_param_update, metrop_step_fn, apply_pmap=apply_pmap
    )

    return burning_step, walker_fn


# TODO: figure out where this should go, perhaps in a physics/molecule.py file?
def _assemble_mol_local_energy_fn(
    local_energy_type: str,
    local_energy_config: ConfigDict,
    ion_pos: Array,
    ion_charges: Array,
    ei_softening: chex.Scalar,
    ee_softening: chex.Scalar,
    log_psi_apply: ModelApply[P],
) -> LocalEnergyApply[P]:
    if local_energy_type == "standard":
        kinetic_fn = physics.kinetic.create_laplacian_kinetic_energy(log_psi_apply)
        ei_potential_fn = physics.potential.create_electron_ion_coulomb_potential(
            ion_charges, softening_term=ei_softening
        )
        ee_potential_fn = physics.potential.create_electron_electron_coulomb_potential(
            softening_term=ee_softening,
        )
        ii_potential_fn = physics.potential.create_ion_ion_coulomb_potential(
            ion_charges,
        )
        # local_energy_fn: LocalEnergyApply[P] = physics.core.combine_local_energy_terms(
        #     [kinetic_fn, ei_potential_fn, ee_potential_fn, ii_potential_fn]
        # )
        # return local_energy_fn
        return kinetic_fn,ei_potential_fn,ee_potential_fn,ii_potential_fn
    
    if local_energy_type == "ibp":
        ibp_parts = local_energy_config.ibp.ibp_parts
        local_energy_fn = physics.ibp.create_ibp_local_energy(
            log_psi_apply,
            ion_pos,
            ion_charges,
            "kinetic" in ibp_parts,
            "ei" in ibp_parts,
            ei_softening,
            "ee" in ibp_parts,
            ee_softening,
        )
        return local_energy_fn
    elif local_energy_type == "random_particle":
        nparticles = local_energy_config.random_particle.nparticles
        sample_parts = local_energy_config.random_particle.sample_parts
        local_energy_fn = (
            physics.random_particle.create_molecular_random_particle_local_energy(
                log_psi_apply,
                ion_pos,
                ion_charges,
                nparticles,
                "kinetic" in sample_parts,
                "ei" in sample_parts,
                ei_softening,
                "ee" in sample_parts,
                ee_softening,
            )
        )
        return local_energy_fn
    else:
        raise ValueError(
            f"Requested local energy type {local_energy_type} is not supported"
        )


# TODO: figure out where this should go -- the act of clipping energies is kind of just
# a training trick rather than a physics thing, so maybe this stays here
def total_variation_clipping_fn(
    local_energies: Array,
    energy_noclip: chex.Numeric,
    threshold=5.0,
    clip_center="mean",
) -> Array:
    """Clip local es to within a multiple of the total variation from a center."""
    if clip_center == "mean":
        center = energy_noclip
    elif clip_center == "median":
        center = jnp.nanmedian(local_energies)
    else:
        raise ValueError(
            "Only mean and median are supported clipping centers, but {} was "
            "requested".format(clip_center)
        )
    total_variation = jnp.nanmean(jnp.abs(local_energies - center))
    clipped_local_e = jnp.clip(
        local_energies,
        center - threshold * total_variation,
        center + threshold * total_variation,
    )
    return clipped_local_e


# TODO: possibly include other types of clipping functions? e.g. using std deviation
# instead of total variation
def _get_clipping_fn(
    vmc_config: ConfigDict,
) -> Optional[ClippingFn]:
    clipping_fn = None
    if vmc_config.clip_threshold > 0.0:
        clipping_fn = functools.partial(
            total_variation_clipping_fn,
            threshold=vmc_config.clip_threshold,
            clip_center=vmc_config.clip_center,
        )
    return clipping_fn


def _get_energy_val_and_grad_fn(
    vmc_config: ConfigDict,
    problem_config: ConfigDict,
    ion_pos: Array,
    ion_charges: Array,
    log_psi_apply: ModelApply[P],
    log_psi_apply_novmap: ModelApply[P],
) -> physics.core.ValueGradEnergyFn[P]:
    ei_softening = problem_config.ei_softening
    ee_softening = problem_config.ee_softening

    # local_energy_fn = _assemble_mol_local_energy_fn(
    kinetic_fn,ei_potential_fn,ee_potential_fn,ii_potential_fn=_assemble_mol_local_energy_fn(
        vmc_config.local_energy_type,
        vmc_config.local_energy,
        ion_pos,
        ion_charges,
        ei_softening,
        ee_softening,
        # log_psi_apply,
        log_psi_apply_novmap,
    )

    clipping_fn = _get_clipping_fn(vmc_config)

    energy_data_val_and_grad = physics.core.create_value_and_grad_energy_fn(
        log_psi_apply,
        # local_energy_fn,
        kinetic_fn,ei_potential_fn,ee_potential_fn,ii_potential_fn,
        vmc_config.nchains * ion_pos.shape[0],
        clipping_fn,
        nan_safe=vmc_config.nan_safe,
        local_energy_type=vmc_config.local_energy_type,
    )

    return energy_data_val_and_grad


# TODO: don't forget to update type hint to be more general when
# _make_initial_distributed_data is more general
def _setup_vmc(
    config: ConfigDict,
    ion_pos: Array,
    ion_charges: Array,
    nelec: Array,
    nspins,
    single_nspins,
    key: PRNGKey,
    dtype=jnp.float32,
    apply_pmap: bool = True,
    do_RFM:bool=False,
    params_fixed:Optional[P]=None,
) -> Tuple[
    ModelApply[flax.core.FrozenDict],
    mcmc.metropolis.BurningStep[flax.core.FrozenDict, dwpa.DWPAData],
    mcmc.metropolis.WalkerFn[flax.core.FrozenDict, dwpa.DWPAData],
    updates.params.UpdateParamFn[flax.core.FrozenDict, dwpa.DWPAData, OptimizerState],
    GetAmplitudeFromData[dwpa.DWPAData],
    flax.core.FrozenDict,
    dwpa.DWPAData,
    OptimizerState,
    PRNGKey,
]:
    nelec_total = int(jnp.sum(nelec))
    key, init_pos = physics.core.initialize_molecular_pos(
        key, 
        config.vmc.nchains,
        ion_pos, 
        ion_charges, 
        nelec_total, 
        single_nspins,
        config.eval.init_width,
        dtype=dtype
    )   #init_pos:(W,B,ne,dim)

    # Make the model
    if config.wfn_type in ["gaoqiao","gq_ferminet"]:
        log_psi_apply, log_psi_apply_novmap,params, key =  _get_gaoqiao_model(
        config_gq=config.gq,
        wfn_type=config.wfn_type,
        nelec=nelec_total,
        charges=ion_charges,
        nspins=nspins,
        key=key,
        apply_pmap=apply_pmap,
        do_RFM=do_RFM,
        params_fixed=params_fixed,
        )
    elif config.wfn_type =="ll" :
        log_psi_apply, params, key = _get_and_init_model(
            config.model,
            ion_pos,
            ion_charges,
            nelec,
            init_pos,
            key,
            dtype=dtype,
            apply_pmap=apply_pmap,
        )
    else:
        raise ValueError("unknown gq_wfn_type: %s "%(config.wfn_type))

    # Make initial data
    data = _make_initial_data(
        log_psi_apply, config.vmc, ion_pos, init_pos, params, dtype=dtype, apply_pmap=apply_pmap
    )   #data:PositionAmplitudeData
    get_amplitude_fn = pacore.get_amplitude_from_data
    update_data_fn = pacore.get_update_data_fn(log_psi_apply)

    # Setup metropolis step
    burning_step, walker_fn = _get_mcmc_fns(
        config.vmc, log_psi_apply, apply_pmap=apply_pmap
    )

    energy_data_val_and_grad = _get_energy_val_and_grad_fn(
        config.vmc, config.problem, ion_pos, ion_charges, log_psi_apply,log_psi_apply_novmap
    )

    # Setup parameter updates
    if apply_pmap:
        key = utils.distribute.make_different_rng_key_on_all_devices(key)

    (   update_param_fn,
        optimizer_state,
        key,
    ) = updates.parse_config.get_update_fn_and_init_optimizer(
        # log_psi_apply,
        log_psi_apply_novmap,
        config.vmc,
        params,
        ion_pos,
        data,
        pacore.get_position_from_data,
        update_data_fn,
        energy_data_val_and_grad,
        key,
        apply_pmap=apply_pmap,
    )

    return (
        log_psi_apply,
        log_psi_apply_novmap,
        burning_step,
        walker_fn,
        update_param_fn,
        get_amplitude_fn,
        params,
        data,
        optimizer_state,
        key,
    )


# TODO: update output type hints when _get_mcmc_fns is made more general
def _setup_eval(
    eval_config: ConfigDict,
    problem_config: ConfigDict,
    ion_pos: Array,
    ion_charges: Array,
    log_psi_apply: ModelApply[P],
    log_psi_apply_novmap: ModelApply[P],
    get_position_fn: GetPositionFromData[dwpa.DWPAData],
    apply_pmap: bool = True,
) -> Tuple[
    updates.params.UpdateParamFn[P, dwpa.DWPAData, OptimizerState],
    mcmc.metropolis.BurningStep[P, dwpa.DWPAData],
    mcmc.metropolis.WalkerFn[P, dwpa.DWPAData],
]:
    ei_softening = problem_config.ei_softening
    ee_softening = problem_config.ee_softening

    # local_energy_fn = _assemble_mol_local_energy_fn(
    kinetic_fn,ei_potential_fn,ee_potential_fn,ii_potential_fn= _assemble_mol_local_energy_fn(
        eval_config.local_energy_type,
        eval_config.local_energy,
        ion_pos,
        ion_charges,
        ei_softening,
        ee_softening,
        log_psi_apply_novmap,
    )
    # local_energy_fn: LocalEnergyApply[P] = physics.core.combine_local_energy_terms(
    #         [kinetic_fn, ei_potential_fn, ee_potential_fn, ii_potential_fn]
    #     )

    eval_update_param_fn = updates.params.create_eval_update_param_fn(
        kinetic_fn,ei_potential_fn,ee_potential_fn,ii_potential_fn,
        # eval_config.nchains,
        get_position_fn,
        record_local_energies=eval_config.record_local_energies,
        nan_safe=eval_config.nan_safe,
        apply_pmap=apply_pmap,
        # use_PRNGKey=eval_config.local_energy_type == "random_particle",
    )
    eval_burning_step, eval_walker_fn = _get_mcmc_fns(
        eval_config, log_psi_apply, apply_pmap=apply_pmap
    )
    return eval_update_param_fn, eval_burning_step, eval_walker_fn


def _make_new_data_for_eval(
    config: ConfigDict,
    log_psi_apply: ModelApply[P],
    params: P,
    ion_pos: Array,
    ion_charges: Array,
    nelec: Array,
    single_nspins: Array,
    key: PRNGKey,
    is_pmapped: bool,
    dtype=jnp.float32,
) -> Tuple[PRNGKey, dwpa.DWPAData]:
    nelec_total = int(jnp.sum(nelec))
    # grab the first key if distributed
    if is_pmapped:
        key = utils.distribute.get_first(key)

    key, init_pos = physics.core.initialize_molecular_pos(
        key,
        config.eval.nchains,
        ion_pos,
        ion_charges,
        nelec_total,
        single_nspins,
        config.eval.init_width,
        dtype=dtype,
    )
    # redistribute if needed
    if config.distribute:
        key = utils.distribute.make_different_rng_key_on_all_devices(key)
    data = _make_initial_data(
        log_psi_apply,
        config.eval,
        ion_pos,
        init_pos,
        params,
        dtype=dtype,
        apply_pmap=config.distribute,
    )

    return key, data


def _burn_and_run_vmc(
    run_config: ConfigDict,
    logdir: str,
    params: P,
    optimizer_state: S,
    data: D,
    burning_step: mcmc.metropolis.BurningStep[P, D],
    walker_fn: mcmc.metropolis.WalkerFn[P, D],
    update_param_fn: updates.params.UpdateParamFn[P, D, S],
    get_amplitude_fn: GetAmplitudeFromData[D],
    key: PRNGKey,
    is_eval: bool,
    is_pmapped: bool,
    skip_burn: bool = False,
    start_epoch: int = 0,
) -> Tuple[P, S, D, PRNGKey, bool]:
    if not is_eval:
        checkpoint_every = run_config.checkpoint_every
        best_checkpoint_every = run_config.best_checkpoint_every
        checkpoint_dir = run_config.checkpoint_dir
        checkpoint_variance_scale = run_config.checkpoint_variance_scale
        nhistory_max = run_config.nhistory_max
        check_for_nans = run_config.check_for_nans
    else:
        checkpoint_every = None
        best_checkpoint_every = None
        checkpoint_dir = ""
        checkpoint_variance_scale = 0
        nhistory_max = 0
        check_for_nans = False

    if not skip_burn:
        data, key = mcmc.metropolis.burn_data(
            burning_step, run_config.nburn, params, data, key
        )
    return train.vmc.vmc_loop(
        params,
        optimizer_state,
        data,
        run_config.nchains,
        run_config.nepochs,
        walker_fn,
        update_param_fn,
        key,
        logdir=logdir,
        checkpoint_every=checkpoint_every,
        best_checkpoint_every=best_checkpoint_every,
        checkpoint_dir=checkpoint_dir,
        checkpoint_variance_scale=checkpoint_variance_scale,
        check_for_nans=check_for_nans,
        record_amplitudes=run_config.record_amplitudes,
        get_amplitude_fn=get_amplitude_fn,
        nhistory_max=nhistory_max,
        is_pmapped=is_pmapped,
        start_epoch=start_epoch,
        down_sample_num=(None if is_eval else run_config.down_sample_num),
        is_eval=is_eval,
    )


def _compute_and_save_energy_statistics(
    local_energies_file_path: str, output_dir: str, output_filename: str,nchains:int ,walkers:int ,nn:int
) -> None:
    local_energies = np.loadtxt(local_energies_file_path)
    eval_statistics = mcmc.statistics.get_stats_summary(local_energies,nchains,walkers,nn)
    # eval_statistics = jax.tree_map(lambda x: x.tolist(), eval_statistics)
    utils.io.save_dict_to_json(
        eval_statistics,
        output_dir,
        output_filename,
    )


def run_molecule() -> None:
    # "Run VMC on a molecule."
    reload_config, config = train.parse_config_flags.parse_flags(FLAGS)

    reload_from_checkpoint = (reload_config.logdir != train.default_config.NO_RELOAD_LOG_DIR and reload_config.use_checkpoint_file)

    if reload_from_checkpoint:
        config.notes = config.notes + " (reloaded from {}/{}{})".format(
            reload_config.logdir, reload_config.checkpoint_relative_file_path,
            ", new optimizer state" if reload_config.new_optimizer_state else "",)

    root_logger = logging.getLogger()
    root_logger.setLevel(config.logging_level)
    logdir = _get_logdir_and_save_config(reload_config, config,"reload")
    _save_git_hash(logdir)

    dtype_to_use = _get_dtype(config)

    ion_pos, ion_charges, nelec ,nspins, single_nspins= _get_electron_ion_config_as_arrays(config.problem, dtype=dtype_to_use)

    key = jax.random.PRNGKey(config.initial_seed)

    (log_psi_apply,log_psi_apply_novmap,burning_step,walker_fn,update_param_fn,
          get_amplitude_fn,params,data,optimizer_state,key,) = _setup_vmc(config,ion_pos,ion_charges,nelec,nspins,single_nspins,key,
                                                                          dtype=dtype_to_use,apply_pmap=config.distribute,)

    start_epoch = 0

    if reload_from_checkpoint:
        checkpoint_file_path = os.path.join(reload_config.logdir, reload_config.checkpoint_relative_file_path)
        directory, filename = os.path.split(checkpoint_file_path)

        (reload_at_epoch,data,params,reloaded_optimizer_state,key,) = utils.io.reload_vmc_state(directory, filename)

        if reload_config.append:
            utils.io.copy_txt_stats(reload_config.logdir, logdir, truncate=reload_at_epoch)

        if config.distribute:
            (data,params,reloaded_optimizer_state,key,) = distribute_vmc_state_from_checkpoint(data, params, reloaded_optimizer_state, key)

        if not reload_config.new_optimizer_state:
            optimizer_state = reloaded_optimizer_state
            start_epoch = reload_at_epoch

    logging.info("Saving to %s", logdir)
    data=pacore.make_position_amplitude_data(
        ion_pos,
        data["walker_data"]["position"],
        data["walker_data"]["amplitude"],
        data["move_metadata"],
    )
    params, optimizer_state, data, key, nans_detected = _burn_and_run_vmc(
                                                                            config.vmc,logdir,params,optimizer_state,data,burning_step,
                                                                            walker_fn,update_param_fn,get_amplitude_fn, key, 
                                                                            is_eval=False, is_pmapped=config.distribute,
                                                                            skip_burn=reload_from_checkpoint and not reload_config.reburn,
                                                                            start_epoch=start_epoch,)

    start_epoch = 0
    (log_psi_apply_RFM,log_psi_apply_novmap_RFM,burning_step,walker_fn,update_param_fn,
          get_amplitude_fn,params_RFM,data,optimizer_state,key,) = _setup_vmc(config,ion_pos,ion_charges,nelec,nspins,single_nspins,key,
                                                            dtype=dtype_to_use,apply_pmap=config.distribute,do_RFM=True,params_fixed=params)

    params_RFM, optimizer_state, data, key, nans_detected = _burn_and_run_vmc(
                                                                            config.vmc,logdir,params_RFM,optimizer_state,data,burning_step,
                                                                            walker_fn,update_param_fn,get_amplitude_fn, key, 
                                                                            is_eval=False, is_pmapped=config.distribute,
                                                                            skip_burn=reload_from_checkpoint and not reload_config.reburn,
                                                                            start_epoch=start_epoch,)
    
    if nans_detected:
        logging.info("VMC terminated due to Nans! Aborting.")
        return
    else:
        logging.info("Completed VMC! Evaluating...")

    # TODO: integrate the stuff in mcmc/statistics and write out an evaluation summary
    # (energy, var, overall mean acceptance ratio, std error, iac) to eval_logdir, post
    # evaluation
    eval_logdir = os.path.join(logdir, "eval")

    ion_pos, ion_charges, nelec ,nspins,single_nspins= _get_electron_ion_config_as_arrays(config.eval, dtype=dtype_to_use)

    eval_update_param_fn, eval_burning_step, eval_walker_fn = _setup_eval(config.eval,config.problem,ion_pos,ion_charges,log_psi_apply,log_psi_apply_novmap,
                                                                          pacore.get_position_from_data, apply_pmap=config.distribute,)
    optimizer_state = None

    if not config.eval.use_data_from_training :
        logging.info("creating new data ...")
        key, data = _make_new_data_for_eval(config,log_psi_apply,params,ion_pos,ion_charges,nelec,single_nspins,key,
                                            is_pmapped=config.distribute, dtype=dtype_to_use,)

    _burn_and_run_vmc(config.eval,eval_logdir,params,optimizer_state,data,eval_burning_step,
                    eval_walker_fn,eval_update_param_fn,get_amplitude_fn,key,
                    is_eval=True, is_pmapped=config.distribute,)

    # need to check for local_energy.txt because when config.eval.nepochs=0 the file is
    # not created regardless of config.eval.record_local_energies
    local_es_were_recorded = os.path.exists(os.path.join(eval_logdir, "local_energies.txt"))
    if config.eval.record_local_energies and local_es_were_recorded:
        local_energies_filepath = os.path.join(eval_logdir, "local_energies.txt")
        _compute_and_save_energy_statistics(local_energies_filepath, eval_logdir, "statistics",config.eval.nchains,ion_pos.shape[0],0)

# def RFM_from_train()->None():
#     FRM_config, config = train.parse_config_flags.RFM_parse_flags(FLAGS)

#     config.notes = config.notes + " (reloaded from {}/{})".format(RFM_config.logdir, RFM_config.checkpoint_relative_file_path,)

#     root_logger = logging.getLogger()
#     root_logger.setLevel(config.logging_level)
#     logdir = _get_logdir_and_save_config(RFM_config, config, "RFM")
#     _save_git_hash(logdir)
#     dtype_to_use = _get_dtype(config)

#     dtype_to_use = _get_dtype(config)

#     ion_pos, ion_charges, nelec ,nspins, single_nspins= _get_electron_ion_config_as_arrays(config.problem, dtype=dtype_to_use)

#     key = jax.random.PRNGKey(config.initial_seed)

#     (log_psi_apply,log_psi_apply_novmap,burning_step,walker_fn,update_param_fn,
#           get_amplitude_fn,params,data,optimizer_state,key,) = _setup_vmc(config,ion_pos,ion_charges,nelec,nspins,single_nspins,key,
#                                                                           dtype=dtype_to_use,apply_pmap=config.distribute,)

#     start_epoch = 0

#     if reload_from_checkpoint:
#         checkpoint_file_path = os.path.join(reload_config.logdir, reload_config.checkpoint_relative_file_path)
#         directory, filename = os.path.split(checkpoint_file_path)

#         (reload_at_epoch,data,params,reloaded_optimizer_state,key,) = utils.io.reload_vmc_state(directory, filename)

#         if reload_config.append:
#             utils.io.copy_txt_stats(reload_config.logdir, logdir, truncate=reload_at_epoch)

#         if config.distribute:
#             (data,params,reloaded_optimizer_state,key,) = distribute_vmc_state_from_checkpoint(data, params, reloaded_optimizer_state, key)

#         if not reload_config.new_optimizer_state:
#             optimizer_state = reloaded_optimizer_state
#             start_epoch = reload_at_epoch

#     logging.info("Saving to %s", logdir)

#     params, optimizer_state, data, key, nans_detected = _burn_and_run_vmc(
#                                                                             config.vmc,logdir,params,optimizer_state,data,burning_step,
#                                                                             walker_fn,update_param_fn,get_amplitude_fn, key, 
#                                                                             is_eval=False, is_pmapped=config.distribute,
#                                                                             skip_burn=reload_from_checkpoint and not reload_config.reburn,
#                                                                             start_epoch=start_epoch,)

#     if nans_detected:
#         logging.info("VMC terminated due to Nans! Aborting.")
#         return
#     else:
#         logging.info("Completed VMC! Evaluating...")

def do_inference()-> None:
    # TODO: integrate the stuff in mcmc/statistics and write out an evaluation summary
    # (energy, var, overall mean acceptance ratio, std error, iac) to eval_logdir, post evaluation
    infer_config, config = train.parse_config_flags.inference_parse_flags(FLAGS)

    config.notes = config.notes + " (reloaded from {}/{})".format(infer_config.logdir, infer_config.checkpoint_relative_file_path,)

    root_logger = logging.getLogger()
    root_logger.setLevel(config.logging_level)
    logdir = _get_logdir_and_save_config(infer_config, config, "infer")
    _save_git_hash(logdir)
    dtype_to_use = _get_dtype(config)

    ion_pos, ion_charges, nelec , nspins, single_nspins= _get_electron_ion_config_as_arrays(config.eval, dtype=dtype_to_use)

    key = jax.random.PRNGKey(config.initial_seed)

    nelec_total = int(jnp.sum(nelec))

    if config.wfn_type in ["gaoqiao","gq_ferminet"]:
        log_psi_apply, log_psi_apply_novmap,params, key =  _get_gaoqiao_model(
                    config_gq=config.gq,
                    wfn_type=config.wfn_type,
                    nelec=nelec_total,
                    charges=ion_charges,
                    nspins=nspins,
                    key=key,
                    apply_pmap=apply_pmap,
                    )
    
    get_amplitude_fn = pacore.get_amplitude_from_data
    
    checkpoint_file_path = os.path.join(infer_config.logdir, infer_config.checkpoint_relative_file_path)
    directory, filename = os.path.split(checkpoint_file_path)

    (reload_at_epoch,data,params,reloaded_optimizer_state,key,) = utils.io.reload_vmc_state(directory, filename)

    if config.distribute:
        (data,params,reloaded_optimizer_state,key,) = distribute_vmc_state_from_checkpoint(data, params, reloaded_optimizer_state, key)

    eval_update_param_fn, eval_burning_step, eval_walker_fn = _setup_eval(
        config.eval,
        config.problem,
        ion_pos,
        ion_charges,
        log_psi_apply,
        log_psi_apply_novmap,
        pacore.get_position_from_data,
        apply_pmap=config.distribute,
    )
    optimizer_state = None

    if not config.eval.use_data_from_training :
        logging.info("new data creating...")
        key, data = _make_new_data_for_eval(
                                            config,log_psi_apply,params,ion_pos,ion_charges,nelec,single_nspins,key,
                                            is_pmapped=config.distribute, dtype=dtype_to_use,)

    if config.density_plot:
        filename=config.density_plot_filename
        data, key = mcmc.metropolis.burn_data(eval_burning_step, config.eval.nburn, params, data, key)
        f = open(filename, "w", buffering=1, newline="\n")
        if os.path.getsize(filename) == 0:
            f.write("                x        y        z\n")
        print("Save configs to file: %s" % filename)
        for _ in range(config.density_plot_nepochs):
            
            accept_ratio, data, key = eval_walker_fn(params, data, key)
            positions = data["walker_data"]["elec_position"].reshape(-1,3)
            logging.info("num: %6d saves" % positions.shape[0])
            for row in positions:
                f.write(f"{'coordinate:'}  {row[0]}  {row[1]}  {row[2]}\n")
        import sys 
        sys.exit()

    
    eval_logdir = os.path.join(logdir, "data")
    logging.info("Saving to %s", eval_logdir)
    _burn_and_run_vmc(
                        config.eval,eval_logdir,params,optimizer_state,data,eval_burning_step,
                        eval_walker_fn,eval_update_param_fn,get_amplitude_fn,key,
                        is_eval=True, is_pmapped=config.distribute,)

    # need to check for local_energy.txt because when config.eval.nepochs=0 the file is
    # not created regardless of config.eval.record_local_energies
    local_es_were_recorded = os.path.exists(os.path.join(eval_logdir, "local_energies.txt"))
    if config.eval.record_local_energies and local_es_were_recorded:
        local_energies_filepath = os.path.join(eval_logdir, "local_energies.txt")
        _compute_and_save_energy_statistics(local_energies_filepath, eval_logdir, "statistics",config.eval.nchains,ion_pos.shape[0],0)


def vmc_statistics() -> None:
    """Calculate statistics from a VMC evaluation run and write them to disc."""
    parser = argparse.ArgumentParser(
        description="Calculate statistics from a VMC evaluation run and write them to disc."
    )
    parser.add_argument(
        "local_energies_file_path",
        type=str,
        help="File path to load local energies from",
    )
    parser.add_argument(
        "output_file_path",
        type=str,
        help="File path to which to write the output statistics. The '.json' suffix "
        "will be appended to the supplied path.",
    )
    parser.add_argument(
        "nchains",
        type=int,
        help="nchains",
    )
    parser.add_argument(
        "walkers",
        type=int,
        help="walkers",
    )
    parser.add_argument(
        "nn",
        type=int,
        help="from nn to the end",
    )

    args = parser.parse_args()

    output_dir, output_filename = os.path.split(os.path.abspath(args.output_file_path))
    _compute_and_save_energy_statistics(args.local_energies_file_path, output_dir, output_filename, args.nchains, args.walkers,args.nn)

if __name__=='__main__':
    run_molecule()