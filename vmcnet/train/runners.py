"""Entry points for running standard jobs."""

import argparse
import datetime
import functools
import logging
import os
import subprocess
from typing import Any, Optional, Tuple, Union
import math
import chex
import flax
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
import numpy as np
from absl import flags
from ml_collections import ConfigDict
# import wandb
from vmcnet.utils.distribute import PMAP_AXIS_NAME
import vmcnet.utils as utils
from functools import partial
import vmcnet.mcmc as mcmc
import vmcnet.mcmc.dynamic_width_position_amplitude as dwpa
import vmcnet.mcmc.position_amplitude_core as pacore
from vmcnet.mcmc.position_amplitude_core import down_sample_data, reform_data_and_metrics
import vmcnet.models as models
import vmcnet.physics as physics
import vmcnet.train as train
import vmcnet.updates as updates
import vmcnet.utils as utils
import kfac_jax
import vmcnet.gaoqiao.fermi_ferminet.fermi_system as fermi_system
import vmcnet.train.pretrain as pretrain
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
import sys
import time
from kfac_jax import utils as kfac_utils

import logging

def create_hf_data(ion_pos, symbol, nspins):
    """
    ion_pos:(W,natoms,dim)
    symbol:["x", "x", ...] total n=natoms strings
    """
    multi_walker_list = []
    walker_num, natoms, _= ion_pos.shape
    assert len(symbol) == natoms
    for i in range(walker_num):
        single_walker_list = []
        for j in range(natoms):
            single_atom_data = fermi_system.Atom(symbol[j], ion_pos[i][j]) 
            single_walker_list.append(single_atom_data)
        multi_walker_list.append(single_walker_list)

    hartree_focks = []
    for single_walker_data in multi_walker_list:
        hartree_fock = pretrain.get_hf(
            pyscf_mol = None,
            molecule = single_walker_data,
            nspins = nspins,
            restricted = False,
            basis = "ccpvdz",
            states = 0,
        )
        hartree_focks.append(hartree_fock)

    hf_wfn = []
    for hartree_fock in hartree_focks:
        hf_wfn_novmap = train.pretrain.get_hf_wfn(hartree_fock, nspins)
        hf_novmap = lambda xe : hf_wfn_novmap(xe)[1]
        hf_wfn.append(hf_novmap)
    return hartree_focks, hf_wfn

def create_hf_data_single(ion_pos, symbol, nspins):
    """
    ion_pos:(W,natoms,dim)
    symbol:["x", "x", ...] total n=natoms strings
    """
    data = ion_pos[0]   #data:(natoms,dim)
    natoms = data.shape[0]
    assert len(symbol) == natoms
    single_walker_list = []
    for i in range(natoms):
        single_atom_data = fermi_system.Atom(symbol[i], data[i]) 
        single_walker_list.append(single_atom_data)
    
    hartree_fock = pretrain.get_hf(
            pyscf_mol = None,
            molecule = single_walker_list,
            nspins = nspins,
            restricted = False,
            basis = "ccpvdz",
            states = 0,
        )
    hf_wfn_novmap = train.pretrain.get_hf_wfn(hartree_fock, nspins)
    hf_novmap = lambda xe : hf_wfn_novmap(xe)[1]
    
    return hartree_fock, hf_novmap


def show_devices():
    devices = jax.devices()
    local_n = jax.local_device_count()
    backend = jax.default_backend()  # "gpu" / "tpu" / "cpu"
    logging.info(
        "Devices=%s\n " \
        "         JAX backend=%s\n " \
        "         process_count=%d\n" \
        "          local_device_count=%d\n" \
        "          global_device_count=%d",
        devices, backend, jax.process_count(), local_n, jax.device_count()
    )


def get_params_initialization_key(deterministic):
  '''
  The key point here is to make sure different hosts uses the same RNG key
  to initialize network parameters.
  '''
  if deterministic:
    seed = 888
  else:
    # We make sure different hosts get the same seed.
    local_seed = time.time()
    float_seed = kfac_utils.compute_mean(jnp.ones(jax.local_device_count()) * local_seed)[0]
    seed = int(1e6 * float_seed)
  print(f'params initialization seed: {seed}')
  return jax.random.PRNGKey(seed)

def _get_logdir_and_save_config(reload_config: ConfigDict, config: ConfigDict,infer:bool) -> str:
    if infer:
        name="infer"
        if config.subfolder_name != train.default_config.NO_NAME:
            config.logdir = os.path.join(config.logdir, config.subfolder_name)
        if config.save_to_current_datetime_subfolder:
            config.logdir = os.path.join(
                config.logdir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            )
        config.logdir = utils.io.add_suffix_for_uniqueness(config.logdir)
    else:
        name="reload"
        if reload_config.same_logdir:
            config.logdir = reload_config.logdir
        else:
            if config.subfolder_name != train.default_config.NO_NAME:
                config.logdir = os.path.join(config.logdir, config.subfolder_name)
            if config.save_to_current_datetime_subfolder:
                config.logdir = os.path.join(config.logdir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            config.logdir = utils.io.add_suffix_for_uniqueness(config.logdir)
    utils.io.save_config_dict_to_json(config, config.logdir, "config")
    utils.io.save_config_dict_to_json(reload_config, config.logdir, name+"_config")
    # logging.info("%s configuration: \n%s", (name,reload_config))
    # logging.info("Running with configuration: \n%s", config)
    logging.info("%s configuration  : %s\n" \
    "          Run with configuration: %s", name, config.logdir+"/"+name+"_config.json", config.logdir+"/"+"config.json")
    return config.logdir


def _save_git_hash(logdir):
    if logdir is None:
        return

    git_hash = (
        subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    )
    git_file = os.path.join(logdir, "git_hash.txt")
    writer = open(git_file, "wt")
    writer.write(git_hash)
    writer.close()


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
    config: ConfigDict, dtype=jnp.float32, repeat_single_molecule=False, repeat_single_molecule_walker=4,
) -> Tuple[Array, Array, Array]:
    ion_pos = jnp.array(config.ion_pos, dtype=dtype)
    if len(ion_pos.shape)==2:
        ion_pos=ion_pos[None,:]

    if repeat_single_molecule:
        if ion_pos.shape[0]==1:
            ion_pos=jnp.repeat(ion_pos,repeat_single_molecule_walker,axis=0)
        else:
            RuntimeError("repeat_single_molecule is only valid for single-molecule walkers")
    
    ion_charges = jnp.array(config.ion_charges, dtype=dtype)
    single_nspins=jnp.array(config.single_nspins,dtype=int)
    nelec = jnp.array(config.nspins)
    nspins=config.nspins
    # logging.info(f"initial_ion_positions:{ion_pos.shape}")
    return ion_pos, ion_charges, nelec, nspins, single_nspins


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
        apply_pmap
):
    import vmcnet.gaoqiao.build as gaoqiaobuild
    from vmcnet.gaoqiao.sr import block_ravel_pytree
    key, subkey = jax.random.split(key)
    # charges=jnp.asarray([7.,7.])
    if wfn_type == "gaoqiao":
        if config_gq.do_attn:
            attn_params = {
                'qkdim' : config_gq.attn_nchnl, 
                'nhead' : config_gq.attn_nhead, 
                'do_gate' : config_gq.attn_do_gate,
                'do_lnorm' : config_gq.attn_do_lnorm,
            }
        else: 
            attn_params = None
        
        feat_params = {
            "do_act": config_gq.feat_do_act,
            "act_func": config_gq.feat_act_func,
            "numb_divid": config_gq.feat_numb_divid,
            "scale": config_gq.scale if config_gq.scale != 1.0 else [],
            "rescale": config_gq.rescale_input,
        }
        # trimul_params = None
        # gemi_params = None
        # feat_params = None
        params, network_wfn, det_fn, orb_fn = gaoqiaobuild.build_network(           #orbitals
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
            envelope_type=config_gq.envelope_type,
            layer_update_scheme=None,
            attn=attn_params, 
            h1_attn=None,
            trimul=None,
            feat_params=feat_params,
            det_mode=config_gq.det_mode, 
            gemi_params=None,
            jastrow_type=config_gq.jastrow_type,
            jastrow_mlp_nlayer=config_gq.jastrow_mlp_nlayer,
            jastrow_mlp_ndim=config_gq.jastrow_mlp_ndim,
            RHF=config_gq.RHF,
            activation_type=config_gq.activation_type,
        )        

    elif wfn_type == "gq_ferminet":
        from vmcnet.gaoqiao.fermi_ferminet import fermi_networks
        from vmcnet.gaoqiao.fermi_ferminet import fermi_envelopes
        envelope = fermi_envelopes.make_isotropic_envelope()
        if config_gq.ferminet_multi==False:
            feature_layer = fermi_networks.make_ferminet_features(
                natoms=charges.shape[0],
                nspins=nspins,
                ndim=3,
                rescale_inputs=True,
            )
        else:
            feature_layer = fermi_networks.make_ferminet_features_multi(
                natoms=charges.shape[0],
                nspins=nspins,
                ndim=3,
                rescale_inputs=True,
            )
        
        (network_init, network_apply, network_options, network_each_det, orbitals) = fermi_networks.make_fermi_net(
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
            rescale_inputs=True,
            complex_output=config_gq.do_complex,
            hidden_dims=tuple([(config_gq.h1,config_gq.h2) for _ in range(config_gq.wfn_depth)]),
            ferminet_multi=config_gq.ferminet_multi,
        )
        key, subkey = jax.random.split(key)
        params = network_init(subkey)
        spins_psi=None
        network_wfn = functools.partial(network_apply,
                                        spins=spins_psi,
                                        charges=charges,
                                        )
        det_fn = functools.partial( network_each_det,
                                    spins=spins_psi,
                                    charges=charges,
                                    )
        orb_fn = functools.partial(orbitals,
                                    spins=spins_psi,
                                    charges=charges,
                                    )
        
    elif wfn_type == 'psiformer':
        from vmcnet.gaoqiao.fermi_ferminet import fermi_networks
        from vmcnet.gaoqiao.fermi_ferminet import fermi_envelopes
        from vmcnet.gaoqiao.fermi_ferminet import psiformer
        envelope = fermi_envelopes.make_isotropic_envelope()
        if config_gq.psiformer_multi == False:
            feature_layer = fermi_networks.make_ferminet_features(
                natoms=charges.shape[0],
                nspins=nspins,
                ndim=3,
                rescale_inputs=True,
            )
            spins_psi=jnp.concatenate([jnp.ones(nspins[0]),-jnp.ones(nspins[1])])
        else:
            feature_layer = fermi_networks.make_ferminet_features_multi(
                natoms=charges.shape[0],
                nspins=nspins,
                ndim=3,
                rescale_inputs=True,
            )
            spins_psi=jnp.concatenate([jnp.ones(nspins[0]),-jnp.ones(nspins[1]),jnp.zeros(charges.shape[0])])
        
        psiformer_config={
              'num_layers': config_gq.psiformer_num_layers,
              'num_heads': config_gq.psiformer_num_heads,
              'heads_dim': config_gq.psiformer_heads_dim,
              'mlp_hidden_dims': (config_gq.psiformer_mlp_hidden_dims,),
              'use_layer_norm': True,
              }
        (network_init, network_apply, network_options, network_each_det, orbitals) = psiformer.make_fermi_net(
            nspins=nspins,
            charges=charges,
            ndim=3,
            determinants=config_gq.ndet,
            states=0,
            envelope=envelope,
            feature_layer=feature_layer,
            jastrow='default',
            bias_orbitals=False,
            rescale_inputs=True,
            complex_output=config_gq.do_complex,
            psiformer_multi=config_gq.psiformer_multi,
            **psiformer_config,
        )
        key, subkey = jax.random.split(key)
        params = network_init(subkey)
        network_wfn = functools.partial(network_apply,
                                        spins=spins_psi,
                                        charges=charges,
                                        )
        det_fn = functools.partial( network_each_det,
                                    spins=spins_psi,
                                    charges=charges,
                                    )
        orb_fn = functools.partial( orbitals,
                                    spins=spins_psi,
                                    charges=charges,
                                    )

    elif wfn_type == 'lapnet':
        from vmcnet.gaoqiao.lapnet import lapnet
        detnet = {
              'hidden_dims': tuple((config_gq.lapnet_mlp_hidden_dims,config_gq.lapnet_num_heads) for i in range(config_gq.lapnet_num_layers)),
              'determinants': 16,
              'after_determinants': (1,),
              }
        (network_init, signed_network, network_options, det_fn, orb_fn) = functools.partial(
            lapnet.make_lapnet,
            envelope='abs-isotropic',
            bias_orbitals=False,
            use_layernorm=False,
            jas_w_init=1.0,
            orbitals_spin_split=True,
            multi=config_gq.lapnet_multi,
            **detnet
            )(nspins, charges, hf_solution=None,)


        # params_initialization_key = get_params_initialization_key(True)
        key, subkey = jax.random.split(key)
        params = network_init(subkey)
        # params = kfac_utils.replicate_all_local_devices(params)
        # Often just need log|psi(x)|.
        network_wfn = lambda *args, **kwargs: signed_network(*args, **kwargs)

    else:
        raise ValueError(f"Unknown electron wavefunction type: {wfn_type}")
    
    def block_fn(block):
        if not isinstance(block, dict):
            return False
        return set() < set(block.keys()) <= {"w", "b"}
    
    print("params.shape:\n", jax.tree_util.tree_map(lambda x: x.shape, params))
    print("params.block.shape:\n", jax.tree_util.tree_map(lambda x: x.shape, block_ravel_pytree(block_fn)(params)))
    raveled_params, _ = jax.flatten_util.ravel_pytree(params)
    logging.info(f"parameters in the wavefunction model: {raveled_params.size}")
    print(f"parameters in the wavefunction model: {raveled_params.size}")

    if apply_pmap:
        params = utils.distribute.replicate_all_local_devices(params)

    # print("params.shape:\n", jax.tree_util.tree_map(lambda x: x.shape, params))
    # print("params.block.shape:\n", jax.tree_util.tree_map(lambda x: x.shape, block_ravel_pytree(block_fn)(params)))
    # raveled_params, _ = jax.flatten_util.ravel_pytree(params)
    # logging.info(f"#parameters in the wavefunction model: {raveled_params.size}")


    # @jax.jit
    def log_psi_apply_novmap(params,xp,xe):
        _, logabsdet = network_wfn(params,xe,xp) #xe(ne,3),xp(na,3)
        return logabsdet

    # @jax.jit
    def log_psi_apply_vmap(params, xp, xe):
        return jax.vmap(jax.vmap(log_psi_apply_novmap, in_axes=(None, None, 0)), in_axes=(None, 0, 0))(params, xp, xe)
        
    def det_fn_novmap(params,xp,xe):
        det = det_fn(params,xe,xp) #xe(ne,3),xp(na,3)
        return det
    
    def orb_fn_novmap(params,xp,xe):
        orb = orb_fn(params,xe,xp)[0] #xe(ne,3),xp(na,3)
        return orb
    
    orb_fn_vmap = jax.vmap(jax.vmap(orb_fn_novmap, in_axes=(None, None, 0)), in_axes=(None, 0, 0))
    # test_xe = jnp.ones((14,3))
    # test_xp = jnp.ones((2,3))

    # orb = orb_fn_novmap(params,test_xp,test_xe)
    # det = det_fn_novmap(params,test_xp,test_xe)
    # log_psi = log_psi_apply_novmap(params,test_xp,test_xe)
    # print("orb.shape:", orb.shape)
    # print("det.shape:", det.shape)
    # print("log_psi.shape:", log_psi.shape)
    # sys.exit()

    return log_psi_apply_vmap, log_psi_apply_novmap, det_fn_novmap, orb_fn_vmap, params, key


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
    ion_pos: Array,
    init_pos: Array,
    params: P,
    dtype=jnp.float32,
) -> dwpa.DWPAData:
    # Need to use distributed_log_psi_apply here, in the case where there is not enough
    # memory to form the initial amplitudes on a single device
    sharded_ion_pos = utils.distribute.default_distribute_data(ion_pos)
    sharded_init_pos = utils.distribute.default_distribute_data(init_pos)
    sharded_amplitudes = distributed_log_psi_apply(params,sharded_ion_pos, sharded_init_pos)
    move_metadata = utils.distribute.replicate_all_local_devices(
        dwpa.MoveMetadata(
            std_move=run_config.std_move,
            move_acceptance_sum=dtype(0.0),
            moves_since_update=0,
        )
    )
    return pacore.make_position_amplitude_data(
        sharded_ion_pos, sharded_init_pos, sharded_amplitudes, move_metadata
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
            utils.distribute.pmap(log_psi_apply), run_config, ion_pos, init_pos, params, dtype
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
    ion_charges: Array,
    ei_softening: chex.Scalar,
    ee_softening: chex.Scalar,
    log_psi_apply: ModelApply[P],
) :

    kinetic_fn = physics.kinetic.create_laplacian_kinetic_energy_new(log_psi_apply)
    ei_potential_fn = physics.potential.create_electron_ion_coulomb_potential(
        ion_charges, softening_term=ei_softening
    )
    ee_potential_fn = physics.potential.create_electron_electron_coulomb_potential(
        softening_term=ee_softening
    )
    ii_potential_fn = physics.potential.create_ion_ion_coulomb_potential(
        ion_charges
    )
    local_energy_fn = physics.core.combine_local_energy_terms([kinetic_fn,ei_potential_fn,ee_potential_fn,ii_potential_fn])
    return local_energy_fn

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
        center = jnp.nanmedian(local_energies, axis=-1, keepdims=True)
    else:
        raise ValueError(
            "Only mean and median are supported clipping centers, but {} was "
            "requested".format(clip_center)
        )
    total_variation = jnp.nanmean(jnp.abs(local_energies - center), axis=-1, keepdims=True)
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
) -> Tuple[
    ModelApply[flax.core.FrozenDict],
    mcmc.metropolis.BurningStep[flax.core.FrozenDict, dwpa.DWPAData],
    mcmc.metropolis.WalkerFn[flax.core.FrozenDict, dwpa.DWPAData],
    updates.update_param_fns.UpdateParamFn[
        flax.core.FrozenDict, dwpa.DWPAData, OptimizerState
    ],
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
        config.vmc.init_width,
        dtype=dtype
    )   #init_pos:(W,B,ne,dim)

    # Make the model
    if config.wfn_type in ["gaoqiao","gq_ferminet","psiformer","lapnet"]:
        log_psi_apply_vmap, log_psi_apply, det_fn_novmap, orb_fn_vmap, params, key =  _get_gaoqiao_model(
        config_gq=config.gq,
        wfn_type=config.wfn_type,
        nelec=nelec_total,
        charges=ion_charges,
        nspins=nspins,
        key=key,
        apply_pmap=apply_pmap,
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
        log_psi_apply_vmap, config.vmc, ion_pos, init_pos, params, dtype=dtype, apply_pmap=apply_pmap
    )
    logging.info("data shapes: %s", jax.tree_util.tree_map(lambda x: getattr(x, "shape", None), data))

    get_amplitude_fn = pacore.get_amplitude_from_data
    update_data_fn = pacore.get_update_data_fn(log_psi_apply_vmap)

    # Setup metropolis step
    burning_step, walker_fn = _get_mcmc_fns(config.vmc, log_psi_apply_vmap, apply_pmap=apply_pmap)

    local_energy_fn = _assemble_mol_local_energy_fn(
        ion_charges,
        config.problem.ei_softening,
        config.problem.ee_softening,
        log_psi_apply,
    )

    clipping_fn = _get_clipping_fn(config.vmc)

    energy_and_statistics_fn = physics.core.create_energy_and_statistics_fn(
            local_energy_fn, clipping_fn, config.vmc.nan_safe
        )
    
    # Setup parameter updates
    if apply_pmap:
        key = utils.distribute.make_different_rng_key_on_all_devices(key)

    down_sample_num = config.vmc.down_sample_num
    if down_sample_num != 0:
        if apply_pmap:
            assert down_sample_num % jax.device_count() == 0, "down_sample_num must be divisible by number of devices"
            down_sample_num = down_sample_num//jax.device_count()
        data_down_sample, _ , _, key = down_sample_data(key, data, down_sample_num, apply_pmap)
    else:
        data_down_sample = data

    (   update_param_fn,
        optimizer_state,
        key,
    ) = updates.parse_optimizer_config.initialize_optimizer(
        log_psi_apply,
        energy_and_statistics_fn,
        det_fn_novmap,
        config.vmc,
        params,
        data_down_sample,
        pacore.get_position_from_data,
        update_data_fn,
        key,
        apply_pmap=apply_pmap,
    )

    return (
        log_psi_apply_vmap,
        log_psi_apply,
        orb_fn_vmap,
        energy_and_statistics_fn,
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
    config: ConfigDict,
    ion_pos: Array,
    ion_charges: Array,
    log_psi_apply_vmap: ModelApply[P],
    log_psi_apply: ModelApply[P],
    get_position_fn: GetPositionFromData[dwpa.DWPAData],
    apply_pmap: bool = True,
) -> Tuple[
    updates.update_param_fns.UpdateParamFn[P, dwpa.DWPAData, OptimizerState],
    mcmc.metropolis.BurningStep[P, dwpa.DWPAData],
    mcmc.metropolis.WalkerFn[P, dwpa.DWPAData],
]:
    problem_config=config.problem
    eval_config=config.eval
    ei_softening = problem_config.ei_softening
    ee_softening = problem_config.ee_softening

    local_energy_fn = _assemble_mol_local_energy_fn(
        ion_charges,
        ei_softening,
        ee_softening,
        log_psi_apply,
    )
    eval_update_param_fn = updates.update_param_fns.construct_eval_update_param_fn(
        local_energy_fn,
        nan_safe=eval_config.nan_safe,
        apply_pmap=apply_pmap,
    )
    eval_burning_step, eval_walker_fn = _get_mcmc_fns(
        eval_config, log_psi_apply_vmap, apply_pmap=apply_pmap
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
    if is_pmapped:
        key = utils.distribute.make_different_rng_key_on_all_devices(key)

    data = _make_initial_data(
        log_psi_apply,
        config.eval,
        ion_pos,
        init_pos,
        params,
        dtype=dtype,
        apply_pmap=is_pmapped,
    )

    return key, data


def _burn_and_run_vmc(
    config: ConfigDict,
    ion_pos: Array,
    nspins,
    logdir: str,
    params: P,
    optimizer_state: S,
    data: D,
    net_orbitals_vmap,
    net_wfn_novmap,
    energy_and_statistics_fn,
    burning_step: mcmc.metropolis.BurningStep[P, D],
    walker_fn: mcmc.metropolis.WalkerFn[P, D],
    update_param_fn: updates.update_param_fns.UpdateParamFn[P, D, S],
    get_amplitude_fn: GetAmplitudeFromData[D],
    key: PRNGKey,
    is_eval: bool,
    is_pmapped: bool,
    skip_burn: bool = False,
    start_epoch: int = 0,
) -> Tuple[P, S, D, PRNGKey, bool]:
    if not is_eval:
        run_config = config.vmc
        checkpoint_every = run_config.checkpoint_every
        best_checkpoint_every = run_config.best_checkpoint_every
        checkpoint_dir = run_config.checkpoint_dir
        checkpoint_variance_scale = run_config.checkpoint_variance_scale
        nhistory_max = run_config.nhistory_max
        check_for_nans = run_config.check_for_nans
    else:
        run_config = config.eval
        checkpoint_every = None
        best_checkpoint_every = None
        checkpoint_dir = ""
        checkpoint_variance_scale = 0
        nhistory_max = 0
        check_for_nans = False

    config_pretrain = config.pretrain
    if config_pretrain.method == "hf" and config_pretrain.iterations > 0 and not is_eval:
        hartree_fock, hf_novmap = create_hf_data_single(ion_pos, config.problem.atoms_symbol, nspins)

        def hf_and_net_sum(params,xp,xe):
            result1 = jnp.exp(hf_novmap(xe))
            result2 = jnp.exp(net_wfn_novmap(params,xp,xe))
            result = jnp.log((result1+result2)/2)
            return result

        def hf_net(params,xp,xe):
            del params,xp
            return hf_novmap(xe)
        
        hf_wfn_sum_vmap = jax.vmap(jax.vmap(hf_and_net_sum, in_axes=(None, None, 0)), in_axes=(None, 0, 0))

        pretrain_burn_step, pretrain_walker_fn = _get_mcmc_fns(config_pretrain, hf_wfn_sum_vmap, apply_pmap=is_pmapped)

        key, subkeys = kfac_jax.utils.p_split(key)

        if not config_pretrain.skip_burn:
            data, key = mcmc.metropolis.burn_data(pretrain_burn_step, config_pretrain.nburn, params, data, key)

        params, data, key = pretrain.pretrain_hartree_fock_gaoqiao_2(
            params=params,
            data=data,
            net_orbitals_vmap=net_orbitals_vmap,
            energy_and_statistics_fn=energy_and_statistics_fn,
            pretrain_walker_fn=pretrain_walker_fn,
            walker_fn=walker_fn,
            burning_step=burning_step,
            key=key,
            nspins=nspins,
            scf_approx=hartree_fock,
            iterations=config_pretrain.iterations,
            )

    if not skip_burn:
        data, key = mcmc.metropolis.burn_data(burning_step, run_config.nburn, params, data, key)
    nchains = math.prod(data["atoms_position"].shape[:-2])
    return train.vmc.vmc_loop(
        params,
        optimizer_state,
        data,
        nchains,
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
        down_sample_num=(0 if is_eval else run_config.down_sample_num),
        is_eval=is_eval,
    )


def _compute_and_save_energy_statistics(
    local_energies_file_path: str, output_dir: str, output_filename: str,nchains:int ,walkers:int ,repeat_single_mol:bool
) -> None:
    local_energies = np.loadtxt(local_energies_file_path)
    eval_statistics = mcmc.statistics.get_stats_summary(local_energies,nchains,walkers,repeat_single_mol)
    # eval_statistics = jax.tree_map(lambda x: x.tolist(), eval_statistics)
    utils.io.save_dict_to_json(
        eval_statistics,
        output_dir,
        output_filename,
    )

import os
def run_molecule() -> None:
    """Run VMC on a molecule."""
    os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=2'
    # jax.config.update("jax_debug_nans", True)  # 发现 NaN/Inf 的原语会报错
    reload_config, config = train.parse_config_flags.parse_flags(FLAGS)
    
    reload_from_checkpoint = (
        reload_config.logdir != train.default_config.NO_RELOAD_LOG_DIR
        and reload_config.use_checkpoint_file
    )

    if reload_from_checkpoint:
        config.notes = config.notes + " (reloaded from {}/{}{})".format(
            reload_config.logdir,
            reload_config.checkpoint_relative_file_path,
            ", new optimizer state" if reload_config.new_optimizer_state else "",
        )

    root_logger = logging.getLogger()
    root_logger.setLevel(config.logging_level)
    logdir = _get_logdir_and_save_config(reload_config, config, False)
    # _save_git_hash(logdir)

    show_devices()

    apply_pmap = True

    dtype_to_use = _get_dtype(config)

    ion_pos, ion_charges, nelec ,nspins, single_nspins= _get_electron_ion_config_as_arrays(
        config.problem, 
        dtype=dtype_to_use, 
        repeat_single_molecule=config.vmc.repeat_single_mol ,
        repeat_single_molecule_walker=config.vmc.repeat_single_molecule_walker
    )

    key = jax.random.PRNGKey(config.initial_seed)

    (
        log_psi_apply_vmap,
        log_psi_apply_novmap,
        orb_fn_vmap,
        energy_and_statistics_fn,
        burning_step,
        walker_fn,
        update_param_fn,
        get_amplitude_fn,
        params,
        data,
        optimizer_state,
        key,
    ) = _setup_vmc(
        config,
        ion_pos,
        ion_charges,
        nelec,
        nspins,
        single_nspins,
        key,
        dtype=dtype_to_use,
        apply_pmap=apply_pmap,
    )

    start_epoch = 0

    if reload_from_checkpoint:
        checkpoint_file_path = os.path.join(
            reload_config.logdir, reload_config.checkpoint_relative_file_path
        )
        directory, filename = os.path.split(checkpoint_file_path)

        (
            reload_at_epoch,
            data,
            params,
            reloaded_optimizer_state,
            key,
        ) = utils.io.reload_vmc_state(directory, filename)

        if reload_config.append:
            utils.io.copy_txt_stats(
                reload_config.logdir, logdir, truncate=reload_at_epoch
            )

        if apply_pmap:
            (
                data,
                params,
                reloaded_optimizer_state,
                key,
            ) = utils.distribute.distribute_vmc_state_from_checkpoint(
                data, params, reloaded_optimizer_state, key
            )

        if not reload_config.new_optimizer_state:
            optimizer_state = reloaded_optimizer_state
            start_epoch = reload_at_epoch

    logging.info("Saving to %s", logdir)

    params, optimizer_state, data, key, nans_detected = _burn_and_run_vmc(
        config=config,
        ion_pos=ion_pos,
        nspins=nspins,
        logdir=logdir,
        params=params,
        optimizer_state=optimizer_state,
        data=data,
        net_orbitals_vmap=orb_fn_vmap,
        net_wfn_novmap=log_psi_apply_novmap,
        energy_and_statistics_fn=energy_and_statistics_fn,
        burning_step=burning_step,
        walker_fn=walker_fn,
        update_param_fn=update_param_fn,
        get_amplitude_fn=get_amplitude_fn,
        key=key,
        is_eval=False,
        is_pmapped=apply_pmap,
        skip_burn=reload_from_checkpoint and not reload_config.reburn,
        start_epoch=start_epoch,
    )

    if nans_detected:
        logging.info("VMC terminated due to Nans! Aborting.")
        return
    else:
        logging.info("Completed VMC! Evaluating")

    # TODO: integrate the stuff in mcmc/statistics and write out an evaluation summary
    # (energy, var, overall mean acceptance ratio, std error, iac) to eval_logdir, post
    # evaluation
    eval_logdir = os.path.join(logdir, "eval")

    ion_pos, ion_charges, nelec ,nspins,single_nspins= _get_electron_ion_config_as_arrays(
        config.eval, 
        dtype=dtype_to_use,
        repeat_single_molecule=config.eval.repeat_single_mol,
        repeat_single_molecule_walker=config.eval.repeat_single_molecule_walker,
        )

    eval_update_param_fn, eval_burning_step, eval_walker_fn = _setup_eval(
        config,
        ion_pos,
        ion_charges,
        log_psi_apply_vmap,
        log_psi_apply_novmap,
        pacore.get_position_from_data,
        apply_pmap=apply_pmap,
    )
    optimizer_sxtate = None

    eval_and_vmc_nchains_match = config.vmc.nchains == config.eval.nchains
    if not config.eval.use_data_from_training or not eval_and_vmc_nchains_match:
        logging.info("creating new data ...")
        key, data = _make_new_data_for_eval(
            config,
            log_psi_apply_vmap,
            params,
            ion_pos,
            ion_charges,
            nelec,
            single_nspins,
            key,
            is_pmapped=apply_pmap,
            dtype=dtype_to_use,
        )

    _burn_and_run_vmc(
        config=config,
        ion_pos=None,
        nspins=None,
        logdir=eval_logdir,
        params=params,
        optimizer_state=optimizer_state,
        data=data,
        net_orbitals_vmap=None,
        burning_step=eval_burning_step,
        walker_fn=eval_walker_fn,
        update_param_fn=eval_update_param_fn,
        get_amplitude_fn=get_amplitude_fn,
        key=key,
        is_eval=True,
        is_pmapped=apply_pmap,
    )

    # need to check for local_energy.txt because when config.eval.nepochs=0 the file is
    # not created regardless of config.eval.record_local_energies
    local_es_were_recorded = os.path.exists(
        os.path.join(eval_logdir, "multi_energy.txt")
    )
    if config.eval.record_local_energies and local_es_were_recorded:
        local_energies_filepath = os.path.join(eval_logdir, "multi_energy.txt")
        _compute_and_save_energy_statistics(
            local_energies_filepath, eval_logdir, "statistics", config.eval.nchains, ion_pos.shape[0], config.eval.repeat_single_mol
        )


def do_inference()-> None:
    # TODO: integrate the stuff in mcmc/statistics and write out an evaluation summary
    # (energy, var, overall mean acceptance ratio, std error, iac) to eval_logdir, post evaluation
    infer_config, config = train.parse_config_flags.inference_parse_flags(FLAGS)

    config.notes = config.notes + " (reloaded from {}/{})".format(infer_config.logdir, infer_config.checkpoint_relative_file_path,)

    root_logger = logging.getLogger()
    root_logger.setLevel(config.logging_level)
    logdir = _get_logdir_and_save_config(infer_config, config, True)
    # _save_git_hash(logdir)
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
        description="Calculate statistics from a VMC evaluation run and write them "
        "to disc."
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
    _compute_and_save_energy_statistics(
        args.local_energies_file_path, output_dir, output_filename, args.nchains, args.walkers,args.nn
    )


if __name__ == "__main__":
    run_molecule()