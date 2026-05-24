"""Entry points for running standard jobs."""

import argparse
import datetime
import functools
import logging
import os
import subprocess
import json
import shutil
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
# from vmcnet.mcmc.position_amplitude_core import make_down_sample_data_fn, make_reform_data_and_metrics_fn
import vmcnet.models as models
import vmcnet.physics as physics
import vmcnet.train as train
import vmcnet.updates as updates
import vmcnet.gaoqiao.envelopes as envelopes
import vmcnet.gaoqiao.jastrows as jastrows
import vmcnet.utils as utils
import kfac_jax
import vmcnet.gaoqiao.fermi_ferminet.fermi_system as fermi_system
import vmcnet.train.pretrain_demo as pretrain
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
        )
        hartree_focks.append(hartree_fock)

    hf_novmaps = []
    for hartree_fock in hartree_focks:
        hf_novmap = pretrain.make_HF_ansatz(hartree_fock)
        hf_novmaps.append(hf_novmap)
    return hartree_focks, hf_novmaps

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
    hf_novmap = train.pretrain.get_hf_wfn(hartree_fock, nspins)
    
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

        if config_gq.psiformer_multi == False:
            feature_layer = psiformer.make_ferminet_features(
                natoms=charges.shape[0],
                ndim=3,
                rescale_inputs=True,
            )
            spins_psi=jnp.concatenate([jnp.ones(nspins[0]),-jnp.ones(nspins[1])])
            envelope = envelopes.make_isotropic_envelope()
            jastrow = jastrows.make_simple_ee_jastrow(nspins = nspins)
        else:
            feature_layer = psiformer.make_ferminet_features_multi(
                natoms=charges.shape[0],
                ndim=3,
                rescale_inputs=True,
            )
            spins_psi=jnp.concatenate([jnp.ones(nspins[0]),-jnp.ones(nspins[1]),jnp.zeros(charges.shape[0])])
            make_envelope_kwargs = {"hiddens": [] if config_gq.nh == 0 else [config_gq.nh],}
            envelope = envelopes.make_ds_hz_envelope(**make_envelope_kwargs)
            jastrow = jastrows.make_mlp_jastrow(
                                                nspins = nspins,
                                                hiddenlayers_num=4,
                                                hiddenlayers_size=32,
                                                activation_fn=jax.nn.tanh,
                                                residual=True,
                                                )
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
            jastrow=jastrow,
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
                                        )
        det_fn = functools.partial( network_each_det,
                                    spins=spins_psi,
                                    )
        orb_fn = functools.partial( orbitals,
                                    spins=spins_psi,
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
    def log_psi_apply_vmap_two_dim(params, xp, xe):
        return jax.vmap(jax.vmap(log_psi_apply_novmap, in_axes=(None, None, 0)), in_axes=(None, 0, 0))(params, xp, xe)
        
    def log_psi_apply_vmap_one_dim(params, xp, xe):
        return jax.vmap(log_psi_apply_novmap, in_axes=(None, None, 0))(params, xp, xe)

    def det_fn_novmap(params,xp,xe):
        det = det_fn(params,xe,xp) #xe(ne,3),xp(na,3)
        return det
    
    det_fn_vmap = jax.vmap(jax.vmap(det_fn_novmap, in_axes=(None, None, 0)), in_axes=(None, 0, 0))
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

    return log_psi_apply_vmap_two_dim, log_psi_apply_vmap_one_dim, log_psi_apply_novmap, det_fn_novmap, det_fn_vmap, orb_fn_vmap, params, key


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
    W = sharded_init_pos.shape[1]
    move_metadata = utils.distribute.replicate_all_local_devices(
        dwpa.MoveMetadata(
            std_move=jnp.full((W,), run_config.std_move, dtype=dtype),
            move_acceptance_sum=jnp.full((W,), 0.0, dtype=dtype),
            moves_since_update=jnp.full((W,), 0),
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
    W = ion_pos.shape[0]
    return dwpa.make_dynamic_width_position_amplitude_data(
        ion_pos,
        init_pos,
        amplitudes,
        std_move=jnp.full((W,), run_config.std_move, dtype=dtype),
        move_acceptance_sum=jnp.full((W,), 0.0, dtype=dtype),
        moves_since_update=jnp.full((W,), 0),
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
    config_vmc,
) :

    kinetic_fn = physics.kinetic.create_laplacian_kinetic_energy_new(log_psi_apply, config_vmc.fwdlap_inner_size)
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
    reload_from_checkpoint: bool = False,
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
        log_psi_apply_vmap_two_dim, log_psi_apply_vmap_one_dim, log_psi_apply, det_fn_novmap,det_fn_vmap,orb_fn_vmap, params, key =  _get_gaoqiao_model(
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
    
    get_amplitude_fn = pacore.get_amplitude_from_data
    update_data_fn = pacore.get_update_data_fn(log_psi_apply_vmap_two_dim)

    # Setup metropolis step
    burning_step, walker_fn = _get_mcmc_fns(config.vmc, log_psi_apply_vmap_one_dim, apply_pmap=apply_pmap)
    
    local_energy_fn = _assemble_mol_local_energy_fn(
        ion_charges,
        config.problem.ei_softening,
        config.problem.ee_softening,
        log_psi_apply,
        config.vmc,
    )
    clipping_fn = _get_clipping_fn(config.vmc)
    energy_and_statistics_fn = physics.core.create_energy_and_statistics_fn(local_energy_fn, clipping_fn, config.vmc.nan_safe)

    if not reload_from_checkpoint:
        # Make initial data
        data = _make_initial_data(
            log_psi_apply_vmap_two_dim, config.vmc, ion_pos, init_pos, params, dtype=dtype, apply_pmap=apply_pmap
        )
        logging.info("data shapes: %s", jax.tree_util.tree_map(lambda x: getattr(x, "shape", None), data))
    else:
        data = None

    # Setup parameter updates
    if apply_pmap:
        key = utils.distribute.make_different_rng_key_on_all_devices(key)

    # down_sample_num = config.vmc.down_sample_num
    # if down_sample_num != 0:
    #     down_sample_data = make_down_sample_data_fn(apply_pmap)
    #     if apply_pmap:
    #         assert down_sample_num % jax.device_count() == 0, "down_sample_num must be divisible by number of devices"
    #         down_sample_num = down_sample_num//jax.device_count()
    #     data_down_sample, _ , _, key = down_sample_data(key, data, down_sample_num, apply_pmap)
    # else:
    #     data_down_sample = data
    

    energy_data_val_and_grad = physics.core.create_value_and_grad_energy_fn(   #for kfac
            log_psi_apply,
            local_energy_fn,
            config.vmc.repeat_single_mol ,
            init_pos.shape[0]*init_pos.shape[1],
            clipping_fn,
            nan_safe=config.vmc.nan_safe,
        )
    data_down_sample=None
    (   get_grad_and_E,
        update_param_fn,
        optimizer_state,
        key,
    ) = updates.parse_optimizer_config.initialize_optimizer(
        log_psi_apply,
        energy_and_statistics_fn,
        energy_data_val_and_grad,
        det_fn_novmap,
        det_fn_vmap,
        config.vmc,
        params,
        data_down_sample,   #only needed in kfac
        pacore.get_position_from_data,
        update_data_fn,
        key,
        apply_pmap=apply_pmap,

    )

    return (
        log_psi_apply_vmap_two_dim,
        log_psi_apply_vmap_one_dim,
        log_psi_apply,
        orb_fn_vmap,
        energy_and_statistics_fn,
        burning_step,
        walker_fn,
        get_grad_and_E,
        update_param_fn,
        get_amplitude_fn,
        update_data_fn,
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
        config.vmc,
    )
    eval_get_grad_and_E, eval_update_param_fn = updates.update_param_fns.construct_eval_update_param_fn(
        local_energy_fn,
        nan_safe=eval_config.nan_safe,
        apply_pmap=apply_pmap,
    )
    eval_burning_step, eval_walker_fn = _get_mcmc_fns(
        eval_config, log_psi_apply_vmap, apply_pmap=apply_pmap
    )
    return eval_get_grad_and_E, eval_update_param_fn, eval_burning_step, eval_walker_fn


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

def _make_new_data_for_reload(
    config: ConfigDict,
    reload_config: ConfigDict,
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
        reload_config.nchains,
        ion_pos,
        ion_charges,
        nelec_total,
        single_nspins,
        config.vmc.init_width,
        dtype=dtype,
    )
    # redistribute if needed
    if is_pmapped:
        key = utils.distribute.make_different_rng_key_on_all_devices(key)

    data = _make_initial_data(
        log_psi_apply,
        config.vmc,
        ion_pos,
        init_pos,
        params,
        dtype=dtype,
        apply_pmap=is_pmapped,
    )

    return key, data


def _burn_and_run_vmc(
    config: ConfigDict,
    logdir: str,
    params: P,
    optimizer_state: S,
    data: D,
    burning_step: mcmc.metropolis.BurningStep[P, D],
    nburn: int,
    walker_fn: mcmc.metropolis.WalkerFn[P, D],
    get_grad_and_E,
    update_param_fn: updates.update_param_fns.UpdateParamFn[P, D, S],
    get_amplitude_fn: GetAmplitudeFromData[D],
    key: PRNGKey,
    is_eval: bool,
    is_pmapped: bool,
    skip_burn: bool = False,
    start_epoch: int = 0,
    end_epochs: int = 0,
    down_sample_num: int =0,
) -> Tuple[P, S, D, PRNGKey, bool]:
    if not is_eval:
        run_config = config.vmc
        checkpoint_every = run_config.checkpoint_every
        best_checkpoint_every = run_config.best_checkpoint_every
        checkpoint_dir = run_config.checkpoint_dir
        checkpoint_variance_scale = run_config.checkpoint_variance_scale
        nhistory_max = run_config.nhistory_max
        check_for_nans = run_config.check_for_nans
        n_inner = run_config.n_inner
        acc_steps = run_config.acc_steps
        down_sample_mode = run_config.down_sample_mode
    else:
        run_config = config.eval
        checkpoint_every = None
        best_checkpoint_every = None
        checkpoint_dir = ""
        checkpoint_variance_scale = 0
        nhistory_max = 0
        check_for_nans = False
        down_sample_num = 0
        n_inner = 0
        acc_steps = 1
        down_sample_mode = ""

    if not skip_burn:
        data, key = mcmc.metropolis.burn_data(burning_step, nburn, params, data, key, is_pmapped)
    nchains = math.prod(data["atoms_position"].shape[:-2])
    return train.vmc.vmc_loop(
        params,
        optimizer_state,
        data,
        nchains,
        end_epochs,
        walker_fn,
        get_grad_and_E,
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
        down_sample_num=down_sample_num,
        down_sample_mode=down_sample_mode,
        n_inner=n_inner,
        acc_steps=acc_steps,
        is_eval=is_eval,
    )


def _compute_and_save_energy_statistics(
    local_energies_file_path: str, output_dir: str, output_filename: str,nchains:int ,walkers:int ,repeat_single_mol:bool, cut:int
) -> None:
    local_energies = np.loadtxt(local_energies_file_path)
    eval_statistics = mcmc.statistics.get_stats_summary(local_energies,nchains,walkers,repeat_single_mol,cut)
    # eval_statistics = jax.tree_map(lambda x: x.tolist(), eval_statistics)
    utils.io.save_dict_to_json(
        eval_statistics,
        output_dir,
        output_filename,
    )






# =========================
# MCMC diagnostics for dissociated H
# =========================

import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _device_to_np(x):
    return np.asarray(jax.device_get(x))


def _get_H_radius(atoms_np, geom_idx, H_idx):
    """Return R_H, min distance from H to body, and counting radius r_c."""
    R = atoms_np[geom_idx]
    R_H = R[H_idx]
    body_idx = [i for i in range(R.shape[0]) if i != H_idx]
    d_body = np.linalg.norm(R[body_idx] - R_H[None, :], axis=-1)
    dmin = float(np.min(d_body))
    r_c = min(2.5, 0.35 * dmin)
    return R_H, dmin, r_c


def _compute_H_observables(elec_np, atoms_np, geom_indices, H_idx, n_up):
    """
    elec_np:  [W, B, Ne, 3]
    atoms_np: [W, Nat, 3]
    """
    rows = []
    dmin_samples = {}

    for g in geom_indices:
        R_H, d_H_body_min, r_c = _get_H_radius(atoms_np, g, H_idx)

        d = np.linalg.norm(elec_np[g] - R_H[None, None, :], axis=-1)  # [B, Ne]

        nH = (d < r_c).sum(axis=1)
        nH_up = (d[:, :n_up] < r_c).sum(axis=1)
        nH_down = (d[:, n_up:] < r_c).sum(axis=1)

        dmin_eH = d.min(axis=1)   #（B)

        rows.append({
            "geom": int(g),
            "d_H_body_min": d_H_body_min,
            "r_c": float(r_c),
            "nH_mean": float(np.mean(nH)),
            "nH_std": float(np.std(nH)),
            "nH_up_mean": float(np.mean(nH_up)),
            "nH_down_mean": float(np.mean(nH_down)),
            "dmin_mean": float(np.mean(dmin_eH)),
            "dmin_p10": float(np.percentile(dmin_eH, 10)),
            "dmin_p50": float(np.percentile(dmin_eH, 50)),
            "dmin_p90": float(np.percentile(dmin_eH, 90)),
        })

        dmin_samples[int(g)] = dmin_eH.copy()

    return rows, dmin_samples


def _sample_H_1s(key, R_H, n_samples, dtype, zeta=1.0):
    """
    Approx hydrogen 1s distribution:
        radial pdf proportional to r^2 exp(-2 zeta r)
    therefore:
        r ~ Gamma(shape=3, scale=1/(2*zeta))
    """
    key, key_r, key_u = jax.random.split(key, 3)

    r = jax.random.gamma(
        key_r,
        a=jnp.asarray(3.0, dtype=dtype),
        shape=(n_samples,),
        dtype=dtype,
    ) / jnp.asarray(2.0 * zeta, dtype=dtype)

    u = jax.random.normal(key_u, shape=(n_samples, 3), dtype=dtype)
    u = u / jnp.linalg.norm(u, axis=-1, keepdims=True)

    pos = R_H[None, :] + r[:, None] * u
    return key, pos


def _get_seed_electron_indices(single_nspins, H_idx, n_up):
    """
    Electron ordering follows your initialize_molecular_pos logic:
        first all spin-up electrons atom by atom,
        then all spin-down electrons atom by atom.

    For H_idx=4:
        single_nspins[4] = [0, 1]
    so H has no initial up electron and one initial down electron.
    """
    s = _device_to_np(single_nspins).astype(int)
    natoms = s.shape[0]

    if H_idx < 0:
        H_idx = natoms + H_idx

    # If dissociated H has no up electron assigned, use the first up electron as donor.
    if s[H_idx, 0] > 0:
        up_eidx = int(np.sum(s[:H_idx, 0]))
    else:
        up_eidx = 0

    # For down electron, use the electron assigned to dissociated H.
    if s[H_idx, 1] > 0:
        down_eidx = int(n_up + np.sum(s[:H_idx, 1]))
    else:
        down_eidx = int(n_up)

    return up_eidx, down_eidx


def _make_seeded_elec_position(
    key,
    elec_position,
    atoms_position,
    geom_indices,
    H_idx,
    n_up,
    single_nspins,
    dtype,
    mode,
    zeta=1.0,
):
    """
    mode:
      normal      : unchanged
      H_down_seed : force the H-assigned down electron near dissociated H
      H_up_seed   : force one up electron near dissociated H, and move H-down electron to old up position
      mixed_seed  : first half H_up_seed, second half H_down_seed
    """
    x = jnp.array(elec_position)
    atoms = jnp.array(atoms_position)

    if H_idx < 0:
        H_idx = atoms.shape[1] + H_idx

    B = x.shape[1]
    up_eidx, down_eidx = _get_seed_electron_indices(single_nspins, H_idx, n_up)

    print(f"seed indices: up_eidx={up_eidx}, down_eidx={down_eidx}")

    if mode == "normal":
        return key, x

    for g in geom_indices:
        R_H = atoms[g, H_idx]

        if mode == "H_down_seed":
            key, pos_down = _sample_H_1s(key, R_H, B, dtype=dtype, zeta=zeta)
            x = x.at[g, :, down_eidx, :].set(pos_down)

        elif mode == "H_up_seed":
            key, pos_up = _sample_H_1s(key, R_H, B, dtype=dtype, zeta=zeta)

            # Avoid putting both selected up and H-down on H:
            # move H-down electron to old up position.
            old_up_pos = x[g, :, up_eidx, :]
            x = x.at[g, :, up_eidx, :].set(pos_up)
            x = x.at[g, :, down_eidx, :].set(old_up_pos)

        elif mode == "mixed_seed":
            half = B // 2

            key, pos_up = _sample_H_1s(key, R_H, half, dtype=dtype, zeta=zeta)
            key, pos_down = _sample_H_1s(key, R_H, B - half, dtype=dtype, zeta=zeta)

            old_up_pos = x[g, :half, up_eidx, :]

            # first half: H-up sector
            x = x.at[g, :half, up_eidx, :].set(pos_up)
            x = x.at[g, :half, down_eidx, :].set(old_up_pos)

            # second half: H-down sector
            x = x.at[g, half:, down_eidx, :].set(pos_down)

        else:
            raise ValueError(f"Unknown seed mode: {mode}")

    return key, x


def _rebuild_data_with_new_electrons(
    log_psi_apply_vmap_two_dim,
    config,
    params,
    old_data,
    new_elec_position,
    dtype,
):
    """After changing electron positions, recompute amplitude consistently."""
    atoms_position = old_data["atoms_position"]
    amplitude = log_psi_apply_vmap_two_dim(params, atoms_position, new_elec_position)

    old_meta = old_data["move_metadata"]

    # Keep exactly the same dtype/shape as the checkpoint metadata.
    return dwpa.make_dynamic_width_position_amplitude_data(
        atoms_position,
        new_elec_position,
        amplitude,
        std_move=old_meta["std_move"],
        move_acceptance_sum=jnp.zeros_like(old_meta["move_acceptance_sum"]),
        moves_since_update=jnp.zeros_like(old_meta["moves_since_update"]),
    )



def _energy_rows(local_es_np, geom_indices):
    rows = {}
    for g in geom_indices:
        e = np.asarray(local_es_np[g]).reshape(-1)
        e = e[np.isfinite(e)]
        if len(e) == 0:
            rows[int(g)] = {
                "E_mean": np.nan,
                "E_std": np.nan,
                "E_stderr": np.nan,
            }
        else:
            rows[int(g)] = {
                "E_mean": float(np.mean(e)),
                "E_std": float(np.std(e)),
                "E_stderr": float(np.std(e) / np.sqrt(len(e))),
            }
    return rows


def _save_trace_csv(trace_rows, path):
    if len(trace_rows) == 0:
        return

    keys = sorted(set(k for r in trace_rows for k in r.keys()))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in trace_rows:
            writer.writerow(r)


def _plot_mcmc_diagnostics(trace_rows, dmin_bank, outdir, geom_indices):
    os.makedirs(outdir, exist_ok=True)

    variants = sorted(set(r["variant"] for r in trace_rows))

    for g in geom_indices:
        # nH trace
        plt.figure(figsize=(8, 5))
        for variant in variants:
            rs = [
                r for r in trace_rows
                if r["geom"] == int(g) and r["variant"] == variant
            ]
            rs = sorted(rs, key=lambda x: x["macro_step"])
            plt.plot(
                [r["macro_step"] for r in rs],
                [r["nH_mean"] for r in rs],
                label=variant,
            )
        plt.xlabel("macro MCMC step")
        plt.ylabel("mean n_H")
        plt.title(f"geom {g}: H occupation")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"nH_trace_geom{g}.png"), dpi=200)
        plt.close()

        # spin-resolved nH trace
        plt.figure(figsize=(8, 5))
        for variant in variants:
            rs = [
                r for r in trace_rows
                if r["geom"] == int(g) and r["variant"] == variant
            ]
            rs = sorted(rs, key=lambda x: x["macro_step"])
            plt.plot(
                [r["macro_step"] for r in rs],
                [r["nH_up_mean"] for r in rs],
                linestyle="-",
                label=f"{variant}: up",
            )
            plt.plot(
                [r["macro_step"] for r in rs],
                [r["nH_down_mean"] for r in rs],
                linestyle="--",
                label=f"{variant}: down",
            )
        plt.xlabel("macro MCMC step")
        plt.ylabel("mean n_H by spin")
        plt.title(f"geom {g}: spin-resolved H occupation")
        plt.grid(True)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"nH_spin_trace_geom{g}.png"), dpi=200)
        plt.close()

        # closest electron-H distance histogram
        plt.figure(figsize=(8, 5))
        for variant in variants:
            key = (variant, int(g))
            if key not in dmin_bank:
                continue
            samples = np.concatenate(dmin_bank[key], axis=0)
            plt.hist(samples, bins=80, density=True, histtype="step", label=variant)
        plt.xlabel("closest electron-H distance / bohr")
        plt.ylabel("density")
        plt.title(f"geom {g}: closest electron-H distance")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"dmin_hist_geom{g}.png"), dpi=200)
        plt.close()

        # energy trace
        if any(("E_mean" in r) for r in trace_rows):
            plt.figure(figsize=(8, 5))
            for variant in variants:
                rs = [
                    r for r in trace_rows
                    if r["geom"] == int(g)
                    and r["variant"] == variant
                    and "E_mean" in r
                    and np.isfinite(r["E_mean"])
                ]
                rs = sorted(rs, key=lambda x: x["macro_step"])
                if len(rs) == 0:
                    continue
                plt.errorbar(
                    [r["macro_step"] for r in rs],
                    [r["E_mean"] for r in rs],
                    yerr=[r["E_stderr"] for r in rs],
                    label=variant,
                    capsize=2,
                )
            plt.xlabel("macro MCMC step")
            plt.ylabel("local energy / Ha")
            plt.title(f"geom {g}: energy")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"energy_trace_geom{g}.png"), dpi=200)
            plt.close()


import re


def _extract_ckpt_step(reload_at_epoch, reload_config):
    """
    Determine checkpoint step for output folder name.

    Priority:
      1. the last integer in reload_config.checkpoint_relative_file_path, e.g. checkpoint_150000.npz -> 150000
      2. reload_at_epoch + 1, matching the original zero-based epoch convention
    """
    ckpt_path = str(reload_config.checkpoint_relative_file_path)
    nums = re.findall(r"\d+", ckpt_path)
    if len(nums) > 0:
        return int(nums[-1])

    try:
        return int(reload_at_epoch + 1)
    except Exception:
        raise RuntimeError(
            f"Cannot extract checkpoint step from path={ckpt_path} "
            f"or reload_at_epoch={reload_at_epoch}."
        )


def _make_unique_test_outdir(base_dir, ckpt_step, suffix="test"):
    """
    生成类似：
        ../test_results/ckpt150000test
        ../test_results/ckpt150000test_1
        ../test_results/ckpt150000test_2

    如果目录已存在，自动加 _1, _2, ...
    """
    os.makedirs(base_dir, exist_ok=True)

    base_name = f"ckpt{ckpt_step}{suffix}"
    outdir = os.path.join(base_dir, base_name)

    if not os.path.exists(outdir):
        os.makedirs(outdir)
        return outdir

    i = 1
    while True:
        candidate = os.path.join(base_dir, f"{base_name}_{i}")
        if not os.path.exists(candidate):
            os.makedirs(candidate)
            return candidate
        i += 1



def _str2bool(x):
    """Parse boolean values from command-line strings."""
    if isinstance(x, bool):
        return x
    x = str(x).strip().lower()
    if x in ["true", "1", "yes", "y", "t"]:
        return True
    if x in ["false", "0", "no", "n", "f"]:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value from {x}")


def _parse_test_mcmc_args():
    """
    Parse diagnostic-only arguments with argparse, then remove them from sys.argv
    so train.parse_config_flags.parse_flags(FLAGS) will not see unknown --test_* flags.
    """
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--test_nwalkers", type=int, default=128)
    parser.add_argument("--test_geom_indices", type=str, default="16,17,18,19")
    parser.add_argument("--test_H_idx", type=int, default=4)

    parser.add_argument("--test_n_macro_steps", type=int, default=300)
    parser.add_argument("--test_burn_in_macro", type=int, default=50)
    parser.add_argument("--test_thin_macro", type=int, default=1)

    parser.add_argument("--test_compute_energy", type=_str2bool, default=True)
    parser.add_argument("--test_energy_every", type=int, default=10)

    parser.add_argument("--test_zeta_H", type=float, default=1.0)
    parser.add_argument(
        "--test_variants",
        type=str,
        default="normal,H_down_seed,H_up_seed,mixed_seed",
    )

    parser.add_argument("--test_results_root", type=str, default="../test_results")
    parser.add_argument("--test_suffix", type=str, default="test")
    parser.add_argument(
        "--test_ckpt_step_override",
        type=int,
        default=None,
        help="Optional manual checkpoint step used in output folder name.",
    )

    original_argv = list(sys.argv)
    test_args, remaining_argv = parser.parse_known_args(sys.argv[1:])

    # Make absl/ml_collections flags ignore these diagnostic-only arguments.
    sys.argv = [sys.argv[0]] + remaining_argv

    return test_args, original_argv


def _parse_geom_indices(s, n_geometries):
    """
    Supported formats:
        "all"       -> all geometries
        "16:20"     -> [16, 17, 18, 19]
        "16,17,19"  -> [16, 17, 19]
        "19"        -> [19]
    """
    s = str(s).strip()
    if s.lower() == "all":
        geom_indices = np.arange(n_geometries, dtype=int)
    elif ":" in s:
        a, b = s.split(":")
        geom_indices = np.arange(int(a), int(b), dtype=int)
    else:
        geom_indices = np.array(
            [int(x.strip()) for x in s.split(",") if x.strip() != ""],
            dtype=int,
        )

    if len(geom_indices) == 0:
        raise ValueError("test_geom_indices selected zero geometries.")

    bad = [int(g) for g in geom_indices if g < 0 or g >= n_geometries]
    if bad:
        raise ValueError(
            f"Invalid geometry indices {bad}; valid range is [0, {n_geometries - 1}]."
        )

    return geom_indices


def _parse_variants(s):
    variants = [x.strip() for x in str(s).split(",") if x.strip() != ""]
    allowed = {"normal", "H_down_seed", "H_up_seed", "mixed_seed"}

    if len(variants) == 0:
        raise ValueError("test_variants must contain at least one variant.")

    bad = [x for x in variants if x not in allowed]
    if bad:
        raise ValueError(f"Unknown test variants: {bad}. Allowed variants are {sorted(allowed)}")

    return variants


def _slice_data_nwalkers(data, nwalkers):
    """
    Keep only the first nwalkers walkers in the checkpoint data.

    For normal:
      - if checkpoint walkers > nwalkers: truncate
      - if checkpoint walkers == nwalkers: unchanged
      - if checkpoint walkers < nwalkers: raise error
    """
    nwalkers = int(nwalkers)
    if nwalkers <= 0:
        raise ValueError(f"test_nwalkers must be positive, got {nwalkers}")

    old_nwalkers = int(data["walker_data"]["elec_position"].shape[1])
    if old_nwalkers < nwalkers:
        raise RuntimeError(
            f"Requested test_nwalkers={nwalkers}, but checkpoint only has "
            f"{old_nwalkers} walkers. Cannot create extra normal walkers."
        )

    if old_nwalkers == nwalkers:
        return data

    new_data = dict(data)
    new_walker_data = dict(data["walker_data"])
    new_walker_data["elec_position"] = data["walker_data"]["elec_position"][:, :nwalkers, ...]
    new_walker_data["amplitude"] = data["walker_data"]["amplitude"][:, :nwalkers]
    new_data["walker_data"] = new_walker_data

    return new_data


def _find_config_files_from_argv(original_argv):
    """
    Copy raw config-like files if they appear in the command line, e.g.
        --config=/path/to/config.py
        --config /path/to/config.py
        --reload_config=/path/to/reload.py
    """
    paths = []
    for i, arg in enumerate(original_argv):
        if arg.startswith("--") and "config" in arg and "=" in arg:
            value = arg.split("=", 1)[1]
            if os.path.isfile(value):
                paths.append(value)
        elif arg.startswith("--") and "config" in arg and "=" not in arg:
            if i + 1 < len(original_argv):
                value = original_argv[i + 1]
                if os.path.isfile(value):
                    paths.append(value)

    unique = []
    seen = set()
    for p in paths:
        ap = os.path.abspath(p)
        if ap not in seen:
            unique.append(ap)
            seen.add(ap)
    return unique


def _json_safe_obj(obj):
    """Convert common numpy/jax/ml_collections values to JSON-safe Python objects."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    try:
        if isinstance(obj, (jnp.ndarray,)):
            return np.asarray(obj).tolist()
    except Exception:
        pass
    return str(obj)


def _backup_test_run_inputs(outdir, config, reload_config, test_args, original_argv, ckpt_step):
    """
    Store the exact diagnostic parameters and config backups in the result folder.
    """
    # Evaluated config backups.
    utils.io.save_config_dict_to_json(config, outdir, "config_backup")
    utils.io.save_config_dict_to_json(reload_config, outdir, "reload_config_backup")

    # Argparse diagnostic parameters.
    with open(os.path.join(outdir, "test_args.json"), "w") as f:
        json.dump(vars(test_args), f, indent=2, default=_json_safe_obj)

    # Original shell command.
    with open(os.path.join(outdir, "command.txt"), "w") as f:
        f.write(" ".join(original_argv))
        f.write("\n")

    run_info = {
        "ckpt_step": int(ckpt_step),
        "checkpoint_logdir": str(reload_config.logdir),
        "checkpoint_relative_file_path": str(reload_config.checkpoint_relative_file_path),
        "output_directory": str(outdir),
    }
    with open(os.path.join(outdir, "run_info.json"), "w") as f:
        json.dump(run_info, f, indent=2, default=_json_safe_obj)

    # Copy raw config files if they were supplied on command line.
    input_backup_dir = os.path.join(outdir, "input_files")
    os.makedirs(input_backup_dir, exist_ok=True)
    for src in _find_config_files_from_argv(original_argv):
        dst = os.path.join(input_backup_dir, os.path.basename(src))
        shutil.copy2(src, dst)




import os



def test_mcmc() -> None:
    """
    Diagnostics:
      1. n_H trace
      2. spin-resolved n_H trace
      3. closest electron-to-H distance histogram
      4. normal vs H_up_seed vs H_down_seed vs mixed_seed local energy

    This function does not train parameters.
    It freezes the checkpoint params and only runs MCMC.

    Diagnostic settings are controlled by argparse flags:
      --test_nwalkers
      --test_geom_indices
      --test_H_idx
      --test_n_macro_steps
      --test_burn_in_macro
      --test_thin_macro
      --test_compute_energy
      --test_energy_every
      --test_zeta_H
      --test_variants
      --test_results_root
      --test_suffix
      --test_ckpt_step_override
    """

    test_args, original_argv = _parse_test_mcmc_args()

    reload_config, config = train.parse_config_flags.parse_flags(FLAGS)

    reload_from_checkpoint = (
        reload_config.logdir != train.default_config.NO_RELOAD_LOG_DIR
        and reload_config.use_checkpoint_file
    )
    if not reload_from_checkpoint:
        raise RuntimeError("test_mcmc requires reload_config.use_checkpoint_file=True")

    root_logger = logging.getLogger()
    root_logger.setLevel(config.logging_level)

    logdir = _get_logdir_and_save_config(reload_config, config, False)
    show_devices()

    apply_pmap = config.distribute or reload_config.to_pmap
    if apply_pmap:
        raise RuntimeError("This test_mcmc diagnostic is written for apply_pmap=False.")

    dtype_to_use = _get_dtype(config)

    ion_pos, ion_charges, nelec, nspins, single_nspins = _get_electron_ion_config_as_arrays(
        config.problem,
        dtype=dtype_to_use,
        repeat_single_molecule=config.vmc.repeat_single_mol,
        repeat_single_molecule_walker=config.vmc.repeat_single_molecule_walker,
    )

    key = jax.random.PRNGKey(config.initial_seed)

    (
        log_psi_apply_vmap_two_dim,
        log_psi_apply_vmap_one_dim,
        log_psi_apply_novmap,
        orb_fn_vmap,
        energy_and_statistics_fn,
        burning_step,
        walker_fn,
        get_grad_and_E,
        update_param_fn,
        get_amplitude_fn,
        update_data_fn,
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
        reload_from_checkpoint=True,
    )

    checkpoint_file_path = os.path.join(
        reload_config.logdir,
        reload_config.checkpoint_relative_file_path,
    )
    logging.info("Reloading from %s", checkpoint_file_path)

    directory, filename = os.path.split(checkpoint_file_path)

    (
        reload_at_epoch,
        data,
        params,
        reloaded_optimizer_state,
        key,
    ) = utils.io.reload_vmc_state(directory, filename)

    data = _slice_data_nwalkers(data, test_args.test_nwalkers)

    print("\n===== loaded checkpoint =====")
    print("reload_at_epoch =", reload_at_epoch)
    print("atoms_position shape =", data["atoms_position"].shape)
    print("elec_position shape =", data["walker_data"]["elec_position"].shape)
    print("nspins =", nspins)
    print("single_nspins =", single_nspins)
    print("test_nwalkers =", test_args.test_nwalkers)

    # -------------------------
    # Diagnostic settings
    # -------------------------
    H_idx = int(test_args.test_H_idx)
    if H_idx < 0:
        H_idx = int(data["atoms_position"].shape[1]) + H_idx

    geom_indices = _parse_geom_indices(
        test_args.test_geom_indices,
        n_geometries=int(data["atoms_position"].shape[0]),
    )

    n_up = int(nspins[0])

    n_macro_steps = int(test_args.test_n_macro_steps)
    burn_in_macro = int(test_args.test_burn_in_macro)
    thin_macro = int(test_args.test_thin_macro)

    compute_energy = bool(test_args.test_compute_energy)
    energy_every = int(test_args.test_energy_every)

    if n_macro_steps < 0:
        raise ValueError(f"test_n_macro_steps must be non-negative, got {n_macro_steps}")
    if burn_in_macro < 0:
        raise ValueError(f"test_burn_in_macro must be non-negative, got {burn_in_macro}")
    if thin_macro <= 0:
        raise ValueError(f"test_thin_macro must be positive, got {thin_macro}")
    if energy_every <= 0:
        raise ValueError(f"test_energy_every must be positive, got {energy_every}")

    zeta_H = float(test_args.test_zeta_H)
    variants = _parse_variants(test_args.test_variants)

    if test_args.test_ckpt_step_override is not None:
        ckpt_step = int(test_args.test_ckpt_step_override)
    else:
        ckpt_step = _extract_ckpt_step(reload_at_epoch, reload_config)

    test_results_root = os.path.abspath(test_args.test_results_root)
    outdir = _make_unique_test_outdir(
        base_dir=test_results_root,
        ckpt_step=ckpt_step,
        suffix=str(test_args.test_suffix),
    )

    _backup_test_run_inputs(
        outdir=outdir,
        config=config,
        reload_config=reload_config,
        test_args=test_args,
        original_argv=original_argv,
        ckpt_step=ckpt_step,
    )

    print("\n===== output directory =====")
    print("ckpt_step =", ckpt_step)
    print("outdir =", outdir)
    print("variants =", variants)
    print("geom_indices =", geom_indices)
    print("H_idx =", H_idx)
    print("n_macro_steps =", n_macro_steps)
    print("burn_in_macro =", burn_in_macro)
    print("thin_macro =", thin_macro)
    print("compute_energy =", compute_energy)
    print("energy_every =", energy_every)
    print("zeta_H =", zeta_H)

    atoms_np = _device_to_np(data["atoms_position"])
    print("\n===== selected dissociation geometries =====")
    for g in geom_indices:
        _, d_H_body_min, r_c = _get_H_radius(atoms_np, int(g), H_idx)
        print(f"geom {int(g):02d}: min d(H-body) = {d_H_body_min:.6f} bohr, r_c = {r_c:.6f}")

    # Local energy fn for per-geometry energy comparison.
    if compute_energy:
        local_energy_fn = _assemble_mol_local_energy_fn(
            ion_charges,
            config.problem.ei_softening,
            config.problem.ee_softening,
            log_psi_apply_novmap,
            config.vmc,
        )
        local_energy_vmap = jax.jit(
            jax.vmap(
                jax.vmap(local_energy_fn, in_axes=(None, None, 0)),
                in_axes=(None, 0, 0),
            )
        )
    else:
        local_energy_vmap = None

    trace_rows = []
    dmin_bank = {}

    for variant in variants:
        print(f"\n========== running variant: {variant} ==========")

        key, seeded_elec = _make_seeded_elec_position(
            key=key,
            elec_position=data["walker_data"]["elec_position"],
            atoms_position=data["atoms_position"],
            geom_indices=geom_indices,
            H_idx=H_idx,
            n_up=n_up,
            single_nspins=single_nspins,
            dtype=dtype_to_use,
            mode=variant,
            zeta=zeta_H,
        )

        diag_data = _rebuild_data_with_new_electrons(
            log_psi_apply_vmap_two_dim=log_psi_apply_vmap_two_dim,
            config=config,
            params=params,
            old_data=data,
            new_elec_position=seeded_elec,
            dtype=dtype_to_use,
        )

        last_accept_mean = np.nan
        for macro_step in range(n_macro_steps + 1):
            atoms_now_np = _device_to_np(diag_data["atoms_position"])
            elec_now_np = _device_to_np(diag_data["walker_data"]["elec_position"])

            obs_rows, dmin_samples = _compute_H_observables(
                elec_now_np,
                atoms_now_np,
                geom_indices,
                H_idx,
                n_up,
            )

            # Energy is expensive; compute every energy_every macro steps after burn-in,
            # and also at macro_step=0 to see the seeded initial sector.
            energy_by_geom = {}
            do_energy = (
                compute_energy
                and (
                    macro_step == 0
                    or (
                        macro_step >= burn_in_macro
                        and macro_step % energy_every == 0
                    )
                )
            )

            if do_energy:
                local_es = local_energy_vmap(
                    params,
                    diag_data["atoms_position"],
                    diag_data["walker_data"]["elec_position"],
                )
                local_es_np = _device_to_np(local_es)
                energy_by_geom = _energy_rows(local_es_np, geom_indices)

            for row in obs_rows:
                g = int(row["geom"])
                out = dict(row)
                out["variant"] = variant
                out["macro_step"] = int(macro_step)
                out["accept_mean"] = float(last_accept_mean)
                out["test_nwalkers"] = int(test_args.test_nwalkers)

                if g in energy_by_geom:
                    out.update(energy_by_geom[g])

                trace_rows.append(out)

            if macro_step >= burn_in_macro and macro_step % thin_macro == 0:
                for g, arr in dmin_samples.items():
                    dmin_bank.setdefault((variant, int(g)), []).append(arr)

            if macro_step % 10 == 0:
                msg = [f"{variant} step {macro_step:04d}"]
                for row in obs_rows:
                    g = int(row["geom"])
                    extra = ""
                    if g in energy_by_geom:
                        extra = f", E={energy_by_geom[g]['E_mean']:.8f}"
                    msg.append(
                        f"g{g}: nH={row['nH_mean']:.3f}, "
                        f"up={row['nH_up_mean']:.3f}, "
                        f"dn={row['nH_down_mean']:.3f}, "
                        f"d50={row['dmin_p50']:.3f}"
                        f"{extra}"
                    )
                print(" | ".join(msg))

            if macro_step < n_macro_steps:
                accept, diag_data, key = walker_fn(params, diag_data, key)
                last_accept_mean = float(np.mean(_device_to_np(accept)))

    # Save CSV
    csv_path = os.path.join(outdir, "trace.csv")
    _save_trace_csv(trace_rows, csv_path)

    # Save dmin raw samples
    npz_dict = {}
    for (variant, g), chunks in dmin_bank.items():
        if len(chunks) > 0:
            npz_dict[f"{variant}_geom{g}_dmin"] = np.concatenate(chunks, axis=0)
    np.savez_compressed(os.path.join(outdir, "dmin_samples.npz"), **npz_dict)

    # Save plots
    _plot_mcmc_diagnostics(trace_rows, dmin_bank, outdir, geom_indices)

    print("\n===== MCMC H diagnostics finished =====")
    print("saved trace:", csv_path)
    print("saved plots:", outdir)


if __name__ == "__main__":
    test_mcmc()