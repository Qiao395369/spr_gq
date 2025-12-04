# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for pretraining and importing PySCF models."""

from typing import Callable, Optional, Sequence, Tuple, Union

import chex
# import ferminet_constants as constants
# import ferminet_mcmc as mcmc
# from ferminet import networks
from vmcnet.gaoqiao.fermi_ferminet import fermi_scf
from vmcnet.gaoqiao.fermi_ferminet import fermi_system
import jax
from jax import numpy as jnp
import kfac_jax
import numpy as np
import optax
import pyscf
import vmcnet.utils as utils
import logging
import vmcnet.mcmc as mcmc

def get_hf(molecule: Optional[Sequence[fermi_system.Atom]] = None,
           nspins: Optional[Tuple[int, int]] = None,
           basis: Optional[str] = 'sto-3g',
           pyscf_mol: Optional[pyscf.gto.Mole] = None,
           restricted: Optional[bool] = False,
           states: int = 0) -> fermi_scf.Scf:
  """Returns an Scf object with the Hartree-Fock solution to the system.

  Args:
    molecule: the molecule in internal format.
    nspins: tuple with number of spin up and spin down electrons.
    basis: basis set to use in Hartree-Fock calculation.
    pyscf_mol: pyscf Mole object defining the molecule. If supplied,
      molecule, nspins and basis are ignored.
    restricted: If true, perform a restricted Hartree-Fock calculation,
      otherwise perform an unrestricted Hartree-Fock calculation.
    states: Number of excited states.  If nonzero, compute all single and double
      excitations of the Hartree-Fock solution and return coefficients for the
      lowest ones.
  """
  if pyscf_mol:
    scf_approx = fermi_scf.Scf(pyscf_mol=pyscf_mol, restricted=restricted)
  else:
    scf_approx = fermi_scf.Scf(
        molecule, nelectrons=nspins, basis=basis, restricted=restricted)
  scf_approx.run(excitations=max(states - 1, 0))
  return scf_approx


def eval_orbitals(scf_approx: fermi_scf.Scf, pos: Union[np.ndarray, jnp.ndarray],
                  nspins: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
  """Evaluates SCF orbitals from PySCF at a set of positions.

  Args:
    scf_approx: an scf.Scf object that contains the result of a PySCF
      calculation.
    pos: an array of electron positions to evaluate the orbitals at, of shape
      (..., nelec*3), where the leading dimensions are arbitrary, nelec is the
      number of electrons and the spin up electrons are ordered before the spin
      down electrons.
    nspins: tuple with number of spin up and spin down electrons.

  Returns:
    tuple with matrices of orbitals for spin up and spin down electrons, with
    the same leading dimensions as in pos.
  """
  if not isinstance(pos, np.ndarray):  # works even with JAX array
    try:
      pos = pos.copy()
    except AttributeError as exc:
      raise ValueError('Input must be either NumPy or JAX array.') from exc
  leading_dims = pos.shape[:-1]
  # split into separate electrons
  pos = np.reshape(pos, [-1, 3])  # (batch*nelec, 3)
  mos = scf_approx.eval_mos(pos)  # (batch*nelec, nbasis), (batch*nelec, nbasis)
  # Reshape into (batch, nelec, nbasis) for each spin channel.
  mos = [np.reshape(mo, leading_dims + (sum(nspins), -1)) for mo in mos]
  # Return (using Aufbau principle) the matrices for the occupied alpha and
  # beta orbitals. Number of alpha electrons given by nspins[0].
  alpha_spin = mos[0][..., :nspins[0], :nspins[0]]
  beta_spin = mos[1][..., nspins[0]:, :nspins[1]]
  return alpha_spin, beta_spin


def eval_slater(scf_approx: fermi_scf.Scf, pos: Union[jnp.ndarray, np.ndarray],
                nspins: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
  """Evaluates the Slater determinant.

  Args:
    scf_approx: an object that contains the result of a PySCF calculation.
    pos: an array of electron positions to evaluate the orbitals at.
    nspins: tuple with number of spin up and spin down electrons.

  Returns:
    tuple with sign and log absolute value of Slater determinant.
  """
  matrices = eval_orbitals(scf_approx, pos, nspins)
  slogdets = [np.linalg.slogdet(elem) for elem in matrices]
  sign_alpha, sign_beta = [elem[0] for elem in slogdets]
  log_abs_wf_alpha, log_abs_wf_beta = [elem[1] for elem in slogdets]
  log_abs_slater_determinant = log_abs_wf_alpha + log_abs_wf_beta
  sign = sign_alpha * sign_beta
  return sign, log_abs_slater_determinant


# def make_pretrain_step(
#     batch_orbitals: networks.OrbitalFnLike,
#     batch_network: networks.LogFermiNetLike,
#     optimizer_update: optax.TransformUpdateFn,
#     scf_approx: scf.Scf,
#     electrons: Tuple[int, int],
#     full_det: bool = False,
#     states: int = 0,
# ):
#   """Creates function for performing one step of Hartre-Fock pretraining.

#   Args:
#     batch_orbitals: callable with signature f(params, data), which given network
#       parameters and a batch of electron positions, returns the orbitals in the
#       network evaluated at those positions.
#     batch_network: callable with signature f(params, data), which given network
#       parameters and a batch of electron positions, returns the log of the
#       magnitude of the (wavefunction) network  evaluated at those positions.
#     optimizer_update: callable for transforming the gradients into an update (ie
#       conforms to the optax API).
#     scf_approx: an scf.Scf object that contains the result of a PySCF
#       calculation.
#     electrons: number of spin-up and spin-down electrons.
#     full_det: If true, evaluate all electrons in a single determinant.
#       Otherwise, evaluate products of alpha- and beta-spin determinants.
#     states: Number of excited states, if not 0.

#   Returns:
#     Callable for performing a single pretraining optimisation step.
#   """

#   def pretrain_step(data, params, state, key, logprob):
#     """One iteration of pretraining to match HF."""

#     cnorm = lambda x, y: (x - y) * jnp.conj(x - y)  # complex norm
#     def loss_fn(
#         params: networks.ParamTree,
#         data: networks.FermiNetData,
#     ):
#       pos = data.positions
#       spins = data.spins
#       if states:
#         # Make vmap-ed versions of eval_orbitals and batch_orbitals over the
#         # states dimension.
#         # (batch, states, nelec*ndim)
#         pos = jnp.reshape(pos, pos.shape[:-1] + (states, -1))
#         # (batch, states, nelec)
#         spins = jnp.reshape(spins, spins.shape[:-1] + (states, -1))

#         scf_orbitals = jax.vmap(
#             scf_approx.eval_orbitals, in_axes=(-2, None), out_axes=-4
#         )

#         def net_orbitals(params, pos, spins, atoms, charges):
#           vmapped_orbitals = jax.vmap(
#               batch_orbitals, in_axes=(None, -2, -2, None, None), out_axes=-4
#           )
#           # Dimensions of result are
#           # [(batch, states, ndet*states, nelec, nelec)]
#           result = vmapped_orbitals(params, pos, spins, atoms, charges)
#           result = [
#               jnp.reshape(r, r.shape[:-3] + (states, -1) + r.shape[-2:])
#               for r in result
#           ]
#           result = [jnp.transpose(r, (0, 3, 1, 2, 4, 5)) for r in result]
#           # We draw distinct samples for each excited state (electron
#           # configuration), and then evaluate each state within each sample.
#           # Output dimensions are:
#           # (batch, det, electron configuration,
#           # excited state, electron, orbital)
#           return result

#       else:
#         scf_orbitals = scf_approx.eval_orbitals
#         net_orbitals = batch_orbitals

#       target = scf_orbitals(pos, electrons)
#       orbitals = net_orbitals(params, pos, spins, data.atoms, data.charges)
#       if full_det:
#         dims = target[0].shape[:-2]  # (batch) or (batch, states).
#         na = target[0].shape[-2]
#         nb = target[1].shape[-2]
#         target = jnp.concatenate(
#             (
#                 jnp.concatenate(
#                     (target[0], jnp.zeros(dims + (na, nb))), axis=-1),
#                 jnp.concatenate(
#                     (jnp.zeros(dims + (nb, na)), target[1]), axis=-1),
#             ),
#             axis=-2,
#         )
#         result = jnp.mean(cnorm(target[:, None, ...], orbitals[0])).real
#       else:
#         result = jnp.array([
#             jnp.mean(cnorm(t[:, None, ...], o)).real
#             for t, o in zip(target, orbitals)
#         ]).sum()
#       return constants.pmean(result)

#     val_and_grad = jax.value_and_grad(loss_fn, argnums=0)
#     loss_val, search_direction = val_and_grad(params, data)
#     search_direction = constants.pmean(search_direction)
#     updates, state = optimizer_update(search_direction, state, params)
#     params = optax.apply_updates(params, updates)
#     data, key, logprob, _ = mcmc.mh_update(params, batch_network, data, key,
#                                            logprob, 0)
#     return data, params, state, loss_val, logprob

#   return pretrain_step

# def pretrain_hartree_fock(
#     *,
#     params: networks.ParamTree,
#     positions: jnp.ndarray,
#     spins: jnp.ndarray,
#     atoms: jnp.ndarray,
#     charges: jnp.ndarray,
#     batch_network: networks.FermiNetLike,
#     batch_orbitals: networks.OrbitalFnLike,
#     network_options: networks.BaseNetworkOptions,
#     sharded_key: chex.PRNGKey,
#     electrons: Tuple[int, int],
#     scf_approx: scf.Scf,
#     iterations: int = 1000,
#     logger: Optional[Callable[[int, float], None]] = None,
#     states: int = 0,
# ):
#   """Performs training to match initialization as closely as possible to HF.

#   Args:
#     params: Network parameters.
#     positions: Electron position configurations.
#     spins: Electron spin configuration (1 for alpha electrons, -1 for beta), as
#       a 1D array. Note we always use the same spin configuration for the entire
#       batch in pretraining.
#     atoms: atom positions (batched).
#     charges: atomic charges (batched).
#     batch_network: callable with signature f(params, data), which given network
#       parameters and a batch of electron positions, returns the log of the
#       magnitude of the (wavefunction) network  evaluated at those positions.
#     batch_orbitals: callable with signature f(params, data), which given network
#       parameters and a batch of electron positions, returns the orbitals in the
#       network evaluated at those positions.
#     network_options: FermiNet network options.
#     sharded_key: JAX RNG state (sharded) per device.
#     electrons: tuple of number of electrons of each spin.
#     scf_approx: an scf.Scf object that contains the result of a PySCF
#       calculation.
#     iterations: number of pretraining iterations to perform.
#     logger: Callable with signature (step, value) which externally logs the
#       pretraining loss.
#     states: Number of excited states, if not 0.

#   Returns:
#     params, positions: Updated network parameters and MCMC configurations such
#     that the orbitals in the network closely match Hartree-Fock and the MCMC
#     configurations are drawn from the log probability of the network.
#   """
#   # Pretraining is slow on larger systems (very low GPU utilization) because the
#   # Hartree-Fock orbitals are evaluated on CPU and only on a single host.
#   # Implementing the basis set in JAX would enable using GPUs and allow
#   # eval_orbitals to be pmapped.

#   optimizer = optax.adam(3.e-4)
#   opt_state_pt = constants.pmap(optimizer.init)(params)

#   pretrain_step = make_pretrain_step(
#       batch_orbitals,
#       batch_network,
#       optimizer.update,
#       scf_approx=scf_approx,
#       electrons=electrons,
#       full_det=network_options.full_det,
#       states=states,
#   )
#   pretrain_step = constants.pmap(pretrain_step)
#   pnetwork = constants.pmap(batch_network)

#   batch_spins = jnp.tile(spins[None], [positions.shape[1], 1])
#   pmap_spins = kfac_jax.utils.replicate_all_local_devices(batch_spins)
#   data = networks.FermiNetData(
#       positions=positions, spins=pmap_spins, atoms=atoms, charges=charges
#   )
#   logprob = 2.0 * pnetwork(params, positions, pmap_spins, atoms, charges)

#   for t in range(iterations):
#     sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
#     data, params, opt_state_pt, loss, logprob = pretrain_step(
#         data, params, opt_state_pt, subkeys, logprob)
#     logging.info('Pretrain iter %05d: %g', t, loss[0])
#     if logger:
#       logger(t, loss[0])
#   return params, data.positions




##########################################################gaoqiao###################################################################

import vmcnet.gaoqiao.network_blocks as network_blocks
def get_hf_wfn(scf_approx:fermi_scf.Scf,nspins):
  scf_orbitals=scf_approx.eval_orbitals
  def hf_wfn(params, atoms, electrons):
    del params, atoms
    electrons=electrons.reshape((-1,))
    orbitals=scf_orbitals(electrons,nspins)
    # orbitals=[orbital[None,...] for orbital in orbitals]
    orbitals = [create_full_det(orbitals)]
    return network_blocks.logdet_matmul(orbitals)[1]
  return hf_wfn

def create_full_det(x):
    """
    x: ((...,n_up,n_up),(...,n_dn,n_dn))
    """
    dims = x[0].shape[:-2]
    n_up = x[0].shape[-2]
    n_dn = x[1].shape[-2]
    det_up = jnp.concatenate((x[0], jnp.zeros((dims + (n_up, n_dn)))), axis=-1)
    det_dn = jnp.concatenate((jnp.zeros((dims + (n_dn, n_up))), x[1]), axis=-1)
    return jnp.concatenate((det_up, det_dn),axis=-2)

def make_pretrain_step_gaoqiao_2(
    net_orbitals_vmap,
    optimizer_update: optax.TransformUpdateFn,
    scf_approx,
    nspins: Tuple[int, int],
    apply_pmap: bool = True,
):
  def loss_fn(params, data):
    xp = data["atoms_position"]
    xe = data["walker_data"]["elec_position"]
    pos = xe.reshape(xe.shape[:-2]+(-1,)) 
    target = scf_approx.eval_orbitals(pos,nspins)
    target_full_det = create_full_det(target)
    orbitals = net_orbitals_vmap(params,xp,xe) #(w_per_device, B, ndet, nele, nele)
    # result = jnp.mean(cnorm(target_full_det[:,:,None,...],orbitals)).real
    # result = jnp.mean((target_full_det[:,:, None, ...] - orbitals)**2)
    result = jnp.mean(jnp.abs(target_full_det[:,:, None, ...] - orbitals))

    return result

  def pretrain_step(data, params, state):
    """One iteration of pretraining to match HF."""
    val_and_grad = jax.value_and_grad(loss_fn, argnums=0)
    loss_val, grad = val_and_grad(params, data)
    loss_val = utils.distribute.pmean_if_pmap(loss_val)
    grad = utils.distribute.pmean_if_pmap(grad)
    # grad = jax.tree_util.tree_map(lambda g: jnp.clip(g, -0.001, 0.001), grad)  # 梯度裁剪
    updates, state = optimizer_update(grad, state, params)
    params = optax.apply_updates(params, updates)
    return data, params, state, loss_val
  
  if not apply_pmap:
    return jax.jit(pretrain_step)

  pmapped_pretrain_step = utils.distribute.pmap(pretrain_step)


  def pmapped_pretrain_step_with_single_loss_val(data, params, state):
      data, params, state, loss_val = pmapped_pretrain_step(data, params, state)
      loss_val = utils.distribute.get_first(loss_val)
      return data, params, state, loss_val 
  
  return pmapped_pretrain_step_with_single_loss_val

def pretrain_hartree_fock_gaoqiao_2(
    params,
    data,
    net_orbitals_vmap,
    energy_and_statistics_fn,
    pretrain_walker_fn,
    walker_fn,
    burning_step,
    key: chex.PRNGKey,
    nspins: Tuple[int, int],
    scf_approx: fermi_scf.Scf,
    iterations: int = 1000,
    optim: str='adam',
    apply_pmap: bool=True,
):
  if optim == 'adam':
    optimizer = optax.adam(1.e-3)
  elif optim == 'lamb':
    optimizer = optax.lamb(1e-2)
  else:
    raise NotImplementedError
  
  pretrain_step = make_pretrain_step_gaoqiao_2(
      net_orbitals_vmap,
      optimizer.update,
      scf_approx=scf_approx,
      nspins=nspins,
      apply_pmap=apply_pmap,
  )
  if apply_pmap:
    energy_and_statistics_fn = utils.distribute.pmap(energy_and_statistics_fn)
    optimizer_init = utils.distribute.pmap(optimizer.init)

  else :
    pretrain_step = jax.jit(pretrain_step)
    energy_and_statistics_fn = jax.jit(energy_and_statistics_fn)
    optimizer_init = optimizer.init

  opt_state = optimizer_init(params)

  for t in range(1,iterations):
    accept_ratio_0, data, key = pretrain_walker_fn(params, data, key)
    # accept_ratio_1, data_1, key = walker_fn(params, data_1, key)
    data, params, opt_state, loss = pretrain_step(data, params, opt_state)
    # loss=0
    # energy_per_w, _, _ = energy_and_statistics_fn(params, data_1["atoms_position"], data_1["walker_data"]["elec_position"])
    # Energy = jnp.mean(energy_per_w)
    # logging.info(f'Pretrain iter: {t:05d},loss: {loss:g}, E: {Energy}, acc_r_0: {accept_ratio_0}, acc_r_1: {accept_ratio_1}')
    logging.info(f'Pretrain iter: {t:05d},loss: {loss:g}')
    # logging.info(f'Pretrain iter: {t:05d}, loss: {loss:g}, acc_r: {accept_ratio}, logprob: {jnp.mean(2 * data["walker_data"]["amplitude"])}, move: {data["move_metadata"]["std_move"]}, acc_sum: {data["move_metadata"]["move_acceptance_sum"]}')

  data, key = mcmc.metropolis.burn_data(burning_step, 3000, params, data, key)
  for t in range(100):
    accept_ratio, data, key = walker_fn(params, data, key)
    data, params, opt_state, loss = pretrain_step(data, params, opt_state)
    energy_per_w, E_loc, stats = energy_and_statistics_fn(params, data["atoms_position"], data["walker_data"]["elec_position"])
    # Energy = jax.pmap(lambda x: jax.lax.pmean(jnp.mean(x),axis_name="ii"),axis_name="ii")(energy_per_w)
    Energy = jnp.mean(energy_per_w)
    logging.info(f'Pretrain iter: {t:05d}, loss: {loss}')
  return params, data, key