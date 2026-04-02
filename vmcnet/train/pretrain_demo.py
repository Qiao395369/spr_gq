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
import matplotlib.pyplot as plt
from typing import Callable, Optional, Sequence, Tuple, Union
import vmcnet.gaoqiao.network_blocks as network_blocks
import chex
import copy

# import ferminet_constants as constants
# import ferminet_mcmc as mcmc
# from ferminet import networks
from vmcnet.gaoqiao.lapnet import scf
from vmcnet.gaoqiao.lapnet import system
import jax
from jax import numpy as jnp
import kfac_jax
import numpy as np
import optax
import pyscf
import vmcnet.utils as utils
import logging
import vmcnet.mcmc as mcmc
import time
import functools
import sys
def get_hf(molecule: Optional[Sequence[system.Atom]] = None,
           nspins: Optional[Tuple[int, int]] = None,
           basis: Optional[str] = 'sto-3g',
           pyscf_mol: Optional[pyscf.gto.Mole] = None,
           restricted: Optional[bool] = False) -> scf.Scf:
  """Returns an Scf object with the Hartree-Fock solution to the system.

  Args:
    molecule: the molecule in internal format.
    nspins: tuple with number of spin up and spin down electrons.
    basis: basis set to use in Hartree-Fock calculation.
    pyscf_mol: pyscf Mole object defining the molecule. If supplied,
      molecule, nspins and basis are ignored.
    restricted: If true, perform a restricted Hartree-Fock calculation,
      otherwise perform an unrestricted Hartree-Fock calculation.
  """
  if pyscf_mol:
    scf_approx = scf.Scf(pyscf_mol=pyscf_mol, restricted=restricted)
  else:
    scf_approx = scf.Scf(
        molecule, nelectrons=nspins, basis=basis, restricted=restricted)
  scf_approx.run()
  return scf_approx



def jax_eval_orbitals(scf_approx: scf.Scf, pos: Union[np.ndarray, jnp.ndarray],
                  ) -> Tuple[np.ndarray, np.ndarray]:
  """Evaluates SCF orbitals from PySCF at a set of positions.

  Args:
    scf_approx: an scf.Scf object that contains the result of a PySCF
      calculation.
    pos: an array of electron positions to evaluate the orbitals at, of shape
      (nelec, 3), where the leading dimensions are arbitrary, nelec is the
      number of electrons and the spin up electrons are ordered before the spin
      down electrons.
    nspins: tuple with number of spin up and spin down electrons.

  Returns:
    tuple with matrices of orbitals for spin up and spin down electrons, with
    the same leading dimensions as in pos.
  """
  if not isinstance(pos, jnp.ndarray):  # works even with JAX array
    try:
      pos = pos.copy()
    except AttributeError as exc:
      raise ValueError('Input must be either NumPy or JAX array.') from exc
#   logging.info(f"pos.shape:{pos.shape}")
  pos = pos.reshape((-1))
  alpha_spin, beta_spin = scf_approx.jax_scf_(pos)
#   logging.info(f"alpha_spin.shape:{alpha_spin.shape}")
#   logging.info(f"alpha_spin:{alpha_spin}")
  return alpha_spin, beta_spin


def eval_slater(scf_approx: scf.Scf, pos: Union[jnp.ndarray, np.ndarray],
                ) -> Tuple[np.ndarray, np.ndarray]:
  """Evaluates the Slater determinant.

  Args:
    scf_approx: an object that contains the result of a PySCF calculation.
    pos: an array of electron positions to evaluate the orbitals at.
    nspins: tuple with number of spin up and spin down electrons.

  Returns:
    tuple with sign and log absolute value of Slater determinant.
  """
  matrices = jax_eval_orbitals(scf_approx, pos)
  slogdets = [jnp.linalg.slogdet(elem) for elem in matrices]
  sign_alpha, sign_beta = [elem[0] for elem in slogdets]
  log_abs_wf_alpha, log_abs_wf_beta = [elem[1] for elem in slogdets]
  log_abs_slater_determinant = log_abs_wf_alpha + log_abs_wf_beta
  sign = sign_alpha * sign_beta
  return sign, log_abs_slater_determinant


def make_pretrain_step(
    net_orbitals_vmap,
    optimizer_update: optax.TransformUpdateFn,
    scf_approx,
    update_data_fn,
    nspins: Tuple[int, int],
    apply_pmap: bool = False,
    loss_mode: str = 'fulldet',
    eps: float = 1e-6,
    offblock_lambda: float = 0.0,
):
  
  if isinstance(scf_approx, list):
    print("pretrain list length:",len(scf_approx))
  else:
    raise ValueError("scf_approx should be a list")
  def hf_full_orbitals(scf_instance,x):
    res_ = jax_eval_orbitals(scf_instance, x)
    # logging.info(f"res_:{res_[0]}")
    # logging.info(f"res_.shape:{res_[0].shape}")
    res = create_full_det(res_)
    return res

  hf_orbital = [jax.vmap(functools.partial(hf_full_orbitals,scf_instance=scf),in_axes=0) for scf in scf_approx]

  def loss_fn(params, data):
    xp = data["atoms_position"]
    xe = data["walker_data"]["elec_position"]    #(W,B,ne,3)
    # logging.info(f"xe:{xe}")
    assert len(hf_orbital) == xe.shape[0]
    target_ = [hf(x=pos_) for hf, pos_ in zip(hf_orbital, xe)]  #[(B,ne,ne) ,...]
    target = jnp.stack(target_,axis=0)   #(W, B, ne, ne)
    # logging.info(f"target:{target}")
    orbitals = net_orbitals_vmap(params,xp,xe) #(W, B, ndet, nele, nele)
    result = compute_pretrain_loss(
                                    target,
                                    orbitals,
                                    nspins=nspins,
                                    loss_mode=loss_mode,
                                    eps=eps,
                                    offblock_lambda=offblock_lambda,
                                    )
    return result

  def pretrain_step(data, params, state):
    val_and_grad = jax.value_and_grad(loss_fn, argnums=0)
    loss_val, grad = val_and_grad(params, data)

    grad_norm = tree_l2_norm(grad)
    grad_max = tree_max_abs(grad)
    param_max = tree_max_abs(params)

    loss_val = utils.distribute.pmean_if_pmap(loss_val)
    grad = utils.distribute.pmean_if_pmap(grad)

    updates, state = optimizer_update(grad, state, params)
    update_max = tree_max_abs(updates)

    params = optax.apply_updates(params, updates)
    # data = update_data_fn(data, params)

    return data, params, state, loss_val, grad_norm, grad_max, param_max, update_max
  
  if not apply_pmap:
    return jax.jit(pretrain_step)
    # return loss_fn
  else:
    pmapped_pretrain_step = utils.distribute.pmap(pretrain_step)
    def pmapped_pretrain_step_with_single_loss_val(data, params, state):
      data, params, state, loss_val = pmapped_pretrain_step(data, params, state)
      loss_val = utils.distribute.get_first(loss_val)
      return data, params, state, loss_val 
    return pmapped_pretrain_step_with_single_loss_val

def pretrain_hartree_fock(
    params,
    data,
    net_orbitals_vmap,
    energy_and_statistics_fn,
    pretrain_walker_fn,    #hf波函数mcmc
    pretrain_burn_step,
    pretrain_nburn,
    update_data_fn,
    walker_fn,      #网络波函数mcmc
    burning_step,  #网络波函数burn
    key: chex.PRNGKey,
    nspins: Tuple[int, int],
    scf_approx: scf.Scf,
    iterations: int = 1000,
    optim: str='adam',
    apply_pmap: bool=False,
    loss_mode: str='fulldet',
    eps: float=1e-6,
    offblock_lambda: float=0.0,
):
  if optim == 'adam':
    optimizer = optax.adam(3.e-4)
  elif optim == 'lamb':
    optimizer = optax.lamb(1e-3)
  else:
    raise NotImplementedError
#   logging.info("make pretrain step.")
  pretrain_step = make_pretrain_step(
      net_orbitals_vmap,
      optimizer.update,
      scf_approx=scf_approx,
      update_data_fn=update_data_fn,
      nspins=nspins,
      apply_pmap=apply_pmap,
      loss_mode=loss_mode,
      eps=eps,
      offblock_lambda=offblock_lambda,
  )

  if apply_pmap:
    energy_and_statistics_fn = utils.distribute.pmap(energy_and_statistics_fn)
    optimizer_init = utils.distribute.pmap(optimizer.init)
  else :
    energy_and_statistics_fn = jax.jit(energy_and_statistics_fn)
    optimizer_init = optimizer.init

  opt_state = optimizer_init(params)
  
#   data_eval_init = copy.deepcopy(data)
#   start_time = time.time()
#   saveds = []
#   plot_titles = []
#   xes=[]
#   xe_titles=[]
#   xes_params=[]
#   xes_params_titles=[]

  

#   key, saved, mean_acc, data_eval = evaluate_current_params_samples(params,data_eval_init,key,pretrain_walker_fn,pretrain_burn_step,apply_pmap)
#   logging.info(f"HF finished, mean_acc:{mean_acc}")
#   saveds.append(saved)
#   plot_titles.append(f"HF")
#   xes.append(data_eval)
#   xe_titles.append("HF")

#   key, saved, mean_acc, data_eval = evaluate_current_params_samples(params,data_eval_init,key,walker_fn,burning_step,apply_pmap)
#   logging.info(f"gq_0 finished, mean_acc:{mean_acc}")
#   saveds.append(saved)
#   plot_titles.append(f"gq_0")
#   xes.append(data_eval)
#   xe_titles.append("gq")

#   xes.append(data["walker_data"]["elec_position"])
#   xe_titles.append("initial")

  data, key = mcmc.metropolis.burn_data(pretrain_burn_step, pretrain_nburn, params, data, key)
#   data, key = mcmc.metropolis.burn_data(burning_step, pretrain_nburn, params, data, key)
#   xes.append(data["walker_data"]["elec_position"])
#   xe_titles.append("HF burn")

#   data_eval = evaluate_current_params_pos(params,data_eval_init,key,burning_step,apply_pmap)
#   xes_params.append(data_eval)
#   xes_params_titles.append(f"params initial")

  
  for t in range(iterations):
    accept_ratio, data, key = pretrain_walker_fn(params, data, key)
    # accept_ratio, data, key = walker_fn(params, data, key)
    data, params, opt_state, loss ,grad_norm, grad_max, param_max, update_max = pretrain_step(data, params, opt_state)
    if (t+1) % 100 == 0 or t==0:
    #   logging.info(f"iter={t+1:05d}, loss={loss:g} ")
      logging.info(f"iter={t+1:05d}, loss={loss:g}, acc_r: {accept_ratio}, std_move={data['move_metadata']['std_move']} ")

    # if (t+1) % 1000 == 0 :
    #   key, saved, mean_acc, data_eval = evaluate_current_params_samples(params,data_eval_init,key,walker_fn,burning_step,apply_pmap)
    #   logging.info(f"samples finished, mean_acc:{mean_acc}")
    #   saveds.append(saved)
    #   plot_titles.append(f"HF{t+1}")

    #   xes.append(data["walker_data"]["elec_position"])
    #   xe_titles.append(f"HF{t+1}")
    #   data_eval = evaluate_current_params_pos(params,data_eval_init,key,burning_step,apply_pmap)
    #   xes_params.append(data_eval)
    #   xes_params_titles.append(f"params{t+1}")
    


  data = update_data_fn(data, params)
#   data, key = mcmc.metropolis.burn_data(burning_step, (iterations) , params, data_eval_init, key, apply_pmap)

#   for t in range(10):
    # accept_ratio, data, key = walker_fn(params, data, key)
#     data, params, opt_state, loss ,grad_norm, grad_max, param_max, update_max = pretrain_step(data, params, opt_state)
#     if (t+1) % 100 == 0 or t==0 :
    # logging.info(f"iter={t+1:05d}, loss={loss:g}, acc_r: {accept_ratio}, std_move={data['move_metadata']['std_move']}, ")
    # if (t+1) % 1000 == 0:
#       key, saved, mean_acc = evaluate_current_params_samples(params,data_eval_init,key,walker_fn,burning_step,apply_pmap)
#       logging.info(f"samples finished, mean_acc:{mean_acc}")
#       saveds.append(saved)
#       plot_titles.append(f"Net{t+1}")
    #   xes.append(data["walker_data"]["elec_position"])
    #   xe_titles.append(f"net{t+1}")
#   plot_xe(xes,xe_titles)
#   plot_xe(xes_params,xes_params_titles)
#   stacked = [np.asarray(saved).reshape(-1, 3) for saved in saveds]
#   plot_density_compare(
#     data["atoms_position"][0],
#     stacked,
#     plane="xz",
#     bins=200,
#     titles=plot_titles,
# ) 
  

  return params, data, key

def evaluate_current_params_samples(
    params,
    data_eval_init,
    key,
    walker_fn,
    burning_step,
    apply_pmap,
    n_burn=1000,
    n_sample_steps=800,
    sample_stride=1,
):
    data_eval = copy.deepcopy(data_eval_init)
    data_eval, key = mcmc.metropolis.burn_data(burning_step,n_burn,params,data_eval,key,apply_pmap,)

    saved = []
    accs = []
    for t in range(n_sample_steps):
        accept_ratio, data_eval, key = walker_fn(params, data_eval, key)
        accs.append(np.array(accept_ratio))
        if t % sample_stride == 0:
            saved.append(np.array(data_eval["walker_data"]["elec_position"]))
    saved = np.stack(saved, axis=0)   # (T, W, B, ne, 3) or similar
    mean_acc = float(np.mean(np.asarray(accs)))

    return key, saved, mean_acc, data_eval["walker_data"]["elec_position"]

def evaluate_current_params_pos(
    params,
    data_eval_init,
    key,
    burning_step,
    apply_pmap,
    n_burn=1000,
):
    data_eval = copy.deepcopy(data_eval_init)
    data_eval, key = mcmc.metropolis.burn_data(burning_step,n_burn,params,data_eval,key,apply_pmap,)
    return data_eval["walker_data"]["elec_position"]

def plot_density_compare(ion_pos, stacked, plane="xz", bins=200,titles=[]):
    def select_plane(samples, plane):
        if plane == "xy":
            return samples[:, 0], samples[:, 1], "x", "y"
        elif plane == "xz":
            return samples[:, 0], samples[:, 2], "x", "z"
        elif plane == "yz":
            return samples[:, 1], samples[:, 2], "y", "z"
        else:
            raise ValueError(plane)
        
    nplots = len(titles)
    fig, axes = plt.subplots(1, nplots, figsize=(6 * nplots, 4))

    for ax, samples, title in zip(axes, stacked, titles):
        a, b, xlabel, ylabel = select_plane(samples, plane)
        h = ax.hist2d(a, b, bins=bins, density=True, cmap="magma")
        ax.set_ylim(-3,3)
        ax.set_xlim(-3,3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    
    fig.colorbar(h[3], ax=axes.ravel().tolist(), label="density")
    # plt.tight_layout()
    # ax.legend(loc="best")
    plt.show()






def make_HF_ansatz(scf_approx: scf.Scf):
  """ For drawing samples from the HF orbitals,
      HF_sampling evaluates SCF orbitals from PySCF at a set of positions and computes corresponding Slater determinants.

  Args:
    scf_approx: an scf.Scf object that contains the result of a PySCF calculation.
    electrons: tuple with number of spin up and spin down electrons.
  """
  sampling_func = functools.partial(eval_slater, scf_approx=scf_approx)

  def HF_sampling(pos):
    """
      Args:
        params: unused
        data: MCMC configurations.

      Returns: ln |psi|
    """
    return sampling_func(pos=pos)[1]

#  batch_sampling = jax.vmap(HF_sampling, in_axes=(None, 0), out_axes=0)
#  return batch_sampling
  return HF_sampling

def create_full_det(x):
    """
    x: ((...,n_up,n_up),(...,n_dn,n_dn))
    """
    dims = x[0].shape[:-2]
    n_up = x[0].shape[-2]
    n_dn = x[1].shape[-2]
    det_up = jnp.concatenate((x[0], jnp.zeros(dims + (n_up, n_dn), dtype=x[0].dtype)), axis=-1)
    det_dn = jnp.concatenate((jnp.zeros(dims + (n_dn, n_up), dtype=x[1].dtype), x[1]), axis=-1)
    return jnp.concatenate((det_up, det_dn),axis=-2)

def split_full_det_blocks(phi_full, nspins):
    n_up, n_dn = nspins
    up_block = phi_full[..., :n_up, :n_up]
    dn_block = phi_full[..., n_up:n_up + n_dn, n_up:n_up + n_dn]
    off_updn = phi_full[..., :n_up, n_up:n_up + n_dn]
    off_dnup = phi_full[..., n_up:n_up + n_dn, :n_up]
    return up_block, dn_block, off_updn, off_dnup

# def orthonormalize_columns(phi, eps=1e-4):
#     phi = jnp.nan_to_num(phi, nan=0.0, posinf=1e6, neginf=-1e6)
#     gram = jnp.einsum('...pi,...pj->...ij', phi, phi)
#     eye = jnp.eye(gram.shape[-1], dtype=phi.dtype)
#     gram = jnp.nan_to_num(gram, nan=0.0, posinf=1e12, neginf=-1e12)
#     chol = jnp.linalg.cholesky(gram + eps * eye)
#     q_t = jax.scipy.linalg.solve_triangular(
#         chol, jnp.swapaxes(phi, -1, -2), lower=True
#     )
#     return jnp.swapaxes(q_t, -1, -2)

def orthonormalize_columns(phi,eps):
    q, _ = jnp.linalg.qr(phi, mode='reduced')
    return q


def occupied_subspace_loss(phi_ref, phi_net, eps=1e-6):
    q_ref = orthonormalize_columns(phi_ref, eps)
    q_net = orthonormalize_columns(phi_net, eps)
    overlap = jnp.einsum('...pi,...pj->...ij', q_ref, q_net)
    # print("overlap:",overlap.shape)
    fro2 = jnp.sum(overlap ** 2, axis=(-2, -1))
    nocc = phi_ref.shape[-1]
    return jnp.mean(nocc - fro2)

def fulldet_subspace_loss(target_full, orbitals_full, eps=1e-6):
    if target_full.ndim == orbitals_full.ndim - 1:
        orbitals_full = jnp.mean(orbitals_full,axis=-3)
    W,B,ne,no = orbitals_full.shape
    orbitals_full = orbitals_full.reshape((W,B*ne,no))
    target_full = target_full.reshape((W,B*ne,no))
    return occupied_subspace_loss(target_full, orbitals_full, eps=eps)

def blockdiag_subspace_loss(target_full, orbitals_full, nspins, eps=1e-6, offblock_lambda=0.0):
    if target_full.ndim == orbitals_full.ndim - 1:
        orbitals_full = jnp.mean(orbitals_full,axis=-3)
    tgt_up, tgt_dn, _, _ = split_full_det_blocks(target_full, nspins)
    net_up, net_dn, net_off_updn, net_off_dnup = split_full_det_blocks(orbitals_full, nspins)
    loss_up = occupied_subspace_loss(tgt_up, net_up, eps=eps)
    loss_dn = occupied_subspace_loss(tgt_dn, net_dn, eps=eps)
    loss = loss_up + loss_dn
    if offblock_lambda > 0.0:
        offblock_penalty = jnp.mean(net_off_updn ** 2) + jnp.mean(net_off_dnup ** 2)
        loss = loss + offblock_lambda * offblock_penalty
    return loss

def compute_pretrain_loss(target_full, orbitals_full, nspins,
                          loss_mode='fulldet_subspace',
                          eps=1e-6,
                          offblock_lambda=0.1):
    if loss_mode == 'fulldet':
        return fulldet_subspace_loss(target_full, orbitals_full, eps=eps)
    if loss_mode == 'blockdiag':
        return blockdiag_subspace_loss(
            target_full,
            orbitals_full,
            nspins=nspins,
            eps=eps,
            offblock_lambda=offblock_lambda,
        )
    if loss_mode == 'l1':
        if target_full.ndim == orbitals_full.ndim - 1:
            orbitals_full = jnp.mean(orbitals_full,axis=-3)
        return jnp.mean(jnp.abs(target_full - orbitals_full))
    if loss_mode == 'l2':
        if target_full.ndim == orbitals_full.ndim - 1:
            orbitals_full = jnp.mean(orbitals_full,axis=-3)
        return jnp.mean((target_full - orbitals_full)**2)
    
    raise ValueError(
        f"Unknown loss_mode={loss_mode}. "
        "Choose from {'fulldet', 'blockdiag', 'l1', 'l2' }"
    )





def tree_l2_norm(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.sqrt(sum(jnp.sum(x * x) for x in leaves if x is not None))

def tree_max_abs(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.max(jnp.array([jnp.max(jnp.abs(x)) for x in leaves if x is not None]))




def plot_xe(xes, titles, plane="xz"):
    def select_plane(samples, plane):
        if plane == "xy":
            return samples[:, 0], samples[:, 1], "x", "y"
        elif plane == "xz":
            return samples[:, 0], samples[:, 2], "x", "z"
        elif plane == "yz":
            return samples[:, 1], samples[:, 2], "y", "z"
        else:
            raise ValueError(f"Invalid plane: {plane}")

    nplots = len(xes)
    fig, axes = plt.subplots(1, nplots, figsize=(6 * nplots, 4))

    for ax, xe, title in zip(axes, xes, titles):
        positions = xe.reshape(-1, 3)  # Flatten to (batch * ne, 3)
        a, b, xlabel, ylabel = select_plane(positions, plane)
        ax.scatter(a, b, s=1, alpha=0.5, cmap="magma")  # Scatter plot, 's' is the point size
        ax.set_ylim(-3, 3)  # Adjust limits as needed
        ax.set_xlim(-3, 3)  # Adjust limits as needed
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    plt.tight_layout()  # Ensures no overlap in subplots
    plt.show()
