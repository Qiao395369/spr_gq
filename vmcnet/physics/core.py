"""Core local energy and gradient construction routines."""

from typing import Callable, Optional, Sequence, Tuple, cast
import logging
import chex
import jax
import jax.numpy as jnp
import kfac_jax

import vmcnet.utils as utils
from vmcnet.utils.typing import (
    Array,
    ArrayLike,
    P,
    ClippingFn,
    PRNGKey,
    LocalEnergyApply,
    ModelApply,
    Dict,
    Any,
)
from vmcnet.utils.distribute import PMAP_AXIS_NAME
EnergyAuxData = Dict[str, Any]
ValueGradEnergyFn = Callable[[P, Array, Array], Tuple[Array, EnergyAuxData, P]]


# def initialize_molecular_pos(
#     key: PRNGKey,
#     nchains: int,
#     ion_pos: Array,
#     ion_charges: Array,
#     nelec_total: int,
#     single_spins:Array,
#     init_width: float = 1.0,
#     dtype=chex.Numeric,
# ) -> Tuple[PRNGKey, Array]:
#     """Initialize a set of plausible initial electron positions.

#     For each chain, each electron is assigned to a random ion and then its position is
#     sampled from a normal distribution centered at that ion with diagonal covariance
#     with diagonal entries all equal to init_width.

#     If there are no more electrons than there are ions, the assignment is done without
#     replacement. If there are more electrons than ions, the assignment is done with
#     replacement, and the probability of choosing ion i is its relative charge (as a
#     fraction of the sum of the ion charges).
#     """
#     nion = len(ion_charges)
#     replace = True

#     if nelec_total <= nion:
#         replace = False

#     elecs_at_ions=[]
#     # print(ion_pos.shape)
#     assert len(ion_pos.shape)==3
#     for i in range(ion_pos.shape[0]):
#         assignments = []
#         for _ in range(nchains):
#             key, subkey = jax.random.split(key)
#             # choices = jax.random.choice(
#             #     subkey,
#             #     nion,
#             #     shape=(nelec_total,),
#             #     replace=replace,
#             #     p=ion_charges / jnp.sum(ion_charges),
#             # )

#             # 从0到6中随机选择两个不重复的数
#             choices=jnp.ones(nelec_total)
#             random_numbers = jnp.random.choice(jnp.arange(7), size=2, replace=False)

#             print(random_numbers)
#             # 随机选择两个不重复的数

#             choices=jnp.asarray([1 ,0 ,0 ,1 ,0 ,1 ,0 ,  0 ,1 ,0 ,1 ,1 ,0 ,1])
#             print(choices)
#             assignments.append(ion_pos[i][choices])
#         elecs_at_ions.append(jnp.stack(assignments, axis=0)) #[(nchains, nelec_total, 3),...]
#     elecs=jnp.stack(elecs_at_ions,axis=0) #(walker,nchain,ne,dim)
#     print(f"xe.shape:{elecs.shape}")
#     key, subkey = jax.random.split(key)
#     return key, elecs + init_width * jax.random.normal(
#         subkey, elecs.shape, dtype=dtype
#     )


def initialize_molecular_pos(
    key: PRNGKey,
    nchains: int,
    ion_pos: Array,
    ion_charges: Array,
    nelec_total: int,
    single_spins:Array,
    init_width: float = 0.2,
    dtype=chex.Numeric,
) -> Tuple[PRNGKey, Array]:
    natoms=len(ion_charges)
    walker=ion_pos.shape[0]
    assert ion_pos.shape==(walker,natoms,3)
    assert nelec_total==jnp.sum(single_spins)
    # print(single_spins)  # single_spins=[[5,2],[2,5]]
    # Assign each electron to an atom initially.
    ppp=[]
    for k in range(walker):
        electron_positions = []
        for i in range(2):
            for j in range(natoms):
                position = jnp.asarray(ion_pos[k][j])
                electron_positions.append(jnp.tile(position, single_spins[j][i]))
        electron_positions = jnp.concatenate(electron_positions)
        # print(electron_positions)
        # print(electron_positions.shape)
        ppp.append(electron_positions.reshape((-1,3)))
    ppp=jnp.stack(ppp,axis=0)
    ppp=ppp[:,None,...]
    key, subkey = jax.random.split(key)
    ppp += jax.random.normal(subkey, shape=(walker,nchains,)+ppp.shape[-2:] , dtype=dtype) * init_width
    logging.info("init_xp: %s \n " \
    "         init_xe: %s", ion_pos.shape, ppp.shape)
    return key, ppp

def combine_local_energy_terms(
    local_energy_terms: Sequence[ModelApply[P]],
) -> LocalEnergyApply[P]:
    """Combine a sequence of local energy terms by adding them.

    Args:
        local_energy_terms (Sequence): sequence of local energy terms, each with the
            signature (params, x) -> array of terms of shape (x.shape[0],)

    Returns:
        Callable: local energy function which computes the sum of the local energy
        terms. Has the signature
        (params, x) -> local energy array of shape (x.shape[0],)
    """

    def local_energy_fn(params: P, x: Array) -> Array:
        local_energy_sum = local_energy_terms[0](params, x)
        for term in local_energy_terms[1:]:
            local_energy_sum = cast(Array, local_energy_sum + term(params, x))
        return local_energy_sum

    return local_energy_fn

def allreduce_mean(x,axis):
    x=jnp.mean(x,axis)
    return jax.lax.pmean(x,axis_name=utils.distribute.PMAP_AXIS_NAME)

def get_statistics_from_other_energy(
    energy1: Array, energy2: Array, energy3: Array, energy4: Array, nan_safe: bool = True
) -> Tuple[Array, Array]:
    # if nan_safe:
    #     allreduce_mean = utils.distribute.nanmean_all_local_devices
    # else:
    #     allreduce_mean = utils.distribute.mean_all_local_devices
    energy1_mean = allreduce_mean(energy1,axis=(0,1))
    energy2_mean = allreduce_mean(energy2,axis=(0,1))
    energy3_mean = allreduce_mean(energy3,axis=(0,1))
    energy4_mean = allreduce_mean(energy4,axis=(0,1))
    return energy1_mean, energy2_mean, energy3_mean, energy4_mean

def get_statistics_from_local_energy(
    local_energies: Array, nan_safe: bool ,
) -> Tuple[Array, Array]:
    """Collectively reduce local energies to an average energy and variance.

    Args:
        local_energies (Array): local energies of shape (nchains,), possibly
            distributed across multiple devices via utils.distribute.pmap.
        nchains (int): total number of chains across all devices, used to compute a
            sample variance estimate of the local energy
        nan_safe (bool, optional): flag which controls if jnp.nanmean is used instead of
            jnp.mean. Can be set to False when debugging if trying to find the source of
            unexpected nans. Defaults to True.

    Returns:
        (chex.Numeric, chex.Numeric): local energy average, local energy (sample)
        variance
    """
    # TODO(Jeffmin) might be worth investigating the numerical stability of the XLA
    # compiled version of these two computations, since the quality of the gradients
    # is fairly crucial to the success of the algorithm
    assert len(local_energies.shape) == 2  # local_energies:(W,B)
    W, B = local_energies.shape
    ridx = jax.lax.axis_index(utils.distribute.PMAP_AXIS_NAME) 
    # ridx=0
    jax.debug.print(f"in get_statistics_from_local_energy: [replica {ridx}] local_energies shape={local_energies.shape} dtype={local_energies.dtype.name}")
    # if nan_safe:
    #     allreduce_mean = utils.distribute.nanmean_all_local_devices
    #     w_mean = jnp.nanmean
    # else:
    #  allreduce_mean = utils.distribute.mean_all_local_devices
    #     w_mean = jnp.mean
    w_mean = jnp.mean

    energy_per_w = w_mean(local_energies, axis=1, keepdims=True)  # (W,1)
    jax.debug.print(f"in get_statistics_from_local_energy: [replica {ridx}] energy_per_w shape={energy_per_w.shape} dtype={energy_per_w.dtype.name}")
    
    var_per_w = jnp.sum(jnp.square(local_energies - energy_per_w), axis=1) / jnp.maximum(B - 1, 1)  # (W,)

    variance = allreduce_mean(var_per_w, axis=0)  # ()
    jax.debug.print(f"in get_statistics_from_local_energy: [replica {ridx}] variance shape={variance.shape} dtype={variance.dtype.name}")
    
    return energy_per_w, variance 


def get_clipped_energies_and_stats(
    local_energies_noclip: Array,
    clipping_fn: Optional[ClippingFn],
    nan_safe: bool,
) -> Tuple[Array, Array, EnergyAuxData]:
    """Clip local energies if requested and return auxiliary data."""
    energy_noclip, variance_noclip = get_statistics_from_local_energy(local_energies_noclip, nan_safe=False)

    if clipping_fn is not None:
        local_energies = clipping_fn(local_energies_noclip, energy_noclip)  #local_energies_noclip:(W,B)， energy_noclip:(W,1)-->local_energies: (W,B)
        energy, variance = get_statistics_from_local_energy(local_energies, nan_safe=nan_safe)  #energy: (W,1)  variance:(1,)
    else:
        local_energies, energy, variance= local_energies_noclip, energy_noclip, variance_noclip
    
    energy_stats = dict(
        variance=variance,  #()
        energy_noclip=allreduce_mean(energy_noclip,axis=(0,1)),  #()
        variance_noclip=variance_noclip,  #()
    )

    return energy, local_energies, energy_stats


def create_value_and_grad_energy_fn(
    log_psi_apply: ModelApply[P],
    kinetic_fn,ei_potential_fn,ee_potential_fn,ii_potential_fn,
    nchains: int,
    clipping_fn: Optional[ClippingFn] = None,
    nan_safe: bool = True,
) -> ValueGradEnergyFn[P]:
    """Create a function which computes unbiased energy gradients.

    Due to the Hermiticity of the Hamiltonian, we can get an unbiased lower variance
    estimate of the gradient of the expected energy than the naive gradient of the
    mean of sampled local energies. Specifically, the gradient of the expected energy
    expect[E_L] takes the form

        2 * expect[(E_L - expect[E_L]) * (grad_psi / psi)(x)],

    where E_L is the local energy and expect[] denotes the expectation with respect to
    the distribution |psi|^2.

    Args:
        log_psi_apply (Callable): computes log|psi(x)|, where the signature of this
            function is (params, x) -> log|psi(x)|
        local_energy_fn (Callable): computes local energies Hpsi / psi. Has signature
            (params, x) -> (Hpsi / psi)(x)
        nchains (int): total number of chains across all devices, used to compute a
            sample variance estimate of the local energy
        clipping_fn (Callable, optional): post-processing function on the local energy,
            e.g. a function which clips the values to be within some multiple of the
            total variation from the median. The post-processed values are used for
            the gradient calculation, if available. Defaults to None.
        nan_safe (bool, optional): flag which controls if jnp.nanmean and jnp.nansum are
            used instead of jnp.mean and jnp.sum for the terms in the gradient
            calculation. Can be set to False when debugging if trying to find the source
            of unexpected nans. Defaults to True.

    Returns:
        Callable: function which computes the clipped energy value and gradient. Has the
        signature
            (params, x)
            -> ((expected_energy, auxiliary_energy_data), grad_energy),
        where auxiliary_energy_data is the tuple
        (expected_variance, local_energies, unclipped_energy, unclipped_variance)
    """
    # mean_grad_fn = utils.distribute.get_mean_over_first_axis_fn(nan_safe=nan_safe)
    mean_grad_fn = utils.distribute.get_mean_over_first_and_second_axis_fn(nan_safe=nan_safe)

    def standard_estimator_forward(
        params: P,
        atoms_positions: Array,
        positions: Array,
        centered_local_energies: Array,
    ) -> ArrayLike:
        '''
        atoms_position:(W,natom,dim)
        positions(W,B,nele,dim)
        centerd_local_energies:(W,B)
        '''
        log_psi = log_psi_apply(params, atoms_positions, positions)  # log_psi:(W,B)
        kfac_jax.register_normal_predictive_distribution(log_psi[:, None])
        # NOTE: for the generic gradient estimator case it may be important to include
        # the (nchains / nchains -1) factor here to make sure the standard and generic
        # gradient terms aren't mismatched by a slight scale factor.
        return (
            2.0
            * nchains
            / (nchains - 1)
            * mean_grad_fn(centered_local_energies * log_psi)
        )  # shape:()

    def get_standard_contribution(local_energies_noclip, params, atoms_positions, positions):
        '''
        local_energies_noclip:(W,B)
        atoms_position:(W,natom,dim)
        positions(W,B,nele,dim)
        '''
        energy, local_energies, stats = get_clipped_energies_and_stats(
            local_energies_noclip, clipping_fn, nan_safe
        )  # energy:(W,1), local_energies:(W,B)
        centered_local_energies = local_energies - energy  #(W,B)
        grad_E = jax.grad(standard_estimator_forward, argnums=0)(
            params, atoms_positions, positions, centered_local_energies
        )  # grad_E has the same shape as params.
        return energy, stats, grad_E

    def energy_val_and_grad(params,atoms_positions, positions):
        '''
        atoms_positions:(W,natom,dim)
        positions:(W,B,nele,dim)
        '''
        kinetic=jax.vmap(jax.vmap(kinetic_fn, in_axes=(None,None,0)),in_axes=(None,0,0))(params,atoms_positions, positions) #(W,B)
        ei_potential= jax.vmap(jax.vmap(ei_potential_fn,in_axes=(None,None,0)),in_axes=(None,0,0))(params,atoms_positions, positions)  #(W,B)
        ee_potential=jax.vmap(jax.vmap(ee_potential_fn, in_axes=(None,None,0)),in_axes=(None,0,0))(params,atoms_positions,positions)#(W,B)
        ii_potential=jax.vmap(jax.vmap(ii_potential_fn, in_axes=(None,None,0)),in_axes=(None,0,0))(params,atoms_positions,positions)#(W,B)

        local_energies_noclip=kinetic+ei_potential+ee_potential+ii_potential  #(W,B)

        kinetic,ei_potential,ee_potential,ii_potential = get_statistics_from_other_energy(kinetic,ei_potential,ee_potential,ii_potential, nan_safe=nan_safe) #()

        energy, stats, grad_E = get_standard_contribution(
            local_energies_noclip, params, atoms_positions, positions
        )
        multi_energy = energy.reshape((-1))
        stats.update({"kinetic": kinetic, "ei_potential": ei_potential ,"ee_potential":ee_potential,"ii_potential":ii_potential,"multi_energy":multi_energy})
        return energy, stats, grad_E

    return energy_val_and_grad


def create_energy_and_statistics_fn(
    kinetic_fn,ei_potential_fn,ee_potential_fn,ii_potential_fn,
    debug: str,
    clipping_fn: Optional[ClippingFn] = None,
    nan_safe: bool = True,
) -> ValueGradEnergyFn[P]:
    """Create a function which computes energies and associated statistics.

    Args:
        log_psi_apply (Callable): computes log|psi(x)|, where the signature of this
            function is (params, x) -> log|psi(x)|
        local_energy_fn (Callable): computes local energies Hpsi / psi. Has signature
            (params, x) -> (Hpsi / psi)(x)
        nchains (int): total number of chains across all devices, used to compute a
            sample variance estimate of the local energy
        clipping_fn (Callable, optional): post-processing function on the local energy,
            e.g. a function which clips the values to be within some multiple of the
            total variation from the median. The post-processed values are used for
            the gradient calculation, if available. Defaults to None.
        nan_safe (bool, optional): flag which controls if jnp.nanmean and jnp.nansum are
            used instead of jnp.mean and jnp.sum for the terms in the gradient
            calculation. Can be set to False when debugging if trying to find the source
            of unexpected nans. Defaults to True.

    Returns:
        Callable: function which computes the clipped energy and associated statistics.
        Has the signature
            (params, positions)
            -> (expected_energy, auxiliary_energy_data)
        where auxiliary_energy_data is the tuple
        (expected_variance, local_energies, unclipped_energy, unclipped_variance, centered_local_energies)
    """

    def energy_and_statistics(params, atoms_positions, positions):
        """
        atoms_positions: (W, natom, dim)
        positions:       (W, B, nele, dim)
        Returns:
            energy_per_w: (W, 1)
            E_loc:        (W, B)
            stats: dict(...)
        """
        jax.debug.print(
            f"energy_and_statistics: atoms_positions shape={atoms_positions.shape} dtype={atoms_positions.dtype.name} || positions shape={positions.shape} dtype={positions.dtype.name}"
        )

        # --- Compute per-replica tensors (shape (W,B)) ---
        kinetic = jax.vmap(jax.vmap(kinetic_fn,      in_axes=(None, None, 0)), in_axes=(None, 0, 0))(params, atoms_positions, positions)
        ei_pot  = jax.vmap(jax.vmap(ei_potential_fn, in_axes=(None, None, 0)), in_axes=(None, 0, 0))(params, atoms_positions, positions)
        ee_pot  = jax.vmap(jax.vmap(ee_potential_fn, in_axes=(None, None, 0)), in_axes=(None, 0, 0))(params, atoms_positions, positions)
        ii_pot  = jax.vmap(jax.vmap(ii_potential_fn, in_axes=(None, None, 0)), in_axes=(None, 0, 0))(params, atoms_positions, positions)

        local_energies_noclip = kinetic + ei_pot + ee_pot + ii_pot  # (W,B)
        W, B = local_energies_noclip.shape
        out_dtype = local_energies_noclip.dtype  # 保持对外 dtype 一致

        # --- Per-replica reductions to scalars (shape ()) ---
        # 注意：对 (W,B) 做 mean → 标量，维度完全一致，便于后续 stack 一次 pmean
        kinetic_mean = jnp.mean(kinetic, axis=(0, 1))
        ei_mean      = jnp.mean(ei_pot,  axis=(0, 1))
        ee_mean      = jnp.mean(ee_pot,  axis=(0, 1))
        ii_mean      = jnp.mean(ii_pot,  axis=(0, 1))

        # --- 必须在 pmap 作用域；更友好地报错 ---
        _assert_in_pmap_or_explain(PMAP_AXIS_NAME)

        # --- Sentinel sync: 确保所有副本已就绪（对齐 collective 序列） ---
        _ = jax.lax.psum(jnp.array(0, jnp.int32), axis_name=PMAP_AXIS_NAME)
        jax.debug.print("sentinel psum ok")

        # --- 把 4 个标量拼成向量，统一到 fp32 后只做一次 pmean ---
        stacked = jnp.stack([kinetic_mean, ei_mean, ee_mean, ii_mean], axis=0)  # (4,)
        comm_vec = stacked.astype(jnp.float32)

        # --- 轻量一致性自检：所有副本的元素数应一致 ---
        vec_size  = jnp.array(comm_vec.size, jnp.int32)
        world_sz  = jax.lax.psum(jnp.array(1, jnp.int32), axis_name=PMAP_AXIS_NAME)
        sum_sizes = jax.lax.psum(vec_size, axis_name=PMAP_AXIS_NAME)
        jax.debug.print(f"comm check: world={world_sz}, vec_size={vec_size}, sum_sizes={sum_sizes}")
        # 若有需要，可在这里加 assert（训练时不建议抛异常）：
        # assert_fn = lambda cond, msg: jax.lax.cond(cond, lambda _: None, lambda _: (_ for _ in ()).throw(AssertionError(msg)), operand=None)
        # assert_fn(sum_sizes == world_sz * vec_size, "collective vector size mismatch across replicas")

        # --- 单次 AllReduce（pmean） ---
        comm_vec = jax.lax.pmean(comm_vec, axis_name=PMAP_AXIS_NAME)  # (4,)
        comm_vec = comm_vec.astype(out_dtype)
        kinetic_pmean, ei_pmean, ee_pmean, ii_pmean = comm_vec

        # --- 下面保持你的原有返回结构（占位实现） ---
        # 如需真正能量/方差统计，可在此处替换为实际计算逻辑。
        energy_per_w = jnp.ones((W, 1), dtype=out_dtype)
        E_loc        = jnp.ones((W, B), dtype=out_dtype)

        stats = dict(
            variance=jnp.ones((),   dtype=out_dtype),
            energy_noclip=jnp.ones((1,), dtype=out_dtype),
            variance_noclip=jnp.ones((), dtype=out_dtype),
            kinetic=kinetic_pmean,
            ei_potential=ei_pmean,
            ee_potential=ee_pmean,
            ii_potential=ii_pmean,
            multi_energy=jnp.squeeze(energy_per_w, axis=-1),  # (W,)
            world_size=world_sz,  # 方便排查
        )

        # multi_energy=jnp.squeeze(energy_per_w, axis=-1)

        # stats.update({"kinetic": kinetic_pmean, "ei_potential": ei_potential_pmean ,"ee_potential":ee_potential_pmean,"ii_potential":ii_potential_pmean,"multi_energy":multi_energy})

        return energy_per_w, E_loc, stats

    return energy_and_statistics


        # ridx = jax.lax.axis_index(utils.distribute.PMAP_AXIS_NAME) 
        # jax.debug.print(f"in energy_and_statistics:[replica {ridx}] kinetic shape={kinetic.shape} dtype={kinetic.dtype.name}")
        # jax.debug.print(f"in energy_and_statistics: [replica {ridx}] energy_per_w shape={energy_per_w.shape} dtype={energy_per_w.dtype.name}||E_loc shape={E_loc.shape} dtype={E_loc.dtype.name}")
        # jax.debug.print(f"in energy_and_statistics: [replica {ridx}] multi_energy shape={multi_energy.shape} dtype={multi_energy.dtype.name}")
        # jax.debug.print(f"in energy_and_statistics:[replica {ridx}] kinetic shape={kinetic.shape} dtype={kinetic.dtype.name}||local_energies_noclip shape={local_energies_noclip.shape} dtype={local_energies_noclip.dtype.name}")

def _assert_in_pmap_or_explain(axis_name: str):
    # 如果当前不在 pmap 作用域，下面这行会抛错；我们捕获后抛出更友好的信息
    try:
        ridx = jax.lax.axis_index(axis_name)
        jax.debug.print(f"pmap preflight ok: axis_name={axis_name}, replica_index={ridx}")
    except Exception as e:
        # 这里故意抛清晰的错误，告诉你在哪里、为什么、怎么修
        raise RuntimeError(
            f"[allreduce_mean] Not inside jax.pmap(axis_name={axis_name}). "
            f"You're calling a collective (pmean) outside pmap, or the axis_name mismatches.\n"
            f"Tips:\n"
            f"  - Wrap the caller with @jax.pmap(axis_name={axis_name}).\n"
            f"  - Ensure this axis_name matches everywhere (including utils.distribute.PMAP_AXIS_NAME).\n"
            f"  - Single-device is fine, but still must be inside pmap if you call pmean."
        ) from e