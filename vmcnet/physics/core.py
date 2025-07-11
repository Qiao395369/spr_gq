"""Core local energy and gradient construction routines."""
from typing import Callable, Optional, Sequence, Tuple, cast

import chex
import jax
import jax.numpy as jnp
import kfac_jax

import vmcnet.utils as utils
from vmcnet.utils.pytree_helpers import tree_sum
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

EnergyAuxData = Dict[str, Any]
EnergyData = Tuple[Array, EnergyAuxData]
ValueGradEnergyFn = Callable[[P, PRNGKey, Array], Tuple[EnergyData, P]]


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
    assert len(ion_pos.shape)==3
    natoms=len(ion_charges)
    walker=ion_pos.shape[0]
    assert ion_pos.shape==(walker,natoms,3)
    # single_spins=[(5,2),(2,5)]
    assert nelec_total==jnp.sum(single_spins)
    print(single_spins)
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
    print("xe.shape:",ppp.shape)
    return key, ppp

def combine_local_energy_terms(
    local_energy_terms: Sequence[ModelApply[P]],
) -> LocalEnergyApply[P]:
    """Combine a sequence of local energy terms by adding them.

    Args:
        local_energy_terms (Sequence): sequence of local energy terms, each with the
            signature (params,atoms_positions, x) -> array of terms of shape (x.shape[0],)

    Returns:
        Callable: local energy function which computes the sum of the local energy
        terms. Has the signature
        (params, x) -> local energy array of shape (x.shape[0],)
    """

    def local_energy_fn(params: P,atoms_positions, x: Array, key: Optional[PRNGKey]) -> Array:
        del key
        local_energy_sum = local_energy_terms[0](params, atoms_positions, x)
        for term in local_energy_terms[1:]:
            local_energy_sum = cast(Array, local_energy_sum + term(params, atoms_positions, x))
        return local_energy_sum

    return local_energy_fn


def laplacian_psi_over_psi(
    grad_log_psi_apply: ModelApply,
    params: P,
    ion_pos: Array,
    x: Array,
    nparticles: Optional[int] = None,
    particle_perm: Optional[Array] = None,
) -> Array:
    """Compute (nabla^2 psi) / psi at x given a function which evaluates psi'(x)/psi.

    The computation is done by computing (forward-mode) derivatives of the gradient to
    get the columns of the Hessian, and accumulating the (i, i)th entries (but this
    implementation is significantly more memory efficient than directly computing the
    Hessian).

    This function uses the identity

        (nabla^2 psi) / psi = (nabla^2 log|psi|) + (nabla log|psi|)^2

    to avoid leaving the log domain during the computation.

    This function should be vmapped in order to be applied to batches of inputs, as it
    completely flattens x in order to take second derivatives w.r.t. each component.

    This is approach is extremely similar to the one in the FermiNet repo
    (in the jax branch, as of this writing -- see
    https://github.com/deepmind/ferminet/blob/aade61b3d30883b3238d6b50c85404d0e8176155/ferminet/hamiltonian.py).

    The main difference is that we are being explicit about the flattening of x within
    the Laplacian calculation, so that it does not have to be handled outside of this
    function (psi is free to take x shapes which are not flat).

    Args:
        grad_log_psi_apply (Callable): function which evaluates the derivative of
            log|psi(x)|, i.e. (nabla psi)(x) / psi(x), with respect to x. Has the
            signature (params, x) -> (nabla psi)(x) / psi(x), so the derivative should
            be over the second arg, x, and the output shape should be the same as x
        params (pytree): model parameters, passed as the first arg of grad_log_psi
        x (Array): second input to grad_log_psi
        nparticles (int, Optional): when specified, only the first nparticles particles
            of the particle_perm are used to calculate the laplacian. Defaults to None.
        particle_perm (Array, Optional): permutation of the particle indices whose
            ordering determines which particles' laplacians get evaluated, for use
            in conjunction with the random particle method and the nparticles argument.

    Returns:
        Array: "local" laplacian calculation, i.e. (nabla^2 psi) / psi
    """
    x_shape = x.shape
    flat_x = jnp.reshape(x, (-1,))
    n = flat_x.shape[0]
    identity_mat = jnp.eye(n)

    def flattened_grad_log_psi_of_flat_x(flat_x_in):
        """Flattened input to flattened output version of grad_log_psi."""
        grad_log_psi_out = grad_log_psi_apply(params,ion_pos, jnp.reshape(flat_x_in, x_shape))
        return jnp.reshape(grad_log_psi_out, (-1,))

    length = n
    multiplier = 1.0
    vecs = identity_mat

    if nparticles is not None and particle_perm is not None:
        length = 3 * nparticles
        multiplier = n / (3 * nparticles)

        d = x_shape[-1]
        triple_perm = jnp.stack([particle_perm * d + i for i in range(d)]).T.reshape(
            (n,)
        )
        vecs = identity_mat[triple_perm, :]

    def step_fn(carry, unused):
        del unused
        i = carry[0]
        primals, tangents = jax.jvp(
            flattened_grad_log_psi_of_flat_x, (flat_x,), (vecs[i],)
        )
        return (
            i + 1,
            carry[1]
            + jnp.square(jnp.dot(primals, vecs[i]))
            + jnp.dot(tangents, vecs[i]),
        ), None

    out, _ = jax.lax.scan(step_fn, (0, jnp.array(0.0)), xs=None, length=length)
    return out[1] * multiplier


def get_statistics_from_local_energy(
    local_energies: Array, nchains: int, nan_safe: bool = True
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
    assert len(local_energies.shape)==2
    if nan_safe:
        allreduce_mean = utils.distribute.nanmean_all_local_devices
    else:
        allreduce_mean = utils.distribute.mean_all_local_devices
    #TODO use the last axis instead of 1.
    energy1 = allreduce_mean(local_energies,axis=(0,1))  #()
    energy2=allreduce_mean(local_energies,axis=1)[:,None]       #(W,1)
    #nchains=W*B
    variance = (allreduce_mean(jnp.square(local_energies - energy2),axis=(0,1)) * nchains / (nchains - 1))  # adjust by n / (n - 1) to get an unbiased estimator
    return energy1,energy2, variance

def get_statistics_from_other_energy(
    energies: Array, nan_safe: bool = True
) -> Tuple[Array, Array]:
    assert len(energies.shape)==2
    if nan_safe:
        allreduce_mean = utils.distribute.nanmean_all_local_devices
    else:
        allreduce_mean = utils.distribute.mean_all_local_devices
    energy = allreduce_mean(energies,axis=(0,1))
    return energy

def get_clipped_energies_and_aux_data(
    local_energies_noclip: Array,
    nchains: int,
    clipping_fn: Optional[ClippingFn],
    nan_safe: bool,
) -> Tuple[Array, Array, EnergyAuxData]:
    """Clip local energies if requested and return auxiliary data.
    local_energies_noclip:(W,B)
    
    """
    assert len(local_energies_noclip.shape)==2 #(W,B)
    if clipping_fn is not None:
        # For the unclipped metrics, which are not used in the gradient, don't
        # do these in a nan-safe way. This makes nans more visible and makes sure
        # nans checkpointing will work properly.
        energy_noclip,_, variance_noclip = get_statistics_from_local_energy(local_energies_noclip, nchains, nan_safe=False)  
        #energy_noclip:()  variance_noclip:()
        local_energies = clipping_fn(local_energies_noclip, energy_noclip)  #local_energies:(W,B)
        energy1, energy2, variance = get_statistics_from_local_energy(local_energies, nchains, nan_safe=nan_safe) #energy1:() energy2:(W,1) variance:()
    else:
        local_energies = local_energies_noclip
        energy1,energy2, variance = get_statistics_from_local_energy(local_energies_noclip, nchains, nan_safe=nan_safe)

        # Even though there's no clipping function, still record noclip metrics
        # without nan-safety so that checkpointing epochs with nans can be
        # supported.
        energy_noclip, variance_noclip = get_statistics_from_local_energy(local_energies_noclip, nchains, nan_safe=False)
    aux_data = dict(
        variance=variance,  #()
        local_energies_noclip=local_energies_noclip,  #(W,B)
        energy_noclip=energy_noclip,  #()
        variance_noclip=variance_noclip,  #()
        centered_local_energies=local_energies - energy2,  #(W,B)
    )
    return energy1,energy2, local_energies, aux_data


def create_value_and_grad_energy_fn(
    log_psi_apply: ModelApply[P],
    # local_energy_fn: LocalEnergyApply[P],
    kinetic_fn,ei_potential_fn,ee_potential_fn,ii_potential_fn,
    nchains: int,
    clipping_fn: Optional[ClippingFn] = None,
    nan_safe: bool = True,
    local_energy_type: str = "standard",
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
        local_energy_type (str): one of [standard, ibp, random_particle]. "standard"
            implies the standard VMC local energy in which case we can apply the
            traditional VMC gradient estimator described above. "ibp" means integration
            by parts in which case we use the "generic" VMC gradient estimator which
            does not depend on the local energy having the form APsi/Psi for a Hermitian
            operator A. In practice this means an extra term must be added to the
            gradient in which the local energy itself is differentiated with respect to
            the model parameters. "random_particle" means that the local energy uses one
            or more random particles for each walker, rather than using all the
            particles. In this case the standard gradient estimator can be used but the
            code is slightly modified to pass a distinct PRNGkey to each walker.
            Defaults to standard.

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

    def get_standard_contribution(local_energies_noclip, params, atoms_positions,positions):
        '''
        local_energies_noclip:(W,B)
        atoms_position:(W,natom,dim)
        positions(W,B,nele,dim)
        '''
        energy1,energy2, local_energies, aux_data = get_clipped_energies_and_aux_data(
            local_energies_noclip, nchains, clipping_fn, nan_safe
        )  # energy1:(),energy2:(W,1) local_energies:(W,B)
        centered_local_energies = local_energies - energy2  # centered_local_energies:(W,B)
        grad_E = jax.grad(standard_estimator_forward, argnums=0)(
            params, atoms_positions, positions, centered_local_energies
        )  # grad_E has the same shape as params.
        return aux_data, energy1,energy2.reshape((-1)), grad_E

    def standard_energy_val_and_grad(params, key, atoms_positions, positions):
        '''
        atoms_positions:(W,natom,dim)
        positions:(W,B,nele,dim)
        '''
        del key
        # print(f"positions.shape:{positions.shape}")  #(W,B,n,dim)

        # local_energies_noclip = jax.vmap(local_energy_fn, in_axes=(None, 0, None))(params, positions, None)
        # def batched_f(f):
        #     return lambda p,xa,xe: jax.vmap(lambda xa_, xe_: jax.vmap(lambda xe__: f(p,xa_, xe__),in_axes=0)(xe_),in_axes=(0, 0))(xa, xe)

        kinetic=jax.vmap(jax.vmap(kinetic_fn, in_axes=(None,None,0)),in_axes=(None,0,0))(params,atoms_positions, positions) #(W,B)
        ei_potential= jax.vmap(jax.vmap(ei_potential_fn,in_axes=(None,None,0)),in_axes=(None,0,0))(params,atoms_positions, positions)  #(W,B)
        ee_potential=jax.vmap(jax.vmap(ee_potential_fn, in_axes=(None,None,0)),in_axes=(None,0,0))(params,atoms_positions,positions)#(W,B)
        ii_potential=jax.vmap(jax.vmap(ii_potential_fn, in_axes=(None,None,0)),in_axes=(None,0,0))(params,atoms_positions,positions)#(W,B)
        # print(kinetic.shape,ei_potential.shape,ee_potential.shape,ii_potential.shape)
        local_energies_noclip=kinetic+ei_potential+ee_potential+ii_potential  #(W,B)

        kinetic = get_statistics_from_other_energy(kinetic, nan_safe=nan_safe) #()
        ei_potential = get_statistics_from_other_energy(ei_potential, nan_safe=nan_safe) #()
        ee_potential = get_statistics_from_other_energy(ee_potential, nan_safe=nan_safe) #()
        ii_potential = get_statistics_from_other_energy(ii_potential, nan_safe=nan_safe) #()

        aux_data, energy, multi_energy,grad_E = get_standard_contribution(local_energies_noclip, params, atoms_positions, positions) 
        aux_data.update({"kinetic": kinetic, "ei_potential": ei_potential ,"ee_potential":ee_potential,"ii_potential":ii_potential,"multi_energy":multi_energy})
        return (energy, aux_data), grad_E

    # def random_particle_energy_val_and_grad(params, key, positions):
    #     nbatch = positions.shape[0]
    #     key = jax.random.split(key, nbatch)

    #     local_energies_noclip = jax.vmap(
    #         local_energy_fn, in_axes=(None, 0, 0), out_axes=0
    #     )(params, positions, key)

    #     aux_data, energy, grad_E = get_standard_contribution(
    #         local_energies_noclip, params, positions
    #     )
    #     return (energy, aux_data), grad_E

    # def generic_energy_val_and_grad(params, key, positions):
    #     del key

    #     val_and_grad_local_energy = jax.value_and_grad(local_energy_fn, argnums=0)
    #     val_and_grad_local_energy_vmapped = jax.vmap(
    #         val_and_grad_local_energy, in_axes=(None, 0, None), out_axes=0
    #     )
    #     local_energies_noclip, local_energy_grads = val_and_grad_local_energy_vmapped(
    #         params, positions, None
    #     )
    #     generic_contribution = jax.tree_map(mean_grad_fn, local_energy_grads)

    #     # Gradient clipping seems to make the optimization fail miserably when using
    #     # the generic gradient estimator, so setting clipping_fn=None here.
    #     # TODO (ggoldsh): investigate this phenomenon further.
    #     energy, local_energies, aux_data = get_clipped_energies_and_aux_data(
    #         local_energies_noclip, nchains, clipping_fn=None, nan_safe=nan_safe
    #     )

    #     centered_local_energies = local_energies - energy

    #     standard_contribution = jax.grad(standard_estimator_forward, argnums=0)(
    #         params, positions, centered_local_energies
    #     )

    #     grad_E = tree_sum(standard_contribution, generic_contribution)

    #     return (energy, aux_data), grad_E

    if local_energy_type == "standard":
        return standard_energy_val_and_grad
    # elif local_energy_type == "ibp":
    #     return generic_energy_val_and_grad
    # elif local_energy_type == "random_particle":
    #     return random_particle_energy_val_and_grad
    else:
        raise ValueError(
            f"Requested local energy type {local_energy_type} is not supported"
        )
