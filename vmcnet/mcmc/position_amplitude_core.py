"""Shared routines for position amplitude metropolis data."""

from typing import Any, Callable, Optional, Tuple, TypedDict

import chex
import jax
import jax.numpy as jnp
import functools
import vmcnet.mcmc.metropolis as metropolis
from vmcnet.utils.distribute import (
    replicate_all_local_devices,
    default_distribute_data,
)
from vmcnet.utils.typing import Array, P, PRNGKey, M, ModelApply, UpdateDataFn
from vmcnet.utils.distribute import PMAP_AXIS_NAME


class PositionAmplitudeWalkerData(TypedDict):
    """TypedDict of walker data holding just positions and amplitudes.

    Holding both particle position and wavefn amplitude in the same named
    tuple allows us to simultaneously mask over both in the acceptance function.

    The first dimension of position and amplitude should match, but position can have
    more dimensions.

    Attributes:
        position (Array): array of shape (n, ...)
        amplitude (Array): array of shape (n,)
    """

    elec_position: Array
    amplitude: Array


class PositionAmplitudeData(TypedDict):
    """TypedDict of data holding positions, amplitudes, and optional metadata.

    Holding both particle position and wavefn amplitude in the data can be advantageous
    to avoid recalculating amplitudes in some routines, e.g. acceptance probabilities.
    Furthermore, holding additional metadata can enable more sophisticated metropolis
    algorithms such as dynamically adjusted gaussian step sizes.

    Attributes:
        walker_data (PositionAmplitudeWalkerData): the positions and amplitudes
        move_metadata (any, optional): any metadata needed for the metropolis algorithm
    """
    atoms_position: Array
    walker_data: PositionAmplitudeWalkerData
    move_metadata: Any


def make_position_amplitude_data(atoms_position: Array, elec_position: Array, amplitude: Array, move_metadata: Any):
    """Create PositionAmplitudeData from position, amplitude, and move_metadata.

    Args:
        position (Array): the particle positions
        amplitude (Array): the wavefunction amplitudes
        move_metadata (Any): other required metadata for the metropolis algorithm

    Returns:
        PositionAmplitudeData: data containing positions, wavefn amplitudes, and move
        metadata
    """
    return PositionAmplitudeData(
        atoms_position=atoms_position,
        walker_data=PositionAmplitudeWalkerData(elec_position=elec_position, amplitude=amplitude),
        move_metadata=move_metadata,
    )


def get_position_from_data(data: PositionAmplitudeData) -> Array:
    """Get the position data from PositionAmplitudeData.

    Args:
        data (PositionAmplitudeData): the data

    Returns:
        Array: the particle positions from the data
    """
    return data["walker_data"]["elec_position"]


def get_amplitude_from_data(data: PositionAmplitudeData) -> Array:
    """Get the amplitude data from PositionAmplitudeData.

    Args:
        data (PositionAmplitudeData): the data

    Returns:
        Array: the wave function amplitudes from the data
    """
    return data["walker_data"]["amplitude"]


def to_pam_tuple(data: PositionAmplitudeData) -> Tuple[Array, Array, Array, Any]:
    """Returns data as a (position, amplitude, move_metadata) tuple.

    Useful for quickly assigning all three pieces to local variables for further use.
    """
    return (
        data["atoms_position"],
        data["walker_data"]["elec_position"],
        data["walker_data"]["amplitude"],
        data["move_metadata"],
    )


def get_update_data_fn(
    model_apply: ModelApply[P],
) -> UpdateDataFn[PositionAmplitudeData, P]:
    """Updates data based on new params, by recalculating amplitudes.

    This needs to be done in the VMC loop each time the params are updated.
    """

    def update_data_fn(data: PositionAmplitudeData, params: P) -> PositionAmplitudeData:
        position = data["walker_data"]["elec_position"]
        ion_pos = data["atoms_position"]
        amplitude = model_apply(params, ion_pos, position)
        return make_position_amplitude_data(ion_pos, position, amplitude, data["move_metadata"])

    return update_data_fn


def distribute_position_amplitude_data(
    data: PositionAmplitudeData,
) -> PositionAmplitudeData:
    """Distribute PositionAmplitudeData across devices.

    Args:
        data (PositionAmplitudeData): the data to distribute

    Returns:
        PositionAmplitudeData: the distributed data.
    """
    atoms_position = data["atoms_position"]
    walker_data = data["walker_data"]
    move_metadata = data["move_metadata"]
    atoms_position = replicate_all_local_devices(atoms_position)
    walker_data = default_distribute_data(walker_data)
    move_metadata = replicate_all_local_devices(move_metadata)
    return PositionAmplitudeData(atoms_position=atoms_position, walker_data=walker_data, move_metadata=move_metadata)


def make_position_amplitude_gaussian_proposal(
    model_apply: ModelApply[P],
    get_std_move: Callable[[PositionAmplitudeData], chex.Scalar],
) -> Callable[
    [P, PositionAmplitudeData, PRNGKey], Tuple[PositionAmplitudeData, PRNGKey]
]:
    """Create a gaussian proposal fn on PositionAmplitudeData.

    Positions are perturbed by a guassian; amplitudes are evaluated using the supplied
    model; move_metadata is not modified.

    Args:
        model_apply (Callable): function which evaluates a model. Has signature
            (params, position) -> amplitude
        get_std_move (Callable): function which gets the standard deviation of the
            gaussian move, which can optionally depend on the data. Has signature
            (PositionAmplitudeData) -> std_move

    Returns:
        Callable: proposal function which can be passed to the main VMC routine. Has
        signature (params, PositionAmplitudeData, key) -> (PositionAmplitudeData, key).
    """

    def proposal_fn(params: P, data: PositionAmplitudeData, key: PRNGKey):
        # jax.debug.print("marker_5")
        std_move = get_std_move(data)
        # jax.debug.print("marker_6")
        proposed_position, key = metropolis.gaussian_proposal(data["walker_data"]["elec_position"], std_move, key)
        # jax.debug.print("marker_7")
        atoms_position = data["atoms_position"]
        # jax.debug.print("marker_8")
        proposed_amplitude = model_apply(params, atoms_position, proposed_position)
        # jax.debug.print("marker_9")
        return (
            make_position_amplitude_data(
                atoms_position, proposed_position, proposed_amplitude, data["move_metadata"]
            ),
            key,
        )

    return proposal_fn


def make_position_amplitude_metropolis_symmetric_acceptance(
    logabs: bool = True,
) -> Callable[[P, PositionAmplitudeData, PositionAmplitudeData], Array]:
    """Create a Metropolis acceptance function on PositionAmplitudeData.

    Args:
        logabs (bool, optional): whether amplitudes provided to `acceptance_fn`
            represent psi (logabs = False) or log|psi| (logabs = True). Defaults to
            True.

    Returns:
        Callable: acceptance function which can be passed to the main VMC routine. Has
        signature (params, PositionAmplitudeData, PositionAmplitudeData) -> accept_ratio
    """

    def acceptance_fn(
        params: P, data: PositionAmplitudeData, proposed_data: PositionAmplitudeData
    ):
        del params
        # jax.debug.print("marker_10")
        return metropolis.metropolis_symmetric_acceptance(
            data["walker_data"]["amplitude"],
            proposed_data["walker_data"]["amplitude"],
            logabs=logabs,
        ) #a (W,B) array, elements are the accept ratio 

    return acceptance_fn


def make_position_amplitude_update(
    update_move_metadata_fn: Optional[Callable[[M, Array], M]] = None,
) -> Callable[
    [
        PositionAmplitudeData,
        PositionAmplitudeData,
        Array,
    ],
    PositionAmplitudeData,
]:
    """Factory for an update to PositionAmplitudeData.

    The returned update takes a mask of approved MCMC walker moves `move_mask` and
    accepts those proposed moves from `proposed_data`, for both positions and
    amplitudes. The `std_move` gaussian step width can also be modified by an optional
    `adjust_std_move_fn`.

    The moves in `move_mask` are applied along the first axis of the position data, and
    should be the same shape as the amplitude data (one-dimensional Array).

    Args:
        update_move_metadata_fn (Callable): function which calculates the new
            move_metadata. Has signature
            (old_move_metadata, move_mask) -> new_move_metadata

    Returns:
        Callable: function with signature
            (PositionAmplitudeData, PositionAmplitudeData, Array) ->
                (PositionAmplitudeData),
            which takes in the original PositionAmplitudeData, the proposed
            PositionAmplitudeData, and a move mask. Uses
            the move mask to decide which proposed data to accept.
    """

    def update_position_amplitude(
        data: PositionAmplitudeData,
        proposed_data: PositionAmplitudeData,
        move_mask: Array,
    ) -> PositionAmplitudeData:
        def mask_on_first_dimension(old_data: Array, proposal: Array):
            shaped_mask = jnp.reshape(move_mask, (old_data.shape[0:2])+((1,) * (old_data.ndim - 2)))
            return jnp.where(shaped_mask, proposal, old_data)

        new_walker_data = jax.tree_map(
            mask_on_first_dimension, data["walker_data"], proposed_data["walker_data"]
        )
        # jax.debug.print("marker_11")
        new_move_metadata = proposed_data["move_metadata"]
        # jax.debug.print("marker_12")
        if update_move_metadata_fn is not None:
            new_move_metadata = update_move_metadata_fn(
                data["move_metadata"], move_mask
            )
        # jax.debug.print("marker_13")

        return PositionAmplitudeData(
            atoms_position=data["atoms_position"], walker_data=new_walker_data, move_metadata=new_move_metadata
        )

    return update_position_amplitude


def make_position_amplitude_gaussian_metropolis_step(
    model_apply: ModelApply[P],
    get_std_move: Callable[[PositionAmplitudeData], chex.Scalar],
    update_move_metadata_fn: Optional[Callable[[M, Array], M]] = None,
    logabs: bool = True,
) -> metropolis.MetropolisStep[P, PositionAmplitudeData]:
    """Make a gaussian proposal with Metropolis acceptance for PositionAmplitudeData.

    Args:
        model_apply (Callable): function which evaluates a model. Has signature
            (params, position) -> amplitude
        get_std_move (Callable): function which gets the standard deviation of the
            gaussian move, which can optionally depend on the data. Has signature
            (PositionAmplitudeData) -> std_move
        update_move_metadata_fn (Callable, optional): function which calculates the new
            move_metadata. Has signature
            (old_move_metadata, move_mask) -> new_move_metadata.
        logabs (bool, optional): whether the provided amplitudes represent psi
            (logabs = False) or log|psi| (logabs = True). Defaults to True.

    Returns:
        Callable: function which does a metropolis step. Has the signature
            (params, PositionAmplitudeData, key)
            -> (mean acceptance probability, PositionAmplitudeData, new_key)
    """
    proposal_fn = make_position_amplitude_gaussian_proposal(model_apply, get_std_move) #proposal is callable,and returns a proposal data
    accept_fn = make_position_amplitude_metropolis_symmetric_acceptance(logabs=logabs) #accept_fn is callable, reveive data and data_proposal, return a ratio array (W,B)
    metrop_step_fn = metropolis.make_metropolis_step(
        proposal_fn,
        accept_fn,
        make_position_amplitude_update(update_move_metadata_fn),
    )
    return metrop_step_fn


def shuffle_and_split_data(x, idx, down_sample_num):
    x = x[idx, ...]
    x0, x1 = jnp.split(x, [down_sample_num], axis=0)
    return x0, x1

# def down_sample_data_pre(key, data, down_sample_num):
#     xp = data["atoms_position"]
#     xe = data["walker_data"]["elec_position"]
#     amp = data["walker_data"]["amplitude"]

#     key, subkey = jax.random.split(key)
#     walker_num = xp.shape[0]
#     idx = jax.random.permutation(subkey, jnp.arange(walker_num))

#     (xp0,xp1), (xe0,xe1), (amp0,amp1) = map(functools.partial(shuffle_and_split_data, idx=idx, down_sample_num=down_sample_num) , [xp, xe, amp])

#     return (xp0,xp1), (xe0,xe1), (amp0,amp1), idx, key

def down_sample_data_pre(data, down_sample_num, variance, multi_energy):
    xp  = data["atoms_position"]
    xe  = data["walker_data"]["elec_position"]
    amp = data["walker_data"]["amplitude"]

    idx_sorted = jnp.argsort(-variance)
    idx0 = idx_sorted[:down_sample_num]      # 方差最大的那几个
    idx1 = idx_sorted[down_sample_num:]      # 剩余

    xp0, xp1 = xp[idx0], xp[idx1]
    xe0, xe1 = xe[idx0], xe[idx1]
    amp0, amp1 = amp[idx0], amp[idx1]
    variance0, variance1 = variance[idx0], variance[idx1]
    multi_energy0, multi_energy1 = multi_energy[idx0], multi_energy[idx1]

    return (xp0, xp1), (xe0, xe1), (amp0, amp1), idx_sorted, variance1, multi_energy1


def make_down_sample_data_fn(apply_pmap):
    if apply_pmap:
        def down_sample_data(data, down_sample_num, variance, multi_energy):
            move_metadata = data["move_metadata"]
            (xp0,xp1), (xe0,xe1), (amp0,amp1), idx, variance1, multi_energy1 = jax.pmap(down_sample_data_pre,
                                                               axis_name=PMAP_AXIS_NAME,
                                                               in_axes=(0, None, 0, 0),
                                                               static_broadcasted_argnums=(1,),
                                                               )(data, down_sample_num, variance, multi_energy)
            data = make_position_amplitude_data(xp0, xe0, amp0, move_metadata)
            rest_data = make_position_amplitude_data(xp1, xe1, amp1, move_metadata)
            return data, rest_data, idx, variance1, multi_energy1
    else:
        def down_sample_data(data, down_sample_num, variance, multi_energy):
            move_metadata = data["move_metadata"]
            (xp0,xp1), (xe0,xe1), (amp0,amp1), idx, variance1, multi_energy1 = jax.jit(down_sample_data_pre,
                                                                static_argnums=(1,),
                                                                )(data, down_sample_num, variance, multi_energy)
            data = make_position_amplitude_data(xp0, xe0, amp0, move_metadata)
            rest_data = make_position_amplitude_data(xp1, xe1, amp1, move_metadata)
            return data, rest_data, idx, variance1, multi_energy1
    return down_sample_data

def scatter_to_original_order_data(x0,x1,idx):
    n0 = x0.shape[0]
    n1 = x1.shape[0]
    walker_num = n0 + n1
    x = jnp.zeros((walker_num, *x0.shape[1:]), dtype=x0.dtype)
    x = x.at[idx[:n0]].set(x0)
    x = x.at[idx[n0:]].set(x1)
    return x

def make_reform_data_and_metrics_fn(apply_pmap):
    if apply_pmap:
        reform_data = jax.pmap(scatter_to_original_order_data, axis_name=PMAP_AXIS_NAME,in_axes=(0,0,0))
    else:
        reform_data = jax.jit(scatter_to_original_order_data)
    def reform_data_and_metrics(data, rest_data, metrics, idx, variance1, multi_energy1):
        xp0 = data["atoms_position"]                   # 选中子集（大小 n0）
        xe0 = data["walker_data"]["elec_position"]
        amp0 = data["walker_data"]["amplitude"]

        xp1 = rest_data["atoms_position"]              # 剩余子集（大小 n1）
        xe1 = rest_data["walker_data"]["elec_position"]
        amp1 = rest_data["walker_data"]["amplitude"]

        xp  = reform_data(xp0,  xp1,  idx)
        xe  = reform_data(xe0,  xe1,  idx)
        amp = reform_data(amp0, amp1, idx)
        data_out = make_position_amplitude_data(xp, xe, amp, data["move_metadata"])
        
        multi_energy0 = metrics["multi_energy"]    # 形状 [n0, ...]
        multi_energy = reform_data(multi_energy0, multi_energy1, idx)

        variance0 = metrics["multi_variance"]
        multi_variance = reform_data(variance0,  variance1,  idx)

        new_metrics = dict(metrics)
        new_metrics["multi_energy"] = multi_energy
        new_metrics["multi_variance"] = multi_variance

        return data_out, new_metrics
    return reform_data_and_metrics




def init_dummy_metrics_for_downsample(is_pmapped: bool):
    def aaa(atoms):
        W = atoms.shape[0]
        # multi_variance = jnp.zeros((W,), dtype=atoms.dtype)        # (W,)
        multi_variance = jnp.full((W,), jnp.inf, dtype=atoms.dtype)
        multi_energy = jnp.zeros((W,), dtype=atoms.dtype)        # (W,)
        return multi_variance, multi_energy
    
    if is_pmapped:
        create_dummy = jax.pmap(aaa, axis_name=PMAP_AXIS_NAME)
    else:
        create_dummy = aaa
    return create_dummy