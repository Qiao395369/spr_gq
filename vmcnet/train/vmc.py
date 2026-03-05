"""Main VMC loop."""

from typing import Tuple, Optional

import jax
# import wandb
import jax.numpy as jnp
import time
from vmcnet.mcmc.metropolis import WalkerFn
from vmcnet.updates.update_param_fns import UpdateParamFn
from vmcnet.utils.checkpoint import CheckpointWriter, MetricsWriter
import vmcnet.utils as utils
from vmcnet.utils.typing import D, GetAmplitudeFromData, P, PRNGKey, S
from vmcnet.mcmc.position_amplitude_core import make_down_sample_data_fn, make_reform_data_and_metrics_fn, init_dummy_metrics_for_downsample
import logging
def vmc_loop(
    params: P,
    optimizer_state: S,
    data: D,
    nchains: int,
    nepochs: int,
    walker_fn: WalkerFn[P, D],
    get_grad_and_E: UpdateParamFn[P, D, S],
    update_param_fn: UpdateParamFn[P, D, S],
    key: PRNGKey,
    logdir: Optional[str] = None,
    checkpoint_every: Optional[int] = 1000,
    best_checkpoint_every: Optional[int] = 100,
    checkpoint_dir: str = "checkpoints",
    checkpoint_variance_scale: float = 10.0,
    check_for_nans: bool = False,
    record_amplitudes: bool = False,
    get_amplitude_fn: Optional[GetAmplitudeFromData[D]] = None,
    nhistory_max: int = 200,
    is_pmapped=True,
    start_epoch: int = 0,
    down_sample_num: int = None,
    down_sample_mode: str = "sto",
    n_inner: int = 1,
    acc_steps: int=1,
    is_eval: bool = False,
) -> Tuple[P, S, D, PRNGKey, bool]:
    """Main Variational Monte Carlo loop routine.

    Variational Monte Carlo (VMC) can be generically viewed as minimizing a
    parameterized variational loss stochastically by sampling over a data distribution
    via Monte Carlo sampling. This function implements this idea at a high level, using
    a walker_fn to sample the data distribution, and passing the optimization step to a
    generic function `update_param_fn`.

    Args:
        params (pytree-like): model parameters which are trained
        optimizer_state (pytree-like): initial state of the optimizer
        data (pytree-like): initial data
        nchains (int): number of parallel MCMC chains being run. This can be difficult
            to infer from data, depending on the structure of data, whether data has
            been pmapped, etc.
        nepochs (int): number of parameter updates to do
        walker_fn (Callable): function which does a number of walker steps between each
            parameter update. Has the signature
            (data, params, key) -> (mean accept prob, new data, new key)
        update_param_fn (Callable): function which updates the parameters. Has signature
            (data, params, optimizer_state, key)
                -> (new_params, optimizer_state, dict: metrics, key).
            If metrics is not None, it is required to have the entries "energy" and
            "variance" at a minimum. If metrics is None, no checkpointing is done.
        key (PRNGKey): an array with shape (2,) representing a jax PRNG key passed
            to proposal_fn and used to randomly accept proposals with probabilities
            output by acceptance_fn
        logdir (str, optional): name of parent log directory. If None, no checkpointing
            is done. Defaults to None.
        checkpoint_every (int, optional): how often to regularly save checkpoints. If
            None, checkpoints are only saved when the error-adjusted running avg of the
            energy improves. Defaults to 1000.
        best_checkpoint_every (int, optional): limit on how often to save best
            checkpoint, even if energy is improving. When the error-adjusted running avg
            of the energy improves, instead of immediately saving a checkpoint, we hold
            onto the data from that epoch in memory, and if it's still the best one when
            we hit an epoch which is a multiple of `best_checkpoint_every`, we save it
            then. This ensures we don't waste time saving best checkpoints too often
            when the energy is on a downward trajectory (as we hope it often is!).
            Defaults to 100.
        checkpoint_dir (str, optional): name of subdirectory to save the regular
            checkpoints. These are saved as "logdir/checkpoint_dir/(epoch + 1).npz".
            Defaults to "checkpoints".
        checkpoint_variance_scale (float, optional): scale of the variance term in the
            error-adjusted running avg of the energy. Higher means the variance is more
            important, and lower means the energy is more important. See
            :func:`~vmctrain.train.vmc.get_checkpoint_metric`. Defaults to 10.0.
        check_for_nans (bool, optional): whether to check for nans in the vmc loop. If
            so, then after nans are detected, a checkpoint will be saved and the loop
            will be aborted. Defaults to False.
        nhistory_max (int, optional): How much history to keep in the running histories
            of the energy and variance. Defaults to 200.

    Returns:
        A tuple of (trained parameters, final optimizer state, final data, final key,
        nans_detected). The first four entries are the same structure as
        (params, optimizer_state, initial_data, key).
    """
    (
        checkpoint_dir,
        checkpoint_metric,
        running_energy_and_variance,
        best_checkpoint_data,
    ) = utils.checkpoint.initialize_checkpointing(
        checkpoint_dir, nhistory_max, logdir, checkpoint_every
    )
    nans_detected = False
    down_sample = (not is_eval) and (down_sample_num is not None) and (down_sample_num != 0)
    if is_pmapped and down_sample:
        assert down_sample_num % jax.device_count() == 0, "down_sample_num must be divisible by number of devices"
        down_sample_num = down_sample_num//jax.device_count()
    
    if down_sample:
        down_sample_data = make_down_sample_data_fn(is_pmapped, down_sample_mode)
        reform_data_and_metrics = make_reform_data_and_metrics_fn(is_pmapped)
        create_dummy = init_dummy_metrics_for_downsample(is_pmapped)
        variance , multi_energy , accept_ratio= create_dummy(data["atoms_position"])
        logging.info("Downsample data with down_sample_num = %d, n_inner = %d "%(down_sample_num, n_inner))

    with CheckpointWriter(is_pmapped) as checkpoint_writer, MetricsWriter() as metrics_writer:
        time_mark=time.time()
        for epoch in range(start_epoch, nepochs):
            # Save state for checkpointing at the start of the epoch for two reasons:
            # 1. To save the model that generates the best energy and variance metrics,
            # rather than the model one parameter UPDATE after the best metrics.
            # 2. To ensure a fully consistent state can be reloaded from a checkpoint, &
            # the exact subsequent behavior can be reproduced (if run on same machine).
            # NOTE: jax deletes the old arrays if we don't make copies.
            old_params = jax.tree_util.tree_map(lambda x: x.copy(), params)
            old_state = jax.tree_util.tree_map(lambda x: x.copy(), optimizer_state)
            old_data = data.copy()
            old_key = key.copy()

            if down_sample :
                key, data, rest_data, idx, variance1, multi_energy1, accept_ratio1 = down_sample_data(key, data, down_sample_num, variance, multi_energy, accept_ratio)
                for _ in range(n_inner):
                    accept_ratio0, data, key = walker_fn(params, data, key)
                    params, data, optimizer_state, metrics ,key = update_param_fn(key, params, optimizer_state, data)
                data, metrics = reform_data_and_metrics(data, rest_data, metrics, idx, variance1, multi_energy1, accept_ratio0, accept_ratio1)
                variance = metrics["multi_variance"]
                multi_energy = metrics["multi_energy"]
                accept_ratio = metrics["accept_ratio"]
            else:
                def zeros_like_tree(tree):
                    return jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), tree)

                metrics_template = {k: jnp.array(0.0, dtype=jnp.float32) for k in [
                                        "energy","variance","variance_noclip","multi_variance","energy_noclip","multi_energy"
                                    ]}
                if is_pmapped:
                    grad_acc = utils.distribute.replicate_all_local_devices(zeros_like_tree(optimizer_state["prev_grad"]))
                    metrics_acc = utils.distribute.replicate_all_local_devices(zeros_like_tree(metrics_template))
                else:
                    grad_acc = zeros_like_tree(optimizer_state["prev_grad"])
                    metrics_acc = zeros_like_tree(metrics_template)

                for _ in range(acc_steps):
                    accept_ratio, data, key = walker_fn(params, data, key)
                    grad_acc, metrics_acc, optimizer_state, data, key = get_grad_and_E(key, params, optimizer_state, data, grad_acc, metrics_acc)
                    
                grad = jax.tree_util.tree_map(lambda acc: acc / acc_steps, grad_acc)
                params, data, optimizer_state, key = update_param_fn(key, params, optimizer_state, data, grad)
                
                metrics = jax.tree_util.tree_map(lambda acc: acc / acc_steps, metrics_acc)
                metrics["accept_ratio"] = accept_ratio
                metrics["accept_ratio_mean"] = utils.distribute.pmap(lambda x: jnp.mean(x))(accept_ratio) if is_pmapped else jnp.mean(accept_ratio)
                metrics["std_move"] = data["move_metadata"]["std_move"]
                metrics["move_acceptance_sum"] = data["move_metadata"]["move_acceptance_sum"]
                metrics["moves_since_update"] = data["move_metadata"]["moves_since_update"]
        
            # Don't checkpoint if no metrics to checkpoint
            if metrics is None :
                continue

            if is_pmapped:
                metrics_cpu = dict(metrics)
                # logging.info(f"metrics_cpu: {metrics_cpu}")
                metrics_cpu["multi_energy"] = jax.device_get(metrics["multi_energy"])[None, ...]
                metrics_cpu["multi_variance"] = jax.device_get(metrics["multi_variance"])[None, ...]
                metrics_cpu["accept_ratio"] = jax.device_get(metrics["accept_ratio"])[None, ...]
                metrics_cpu["std_move"] = jax.device_get(metrics["std_move"])[None, ...]
                metrics_cpu["move_acceptance_sum"] = jax.device_get(metrics["move_acceptance_sum"])[None, ...]
                metrics_cpu["moves_since_update"] = jax.device_get(metrics["moves_since_update"])[None, ...]
                metrics_cpu = jax.tree_map(lambda x: x[0], metrics_cpu)
                metrics_cpu = jax.device_put(metrics_cpu, jax.devices("cpu")[0])
            else:
                metrics_cpu = metrics  

            (
                checkpoint_metric,
                best_checkpoint_data,
                time_mark,
                nans_detected,
            ) = utils.checkpoint.save_metrics_and_handle_checkpoints(
                time_mark,
                epoch,
                old_params,
                params,
                old_state,
                old_data,
                data,
                old_key,
                metrics_cpu,
                nchains,
                running_energy_and_variance,
                checkpoint_writer,
                metrics_writer,
                checkpoint_metric,
                logdir=logdir,
                variance_scale=checkpoint_variance_scale,
                checkpoint_every=checkpoint_every,
                best_checkpoint_every=best_checkpoint_every,
                best_checkpoint_data=best_checkpoint_data,
                checkpoint_dir=checkpoint_dir,
                check_for_nans=check_for_nans,
                record_amplitudes=record_amplitudes,
                get_amplitude_fn=get_amplitude_fn,
            )

            if nans_detected:
                break

        utils.checkpoint.finish_checkpointing(
            checkpoint_writer, best_checkpoint_data, logdir
        )

    return params, optimizer_state, data, key, nans_detected
