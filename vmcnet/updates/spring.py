"""SPRING implementation, see https://doi.org/10.1016/j.jcp.2024.113351."""

from typing import Callable, Dict, Tuple, Any
import jax
import jax.flatten_util
import jax.numpy as jnp
import neural_tangents as nt  # type: ignore
from ml_collections import ConfigDict
import chex
import optax

from vmcnet.utils.typing import Array, D, ModelApply, P, S, Tuple
from vmcnet.utils.pytree_helpers import (
    multiply_tree_by_scalar,
    tree_inner_product,
    tree_reduce_l1,
)
from vmcnet.utils.distribute import pmean_if_pmap
from vmcnet.utils.typing import UpdateDataFn, GetPositionFromData, LearningRateSchedule

from .update_param_fns import (
    UpdateParamFn,
    make_traced_fn_with_single_metrics,
    update_metrics_with_noclip,
)
from .optax_utils import initialize_optax_optimizer
import psutil
import logging

# 定义空标记（例如 1e30，确保不会与真实能量值冲突）
EMPTY_MARKER = jnp.array(1e30, dtype=jnp.float32)

def print_memory_usage(message: str):
    # 主机内存
    host_mem = psutil.virtual_memory().used / (1024**3)
    # GPU内存（若使用GPU）
    # gpu_mem = jax.device_get(jax.numpy.array([0])).devices().memory_stats()["bytes_used"] / (1024**3)
    logging.info(f"[{message}] 主机内存: {host_mem:.2f} GB")

def construct_spring_update_param_fn(
    energy_and_statistics_fn,
    optimizer_apply: Callable[[P, P, S, D, Dict[str, Array]], Tuple[P, S]],
    get_position_fn: GetPositionFromData[D],
    update_data_fn: UpdateDataFn[D, P],
    apply_pmap: bool = True,
    record_param_l1_norm: bool = False,
) -> UpdateParamFn[P, D, S]:
    """Create the `update_param_fn` based on the gradient of the total energy."""

    def update_param_fn(params, data, optimizer_state, key):
        position = get_position_fn(data)
        atoms_position = data["atoms_position"]
        energy, local_energies, stats = energy_and_statistics_fn(params, atoms_position, position)
        params, optimizer_state = optimizer_apply(
            energy,
            local_energies,
            params,
            optimizer_state,
            data,
        )
        data = update_data_fn(data, params)
        metrics = {
                    "energy": energy, "variance": stats["variance"],
                    "kinetic": stats["kinetic"],
                    "ei_potential": stats["ei_potential"],
                    "ee_potential": stats["ee_potential"],
                    "ii_potential": stats["ii_potential"],
                    "multi_energy": stats["multi_energy"],
            }
        metrics = update_metrics_with_noclip(
            stats["energy_noclip"],
            stats["variance_noclip"],
            metrics,
        )
        if record_param_l1_norm:
            metrics.update({"param_l1_norm": tree_reduce_l1(params)})
        return params, data, optimizer_state, metrics, key

    traced_fn = make_traced_fn_with_single_metrics(update_param_fn, apply_pmap)

    return traced_fn


def initialize_spring(
    log_psi_apply: ModelApply[P],
    energy_and_statistics_fn,
    params: P,
    get_position_fn: GetPositionFromData[D],
    update_data_fn: UpdateDataFn[D, P],
    learning_rate_schedule: LearningRateSchedule,
    optimizer_config: ConfigDict,
    record_param_l1_norm: bool = False,
    apply_pmap: bool = True,
) -> Tuple[UpdateParamFn[P, D, optax.OptState], optax.OptState]:
    """Get an update param function and initial state for SPRING."""
    if optimizer_config.type == "old":
        get_spring_step = get_spring_step_old
    elif optimizer_config.type == "new":
        get_spring_step = get_spring_step_new
    else:
        raise ValueError("optimizer_config.type should be 'old' or 'new'")
    spring_step = get_spring_step(
        log_psi_apply,
        optimizer_config.damping,
        optimizer_config.mu,
    )

    descent_optimizer = optax.sgd(
        learning_rate=learning_rate_schedule, momentum=0, nesterov=False
    )

    def prev_update(optimizer_state):
        return optimizer_state[0].trace

    def optimizer_apply(energy, local_energies, params, optimizer_state, data):
        positions = get_position_fn(data)

        centered_local_energies = local_energies - energy
        grad = spring_step(
            centered_local_energies,
            params,
            prev_update(optimizer_state),
            data["atoms_position"],
            positions,
        )
        updates, optimizer_state = descent_optimizer.update(
            grad, optimizer_state, params
        )
        if optimizer_config.constrain_norm:
            updates = constrain_norm(
                updates,
                optimizer_config.norm_constraint,
            )
        params = optax.apply_updates(params, updates)
        return params, optimizer_state
    update_param_fn = construct_spring_update_param_fn(
        energy_and_statistics_fn,
        optimizer_apply,
        get_position_fn=get_position_fn,
        update_data_fn=update_data_fn,
        record_param_l1_norm=record_param_l1_norm,
        apply_pmap=apply_pmap,
    )
    optimizer_state = initialize_optax_optimizer(
        descent_optimizer, params, apply_pmap=apply_pmap
    )

    return update_param_fn, optimizer_state


def get_spring_step_new(
    log_psi_apply: ModelApply[P],
    damping: chex.Scalar = 0.001,
    mu: chex.Scalar = 0.99,
):
    """Get the SPRING update function."""
    def joint_log_psi_apply(params,joint_x):
        xp=joint_x[:2,:]
        xe=joint_x[2:,:]
        # print("xp:",xp)
        # print("xe:",xe)
        return log_psi_apply(params,xp,xe)
    joint_log_psi_apply_vmap=jax.vmap(joint_log_psi_apply,in_axes=(None,0))
    kernel_fn = nt.empirical_kernel_fn(joint_log_psi_apply_vmap, vmap_axes=0, trace_axes=())

    def spring_step(
        centered_energies: P,
        params: P,
        prev_grad,
        atoms_positions: Array,
        positions: Array,
    ) -> Tuple[Array, P]:
        nchains = positions.shape[1]*positions.shape[0]
        joint_positions = jnp.reshape(positions, (nchains, *positions.shape[-2:]))
        joint_atoms_positions = jnp.repeat(atoms_positions[:, None, ...], positions.shape[1], axis=1).reshape(nchains, *atoms_positions.shape[-2:])
        joint_x=jnp.concatenate([joint_atoms_positions,joint_positions],axis=-2)
        # print("joint_x:",joint_x.shape)
        mu_prev = jax.tree_map(lambda x: mu * x, prev_grad)
        ones = jnp.ones((nchains, 1))

        # Calculate T = Ohat @ Ohat^T using neural-tangents
        # Some GPUs, particularly A100s and A5000s, can exhibit large numerical
        # errors in these calculations. As a result, we explicitly symmetrize T
        # and, rather than using a Cholesky solver to solve against T, we
        # calculate its eigendecomposition and explicitly fix any negative
        # eigenvalues. We then use the fixed and regularized igendecomposition
        # to solve against T. This appears to be more stable than Cholesky
        # in practice.
        T = kernel_fn(joint_x,joint_x, "ntk", params) / nchains
        T = T - jnp.mean(T, axis=0, keepdims=True)
        T = T - jnp.mean(T, axis=1, keepdims=True)
        T = T + ones @ ones.T / nchains
        T = (T + T.T) / 2
        Tvals, Tvecs = jnp.linalg.eigh(T)
        Tvals = jnp.maximum(Tvals, 0) + damping

        epsilon_bar = centered_energies.reshape((-1,)) / jnp.sqrt(nchains)
        O_prev = jax.jvp(
            joint_log_psi_apply_vmap,
            (params, joint_x),
            (mu_prev, jnp.zeros_like(joint_x)),
        )[1] / jnp.sqrt(nchains)
        Ohat_prev = O_prev - jnp.mean(O_prev, axis=0, keepdims=True)
        epsilon_tilde = epsilon_bar - Ohat_prev

        zeta = Tvecs @ jnp.diag(1 / Tvals) @ Tvecs.T @ epsilon_tilde
        zeta_hat = zeta - jnp.mean(zeta)
        dtheta_residual = jax.vjp(joint_log_psi_apply_vmap, params, joint_x)[1](zeta_hat)[0]
        # memory_show()
        return jax.tree_map(
            lambda dt, mup: dt / jnp.sqrt(nchains) + mup, dtheta_residual, mu_prev
        )

    return spring_step

def get_spring_step_old(
    log_psi_apply: ModelApply[P],
    damping: chex.Scalar = 0.001,
    mu: chex.Scalar = 0.99,
):

    def raveled_log_psi_grad(params: P,ion_pos: Array, positions: Array) -> Array:
        log_grads = jax.grad(log_psi_apply)(params,ion_pos, positions)
        return jax.flatten_util.ravel_pytree(log_grads)[0]

    batch_raveled_log_psi_grad = jax.vmap(jax.vmap(raveled_log_psi_grad, in_axes=(None, None,0)),in_axes=(None,0,0))

    def spring_update_fn(
        centered_energies: P,
        params: P,
        prev_grad,
        atoms_positions: Array,
        positions: Array,
    ) -> Tuple[Array, P]:
        nchains = positions.shape[1]*positions.shape[0]
        # print_memory_usage("开始spring_update_fn")
        # logging.info(f"nchains: {nchains}, positions形状: {positions.shape}")
        prev_grad, unravel_fn = jax.flatten_util.ravel_pytree(prev_grad)
        prev_grad_decayed = mu * prev_grad  #(nparams,)
        # print_memory_usage("计算log_psi_grads前")
        log_psi_grads_pre = batch_raveled_log_psi_grad(params,atoms_positions, positions) 
        # logging.info(f"log_psi_grads_pre形状: {log_psi_grads_pre.shape}")
        # print_memory_usage("计算log_psi_grads后")
        W,B,nparams=log_psi_grads_pre.shape
        log_psi_grads=log_psi_grads_pre.reshape((W*B,nparams)) /jnp.sqrt(nchains)  #(W*B,nparams)
        Ohat = log_psi_grads - jnp.mean(log_psi_grads, axis=0, keepdims=True)  #(W*B,nparams)
        # logging.info(f"Ohat形状: {Ohat.shape}")
        # print_memory_usage("计算Ohat后")
        T = Ohat @ Ohat.T  #(W*B,W*B)
        # logging.info(f"T矩阵形状: {T.shape}")
        # print_memory_usage("计算T矩阵后")  # 若此处内存骤增到接近总容量，则是溢出点
        ones = jnp.ones((nchains, 1)) #(W*B,1)
        T_reg = T + ones @ ones.T / nchains + damping * jnp.eye(nchains)  #(W*B,W*B)
        # logging.info(f"T_reg形状: {T_reg.shape}")
        # print_memory_usage("计算T_reg后")
        epsilon_bar = centered_energies.reshape((-1,)) / jnp.sqrt(nchains) #(W*B,)
        epsion_tilde = epsilon_bar - Ohat @ prev_grad_decayed   #(W*B,)
        dtheta_residual = Ohat.T @ jax.scipy.linalg.solve(T_reg, epsion_tilde, assume_a="pos") #(nparams,)
        # print_memory_usage("计算solve后")
        # print(f"dtheta_residual:{dtheta_residual.shape}")   #(nparams,)
        # print(f"prev_grad_decayed:{prev_grad_decayed.shape}")   #(nparams,)
        SR_G = dtheta_residual + prev_grad_decayed

        return unravel_fn(SR_G)

    return spring_update_fn

def constrain_norm(
    grad: P,
    norm_constraint: chex.Numeric = 0.001,
) -> P:
    """Euclidean norm constraint."""
    sq_norm_scaled_grads = tree_inner_product(grad, grad)

    # Sync the norms here, see:
    # https://github.com/deepmind/deepmind-research/blob/30799687edb1abca4953aec507be87ebe63e432d/kfac_ferminet_alpha/optimizer.py#L585
    sq_norm_scaled_grads = pmean_if_pmap(sq_norm_scaled_grads)

    norm_scale_factor = jnp.sqrt(norm_constraint / sq_norm_scaled_grads)
    coefficient = jnp.minimum(norm_scale_factor, 1)
    constrained_grads = multiply_tree_by_scalar(grad, coefficient)

    return constrained_grads


def raveled_log_psi_grad(params: P, ion_pos: Array, positions: Array) -> Array:
    # 原单个chain的梯度计算（不变）
    log_grads = jax.grad(log_psi_apply)(params, ion_pos, positions)
    return jax.flatten_util.ravel_pytree(log_grads)[0]

# 单批梯度计算：处理一批positions（形状(W, batch_size, N, 3)）
def process_batch(params, atoms_pos, positions_batch):
    # 对单批内的chain用vmap计算梯度（仅对batch_size维度vmap）
    batch_grads = jax.vmap(jax.vmap(raveled_log_psi_grad, in_axes=(None, None, 0)),in_axes=(None, 0, 0)
    )(params, atoms_pos, positions_batch)  # 输出形状(W, batch_size, nparams)
    return batch_grads.reshape(-1, batch_grads.shape[-1])  # 合并为(512, nparams)

# 用scan分批次计算并合并结果
def batch_raveled_log_psi_grad_split(params, atoms_pos, positions, batch_size=128):
    # 拆分输入为批次列表
    positions_batches = jnp.split(positions, indices_or_sections=positions.shape[1]//batch_size, axis=1)
    
    # 用scan循环处理所有批次，累积结果
    def scan_fn(acc, batch):
        # 计算当前批次的梯度
        batch_grads = process_batch(params, atoms_pos, batch)
        # 拼接当前批次结果到累积器
        return jnp.concatenate([acc, batch_grads], axis=0), None
    
    # 初始化累积器（空数组），开始scan
    total_grads, _ = jax.lax.scan(
        scan_fn,
        init=jnp.empty((0, 1961218), dtype=jnp.float64),  # 与nparams匹配的空数组
        xs=positions_batches
    )
    return total_grads  # 最终形状(2048, 1961218)，与原结果一致


def construct_spring_update_param_fn_with_accum(
    energy_and_statistics_fn,
    optimizer_apply,
    get_position_fn: GetPositionFromData[D],
    update_data_fn: UpdateDataFn[D, P],
    acc_steps: int = 4,  # 梯度累积步数
    apply_pmap: bool = True,
    record_param_l1_norm: bool = False,
) -> UpdateParamFn[P, D, S]:
    def update_param_fn(params, data, optimizer_state, key):
        position = get_position_fn(data)
        atoms_position = data["atoms_position"]
        nwalker = atoms_position.shape[0]
        energy, local_energies, stats = energy_and_statistics_fn(params, atoms_position, position)

        params, optimizer_state, is_updated = optimizer_apply(
            energy,
            local_energies,
            params,
            optimizer_state,
            data,
            acc_steps
        )

        def update(_):
            return update_data_fn(data, params)
        
        def no_update(_):
            return data
        
        data = jax.lax.cond(
            is_updated,  # 条件：JAX布尔数组
            update,  
            no_update,  
            operand=None  
        )
        # metrics = {
        #         "energy": energy, 
        #         "variance": stats["variance"],
        #         "kinetic": stats["kinetic"],
        #         "ei_potential": stats["ei_potential"],
        #         "ee_potential": stats["ee_potential"],
        #         "ii_potential": stats["ii_potential"],
        #         "multi_energy": stats["multi_energy"],
        #     }
        # metrics = update_metrics_with_noclip(
        #         stats["energy_noclip"],
        #         stats["variance_noclip"],
        #         metrics,
        #     )
        # if record_param_l1_norm:
        #     metrics["param_l1_norm"] = tree_reduce_l1(params)

        def create_valid_metrics(_):
            metrics = {
                "energy": energy, 
                "variance": stats["variance"],
                "kinetic": stats["kinetic"],
                "ei_potential": stats["ei_potential"],
                "ee_potential": stats["ee_potential"],
                "ii_potential": stats["ii_potential"],
                "multi_energy": stats["multi_energy"],
            }
            metrics = update_metrics_with_noclip(stats["energy_noclip"], stats["variance_noclip"], metrics)
            if record_param_l1_norm:
                metrics["param_l1_norm"] = tree_reduce_l1(params)
            return metrics

        def create_empty_metrics(_):
            empty_metrics = {
                "energy": EMPTY_MARKER,
                "variance": jnp.nan,
                "kinetic": jnp.nan,
                "ei_potential": jnp.nan,
                "ee_potential": jnp.nan,
                "ii_potential": jnp.nan,
                "multi_energy": jnp.full(shape=stats["multi_energy"].shape, fill_value=jnp.nan, dtype=stats["multi_energy"].dtype),
                "energy_noclip": jnp.nan,
                "variance_noclip": jnp.nan,
            }
            if record_param_l1_norm:
                metrics["param_l1_norm"] = jnp.nan
            return empty_metrics


        operand = (energy, stats, record_param_l1_norm, params)
        metrics = jax.lax.cond(
            is_updated,
            create_valid_metrics,  # 真分支：有效metrics
            create_empty_metrics,  # 假分支：空metrics
            operand=None,
        )

        return params, data, optimizer_state, metrics, key

    traced_fn = make_traced_fn_with_single_metrics(update_param_fn, apply_pmap)

    return traced_fn


def initialize_spring_with_accum(
    log_psi_apply: ModelApply[P],
    energy_and_statistics_fn,
    params: P,
    get_position_fn: GetPositionFromData[D],
    update_data_fn: UpdateDataFn[D, P],
    learning_rate_schedule: LearningRateSchedule,
    optimizer_config: ConfigDict,
    acc_steps: int = 4,  # 梯度累积步数
    record_param_l1_norm: bool = False,
    apply_pmap: bool = True,
) -> Tuple[Callable, Tuple[optax.OptState, P, int]]:
    """初始化带梯度累积的SPRING优化器"""
    if optimizer_config.type == "old":
        get_spring_step = get_spring_step_old
    elif optimizer_config.type == "new":
        get_spring_step = get_spring_step_new
    else:
        raise ValueError("optimizer_config.type should be 'old' or 'new'")
    spring_step = get_spring_step(
        log_psi_apply,
        optimizer_config.damping,
        optimizer_config.mu,
    )

    descent_optimizer = optax.sgd(
        learning_rate=learning_rate_schedule, momentum=0, nesterov=False
    )

    def prev_update(optimizer_state):
        return optimizer_state[0].trace

    def optimizer_apply(energy, local_energies, params, optimizer_state, data, acc_steps):
        opt_state, grad_acc, acc_count = optimizer_state
        positions = get_position_fn(data)

        centered_local_energies = local_energies - energy
        current_grad = spring_step(
            centered_local_energies,
            params,
            prev_update(opt_state),
            data["atoms_position"],
            positions,
        )

        # if grad_acc is None:
            # grad_acc = jax.tree_map(lambda x: jnp.zeros_like(x), current_grad)

        grad_acc = jax.tree_map(lambda acc, g: acc + g, grad_acc, current_grad)
        acc_count = jnp.add(acc_count, 1)

        def true_branch(_):
            avg_grad = jax.tree_map(lambda g: g / float(acc_steps), grad_acc)
            updates, new_opt_state = descent_optimizer.update(
                avg_grad, opt_state, params
            )
            if optimizer_config.constrain_norm:
                updates = constrain_norm(
                    updates,
                    optimizer_config.norm_constraint,
                )
            new_params = optax.apply_updates(params, updates)
            # 重置累积器
            new_grad_acc = jax.tree_map(lambda x: jnp.zeros_like(x), avg_grad)
            new_acc_count = 0
            is_updated = jnp.array(True)  # 用JAX布尔数组替代Python布尔值
            return new_params, new_opt_state, new_grad_acc, new_acc_count, is_updated

        def false_branch(_):
            new_params = params
            new_opt_state = opt_state
            new_grad_acc = grad_acc
            new_acc_count = acc_count
            is_updated = jnp.array(False)  # 用JAX布尔数组替代Python布尔值
            return new_params, new_opt_state, new_grad_acc, new_acc_count, is_updated

        # 用jax.lax.cond判断条件，执行对应分支
        # 条件：acc_count >= acc_steps（用JAX函数比较，确保追踪兼容性）
        new_params, new_opt_state, new_grad_acc, new_acc_count, is_updated = jax.lax.cond(
            jnp.greater_equal(acc_count, acc_steps),  # 条件（JAX数组布尔值）
            true_branch,  # 条件为真时执行
            false_branch,  # 条件为假时执行
            operand=None  # 传给分支函数的额外参数（这里不需要）
        )

        # 更新优化器状态
        new_optimizer_state = (new_opt_state, new_grad_acc, new_acc_count)
        return new_params, new_optimizer_state, is_updated

    # 构造带梯度累积的参数更新函数
    update_param_fn = construct_spring_update_param_fn_with_accum(
        energy_and_statistics_fn,
        optimizer_apply,
        get_position_fn=get_position_fn,
        update_data_fn=update_data_fn,
        acc_steps=acc_steps,
        record_param_l1_norm=record_param_l1_norm,
        apply_pmap=apply_pmap,
    )
    optimizer_state = initialize_optax_optimizer(
        descent_optimizer, params, apply_pmap=apply_pmap
    )
    optimizer_state = (optimizer_state, 
                        jax.tree_map(lambda x: jnp.zeros_like(x),prev_update(optimizer_state)), 
                        jnp.array(0, dtype=jnp.int32)
                        )
    # if apply_pmap:
    #     optimizer_state = jax.pmap(lambda _: optimizer_state)(jax.arange(jax.device_count()))

    return update_param_fn, optimizer_state
