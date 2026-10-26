"""Distill a trained real-valued gaoqiao wavefunction into a fresh gaoqiao net.

The teacher checkpoint and its *original* config are copied into every result
directory.  The student is trained on signed wavefunction values, not orbital
matrices or log magnitudes.  This is deliberately separate from HF pretraining.

Run ``python -m vmcnet.train.pretrain_from_trained_net --help`` for options.
"""

import argparse
import csv
import json
import logging
import math
import shutil
from pathlib import Path
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from vmcnet.gaoqiao.build import build_network
from vmcnet.mcmc import dynamic_width_position_amplitude as dwpa
from vmcnet.physics import kinetic, potential
from vmcnet.physics.core import initialize_molecular_pos
from vmcnet.train.config_compat import complete_config
from vmcnet.utils import io


class Molecule(NamedTuple):
    ions: jax.Array
    charges: jax.Array
    nspins: tuple[int, int]
    single_nspins: jax.Array


def validate_problem_match(teacher: Any, student: Any) -> None:
    """检查原始配置的完整 problem 和精度；不补字段、不改写任一配置。

    JSON 列表与配置元组视为相同；单分子 ion_pos 的 (N,3) 与
    (1,N,3) 也视为相同。字段缺失不能通过默认值掩盖，必须明确报错。
    """
    if "problem" not in teacher or "problem" not in student:
        raise ValueError("teacher/student 必须显式提供 problem，不能自动补全物理体系。")
    differences = []
    for field in sorted(set(teacher.problem) | set(student.problem)):
        if field not in teacher.problem or field not in student.problem:
            differences.append(f"problem.{field} 缺失")
            continue
        left, right = teacher.problem[field], student.problem[field]
        if field == "ion_pos":
            left, right = _ions_from_config(teacher), _ions_from_config(student)
        if not np.array_equal(np.asarray(left), np.asarray(right)):
            differences.append(f"problem.{field}: teacher={left!r}, student={right!r}")
    if differences:
        raise ValueError("teacher/student 的 problem 不一致，停止预训练：\n" +
                         "\n".join(differences))
    if "dtype" not in teacher or "dtype" not in student or teacher.dtype != student.dtype:
        raise ValueError("teacher/student 必须显式提供相同 dtype；不会自动继承。")


def _ions_from_config(config: Any) -> np.ndarray:
    positions = np.asarray(config.problem.ion_pos, dtype=float)
    if positions.ndim == 3 and positions.shape[0] == 1:
        positions = positions[0]
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("目前只支持单个固定几何的分子，ion_pos 应为 (N,3) 或 (1,N,3)。")
    return positions


def validate_configs(teacher: Any, student: Any) -> Molecule:
    """Check the resolved student matches the teacher's physical system."""
    validate_problem_match(teacher, student)
    if teacher.wfn_type != "gaoqiao" or student.wfn_type != "gaoqiao":
        raise ValueError("teacher 和 student 的 wfn_type 都必须是 'gaoqiao'。")
    if teacher.gq.type != "ef_shrd_sym":
        raise ValueError("当前 teacher 要求 gq.type='ef_shrd_sym'。")
    if teacher.dtype != student.dtype:
        raise ValueError("teacher 和 student 必须使用同一 dtype。")
    if bool(teacher.gq.do_complex) or bool(student.gq.do_complex):
        raise ValueError("带符号实波函数损失要求 do_complex=False。")
    if (student.gq.type == "so3"
            and str(student.gq.so3_params.basis).lower() == "complex"):
        raise ValueError("当前预训练要求实值 SO(3) 输出；请用 real 或 complex_to_real。")

    teacher_ions = _ions_from_config(teacher)
    student_ions = _ions_from_config(student)
    if not np.array_equal(teacher_ions, student_ions):
        raise ValueError("teacher/student 的核坐标或核顺序不同。")
    teacher_charges = np.asarray(teacher.problem.ion_charges, dtype=float)
    student_charges = np.asarray(student.problem.ion_charges, dtype=float)
    if not np.array_equal(teacher_charges, student_charges):
        raise ValueError("teacher/student 的核电荷或核顺序不同。")
    nspins = tuple(int(n) for n in teacher.problem.nspins)
    if nspins != tuple(int(n) for n in student.problem.nspins) or len(nspins) != 2:
        raise ValueError("teacher/student 的两块自旋电子数必须相同。")
    allocations = np.asarray(teacher.problem.single_nspins, dtype=int)
    if allocations.shape != (len(teacher_charges), 2):
        raise ValueError("single_nspins 应为 (核数, 2)。")
    if not np.array_equal(allocations.sum(axis=0), nspins):
        raise ValueError("single_nspins 按核分配的电子数与 nspins 不一致。")
    if not np.array_equal(allocations,
                          np.asarray(student.problem.single_nspins, dtype=int)):
        raise ValueError("teacher/student 的 single_nspins 必须相同。")
    dtype = jnp.float64 if teacher.dtype == "float64" else jnp.float32
    return Molecule(jnp.asarray(teacher_ions, dtype),
                    jnp.asarray(teacher_charges, dtype), nspins,
                    jnp.asarray(allocations, dtype=jnp.int32))


def construct_network(config: Any, molecule: Molecule, key: jax.Array):
    """Use the same build_network arguments as the gaoqiao branch of runners.py."""
    gq = config.gq
    attention = None
    if gq.do_attn:
        attention = {
            "qkdim": gq.attn_nchnl,
            "nhead": gq.attn_nhead,
            "do_gate": gq.attn_do_gate,
            "do_lnorm": gq.attn_do_lnorm,
        }
    features = {
        "do_act": gq.feat_do_act,
        "act_func": gq.feat_act_func,
        "numb_divid": gq.feat_numb_divid,
        "scale": gq.scale if gq.scale != 1.0 else [],
        "rescale": gq.rescale_input,
    }
    return build_network(
        n=sum(molecule.nspins),
        charges=molecule.charges,
        nspins=molecule.nspins,
        key=key,
        ndet=gq.ndet,
        depth=gq.wfn_depth,
        h1=gq.h1,
        h2=gq.h2,
        nh=gq.nh,
        do_complex=False,
        gq_type=gq.type,
        envelope_type=gq.envelope_type,
        attn=attention,
        feat_params=features,
        det_mode=gq.det_mode,
        jastrow_type=gq.jastrow_type,
        jastrow_mlp_nlayer=gq.jastrow_mlp_nlayer,
        jastrow_mlp_ndim=gq.jastrow_mlp_ndim,
        RHF=gq.RHF,
        activation_type=gq.activation_type,
        jastrow_activation_type=gq.jastrow_activation_type,
        dp_type=gq.dp_type,
        alpha=gq.alpha,
        so3_params=gq.so3_params.to_dict() if gq.type == "so3" else None,
        dtype=config.dtype,
    )[:2]


def check_checkpoint_params(initial: Any, loaded: Any) -> Any:
    """Fail early when config.json does not describe its paired checkpoint."""
    if jax.tree_util.tree_structure(initial) != jax.tree_util.tree_structure(loaded):
        raise ValueError("teacher checkpoint 的参数结构与 teacher config 不匹配。")
    initial_leaves = jax.tree_util.tree_leaves(initial)
    loaded_leaves = jax.tree_util.tree_leaves(loaded)
    for index, (expected, actual) in enumerate(zip(initial_leaves, loaded_leaves)):
        if np.shape(expected) != np.shape(actual):
            raise ValueError(f"teacher 参数叶子 {index} 的形状与配置不符："
                             f"{np.shape(actual)} != {np.shape(expected)}")
    return jax.tree_util.tree_map(jnp.asarray, loaded)


def check_checkpoint_amplitudes(network: Callable, params: Any,
                                checkpoint_data: Any,
                                molecule: Molecule) -> bool:
    """Check that today's reconstructed teacher matches stored log|psi|.

    Parameter shapes alone cannot detect a changed activation, particle-type
    encoding, or other parameter-free change to an old network definition.
    Saved walker amplitudes provide a direct check at actual configurations.
    Return False only if that reference data was not saved in the checkpoint.
    """
    if not isinstance(checkpoint_data, dict):
        return False
    walker = checkpoint_data.get("walker_data")
    if not isinstance(walker, dict):
        return False
    if "elec_position" not in walker or "amplitude" not in walker:
        return False
    positions = np.asarray(walker["elec_position"])
    reference = np.asarray(walker["amplitude"])
    expected_shape = (sum(molecule.nspins), 3)
    if positions.shape[-2:] != expected_shape or reference.shape != positions.shape[:-2]:
        raise ValueError("teacher checkpoint 中的 walker 坐标与幅值形状不匹配。")
    positions = positions.reshape(-1, *expected_shape)
    reference = reference.reshape(-1)
    good = np.isfinite(positions).all(axis=(1, 2)) & np.isfinite(reference)
    if not np.any(good):
        raise ValueError("teacher checkpoint 中没有可用于校验的有限 walker 幅值。")
    indices = np.flatnonzero(good)[:min(4, int(np.sum(good)))]
    actual = jax.vmap(lambda x: network(params, x, molecule.ions)[1])(
        jnp.asarray(positions[indices], dtype=molecule.ions.dtype))
    atol = 2e-4 if molecule.ions.dtype == jnp.float32 else 1e-8
    if not np.allclose(np.asarray(actual), reference[indices], rtol=1e-5,
                       atol=atol, equal_nan=False):
        raise ValueError(
            "teacher checkpoint 保存的 log|psi| 与当前代码重建的网络不同；"
            "旧配置补字段不足以保证原波函数定义不变，请核对训练时的代码版本。")
    return True


def make_metropolis_sampler(network: Callable, ions: jax.Array,
                            nsteps: int, proposal_std: float,
                            nmoves_per_update: int = 100):
    """复用主 VMC 的全电子高斯提议、整构型接受和动态步宽。

    返回的第四项是步宽及接受率累计状态；调用者须传给下一次采样，
    让热化和训练之间的步宽调节连续。teacher/student 各自维护状态。
    每次重新计算当前幅值，因为 student 参数可能刚刚更新。
    """
    def batched_logabs(params, atoms, positions):
        return jax.vmap(lambda x: network(params, x, atoms)[1])(positions)

    metrop_step = dwpa.make_dynamic_pos_amp_gaussian_step(
        batched_logabs, nmoves_per_update,
        dwpa.make_threshold_adjust_std_move(0.5, 0.05, 0.1))

    @jax.jit
    def sample(params, positions, key, move_metadata=None):
        data = dwpa.make_dynamic_width_position_amplitude_data(
            ions[None], positions[None], batched_logabs(params, ions, positions)[None],
            std_move=jnp.full((1,), proposal_std, dtype=positions.dtype),
            move_acceptance_sum=jnp.zeros((1,), dtype=positions.dtype),
            # 与主采样器计数重置时的整数精度一致（x64 开启时为 int64）。
            moves_since_update=jnp.full((1,), 0))
        if move_metadata is not None:
            data["move_metadata"] = move_metadata

        def move(carry, _):
            current, rng = carry
            acceptance, current, rng = metrop_step(params, current, rng)
            return (current, rng), acceptance

        # 主 VMC kernel 自带分子轴 vmap；这里只有一个固定几何。
        (data, keys), accepted = jax.lax.scan(move, (data, key[None]), None, length=nsteps)
        return (data["walker_data"]["elec_position"][0], keys[0],
                jnp.mean(accepted), data["move_metadata"])

    return sample


def signed_amplitude_loss(teacher_sign, teacher_logabs, student_sign,
                          student_logabs, mode: str, student_log_scale=0.0):
    """Compute a signed loss without directly exponentiating raw log|psi|.

    A common stopped scale changes the *numerical size* of the batch loss, but
    not its minimizer.  ``signed_mse`` still fits the teacher's global scale.
    ``scale_aligned`` fits an optimal global signed multiplier in each batch.
    """
    student_logabs = student_logabs + student_log_scale
    if mode == "scale_aligned":
        # This mode is invariant to an overall scale.  Match its numerical
        # range before exponentiation, or the student norm can underflow for
        # a randomly initialized many-electron determinant.
        numeric_shift = jax.lax.stop_gradient(
            jnp.median(teacher_logabs - student_logabs))
        student_logabs = student_logabs + numeric_shift
    shift = jax.lax.stop_gradient(jnp.maximum(jnp.max(teacher_logabs),
                                              jnp.max(student_logabs)))
    target = teacher_sign * jnp.exp(teacher_logabs - shift)
    student = student_sign * jnp.exp(student_logabs - shift)
    target_norm = jax.lax.stop_gradient(jnp.mean(target * target))
    tiny = jnp.finfo(target.dtype).tiny
    if mode == "signed_mse":
        alignment = jnp.asarray(1.0, dtype=target.dtype)
    elif mode == "scale_aligned":
        alignment = jnp.mean(target * student) / (
            jnp.mean(student * student) + tiny)
    else:
        raise ValueError("loss_mode 必须是 signed_mse 或 scale_aligned。")
    residual = alignment * student - target
    loss = jnp.mean(residual * residual) / (target_norm + tiny)
    sign_agreement = jnp.mean(teacher_sign == student_sign)
    return loss, (alignment, sign_agreement)


def make_training_step(teacher_network: Callable,
                       student_network: Callable, ions: jax.Array,
                       loss_mode: str, optimizer: optax.GradientTransformation):
    """Create compiled update/evaluation functions with cached teacher targets."""
    evaluate_teacher = jax.jit(jax.vmap(
        lambda params, x: teacher_network(params, x, ions), in_axes=(None, 0)))
    evaluate_student = jax.vmap(lambda params, x: student_network(params, x, ions),
                                in_axes=(None, 0))

    def objective(trainable, positions, teacher_targets):
        teacher_sign, teacher_logabs = teacher_targets
        student_sign, student_logabs = evaluate_student(
            trainable["network"], positions)
        return signed_amplitude_loss(
            teacher_sign, teacher_logabs, student_sign, student_logabs,
            loss_mode, trainable["log_scale"])

    @jax.jit
    def step(trainable, optimizer_state, positions, teacher_targets):
        (loss, aux), grads = jax.value_and_grad(objective, has_aux=True)(
            trainable, positions, teacher_targets)
        updates, optimizer_state = optimizer.update(grads, optimizer_state,
                                                    trainable)
        trainable = optax.apply_updates(trainable, updates)
        return trainable, optimizer_state, loss, aux

    return step, jax.jit(objective), evaluate_teacher


def _fresh_positions(config: Any, molecule: Molecule, count: int,
                     key: jax.Array):
    key, positions = initialize_molecular_pos(
        key, count, molecule.ions[None], molecule.charges,
        sum(molecule.nspins), molecule.single_nspins,
        config.vmc.init_width, dtype=molecule.ions.dtype)
    return key, positions[0]


def initial_teacher_positions(checkpoint_data: Any, config: Any,
                              molecule: Molecule, count: int,
                              key: jax.Array):
    """Prefer equilibrated teacher walkers; fall back to molecular initialization.

    A checkpoint can have extra device and geometry axes.  This first version
    supports one geometry, so flatten every leading walker/device axis only
    after checking the last two electron dimensions and recorded ion positions.
    """
    if isinstance(checkpoint_data, dict):
        walker_data = checkpoint_data.get("walker_data")
        if isinstance(walker_data, dict) and "elec_position" in walker_data:
            old_atoms = checkpoint_data.get("atoms_position")
            if old_atoms is not None:
                old_atoms = np.asarray(old_atoms)
                if (old_atoms.shape[-2:] != tuple(molecule.ions.shape)
                        or not np.allclose(old_atoms.reshape(-1, *molecule.ions.shape),
                                           np.asarray(molecule.ions))):
                    raise ValueError("teacher checkpoint 的核坐标与 teacher config 不同。")
            saved = np.asarray(walker_data["elec_position"])
            if (saved.ndim >= 3
                    and saved.shape[-2:] == (sum(molecule.nspins), 3)):
                saved = saved.reshape(-1, sum(molecule.nspins), 3)
                saved = saved[np.all(np.isfinite(saved), axis=(1, 2))]
                if len(saved):
                    key, choose_key, noise_key = jax.random.split(key, 3)
                    choices = jax.random.choice(
                        choose_key, len(saved), (count,),
                        replace=len(saved) < count)
                    positions = jnp.asarray(saved, dtype=molecule.ions.dtype)[choices]
                    if len(saved) < count:
                        # Replicated old walkers need decorrelation; burn-in
                        # below still runs under the frozen teacher density.
                        positions += 0.01 * jax.random.normal(
                            noise_key, positions.shape, dtype=positions.dtype)
                    return key, positions, "checkpoint"
    key, positions = _fresh_positions(config, molecule, count, key)
    return key, positions, "fresh"


def _snapshot_inputs(args: argparse.Namespace, output: Path,
                     teacher_config: Any, student_config: Any,
                     teacher_added: tuple[str, ...]):
    teacher_dir = output / "teacher_source"
    teacher_dir.mkdir()
    shutil.copy2(args.teacher_checkpoint, teacher_dir / "teacher.npz")
    shutil.copy2(args.teacher_config, teacher_dir / "config_original.json")
    io.save_config_dict_to_json(teacher_config, str(teacher_dir), "config_resolved")
    shutil.copy2(args.student_config, output / "student_config_original.json")
    # Standard VMC reload expects config.json in the checkpoint directory.
    # Send the subsequent VMC run to a sibling directory, not to the old
    # student's run directory recorded in a reused student config.
    student_config.logdir = str(output.parent)
    student_config.base_logdir = str(output.parent)
    student_config.subfolder_name = output.name + "_vmc"
    student_config.save_to_current_datetime_subfolder = False
    student_config.pretrain.iterations = 0
    io.save_config_dict_to_json(student_config, str(output), "config")
    metadata = {
        "pretrain_arguments": {key: str(value) if isinstance(value, Path) else value
                               for key, value in vars(args).items()},
        "student_molecule_and_dtype_source": str(args.student_config),
        "teacher_student_problem_validated": True,
        "sampling_proposal": "main VMC all-electron Gaussian with dynamic width",
        "nmoves_per_width_update": student_config.vmc.nmoves_per_width_update,
        "teacher_fields_filled_from_defaults": teacher_added,
        "resume_vmc": "Use --reload.new_optimizer_state=True with pretrained_student.npz.",
    }
    with (output / "pretrain_settings.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)


def _save_student_checkpoint(output: Path, name: str, config: Any,
                             molecule: Molecule, student_network: Callable,
                             student_params: Any, positions: jax.Array,
                             key: jax.Array):
    batched_logabs = jax.vmap(lambda x: student_network(
        student_params, x, molecule.ions)[1])(positions)
    nwalkers = 1
    data = dwpa.make_dynamic_width_position_amplitude_data(
        molecule.ions[None], positions[None], batched_logabs[None],
        std_move=jnp.full((nwalkers,), config.vmc.std_move,
                           dtype=molecule.ions.dtype),
        move_acceptance_sum=jnp.zeros((nwalkers,), dtype=molecule.ions.dtype),
        moves_since_update=jnp.zeros((nwalkers,), dtype=jnp.int32),
    )
    # A distillation Adam state is not a Spring/KFAC VMC optimizer state.
    io.save_vmc_state(str(output), name, (0, data, student_params, None, key))


def _fold_so3_log_scale(params: Any, log_scale: jax.Array) -> Any:
    """Absorb a global wavefunction factor into the SO(3) determinant weights.

    ``alpha`` enters the final determinant sum linearly.  Multiplying every
    alpha by exp(log_scale) multiplies the signed wavefunction by that factor
    without changing its nodes or sampling distribution.
    """
    alpha = params["so3_det"]["alpha"]
    dtype = next(iter(alpha.values())).dtype
    max_log = math.log(float(jnp.finfo(dtype).max))
    scale_value = float(log_scale)
    if abs(scale_value) >= max_log - 5:
        raise FloatingPointError(
            "整体幅度超出当前 dtype 可安全存储的范围；请使用 float64 "
            "或检查 teacher/student 的初始尺度差异。")
    scale = jnp.exp(jnp.asarray(log_scale, dtype=dtype))
    result = dict(params)
    determinant = dict(params["so3_det"])
    determinant["alpha"] = {key: value * scale
                            for key, value in alpha.items()}
    result["so3_det"] = determinant
    return result


def _fold_student_log_scale(params: Any, log_scale: jax.Array,
                            config: Any) -> Any:
    """将预训练的整体幅度存入普通网络参数，保持 checkpoint 可直接用于 VMC。

    SO(3) 仍缩放最终 alpha。其他 gaoqiao 网络的轨道输出为线性层：
    将所有自旋的轨道权重（及偏置）乘 exp(log_scale / N)，每个 N 电子
    行列式就乘 exp(log_scale)。RHF 使用共享的 RHF_orbital，同样只缩放一次。
    不新增参数叶子，不改变 envelope、Jastrow 或网络结构。
    """
    if config.gq.type == "so3":
        return _fold_so3_log_scale(params, log_scale)
    if config.gq.det_mode != "det":
        raise ValueError("普通 gaoqiao student 的尺度保存要求 det_mode='det'。")
    count = sum(int(n) for n in config.problem.nspins)
    field = "RHF_orbital" if config.gq.RHF else "orbital"
    layers = params.get(field)
    if count <= 0 or not layers:
        raise ValueError(f"无法将整体幅度写入 student 的 {field} 参数。")
    dtype = jax.tree_util.tree_leaves(layers)[0].dtype
    scale = jnp.exp(jnp.asarray(log_scale / count, dtype=dtype))
    scaled = jax.tree_util.tree_map(lambda value: value * scale, layers)
    if (not bool(jnp.isfinite(scale)) or float(scale) == 0.
            or not all(bool(jnp.all(jnp.isfinite(value)))
                       for value in jax.tree_util.tree_leaves(scaled))):
        raise FloatingPointError("student 轨道整体缩放超出当前 dtype 的有限范围。")
    result = dict(params)
    result[field] = scaled
    return result


def _unique_output_directory(requested: Path) -> Path:
    requested = requested.resolve()
    output = requested
    suffix = 1
    while output.exists():
        output = requested.with_name(f"{requested.name}_{suffix}")
        suffix += 1
    output.mkdir(parents=True)
    return output


def make_energy_components(network: Callable, molecule: Molecule, config: Any):
    """Build the same local kinetic and Coulomb terms used by VMC.

    The overall amplitude scale and sign do not affect local energy away from
    nodes, so the trained network parameters can be used before folding the
    learned global scale into the saved checkpoint.
    """
    log_psi = lambda params, ions, x: network(params, x, ions)[1]
    kinetic_fn = kinetic.create_laplacian_kinetic_energy_new(
        log_psi, config.vmc.fwdlap_inner_size)
    ei_fn = potential.create_electron_ion_coulomb_potential(
        molecule.charges, softening_term=config.problem.ei_softening)
    ee_fn = potential.create_electron_electron_coulomb_potential(
        softening_term=config.problem.ee_softening)
    ii_fn = potential.create_ion_ion_coulomb_potential(molecule.charges)

    @jax.jit
    def components(params, positions):
        return jnp.stack((kinetic_fn(params, molecule.ions, positions),
                          ei_fn(params, molecule.ions, positions),
                          ee_fn(params, molecule.ions, positions),
                          ii_fn(params, molecule.ions, positions)))

    return components


def evaluate_student_energy(network: Callable, params: Any, molecule: Molecule,
                            config: Any, initial_positions: jax.Array,
                            key: jax.Array, args: argparse.Namespace,
                            output: Path) -> None:
    """Estimate the student's energy using fresh samples from its own |psi|².

    This is a debugging estimate, not an independent production VMC run.
    Save every local energy so outliers and non-finite values remain visible.
    """
    nwalkers = min(args.energy_samples, initial_positions.shape[0])
    positions = initial_positions[:nwalkers]
    move_metadata = None
    width_interval = config.vmc.nmoves_per_width_update
    if args.energy_burn_in:
        burn_sampler = make_metropolis_sampler(
            network, molecule.ions, args.energy_burn_in, args.proposal_std, width_interval)
        positions, key, _, move_metadata = burn_sampler(params, positions, key)
    sampler = make_metropolis_sampler(
        network, molecule.ions, args.mcmc_steps, args.proposal_std, width_interval)
    components = make_energy_components(network, molecule, config)
    rows = []
    while len(rows) < args.energy_samples:
        positions, key, _, move_metadata = sampler(params, positions, key, move_metadata)
        for x in positions[:args.energy_samples - len(rows)]:
            terms = np.asarray(components(params, x), dtype=float)
            rows.append((*terms, float(np.sum(terms))))

    names = ("kinetic", "electron_ion", "electron_electron", "ion_ion",
             "local_energy")
    with (output / "energy_samples.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("sample", *names))
        for index, row in enumerate(rows, 1):
            writer.writerow((index, *row))

    values = np.asarray(rows)
    finite = np.all(np.isfinite(values), axis=1)
    valid = values[finite]
    summary = {
        "sampling_distribution": "student |psi|^2",
        "sample_count": len(rows),
        "finite_count": int(np.sum(finite)),
        "nonfinite_count": int(np.sum(~finite)),
        "burn_in_steps": args.energy_burn_in,
        "metropolis_steps_between_samples": args.mcmc_steps,
        "mean_hartree": float(np.mean(valid[:, -1])) if len(valid) else None,
        "std_hartree": float(np.std(valid[:, -1])) if len(valid) else None,
        "min_hartree": float(np.min(valid[:, -1])) if len(valid) else None,
        "max_hartree": float(np.max(valid[:, -1])) if len(valid) else None,
        "mean_components_hartree": {
            name: float(np.mean(valid[:, i])) if len(valid) else None
            for i, name in enumerate(names[:-1])
        },
    }
    with (output / "energy_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    logging.info("学生能量诊断: mean=%s Ha, std=%s Ha, finite=%d/%d; %s",
                 summary["mean_hartree"], summary["std_hartree"],
                 summary["finite_count"], len(rows), output / "energy_summary.json")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-config", type=Path, required=True)
    parser.add_argument("--student-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--loss-mode", choices=("signed_mse", "scale_aligned"),
                        default="signed_mse")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--mix-start-step", type=int, default=None,
                        help="Before this update use only teacher walkers; default: steps//4.")
    parser.add_argument("--student-fraction", type=float, default=0.5)
    parser.add_argument("--burn-in", type=int, default=200)
    parser.add_argument("--student-burn-in", type=int, default=100)
    parser.add_argument("--mcmc-steps", type=int, default=10)
    parser.add_argument("--proposal-std", type=float, default=0.25,
                        help="全电子高斯提议的初始步宽；使用 student.vmc.nmoves_per_width_update 调节。")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--max-backtracks", type=int, default=16,
                        help="Halve an Adam update until same-batch loss decreases.")
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-energy", action="store_true",
                        help="预训练结束后，从学生 |psi|² 分布采样并计算局部能量；默认关闭。")
    parser.add_argument("--energy-samples", type=int, default=16,
                        help="能量诊断的局部能量样本数。")
    parser.add_argument("--energy-burn-in", type=int, default=100,
                        help="能量诊断前在学生分布下的 Metropolis 热化步数。")
    args = parser.parse_args(argv)
    if args.steps <= 0 or args.batch_size < 2 or args.mcmc_steps <= 0:
        parser.error("steps、mcmc-steps 必须为正，batch-size 至少为 2。")
    if (args.burn_in < 0 or args.student_burn_in < 0
            or args.save_every <= 0 or args.max_backtracks < 0):
        parser.error("burn-in/student-burn-in/max-backtracks 不可为负，"
                     "save-every 必须为正。")
    if not 0.0 < args.student_fraction < 1.0:
        parser.error("student-fraction 必须在 0 与 1 之间。")
    if args.proposal_std <= 0 or args.learning_rate <= 0 or args.gradient_clip <= 0:
        parser.error("proposal-std、learning-rate、gradient-clip 必须为正。")
    if args.energy_samples <= 0 or args.energy_burn_in < 0:
        parser.error("energy-samples 必须为正，energy-burn-in 不可为负。")
    if args.mix_start_step is None:
        args.mix_start_step = max(1, args.steps // 4)
    if not 0 <= args.mix_start_step <= args.steps:
        parser.error("mix-start-step 必须在 0..steps 范围内。")
    return args


def main(argv=None):
    args = parse_args(argv)
    for path in (args.teacher_checkpoint, args.teacher_config, args.student_config):
        if not path.is_file():
            raise FileNotFoundError(path)

    teacher_config = io.load_config_dict("", str(args.teacher_config))
    student_config = io.load_config_dict("", str(args.student_config))
    # 先核对原始物理体系，不能让默认值或继承操作掩盖不一致。
    validate_problem_match(teacher_config, student_config)
    teacher_config, teacher_added = complete_config(
        teacher_config, source=str(args.teacher_config))
    # student JSON 直接使用，不自动补全；仅重建旧 teacher 时允许兼容补全。
    if teacher_config.dtype not in ("float32", "float64"):
        raise ValueError("仅支持 float32/float64 实值配置。")
    jax.config.update("jax_enable_x64", teacher_config.dtype == "float64")
    molecule = validate_configs(teacher_config, student_config)
    key = jax.random.PRNGKey(args.seed)
    key, teacher_key, student_key = jax.random.split(key, 3)
    expected_teacher_params, teacher_network = construct_network(
        teacher_config, molecule, teacher_key)
    _, teacher_data, loaded_teacher_params, _, _ = io.reload_vmc_state(
        str(args.teacher_checkpoint.parent), args.teacher_checkpoint.name)
    teacher_params = check_checkpoint_params(expected_teacher_params,
                                             loaded_teacher_params)
    if not check_checkpoint_amplitudes(teacher_network, teacher_params,
                                       teacher_data, molecule):
        logging.warning("teacher checkpoint 未保存 walker 幅值，"
                        "只能验证参数结构，无法数值核对旧波函数。")
    student_params, student_network = construct_network(
        student_config, molecule, student_key)

    key, teacher_positions, initial_source = initial_teacher_positions(
        teacher_data, teacher_config, molecule, args.batch_size, key)
    logging.info("teacher walker 起点: %s", initial_source)
    width_interval = student_config.vmc.nmoves_per_width_update
    teacher_move_metadata = None
    teacher_sampler = make_metropolis_sampler(
        teacher_network, molecule.ions, args.mcmc_steps, args.proposal_std, width_interval)
    if args.burn_in:
        burn_sampler = make_metropolis_sampler(
            teacher_network, molecule.ions, args.burn_in, args.proposal_std, width_interval)
        teacher_positions, key, _, teacher_move_metadata = burn_sampler(
            teacher_params, teacher_positions, key)

    student_count = max(1, min(args.batch_size - 1,
                               round(args.batch_size * args.student_fraction)))
    teacher_count = args.batch_size - student_count
    student_sampler = make_metropolis_sampler(
        student_network, molecule.ions, args.mcmc_steps, args.proposal_std, width_interval)
    student_positions = None
    student_move_metadata = None
    # The two random networks may differ in log amplitude by tens or more.
    # A trainable log scale avoids vanishing signed-MSE gradients.  It is
    # folded into the student's output parameters when saving the checkpoint.
    if args.loss_mode == "signed_mse":
        teacher_logs = jax.vmap(lambda x: teacher_network(
            teacher_params, x, molecule.ions)[1])(teacher_positions)
        student_logs = jax.vmap(lambda x: student_network(
            student_params, x, molecule.ions)[1])(teacher_positions)
        finite = np.isfinite(np.asarray(teacher_logs - student_logs))
        if not np.any(finite):
            raise FloatingPointError("初始 teacher/student 对数幅度均不可比较。")
        initial_scale = float(np.median(
            np.asarray(teacher_logs - student_logs)[finite]))
    else:
        initial_scale = 0.0
    trainable = {"network": student_params,
                 "log_scale": jnp.asarray(initial_scale, dtype=molecule.ions.dtype)}
    optimizer = optax.chain(optax.clip_by_global_norm(args.gradient_clip),
                            optax.adam(args.learning_rate))
    optimizer_state = optimizer.init(trainable)
    train_step, evaluate_loss, evaluate_teacher = make_training_step(
        teacher_network, student_network, molecule.ions,
        args.loss_mode, optimizer)

    output = _unique_output_directory(args.output_dir)
    _snapshot_inputs(args, output, teacher_config, student_config,
                     teacher_added)
    logging.info("结果目录: %s", output)
    with (output / "pretrain_metrics.csv").open("w", newline="") as metrics_file:
        writer = csv.writer(metrics_file)
        writer.writerow(("step", "loss_before", "loss_after", "update_scale",
                         "teacher_accept", "student_accept", "alignment",
                         "sign_agreement", "log_scale"))
        for step in range(args.steps):
            teacher_positions, key, teacher_accept, teacher_move_metadata = teacher_sampler(
                teacher_params, teacher_positions, key, teacher_move_metadata)
            student_accept = math.nan
            if step >= args.mix_start_step:
                if student_positions is None:
                    student_positions = teacher_positions[:student_count]
                    if args.student_burn_in:
                        student_burn = make_metropolis_sampler(
                            student_network, molecule.ions,
                            args.student_burn_in, args.proposal_std, width_interval)
                        student_positions, key, _, student_move_metadata = student_burn(
                            trainable["network"], student_positions, key)
                student_positions, key, student_accept, student_move_metadata = student_sampler(
                    trainable["network"], student_positions, key, student_move_metadata)
                batch = jnp.concatenate(
                    (teacher_positions[:teacher_count], student_positions), axis=0)
            else:
                batch = teacher_positions

            teacher_targets = evaluate_teacher(teacher_params, batch)
            proposed, proposed_state, loss, (alignment, agreement) = (
                train_step(trainable, optimizer_state, batch, teacher_targets))
            if not bool(jnp.isfinite(loss)):
                raise FloatingPointError(f"step {step + 1}: 更新前损失为 NaN/Inf。")
            candidate_loss, _ = evaluate_loss(proposed, batch, teacher_targets)
            update_scale = 1.0
            if not bool(jnp.isfinite(candidate_loss)) or float(candidate_loss) > float(loss):
                update_scale = 0.0
                for attempt in range(1, args.max_backtracks + 1):
                    trial_scale = 0.5 ** attempt
                    trial = jax.tree_util.tree_map(
                        lambda old, new: old + trial_scale * (new - old),
                        trainable, proposed)
                    trial_loss, _ = evaluate_loss(trial, batch, teacher_targets)
                    if bool(jnp.isfinite(trial_loss)) and float(trial_loss) <= float(loss):
                        proposed, candidate_loss = trial, trial_loss
                        update_scale = trial_scale
                        break
            if update_scale:
                trainable, optimizer_state = proposed, proposed_state
                after_loss = candidate_loss
            else:
                after_loss = loss
            if not all(bool(jnp.all(jnp.isfinite(leaf)))
                       for leaf in jax.tree_util.tree_leaves(trainable)):
                raise FloatingPointError(f"step {step + 1}: 学生参数出现 NaN/Inf。")
            writer.writerow((step + 1, float(loss), float(after_loss),
                             update_scale, float(teacher_accept),
                             float(student_accept), float(alignment),
                             float(agreement), float(trainable["log_scale"])))
            if (step + 1) % args.save_every == 0 or step + 1 == args.steps:
                metrics_file.flush()
                source = student_positions if student_positions is not None else teacher_positions
                saved_params = _fold_student_log_scale(
                    trainable["network"], trainable["log_scale"], student_config)
                _save_student_checkpoint(output, "pretrained_student.npz",
                                         student_config, molecule, student_network,
                                         saved_params, source, key)
                logging.info("step=%d loss=%.6g -> %.6g step_scale=%.3g"
                             " sign_match=%.3f teacher_accept=%.3f"
                             " student_accept=%.3f", step + 1, float(loss),
                             float(after_loss), update_scale,
                             float(agreement), float(teacher_accept),
                             float(student_accept))
    with (output / "calibration.json").open("w", encoding="utf-8") as file:
        json.dump({"final_student_log_scale": float(trainable["log_scale"]),
                   "student_type": student_config.gq.type,
                   "folded_into_so3_alpha": student_config.gq.type == "so3",
                   "folded_into": ("so3_det.alpha" if student_config.gq.type == "so3"
                                   else "RHF_orbital" if student_config.gq.RHF else "orbital")},
                  file, indent=2)
    if args.test_energy:
        # The student walkers have already been sampled under the final network
        # unless training ended before the student chain was introduced.
        source = student_positions if student_positions is not None else teacher_positions
        evaluate_student_energy(student_network, trainable["network"], molecule,
                                student_config, source, key, args, output)
    print(f"Pretraining complete: {output}")
    return output


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
