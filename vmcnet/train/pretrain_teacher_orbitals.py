"""在固定 teacher 构型上，独立预训练 student 的完整轨道矩阵。

本入口不调用 runners，不做 HF 计算或 VMC 能量优化；默认在预训练
结束后，用最终 student 自己的分布重新热化、采样，测试局部能量。
student 从 get_default_config() 开始，由 --config.xxx=value 覆盖；只有
重建历史 teacher 时才补全旧配置，并由 config_compat 逐项记录补全日志。

teacher 为实值 ef_shrd_sym；student 可为 ef_shrd_sym 或 so3_l0。
两者返回含原生包络和 Jastrow 的完整 (K,N,N) 轨道，按序号一一监督。
旧 so3 含外部 A 与 rank 库，仍由完整波函数预训练入口处理。
"""

import argparse
import csv
import json
import logging
import shutil
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl import flags
from ml_collections.config_flags import config_flags

from vmcnet.gaoqiao.build import build_network
from vmcnet.train.config_compat import complete_config
from vmcnet.train.default_config import (
    choose_model_type_in_model_config, get_default_config,
)
# 只复用配置校验、主 VMC 的采样器及 checkpoint 工具，不调用原蒸馏训练流程。
from vmcnet.train.pretrain_from_trained_net import (
    _save_student_checkpoint, _unique_output_directory,
    check_checkpoint_amplitudes, check_checkpoint_params,
    collect_fixed_teacher_samples, evaluate_student_energy, initial_teacher_positions,
    make_metropolis_sampler, validate_configs, validate_problem_match,
)
from vmcnet.utils import io


def construct_orbital_network(config, molecule, key):
    """按现有 gaoqiao 构造参数，同时取得 signed_network 和轨道接口。

    返回轨道为 (K,N,N)，含该网络正常前向使用的 envelope 和 Jastrow
    乘子。两边都使用这一约定，不把 teacher 的 Jastrow 重复乘一次。
    """
    gq = config.gq
    attention = None
    if gq.do_attn:
        attention = dict(qkdim=gq.attn_nchnl, nhead=gq.attn_nhead,
                         do_gate=gq.attn_do_gate, do_lnorm=gq.attn_do_lnorm)
    features = dict(do_act=gq.feat_do_act, act_func=gq.feat_act_func,
                    numb_divid=gq.feat_numb_divid,
                    scale=gq.scale if gq.scale != 1.0 else [],
                    rescale=gq.rescale_input)
    params, signed_network, _, orbitals, _ = build_network(
        n=sum(molecule.nspins), charges=molecule.charges,
        nspins=molecule.nspins, key=key, ndet=gq.ndet,
        depth=gq.wfn_depth, h1=gq.h1, h2=gq.h2, nh=gq.nh,
        do_complex=False, gq_type=gq.type, envelope_type=gq.envelope_type,
        attn=attention, feat_params=features, det_mode=gq.det_mode,
        jastrow_type=gq.jastrow_type,
        jastrow_mlp_nlayer=gq.jastrow_mlp_nlayer,
        jastrow_mlp_ndim=gq.jastrow_mlp_ndim, RHF=gq.RHF,
        activation_type=gq.activation_type,
        jastrow_activation_type=gq.jastrow_activation_type,
        dp_type=gq.dp_type, alpha=gq.alpha, dtype=config.dtype,
        so3_l0_params=gq.so3_l0.to_dict() if gq.type == "so3_l0" else None,
    )

    def full_orbitals(parameters, positions, atoms):
        # 普通网络与 so3_l0 均返回长度为 1 的完整行列式 list。
        matrices = orbitals(parameters, positions, atoms)
        if not isinstance(matrices, (tuple, list)) or len(matrices) != 1:
            raise ValueError("要求 full determinant 轨道接口返回一组 (K,N,N) 矩阵。")
        return matrices[0]

    return params, signed_network, full_orbitals


def validate_orbital_types(teacher, student):
    """在加载 checkpoint 前检查轨道监督约定，不补全任何配置。"""
    if teacher.gq.type != "ef_shrd_sym":
        raise NotImplementedError("轨道预训练 teacher 当前要求 ef_shrd_sym。")
    if student.gq.type not in ("ef_shrd_sym", "so3_l0"):
        raise NotImplementedError(
            "轨道预训练 student 支持 ef_shrd_sym 或 so3_l0；旧 so3 含外部 A "
            "和 rank 库，请使用完整波函数预训练入口或改用 so3_l0。")
    for role, config in (("teacher", teacher), ("student", student)):
        if config.gq.det_mode != "det":
            raise ValueError(f"{role} 要求 det_mode=det。")


def orbital_loss(target, prediction, mode):
    """完整矩阵损失；输入均为 (B,K,N,N)，按相同 k 一一对应。

    l2：与 pretrain_demo.py 一样，直接平均矩阵元素的误差平方。
    fulldet：沿用该文件的 QR occupied-subspace 定义，但分别比较每个
    teacher/student 行列式，不再平均行列式轴。对每个 k，先把 B 个
    构型的电子行堆叠成 (B*N,N)，再 QR；否则对单个满秩方阵求列空间
    会总得到整个 R^N，失去监督作用。

    fulldet 约束列空间而非逐元素数值；列的可逆混合、尺度等自由度
    未被完全固定，所以它的损失为零不代表最终波函数已经相同。
    """
    if target.shape != prediction.shape or target.ndim != 4:
        raise ValueError("teacher/student 轨道必须具有相同的 (B,K,N,N) 形状。")
    if mode == "l2":
        return jnp.mean(jnp.square(prediction - target))
    if mode != "fulldet":
        raise ValueError("loss-mode 只支持 fulldet 或 l2。")
    batch, determinants, nelec, norb = target.shape
    if batch < 2 or nelec != norb:
        raise ValueError("fulldet 要求每批至少两个构型，并使用完整方阵。")
    reference = target.transpose(1, 0, 2, 3).reshape(
        determinants, batch * nelec, norb)
    predicted = prediction.transpose(1, 0, 2, 3).reshape(
        determinants, batch * nelec, norb)
    # 与 pretrain_demo.py 相同：reduced QR，没有另外引入 eps 正则或裁剪。
    q_ref, _ = jnp.linalg.qr(reference, mode="reduced")
    q_net, _ = jnp.linalg.qr(predicted, mode="reduced")
    overlap = jnp.einsum("kpi,kpj->kij", q_ref, q_net)
    return jnp.mean(norb - jnp.sum(jnp.square(overlap), axis=(-2, -1)))


def fixed_batches(positions, targets, batch_size, key, mode):
    """每轮打乱并遍历全部固定点；不分验证集，不改变坐标和标签。"""
    count = len(positions)
    while True:
        key, shuffle_key = jax.random.split(key)
        order = jax.random.permutation(shuffle_key, count)
        start = 0
        while start < count:
            end = min(start + batch_size, count)
            # QR 子空间损失不能使用单构型尾批；把最后一点并入上一批。
            if mode == "fulldet" and count - end == 1:
                end = count
            indices = order[start:end]
            yield positions[indices], targets[indices]
            start = end


def require_finite(tree, label):
    """损失、轨道、梯度或参数含 NaN/Inf 时停止，不静默过滤数据。"""
    leaves = jax.tree_util.tree_leaves(tree)
    if not all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves):
        raise FloatingPointError(f"{label} 含 NaN/Inf，停止轨道预训练。")


def make_train_step(student_orbitals, atoms, mode, learning_rate):
    """沿用 HF 预训练的标准 Adam；不做损失回溯或学习整体 log_scale。"""
    optimizer = optax.adam(learning_rate)
    evaluate = jax.vmap(lambda p, x: student_orbitals(p, x, atoms),
                        in_axes=(None, 0))

    def objective(params, positions, target):
        prediction = evaluate(params, positions)
        return orbital_loss(target, prediction, mode), prediction

    @jax.jit
    def step(params, state, positions, target):
        (loss, prediction), gradient = jax.value_and_grad(
            objective, has_aux=True)(params, positions, target)
        updates, state = optimizer.update(gradient, state, params)
        params = optax.apply_updates(params, updates)
        diagnostics = dict(gradient_norm=optax.global_norm(gradient),
                           orbital_l2=jnp.mean(jnp.square(prediction - target)),
                           gradient_finite=jnp.all(jnp.stack([
                               jnp.all(jnp.isfinite(x))
                               for x in jax.tree_util.tree_leaves(gradient)])))
        return params, state, loss, diagnostics

    return optimizer, step, jax.jit(objective)


def parse_args(argv=None):
    """独立训练参数交给 argparse；--config.* 交给项目原有 flags 解析器。"""
    parser = argparse.ArgumentParser(
        description=__doc__, allow_abbrev=False,
        epilog="网络和分子参数使用 --config.xxx=value；未指定项来自默认配置。")
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--loss-mode", choices=("fulldet", "l2"), default="l2")
    parser.add_argument("--sample-count", type=int, default=4096,
                        help="预先采集并冻结的构型总数，全部用于训练")
    parser.add_argument("--sampler-chains", type=int, default=128,
                        help="采集阶段的并行 teacher 链数")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=None,
                        help="Adam 更新次数；默认使用 config.pretrain.iterations")
    parser.add_argument("--burn-in", type=int, default=None,
                        help="teacher 热化步数；默认使用 config.pretrain.nburn")
    parser.add_argument("--mcmc-steps", type=int, default=None,
                        help="采集两批点的间隔；默认使用 pretrain.nsteps_per_param_update")
    parser.add_argument("--proposal-std", type=float, default=None,
                        help="初始高斯步宽；默认使用 config.vmc.std_move")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="与 pretrain_demo.py 的默认 Adam 学习率相同")
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-energy", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="默认在训练后测试 student 能量；--no-test-energy 可关闭")
    parser.add_argument("--energy-samples", type=int, default=2048,
                        help="能量测试的总构型数，重新从 student 分布采集")
    parser.add_argument("--energy-burn-in", type=int, default=None,
                        help="student 能量测试的热化步数；默认使用 config.eval.nburn")
    args, overrides = parser.parse_known_args(argv)
    if any(not item.startswith("--config.") or "=" not in item for item in overrides):
        parser.error("未知参数；网络覆盖项请写成 --config.xxx=value。")
    config_argv = [sys.argv[0], *overrides]
    values = flags.FlagValues()
    config_flags.DEFINE_config_dict(
        "config", get_default_config(), flag_values=values,
        lock_config=False, sys_argv=config_argv)
    values(config_argv)
    student = values.config
    student.model = choose_model_type_in_model_config(student.model)
    # 从默认配置构建新 student，不调用任何旧配置补全函数。
    for name, value in (("steps", student.pretrain.iterations),
                        ("burn_in", student.pretrain.nburn),
                        ("energy_burn_in", student.eval.nburn),
                        ("mcmc_steps", student.pretrain.nsteps_per_param_update),
                        ("proposal_std", student.vmc.std_move)):
        if getattr(args, name) is None:
            setattr(args, name, value)
    for name in ("sample_count", "sampler_chains", "batch_size", "steps",
                 "mcmc_steps", "save_every"):
        if getattr(args, name) <= 0:
            parser.error(f"{name} 必须为正整数。")
    if args.burn_in < 0 or not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("burn-in 不可为负，learning-rate 必须为有限正数。")
    if not np.isfinite(args.proposal_std) or args.proposal_std <= 0:
        parser.error("proposal-std 必须为有限正数。")
    if args.energy_samples <= 0 or args.energy_burn_in < 0:
        parser.error("energy-samples 必须为正，energy-burn-in 不可为负。")
    if args.loss_mode == "fulldet" and min(args.sample_count, args.batch_size) < 2:
        parser.error("fulldet 的训练集及 batch-size 都必须至少为 2。")
    if args.steps * args.batch_size < args.sample_count:
        parser.error("当前 steps 不足以遍历全部固定点，请增加 steps。")
    return args, student


def snapshot_inputs(args, output, teacher, student, added):
    """保存配对的 teacher 文件、解析后的 student 配置和本次训练设置。"""
    teacher_dir = output / "teacher_source"
    teacher_dir.mkdir()
    shutil.copy2(args.teacher_checkpoint, teacher_dir / "teacher.npz")
    shutil.copy2(args.teacher_config, teacher_dir / "config_original.json")
    io.save_config_dict_to_json(teacher, str(teacher_dir), "config_resolved")
    io.save_config_dict_to_json(student, str(output), "config")
    settings = {name: str(value) if isinstance(value, Path) else value
                for name, value in vars(args).items()}
    settings.update(
        student_type=student.gq.type, teacher_fields_filled=list(added),
        orbital_pairing="same determinant index; no determinant averaging",
        orbital_definition="full matrix including native envelope and Jastrow multiplier",
        sampling_distribution="teacher |sum_k w_k det(Phi_k)|^2",
        sampling_proposal="main VMC all-electron Gaussian with dynamic width",
        training_data="all fixed points; no validation split; no sampling during training",
        optimizer="standard Adam; no backtracking or gradient clipping",
        vmc_optimization=False, energy_evaluation=args.test_energy,
        energy_sampling_distribution="final student |psi|^2; fresh burn-in and sampling",
    )
    (output / "pretrain_settings.json").write_text(
        json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv=None):
    args, student = parse_args(argv)
    for path in (args.teacher_checkpoint, args.teacher_config):
        if not path.is_file():
            raise FileNotFoundError(path)
    teacher = io.load_config_dict("", str(args.teacher_config))
    validate_problem_match(teacher, student)
    validate_orbital_types(teacher, student)
    if teacher.gq.ndet != student.gq.ndet:
        raise ValueError("按行列式序号一一监督，teacher/student 的 ndet 必须相同。")
    # 仅在这里重建历史 teacher 前补全；补全明细由 complete_config 输出日志。
    teacher, added = complete_config(teacher, source=str(args.teacher_config))
    jax.config.update("jax_enable_x64", student.dtype == "float64")
    molecule = validate_configs(teacher, student)
    key, teacher_key, student_key = jax.random.split(jax.random.PRNGKey(args.seed), 3)
    initial_teacher, teacher_network, teacher_orbitals = construct_orbital_network(
        teacher, molecule, teacher_key)
    _, teacher_data, loaded, _, _ = io.reload_vmc_state(
        str(args.teacher_checkpoint.parent), args.teacher_checkpoint.name)
    teacher_params = check_checkpoint_params(initial_teacher, loaded)
    if not check_checkpoint_amplitudes(teacher_network, teacher_params, teacher_data, molecule):
        logging.warning("checkpoint 没有保存幅值，无法数值核对重建 teacher。")
    params, student_network, student_orbitals = construct_orbital_network(
        student, molecule, student_key)
    # 行列式组合权重不进入轨道损失，因此不把其初始化差异藏在监督之外。
    # 当前 ef_shrd_sym 的这项通常为 None，即各行列式直接求和。
    if teacher_params.get("det") is not None or params.get("det") is not None:
        raise NotImplementedError("当前轨道实验要求 det=None；独立组合权重尚未适配。")

    chain_count = min(args.sampler_chains, args.sample_count)
    key, positions, source = initial_teacher_positions(
        teacher_data, teacher, molecule, chain_count, key)
    logging.info("teacher 起点=%s，链数=%d；只在收集数据时采样。", source, chain_count)
    metadata = None
    # signed_network 本身就是 teacher 轨道经行列式组合后的波函数。
    # 因此采的是完整 teacher 的 |Psi|^2，而非各轨道平方或单个行列式。
    if args.burn_in:
        burn = make_metropolis_sampler(
            teacher_network, molecule.ions, args.burn_in, args.proposal_std,
            student.pretrain.nmoves_per_width_update)
        positions, key, acceptance, metadata = burn(teacher_params, positions, key)
        logging.info("teacher 热化完成，平均接受率=%.4f。", float(acceptance))
    sampler = make_metropolis_sampler(
        teacher_network, molecule.ions, args.mcmc_steps, args.proposal_std,
        student.pretrain.nmoves_per_width_update)
    fixed_positions, positions, key, metadata = collect_fixed_teacher_samples(
        sampler, teacher_params, positions, key, metadata, args.sample_count)
    require_finite(fixed_positions, "固定构型")
    evaluate_teacher = jax.jit(jax.vmap(
        lambda x: teacher_orbitals(teacher_params, x, molecule.ions)))
    # teacher 只在采集后计算一次标签；训练中不再调用 teacher 或更新坐标。
    targets = jnp.concatenate([
        evaluate_teacher(fixed_positions[start:start + args.batch_size])
        for start in range(0, args.sample_count, args.batch_size)])
    expected = (args.sample_count, int(teacher.gq.ndet),
                sum(molecule.nspins), sum(molecule.nspins))
    if targets.shape != expected:
        raise ValueError(f"teacher 轨道形状 {targets.shape}，预期 {expected}。")
    require_finite(targets, "teacher 轨道标签")

    optimizer, train_step, objective = make_train_step(
        student_orbitals, molecule.ions, args.loss_mode, args.learning_rate)
    state = optimizer.init(params)
    batches = fixed_batches(fixed_positions, targets, args.batch_size, key, args.loss_mode)
    output = _unique_output_directory(args.output_dir)
    snapshot_inputs(args, output, teacher, student, added)
    np.savez_compressed(output / "fixed_teacher_orbitals.npz",
                        positions=np.asarray(fixed_positions), teacher_orbitals=np.asarray(targets))
    logging.info("结果目录: %s", output)
    logging.info("固定 %d 个构型，轨道标签 %s；loss=%s，全部用于训练。",
                 args.sample_count, targets.shape, args.loss_mode)
    with (output / "pretrain_metrics.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("step", "loss_before", "loss_after", "orbital_l2_before",
                         "orbital_l2_after", "gradient_norm"))
        for step in range(1, args.steps + 1):
            batch, target = next(batches)
            params, state, loss, diagnostics = train_step(params, state, batch, target)
            if not bool(diagnostics["gradient_finite"]):
                raise FloatingPointError(f"step={step}: 梯度含 NaN/Inf。")
            require_finite((params, state, loss, diagnostics), f"step={step} Adam 更新")
            after, prediction = objective(params, batch, target)
            require_finite((after, prediction), f"step={step} 更新后轨道/损失")
            after_l2 = jnp.mean(jnp.square(prediction - target))
            writer.writerow((step, float(loss), float(after),
                             float(diagnostics["orbital_l2"]), float(after_l2),
                             float(diagnostics["gradient_norm"])))
            if step % args.save_every == 0 or step == args.steps:
                file.flush()
                # 保存兼容项目 IO 的参数 checkpoint；不把 Adam 状态当成 VMC 状态。
                _save_student_checkpoint(
                    output, "pretrained_student.npz", student, molecule,
                    student_network, params, fixed_positions[:chain_count], key)
                logging.info("step=%d %s=%.6g -> %.6g orbital_l2=%.6g grad_norm=%.4g",
                             step, args.loss_mode, float(loss), float(after),
                             float(after_l2), float(diagnostics["gradient_norm"]))
    if args.test_energy:
        # 固定 teacher 数据只作为链的起点；必须先改用最终 student 热化，
        # 再从其 |Psi_S|^2 采新点，不能直接平均 teacher 固定点上的局部能量。
        # 参数已在上面保存；测试能量时不做任何参数更新。
        logging.info("轨道预训练完成，开始 student 能量测试：%d 个构型，热化 %d 步。",
                     args.energy_samples, args.energy_burn_in)
        energy_key = jax.random.fold_in(key, 0x454E4552)
        evaluate_student_energy(
            student_network, params, molecule, student,
            fixed_positions[:chain_count], energy_key, args, output)
    print(f"Orbital pretraining complete: {output}")
    return output


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
