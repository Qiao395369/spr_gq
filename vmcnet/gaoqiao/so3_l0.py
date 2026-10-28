"""仅以零阶输出构造轨道的 SO(3) 网络。

每层按 Conv -> WBN 更新；最后的 Conv 保留所有阶以便 WBN 耦合，
最后的 WBN 只计算输出 l=0。没有整体 A 读出、rank 或 sector 行列式库。
K = gq.ndet 就是全部行列式数，波函数为 sum_k det(Phi_k)。

沿用 so3_utils 的几何、球谐、CG、残差卷积和 Kato Padé Jastrow。
Jastrow 以 exp(J / Ne) 乘每个轨道元素，所以 orbital 接口返回的矩阵
已经包含完整波函数需要的因子，不在 det 外再次加 J。
"""

from __future__ import annotations

import dataclasses
import math
from typing import Mapping, Sequence

import jax
import jax.numpy as jnp

from vmcnet.gaoqiao import so3_utils


@dataclasses.dataclass(frozen=True)
class SO3L0Config:
    """独立的 so3_l0 配置；精度由构建入口的顶层 dtype 传入。

    n_layers 是 Conv+WBN 的对数，至少为 1。lmax 只限制隐藏阶数，
    最后的 WBN 总是只输出 l=0。wbn_bottleneck=None 表示与 channels 相同。
    basis 控制主干：real、complex、complex_to_real 均可；轨道和最终
    波函数始终为实数。复零阶的实、虚部作为两组实输入交给轨道头。
    """

    lmax: int = 2
    channels: int = 16
    n_layers: int = 3
    n_rbf: int = 8
    cutoff: float | None = None
    wbn_bottleneck: int | None = None
    basis: str = "real"
    init_l1: bool = True
    regularize_angular_origin: bool = True
    eps: float = 1e-12
    use_envelope: bool = True
    envelope_sigma_init: float = 1.
    envelope_sigma_floor: float = 1e-6
    envelope_pi_init: float = 1.
    use_kato_jastrow: bool = True
    jastrow_b_init: float = 1.
    float_dtype: object = jnp.float32


def _common_config(config: SO3L0Config) -> so3_utils.SO3Config:
    """仅适配复用工具的静态配置，不在新网络中创建 rank/读出参数。"""
    if config.n_layers < 1:
        raise ValueError("so3_l0.n_layers 必须至少为 1（每层 Conv+WBN）。")
    if config.eps <= 0:
        raise ValueError("so3_l0.eps 必须为正。")
    if config.wbn_bottleneck is not None and config.wbn_bottleneck <= 0:
        raise ValueError("so3_l0.wbn_bottleneck 必须为正或 None。")
    fields = {field.name: getattr(config, field.name)
              for field in dataclasses.fields(config)}
    fields["basis"] = str(config.basis).lower()
    fields.update(target_ls=(0,), det_rank=1, wbn_layers=config.n_layers,
                  use_invariant_orbital_features=False)
    return so3_utils.SO3Config(**fields)


def init_backbone(key, type_dim: int, config: SO3L0Config):
    """抽取每层独立参数，去掉不用的全局读出及最后 WBN 的高阶输出。

    复用旧工具的高斯初始化与相位约定；临时 readout 不进入返回参数树。
    最后 WBN 仍有 l x l -> 0 路径，故高阶信息可以在最后一步进入标量。
    """
    common = _common_config(config)
    tree = so3_utils.init_so3_params(key, type_dim, common)
    layers = []
    for index, (conv, wbn) in enumerate(zip(tree["conv_layers"], tree["wbn"])):
        if index == config.n_layers - 1:
            wbn = {
                "self": {"0": wbn["self"]["0"]},
                "paths": {path: value for path, value in wbn["paths"].items()
                          if path.rsplit("_", 1)[1] == "0"},
            }
        layers.append({"conv": conv, "wbn": wbn})
    result = {"init_w0": tree["init_w0"], "layers": tuple(layers)}
    for name in ("init_l1_type", "init_l1_rbf"):
        if name in tree:
            result[name] = tree[name]
    return result


def _wbn_update(h, cg, params, common, *, final: bool):
    """最后一块仅建立 l=0 输出；左右输入仍可取任意隐藏阶数。"""
    if not final:
        return so3_utils._bilinear(h, cg, params, common)
    out = jnp.einsum("oc,icq->ioq", params["self"]["0"], h[0])
    for (left_l, right_l, output_l), tensor in cg.items():
        if output_l != 0:
            continue
        weights = params["paths"][f"{left_l}_{right_l}_0"]
        left = jnp.einsum("bc,icm->ibm", weights["left"], h[left_l])
        right = jnp.einsum("bc,icn->ibn", weights["right"], h[right_l])
        coupled = jnp.einsum("ibm,ibn,pmn->ibp", left, right, tensor,
                             optimize=True)
        out = out + jnp.einsum("ob,ibp->iop", weights["out"], coupled)
    return {0: (h[0] + out) / math.sqrt(2.)}


def apply_backbone(params, positions, type_features, config: SO3L0Config):
    """返回仅含 0 键的零阶特征字典，形状为 (Ne+Na,C_out,1)。

    real 的 C_out=C；两个复主干模式均把零阶实/虚部拼为 2C 个实通道。
    因为 l=0 的实复换基矩阵就是 1，纯复主干也可直接拆出两组实标量。
    """
    common = _common_config(config)
    dtype = so3_utils.config_dtype(common)
    features = jnp.asarray(type_features, so3_utils.feature_dtype(common))
    positions = jnp.asarray(positions, dtype)
    if positions.ndim != 2 or positions.shape[-1] != 3:
        raise ValueError("so3_l0 位置必须为 (N,3)。")
    if features.ndim != 2 or features.shape[0] != positions.shape[0]:
        raise ValueError("so3_l0 类型描述符必须与粒子逐行对应。")
    g, harmonics = so3_utils._edge_data(positions, common)
    basis = "real" if common.basis == "real" else "complex"
    cg = so3_utils._cg_table(common.lmax, basis,
                             so3_utils.feature_dtype(common))
    h = so3_utils._initial_features(params, features, g, harmonics, common)
    for index, layer in enumerate(params["layers"]):
        h = so3_utils._convolution(h, g, harmonics, cg, layer["conv"], common)
        h = _wbn_update(h, cg, layer["wbn"], common,
                        final=index == len(params["layers"]) - 1)
    if common.basis != "real":
        real, imag = so3_utils.split_real_multiplet(h[0], 0)
        h = {0: jnp.concatenate((real, imag), axis=1)}
    return h


def init_params(key, type_dim: int, nspins: Sequence[int], natom: int,
                ndet: int, config: SO3L0Config):
    """构造主干、每个非空自旋组的实线性头、核包络和 Jastrow。

    完整矩阵有 Ne 个轨道列；每个头的 w 为 (F,ndet*Ne)，无偏置。
    行列式直接求和，不另外建立组合权重或 A 参数。
    """
    common = _common_config(config)
    dtype = so3_utils.config_dtype(common)
    nelec = sum(nspins)
    if len(nspins) != 2 or min(nspins) < 0 or nelec <= 0 or ndet <= 0:
        raise ValueError("so3_l0 要求两组非负自旋数、正电子数和正 ndet。")
    if config.use_envelope and natom <= 0:
        raise ValueError("启用 so3_l0 核包络时至少需要一个核。")
    kb, kj, key = jax.random.split(key, 3)
    width = config.channels * (1 if common.basis == "real" else 2)
    heads, envelopes = [], []
    for count in nspins:
        if count == 0:
            continue
        key, kh, ke = jax.random.split(key, 3)
        heads.append({"w": so3_utils._randn(
            kh, (width, ndet * nelec), 1 / math.sqrt(width), dtype)})
        envelopes.append(
            so3_utils._init_envelope(
                ke, ndet, nelec, natom, config.envelope_sigma_init,
                config.envelope_sigma_floor, config.envelope_pi_init, dtype)
            if config.use_envelope else None)
    return {
        "backbone": init_backbone(kb, type_dim, config),
        "orbital": tuple(heads),
        "envelope": tuple(envelopes),
        "jastrow": so3_utils.init_kato_jastrow_params(
            kj, nspins, config.jastrow_b_init, dtype)
        if config.use_kato_jastrow else None,
    }


def orbital_matrices(params: Mapping, electron_pos, atoms, type_features,
                     nspins: Sequence[int], ndet: int, config: SO3L0Config):
    """返回 (ndet,Ne,Ne) 实矩阵，已包含包络与 exp(J/Ne)。

    全体电子共用同一个标量乘子，交换同自旋电子时仍只交换矩阵行。
    det(exp(J/Ne)*Phi)=exp(J)*det(Phi)，所以 signed_network 不再加 J。
    """
    dtype = so3_utils.config_dtype(_common_config(config))
    nelec = sum(nspins)
    electron_pos = jnp.asarray(electron_pos, dtype).reshape((nelec, 3))
    atoms = jnp.asarray(atoms, dtype).reshape((-1, 3))
    positions = jnp.concatenate((electron_pos, atoms), axis=0)
    scalar = apply_backbone(params["backbone"], positions, type_features,
                             config)[0][:nelec, :, 0]
    active = tuple(count for count in nspins if count)
    cuts = tuple(sum(active[:i]) for i in range(1, len(active)))
    features = jnp.split(scalar, cuts, axis=0)
    electrons = jnp.split(electron_pos, cuts, axis=0)
    blocks = []
    for f, r, head, env in zip(
            features, electrons, params["orbital"], params["envelope"]):
        values = (f @ head["w"]).reshape((f.shape[0], ndet, nelec))
        if env is not None:
            values = values * so3_utils.apply_orbital_envelope(
                r, atoms, env, config.envelope_sigma_floor, config.eps)
        blocks.append(jnp.transpose(values, (1, 0, 2)))
    matrices = jnp.concatenate(blocks, axis=1)
    if config.use_kato_jastrow and params["jastrow"] is not None:
        log_j = so3_utils.kato_ee_jastrow(
            electron_pos, nspins, params["jastrow"], config.eps)
        matrices = matrices * jnp.exp(log_j / nelec)
    return matrices


def make_network(key, charges, nspins, ndet: int, type_features,
                 config: SO3L0Config):
    """返回与 gaoqiao build_network 一致的五元组，供 VMC 和轨道预训练使用。"""
    params = init_params(key, type_features.shape[-1], nspins, len(charges),
                         ndet, config)

    def orbitals(parameters, positions, atoms):
        # 和普通 full determinant 网络相同：长度为 1 的列表。
        return [orbital_matrices(parameters, positions, atoms, type_features,
                                  nspins, ndet, config)]

    def signed_network(parameters, positions, atoms):
        signs, logs = jnp.linalg.slogdet(orbitals(parameters, positions, atoms)[0])
        return so3_utils.phase_logsumexp(signs, logs, axis=0)

    def det_distribution(parameters, positions, atoms):
        _, logs = jnp.linalg.slogdet(orbitals(parameters, positions, atoms)[0])
        return jax.nn.log_softmax(logs)

    def nuclear_features(parameters, positions, atoms):
        x = so3_utils.combine_particle_positions(
            positions, atoms, dtype=so3_utils.config_dtype(_common_config(config)))
        return apply_backbone(parameters["backbone"], x, type_features,
                                config)[0][sum(nspins):, :, 0]

    return params, signed_network, det_distribution, orbitals, nuclear_features
