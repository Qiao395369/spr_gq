"""完全独立的三种 SO(3) 表示模式与 Slater 波函数工具。

约定
----
* ``basis='real'``：使用实球谐与实 CG；q=-l,...,l。
* ``basis='complex'``：使用 Condon--Shortley 复球谐与标准 CG；m=-l,...,l。
* ``basis='complex_to_real'``：复数主干结束后换到实基，把实部、虚部
  固定拼成 2C 个实 channel；读出层及其后的波函数计算使用实数。
* ``complex`` 模式的复基特征保持复数直到波函数输出；输出为
  ``(phase, logabs)``，其中 ``phase`` 是单位模复数。另两种模式
  的 phase 是实数 sign。
* 坐标、核包络、Jastrow 与 Slater 行列式仍用实数。复数标量特征的
  实部和虚部，以及高阶特征的 Hermitian 范数，组成行列式轨道的实数输入。

配置、球谐、CG、等变网络、标量行列式与 Jastrow 都在本文件实现。
``networks.py`` 尚未改为导入本文件；可先使用下面的独立入口。

最短调用示例（type_features 的行顺序为电子在前、原子核在后）::

    cfg = SO3Config(basis="complex", lmax=2)
    params = init_wavefunction_params(key, type_dim, nspins, natom, K, cfg)
    out = evaluate_wavefunction(params, electrons, atoms, type_features,
                                nspins, K, cfg)
    psi = out["phase"] * jnp.exp(out["logabs"])

改成 ``basis="real"`` 即沿用原实基；``basis="complex_to_real"`` 则
保留复数主干，在主干末端换基并分离两组实特征。不能直接取复基
分量的实部。固定 q 投影后的标量值一般不具备完整 l>0 等变性。
"""

from __future__ import annotations

import dataclasses
import functools
import math
from typing import Dict, Mapping, Sequence, Tuple

import jax
import jax.numpy as jnp


Array = jax.Array
IrrepFeatures = Dict[int, Array]


@dataclasses.dataclass(frozen=True)
class SO3Config:
    """网络结构、精度、行列式和输出方式的静态配置。

    ``float_dtype`` 指坐标及标量头的实数精度：float32 对应 complex64，
    float64 对应 complex128。使用 float64 时须在 JAX 初始化前开启 x64。
    ``basis="complex_to_real"`` 在主干末端把 C 个复 channel 换成
    2C 个实 channel，随后使用实数读出层和实数行列式。
    ``target_ls=(0,)`` 只为 l=0 构造行列式和最终 multiplet；若
    ``lmax>0``，主干仍会计算高阶隐藏特征，轨道头可读取其不变量。
    """

    lmax: int = 2
    channels: int = 16
    n_layers: int = 3
    n_rbf: int = 8
    cutoff: float | None = None
    wbn_layers: int = 1
    wbn_bottleneck: int | None = None
    target_ls: Tuple[int, ...] = (0, 1, 2)
    init_l1: bool = True
    eps: float = 1.0e-12
    float_dtype: object = jnp.float32
    use_envelope: bool = True
    envelope_sigma_init: float = 1.0
    envelope_pi_init: float = 1.0
    use_kato_jastrow: bool = True
    jastrow_b_init: float = 1.0
    det_rank: int = 4
    det_bank_mode: str = "shared"
    use_invariant_orbital_features: bool = True
    output_projection: str = "q0"
    envelope_sigma_floor: float = 1.0e-6
    basis: str = "real"


def _basis(config: SO3Config) -> str:
    """检查模式名；模式决定主干的基和最终读出的数域。"""
    name = str(config.basis).lower()
    if name not in ("real", "complex", "complex_to_real"):
        raise ValueError("SO3Config.basis 只支持 'real'、'complex' 或 'complex_to_real'。")
    return name


def _backbone_basis(config: SO3Config) -> str:
    """末端转换模式的主干仍完全使用标准复基。"""
    return "real" if _basis(config) == "real" else "complex"


def config_dtype(config: SO3Config):
    """返回坐标、距离、行列式等实数部分使用的 dtype。"""
    dtype = jnp.dtype(config.float_dtype)
    if dtype == jnp.dtype(jnp.float32):
        return jnp.float32
    if dtype == jnp.dtype(jnp.float64):
        if not jax.config.jax_enable_x64:
            raise ValueError("使用 float64 前须启用 jax_enable_x64。")
        return jnp.float64
    raise ValueError("float_dtype 只支持 float32 或 float64。")


def normalize_det_bank_mode(mode: str) -> str:
    """行列式库可按 l 分开，或由所有 l 共享。"""
    mode = str(mode).lower()
    if mode not in ("per_l", "shared"):
        raise ValueError("det_bank_mode 只支持 'per_l' 或 'shared'。")
    return mode


def unique_determinant_count(determinants_per_rank: int, det_rank: int,
                             sectors: Sequence[int], det_bank_mode: str) -> int:
    """返回实际计算的唯一行列式数；shared 不随 sector 数增加。"""
    count = int(determinants_per_rank) * int(det_rank)
    return count if normalize_det_bank_mode(det_bank_mode) == "shared" else count * len(tuple(sectors))


def sector_list(config: SO3Config) -> Tuple[int, ...]:
    """使用 target_ls 指定的阶数；要求显式包含 0，并保序去重。

    ``target_ls=(0,)`` 只构造标量 sector，是合法配置。
    """
    sectors = []
    for l in config.target_ls:
        l = int(l)
        if not 0 <= l <= config.lmax:
            raise ValueError(f"target l={l} 超出 0..{config.lmax}。")
        if l not in sectors:
            sectors.append(l)
    if 0 not in sectors:
        raise ValueError("target_ls 必须显式包含 l=0。")
    return tuple(sectors)


def admissible_paths(lmax: int) -> Tuple[Tuple[int, int, int], ...]:
    """枚举 |l1-l2|<=L<=l1+l2 且 L<=lmax 的 SO(3) 耦合路径。"""
    return tuple((l1, l2, L)
                 for l1 in range(lmax + 1)
                 for l2 in range(lmax + 1)
                 for L in range(abs(l1 - l2), min(l1 + l2, lmax) + 1))


bilinear_paths = admissible_paths


def combine_particle_positions(pos: Array, atoms: Array, ndim: int = 3, dtype=None) -> Array:
    """按电子在前、原子核在后的顺序拼成 (Ne+Na,ndim)。"""
    dtype = dtype or jnp.result_type(pos, atoms)
    return jnp.concatenate((jnp.asarray(pos, dtype).reshape((-1, ndim)),
                            jnp.asarray(atoms, dtype).reshape((-1, ndim))), axis=0)


def pair_geometry(positions: Array, eps: float = 1e-12):
    """返回 rel、dist、unit、nonself；自边用安全方向避免 norm(0) 导数。"""
    n = positions.shape[0]
    rel = positions[None, :, :] - positions[:, None, :]
    nonself = ~jnp.eye(n, dtype=bool)
    safe_rel = jnp.where(nonself[..., None], rel,
                         jnp.asarray((0., 0., 1.), dtype=positions.dtype))
    safe_dist = jnp.linalg.norm(safe_rel, axis=-1)
    unit = safe_rel / jnp.maximum(safe_dist[..., None],
                                 jnp.asarray(eps, dtype=positions.dtype))
    dist = jnp.where(nonself, safe_dist, 0.)
    return rel, dist, unit, nonself


def cutoff_envelope(r: Array, cutoff: float | None) -> Array:
    """余弦径向截断；None 表示全连接。"""
    if cutoff is None:
        return jnp.ones_like(r)
    if cutoff <= 0:
        raise ValueError("cutoff 必须为正。")
    c = jnp.asarray(cutoff, dtype=r.dtype)
    return jnp.where(r < c, .5 * (1 + jnp.cos(jnp.pi * r / c)), 0.)


def radial_basis(r: Array, n_rbf: int, cutoff: float | None) -> Array:
    """多个长度尺度的 exp(-beta*r)，返回 (...,n_rbf)。
    r:(np,np)-->return:(np,np,n_rbf)
    """
    if n_rbf <= 0:
        raise ValueError("n_rbf 必须为正。")
    betas = jnp.exp(jnp.linspace(jnp.log(jnp.asarray(.35, r.dtype)),
                                   jnp.log(jnp.asarray(4., r.dtype)), n_rbf,
                                   dtype=r.dtype))
    return jnp.exp(-r[..., None] * betas) * cutoff_envelope(r, cutoff)[..., None]


@functools.lru_cache(maxsize=None)
def clebsch_gordan_value(l1: int, m1: int, l2: int, m2: int, L: int, M: int) -> float:
    """标准 |l,m> 基中的 CG 系数，由 Racah 阶乘有限和得到。

    这里处理 SO(3) 的非负整数 l。选择规则不满足时返回 0；复球谐基
    下这些 CG 数值本身是实数，复数的是球谐值和表示矩阵。
    """
    if (min(l1, l2, L) < 0 or abs(m1) > l1 or abs(m2) > l2
            or abs(M) > L or M != m1 + m2
            or not abs(l1 - l2) <= L <= l1 + l2):
        return 0.
    fact = math.factorial
    triangle = ((2 * L + 1) * fact(L + l1 - l2) * fact(L - l1 + l2)
                * fact(l1 + l2 - L) / fact(l1 + l2 + L + 1))
    magnetic = (fact(L + M) * fact(L - M) * fact(l1 - m1) * fact(l1 + m1)
                * fact(l2 - m2) * fact(l2 + m2))
    total = 0.
    for k in range(l1 + l2 + L + 2):
        args = (k, l1 + l2 - L - k, l1 - m1 - k, l2 + m2 - k,
                L - l2 + m1 + k, L - l1 - m2 + k)
        if min(args) >= 0:
            denominator = math.prod(fact(a) for a in args)
            total += (-1) ** k / denominator
    return math.sqrt(triangle * magnetic) * total


def _real_basis_entry(l: int, q: int, m: int) -> complex:
    """实球谐 q 与标准复球谐 m 之间的单个酉换基元素。"""
    if q == 0:
        return 1. + 0j if m == 0 else 0j
    k = abs(q)
    root = 1 / math.sqrt(2)
    if q > 0:
        return ((-1) ** k if m == k else 1 if m == -k else 0) * root
    if m == k:
        return -1j * (-1) ** k * root
    if m == -k:
        return 1j * root
    return 0j


@functools.lru_cache(maxsize=None)
def _real_cg_data(l1: int, l2: int, L: int):
    """在三个角动量指标上换基；纯虚通道选择整体相位使结果为实数。"""
    if min(l1, l2, L) < 0 or not abs(l1 - l2) <= L <= l1 + l2:
        raise ValueError(f"非法 CG 路径 {l1} x {l2} -> {L}")
    entries = []
    for Q in range(-L, L + 1):
        plane = []
        for q1 in range(-l1, l1 + 1):
            row = []
            for q2 in range(-l2, l2 + 1):
                value = sum(
                    _real_basis_entry(L, Q, M)
                    * _real_basis_entry(l1, q1, m1).conjugate()
                    * _real_basis_entry(l2, q2, m2).conjugate()
                    * clebsch_gordan_value(l1, m1, l2, m2, L, M)
                    for M in range(-L, L + 1)
                    for m1 in range(-l1, l1 + 1)
                    for m2 in range(-l2, l2 + 1))
                row.append(value)
            plane.append(row)
        entries.append(plane)
    norm_re = math.sqrt(sum(z.real ** 2 for p in entries for r in p for z in r))
    norm_im = math.sqrt(sum(z.imag ** 2 for p in entries for r in p for z in r))
    if min(norm_re, norm_im) > 1e-10 * max(norm_re, norm_im, 1.):
        raise RuntimeError("实 CG 换基出现异常的混合相位。")
    return tuple(tuple(tuple(float(z.real if norm_re >= norm_im else z.imag)
                             for z in row) for row in plane) for plane in entries)


def feature_dtype(config: SO3Config):
    """返回主干表示分量的 dtype；末端转换模式的主干也是复数。"""
    real = config_dtype(config)
    if _basis(config) == "real":
        return real
    return jnp.complex64 if real == jnp.float32 else jnp.complex128


@functools.lru_cache(maxsize=None)
def _standard_cg_data(l1: int, l2: int, L: int):
    """缓存标准复球谐基的 CG 数值；CG 本身为实数常数。"""
    if min(l1, l2, L) < 0 or not abs(l1 - l2) <= L <= l1 + l2:
        raise ValueError(f"非法 CG 路径 {l1} x {l2} -> {L}")
    return tuple(
        tuple(
            tuple(
                clebsch_gordan_value(l1, m1, l2, m2, L, M)
                for m2 in range(-l2, l2 + 1)
            )
            for m1 in range(-l1, l1 + 1)
        )
        for M in range(-L, L + 1)
    )


def cg_array(l1: int, l2: int, L: int, *, basis: str = "real", dtype=None) -> Array:
    """返回 CG[L 分量, 左输入分量, 右输入分量]。

    复基直接使用标准 CG；实基在三个指标上换基并选定整体相位。
    同一条耦合路径的两个输入及输出必须采用与此张量一致的基。
    """
    if basis == "real":
        return jnp.asarray(_real_cg_data(l1, l2, L), dtype=dtype or jnp.float32)
    if basis != "complex":
        raise ValueError(f"未知 basis={basis!r}")
    return jnp.asarray(_standard_cg_data(l1, l2, L), dtype=dtype or jnp.complex64)


def basis_change_matrix(l: int, dtype=jnp.complex64) -> Array:
    """复分量到本项目实球谐分量的矩阵 S_l：Y_real=S_l @ Y_complex。

    S_l 元素可能是虚数，但乘标准复球谐后得到实值函数。网络中的
    复特征即使没有球谐的共轭约束，乘 S_l 后仍可能保留虚部。
    """
    return jnp.asarray(
        [
            [_real_basis_entry(l, q, m)
             for m in range(-l, l + 1)]
            for q in range(-l, l + 1)
        ],
        dtype=dtype,
    )


def split_real_multiplet(complex_features: Array, l: int) -> Tuple[Array, Array]:
    """把标准复基分量 (...,2l+1) 换到实基，再分开实部和虚部。

    实基旋转矩阵为实矩阵，所以两份输出分别保持 l 阶等变性。
    任意网络复特征不一定满足球谐函数的共轭关系，虚部不一定为零。
    """
    complex_features = jnp.asarray(complex_features)
    if not jnp.issubdtype(complex_features.dtype, jnp.complexfloating):
        raise ValueError("split_real_multiplet 需要复基特征。")
    if complex_features.shape[-1] != 2 * l + 1:
        raise ValueError("最后一维必须是 2*l+1。")
    S = basis_change_matrix(l, complex_features.dtype)
    converted = jnp.einsum("qm,...m->...q", S, complex_features)
    return jnp.real(converted), jnp.imag(converted)


def _associated_legendre(l: int, m: int, x: Array) -> Array:
    """计算含 Condon--Shortley 相位的 P_l^m(x)，供高阶球谐使用。"""
    if not 0 <= m <= l:
        raise ValueError("要求 0<=m<=l。")
    double_factorial = math.prod(range(1, 2 * m, 2))
    pmm = ((-1) ** m * double_factorial
           * jnp.maximum(0., 1. - x * x) ** (m / 2))
    if l == m:
        return pmm
    pm1 = (2 * m + 1) * x * pmm
    if l == m + 1:
        return pm1
    pm2 = pmm
    for ell in range(m + 2, l + 1):
        pm2, pm1 = pm1, ((2 * ell - 1) * x * pm1
                           - (ell + m - 1) * pm2) / (ell - m)
    return pm1


def _real_spherical_harmonics(l: int, unit: Array) -> Array:
    """实数路径直接用多项式或三角函数，不经过复数中间数组。"""
    x, y, z = unit[..., 0], unit[..., 1], unit[..., 2]
    if l == 0:
        return (jnp.ones_like(x) / math.sqrt(4 * math.pi))[..., None]
    if l == 1:
        return math.sqrt(3 / (4 * math.pi)) * jnp.stack((y, z, x), -1)
    if l == 2:
        c0 = math.sqrt(5 / (16 * math.pi))
        c1 = math.sqrt(15 / (4 * math.pi))
        c2 = math.sqrt(15 / (16 * math.pi))
        return jnp.stack((2 * c2 * x * y, c1 * y * z,
                          c0 * (3 * z * z - 1), c1 * x * z,
                          c2 * (x * x - y * y)), -1)
    phi = jnp.arctan2(y, x)
    z = jnp.clip(z, -1., 1.)
    values = []
    for q in range(-l, l + 1):
        m = abs(q)
        norm = math.sqrt((2 * l + 1) / (4 * math.pi)
                         * math.factorial(l - m) / math.factorial(l + m))
        p = _associated_legendre(l, m, z)
        if q == 0:
            values.append(norm * p)
        elif q > 0:
            values.append(math.sqrt(2) * (-1) ** m * norm * p * jnp.cos(m * phi))
        else:
            values.append(math.sqrt(2) * (-1) ** m * norm * p * jnp.sin(m * phi))
    return jnp.stack(values, -1)


def spherical_harmonics(l: int, unit: Array, *, basis: str = "real") -> Array:
    """单位方向 (...,3) -> 同阶的 (2l+1) 个球谐值。

    l<=2 直接使用 Cartesian 多项式，避免 atan2 和极点导数；高阶
    高阶使用缔合 Legendre 递推。两种基都按 -l,...,+l 排列。
    此处实基的 l=1 为比例系数乘 [y,z,x]。
    """
    unit = jnp.asarray(unit)
    if unit.shape[-1] != 3:
        raise ValueError("球谐函数要求最后一维为三维方向。")
    if l < 0:
        raise ValueError("l 必须为非负整数。")
    if basis == "real":
        return _real_spherical_harmonics(l, unit)
    if basis != "complex":
        raise ValueError(f"未知 basis={basis!r}")

    x, y, z = unit[..., 0], unit[..., 1], unit[..., 2]
    cdtype = jnp.complex64 if unit.dtype == jnp.float32 else jnp.complex128
    xy = x + 1j * y
    if l == 0:
        return jnp.stack([jnp.ones_like(x) / math.sqrt(4 * math.pi)], -1).astype(cdtype)
    if l == 1:
        c = math.sqrt(3 / (8 * math.pi))
        return jnp.stack([c * jnp.conj(xy),
                          math.sqrt(3 / (4 * math.pi)) * z,
                          -c * xy], -1).astype(cdtype)
    if l == 2:
        c2 = math.sqrt(15 / (32 * math.pi))
        c1 = math.sqrt(15 / (8 * math.pi))
        return jnp.stack([
            c2 * jnp.conj(xy) ** 2,
            c1 * z * jnp.conj(xy),
            math.sqrt(5 / (16 * math.pi)) * (3 * z * z - 1),
            -c1 * z * xy,
            c2 * xy ** 2,
        ], -1).astype(cdtype)

    # l>2 使用 atan2；极点自动微分敏感，生产用途建议改用 solid harmonics。
    z = jnp.clip(z, -1, 1)
    phi = jnp.arctan2(y, x)
    positive = {}
    for m in range(l + 1):
        norm = math.sqrt((2 * l + 1) / (4 * math.pi)
                         * math.factorial(l - m) / math.factorial(l + m))
        positive[m] = norm * _associated_legendre(l, m, z) * jnp.exp(1j * m * phi)
    return jnp.stack([
        (-1) ** (-m) * jnp.conj(positive[-m]) if m < 0 else positive[m]
        for m in range(-l, l + 1)
    ], axis=-1).astype(cdtype)


def all_spherical_harmonics(lmax: int, unit: Array, *, basis: str = "real") -> Dict[int, Array]:
    """同一次几何计算得到 l=0,...,lmax 的角向边特征。"""
    return {l: spherical_harmonics(l, unit, basis=basis) for l in range(lmax + 1)}


def _randn(key, shape, scale, dtype) -> Array:
    """显式随机 key 的缩放高斯初始化。"""
    return jnp.asarray(scale, dtype) * jax.random.normal(key, shape, dtype)


def _init_real_so3_params(key, type_dim: int, config: SO3Config):
    """构造共享参数树；只混 channel，不混同阶表示的角分量。"""
    dtype = config_dtype(config)
    C = int(config.channels)
    B = int(config.wbn_bottleneck or C)
    R = int(config.det_rank)
    if min(C, B, R, type_dim, config.n_rbf) <= 0 or config.lmax < 0:
        raise ValueError("channels、det_rank、type_dim、n_rbf 必须为正，lmax 必须非负。")
    if config.output_projection not in ("q0", "sum"):
        raise ValueError("output_projection 只支持 'q0' 或 'sum'。")

    key, k = jax.random.split(key)
    params = {"init_w0": _randn(k, (C, type_dim), 1 / math.sqrt(type_dim), dtype)}
    if config.init_l1 and config.lmax >= 1:
        key, k1, k2 = jax.random.split(key, 3)
        params["init_l1_type"] = _randn(k1, (C, type_dim), 1 / math.sqrt(type_dim), dtype)
        params["init_l1_rbf"] = _randn(k2, (C, config.n_rbf),
                                        1 / math.sqrt(config.n_rbf), dtype)

    paths = admissible_paths(config.lmax)
    conv = []
    for _ in range(config.n_layers):
        layer = {"self": {}, "paths": {}}
        for l in range(config.lmax + 1):
            key, k = jax.random.split(key)
            layer["self"][str(l)] = _randn(k, (C, C), 1 / math.sqrt(C), dtype)
        for li, lf, lo in paths:
            key, k = jax.random.split(key)
            layer["paths"][f"{li}_{lf}_{lo}"] = _randn(
                k, (config.n_rbf, C, C), 1 / math.sqrt(config.n_rbf * C), dtype)
        conv.append(layer)
    params["conv_layers"] = tuple(conv)

    blocks = []
    for _ in range(config.wbn_layers):
        block = {"self": {}, "paths": {}}
        for l in range(config.lmax + 1):
            key, k = jax.random.split(key)
            block["self"][str(l)] = _randn(k, (C, C), 1 / math.sqrt(C), dtype)
        for l1, l2, lo in paths:
            key, kl, kr, ko = jax.random.split(key, 4)
            block["paths"][f"{l1}_{l2}_{lo}"] = {
                "left": _randn(kl, (B, C), 1 / math.sqrt(C), dtype),
                "right": _randn(kr, (B, C), 1 / math.sqrt(C), dtype),
                "out": _randn(ko, (C, B), 1 / math.sqrt(B), dtype),
            }
        blocks.append(block)
    params["wbn"] = tuple(blocks)
    params["readout"] = {}
    for l in range(config.lmax + 1):
        key, k = jax.random.split(key)
        params["readout"][str(l)] = _randn(k, (R, C), 1 / math.sqrt(C), dtype)
    return params


def init_so3_params(key, type_dim: int, config: SO3Config) -> Mapping[str, object]:
    """初始化主干；末端转换模式单独初始化实数的 2C 输入读出层。"""
    mode = _basis(config)
    if mode == "real":
        return _init_real_so3_params(key, type_dim, config)
    if mode == "complex":
        kr, ki = jax.random.split(key)
    else:
        kr, ki, kout = jax.random.split(key, 3)
    real = _init_real_so3_params(kr, type_dim, config)
    imag = _init_real_so3_params(ki, type_dim, config)
    params = jax.tree_util.tree_map(lambda r, i: (r + 1j * i) / math.sqrt(2), real, imag)
    if mode == "complex_to_real":
        C, R = int(config.channels), int(config.det_rank)
        keys = jax.random.split(kout, config.lmax + 1)
        params["readout"] = {
            str(l): _randn(keys[l], (R, 2 * C), 1 / math.sqrt(2 * C), config_dtype(config))
            for l in range(config.lmax + 1)
        }
    return params


def _edge_data(positions: Array, config: SO3Config):
    """几何、径向基和球谐只算一次，供初始化与所有卷积层复用。"""
    _, dist, unit, nonself = pair_geometry(positions, config.eps)
    rbf = radial_basis(dist, config.n_rbf, config.cutoff)
    rbf = rbf * nonself[..., None]  # 排除 i=j 自边。
    Y = all_spherical_harmonics(config.lmax, unit, basis=_backbone_basis(config))
    return rbf, Y


def _initial_features(params, type_features, rbf, Y, config):
    """l=0 是类型标量；可选 l=1 由邻居方向和径向标量生成。"""
    n = type_features.shape[0]
    C = config.channels
    dtype = feature_dtype(config)
    h = {l: jnp.zeros((n, C, 2 * l + 1), dtype=dtype)
         for l in range(config.lmax + 1)}
    h[0] = jnp.einsum("ct,nt->nc", params["init_w0"], type_features)[..., None]
    if config.init_l1 and config.lmax >= 1:
        sender = jnp.einsum("ct,nt->nc", params["init_l1_type"], type_features)
        radial = jnp.einsum("cr,ijr->ijc", params["init_l1_rbf"], rbf)
        h[1] = jnp.einsum("ijc,ijq->icq", radial * sender[None, :, :], Y[1])
    return h


def _cg_table(lmax: int, basis: str, dtype) -> Dict[Tuple[int, int, int], Array]:
    """静态路径的 CG 张量在一次前向传播内共享。"""
    return {path: cg_array(*path, basis=basis, dtype=dtype)
            for path in admissible_paths(lmax)}


def _convolution(h, rbf, Y, CG, params, config):
    """一层等变消息：径向标量 × 发送特征 × 方向球谐 × CG。

    einsum 指标：i/j=接收/发送粒子；o/c=输出/输入通道；
    p/m/n=输出/输入/球谐分量；r=径向基。直接收缩，避免显式
    构造原实现的 (N,N,Cout,Cin) 边权重张量。
    """
    out = {l: jnp.einsum("oc,icq->ioq", params["self"][str(l)], h[l])
           for l in h}
    for li, lf, lo in CG:
        W = params["paths"][f"{li}_{lf}_{lo}"]
        msg = jnp.einsum("ijr,roc,jcm,ijn,pmn->iop",
                         rbf, W, h[li], Y[lf], CG[li, lf, lo], optimize=True)
        out[lo] = out[lo] + msg
    root2 = jnp.asarray(math.sqrt(2), dtype=config_dtype(config))
    return {l: (h[l] + out[l]) / root2 for l in h}


def _bilinear(h, CG, params, config):
    """逐粒子 CG 张量积：提供不破坏表示类型的双线性非线性。"""
    out = {l: jnp.einsum("oc,icq->ioq", params["self"][str(l)], h[l])
           for l in h}
    for l1, l2, lo in CG:
        p = params["paths"][f"{l1}_{l2}_{lo}"]
        left = jnp.einsum("bc,icm->ibm", p["left"], h[l1])
        right = jnp.einsum("bc,icn->ibn", p["right"], h[l2])
        coupled = jnp.einsum("ibm,ibn,pmn->ibp",
                             left, right, CG[l1, l2, lo], optimize=True)
        out[lo] = out[lo] + jnp.einsum("ob,ibp->iop", p["out"], coupled)
    root2 = jnp.asarray(math.sqrt(2), dtype=config_dtype(config))
    return {l: (h[l] + out[l]) / root2 for l in h}


def apply_so3_network(params, positions: Array, type_features: Array, config: SO3Config):
    """执行 SO(3) 主干，返回 ``h/y/A_channels/A`` 四级结果。

    positions 为 (N,3)，type_features 为 (N,T)。变更模式后需重新
    初始化参数，不能直接复用旧参数。
    complex_to_real 模式的 h 是复数主干结果，y 是 2C 个实 channel；
    real 和 complex 模式的 h 与 y 相同。
    """
    mode = _basis(config)
    basis = _backbone_basis(config)
    positions = jnp.asarray(positions, dtype=config_dtype(config))
    type_features = jnp.asarray(type_features, dtype=feature_dtype(config))
    if positions.ndim != 2 or positions.shape[-1] != 3:
        raise ValueError("positions 的 shape 必须是 (N,3)。")
    if type_features.ndim != 2 or positions.shape[0] != type_features.shape[0]:
        raise ValueError("type_features 的 shape 必须是 (N,T)，与坐标逐粒子对应。")
    if (jnp.issubdtype(params["init_w0"].dtype, jnp.complexfloating)
            != (basis == "complex")):
        raise ValueError("当前模式需要重新初始化对应数域的 SO(3) 参数。")
    rbf, Y = _edge_data(positions, config)
    CG = _cg_table(config.lmax, basis, feature_dtype(config))
    h = _initial_features(params, type_features, rbf, Y, config)
    for layer in params["conv_layers"]:
        h = _convolution(h, rbf, Y, CG, layer, config)
    for block in params["wbn"]:
        h = _bilinear(h, CG, block, config)
    if mode == "complex_to_real":
        y = {}
        for l, values in h.items():
            real, imag = split_real_multiplet(values, l)
            # channel 数固定为 2C，与当前输入的虚部是否恰为零无关。
            y[l] = jnp.concatenate((real, imag), axis=1)
    else:
        y = h
    channels = {l: jnp.sum(y[l], axis=0) for l in y}
    A = {l: jnp.einsum("rc,cq->rq", params["readout"][str(l)], channels[l])
         for l in y}
    return {"h": h, "y": y, "A_channels": channels, "A": A}


def scalar_orbital_feature_dim(config: SO3Config) -> int:
    """实轨道输入维度；末端转换模式对每个 l 都有 2C 个实 channel。"""
    C = int(config.channels)
    mode = _basis(config)
    scalar_channels = 2 * C if mode != "real" else C
    higher_channels = 2 * C if mode == "complex_to_real" else C
    return scalar_channels + (int(config.lmax) * higher_channels
                              if config.use_invariant_orbital_features else 0)


def make_scalar_orbital_features(y: Mapping[int, Array], nelec: int, config: SO3Config) -> Array:
    """将电子表示映射为实数旋转不变量，供原有 Slater 轨道头使用。

    复 l=0 分量本身是复标量，实部与虚部各自不变。高阶部分使用
    sqrt(mean_q |y_q|²+eps²)；Wigner D 的幺正性保证其旋转不变。
    因为使用模长，高阶特征的相位及方向不会进入行列式。
    complex_to_real 的 y 已是实数、channel 数已翻倍，不再分离虚部。
    """
    y0 = jnp.asarray(y[0][:nelec, :, 0])
    parts = [jnp.real(y0)]
    if _basis(config) == "complex":
        parts.append(jnp.imag(y0))
    if config.use_invariant_orbital_features:
        for l in range(1, config.lmax + 1):
            yl = jnp.asarray(y[l][:nelec])
            norm2 = jnp.mean(jnp.real(yl * jnp.conj(yl)), axis=-1)
            parts.append(jnp.sqrt(norm2 + jnp.asarray(config.eps ** 2, norm2.dtype)))
    out = jnp.concatenate(parts, axis=-1)
    if out.shape[-1] != scalar_orbital_feature_dim(config):
        raise ValueError("轨道标量输入维度与配置不符。")
    return out


def phase_logsumexp(phases: Array, logabs: Array, axis=0):
    """稳定求和复数项 ``phase*exp(logabs)``；实数 sign 也适用。

    输出 phase 在非零处为单位模；精确抵消时返回 (0,-inf)。
    非有限 logabs 当作无效项跳过，不会凭空修复上游 NaN。
    """
    finite = jnp.isfinite(logabs)
    safe = jnp.where(finite, logabs, -jnp.inf)
    maximum = jnp.max(safe, axis=axis)
    base = jnp.where(jnp.isfinite(maximum), maximum, 0)
    scaled = jnp.sum(phases * jnp.exp(safe - jnp.expand_dims(base, axis)), axis=axis)
    magnitude = jnp.abs(scaled)
    floor = jnp.asarray(jnp.finfo(logabs.dtype).tiny, dtype=logabs.dtype)
    phase = jnp.where(magnitude > 0, scaled / jnp.maximum(magnitude, floor), 0)
    out_log = jnp.where(magnitude > 0, base + jnp.log(jnp.maximum(magnitude, floor)), -jnp.inf)
    return phase, out_log


def _weighted_phase_sum(weights: Array, logabs: Array, axis=0):
    """把普通复系数乘到 signed-log 项，再沿给定轴稳定相加。"""
    mag = jnp.abs(weights)
    floor = jnp.asarray(jnp.finfo(logabs.dtype).tiny, dtype=logabs.dtype)
    phases = weights / jnp.maximum(mag, floor)
    logs = jnp.where((mag > 0) & jnp.isfinite(logabs),
                     logabs + jnp.log(jnp.maximum(mag, floor)), -jnp.inf)
    return phase_logsumexp(phases, logs, axis=axis)


def sector_wavefunction_multiplets(det_signs, det_logabs, A, det_params, sectors):
    """构造每阶完整波函数 multiplet，返回 phase[l] 与 logabs[l]。

    实 A 得到实 sign，复 A 得到复 phase；两者都用缩放求和。
    行列式仍是实标量且不带 q，所以每个 l 的复 multiplet 保持等变。
    """
    sectors = tuple(int(l) for l in sectors)
    if not sectors:
        raise ValueError("sectors 不能为空。")
    rank_sign, rank_logabs = rank_determinant_sums(
        det_signs, det_logabs, det_params, sectors)
    phases, logs = {}, {}
    for l in sectors:
        a = jnp.asarray(A[l])
        if a.shape != (rank_sign[l].shape[0], 2 * l + 1):
            raise ValueError(f"l={l} 的 A 应为 (R,{2*l+1})。")
        phases[l], logs[l] = _weighted_phase_sum(
            a * rank_sign[l][:, None], rank_logabs[l][:, None], axis=0)
    return phases, logs


def project_sector_multiplets(multiplet_phase, multiplet_logabs, sectors, projection="q0"):
    """取 q=0 或全部 q 的固定和，并进一步对 l 求和。

    这是为了适配标量波函数接口；l>0 的固定投影一般不是旋转不变量。
    """
    sectors = tuple(int(l) for l in sectors)
    if not sectors:
        raise ValueError("sectors 不能为空。")
    phases, logs = [], []
    for l in sectors:
        if projection == "q0":
            p, q = multiplet_phase[l][l], multiplet_logabs[l][l]
        elif projection == "sum":
            p, q = phase_logsumexp(multiplet_phase[l], multiplet_logabs[l], axis=0)
        else:
            raise ValueError("projection 只支持 'q0' 或 'sum'。")
        phases.append(p)
        logs.append(q)
    return phase_logsumexp(jnp.stack(phases), jnp.stack(logs), axis=0)


def assemble_sector_wavefunction(det_signs, det_logabs, A, det_params, sectors,
                                 projection="q0", jastrow_log=None, eps=1e-30):
    """完整 multiplet -> 固定投影 -> 公共 Jastrow，返回 (phase,logabs)。

    ``eps`` 仅为与原函数兼容。exp(J)>0，所以 J 只加在 logabs 上。
    """
    del eps
    multiplet_phase, multiplet_logabs = sector_wavefunction_multiplets(
        det_signs, det_logabs, A, det_params, sectors)
    phase, logabs = project_sector_multiplets(
        multiplet_phase, multiplet_logabs, sectors, projection)
    if jastrow_log is not None:
        logabs = logabs + jnp.asarray(jastrow_log, dtype=logabs.dtype)
    return phase, logabs


def sector_term_log_distribution(det_logabs, A, det_params, sectors,
                                 projection="q0", eps=1e-30):
    """诊断各 (l,r,k) 项的对数模权重；复相位与符号抵消不计入。"""
    sectors = tuple(int(l) for l in sectors)
    if not sectors:
        raise ValueError("sectors 不能为空。")
    logs = []
    for l in sectors:
        a = jnp.asarray(A[l])
        if projection == "q0":
            b = a[:, l]
        elif projection == "sum":
            b = jnp.sum(a, axis=-1)
        else:
            raise ValueError("projection 只支持 'q0' 或 'sum'。")
        alpha = jnp.asarray(det_params["alpha"][str(l)])
        if alpha.shape != det_logabs[l].shape or b.shape[0] != alpha.shape[0]:
            raise ValueError(f"l={l} 的 alpha、A 或 determinant 维度不匹配。")
        floor = jnp.asarray(eps, dtype=det_logabs[l].dtype)
        logs.append((det_logabs[l] + jnp.log(jnp.maximum(jnp.abs(alpha), floor))
                     + jnp.log(jnp.maximum(jnp.abs(b), floor))[:, None]).reshape(-1))
    joined = jnp.concatenate(logs)
    return jax.nn.log_softmax(jnp.where(jnp.any(jnp.isfinite(joined)), joined,
                                       jnp.zeros_like(joined)))


def init_wavefunction_params(key, type_dim: int, nspins: Sequence[int], natom: int,
                             determinants_per_rank: int, config: SO3Config,
                             *, full_det=True, include_bias=False):
    """一次初始化主干、实标量行列式库和可选 Jastrow 参数。"""
    k_so3, k_det, k_jastrow = jax.random.split(key, 3)
    sectors = sector_list(config)
    return {
        "so3": init_so3_params(k_so3, type_dim, config),
        "so3_det": init_sector_determinant_params(
            k_det, scalar_orbital_feature_dim(config), nspins,
            determinants_per_rank, sectors, config.det_rank, natom,
            full_det=full_det, include_bias=include_bias,
            float_dtype=config.float_dtype, use_envelope=config.use_envelope,
            envelope_sigma_init=config.envelope_sigma_init,
            envelope_sigma_floor=config.envelope_sigma_floor,
            envelope_pi_init=config.envelope_pi_init,
            det_bank_mode=config.det_bank_mode),
        "so3_jastrow": (
            init_kato_jastrow_params(k_jastrow, nspins,
                                           config.jastrow_b_init, config.float_dtype)
            if config.use_kato_jastrow else None),
    }


def evaluate_wavefunction(params, electron_pos: Array, atoms: Array,
                          type_features: Array, nspins: Sequence[int],
                          determinants_per_rank: int, config: SO3Config,
                          *, full_det=True):
    """端到端求值；返回完整 multiplet 与固定投影的 phase/logabs。

    type_features 必须按“全部电子、全部原子核”顺序排列。实模式和
    末端转换模式的 phase 为 +/-1/0；复模式为单位模复数或零。实际值为
    ``phase * exp(logabs)``。返回的 ``multiplet_*`` 按整数 l 索引。
    """
    dtype = config_dtype(config)
    nelec = int(sum(nspins))
    electron_pos = jnp.asarray(electron_pos, dtype=dtype).reshape((nelec, 3))
    atoms = jnp.asarray(atoms, dtype=dtype).reshape((-1, 3))
    positions = jnp.concatenate((electron_pos, atoms), axis=0)
    backbone = apply_so3_network(params["so3"], positions, type_features, config)
    scalar = make_scalar_orbital_features(backbone["y"], nelec, config)
    sectors = sector_list(config)
    det_sign, det_logabs, orbitals = sector_determinants_from_scalar_features(
        scalar, electron_pos, atoms, params["so3_det"], nspins,
        determinants_per_rank, sectors, config.det_rank, full_det=full_det,
        envelope_sigma_floor=config.envelope_sigma_floor, eps=config.eps,
        det_bank_mode=config.det_bank_mode)
    multiplet_phase, multiplet_logabs = sector_wavefunction_multiplets(
        det_sign, det_logabs, backbone["A"], params["so3_det"], sectors)
    phase, logabs = project_sector_multiplets(
        multiplet_phase, multiplet_logabs, sectors, config.output_projection)
    if params.get("so3_jastrow") is not None:
        logabs = logabs + kato_ee_jastrow(
            electron_pos, nspins, params["so3_jastrow"], config.eps)
    return {
        "phase": phase, "logabs": logabs,
        "multiplet_phase": multiplet_phase,
        "multiplet_logabs": multiplet_logabs,
        "backbone": backbone,
        "det_sign": det_sign, "det_logabs": det_logabs,
        "orbitals": orbitals,
    }


def _inverse_softplus(value: float, dtype) -> Array:
    """给正的目标值计算 softplus 的无约束初值。"""
    value = max(float(value), 1e-8)
    return jnp.asarray(math.log(math.expm1(value)), dtype=dtype)


def _init_envelope(key, K, norb, natom, sigma_init, sigma_floor, pi_init, dtype):
    """各向同性核包络：sigma、pi 的 shape 都是 (K,norb,natom)。"""
    if sigma_init <= sigma_floor:
        raise ValueError("envelope_sigma_init 必须大于 envelope_sigma_floor。")
    raw = _inverse_softplus(sigma_init - sigma_floor, dtype)
    raw_sigma = jnp.full((K, norb, natom), raw, dtype=dtype)
    raw_sigma = raw_sigma + _randn(key, raw_sigma.shape, 1e-3, dtype)
    pi = jnp.full((K, norb, natom), pi_init / natom, dtype=dtype)
    return {"raw_sigma": raw_sigma, "pi": pi}


def _init_bank(key, input_dim, nspins, K, R, natom, full_det, include_bias,
               dtype, use_envelope, sigma_init, sigma_floor, pi_init):
    """一个 R×K 标量 Slater 库；每个 rank 下按非空自旋块保存头。"""
    active = tuple(int(n) for n in nspins if int(n) > 0)
    nelec = sum(active)
    heads, envelopes = [], []
    for _ in range(R):
        spin_heads, spin_envs = [], []
        for nspin in active:
            norb = nelec if full_det else nspin
            key, kh = jax.random.split(key)
            head = {"w": _randn(kh, (input_dim, K * norb),
                                 1 / math.sqrt(input_dim), dtype)}
            if include_bias:
                head["b"] = jnp.zeros((K * norb,), dtype=dtype)
            spin_heads.append(head)
            if use_envelope:
                key, ke = jax.random.split(key)
                spin_envs.append(_init_envelope(ke, K, norb, natom,
                                                sigma_init, sigma_floor, pi_init, dtype))
            else:
                spin_envs.append(None)
        heads.append(tuple(spin_heads))
        envelopes.append(tuple(spin_envs))
    return key, tuple(heads), tuple(envelopes)


def init_sector_determinant_params(
    key, input_dim: int, nspins: Sequence[int], determinants_per_rank: int,
    sectors: Sequence[int], det_rank: int, natom: int, full_det: bool = True,
    include_bias: bool = False, float_dtype=jnp.float32, use_envelope: bool = True,
    envelope_sigma_init: float = 1., envelope_sigma_floor: float = 1e-6,
    envelope_pi_init: float = 1., det_bank_mode: str = "per_l",
):
    """初始化实标量轨道头、可选核包络与按 l 独立的 alpha[l]:(R,K)。

    shared 仅生成一组昂贵的轨道库；per_l 为每阶分别生成。两种模式的
    alpha 都不共享。与 SO3Config 默认 shared 配合时须显式传 det_bank_mode。
    """
    dtype = jnp.dtype(float_dtype)
    if dtype not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)):
        raise ValueError("行列式参数只支持 float32/float64。")
    if dtype == jnp.dtype(jnp.float64) and not jax.config.jax_enable_x64:
        raise ValueError("使用 float64 前须启用 jax_enable_x64。")
    K, R, input_dim = int(determinants_per_rank), int(det_rank), int(input_dim)
    sectors = tuple(int(l) for l in sectors)
    mode = normalize_det_bank_mode(det_bank_mode)
    if min(K, R, input_dim) <= 0 or not sectors or sum(map(int, nspins)) <= 0:
        raise ValueError("K、R、input_dim、电子数和 sector 数都必须为正。")
    if use_envelope and natom <= 0:
        raise ValueError("启用核包络时至少需要一个原子核。")

    def create(k):
        return _init_bank(k, input_dim, nspins, K, R, int(natom), full_det,
                          include_bias, dtype, use_envelope, envelope_sigma_init,
                          envelope_sigma_floor, envelope_pi_init)

    if mode == "shared":
        key, heads, envelope = create(key)
    else:
        heads, envelope = {}, {}
        for l in sectors:
            key, heads[str(l)], envelope[str(l)] = create(key)
    alpha = {str(l): jnp.full((R, K), 1 / math.sqrt(R * K), dtype=dtype)
             for l in sectors}
    return {"heads": heads, "envelope": envelope, "alpha": alpha}


def apply_orbital_envelope(electron_pos: Array, atoms: Array,
                           envelope_params: Mapping[str, Array] | None,
                           sigma_floor: float = 1e-6, eps: float = 1e-12) -> Array:
    """按电子-核距离计算各向同性 exp(-sigma*r) 包络 (nspin,K,norb)。"""
    if envelope_params is None:
        raise ValueError("禁用 envelope 时不应调用此函数。")
    raw = envelope_params["raw_sigma"]
    pi = envelope_params["pi"]
    delta = jnp.asarray(electron_pos, raw.dtype)[:, None, :] - jnp.asarray(atoms, raw.dtype)[None, :, :]
    radius = jnp.sqrt(jnp.sum(delta * delta, axis=-1) +
                      jnp.asarray(eps, raw.dtype) ** 2)
    sigma = jax.nn.softplus(raw) + jnp.asarray(sigma_floor, raw.dtype)
    return jnp.sum(pi[None, ...] * jnp.exp(-radius[:, None, None, :]
                                            * sigma[None, ...]), axis=-1)


def _build_bank_orbitals(features, electron_pos, atoms, heads, envelopes,
                         nspins, K, R, full_det, sigma_floor, eps):
    """返回 R 个行列式矩阵组：完整矩阵或按自旋分块的矩阵。"""
    active = tuple(int(n) for n in nspins if int(n) > 0)
    nelec = sum(active)
    cut = tuple(sum(active[:i]) for i in range(1, len(active)))
    x_blocks = jnp.split(features, cut, axis=0)
    pos_blocks = jnp.split(electron_pos, cut, axis=0)
    results = []
    for r in range(R):
        spin_mats = []
        for x, pos, head, env, nspin in zip(x_blocks, pos_blocks,
                                             heads[r], envelopes[r], active):
            norb = nelec if full_det else nspin
            raw = x @ head["w"]
            if "b" in head:
                raw = raw + head["b"]
            mat = raw.reshape((nspin, K, norb))
            if env is not None:
                mat = mat * apply_orbital_envelope(pos, atoms, env, sigma_floor, eps)
            spin_mats.append(jnp.transpose(mat, (1, 0, 2)))
        results.append(jnp.concatenate(spin_mats, axis=1) if full_det
                       else tuple(spin_mats))
    return tuple(results)


def build_sector_orbitals(
    scalar_electron_features: Array, electron_pos: Array, atoms: Array,
    det_params: Mapping[str, object], nspins: Sequence[int],
    determinants_per_rank: int, sectors: Sequence[int], det_rank: int,
    full_det: bool = True, envelope_sigma_floor: float = 1e-6,
    eps: float = 1e-12, det_bank_mode: str = "per_l",
):
    """返回 orbitals[l][r]；共享库只计算一次并复用于全部 l。"""
    mode = normalize_det_bank_mode(det_bank_mode)
    sectors = tuple(int(l) for l in sectors)
    nelec = sum(int(n) for n in nspins if int(n) > 0)
    features = jnp.asarray(scalar_electron_features)
    if features.ndim != 2 or features.shape[0] != nelec or not sectors:
        raise ValueError("标量电子特征应为 (Ne,F)，且 sectors 非空。")
    positions = jnp.asarray(electron_pos, features.dtype).reshape((nelec, 3))
    atoms = jnp.asarray(atoms, features.dtype).reshape((-1, 3))
    K, R = int(determinants_per_rank), int(det_rank)

    def build(heads, envelopes):
        return _build_bank_orbitals(features, positions, atoms, heads, envelopes,
                                    nspins, K, R, full_det, envelope_sigma_floor, eps)

    if mode == "shared":
        shared = build(det_params["heads"], det_params["envelope"])
        return {l: shared for l in sectors}
    return {l: build(det_params["heads"][str(l)], det_params["envelope"][str(l)])
            for l in sectors}


def _bank_slogdet(orbitals, K: int, dtype, full_det: bool):
    """批量 slogdet；分块模式相乘符号、相加 log|det|。"""
    signs, logs = [], []
    for bank in orbitals:
        if full_det:
            sign, logabs = jnp.linalg.slogdet(bank)
        else:
            sign = jnp.ones((K,), dtype=dtype)
            logabs = jnp.zeros((K,), dtype=dtype)
            for block in bank:
                s, q = jnp.linalg.slogdet(block)
                sign, logabs = sign * s, logabs + q
        signs.append(sign)
        logs.append(logabs)
    return jnp.stack(signs), jnp.stack(logs)


def sector_determinants_from_scalar_features(
    scalar_electron_features: Array, electron_pos: Array, atoms: Array,
    det_params: Mapping[str, object], nspins: Sequence[int],
    determinants_per_rank: int, sectors: Sequence[int], det_rank: int,
    full_det: bool = True, envelope_sigma_floor: float = 1e-6,
    eps: float = 1e-12, det_bank_mode: str = "per_l",
):
    """返回 det_sign[l]、det_logabs[l]（均为 R×K）和轨道矩阵。"""
    mode = normalize_det_bank_mode(det_bank_mode)
    sectors = tuple(int(l) for l in sectors)
    orbitals = build_sector_orbitals(
        scalar_electron_features, electron_pos, atoms, det_params, nspins,
        determinants_per_rank, sectors, det_rank, full_det,
        envelope_sigma_floor, eps, mode)
    signs, logs = {}, {}
    if mode == "shared":
        s, q = _bank_slogdet(orbitals[sectors[0]], determinants_per_rank,
                            jnp.asarray(scalar_electron_features).dtype, full_det)
        signs = {l: s for l in sectors}
        logs = {l: q for l in sectors}
    else:
        for l in sectors:
            signs[l], logs[l] = _bank_slogdet(
                orbitals[l], determinants_per_rank,
                jnp.asarray(scalar_electron_features).dtype, full_det)
    return signs, logs, orbitals


def sector_determinants_from_y0(y0_electrons, electron_pos, atoms,
                                det_params, nspins, determinants_per_l,
                                sectors, full_det=True, eps=1e-12):
    """旧 det_rank=1 接口；高阶不变量应改用上一函数。"""
    heads = det_params["heads"]
    first = int(tuple(sectors)[0])
    R = len(heads[str(first)]) if isinstance(heads, dict) else len(heads)
    if R != 1:
        raise ValueError("此兼容入口只支持 det_rank=1。")
    return sector_determinants_from_scalar_features(
        y0_electrons, electron_pos, atoms, det_params, nspins,
        determinants_per_l, sectors, 1, full_det, eps=eps,
        det_bank_mode="per_l" if isinstance(heads, dict) else "shared")


def init_kato_jastrow_params(key, nspins, b_init=1., float_dtype=jnp.float32):
    """初始化同/异自旋 Padé 范围参数；key 保留作接口扩展。"""
    del key
    dtype = jnp.dtype(float_dtype)
    if sum(map(int, nspins)) <= 0:
        raise ValueError("至少需要一个电子。")
    raw = _inverse_softplus(b_init, dtype)
    return {"raw_b_same": raw, "raw_b_opposite": raw}


def kato_ee_jastrow(electron_pos, nspins, params, eps=1e-12) -> Array:
    """电子对 Padé Jastrow 的对数；同/异自旋目标斜率为 1/4、1/2。

    为避免 r=0 的导数问题，使用 sqrt(r²+eps²)；有限 eps 时严格
    零距离 cusp 斜率不再等于未平滑模型的 a。
    """
    pos = jnp.asarray(electron_pos)
    labels = jnp.asarray([i for i, n in enumerate(nspins) for _ in range(int(n))])
    if labels.shape[0] != pos.shape[0]:
        raise ValueError("nspins 总电子数与坐标行数不一致。")
    ii, jj = jnp.triu_indices(pos.shape[0], k=1)
    delta = pos[ii] - pos[jj]
    r = jnp.sqrt(jnp.sum(delta * delta, axis=-1) +
                 jnp.asarray(eps, pos.dtype) ** 2)
    same = labels[ii] == labels[jj]
    a = jnp.where(same, jnp.asarray(.25, pos.dtype), jnp.asarray(.5, pos.dtype))
    b_same = jax.nn.softplus(jnp.asarray(params["raw_b_same"], pos.dtype))
    b_opp = jax.nn.softplus(jnp.asarray(params["raw_b_opposite"], pos.dtype))
    b = jnp.where(same, b_same, b_opp)
    return jnp.sum(a * r / (1 + b * r))


def real_signed_logsumexp(signs: Array, logabs: Array):
    """所有项的稳定实数 signed-log 和；精确抵消返回 (0,-inf)。"""
    return phase_logsumexp(jnp.ravel(signs), jnp.ravel(logabs), axis=0)


def rank_determinant_sums(det_signs, det_logabs, det_params, sectors):
    """在 k 轴求 S[l,r]=sum_k alpha[l,r,k]*D[l,r,k]。"""
    rank_sign, rank_logabs = {}, {}
    for l in sectors:
        l = int(l)
        alpha = jnp.asarray(det_params["alpha"][str(l)], dtype=det_logabs[l].dtype)
        if alpha.shape != det_logabs[l].shape:
            raise ValueError(f"l={l} 的 alpha 与 determinant 维度不匹配。")
        rank_sign[l], rank_logabs[l] = _weighted_phase_sum(
            det_signs[l] * alpha, det_logabs[l], axis=1)
    return rank_sign, rank_logabs
