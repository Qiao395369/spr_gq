# ============================================================================
# SO(3) 等变网络工具函数库 —— 纯实数版本、中文逐行详注
# ============================================================================
# 这个文件只负责 SO(3) 网络中的数学与张量运算，不负责 VMC 优化器、Hamiltonian 或采样。
# 整个 SO(3) 分支从参数、隐藏特征、球谐函数、CG 张量、Slater 轨道到最终波函数全部使用实数浮点类型。
# 浮点类型通过 SO3Config.float_dtype 自定义，推荐使用 jnp.float32 或 jnp.float64。
# 如果选择 jnp.float64，必须在程序入口提前执行 jax.config.update("jax_enable_x64", True)。
# 整体数据流为：粒子坐标与类型 -> 初始 h_j^(l) -> 等变卷积 -> Wigner 双线性块 -> y_j^(l) -> A^(l)。
# 然后电子的 y_i^(0) 经过每个 l 独立的线性 orbital head，得到每个 l 对应的 K 个 Slater determinant D_{l,k}。
# 最终波函数为 Psi=sum_l sum_k alpha_{l,k} D_{l,k} q_l，其中 q_0=1，q_l=c_l^T A^(l)。
# 这里所有 c_l、alpha_{l,k}、D_{l,k}、q_l 与 Psi 都是实数，不再拆分实部/虚部。
# 本文件使用“实球谐基”表示整数阶 SO(3) 不可约表示，因此 l=1 可以理解为普通三维实向量，l=2 是五维实表示。
# 对每个 l，最后一维仍按 q=-l,...,+l 排列；q 只是实球谐基的分量标签，不再表示复数 Y_l^m 的相位分量。
# ==========================================================================
from __future__ import annotations
# dataclasses 用于定义不可变的网络配置对象。
import dataclasses
# functools 用于缓存 CG 常数，避免每次前向传播重复计算。
import functools
# math 提供阶乘、平方根、pi 等普通 Python 常数计算。
import math
# Any 用于允许 float_dtype 直接传入 jnp.float32、jnp.float64 或字符串。
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple
# JAX 提供自动微分、JIT 与随机数。
import jax
# jax.numpy 是本文件全部张量计算的后端。
import jax.numpy as jnp
# Array 只是为了让后面的类型注解更简洁。
Array = jnp.ndarray
# IrrepFeatures 的键是 l，值的统一 shape 是 (nparticle, channels, 2*l+1)。
IrrepFeatures = Dict[int, Array]
# frozen=True 保证配置对象创建后不被意外修改。
@dataclasses.dataclass(frozen=True)
class SO3Config:
    # lmax 是网络中允许出现的最大角动量阶数；实现本身支持任意非负整数，但实际计算量会随 lmax 快速增大。
    lmax: int = 2
    # channels 是每个 l sector 的 multiplicity/channel 数。
    channels: int = 16
    # n_layers 是 SO(3) 等变消息传递层数。
    n_layers: int = 3
    # n_rbf 是径向基函数个数。
    n_rbf: int = 8
    # cutoff 是可选截断半径；None 表示所有粒子对都参与消息传递。
    cutoff: Optional[float] = None
    # wbn_layers 是消息传递之后逐粒子 Wigner bilinear block 的数量。
    wbn_layers: int = 1
    # wbn_bottleneck 是 WBN 内部 bilinear channel 数；None 时等于 channels。
    wbn_bottleneck: Optional[int] = None
    # target_ls 指定最终真正进入波函数的非零 l sector；l=0 会自动加入，不需要写在这里。
    target_ls: Tuple[int, ...] = (1, 2)
    # init_l1 控制初始化时是否从粒子相对方向显式构造 l=1 几何特征。
    init_l1: bool = True
    # eps 是数值稳定用的小正数。
    eps: float = 1.0e-12
    # float_dtype 是整个 SO(3) 分支统一使用的实数浮点类型，可设为 jnp.float32、jnp.float64、"float32"、"float64"。
    float_dtype: Any = jnp.float32
    # use_envelope 控制是否在每个 Slater orbital 上乘 FermiNet 风格的原子核指数包络。
    use_envelope: bool = True
    # envelope_sigma_init 是 Sigma 矩阵的对角初值；1.0 对应初始各向同性 exp(-|r-R_m|) 尺度。
    envelope_sigma_init: float = 1.0
    # envelope_pi_init 是每个原子核 envelope mixing coefficient pi 的总尺度；内部会除以 natom。
    envelope_pi_init: float = 1.0
    # use_kato_jastrow 控制是否在最终波函数上乘满足电子-电子 Kato cusp 的 Padé Jastrow。
    use_kato_jastrow: bool = True
    # jastrow_b_init 是 Padé 分母 1+b*r 中 b 的正初值；cusp 斜率由固定 a 系数严格控制。
    jastrow_b_init: float = 1.0
# 把用户给出的 dtype 统一解析成 JAX dtype，并限制为 float32 或 float64。
def resolve_float_dtype(dtype_like: Any):
    # jnp.dtype 可以同时接受 jnp.float32、jnp.float64 和字符串。
    dtype = jnp.dtype(dtype_like)
    # float32 是默认推荐类型，速度快、显存占用较低。
    if dtype == jnp.dtype(jnp.float32):
        return jnp.float32
    # float64 更适合高精度 QMC，但 JAX 必须显式开启 x64。
    if dtype == jnp.dtype(jnp.float64):
        # 如果没有启用 x64，这里直接报错，避免 JAX 静默把 float64 截成 float32。
        if not bool(jax.config.jax_enable_x64):
            raise ValueError(
                'SO3Config.float_dtype=float64，但 JAX x64 尚未开启；请在程序入口先执行 '
                'jax.config.update("jax_enable_x64", True)。'
            )
        return jnp.float64
    # 其它低精度类型暂时不开放，因为 slogdet 与角动量耦合对数值精度比较敏感。
    raise ValueError(f'float_dtype 目前只支持 float32 或 float64，收到 {dtype}.')
# 返回配置中实际使用的 JAX 浮点类型。
def config_dtype(config: SO3Config):
    # 所有参数初始化和前向传播都应调用同一个解析函数，保证 dtype 全局一致。
    return resolve_float_dtype(config.float_dtype)

# ============================================================================
# NaN / Inf 调试开关。
# SO3_DEBUG=True: 打印每个主要阶段的统计量。
# SO3_DEBUG_VERBOSE=True: 额外打印每条 CG / Wigner coupling path 的统计量。
# 注意：这里使用 jax.debug.print，而不是 Python print，因此在 jit/vmap/pmap 下也能看到运行时数值。
# ============================================================================
SO3_DEBUG = False
SO3_DEBUG_VERBOSE = False


def _debug_array(name: str, x: Array):
    """打印数组的数值健康度摘要，不直接打印整个大数组。"""
    if not SO3_DEBUG:
        return
    x = jnp.asarray(x)
    finite = jnp.isfinite(x)
    safe = jnp.where(finite, x, jnp.asarray(0.0, dtype=x.dtype))
    nfinite = jnp.sum(finite)
    denom = jnp.maximum(nfinite, jnp.asarray(1, dtype=nfinite.dtype))
    xmin = jnp.min(jnp.where(finite, x, jnp.asarray(jnp.inf, dtype=x.dtype)))
    xmax = jnp.max(jnp.where(finite, x, jnp.asarray(-jnp.inf, dtype=x.dtype)))
    mean = jnp.sum(safe) / denom.astype(x.dtype)
    norm = jnp.linalg.norm(safe)
    jax.debug.print(
        f"[SO3 DEBUG] {name} shape={x.shape} dtype={x.dtype} "
        "nan={nan} +inf={pinf} -inf={ninf} finite={nfinite} "
        "min={xmin:.6e} max={xmax:.6e} mean={mean:.6e} norm={norm:.6e}",
        nan=jnp.sum(jnp.isnan(x)),
        pinf=jnp.sum(jnp.isposinf(x)),
        ninf=jnp.sum(jnp.isneginf(x)),
        nfinite=nfinite,
        xmin=xmin,
        xmax=xmax,
        mean=mean,
        norm=norm,
    )


def _debug_scalar(name: str, x):
    """打印标量；仍然使用 jax.debug.print 以兼容 jit。"""
    if not SO3_DEBUG:
        return
    x = jnp.asarray(x)
    jax.debug.print(
        f"[SO3 DEBUG] {name}={{x}} finite={{finite}} nan={{nan}} inf={{inf}}",
        x=x,
        finite=jnp.isfinite(x),
        nan=jnp.isnan(x),
        inf=jnp.isinf(x),
    )


# Python 整数阶乘包装，用于 CG 封闭公式。
def _fact(n: int) -> int:
    # math.factorial 返回精确整数。
    return math.factorial(n)
# 计算标准复球谐基中的 CG 数值；对于整数角动量和 Condon--Shortley 约定，这个数本身是实数。
@functools.lru_cache(maxsize=None)
def clebsch_gordan_value(l1: int, m1: int, l2: int, m2: int, L: int, M: int) -> float:
    # CG 系数要求磁量子数守恒 M=m1+m2。
    if M != m1 + m2:
        return 0.0
    # 三角条件和输出磁量子数范围必须同时满足。
    if L < abs(l1 - l2) or L > l1 + l2 or abs(M) > L:
        return 0.0
    # 这一部分是 Racah 封闭公式中的三角归一化因子。
    pref_triangle = (
        (2 * L + 1)
        * _fact(L + l1 - l2)
        * _fact(L - l1 + l2)
        * _fact(l1 + l2 - L)
        / _fact(l1 + l2 + L + 1)
    )
    # 这一部分包含所有 m 相关阶乘。
    pref_m = (
        _fact(L + M)
        * _fact(L - M)
        * _fact(l1 - m1)
        * _fact(l1 + m1)
        * _fact(l2 - m2)
        * _fact(l2 + m2)
    )
    # 两个正归一化因子的乘积取平方根。
    prefactor = math.sqrt(pref_triangle * pref_m)
    # total 累加 Racah 求和。
    total = 0.0
    # 这个有限上界足够覆盖所有合法 k，非法项会被下面的非负阶乘条件过滤。
    for k in range(l1 + l2 + L + 2):
        # 分母中六个阶乘的参数。
        args = (
            k,
            l1 + l2 - L - k,
            l1 - m1 - k,
            l2 + m2 - k,
            L - l2 + m1 + k,
            L - l1 - m2 + k,
        )
        # 只要有一个阶乘参数为负，这一项就不存在。
        if min(args) < 0:
            continue
        # denom 是六个阶乘的乘积。
        denom = 1
        # 逐项乘入分母。
        for a in args:
            denom *= _fact(a)
        # Racah 和具有交替符号 (-1)^k。
        total += ((-1) ** k) / denom
    # 返回标准 CG 系数。
    return prefactor * total
# 以下三个小函数用两个普通实数 (re,im) 模拟一次性的二维代数，只用于把标准 CG 常数换到实球谐基。
# 注意：这里没有创建任何复数 dtype，最终输出的 CG 张量也严格是实数。
def _pair_mul(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    # (a_re+i a_im)(b_re+i b_im) 的实部。
    re = a[0] * b[0] - a[1] * b[1]
    # 同一个乘积的虚部。
    im = a[0] * b[1] + a[1] * b[0]
    # 用两个 Python float 返回，不产生复数对象。
    return re, im
# 对二维代数元素做共轭，相当于把第二个分量变号。
def _pair_conj(a: Tuple[float, float]) -> Tuple[float, float]:
    # 实部不变，虚部取负。
    return a[0], -a[1]
# 返回实球谐基变换矩阵 S_l 的单个元素，用 (实部,虚部) 两个 float 表示。
def _real_basis_transform_entry(l: int, q: int, m: int) -> Tuple[float, float]:
    # q=0 的实球谐就是标准 m=0 球谐。
    if q == 0:
        return (1.0, 0.0) if m == 0 else (0.0, 0.0)
    # k=|q| 是成对出现的正磁量子数大小。
    k = abs(q)
    # inv_sqrt2 是成对基变换的归一化常数。
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    # q>0 对应 cosine 型实球谐：Y_real,+k=[(-1)^k Y_k + Y_-k]/sqrt(2)。
    if q > 0:
        if m == k:
            return ((-1) ** k * inv_sqrt2, 0.0)
        if m == -k:
            return (inv_sqrt2, 0.0)
        return (0.0, 0.0)
    # q<0 对应 sine 型实球谐；这里仍只用两个实数记录基变换系数。
    if m == k:
        return (0.0, -((-1) ** k) * inv_sqrt2)
    if m == -k:
        return (0.0, inv_sqrt2)
    return (0.0, 0.0)
# 缓存实球谐基中的 CG 常数数据；函数返回普通 Python tuple，真正 JAX dtype 在 cg_array 中指定。
@functools.lru_cache(maxsize=None)
def _real_cg_data(l1: int, l2: int, L: int):
    # 首先检查三角条件。
    if L < abs(l1 - l2) or L > l1 + l2:
        raise ValueError(f'非法 CG 路径 {l1} x {l2} -> {L}')
    # data_re 和 data_im 分别累积基变换后的两个二维代数分量。
    data_re = []
    data_im = []
    # 输出实基分量 Q=-L,...,+L。
    for Q in range(-L, L + 1):
        plane_re = []
        plane_im = []
        # 第一个输入实基分量 q1=-l1,...,+l1。
        for q1 in range(-l1, l1 + 1):
            row_re = []
            row_im = []
            # 第二个输入实基分量 q2=-l2,...,+l2。
            for q2 in range(-l2, l2 + 1):
                total_re = 0.0
                total_im = 0.0
                # 下面执行 C_real=S_L C_standard S_l1^{-1} S_l2^{-1}，但完全用两个实数做代数。
                for M in range(-L, L + 1):
                    # 输出基变换系数 S_L[Q,M]。
                    s_out = _real_basis_transform_entry(L, Q, M)
                    # 零元素直接跳过，减少循环量。
                    if s_out == (0.0, 0.0):
                        continue
                    for m1 in range(-l1, l1 + 1):
                        # S^{-1}=S^†，因此输入侧取共轭。
                        s1 = _pair_conj(_real_basis_transform_entry(l1, q1, m1))
                        if s1 == (0.0, 0.0):
                            continue
                        for m2 in range(-l2, l2 + 1):
                            # 第二个输入同样使用 S^†。
                            s2 = _pair_conj(_real_basis_transform_entry(l2, q2, m2))
                            if s2 == (0.0, 0.0):
                                continue
                            # 标准 CG 系数本身是普通实数。
                            c = clebsch_gordan_value(l1, m1, l2, m2, L, M)
                            if c == 0.0:
                                continue
                            # 依次乘三个基变换系数。
                            term = _pair_mul(s_out, s1)
                            term = _pair_mul(term, s2)
                            # 最后乘实数 CG 系数。
                            total_re += c * term[0]
                            total_im += c * term[1]
                row_re.append(total_re)
                row_im.append(total_im)
            plane_re.append(row_re)
            plane_im.append(row_im)
        data_re.append(plane_re)
        data_im.append(plane_im)
    # 计算两个分量的整体范数，用来判断当前唯一 intertwiner 在这个实基下落在“实方向”还是“虚方向”。
    norm_re = math.sqrt(sum(x * x for plane in data_re for row in plane for x in row))
    norm_im = math.sqrt(sum(x * x for plane in data_im for row in plane for x in row))
    # SO(3) 实表示之间的 intertwiner 总可以乘一个全局相位变成实张量；这里直接选择非零的那个实方向。
    if norm_re >= norm_im:
        # 只要理论正确，另一个分量应当只剩浮点舍入误差。
        if norm_im > 1.0e-10 * max(norm_re, 1.0):
            raise RuntimeError(f'实 CG 基变换出现异常混合：{l1}x{l2}->{L}, re={norm_re}, im={norm_im}')
        chosen = data_re
    else:
        # 对 1x1->1 等反对称通道，标准相位约定下会整体落在第二个分量；取这一分量等价于乘一个全局常数，不影响等变性。
        if norm_re > 1.0e-10 * max(norm_im, 1.0):
            raise RuntimeError(f'实 CG 基变换出现异常混合：{l1}x{l2}->{L}, re={norm_re}, im={norm_im}')
        chosen = data_im
    # 转成不可变 tuple，便于 lru_cache 保存。
    return tuple(tuple(tuple(float(x) for x in row) for row in plane) for plane in chosen)
# 返回实球谐基中的 CG 张量，shape=(2L+1,2l1+1,2l2+1)。
def cg_array(l1: int, l2: int, L: int, dtype=jnp.float32) -> Array:
    # dtype 由调用者传入，保证 float32 与 float64 不混用。
    return jnp.asarray(_real_cg_data(l1, l2, L), dtype=dtype)
# 枚举卷积中全部合法的 li x lf -> lo 路径。
def admissible_paths(lmax: int) -> Tuple[Tuple[int, int, int], ...]:
    # paths 保存所有三元组。
    paths = []
    # li 是输入粒子 feature 的角动量阶数。
    for li in range(lmax + 1):
        # lf 是几何滤波器的角动量阶数。
        for lf in range(lmax + 1):
            # lo 必须满足角动量三角条件，并限制在网络 lmax 内。
            for lo in range(abs(li - lf), min(li + lf, lmax) + 1):
                paths.append((li, lf, lo))
    # tuple 作为静态结构更适合 JAX 外层 Python 控制流。
    return tuple(paths)
# WBN 使用同样的角动量加法规则。
def bilinear_paths(lmax: int) -> Tuple[Tuple[int, int, int], ...]:
    # 直接复用 admissible_paths。
    return admissible_paths(lmax)
# 把电子和原子核坐标拼成统一粒子坐标。
def combine_particle_positions(pos: Array, atoms: Array, ndim: int = 3, dtype=None) -> Array:
    # 未显式给 dtype 时沿用输入 dtype。
    if dtype is None:
        dtype = jnp.result_type(pos, atoms)
    # 电子坐标整理成 (Ne,3)。
    epos = jnp.asarray(pos, dtype=dtype).reshape((-1, ndim))
    # 核坐标整理成 (Natom,3)。
    apos = jnp.asarray(atoms, dtype=dtype).reshape((-1, ndim))
    # 电子在前、原子核在后，这是后面 split_ea 的统一约定。
    return jnp.concatenate([epos, apos], axis=0)
# 计算所有有序粒子对的几何量。
def pair_geometry(positions: Array, eps: float = 1.0e-12):
    # n 是统一粒子数。
    n = positions.shape[0]
    # rel[i,j]=x_j-x_i，i 是接收粒子，j 是发送粒子。
    rel = positions[None, :, :] - positions[:, None, :]
    # nonself 在 i!=j 时为 1，在 i=j 时为 0。
    nonself = ~jnp.eye(n, dtype=bool)
    # 对角零向量没有方向，这里临时替换为 z 轴单位向量避免归一化 NaN。
    zhat = jnp.asarray([0.0, 0.0, 1.0], dtype=positions.dtype)
    # 非对角保留真实位移，对角使用安全方向。
    rel_safe = jnp.where(nonself[..., None], rel, zhat)
    # 安全位移长度。
    dist_safe = jnp.linalg.norm(rel_safe, axis=-1)
    # 单位方向。
    unit = rel_safe / jnp.maximum(dist_safe[..., None], jnp.asarray(eps, dtype=positions.dtype))
    # 真正径向距离；自相互作用位置严格置零。
    # 注意这里不能再次对原始 rel 做 norm；对角线 rel[i,i]=0，虽然前向值正常，但其坐标导数未定义，会污染 VMC kinetic-energy 的梯度/Hessian。
    # 直接复用上面已经用 zhat 替换对角线后的 dist_safe，再把对角线数值置回 0，可保证被 mask 掉的分支也没有 NaN 导数。
    dist = jnp.where(nonself, dist_safe, jnp.asarray(0.0, dtype=positions.dtype))
    if SO3_DEBUG_VERBOSE:
        _debug_array('pair_geometry/positions', positions)
        _debug_array('pair_geometry/dist_safe', dist_safe)
        _debug_array('pair_geometry/dist', dist)
        _debug_array('pair_geometry/unit', unit)
    # 返回相对位移、距离、方向和非自相互作用 mask。
    return rel, dist, unit, nonself
# 平滑余弦 cutoff。
def cutoff_envelope(r: Array, cutoff: Optional[float]) -> Array:
    # cutoff=None 表示全连接。
    if cutoff is None:
        return jnp.ones_like(r)
    # 统一把 cutoff 转成与 r 相同 dtype。
    c = jnp.asarray(cutoff, dtype=r.dtype)
    # 无量纲距离。
    x = r / c
    # cutoff 内平滑衰减，外部严格为零。
    return jnp.where(r < c, 0.5 * (jnp.cos(jnp.asarray(math.pi, dtype=r.dtype) * x) + 1.0), 0.0)
# 指数型径向基。
def radial_basis(r: Array, n_rbf: int, cutoff: Optional[float]) -> Array:
    # beta 在多个长度尺度之间对数均匀分布。
    betas = jnp.exp(jnp.linspace(jnp.log(jnp.asarray(0.35, dtype=r.dtype)), jnp.log(jnp.asarray(4.0, dtype=r.dtype)), n_rbf, dtype=r.dtype))
    # basis[...,n]=exp(-beta_n r)。
    basis = jnp.exp(-r[..., None] * betas)
    # 乘 cutoff 包络。
    return basis * cutoff_envelope(r, cutoff)[..., None]
# 计算带 Condon--Shortley 相位的缔合 Legendre 多项式 P_l^m(x)，只需要 m>=0。
def _associated_legendre(l: int, m: int, x: Array) -> Array:
    # 基本范围检查。
    if m < 0 or m > l:
        raise ValueError(f'需要 0<=m<=l，收到 l={l}, m={m}')
    # P_m^m(x)=(-1)^m(2m-1)!!(1-x^2)^(m/2)。
    pmm = jnp.ones_like(x)
    # m=0 时循环为空，pmm=1 正确。
    if m > 0:
        # 双阶乘 (2m-1)!! 用普通 Python 常数计算。
        df = 1
        for k in range(1, 2 * m, 2):
            df *= k
        # clip 防止单位向量舍入误差导致 1-x^2 略微为负。
        base = jnp.maximum(jnp.asarray(0.0, dtype=x.dtype), 1.0 - x * x)
        # Condon--Shortley 相位 (-1)^m 显式包含在这里。
        pmm = jnp.asarray(((-1) ** m) * df, dtype=x.dtype) * base ** (jnp.asarray(m / 2.0, dtype=x.dtype))
    # 如果 l=m，递推已经结束。
    if l == m:
        return pmm
    # P_{m+1}^m=x(2m+1)P_m^m。
    pmmp1 = x * jnp.asarray(2 * m + 1, dtype=x.dtype) * pmm
    # l=m+1 时直接返回。
    if l == m + 1:
        return pmmp1
    # 继续用三项递推获得更高 l。
    p_lm2 = pmm
    p_lm1 = pmmp1
    # ell 从 m+2 递增到目标 l。
    for ell in range(m + 2, l + 1):
        # 标准递推式。
        pell = (
            jnp.asarray(2 * ell - 1, dtype=x.dtype) * x * p_lm1
            - jnp.asarray(ell + m - 1, dtype=x.dtype) * p_lm2
        ) / jnp.asarray(ell - m, dtype=x.dtype)
        # 滚动保存前两阶。
        p_lm2, p_lm1 = p_lm1, pell
    # 返回 P_l^m。
    return p_lm1
# 计算完整实球谐 multiplet，最后一维按 q=-l,...,+l 排列。
def spherical_harmonics(l: int, unit: Array) -> Array:
    # 统一确保输入是实浮点数组。
    unit = jnp.asarray(unit)
    # 取 Cartesian 单位方向分量；对于本项目默认 lmax<=2，直接使用多项式形式可以避免 atan2(0,0) 和极点 sqrt(1-z^2) 的 AD 奇点。
    x = unit[..., 0]
    y = unit[..., 1]
    z = unit[..., 2]
    # l=0 只有 q=0，Y_00=1/sqrt(4*pi)。
    if l == 0:
        c0 = jnp.asarray(1.0 / math.sqrt(4.0 * math.pi), dtype=unit.dtype)
        return jnp.stack([jnp.ones_like(x) * c0], axis=-1)
    # l=1 的实球谐严格保持本文件原来的 q=-1,0,+1 顺序与相位约定。
    # 对应 [sqrt(3/4pi)*y, sqrt(3/4pi)*z, sqrt(3/4pi)*x]。
    if l == 1:
        c1 = jnp.asarray(math.sqrt(3.0 / (4.0 * math.pi)), dtype=unit.dtype)
        return jnp.stack([c1 * y, c1 * z, c1 * x], axis=-1)
    # l=2 同样使用与原 Legendre+phi 实现完全一致的 q=-2,-1,0,+1,+2 顺序。
    if l == 2:
        c20 = jnp.asarray(math.sqrt(5.0 / (16.0 * math.pi)), dtype=unit.dtype)
        c21 = jnp.asarray(math.sqrt(15.0 / (4.0 * math.pi)), dtype=unit.dtype)
        c22 = jnp.asarray(math.sqrt(15.0 / (16.0 * math.pi)), dtype=unit.dtype)
        return jnp.stack([
            2.0 * c22 * x * y,
            c21 * y * z,
            c20 * (3.0 * z * z - 1.0),
            c21 * x * z,
            c22 * (x * x - y * y),
        ], axis=-1)
    # l>2 仍保留通用 Legendre 实现；如果以后 VMC 需要 l>2，建议继续改成 Cartesian solid-harmonic 递推以完全避免极点 AD 奇点。
    zc = jnp.clip(z, -1.0, 1.0)
    phi = jnp.arctan2(y, x)
    values = []
    for q in range(-l, l + 1):
        m = abs(q)
        P = _associated_legendre(l, m, zc)
        norm = math.sqrt((2 * l + 1) / (4.0 * math.pi) * _fact(l - m) / _fact(l + m))
        norm = jnp.asarray(norm, dtype=unit.dtype)
        if q == 0:
            value = norm * P
        elif q > 0:
            value = jnp.asarray(math.sqrt(2.0) * ((-1) ** m), dtype=unit.dtype) * norm * P * jnp.cos(jnp.asarray(m, dtype=unit.dtype) * phi)
        else:
            value = jnp.asarray(math.sqrt(2.0) * ((-1) ** m), dtype=unit.dtype) * norm * P * jnp.sin(jnp.asarray(m, dtype=unit.dtype) * phi)
        values.append(value)
    return jnp.stack(values, axis=-1)
# 一次计算 l=0,...,lmax 的全部实球谐。
def all_spherical_harmonics(lmax: int, unit: Array) -> Dict[int, Array]:
    # 字典键直接用整数 l。
    return {l: spherical_harmonics(l, unit) for l in range(lmax + 1)}
# 统一实高斯初始化函数。
def _randn(key, shape, scale: float, dtype) -> Array:
    # jax.random.normal 直接用用户指定的 float32 或 float64。
    return jnp.asarray(scale, dtype=dtype) * jax.random.normal(key, shape, dtype=dtype)
# 初始化 SO(3) backbone 参数。
def init_so3_params(key, type_dim: int, config: SO3Config) -> Mapping[str, object]:
    # 解析并验证浮点类型。
    dtype = config_dtype(config)
    # C 是每个 l 的 channel 数。
    C = int(config.channels)
    # B 是 WBN bottleneck channel 数。
    B = int(config.wbn_bottleneck or C)
    # 参数树根字典。
    params = {}
    # 初始化 l=0 类型嵌入。
    key, k = jax.random.split(key)
    # shape=(C,type_dim)。
    params['init_w0'] = _randn(k, (C, type_dim), 1.0 / math.sqrt(max(type_dim, 1)), dtype)
    # 可选的 l=1 几何初始化。
    if config.init_l1 and config.lmax >= 1:
        # 类型权重和径向权重分开初始化。
        key, k1, k2 = jax.random.split(key, 3)
        # 发送粒子类型 -> C 个 l=1 channel 权重。
        params['init_l1_type'] = _randn(k1, (C, type_dim), 1.0 / math.sqrt(max(type_dim, 1)), dtype)
        # n_rbf -> C 个径向 channel 权重。
        params['init_l1_rbf'] = _randn(k2, (C, config.n_rbf), 1.0 / math.sqrt(config.n_rbf), dtype)
    # 初始化等变卷积层。
    conv_layers = []
    # 所有合法角动量路径只需枚举一次。
    paths = admissible_paths(config.lmax)
    # 逐层创建参数。
    for _ in range(config.n_layers):
        # self 保存同 l 线性项，paths 保存几何消息路径。
        layer = {'self': {}, 'paths': {}}
        # 每个 l 有一个 CxC channel 线性变换。
        for l in range(config.lmax + 1):
            key, k = jax.random.split(key)
            layer['self'][str(l)] = _randn(k, (C, C), 1.0 / math.sqrt(C), dtype)
        # 每个 li x lf -> lo 路径有一组径向 channel 核。
        for li, lf, lo in paths:
            key, k = jax.random.split(key)
            layer['paths'][f'{li}_{lf}_{lo}'] = _randn(k, (config.n_rbf, C, C), 1.0 / math.sqrt(config.n_rbf * C), dtype)
        # 保存当前层。
        conv_layers.append(layer)
    # tuple 固定网络层结构。
    params['conv_layers'] = tuple(conv_layers)
    # 初始化 WBN。
    wbn = []
    # WBN 合法路径。
    bpaths = bilinear_paths(config.lmax)
    # 逐个 block 初始化。
    for _ in range(config.wbn_layers):
        # 每个 block 同样包含 self 与 paths。
        block = {'self': {}, 'paths': {}}
        # 同 l 的线性分支。
        for l in range(config.lmax + 1):
            key, k = jax.random.split(key)
            block['self'][str(l)] = _randn(k, (C, C), 1.0 / math.sqrt(C), dtype)
        # bilinear 路径有 left/right/out 三个 channel 映射。
        for l1, l2, lo in bpaths:
            key, kl, kr, ko = jax.random.split(key, 4)
            block['paths'][f'{l1}_{l2}_{lo}'] = {
                'left': _randn(kl, (B, C), 1.0 / math.sqrt(C), dtype),
                'right': _randn(kr, (B, C), 1.0 / math.sqrt(C), dtype),
                'out': _randn(ko, (C, B), 1.0 / math.sqrt(B), dtype),
            }
        # 保存 block。
        wbn.append(block)
    # 固定结构。
    params['wbn'] = tuple(wbn)
    # readout 只收缩 channel，不作用于 irrep 分量 q。
    params['readout'] = {}
    # 每个 l 有独立 readout 向量。
    for l in range(config.lmax + 1):
        key, k = jax.random.split(key)
        params['readout'][str(l)] = _randn(k, (C,), 1.0 / math.sqrt(C), dtype)
    # 新结构不再把 A^(l) 投影成单个 q_l；每个 A_q^(l) 都直接拥有自己的一组 K 个 determinant，因此不再需要 state_coeff。
    # 返回参数树。
    return params
# 初始化每个粒子的全部 irrep feature。
def initialize_irreps(params: Mapping[str, object], positions: Array, type_features: Array, config: SO3Config) -> IrrepFeatures:
    # 使用配置中的统一实 dtype。
    dtype = config_dtype(config)
    # 输入坐标和类型 feature 都强制转换为同一 dtype。
    positions = jnp.asarray(positions, dtype=dtype)
    type_features = jnp.asarray(type_features, dtype=dtype)
    # C 是 channel 数。
    C = config.channels
    # n 是统一粒子总数。
    n = positions.shape[0]
    # 为 l=0,...,lmax 建立全零实张量。
    h: IrrepFeatures = {l: jnp.zeros((n, C, 2 * l + 1), dtype=dtype) for l in range(config.lmax + 1)}
    # 类型特征线性映射得到 l=0 scalar channels。
    h0 = jnp.einsum('ct,nt->nc', params['init_w0'], type_features)
    # l=0 只有一个分量，所以补长度 1 的最后一维。
    h[0] = h0[..., None]
    _debug_array('init/positions', positions)
    _debug_array('init/type_features', type_features)
    _debug_array('init/h_l0', h[0])
    # 显式初始化 l=1 几何特征。
    if config.init_l1 and config.lmax >= 1:
        # 粒子对几何。
        _, dist, unit, nonself = pair_geometry(positions, config.eps)
        # 径向基。
        rbf = radial_basis(dist, config.n_rbf, config.cutoff)
        # 实 l=1 球谐，shape=(n,n,3)。
        y1 = spherical_harmonics(1, unit)
        # 发送粒子类型权重，shape=(n,C)。
        type_weight = jnp.einsum('ct,nt->nc', params['init_l1_type'], type_features)
        # 粒子对径向权重，shape=(n,n,C)。
        radial_weight = jnp.einsum('cr,ijr->ijc', params['init_l1_rbf'], rbf)
        # 每条 i<-j 边的标量 channel 权重。
        pair_weight = radial_weight * type_weight[None, :, :]
        # 去掉 i=j 自相互作用。
        pair_weight = pair_weight * nonself[..., None]
        # 对发送粒子 j 求和，得到每个中心粒子的 l=1 feature。
        h[1] = jnp.einsum('ijc,ijq->icq', pair_weight, y1)
        if SO3_DEBUG_VERBOSE:
            _debug_array('init/rbf', rbf)
            _debug_array('init/Y_l1', y1)
            _debug_array('init/type_weight', type_weight)
            _debug_array('init/radial_weight', radial_weight)
            _debug_array('init/pair_weight', pair_weight)
        _debug_array('init/h_l1', h[1])
    # 其它更高 l 此时通常仍为 0，也打印一次用于确认。
    for l in range(2, config.lmax + 1):
        _debug_array(f'init/h_l{l}', h[l])
    # 返回初始 irreps。
    return h
# 一层 SO(3) 等变卷积。
def equivariant_convolution_layer(h: IrrepFeatures, positions: Array, layer_params: Mapping[str, object], config: SO3Config) -> IrrepFeatures:
    # 坐标统一 dtype。
    positions = jnp.asarray(positions, dtype=config_dtype(config))
    # 重新计算当前构型下的粒子对几何。
    _, dist, unit, nonself = pair_geometry(positions, config.eps)
    # 径向基。
    rbf = radial_basis(dist, config.n_rbf, config.cutoff)
    # 全部实球谐滤波器。
    Y = all_spherical_harmonics(config.lmax, unit)
    if SO3_DEBUG_VERBOSE:
        _debug_array('conv/dist', dist)
        _debug_array('conv/unit', unit)
        _debug_array('conv/rbf', rbf)
        for lf in range(config.lmax + 1):
            _debug_array(f'conv/Y_l{lf}', Y[lf])
    # out 保存每个输出 lo 的更新。
    out = {}
    # 先加 self interaction；只混 channel，不混 irrep 分量。
    for lo in range(config.lmax + 1):
        out[lo] = jnp.einsum('oc,icq->ioq', layer_params['self'][str(lo)], h[lo])
    # 遍历所有 CG 消息路径。
    for li, lf, lo in admissible_paths(config.lmax):
        # 路径参数 shape=(n_rbf,Cout,Cin)。
        W = layer_params['paths'][f'{li}_{lf}_{lo}']
        # 粒子对相关 channel 矩阵 shape=(n,n,Cout,Cin)。
        radial_matrix = jnp.einsum('ijr,roc->ijoc', rbf, W)
        # i=j 路径置零。
        radial_matrix = radial_matrix * nonself[..., None, None]
        # 实 CG 张量 shape=(2lo+1,2li+1,2lf+1)。
        C = cg_array(li, lf, lo, dtype=h[li].dtype)
        # 完成径向矩阵、发送粒子 feature、实球谐与 CG 的联合收缩，并对 j 求和。
        msg = jnp.einsum('ijoc,jcm,ijn,pmn->iop', radial_matrix, h[li], Y[lf], C, optimize=True)
        if SO3_DEBUG_VERBOSE:
            _debug_array(f'conv/path_{li}x{lf}_to_{lo}/W', W)
            _debug_array(f'conv/path_{li}x{lf}_to_{lo}/radial_matrix', radial_matrix)
            _debug_array(f'conv/path_{li}x{lf}_to_{lo}/CG', C)
            _debug_array(f'conv/path_{li}x{lf}_to_{lo}/msg', msg)
        # 累加到对应输出 sector。
        out[lo] = out[lo] + msg
    # 同一 l 的量相加仍属于同一 irrep；sqrt(2) 控制方差。
    root2 = jnp.asarray(math.sqrt(2.0), dtype=h[0].dtype)
    result = {l: (h[l] + out[l]) / root2 for l in range(config.lmax + 1)}
    for l in range(config.lmax + 1):
        _debug_array(f'conv/output_l{l}', result[l])
    return result
# Wigner bilinear block，在每个粒子内部做两组 irrep 的 CG 双线性耦合。
def wigner_bilinear_block(h: IrrepFeatures, block_params: Mapping[str, object], config: SO3Config) -> IrrepFeatures:
    # out 初始化为各 l 的线性 self 分支。
    out = {lo: jnp.einsum('oc,icq->ioq', block_params['self'][str(lo)], h[lo]) for lo in range(config.lmax + 1)}
    # 遍历所有合法 l1 x l2 -> lo 路径。
    for l1, l2, lo in bilinear_paths(config.lmax):
        # 当前路径的 channel 投影参数。
        p = block_params['paths'][f'{l1}_{l2}_{lo}']
        # 左输入投影到 bottleneck channel。
        left = jnp.einsum('bc,icm->ibm', p['left'], h[l1])
        # 右输入投影到同一 bottleneck channel。
        right = jnp.einsum('bc,icm->ibm', p['right'], h[l2])
        # 实 CG 张量。
        C = cg_array(l1, l2, lo, dtype=left.dtype)
        # 在同一粒子、同一 bottleneck channel 内耦合角动量分量。
        coupled = jnp.einsum('ibm,ibn,pmn->ibp', left, right, C, optimize=True)
        projected = jnp.einsum('ob,ibm->iom', p['out'], coupled)
        if SO3_DEBUG_VERBOSE:
            _debug_array(f'wbn/path_{l1}x{l2}_to_{lo}/left', left)
            _debug_array(f'wbn/path_{l1}x{l2}_to_{lo}/right', right)
            _debug_array(f'wbn/path_{l1}x{l2}_to_{lo}/CG', C)
            _debug_array(f'wbn/path_{l1}x{l2}_to_{lo}/coupled', coupled)
            _debug_array(f'wbn/path_{l1}x{l2}_to_{lo}/projected', projected)
        # 输出投影回 C 个 channel。
        out[lo] = out[lo] + projected
    # 残差归一化。
    root2 = jnp.asarray(math.sqrt(2.0), dtype=h[0].dtype)
    result = {l: (h[l] + out[l]) / root2 for l in range(config.lmax + 1)}
    for l in range(config.lmax + 1):
        _debug_array(f'wbn/output_l{l}', result[l])
    return result
# 从最终粒子 feature 得到 y_j,c^(l)、A_channels^(l) 与最终单个 multiplet A^(l)。
def readout_particle_features(h: IrrepFeatures, params: Mapping[str, object], config: SO3Config):
    # y 就是最终逐粒子 feature，不改变 channel 与 irrep 分量。
    y = {l: h[l] for l in range(config.lmax + 1)}
    # A_channels 只对粒子 j 求和，仍保留 channel c。
    A_channels = {}
    # A 再对 channel 做可学习 readout，得到长度 2l+1 的最终 multiplet。
    A = {}
    # 逐个 l 处理。
    for l in range(config.lmax + 1):
        # A_channels[c,q]=sum_j y[j,c,q]。
        A_channels[l] = jnp.sum(y[l], axis=0)
        # w[c] 只收缩 channel，不碰 q。
        w = params['readout'][str(l)]
        # A[q]=sum_c w[c] A_channels[c,q]。
        A[l] = jnp.einsum('c,cq->q', w, A_channels[l])
        _debug_array(f'readout/y_l{l}', y[l])
        _debug_array(f'readout/A_channels_l{l}', A_channels[l])
        _debug_array(f'readout/A_l{l}', A[l])
    # 返回三个层级的结果。
    return y, A_channels, A
# 完整执行 SO(3) backbone。
def apply_so3_network(params: Mapping[str, object], positions: Array, type_features: Array, config: SO3Config):
    # 统一 dtype。
    dtype = config_dtype(config)
    # 坐标强制转换到配置 dtype。
    positions = jnp.asarray(positions, dtype=dtype)
    # 类型特征同样转换。
    type_features = jnp.asarray(type_features, dtype=dtype)
    # 每个粒子必须恰好有一行 type feature。
    if positions.shape[0] != type_features.shape[0]:
        raise ValueError('type_features 必须为每个电子和原子核提供一行特征。')
    _debug_array('network/input_positions', positions)
    _debug_array('network/input_type_features', type_features)
    # 初始化各阶 feature。
    h = initialize_irreps(params, positions, type_features, config)
    # 多层等变卷积。
    for layer_index, layer_params in enumerate(params['conv_layers']):
        if SO3_DEBUG:
            jax.debug.print(f'[SO3 DEBUG] ===== conv layer {layer_index} =====')
        h = equivariant_convolution_layer(h, positions, layer_params, config)
        for l in range(config.lmax + 1):
            _debug_array(f'network/after_conv_{layer_index}/h_l{l}', h[l])
    # 多个 WBN。
    for block_index, block_params in enumerate(params['wbn']):
        if SO3_DEBUG:
            jax.debug.print(f'[SO3 DEBUG] ===== WBN block {block_index} =====')
        h = wigner_bilinear_block(h, block_params, config)
        for l in range(config.lmax + 1):
            _debug_array(f'network/after_wbn_{block_index}/h_l{l}', h[l])
    # 最终 readout。
    y, A_channels, A = readout_particle_features(h, params, config)
    # 返回所有需要的中间量。
    return {'h': h, 'y': y, 'A_channels': A_channels, 'A': A}
# 返回实际进入最终波函数的 sector 列表；l=0 自动放在第一项。
def sector_list(config: SO3Config) -> Tuple[int, ...]:
    # 去重并保持用户 target_ls 的原始顺序。
    seen = {0}
    sectors = [0]
    # 遍历所有目标 l。
    for l in config.target_ls:
        # 只接受 1<=l<=lmax。
        if int(l) < 1 or int(l) > config.lmax:
            raise ValueError(f'target l={l} 不在 1..lmax={config.lmax} 范围内。')
        # 避免重复 sector。
        if int(l) not in seen:
            sectors.append(int(l))
            seen.add(int(l))
    # 返回 tuple。
    return tuple(sectors)
# ============================================================================
# 每个 A^(l) 分量各自对应 K 个 determinant 的新波函数结构。
# 对 l=0：只有 q=0，直接把 K 个 determinant 求和。
# 对 l>0：每个 q=-l,...,+l 都有独立的 K 个 determinant，先对 k 求和，再乘对应 A_q^(l)，最后对全部 (l,q) 求和。
# 最终还在所有 determinant orbital 上乘 FermiNet 风格的原子核指数 envelope，并在总波函数上乘电子-电子 Kato Jastrow。
# ============================================================================

# 把正数 value 转成 softplus 的 unconstrained 参数初值，使 softplus(raw)=value。
def _inverse_softplus(value: float, dtype) -> Array:
    # 数值上要求 value>0；这里同时避免 expm1(0)。
    value = max(float(value), 1.0e-8)
    return jnp.asarray(math.log(math.expm1(value)), dtype=dtype)

# 初始化单个 (l,q,spin) orbital head 对应的 FermiNet 风格 envelope 参数。
def _init_orbital_envelope_params(
    key,
    determinants_per_l: int,
    norb: int,
    natom: int,
    ndim: int,
    sigma_init: float,
    pi_init: float,
    dtype,
):
    # K 是当前 A_q 分量对应的 determinant 数。
    K = int(determinants_per_l)
    # sigma 的 shape=(K,norb,natom,ndim,ndim)，与 FermiNet 公式中的 Sigma_{im}^{k alpha} 一一对应。
    eye = jnp.eye(ndim, dtype=dtype)
    sigma = jnp.broadcast_to(
        jnp.asarray(sigma_init, dtype=dtype) * eye,
        (K, norb, natom, ndim, ndim),
    )
    # 给 Sigma 一个很小的随机扰动，避免所有 determinant/orbital 的 envelope 在初始化时完全相同。
    key, k_sigma = jax.random.split(key)
    sigma = sigma + _randn(k_sigma, sigma.shape, 1.0e-3, dtype)
    # pi 的 shape=(K,norb,natom)；初始让所有原子核等权贡献，总尺度约为 pi_init。
    pi = jnp.full(
        (K, norb, natom),
        jnp.asarray(pi_init / max(natom, 1), dtype=dtype),
        dtype=dtype,
    )
    # pi 可以取正也可以取负，与标准 FermiNet envelope 一样保持为普通可学习参数。
    return {'sigma': sigma, 'pi': pi}

# 初始化“每个 A_q^(l) 都有 K 个 determinant”的全部 orbital head 与 envelope 参数。
def init_sector_determinant_params(
    key,
    channels: int,
    nspins: Sequence[int],
    determinants_per_l: int,
    sectors: Sequence[int],
    natom: int,
    ndim: int = 3,
    full_det: bool = True,
    include_bias: bool = False,
    float_dtype: Any = jnp.float32,
    use_envelope: bool = True,
    envelope_sigma_init: float = 1.0,
    envelope_pi_init: float = 1.0,
):
    # 解析统一实 dtype。
    dtype = resolve_float_dtype(float_dtype)
    # K 是“每个 A 分量”的 determinant 数，而不是每个 l 的总 determinant 数。
    K = int(determinants_per_l)
    if K <= 0:
        raise ValueError('determinants_per_l 必须大于 0。')
    # active_spins 只保留非空自旋通道。
    active_spins = tuple(int(n) for n in nspins if int(n) > 0)
    # 总电子数。
    nelec = int(sum(active_spins))
    if nelec <= 0:
        raise ValueError('至少需要一个电子才能构造 Slater determinant。')
    if int(natom) <= 0:
        raise ValueError('SO(3) envelope 至少需要一个原子核。')
    # heads[l][q_index][spin_index] 保存线性 orbital head。
    params = {'heads': {}, 'envelope': {}}
    # 逐 l 建立参数；l 对应 2*l+1 个 A 分量。
    for l in sectors:
        l = int(l)
        nq = 2 * l + 1
        q_heads = []
        q_envelopes = []
        # q_index=0,...,2l 对应物理标签 q=-l,...,+l。
        for q_index in range(nq):
            spin_heads = []
            spin_envelopes = []
            # 不同 spin row channel 仍然使用独立 orbital head；full_det 时每一行都输出 Ne 个轨道列。
            for nspin in active_spins:
                norb = nelec if full_det else nspin
                out_dim = K * norb
                key, k_head = jax.random.split(key)
                # 每个 (l,q,spin) 都有完全独立的 K*norb 输出，因此 l=1 总共有 3K 个 determinant。
                w = _randn(k_head, (channels, out_dim), 1.0 / math.sqrt(max(channels, 1)), dtype)
                head = {'w': w}
                if include_bias:
                    head['b'] = jnp.zeros((out_dim,), dtype=dtype)
                spin_heads.append(head)
                # envelope 与 orbital head 同样按 (l,q,spin) 独立参数化。
                if use_envelope:
                    key, k_env = jax.random.split(key)
                    env = _init_orbital_envelope_params(
                        k_env,
                        determinants_per_l=K,
                        norb=norb,
                        natom=int(natom),
                        ndim=int(ndim),
                        sigma_init=envelope_sigma_init,
                        pi_init=envelope_pi_init,
                        dtype=dtype,
                    )
                else:
                    env = None
                spin_envelopes.append(env)
            q_heads.append(tuple(spin_heads))
            q_envelopes.append(tuple(spin_envelopes))
        params['heads'][str(l)] = tuple(q_heads)
        params['envelope'][str(l)] = tuple(q_envelopes)
    # 这里故意不再创建 alpha：用户要求同一 A_q 下 K 个 determinant 直接等权求和。
    return params

# 应用某一个实线性 orbital head。
def _apply_orbital_head(x: Array, head: Mapping[str, Array]) -> Array:
    # 读取实权重并统一 dtype。
    w = head['w']
    x = jnp.asarray(x, dtype=w.dtype)
    _debug_array('orbital_head/input_x', x)
    if SO3_DEBUG_VERBOSE:
        _debug_array('orbital_head/weight', w)
    # 输出 shape=(nspin,K*norb)。
    out = x @ w
    if 'b' in head:
        out = out + head['b']
    _debug_array('orbital_head/output', out)
    return out

# 应用图中 FermiNet 风格的原子核指数 envelope：
#   e_{j i}^{k alpha} = sum_m pi_{i m}^{k alpha} exp(-| Sigma_{i m}^{k alpha} (r_j^alpha-R_m) |)。
def apply_orbital_envelope(
    electron_pos: Array,
    atoms: Array,
    envelope_params: Optional[Mapping[str, Array]],
    eps: float = 1.0e-12,
) -> Array:
    # 如果禁用 envelope，调用者会直接跳过；这里仍允许 None 方便接口统一。
    if envelope_params is None:
        raise ValueError('apply_orbital_envelope 收到 None；禁用 envelope 时不应调用该函数。')
    # pi shape=(K,norb,natom)。
    pi = envelope_params['pi']
    # sigma shape=(K,norb,natom,3,3)。
    sigma = envelope_params['sigma']
    dtype = sigma.dtype
    electron_pos = jnp.asarray(electron_pos, dtype=dtype)
    atoms = jnp.asarray(atoms, dtype=dtype)
    # diff[e,m,d]=r_e-R_m，shape=(nspin,natom,3)。
    diff = electron_pos[:, None, :] - atoms[None, :, :]
    # transformed[e,k,o,m,a]=sum_b Sigma[k,o,m,a,b]*diff[e,m,b]。
    transformed = jnp.einsum('komab,emb->ekoma', sigma, diff, optimize=True)
    # norm shape=(nspin,K,norb,natom)。加 eps^2 仅用于避免电子恰落在原子核时 AD 产生 0/0。
    eps_value = jnp.asarray(eps, dtype=dtype)
    radius = jnp.sqrt(jnp.sum(transformed * transformed, axis=-1) + eps_value * eps_value)
    # 对每个原子核求加权指数衰减并求和，得到 shape=(nspin,K,norb)。
    env = jnp.sum(pi[None, ...] * jnp.exp(-radius), axis=-1)
    if SO3_DEBUG_VERBOSE:
        _debug_array('envelope/pi', pi)
        _debug_array('envelope/sigma', sigma)
        _debug_array('envelope/radius', radius)
    _debug_array('envelope/value', env)
    return env

# 从电子 y_i^(0) 为每个 (l,q) 构造 K 个 Slater orbital matrix，并在每个 orbital element 上乘 envelope。
def build_sector_orbitals(
    y0_electrons: Array,
    electron_pos: Array,
    atoms: Array,
    det_params: Mapping[str, object],
    nspins: Sequence[int],
    determinants_per_l: int,
    sectors: Sequence[int],
    full_det: bool = True,
    eps: float = 1.0e-12,
):
    # K 是每个 A_q 分量的 determinant 数。
    K = int(determinants_per_l)
    active_spins = tuple(int(n) for n in nspins if int(n) > 0)
    nelec = int(sum(active_spins))
    y0_electrons = jnp.asarray(y0_electrons)
    electron_pos = jnp.asarray(electron_pos, dtype=y0_electrons.dtype).reshape((nelec, -1))
    atoms = jnp.asarray(atoms, dtype=y0_electrons.dtype)
    _debug_array('det/y0_electrons', y0_electrons)
    _debug_array('det/electron_pos', electron_pos)
    if y0_electrons.shape[0] != nelec:
        raise ValueError(f'y0_electrons 第一维应为 {nelec}，实际为 {y0_electrons.shape[0]}。')
    # 计算自旋 split 边界，并同时切分 feature 与电子坐标。
    split_points = []
    running = 0
    for nspin in active_spins[:-1]:
        running += nspin
        split_points.append(running)
    y_by_spin = jnp.split(y0_electrons, split_points, axis=0)
    pos_by_spin = jnp.split(electron_pos, split_points, axis=0)
    result = {}
    # 每个 l 返回长度 nq=2l+1 的 tuple；每个元素就是该 A_q 对应的 K 个 Slater matrix。
    for l in sectors:
        l = int(l)
        nq = 2 * l + 1
        q_results = []
        for q_index in range(nq):
            q_label = q_index - l
            heads = det_params['heads'][str(l)][q_index]
            envs = det_params['envelope'][str(l)][q_index]
            spin_mats = []
            for x, epos, head, env_params, nspin in zip(y_by_spin, pos_by_spin, heads, envs, active_spins):
                raw = _apply_orbital_head(x, head)
                norb = nelec if full_det else nspin
                # mat 先保持 (nspin,K,norb)，便于 envelope 按电子行逐元素相乘。
                mat = raw.reshape(nspin, K, norb)
                if env_params is not None:
                    env = apply_orbital_envelope(epos, atoms, env_params, eps=eps)
                    mat = mat * env
                # determinant batch 需要 (K,nspin,norb)。
                mat = jnp.transpose(mat, (1, 0, 2))
                spin_mats.append(mat)
            if full_det:
                q_mat = jnp.concatenate(spin_mats, axis=1)
                _debug_array(f'det/orbitals_l{l}_q{q_label}', q_mat)
            else:
                q_mat = tuple(spin_mats)
                for iblock, block in enumerate(q_mat):
                    _debug_array(f'det/orbitals_l{l}_q{q_label}_spinblock{iblock}', block)
            q_results.append(q_mat)
        result[l] = tuple(q_results)
    return result

# 对每个 (l,q,k) 计算 Slater determinant 的 sign 与 log|det|。
def sector_determinants_from_y0(
    y0_electrons: Array,
    electron_pos: Array,
    atoms: Array,
    det_params: Mapping[str, object],
    nspins: Sequence[int],
    determinants_per_l: int,
    sectors: Sequence[int],
    full_det: bool = True,
    eps: float = 1.0e-12,
):
    orbitals = build_sector_orbitals(
        y0_electrons,
        electron_pos,
        atoms,
        det_params,
        nspins,
        determinants_per_l,
        sectors,
        full_det=full_det,
        eps=eps,
    )
    signs = {}
    logabs = {}
    K = int(determinants_per_l)
    # signs[l] 与 logabs[l] 的 shape 都是 (2l+1,K)。
    for l in sectors:
        l = int(l)
        q_signs = []
        q_logs = []
        for q_index, q_orbitals in enumerate(orbitals[l]):
            q_label = q_index - l
            if full_det:
                sign_q, log_q = jnp.linalg.slogdet(q_orbitals)
            else:
                sign_q = jnp.ones((K,), dtype=y0_electrons.dtype)
                log_q = jnp.zeros((K,), dtype=y0_electrons.dtype)
                for block in q_orbitals:
                    p, q = jnp.linalg.slogdet(block)
                    sign_q = sign_q * p
                    log_q = log_q + q
            _debug_array(f'det/slogdet_sign_l{l}_q{q_label}', sign_q)
            _debug_array(f'det/slogdet_logabs_l{l}_q{q_label}', log_q)
            q_signs.append(sign_q)
            q_logs.append(log_q)
        signs[l] = jnp.stack(q_signs, axis=0)
        logabs[l] = jnp.stack(q_logs, axis=0)
    return signs, logabs, orbitals

# 初始化电子-电子 Kato cusp Jastrow 的可学习 Padé range 参数。
def init_kato_jastrow_params(
    key,
    nspins: Sequence[int],
    b_init: float = 1.0,
    float_dtype: Any = jnp.float32,
):
    # key 当前保留在接口中，便于以后扩展更多可学习 Jastrow 参数；当前初始化是确定性的。
    del key
    dtype = resolve_float_dtype(float_dtype)
    if sum(int(x) for x in nspins) <= 0:
        raise ValueError('Kato Jastrow 至少需要一个电子。')
    # b 必须为正，因此实际前向使用 softplus(raw_b)；这里反解 raw_b 使初始 b 恰为 b_init。
    raw = _inverse_softplus(b_init, dtype)
    return {
        'raw_b_same': raw,
        'raw_b_opposite': raw,
    }

# 计算满足 3D 电子-电子 Kato cusp 的 Padé Jastrow 对数 J。
# 波函数最终乘 exp(J)，其中同自旋 cusp 斜率固定为 1/4，异自旋固定为 1/2。
def kato_ee_jastrow(
    electron_pos: Array,
    nspins: Sequence[int],
    params: Mapping[str, Array],
    eps: float = 1.0e-12,
) -> Array:
    electron_pos = jnp.asarray(electron_pos)
    nelec = electron_pos.shape[0]
    dtype = electron_pos.dtype
    # 正的可学习 range 参数 b；softplus 保证优化过程中不会穿过负值导致分母奇异。
    b_same = jax.nn.softplus(jnp.asarray(params['raw_b_same'], dtype=dtype))
    b_opp = jax.nn.softplus(jnp.asarray(params['raw_b_opposite'], dtype=dtype))
    # 根据 nspins 构造每个电子的 spin-channel 标签；例如 (Nup,Ndn)->[0,...,0,1,...,1]。
    labels = []
    for ispin, nspin in enumerate(nspins):
        labels.extend([ispin] * int(nspin))
    spin_label = jnp.asarray(labels, dtype=jnp.int32)
    if spin_label.shape[0] != nelec:
        raise ValueError(f'nspins 总电子数 {spin_label.shape[0]} 与 electron_pos 第一维 {nelec} 不一致。')
    # 只取 i<j，完全避免 ee 距离矩阵对角线 r_ii=0 的自动微分奇点。
    ii, jj = jnp.triu_indices(nelec, k=1)
    diff = electron_pos[ii] - electron_pos[jj]
    eps_value = jnp.asarray(eps, dtype=dtype)
    rij = jnp.sqrt(jnp.sum(diff * diff, axis=-1) + eps_value * eps_value)
    same = spin_label[ii] == spin_label[jj]
    # 3D Coulomb Kato 条件：same-spin a=1/4，opposite-spin a=1/2。
    a = jnp.where(
        same,
        jnp.asarray(0.25, dtype=dtype),
        jnp.asarray(0.50, dtype=dtype),
    )
    b = jnp.where(same, b_same, b_opp)
    # u_ij(r)=a*r/(1+b*r)，因此 u'_ij(0)=a，严格固定 cusp；b 只控制有限距离形状。
    u = a * rij / (1.0 + b * rij)
    J = jnp.sum(u)
    _debug_scalar('jastrow/b_same', b_same)
    _debug_scalar('jastrow/b_opposite', b_opp)
    _debug_scalar('jastrow/J', J)
    return J

# 稳定计算一批实数项的和，每项用 sign_i 与 log|term_i| 表示。
def real_signed_logsumexp(signs: Array, logabs: Array):
    _debug_array('logsumexp/input_signs', signs)
    _debug_array('logsumexp/input_logabs', logabs)
    finite = jnp.isfinite(logabs)
    safe_logs = jnp.where(finite, logabs, -jnp.inf)
    maxlog = jnp.max(safe_logs)
    base = jnp.where(jnp.isfinite(maxlog), maxlog, jnp.asarray(0.0, dtype=logabs.dtype))
    scaled = jnp.sum(jnp.where(finite, signs * jnp.exp(logabs - base), jnp.asarray(0.0, dtype=logabs.dtype)))
    _debug_scalar('logsumexp/maxlog', maxlog)
    _debug_scalar('logsumexp/base', base)
    _debug_scalar('logsumexp/scaled', scaled)
    final_sign = jnp.sign(scaled)
    mag = jnp.abs(scaled)
    final_logabs = jnp.where(mag > 0, base + jnp.log(mag), -jnp.inf)
    _debug_scalar('logsumexp/final_sign', final_sign)
    _debug_scalar('logsumexp/final_logabs', final_logabs)
    return final_sign, final_logabs

# 按用户指定的新公式组装最终实波函数：
#   Psi_0 = sum_k D_{0,0,k}
#   Psi_l = sum_q A_q^(l) [sum_k D_{l,q,k}],  l>0
#   Psi   = exp(J) [Psi_0 + sum_{l>0} Psi_l].
def assemble_sector_wavefunction(
    det_signs: Mapping[int, Array],
    det_logabs: Mapping[int, Array],
    A: Mapping[int, Array],
    sectors: Sequence[int],
    jastrow_log: Optional[Array] = None,
    eps: float = 1.0e-30,
):
    # eps 仅用于兼容旧调用签名；signed-log 求和本身不需要用 eps 修改数学结果。
    del eps
    component_signs = []
    component_logs = []
    # 外层先遍历 l，再遍历该 irrep 的每个 q 分量。
    for l in sectors:
        l = int(l)
        nq = 2 * l + 1
        if det_signs[l].shape[0] != nq or det_logabs[l].shape[0] != nq:
            raise ValueError(f'l={l} 应有 {nq} 个 q 分量的 determinant。')
        for q_index in range(nq):
            q_label = q_index - l
            # 第一步：当前 A_q 对应的 K 个 determinant 先等权求和。
            det_sum_sign, det_sum_log = real_signed_logsumexp(
                det_signs[l][q_index],
                det_logabs[l][q_index],
            )
            # l=0 按用户要求不乘 A^(0)，直接使用 K 个 determinant 的和。
            if l == 0:
                aq = jnp.asarray(1.0, dtype=det_sum_log.dtype)
            else:
                aq = jnp.asarray(A[l][q_index], dtype=det_sum_log.dtype)
            aq_abs = jnp.abs(aq)
            aq_sign = jnp.sign(aq)
            term_sign = det_sum_sign * aq_sign
            term_log = jnp.where(
                (aq_abs > 0) & jnp.isfinite(det_sum_log),
                det_sum_log + jnp.log(aq_abs),
                -jnp.inf,
            )
            _debug_scalar(f'assemble/A_l{l}_q{q_label}', aq)
            _debug_scalar(f'assemble/detsum_sign_l{l}_q{q_label}', det_sum_sign)
            _debug_scalar(f'assemble/detsum_log_l{l}_q{q_label}', det_sum_log)
            _debug_scalar(f'assemble/term_log_l{l}_q{q_label}', term_log)
            component_signs.append(term_sign)
            component_logs.append(term_log)
    # 第二步：把所有 (l,q) 乘积项直接求和。
    signs = jnp.stack(component_signs, axis=0)
    logs = jnp.stack(component_logs, axis=0)
    _debug_array('assemble/component_signs', signs)
    _debug_array('assemble/component_logs', logs)
    final_sign, final_logabs = real_signed_logsumexp(signs, logs)
    # 第三步：全局 Kato Jastrow 乘 exp(J)，在 log|Psi| 上等价于直接加 J；它不会改变波函数符号。
    if jastrow_log is not None:
        final_logabs = final_logabs + jnp.asarray(jastrow_log, dtype=final_logabs.dtype)
        _debug_scalar('assemble/jastrow_log', jastrow_log)
        _debug_scalar('assemble/final_logabs_with_jastrow', final_logabs)
    return final_sign, final_logabs

# 返回每个 (l,q,k) 原始项的归一化 log 模长，主要用于 det_fn/诊断；Jastrow 是所有项共享因子，因此在 log_softmax 中会抵消。
def sector_term_log_distribution(
    det_logabs: Mapping[int, Array],
    A: Mapping[int, Array],
    sectors: Sequence[int],
    eps: float = 1.0e-30,
):
    # dtype 从第一个 sector 的 determinant 读取。
    first_l = int(tuple(sectors)[0])
    dtype = det_logabs[first_l].dtype
    eps_value = jnp.asarray(eps, dtype=dtype)
    logs = []
    for l in sectors:
        l = int(l)
        nq = 2 * l + 1
        for q_index in range(nq):
            # l=0 不乘 A0；其它 l 每一个 q 都乘自己的 A_q。
            if l == 0:
                aqlog = jnp.asarray(0.0, dtype=dtype)
            else:
                aqlog = jnp.log(jnp.maximum(jnp.abs(A[l][q_index]), eps_value))
            logs.append(det_logabs[l][q_index] + aqlog)
    logs = jnp.concatenate(logs, axis=0)
    return jax.nn.log_softmax(logs)
