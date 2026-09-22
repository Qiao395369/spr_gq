# ============================================================================
# SO(3) 等变网络工具函数库 —— 纯实数版本、中文逐行详注
# ============================================================================
# 这个文件只负责 SO(3) 网络中的数学与张量运算，不负责 VMC 优化器、Hamiltonian 或采样。
# 整个 SO(3) 分支从参数、隐藏特征、球谐函数、CG 张量、Slater 轨道到最终波函数全部使用实数浮点类型。
# 浮点类型通过 SO3Config.float_dtype 自定义，推荐使用 jnp.float32 或 jnp.float64。
# 如果选择 jnp.float64，必须在程序入口提前执行 jax.config.update("jax_enable_x64", True)。
# 整体数据流为：粒子坐标与类型 -> 初始 h_j^(l) -> 等变卷积 -> Wigner 双线性块 -> y_j^(l)。
# 对每个 l，通过只混 channel、不混 q 的 readout 产生 R 份完整 multiplet A_{r,q}^(l)。
# determinant 不绑定到 q；可选 per_l 模式使用 D_{l,r,k}，或 shared 模式让全部 l 共用 D_{r,k}。
# 每个 determinant 的 orbital 输入由 y_i^(0) 与高阶 y_i^(l) 的旋转不变量组成，并乘各向同性可学习核指数 envelope。
# 完整 multiplet 为 Psi_q^(l)=sum_r A_{r,q}^(l) S_{l,r}；S 由 per-l 或 shared scalar determinant bank 线性组合得到。
# signed_network 默认取每个 l 的 q=0 分量再对 l 求和；完整 multiplet 可单独调用辅助函数取得。
# 最后整个波函数乘满足电子-电子 Kato cusp 的 Padé Jastrow exp(J)。
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
    # det_rank=R：每个 l 使用 R 份独立的完整 SO(3) multiplet readout；每个 r 再配 K 个 scalar determinant。
    # 因此每个 l 的 determinant 数为 R*K，而不是 (2l+1)*K；同一个 r 的 determinant 因子共享给全部 q 分量。
    det_rank: int = 4
    # det_bank_mode 控制 scalar determinant bank 是否在不同 l 之间共享：
    #   "per_l"  : 之前的方法；每个 l 都有自己独立的 R*K 个 determinant。
    #   "shared" : 新的省算力方法；所有 l 共用同一组 R*K determinant，但每个 l 保留独立 alpha_{l,r,k}。
    # shared 模式下实际需要计算的唯一 Slater determinant 数与 lmax/target_ls 无关。
    det_bank_mode: str = "shared"
    # True 时 determinant orbital 不只读取 y_i^(0)，还读取每个高阶 l 的 channel-wise RMS invariant。
    # 这样 determinant 能感知 p/d/... 几何信息，但本身仍然是 SO(3) scalar。
    use_invariant_orbital_features: bool = True
    # scalar signed_network 如何从每个 l 的完整 multiplet 中选出一个标量分量。
    # "q0"：取 q=0；"sum"：固定等权求和所有 q。完整 multiplet 本身不受这个选择影响。
    output_projection: str = "q0"
    # isotropic envelope 中 sigma=softplus(raw_sigma)+envelope_sigma_floor，保证指数衰减系数严格为正。
    envelope_sigma_floor: float = 1.0e-6
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

# 规范化 determinant bank 模式。
def normalize_det_bank_mode(mode: str) -> str:
    mode = str(mode).lower()
    if mode not in ("per_l", "shared"):
        raise ValueError(
            f"det_bank_mode 只支持 'per_l' 或 'shared'，收到 {mode!r}。"
        )
    return mode

# 返回实际需要计算的唯一 Slater determinant 数。
def unique_determinant_count(
    determinants_per_rank: int,
    det_rank: int,
    sectors: Sequence[int],
    det_bank_mode: str,
) -> int:
    K = int(determinants_per_rank)
    R = int(det_rank)
    mode = normalize_det_bank_mode(det_bank_mode)
    if mode == "shared":
        return R * K
    return len(tuple(sectors)) * R * K

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
    # 每个 l 产生 R 份完整 multiplet；R 是 multiplicity/rank 维，而不是磁量子数维。
    R = int(config.det_rank)
    if R <= 0:
        raise ValueError('SO3Config.det_rank 必须大于 0。')
    params['readout'] = {}
    for l in range(config.lmax + 1):
        key, k = jax.random.split(key)
        # shape=(R,C)；A[l] 最终为 (R,2l+1)。
        params['readout'][str(l)] = _randn(k, (R, C), 1.0 / math.sqrt(C), dtype)
    # determinant 绝不按 q 参数化；q 维只存在于完整 irrep/multiplet 中。
    # scalar 输出如何投影 q 由 config.output_projection 决定，不使用可训练 q-specific 系数。
    if config.output_projection not in ('q0', 'sum'):
        raise ValueError("SO3Config.output_projection 目前只支持 'q0' 或 'sum'。")
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
    # A_channels 只对粒子 j 求和，仍保留 channel c 与 irrep 分量 q。
    A_channels = {}
    # A[l] shape=(R,2l+1)：R 份完整 multiplet。
    A = {}
    # 逐个 l 处理。
    for l in range(config.lmax + 1):
        # A_channels[c,q]=sum_j y[j,c,q]。
        A_channels[l] = jnp.sum(y[l], axis=0)
        # w[r,c] 只混 multiplicity/channel c，绝不混 q，因此每个 r 都仍严格属于同一个 l irrep。
        w = params['readout'][str(l)]
        # A[r,q]=sum_c w[r,c] A_channels[c,q]。
        A[l] = jnp.einsum('rc,cq->rq', w, A_channels[l])
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

# 返回 determinant orbital head 的 scalar 输入维数。
def scalar_orbital_feature_dim(config: SO3Config) -> int:
    # 基础 y^(0) 提供 C 个 scalar channel。
    C = int(config.channels)
    # 如果关闭高阶 invariant，只使用原始 y^(0)。
    if not config.use_invariant_orbital_features:
        return C
    # 对每个 l=1,...,lmax 再加入 C 个 channel-wise RMS invariant，因此总维数 C*(lmax+1)。
    return C * (int(config.lmax) + 1)

# 从逐粒子 irrep feature 构造 determinant 可使用的旋转标量 feature。
def make_scalar_orbital_features(
    y: Mapping[int, Array],
    nelec: int,
    config: SO3Config,
) -> Array:
    # l=0 本身就是 scalar，取唯一 q=0 分量，shape=(Ne,C)。
    scalar_parts = [jnp.asarray(y[0][:nelec, :, 0])]
    # 如果用户只想使用 y^(0)，直接返回。
    if not config.use_invariant_orbital_features:
        out = scalar_parts[0]
        _debug_array('orbital_features/y0_only', out)
        return out
    # 对每个高阶 irrep 取 channel-wise RMS：
    #   s_{i,c}^{(l)} = sqrt(mean_q y_{i,c,q}^2 + eps^2)。
    # 实 SO(3) irrep 的旋转矩阵是正交的，因此 q 上的平方和是严格旋转不变量。
    eps_value = jnp.asarray(config.eps, dtype=scalar_parts[0].dtype)
    for l in range(1, config.lmax + 1):
        yl = jnp.asarray(y[l][:nelec], dtype=scalar_parts[0].dtype)
        rms = jnp.sqrt(jnp.mean(yl * yl, axis=-1) + eps_value * eps_value)
        scalar_parts.append(rms)
        _debug_array(f'orbital_features/invariant_l{l}', rms)
    # 最终 shape=(Ne,C*(lmax+1))。
    out = jnp.concatenate(scalar_parts, axis=-1)
    expected = scalar_orbital_feature_dim(config)
    if out.shape[-1] != expected:
        raise ValueError(f'scalar orbital feature dim 应为 {expected}，实际为 {out.shape[-1]}。')
    _debug_array('orbital_features/combined', out)
    return out

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
# SO(3)-safe rank-R determinant 结构。
#
# 对每个 l：
#   A^(l) 的 shape=(R,2l+1)，每个 r 是一份完整 multiplet。
#   determinant 只带 (l,r,k)，绝不带 q：
#
#       S_{l,r}(X) = sum_k alpha_{l,r,k} D_{l,r,k}(X)
#
#   因为 D 与 alpha 都是 rotation scalar，所以
#
#       Psi_q^(l)(X) = sum_r A_{r,q}^(l)(X) S_{l,r}(X)
#
#   仍严格按照 l 阶 irrep 变换。
#
# determinant orbital 的输入只包含 SO(3) scalar feature：
#   y_i^(0) 以及高阶 y_i^(l) 的 channel-wise RMS invariant。
#
# envelope 使用各向同性的可学习 sigma：
#   e_{e,o}^{k} = sum_m pi_{k,o,m} exp[-sigma_{k,o,m} |r_e-R_m|]
#
# 因此 envelope 本身也是严格的 rotation/translation scalar。
#
# 最后整个波函数乘 electron-electron Kato Padé Jastrow exp(J)。
# ============================================================================

# 把正数 value 转成 softplus 的 unconstrained 参数初值，使 softplus(raw)=value。
def _inverse_softplus(value: float, dtype) -> Array:
    # 数值上要求 value>0；这里同时避免 expm1(0)。
    value = max(float(value), 1.0e-8)
    return jnp.asarray(math.log(math.expm1(value)), dtype=dtype)

# 初始化单个 (l,r,spin) orbital head 对应的各向同性 FermiNet 风格 envelope 参数。
def _init_orbital_envelope_params(
    key,
    determinants_per_rank: int,
    norb: int,
    natom: int,
    sigma_init: float,
    sigma_floor: float,
    pi_init: float,
    dtype,
):
    # K 是当前 scalar rank r 对应的 determinant 数。
    K = int(determinants_per_rank)
    if sigma_init <= sigma_floor:
        raise ValueError(
            f'envelope_sigma_init={sigma_init} 必须大于 envelope_sigma_floor={sigma_floor}。'
        )
    # sigma 必须为正；保存 raw_sigma，前向时通过 softplus+floor 得到真正衰减系数。
    raw0 = _inverse_softplus(float(sigma_init) - float(sigma_floor), dtype)
    raw_sigma = jnp.full((K, norb, natom), raw0, dtype=dtype)
    # 加很小的 scalar 随机扰动；不会引入空间方向，因此不破坏 SO(3)。
    key, k_sigma = jax.random.split(key)
    raw_sigma = raw_sigma + _randn(k_sigma, raw_sigma.shape, 1.0e-3, dtype)
    # pi shape=(K,norb,natom)；初始各核等权，总尺度约为 pi_init。
    pi = jnp.full(
        (K, norb, natom),
        jnp.asarray(pi_init / max(natom, 1), dtype=dtype),
        dtype=dtype,
    )
    return {'raw_sigma': raw_sigma, 'pi': pi}

# 初始化一个 R x K scalar determinant bank；该 bank 可属于某个 l，也可以被所有 l 共享。
def _init_determinant_bank(
    key,
    input_dim: int,
    nspins: Sequence[int],
    determinants_per_rank: int,
    det_rank: int,
    natom: int,
    full_det: bool,
    include_bias: bool,
    dtype,
    use_envelope: bool,
    envelope_sigma_init: float,
    envelope_sigma_floor: float,
    envelope_pi_init: float,
):
    K = int(determinants_per_rank)
    R = int(det_rank)
    active_spins = tuple(int(n) for n in nspins if int(n) > 0)
    nelec = int(sum(active_spins))

    rank_heads = []
    rank_envelopes = []

    for _ in range(R):
        spin_heads = []
        spin_envelopes = []

        for nspin in active_spins:
            norb = nelec if full_det else nspin
            out_dim = K * norb

            key, k_head = jax.random.split(key)
            # determinant head 只读取 SO(3) scalar/invariant feature，因此输出仍是 rotation scalar。
            w = _randn(
                k_head,
                (int(input_dim), out_dim),
                1.0 / math.sqrt(max(int(input_dim), 1)),
                dtype,
            )
            head = {'w': w}
            if include_bias:
                head['b'] = jnp.zeros((out_dim,), dtype=dtype)
            spin_heads.append(head)

            if use_envelope:
                key, k_env = jax.random.split(key)
                env = _init_orbital_envelope_params(
                    k_env,
                    determinants_per_rank=K,
                    norb=norb,
                    natom=int(natom),
                    sigma_init=envelope_sigma_init,
                    sigma_floor=envelope_sigma_floor,
                    pi_init=envelope_pi_init,
                    dtype=dtype,
                )
            else:
                env = None
            spin_envelopes.append(env)

        rank_heads.append(tuple(spin_heads))
        rank_envelopes.append(tuple(spin_envelopes))

    return key, tuple(rank_heads), tuple(rank_envelopes)

# 初始化 rank-R scalar determinant 参数。
#
# det_bank_mode="per_l":
#   heads[l][r][spin]，每个 l 独立计算 R*K 个 determinant（之前的方法）。
#
# det_bank_mode="shared":
#   heads[r][spin]，所有 l 共享同一个 R*K determinant bank（省算力方法）。
#
# 两种模式下 alpha[l][r,k] 始终按 l 独立，因此即便 determinant 共享，
# 不同 l 仍可从同一个 antisymmetric basis 中学习不同线性组合。
def init_sector_determinant_params(
    key,
    input_dim: int,
    nspins: Sequence[int],
    determinants_per_rank: int,
    sectors: Sequence[int],
    det_rank: int,
    natom: int,
    full_det: bool = True,
    include_bias: bool = False,
    float_dtype: Any = jnp.float32,
    use_envelope: bool = True,
    envelope_sigma_init: float = 1.0,
    envelope_sigma_floor: float = 1.0e-6,
    envelope_pi_init: float = 1.0,
    det_bank_mode: str = "per_l",
):
    dtype = resolve_float_dtype(float_dtype)
    K = int(determinants_per_rank)
    R = int(det_rank)
    sectors = tuple(int(l) for l in sectors)
    mode = normalize_det_bank_mode(det_bank_mode)

    if K <= 0:
        raise ValueError('determinants_per_rank 必须大于 0。')
    if R <= 0:
        raise ValueError('det_rank 必须大于 0。')
    if int(input_dim) <= 0:
        raise ValueError('input_dim 必须大于 0。')
    active_spins = tuple(int(n) for n in nspins if int(n) > 0)
    nelec = int(sum(active_spins))
    if nelec <= 0:
        raise ValueError('至少需要一个电子才能构造 Slater determinant。')
    if use_envelope and int(natom) <= 0:
        raise ValueError('启用 SO(3) envelope 时至少需要一个原子核。')

    params = {'heads': None, 'envelope': None, 'alpha': {}}

    if mode == 'per_l':
        params['heads'] = {}
        params['envelope'] = {}
        for l in sectors:
            key, heads, envs = _init_determinant_bank(
                key,
                input_dim=input_dim,
                nspins=nspins,
                determinants_per_rank=K,
                det_rank=R,
                natom=natom,
                full_det=full_det,
                include_bias=include_bias,
                dtype=dtype,
                use_envelope=use_envelope,
                envelope_sigma_init=envelope_sigma_init,
                envelope_sigma_floor=envelope_sigma_floor,
                envelope_pi_init=envelope_pi_init,
            )
            params['heads'][str(l)] = heads
            params['envelope'][str(l)] = envs
    else:
        # 所有 l 只初始化一次 orbital/envelope bank。
        key, heads, envs = _init_determinant_bank(
            key,
            input_dim=input_dim,
            nspins=nspins,
            determinants_per_rank=K,
            det_rank=R,
            natom=natom,
            full_det=full_det,
            include_bias=include_bias,
            dtype=dtype,
            use_envelope=use_envelope,
            envelope_sigma_init=envelope_sigma_init,
            envelope_sigma_floor=envelope_sigma_floor,
            envelope_pi_init=envelope_pi_init,
        )
        params['heads'] = heads
        params['envelope'] = envs

    # 每个 l 始终有独立 alpha，因此 shared 只共享昂贵的 determinant basis，
    # 不共享不同 l 对 determinant basis 的线性组合。
    alpha0 = 1.0 / math.sqrt(R * K)
    for l in sectors:
        params['alpha'][str(l)] = jnp.full(
            (R, K),
            jnp.asarray(alpha0, dtype=dtype),
            dtype=dtype,
        )

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

# 应用 SO(3)-safe 的各向同性原子核指数 envelope：
#   e_{e,o}^{k} = sum_m pi_{k,o,m} exp[-sigma_{k,o,m} |r_e-R_m|]。
def apply_orbital_envelope(
    electron_pos: Array,
    atoms: Array,
    envelope_params: Optional[Mapping[str, Array]],
    sigma_floor: float = 1.0e-6,
    eps: float = 1.0e-12,
) -> Array:
    if envelope_params is None:
        raise ValueError('apply_orbital_envelope 收到 None；禁用 envelope 时不应调用该函数。')

    # pi 与 raw_sigma 都是 (K,norb,natom)；没有 Cartesian matrix index，因此不会挑选固定空间方向。
    pi = envelope_params['pi']
    raw_sigma = envelope_params['raw_sigma']
    dtype = raw_sigma.dtype
    electron_pos = jnp.asarray(electron_pos, dtype=dtype)
    atoms = jnp.asarray(atoms, dtype=dtype)

    # diff[e,m,d]=r_e-R_m。
    diff = electron_pos[:, None, :] - atoms[None, :, :]
    eps_value = jnp.asarray(eps, dtype=dtype)
    # radius[e,m] 是严格 rotation scalar；eps^2 让电子恰在核上时一、二阶 AD 保持有限。
    radius = jnp.sqrt(jnp.sum(diff * diff, axis=-1) + eps_value * eps_value)

    # sigma>0，shape=(K,norb,natom)。
    sigma = jax.nn.softplus(raw_sigma) + jnp.asarray(sigma_floor, dtype=dtype)

    # decay[e,k,o,m]=exp[-sigma[k,o,m] * radius[e,m]]。
    decay = jnp.exp(-radius[:, None, None, :] * sigma[None, ...])
    # 对原子核 m 求和，得到 shape=(nspin,K,norb)。
    env = jnp.sum(pi[None, ...] * decay, axis=-1)

    if SO3_DEBUG_VERBOSE:
        _debug_array('envelope/pi', pi)
        _debug_array('envelope/sigma', sigma)
        _debug_array('envelope/radius', radius)
    _debug_array('envelope/value', env)
    return env

# 从一个 determinant bank 构造 R 组、每组 K 个 Slater orbital matrix。
def _build_orbitals_from_bank(
    scalar_electron_features: Array,
    electron_pos: Array,
    atoms: Array,
    head_bank,
    envelope_bank,
    nspins: Sequence[int],
    determinants_per_rank: int,
    det_rank: int,
    full_det: bool,
    envelope_sigma_floor: float,
    eps: float,
    debug_label: str,
):
    K = int(determinants_per_rank)
    R = int(det_rank)
    active_spins = tuple(int(n) for n in nspins if int(n) > 0)
    nelec = int(sum(active_spins))

    split_points = []
    running = 0
    for nspin in active_spins[:-1]:
        running += nspin
        split_points.append(running)
    x_by_spin = jnp.split(scalar_electron_features, split_points, axis=0)
    pos_by_spin = jnp.split(electron_pos, split_points, axis=0)

    rank_results = []
    for r in range(R):
        heads = head_bank[r]
        envs = envelope_bank[r]
        spin_mats = []

        for x, epos, head, env_params, nspin in zip(
            x_by_spin,
            pos_by_spin,
            heads,
            envs,
            active_spins,
        ):
            raw = _apply_orbital_head(x, head)
            norb = nelec if full_det else nspin
            mat = raw.reshape(nspin, K, norb)

            if env_params is not None:
                env = apply_orbital_envelope(
                    epos,
                    atoms,
                    env_params,
                    sigma_floor=envelope_sigma_floor,
                    eps=eps,
                )
                mat = mat * env

            mat = jnp.transpose(mat, (1, 0, 2))
            spin_mats.append(mat)

        if full_det:
            rank_mat = jnp.concatenate(spin_mats, axis=1)
            _debug_array(f'det/orbitals_{debug_label}_r{r}', rank_mat)
        else:
            rank_mat = tuple(spin_mats)
            for iblock, block in enumerate(rank_mat):
                _debug_array(
                    f'det/orbitals_{debug_label}_r{r}_spinblock{iblock}',
                    block,
                )
        rank_results.append(rank_mat)

    return tuple(rank_results)

# 构造 Slater matrices。
# 返回格式对两种模式保持一致：result[l][r]。
# shared 模式下各 l 指向同一组已经计算好的 matrices，不会重新执行 orbital head。
def build_sector_orbitals(
    scalar_electron_features: Array,
    electron_pos: Array,
    atoms: Array,
    det_params: Mapping[str, object],
    nspins: Sequence[int],
    determinants_per_rank: int,
    sectors: Sequence[int],
    det_rank: int,
    full_det: bool = True,
    envelope_sigma_floor: float = 1.0e-6,
    eps: float = 1.0e-12,
    det_bank_mode: str = "per_l",
):
    mode = normalize_det_bank_mode(det_bank_mode)

    # print("\n========== SO3 DET DEBUG ==========")
    # print("Forward det_bank_mode:", det_bank_mode)

    # print("heads type:", type(det_params["heads"]))
    # print("envelope type:", type(det_params["envelope"]))

    # if isinstance(det_params["heads"], dict):
    #     print("heads keys:", list(det_params["heads"].keys()))
    # else:
    #     print("heads length:", len(det_params["heads"]))

    # if isinstance(det_params["envelope"], dict):
    #     print("envelope keys:", list(det_params["envelope"].keys()))
    # else:
    #     print("envelope length:", len(det_params["envelope"]))

    # print("===================================\n")
    
    active_spins = tuple(int(n) for n in nspins if int(n) > 0)
    nelec = int(sum(active_spins))
    sectors = tuple(int(l) for l in sectors)

    scalar_electron_features = jnp.asarray(scalar_electron_features)
    electron_pos = jnp.asarray(
        electron_pos,
        dtype=scalar_electron_features.dtype,
    ).reshape((nelec, -1))
    atoms = jnp.asarray(atoms, dtype=scalar_electron_features.dtype)

    if scalar_electron_features.shape[0] != nelec:
        raise ValueError(
            f'scalar_electron_features 第一维应为 {nelec}，实际为 {scalar_electron_features.shape[0]}。'
        )

    _debug_array('det/scalar_electron_features', scalar_electron_features)
    _debug_array('det/electron_pos', electron_pos)

    result = {}
    if mode == 'shared':
        shared = _build_orbitals_from_bank(
            scalar_electron_features,
            electron_pos,
            atoms,
            det_params['heads'],
            det_params['envelope'],
            nspins=nspins,
            determinants_per_rank=determinants_per_rank,
            det_rank=det_rank,
            full_det=full_det,
            envelope_sigma_floor=envelope_sigma_floor,
            eps=eps,
            debug_label='shared',
        )
        for l in sectors:
            result[l] = shared
    else:
        for l in sectors:
            result[l] = _build_orbitals_from_bank(
                scalar_electron_features,
                electron_pos,
                atoms,
                det_params['heads'][str(l)],
                det_params['envelope'][str(l)],
                nspins=nspins,
                determinants_per_rank=determinants_per_rank,
                det_rank=det_rank,
                full_det=full_det,
                envelope_sigma_floor=envelope_sigma_floor,
                eps=eps,
                debug_label=f'l{l}',
            )

    return result

# 对一个 R x K orbital bank 求 determinant。
def _determinants_from_orbital_bank(
    rank_orbitals,
    K: int,
    dtype,
    full_det: bool,
    debug_label: str,
):
    rank_signs = []
    rank_logs = []

    for r, orbitals_r in enumerate(rank_orbitals):
        if full_det:
            sign_r, log_r = jnp.linalg.slogdet(orbitals_r)
        else:
            sign_r = jnp.ones((K,), dtype=dtype)
            log_r = jnp.zeros((K,), dtype=dtype)
            for block in orbitals_r:
                p, q = jnp.linalg.slogdet(block)
                sign_r = sign_r * p
                log_r = log_r + q

        _debug_array(f'det/slogdet_sign_{debug_label}_r{r}', sign_r)
        _debug_array(f'det/slogdet_logabs_{debug_label}_r{r}', log_r)
        rank_signs.append(sign_r)
        rank_logs.append(log_r)

    return jnp.stack(rank_signs, axis=0), jnp.stack(rank_logs, axis=0)

# 计算 scalar Slater determinant。
# 输出 API 始终保持 det_sign[l].shape=(R,K)，方便后续每个 l 使用独立 alpha。
# shared 模式只执行一次 R*K 个 slogdet，然后把结果复用于全部 l。
def sector_determinants_from_scalar_features(
    scalar_electron_features: Array,
    electron_pos: Array,
    atoms: Array,
    det_params: Mapping[str, object],
    nspins: Sequence[int],
    determinants_per_rank: int,
    sectors: Sequence[int],
    det_rank: int,
    full_det: bool = True,
    envelope_sigma_floor: float = 1.0e-6,
    eps: float = 1.0e-12,
    det_bank_mode: str = "per_l",
):
    mode = normalize_det_bank_mode(det_bank_mode)
    sectors = tuple(int(l) for l in sectors)
    K = int(determinants_per_rank)

    orbitals = build_sector_orbitals(
        scalar_electron_features,
        electron_pos,
        atoms,
        det_params,
        nspins,
        determinants_per_rank,
        sectors,
        det_rank,
        full_det=full_det,
        envelope_sigma_floor=envelope_sigma_floor,
        eps=eps,
        det_bank_mode=mode,
    )

    signs = {}
    logabs = {}

    if mode == 'shared':
        first_l = sectors[0]
        shared_sign, shared_log = _determinants_from_orbital_bank(
            orbitals[first_l],
            K=K,
            dtype=scalar_electron_features.dtype,
            full_det=full_det,
            debug_label='shared',
        )
        for l in sectors:
            signs[l] = shared_sign
            logabs[l] = shared_log
    else:
        for l in sectors:
            signs[l], logabs[l] = _determinants_from_orbital_bank(
                orbitals[l],
                K=K,
                dtype=scalar_electron_features.dtype,
                full_det=full_det,
                debug_label=f'l{l}',
            )

    return signs, logabs, orbitals

# 兼容旧的函数名；新语义已经不是只从 y0 构造，而是从 scalar/invariant feature 构造。
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
    # 只有在 det_rank=1 且 head 输入维数等于 y0 最后一维时，这个兼容入口才有明确定义。
    first_l = int(tuple(sectors)[0])
    heads_root = det_params['heads']
    R = len(heads_root[str(first_l)]) if isinstance(heads_root, dict) else len(heads_root)
    if R != 1:
        raise ValueError(
            '新版 rank-R 结构请调用 sector_determinants_from_scalar_features；'
            'sector_determinants_from_y0 仅保留给 det_rank=1 的旧调用。'
        )
    return sector_determinants_from_scalar_features(
        y0_electrons,
        electron_pos,
        atoms,
        det_params,
        nspins=nspins,
        determinants_per_rank=determinants_per_l,
        sectors=sectors,
        det_rank=1,
        full_det=full_det,
        eps=eps,
        det_bank_mode='per_l' if isinstance(det_params['heads'], dict) else 'shared',
    )

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

    # 根据 nspins 构造每个电子的 spin-channel 标签。
    labels = []
    for ispin, nspin in enumerate(nspins):
        labels.extend([ispin] * int(nspin))
    spin_label = jnp.asarray(labels, dtype=jnp.int32)
    if spin_label.shape[0] != nelec:
        raise ValueError(
            f'nspins 总电子数 {spin_label.shape[0]} 与 electron_pos 第一维 {nelec} 不一致。'
        )

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

    # u_ij(r)=a*r/(1+b*r)，因此 u'_ij(0)=a；b 只控制有限距离形状。
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
    # 先把 NaN/+Inf/-Inf 全部替换为 -Inf，再进入 exp；不要在 jnp.where 的未选分支里计算 exp(NaN)。
    safe_logs = jnp.where(finite, logabs, -jnp.inf)
    maxlog = jnp.max(safe_logs)
    base = jnp.where(
        jnp.isfinite(maxlog),
        maxlog,
        jnp.asarray(0.0, dtype=logabs.dtype),
    )
    scaled_terms = signs * jnp.exp(safe_logs - base)
    scaled = jnp.sum(
        jnp.where(
            finite,
            scaled_terms,
            jnp.asarray(0.0, dtype=logabs.dtype),
        )
    )

    _debug_scalar('logsumexp/maxlog', maxlog)
    _debug_scalar('logsumexp/base', base)
    _debug_scalar('logsumexp/scaled', scaled)

    final_sign = jnp.sign(scaled)
    mag = jnp.abs(scaled)
    final_logabs = jnp.where(
        mag > 0,
        base + jnp.log(mag),
        -jnp.inf,
    )

    _debug_scalar('logsumexp/final_sign', final_sign)
    _debug_scalar('logsumexp/final_logabs', final_logabs)
    return final_sign, final_logabs

# 把 alpha_{l,r,k} D_{l,r,k} 在 k 上求和，得到每个 rank r 的 scalar factor S_{l,r}。
def rank_determinant_sums(
    det_signs: Mapping[int, Array],
    det_logabs: Mapping[int, Array],
    det_params: Mapping[str, object],
    sectors: Sequence[int],
):
    rank_sign = {}
    rank_logabs = {}

    for l in sectors:
        l = int(l)
        alpha = jnp.asarray(det_params['alpha'][str(l)], dtype=det_logabs[l].dtype)

        if alpha.shape != det_logabs[l].shape:
            raise ValueError(
                f'l={l} 的 alpha shape={alpha.shape} 与 determinant shape={det_logabs[l].shape} 不一致。'
            )

        signs_l = []
        logs_l = []

        for r in range(alpha.shape[0]):
            a = alpha[r]
            a_abs = jnp.abs(a)
            # alpha 可以在训练中穿过 0；先对安全下界取 log，避免未选 jnp.where 分支里的 log(0) 污染梯度。
            tiny = jnp.asarray(jnp.finfo(det_logabs[l].dtype).tiny, dtype=det_logabs[l].dtype)
            safe_alog = jnp.log(jnp.maximum(a_abs, tiny))
            term_sign = det_signs[l][r] * jnp.sign(a)
            term_log = jnp.where(
                (a_abs > 0) & jnp.isfinite(det_logabs[l][r]),
                det_logabs[l][r] + safe_alog,
                -jnp.inf,
            )
            s, q = real_signed_logsumexp(term_sign, term_log)
            signs_l.append(s)
            logs_l.append(q)

        rank_sign[l] = jnp.stack(signs_l, axis=0)
        rank_logabs[l] = jnp.stack(logs_l, axis=0)

        _debug_array(f'rank_sum/sign_l{l}', rank_sign[l])
        _debug_array(f'rank_sum/logabs_l{l}', rank_logabs[l])

    return rank_sign, rank_logabs

# 构造完整 SO(3) multiplet：
#   Psi_q^(l)=sum_r A_{r,q}^(l) S_{l,r}。
# 返回每个 l 的 signed-log multiplet，shape=(2l+1,)。
def sector_wavefunction_multiplets(
    det_signs: Mapping[int, Array],
    det_logabs: Mapping[int, Array],
    A: Mapping[int, Array],
    det_params: Mapping[str, object],
    sectors: Sequence[int],
):
    rank_sign, rank_logabs = rank_determinant_sums(
        det_signs,
        det_logabs,
        det_params,
        sectors,
    )

    multiplet_sign = {}
    multiplet_logabs = {}

    for l in sectors:
        l = int(l)
        A_l = jnp.asarray(A[l], dtype=rank_logabs[l].dtype)

        if A_l.shape[0] != rank_logabs[l].shape[0]:
            raise ValueError(
                f'l={l} 的 A rank={A_l.shape[0]} 与 determinant rank={rank_logabs[l].shape[0]} 不一致。'
            )

        q_signs = []
        q_logs = []

        for q_index in range(A_l.shape[1]):
            aq = A_l[:, q_index]
            aq_abs = jnp.abs(aq)
            # A 分量允许穿过 0；同样先构造安全 log，避免 log(0) 的 AD NaN。
            tiny = jnp.asarray(jnp.finfo(rank_logabs[l].dtype).tiny, dtype=rank_logabs[l].dtype)
            safe_aqlog = jnp.log(jnp.maximum(aq_abs, tiny))

            term_sign = rank_sign[l] * jnp.sign(aq)
            term_log = jnp.where(
                (aq_abs > 0) & jnp.isfinite(rank_logabs[l]),
                rank_logabs[l] + safe_aqlog,
                -jnp.inf,
            )

            s, q = real_signed_logsumexp(term_sign, term_log)
            q_signs.append(s)
            q_logs.append(q)

        multiplet_sign[l] = jnp.stack(q_signs, axis=0)
        multiplet_logabs[l] = jnp.stack(q_logs, axis=0)

        _debug_array(f'multiplet/sign_l{l}', multiplet_sign[l])
        _debug_array(f'multiplet/logabs_l{l}', multiplet_logabs[l])

    return multiplet_sign, multiplet_logabs

# 从完整 multiplet 中选择 scalar network 输出。
# 重要：SO(3) 对称结构保存在 sector_wavefunction_multiplets 的完整 q 向量中；
# 这里的 q0/sum 只是为了适配现有 VMC signed_network 的标量返回接口。
def project_sector_multiplets(
    multiplet_sign: Mapping[int, Array],
    multiplet_logabs: Mapping[int, Array],
    sectors: Sequence[int],
    projection: str = 'q0',
):
    sector_signs = []
    sector_logs = []

    for l in sectors:
        l = int(l)
        signs_l = multiplet_sign[l]
        logs_l = multiplet_logabs[l]

        if projection == 'q0':
            # q=-l,...,+l，因此 q=0 的数组下标就是 l。
            s = signs_l[l]
            q = logs_l[l]
        elif projection == 'sum':
            # 固定等权 covector (1,...,1)；完整 multiplet 本身仍在投影前保留。
            s, q = real_signed_logsumexp(signs_l, logs_l)
        else:
            raise ValueError(f"未知 output_projection={projection!r}；只支持 'q0' 或 'sum'。")

        sector_signs.append(s)
        sector_logs.append(q)

    return real_signed_logsumexp(
        jnp.stack(sector_signs, axis=0),
        jnp.stack(sector_logs, axis=0),
    )

# 组装最终标量波函数，并乘公共 Kato Jastrow。
def assemble_sector_wavefunction(
    det_signs: Mapping[int, Array],
    det_logabs: Mapping[int, Array],
    A: Mapping[int, Array],
    det_params: Mapping[str, object],
    sectors: Sequence[int],
    projection: str = 'q0',
    jastrow_log: Optional[Array] = None,
    eps: float = 1.0e-30,
):
    # eps 保留在接口中便于兼容；signed-log 求和本身不需要人为截断。
    del eps

    multiplet_sign, multiplet_logabs = sector_wavefunction_multiplets(
        det_signs,
        det_logabs,
        A,
        det_params,
        sectors,
    )

    final_sign, final_logabs = project_sector_multiplets(
        multiplet_sign,
        multiplet_logabs,
        sectors,
        projection=projection,
    )

    if jastrow_log is not None:
        final_logabs = final_logabs + jnp.asarray(
            jastrow_log,
            dtype=final_logabs.dtype,
        )
        _debug_scalar('assemble/jastrow_log', jastrow_log)
        _debug_scalar('assemble/final_logabs_with_jastrow', final_logabs)

    return final_sign, final_logabs

# 计算 scalar 输出投影下，每个 rank r 的 A 因子 B_{l,r}。
def projected_rank_amplitudes(
    A: Mapping[int, Array],
    sectors: Sequence[int],
    projection: str,
):
    out = {}

    for l in sectors:
        l = int(l)
        A_l = A[l]

        if projection == 'q0':
            out[l] = A_l[:, l]
        elif projection == 'sum':
            out[l] = jnp.sum(A_l, axis=-1)
        else:
            raise ValueError(f"未知 output_projection={projection!r}；只支持 'q0' 或 'sum'。")

    return out

# 返回每个 (l,r,k) 对 scalar 输出的原始 log 模长，供 det_fn/诊断。
# 公共 Jastrow 因子对所有项相同，因此在 log_softmax 中抵消。
def sector_term_log_distribution(
    det_logabs: Mapping[int, Array],
    A: Mapping[int, Array],
    det_params: Mapping[str, object],
    sectors: Sequence[int],
    projection: str = 'q0',
    eps: float = 1.0e-30,
):
    first_l = int(tuple(sectors)[0])
    dtype = det_logabs[first_l].dtype
    eps_value = jnp.asarray(eps, dtype=dtype)
    B = projected_rank_amplitudes(A, sectors, projection)

    logs = []

    for l in sectors:
        l = int(l)
        alpha = jnp.asarray(det_params['alpha'][str(l)], dtype=dtype)
        b = jnp.asarray(B[l], dtype=dtype)

        if alpha.shape != det_logabs[l].shape:
            raise ValueError(
                f'l={l} alpha shape={alpha.shape} 与 determinant shape={det_logabs[l].shape} 不一致。'
            )
        if b.shape[0] != alpha.shape[0]:
            raise ValueError(
                f'l={l} projected A rank={b.shape[0]} 与 alpha rank={alpha.shape[0]} 不一致。'
            )

        # 每个 (r,k) 的模长：
        # log|D| + log|alpha| + log|B_r|。
        alog = jnp.log(jnp.maximum(jnp.abs(alpha), eps_value))
        blog = jnp.log(jnp.maximum(jnp.abs(b), eps_value))[:, None]
        logs.append(det_logabs[l] + alog + blog)

    logs = jnp.concatenate([x.reshape(-1) for x in logs], axis=0)

    # 正常情况下直接 log_softmax；若极端情况下全部为 -inf，返回均匀诊断分布避免 NaN。
    finite_any = jnp.any(jnp.isfinite(logs))
    safe_logs = jnp.where(
        finite_any,
        logs,
        jnp.zeros_like(logs),
    )
    return jax.nn.log_softmax(safe_logs)

