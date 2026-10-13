"""各测试共享的输入构型、旋转矩阵和网络构造函数。

本模块把“预期怎样变换”与“网络实际算出什么”分开：预期的 D_l
由球谐函数在若干方向上的取值确定，实际结果来自完整的 SO(3) 主干。
这样测试的不只是球谐函数本身，还包括卷积、CG 耦合和读出。
"""

from functools import lru_cache
import math

import jax

# 等变性测试比较两条不同计算路径，单精度的舍入误差可能掩盖真正的偏差。
# 因此在创建任何 JAX 数组之前启用 64 位浮点数。
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from vmcnet.gaoqiao import so3_utils
from vmcnet.gaoqiao.build import build_network


# 三种数域路径：全程实基、全程复基，以及复基主干计算后转为实基。
# 最后一种模式的实部、虚部分别作为实 channel 保存。
BASES = ("real", "complex", "complex_to_real")


def rotation_matrix() -> np.ndarray:
    """给出一个非平凡的三维主动旋转矩阵 R。

    列向量约定下 R = Rz @ Ry：先绕 y 轴，再绕 z 轴。
    选择两个非零角度，是为了让不同 m/q 分量真正发生混合；
    只绕 z 轴可能漏掉某些混合顺序或相位错误。
    """
    azimuth, polar = 0.43, -0.71
    cz, sz = math.cos(azimuth), math.sin(azimuth)
    cy, sy = math.cos(polar), math.sin(polar)
    # rz、ry 都作用于列向量。测试中的坐标按行存储，实际应用时用 u @ R.T。
    rz = np.array(((cz, -sz, 0.), (sz, cz, 0.), (0., 0., 1.)))
    ry = np.array(((cy, 0., sy), (0., 1., 0.), (-sy, 0., cy)))
    return rz @ ry


ROTATION = rotation_matrix()


def analytic_representation_matrix(l: int, basis: str,
                                   rotation=None) -> np.ndarray:
    """从笛卡尔向量/对称无迹张量独立构造 l=1、2 的旋转矩阵。

    本函数不调用 so3_utils 的球谐、换基矩阵或 CG 系数，因此能检验
    representation_matrix 从球谐取值反推的矩阵是否符合已知几何表示。
    l=1 用三维向量；l=2 用五维对称无迹二阶张量。rotation 省略时
    使用默认的 ROTATION；传入其他 3x3 矩阵可检查不同旋转。复基结果再根据
    明写在这里的标准复/实球谐对应关系计算，而不读取生产代码的 S_l。
    """
    rotation = ROTATION if rotation is None else np.asarray(rotation)
    if l == 1:
        # 实球谐顺序为 [y,z,x]；P 将笛卡尔向量 [x,y,z] 排成该顺序。
        permutation = np.array(((0., 1., 0.), (0., 0., 1.), (1., 0., 0.)))
        real_matrix = permutation @ rotation @ permutation.T
        # 行依次为 q=-1,0,1，列依次为复球谐的 m=-1,0,1。
        complex_to_real = np.array((
            (1j, 0., 1j),
            (0., math.sqrt(2.), 0.),
            (1., 0., -1.),
        ), dtype=complex) / math.sqrt(2.)
    elif l == 2:
        # 五个对称无迹矩阵 B_q 满足 u^T B_q u ∝ Y^real_{2,q}(u)。
        # 共同的比例常数为 sqrt(15/(4π))，在求 D_2 时会抵消。
        # 顺序 q=-2,-1,0,1,2，依次对应 xy、yz、3z²-1、xz、x²-y²。
        half = 0.5
        inv_sqrt3 = 1. / math.sqrt(3.)
        tensors = np.array((
            ((0., half, 0.), (half, 0., 0.), (0., 0., 0.)),
            ((0., 0., 0.), (0., 0., half), (0., half, 0.)),
            ((-half * inv_sqrt3, 0., 0.),
             (0., -half * inv_sqrt3, 0.), (0., 0., inv_sqrt3)),
            ((0., 0., half), (0., 0., 0.), (half, 0., 0.)),
            ((half, 0., 0.), (0., -half, 0.), (0., 0., 0.)),
        ))
        # 所有 B_q 的 Frobenius 范数平方均为 1/2，且两两正交。
        # Y_q(Ru)=u^T(R^T B_q R)u；投影到 B_p 上得到 D_{q,p}。
        real_matrix = np.array([
            [2. * np.sum(tensors[p] * (rotation.T @ tensors[q] @ rotation))
             for p in range(5)]
            for q in range(5)
        ])
        # 行 q、列 m 都按 -2,-1,0,1,2 排列。
        complex_to_real = np.array((
            (1j, 0., 0., 0., -1j),
            (0., 1j, 0., 1j, 0.),
            (0., 0., math.sqrt(2.), 0., 0.),
            (0., 1., 0., -1., 0.),
            (1., 0., 0., 0., 1.),
        ), dtype=complex) / math.sqrt(2.)
    else:
        raise ValueError("解析参照只实现 l=1 和 l=2。")

    if basis == "real":
        return real_matrix
    if basis == "complex":
        # Y_real=S Y_complex，故 D_complex=S† D_real S。
        return complex_to_real.conj().T @ real_matrix @ complex_to_real
    raise ValueError(f"未知 basis={basis!r}")


def representation_matrix(l: int, basis: str, rotation=None) -> np.ndarray:
    """从球谐在独立探针方向上的值求同阶旋转矩阵。

    每个 Y_l(u) 都有 2l+1 个分量。把多个方向的列向量并排，记作
    X=[Y_l(u_1), ...]；旋转后的取值记作 X_R=[Y_l(Ru_1), ...]。
    理论上 X_R=D_l(R) X，因此满秩时可由 X_R @ pinv(X) 求出 D_l。

    这里仅调用球谐函数，不调用被测网络的 CG 系数、卷积或读出。
    所得矩阵作为网络各阶特征变换规律的参照，而非网络自己的预测。
    rotation 省略时使用默认的 ROTATION。l=1、2 还需与
    analytic_representation_matrix 独立比较，才能
    检查球谐实现本身的旋转矩阵是否正确。
    """
    rotation = ROTATION if rotation is None else np.asarray(rotation)
    # 使用固定随机种子，令每次运行采用相同的 16 个方向；归一化得到单位向量。
    # 16 大于本组测试的最大分量数 2*2+1=5，便于求稳定的伪逆。
    probes = np.random.default_rng(1207).normal(size=(16, 3))
    probes /= np.linalg.norm(probes, axis=-1, keepdims=True)
    # spherical_harmonics 返回 (探针数, 2l+1)；转置后每列是一个 Y_l(u)。
    original = np.asarray(so3_utils.spherical_harmonics(
        l, jnp.asarray(probes), basis=basis)).T
    rotated = np.asarray(so3_utils.spherical_harmonics(
        l, jnp.asarray(probes @ rotation.T), basis=basis)).T
    # 若列向量没有张成整个 l 阶空间，伪逆不能唯一确定 D_l。
    if np.linalg.matrix_rank(original) != 2 * l + 1:
        raise AssertionError("球谐探针矩阵未满秩，无法确定旋转矩阵。")
    return rotated @ np.linalg.pinv(original)


@lru_cache(maxsize=None)
def backbone_rotation_case(basis: str, regularize_angular_origin=False):
    """用同一套参数计算原坐标和旋转坐标下的 SO(3) 主干输出。

    lmax=2 让网络内部同时产生 l=0、1、2 特征。target_ls 只限制
    最终行列式扇区，不妨碍这里检查中间各阶特征。缓存避免每个阶数
    都重复初始化和运行网络；缓存键是 basis 和正则化开关。
    """
    config = so3_utils.SO3Config(
        basis=basis, lmax=2, target_ls=(0,), channels=4,
        n_layers=1, n_rbf=4, wbn_layers=1, det_rank=2,
        float_dtype=jnp.float64,
        regularize_angular_origin=regularize_angular_origin,
    )
    # 六个位置刻意不沿同一轴排列，确保旋转会改变所有相关角分量。
    positions = jnp.asarray((
        (0.21, -0.37, 0.54), (-0.82, 0.11, 0.29),
        (0.45, 0.73, -0.18), (1.24, -0.51, 0.42),
        (-0.31, -1.07, -0.64), (0.68, 0.27, 1.13),
    ))
    # type_features 为 (N,T)；每行是一个粒子的三维 one-hot 类型信息。
    # 类型属于 l=0 标量，旋转坐标时不应改变它。
    types = jnp.asarray((
        (1., 0., 0.), (1., 0., 0.), (0., 1., 0.),
        (0., 0., 1.), (0., 0., 1.), (0., 1., 0.),
    ))
    # 固定随机参数，只旋转几何输入；若重采样参数，就无法检验等变性。
    params = so3_utils.init_so3_params(jax.random.PRNGKey(7), 3, config)
    original = so3_utils.apply_so3_network(params, positions, types, config)
    rotated = so3_utils.apply_so3_network(
        params, positions @ jnp.asarray(ROTATION.T), types, config)
    return original, rotated


def assert_multiplet_rotation(testcase, l: int, basis: str,
                             regularize_angular_origin=False):
    """断言 l 阶特征满足 F_l(Rx)=D_l(R)F_l(x)。

    所有被检查数组的最后一维都是 2l+1：h/y 是逐粒子特征，
    A_channels 是粒子求和后的 channel，A 是最终线性读出。
    对前面的粒子、channel、读出维度均不施加 D_l；它只混合角分量。
    """
    original, rotated = backbone_rotation_case(basis, regularize_angular_origin)
    # complex_to_real 的主干 h 仍是复基，末端 y/A 已变成实基；
    # 因此同一次测试要为这两个阶段选用不同的表示矩阵。
    backbone_basis = "real" if basis == "real" else "complex"
    output_basis = "complex" if basis == "complex" else "real"
    matrices = {
        "h": representation_matrix(l, backbone_basis),
        "y": representation_matrix(l, output_basis),
        "A_channels": representation_matrix(l, output_basis),
        "A": representation_matrix(l, output_basis),
    }
    for name, matrix in matrices.items():
        value = np.asarray(original[name][l])
        observed = np.asarray(rotated[name][l])
        # 排除全零特征的空测试：零数组在任何矩阵下都会“等变”。
        testcase.assertGreater(np.linalg.norm(value), 1e-9, (basis, l, name))
        # q 是旋转后的角分量，p 是旋转前的角分量；前导维度原样保留。
        expected = np.einsum("qp,...p->...q", matrix, value)
        np.testing.assert_allclose(
            observed, expected, rtol=2e-9, atol=2e-9,
            err_msg=f"{basis} 的 l={l} {name} 旋转不等变",
        )


@lru_cache(maxsize=None)
def wavefunction_case(basis: str, det_bank_mode: str = "shared",
                      target_ls=(0,), regularize_angular_origin=False):
    """构造小型完整波函数，并返回固定的电子/原子核构型。

    电子按 nspins=(2,2) 排列：索引 0、1 为上自旋，2、3 为下自旋。
    两组各有两个电子，便于分别交换并验证反对称性。
    target_ls 决定参与最终行列式读出的 l 阶集合。
    regularize_angular_origin 让高阶边特征在粒子重合处按 r^l 衰减。
    """
    # 核坐标与电子坐标都避开特殊对称位置，减小波函数偶然为零的风险。
    atoms = jnp.asarray(((-0.6, 0.2, 0.1), (0.8, -0.3, 0.4)))
    electrons = jnp.asarray((
        (0.31, 0.54, -0.27), (-0.74, 0.18, 0.63),
        (0.85, -0.61, 0.22), (-0.18, -0.93, -0.45),
    ))
    # 使用真实的 build_network 入口，覆盖轨道、行列式和总波函数，
    # 而不仅仅检查独立的 SO(3) 主干。固定随机种子保证测试可复现。
    params, network, _, _, _ = build_network(
        n=4, charges=jnp.asarray((2., 2.)), nspins=(2, 2),
        key=jax.random.PRNGKey(19), ndet=2,
        # do_complex 是旧接口参数；SO(3) 的数域由 so3_params.basis 控制。
        depth=1, h1=4, h2=4, nh=4, do_complex=False,
        gq_type="so3", dp_type="full", jastrow_type="null",
        so3_params=dict(
            basis=basis, lmax=2, target_ls=target_ls, channels=4,
            n_layers=1, n_rbf=4, wbn_layers=1, det_rank=2,
            det_bank_mode=det_bank_mode, use_envelope=True,
            use_kato_jastrow=True,
            regularize_angular_origin=regularize_angular_origin,
        ),
        dtype="float64",
    )
    return params, network, electrons, atoms


def signed_log_value(network, params, electrons, atoms):
    """取出波函数的 (phase, log|psi|) 表示，并排除无效测试点。

    直接恢复 psi=phase*exp(log|psi|) 可能上溢或下溢。
    对非零的实/复波函数，phase 的模应为 1；交换同自旋电子
    应使 phase 乘 -1，而 log|psi| 保持不变。
    """
    phase, logabs = network(params, electrons, atoms)
    # JAX 标量转成普通 Python 数值，方便 unittest/NumPy 做数值比较。
    phase, logabs = np.asarray(phase).item(), float(logabs)
    if not np.isfinite(logabs) or not np.isclose(abs(phase), 1., atol=1e-10):
        raise AssertionError("测试构型上的波函数为零或非有限，无法检验符号。")
    return phase, logabs
