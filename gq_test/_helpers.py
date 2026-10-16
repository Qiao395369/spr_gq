"""测试共用的 N₂ 输入与小型网络。

保留训练脚本的物理系统和网络分支，只缩小网络宽度与深度，使测试
无需启动采样或优化。固定随机种子保证失败可复现。
"""

from functools import lru_cache

import jax

# 训练配置要求 float64；必须在创建 JAX 数组前启用。
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from vmcnet.gaoqiao.build import build_network


N_ELECTRONS = 14
NSPINS = (7, 7)
CHARGES = (7.0, 7.0)


def nuclear_positions():
    """训练脚本中的两颗氮核；行号只是标签，不是物理可观测量。"""
    return jnp.asarray(((0., 0., 1.034), (0., 0., -1.034)), dtype=jnp.float64)


def electron_positions():
    """构造远离电子-电子、电子-核重合点的固定测试构型。

前 7 行属于自旋向上的电子，后 7 行属于自旋向下的电子。
先把电子分配到两颗核附近，再加确定性的扰动；随机数只在这里用。
"""
    ions = np.asarray(nuclear_positions())
    owners = (0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1)
    noise = np.random.default_rng(134).normal(size=(N_ELECTRONS, 3))
    return jnp.asarray(ions[list(owners)] + 0.43 * noise, dtype=jnp.float64)


@lru_cache(maxsize=2)
def tiny_network(rescale="all"):
    """返回 (参数, signed_network)，且每种缩放方式只初始化一次。

与正式 N₂ 脚本一致：ef_shrd_sym、full 粒子描述、ds_hz_old 包络、
mlp_res Jastrow、实值 float64。缩小 ndet、depth、h1/h2/nh 和
Jastrow 宽度，不改变待测的对称性机制。rescale 默认仍为脚本的 all。
"""
    params, signed_network, *_ = build_network(
        n=N_ELECTRONS,
        charges=jnp.asarray(CHARGES, dtype=jnp.float64),
        nspins=NSPINS,
        key=jax.random.PRNGKey(31),
        ndet=2,
        depth=2,
        h1=8,
        h2=4,
        nh=4,
        do_complex=False,
        gq_type="ef_shrd_sym",
        envelope_type="ds_hz_old",
        feat_params={"rescale": rescale},
        jastrow_type="mlp_res",
        jastrow_mlp_nlayer=2,
        jastrow_mlp_ndim=8,
        dp_type="full",
        RHF=False,
        activation_type="silu",
        jastrow_activation_type="tanh",
        dtype="float64",
    )
    return params, signed_network


def evaluate(electrons=None, ions=None, rescale="all"):
    """按生产接口返回波函数的 (符号, log|ψ|)。"""
    params, network = tiny_network(rescale)
    if electrons is None:
        electrons = electron_positions()
    if ions is None:
        ions = nuclear_positions()
    return network(params, electrons, ions)
