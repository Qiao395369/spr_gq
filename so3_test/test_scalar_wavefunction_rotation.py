"""检验最终波函数在整体旋转下的标量性质。

这里 target_ls=(0,)：最终只用 l=0 扇区构造波函数。
电子和原子核一起旋转，相对距离与物理构型不变，故输出应不变。
若选择 l>0 后再固定取某个 q 分量，最终输出一般不再是旋转标量。
"""

import unittest

import jax.numpy as jnp
import numpy as np

from so3_test._helpers import BASES, ROTATION, signed_log_value, wavefunction_case


class ScalarWavefunctionRotationTest(unittest.TestCase):
    def test_full_wavefunction_is_scalar(self):
        """分别比较原构型和共同旋转后构型的相位及对数模。"""
        for basis in BASES:
            with self.subTest(basis=basis):
                # wavefunction_case 的默认 target_ls=(0,)，但网络内部 lmax=2。
                # 因而仍会经过高阶特征的生成和耦合，再读出标量扇区。
                params, network, electrons, atoms = wavefunction_case(basis)
                phase, logabs = signed_log_value(network, params, electrons, atoms)
                # 坐标数组按行存储；列向量的 R 作用写成 rows @ R.T。
                # 两类坐标必须使用同一个 R，才能保持所有相对几何关系。
                rotation = jnp.asarray(ROTATION.T)
                turned_phase, turned_logabs = signed_log_value(
                    network, params, electrons @ rotation, atoms @ rotation)
                # 对非零波函数，比较 (phase, log|psi|) 等价于比较其数值，
                # 且避免直接计算 exp(log|psi|) 时的数值溢出。
                np.testing.assert_allclose(turned_phase, phase, rtol=2e-9, atol=2e-9)
                np.testing.assert_allclose(turned_logabs, logabs, rtol=2e-9, atol=2e-9)


if __name__ == "__main__":
    unittest.main()
