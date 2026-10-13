"""检查高阶角向边特征在电子接近原子核时的连续性。

若边特征是 e^{-beta*r} Y_l(r_hat)，l>0 时从不同方向靠近 r=0
会得到不同极限。乘上近原点的 r^l 因子后，角向项应消失。
这个测试比较完整随机波函数在两个趋近方向上的值，而不只检查边函数。
"""

import unittest

import jax.numpy as jnp
import numpy as np

from so3_test._helpers import (
    BASES, assert_multiplet_rotation, signed_log_value, wavefunction_case,
)


class CoalescenceTest(unittest.TestCase):
    def test_regularized_edges_remain_equivariant(self):
        """径向正则化是标量乘子，不应改变各阶特征的旋转规律。"""
        for basis in BASES:
            for l in (0, 1, 2):
                with self.subTest(basis=basis, l=l):
                    assert_multiplet_rotation(
                        self, l, basis, regularize_angular_origin=True)

    def test_wavefunction_has_direction_independent_limit(self):
        """电子趋近同一原子核时，两个方向的波函数差应趋于零。"""
        params, network, electrons, atoms = wavefunction_case(
            "real", target_ls=(0,), regularize_angular_origin=True)
        directions = (jnp.array((1., 0., 0.)), jnp.array((0., 1., 0.)))
        differences = []
        for distance in (1e-3, 1e-4):
            values = []
            for direction in directions:
                moved = electrons.at[0].set(atoms[0] + distance * direction)
                phase, logabs = signed_log_value(network, params, moved, atoms)
                # 缩放后的实际波函数值；两方向使用同一参考指数，避免溢出。
                values.append((phase, logabs))
            scale = max(logabs for _, logabs in values)
            scaled = [phase * np.exp(logabs - scale) for phase, logabs in values]
            differences.append(abs(scaled[0] - scaled[1]))

        # 距离缩小十倍后，方向差也应明显缩小；未正则化时它趋向常数。
        self.assertLess(differences[1], differences[0] * 0.2)
        self.assertLess(differences[1], 0.01)


if __name__ == "__main__":
    unittest.main()
