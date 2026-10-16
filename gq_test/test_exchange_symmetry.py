"""同自旋电子交换反对称性和相同原子核换序不变性。"""

import unittest

import jax.numpy as jnp
import numpy as np

from ._helpers import electron_positions, evaluate, nuclear_positions


def permute_rows(array, permutation):
    """只重排标签，不改变任一粒子的坐标。"""
    return array[jnp.asarray(permutation)]


class ExchangeSymmetryTest(unittest.TestCase):
    def assert_exchange_sign(self, permutation, expected_sign):
        original_phase, original_logabs = evaluate()
        electrons = permute_rows(electron_positions(), permutation)
        changed_phase, changed_logabs = evaluate(electrons)
        # 对数模长描述 |ψ|；一次同自旋交换只应改变实波函数符号。
        np.testing.assert_allclose(changed_logabs, original_logabs,
                                   rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(changed_phase, expected_sign * original_phase,
                                   rtol=0, atol=0)

    def test_two_up_electrons_flip_sign(self):
        permutation = list(range(14))
        permutation[0], permutation[1] = permutation[1], permutation[0]
        self.assert_exchange_sign(permutation, -1)

    def test_two_down_electrons_flip_sign(self):
        permutation = list(range(14))
        permutation[7], permutation[8] = permutation[8], permutation[7]
        self.assert_exchange_sign(permutation, -1)

    def test_two_swaps_restore_sign(self):
        """上、下自旋块各做一次交换，是偶置换。"""
        permutation = list(range(14))
        permutation[0], permutation[1] = permutation[1], permutation[0]
        permutation[7], permutation[8] = permutation[8], permutation[7]
        self.assert_exchange_sign(permutation, +1)

    def test_identical_nuclei_can_be_relabelled(self):
        """两颗核都是 Z=7，交换数组行号不应改变波函数。"""
        ions = nuclear_positions()
        reference = evaluate(ions=ions)
        exchanged = evaluate(ions=ions[::-1])
        for actual, expected in zip(exchanged, reference):
            np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)


if __name__ == "__main__":
    unittest.main()
