"""整体平移电子和核时，波函数应不变。"""

import unittest

import jax.numpy as jnp
import numpy as np

from ._helpers import electron_positions, evaluate, nuclear_positions


class TranslationInvarianceTest(unittest.TestCase):
    def test_joint_translation_preserves_sign_and_logabs(self):
        """只检验同时移动所有粒子：相对位置不变，物理构型相同。"""
        electrons, ions = electron_positions(), nuclear_positions()
        reference = evaluate(electrons, ions)
        # 使用非轴向位移，以覆盖三个坐标方向。
        shift = jnp.asarray((0.31, -0.27, 0.19), dtype=jnp.float64)
        translated = evaluate(electrons + shift, ions + shift)
        for actual, expected in zip(translated, reference):
            np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)


if __name__ == "__main__":
    unittest.main()
