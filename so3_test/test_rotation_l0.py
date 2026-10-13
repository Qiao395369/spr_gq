"""检验 l=0 标量表示的旋转性质。

l=0 只有一个角分量，D_0(R)=[1]。因此旋转所有粒子坐标后，
该阶特征的数值应保持不变；这个结论对三种 basis 模式都成立。
"""

import unittest

import numpy as np

from so3_test._helpers import BASES, assert_multiplet_rotation, representation_matrix


class RotationL0Test(unittest.TestCase):
    def test_scalar_multiplets(self):
        """同时检查标量的表示矩阵，以及网络内部/读出的标量特征。"""
        for basis in BASES:
            with self.subTest(basis=basis):
                # 先确认用作参照的 D_0 确实是恒等变换。
                np.testing.assert_allclose(representation_matrix(0, "real"), [[1.]])
                # 再比较同一套网络参数在旋转前后的 h、y、A_channels 和 A。
                assert_multiplet_rotation(self, 0, basis)


if __name__ == "__main__":
    unittest.main()
