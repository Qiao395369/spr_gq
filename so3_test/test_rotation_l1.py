"""检验 l=1 三分量表示的旋转性质。

实球谐的 l=1 与笛卡尔坐标 (x,y,z) 等价，只是分量顺序不同；
所以既要检查这个已知的笛卡尔对应，也要检查网络产生的 l=1 特征。
"""

import unittest

import numpy as np

from so3_test._helpers import (
    BASES, ROTATION, analytic_representation_matrix,
    assert_multiplet_rotation, representation_matrix,
)


class RotationL1Test(unittest.TestCase):
    def test_vector_basis_order(self):
        """用笛卡尔旋转矩阵独立核对实基 l=1 的分量顺序。"""
        # 项目实球谐 l=1 的分量与 [y,z,x] 成比例。
        # P 把笛卡尔列向量 [x,y,z] 改排为 [y,z,x]。
        permutation = np.array(((0., 1., 0.), (0., 0., 1.), (1., 0., 0.)))
        # 若 v_real=P v_xyz，而 v_xyz'=R v_xyz，
        # 则 v_real'=(P R P^T) v_real；P^T 是 P 的逆矩阵。
        expected = permutation @ ROTATION @ permutation.T
        np.testing.assert_allclose(
            representation_matrix(1, "real"), expected, rtol=1e-12, atol=1e-12)

    def test_complex_basis_rotation(self):
        """复基也要满足独立推导的 l=1 旋转矩阵。"""
        # 先由笛卡尔 R 求实基矩阵，再用明写的复/实球谐关系换基。
        # 不调用生产代码的 spherical_harmonics 或 basis_change_matrix。
        expected = analytic_representation_matrix(1, "complex")
        np.testing.assert_allclose(
            representation_matrix(1, "complex"), expected,
            rtol=1e-12, atol=1e-12)

    def test_vector_multiplets(self):
        """检查主干、基转换后特征和读出都按 l=1 表示旋转。"""
        for basis in BASES:
            with self.subTest(basis=basis):
                assert_multiplet_rotation(self, 1, basis)


if __name__ == "__main__":
    unittest.main()
