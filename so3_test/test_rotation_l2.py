"""检验 l=2 五分量表示的旋转性质。

旋转会在同一个 l=2 多重态的五个分量之间混合，不应把它们
变成其他 l 阶。正交/酉表示还应保持分量向量的范数。
"""

import unittest

import numpy as np

from so3_test._helpers import (
    BASES, analytic_representation_matrix, assert_multiplet_rotation,
    representation_matrix,
)


class RotationL2Test(unittest.TestCase):
    def test_rotation_matches_cartesian_tensors(self):
        """把数值反推的 D_2 与独立的对称无迹张量表示逐项比较。"""
        # 这一步检查 D_2 的具体矩阵元；仅检查酉性远远不够，
        # 例如错误的单位矩阵也满足 D†D=I。
        for basis in ("real", "complex"):
            with self.subTest(basis=basis):
                np.testing.assert_allclose(
                    representation_matrix(2, basis),
                    analytic_representation_matrix(2, basis),
                    rtol=1e-12, atol=1e-12)

    def test_rotation_matrix_is_unitary(self):
        """检查从球谐探针求出的 D_2 是否保持五维向量的内积。"""
        for basis in ("real", "complex"):
            with self.subTest(basis=basis):
                matrix = representation_matrix(2, basis)
                # 对实基这是正交条件 D^T D=I；对复基是酉条件 D†D=I。
                # conj().T 同时适用于这两种情况。
                np.testing.assert_allclose(
                    matrix.conj().T @ matrix, np.eye(5),
                    rtol=1e-12, atol=1e-12)

    def test_quadrupole_multiplets(self):
        """检查三种模式下网络各阶段的五分量特征是否等变。"""
        for basis in BASES:
            with self.subTest(basis=basis):
                assert_multiplet_rotation(self, 2, basis)


if __name__ == "__main__":
    unittest.main()
