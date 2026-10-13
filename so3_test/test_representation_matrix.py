"""用多个旋转检查从球谐取值反推的 l=1、2 表示矩阵。

解析参照来自笛卡尔向量和对称无迹二阶张量，不依赖生产代码的
球谐函数或基变换。除矩阵元本身，还检查群乘法关系
D_l(R_z R_y)=D_l(R_z)D_l(R_y)。有限数量的旋转不能构成数学证明，
但能排除只对某一个特定旋转碰巧成立的实现。
"""

import math
import unittest

import numpy as np

from so3_test._helpers import (
    analytic_representation_matrix, representation_matrix,
)


class RepresentationMatrixTest(unittest.TestCase):
    def test_identity_axes_and_composition(self):
        """逐项核对单位、y 轴、z 轴和两轴复合旋转的表示矩阵。"""
        a, b = 0.37, -0.58
        ca, sa = math.cos(a), math.sin(a)
        cb, sb = math.cos(b), math.sin(b)
        rz = np.array(((ca, -sa, 0.), (sa, ca, 0.), (0., 0., 1.)))
        ry = np.array(((cb, 0., sb), (0., 1., 0.), (-sb, 0., cb)))

        for l in (1, 2):
            for basis in ("real", "complex"):
                with self.subTest(l=l, basis=basis):
                    matrices = {
                        "identity": representation_matrix(l, basis, np.eye(3)),
                        "z": representation_matrix(l, basis, rz),
                        "y": representation_matrix(l, basis, ry),
                        "z_after_y": representation_matrix(l, basis, rz @ ry),
                    }
                    for name, rotation in (
                        ("identity", np.eye(3)),
                        ("z", rz),
                        ("y", ry),
                        ("z_after_y", rz @ ry),
                    ):
                        # 每个矩阵元都与独立的笛卡尔/张量解析结果比较。
                        np.testing.assert_allclose(
                            matrices[name],
                            analytic_representation_matrix(l, basis, rotation),
                            rtol=1e-12, atol=1e-12,
                            err_msg=f"l={l}, basis={basis}, rotation={name}",
                        )

                    # 对主动旋转，先 Ry 后 Rz 对应 Rz@Ry；同阶表示
                    # 应保持这个乘法顺序，而不只是分别对单轴旋转正确。
                    np.testing.assert_allclose(
                        matrices["z_after_y"], matrices["z"] @ matrices["y"],
                        rtol=1e-12, atol=1e-12,
                    )


if __name__ == "__main__":
    unittest.main()
