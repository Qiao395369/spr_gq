"""检验完整波函数对同自旋电子交换的反对称性。

费米子波函数满足 psi(...,r_i,...,r_j,...)
                 = -psi(...,r_j,...,r_i,...)，前提是 i、j 属于同一自旋组。
这里分别交换两个上自旋电子和两个下自旋电子；核坐标不动。
"""

import unittest

import numpy as np

from so3_test._helpers import BASES, signed_log_value, wavefunction_case


class ExchangeAntisymmetryTest(unittest.TestCase):
    def _check_swaps(self, basis: str, det_bank_mode: str, target_ls):
        """在固定参数和构型下，逐组验证一次换位使波函数乘以 -1。"""
        params, network, electrons, atoms = wavefunction_case(
            basis, det_bank_mode, target_ls)
        phase, logabs = signed_log_value(network, params, electrons, atoms)
        # nspins=(2,2)：数组前两行是上自旋，后两行是下自旋。
        # (1,0,2,3) 仅交换上自旋；(0,1,3,2) 仅交换下自旋。
        for label, permutation in (
            ("up", (1, 0, 2, 3)),
            ("down", (0, 1, 3, 2)),
        ):
            with self.subTest(basis=basis, det_bank_mode=det_bank_mode,
                              target_ls=target_ls, swap=label):
                # 只重排电子坐标，不修改网络参数、电子自旋数或核坐标。
                swapped_phase, swapped_logabs = signed_log_value(
                    network, params, electrons[np.asarray(permutation)], atoms)
                # psi -> -psi：符号/相位反转，绝对值及其对数保持不变。
                np.testing.assert_allclose(swapped_phase, -phase, rtol=2e-9, atol=2e-9)
                np.testing.assert_allclose(swapped_logabs, logabs, rtol=2e-9, atol=2e-9)

    def test_all_basis_modes_with_shared_determinants(self):
        """多个 l 共用行列式参数时，三种 basis 模式都应保持反对称。"""
        for basis in BASES:
            self._check_swaps(basis, "shared", (0, 1, 2))

    def test_independent_determinant_banks(self):
        """每个 l 使用独立行列式参数时，也检查同自旋交换。"""
        self._check_swaps("real", "per_l", (0, 1, 2))

    def test_scalar_sector(self):
        """仅用 l=0 输出的最简单配置同样必须满足反对称性。"""
        self._check_swaps("real", "shared", (0,))


if __name__ == "__main__":
    unittest.main()
