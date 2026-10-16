"""基本输入和输出检查：在讨论对称性前先确认测试构型有效。"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from ._helpers import (CHARGES, N_ELECTRONS, NSPINS, electron_positions,
                       evaluate, nuclear_positions, tiny_network)


class WavefunctionValueTest(unittest.TestCase):
    def test_n2_system_is_neutral_and_has_consistent_shapes(self):
        """两颗核总电荷 14，电子也为 7+7=14。"""
        self.assertEqual(sum(CHARGES), N_ELECTRONS)
        self.assertEqual(sum(NSPINS), N_ELECTRONS)
        self.assertEqual(electron_positions().shape, (14, 3))
        self.assertEqual(nuclear_positions().shape, (2, 3))

    def test_generic_wavefunction_is_real_finite_and_float64(self):
        """有效构型上的实值波函数必须有可用的符号与对数模长。"""
        phase, logabs = evaluate()
        self.assertEqual(phase.dtype, jnp.float64)
        self.assertEqual(logabs.dtype, jnp.float64)
        self.assertTrue(bool(jnp.isfinite(phase)))
        self.assertTrue(bool(jnp.isfinite(logabs)))
        self.assertEqual(abs(float(phase)), 1.)

    def test_fixture_is_small_and_depends_on_electron_positions(self):
        """防止误用正式大网络，也防止常数输出让对称性测试空过。"""
        params, _ = tiny_network()
        nparams = sum(int(x.size) for x in jax.tree_util.tree_leaves(params))
        self.assertLess(nparams, 5000)
        electrons = electron_positions()
        moved = electrons.at[0, 0].add(0.17)
        original_logabs = float(evaluate(electrons)[1])
        moved_logabs = float(evaluate(moved)[1])
        self.assertGreater(abs(original_logabs - moved_logabs), 1e-7)


if __name__ == "__main__":
    unittest.main()
