"""与 QMC 动能相关的坐标导数检查。"""

import unittest

import jax
import jax.numpy as jnp

from ._helpers import electron_positions, nuclear_positions, tiny_network


class CoordinateDerivativeTest(unittest.TestCase):
    def test_forward_coordinate_derivative_is_finite(self):
        """前向求导在普通构型上应给出有限的一阶导数。"""
        params, network = tiny_network("all")
        ions = nuclear_positions()
        derivative = jax.jacfwd(lambda x: network(params, x, ions)[1])(
            electron_positions())
        self.assertTrue(bool(jnp.all(jnp.isfinite(derivative))))

    def test_reverse_coordinate_gradient_is_finite(self):
        """回传梯度也应有限；当前 all 缩放分支会使此测试失败。"""
        params, network = tiny_network("all")
        ions = nuclear_positions()
        gradient = jax.grad(lambda x: network(params, x, ions)[1])(
            electron_positions())
        self.assertTrue(bool(jnp.all(jnp.isfinite(gradient))),
                        "rescale_input='all' 在普通构型上产生 NaN 坐标梯度")

    def test_all_eps_control_has_finite_reverse_gradient(self):
        """对照现有 all_eps 分支，以帮助定位 all 分支中的 0/0。"""
        params, network = tiny_network("all_eps")
        ions = nuclear_positions()
        gradient = jax.grad(lambda x: network(params, x, ions)[1])(
            electron_positions())
        self.assertTrue(bool(jnp.all(jnp.isfinite(gradient))))


if __name__ == "__main__":
    unittest.main()
