"""so3_l0 的结构、轨道语义和训练接入回归测试。

测试小分子构型，避免正式训练开销。重点核对：Conv/WBN 交替、
末层只输出零阶、没有 A/rank、Jastrow 只乘一次，以及实际轨道监督
和动能所需的自动微分，而不只检查函数能否被导入。
"""
import dataclasses
import unittest
from unittest.mock import patch

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict

from vmcnet.gaoqiao import so3_l0, so3_utils, dp
from vmcnet.gaoqiao.build import build_network
from vmcnet.train.default_config import get_default_config
from vmcnet.train.pretrain_from_trained_net import (
    construct_network, validate_configs, _fold_student_log_scale)
from vmcnet.train.pretrain_teacher_orbitals import (
    construct_orbital_network, make_train_step, validate_orbital_types)


class SO3L0Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ions = jnp.array(((0., 0., -.8), (.2, .1, .8)))
        cls.charges = jnp.array((2., 1.))
        cls.spins = (2, 1)
        cls.positions = jnp.array(((.3, .4, -.4), (-.5, .2, .1),
                                   (.4, -.3, .9)))
        cls.types = dp.ManyElectronSystem(
            cls.charges, cls.spins, "full").get_part_features()
        cls.config = so3_l0.SO3L0Config(
            lmax=2, channels=4, n_layers=2, n_rbf=2, float_dtype=jnp.float64)
        cls.params, network, dist, orbitals, _ = so3_l0.make_network(
            jax.random.PRNGKey(71), cls.charges, cls.spins, 2,
            cls.types, cls.config)
        # 函数存为 staticmethod，防止 unittest 实例访问时被绑定额外的 self。
        cls.network = staticmethod(network)
        cls.dist = staticmethod(dist)
        cls.orbitals = staticmethod(orbitals)

    def test_layers_are_interleaved_and_last_wbn_keeps_only_zero(self):
        layers = self.params["backbone"]["layers"]
        self.assertEqual(len(layers), 2)
        self.assertEqual(set(layers[-1]["conv"]["self"]), {"0", "1", "2"})
        self.assertEqual(set(layers[-1]["wbn"]["self"]), {"0"})
        self.assertIn("2_2_0", layers[-1]["wbn"]["paths"])
        self.assertTrue(all(path.endswith("_0") for path in layers[-1]["wbn"]["paths"]))
        self.assertNotIn("readout", self.params["backbone"])
        self.assertNotIn("so3_det", self.params)
        self.assertNotIn("alpha", self.params)
        events = []
        conv, wbn = so3_utils._convolution, so3_l0._wbn_update

        def record_conv(*args, **kwargs):
            events.append("conv")
            return conv(*args, **kwargs)

        def record_wbn(*args, **kwargs):
            events.append("wbn")
            return wbn(*args, **kwargs)

        with patch.object(so3_utils, "_convolution", side_effect=record_conv), \
             patch.object(so3_l0, "_wbn_update", side_effect=record_wbn):
            y = so3_l0.apply_backbone(
                self.params["backbone"],
                jnp.concatenate((self.positions, self.ions)), self.types, self.config)
        self.assertEqual(events, ["conv", "wbn", "conv", "wbn"])
        self.assertEqual(set(y), {0})
        self.assertEqual(y[0].shape, (5, 4, 1))

    def test_orbital_interface_and_jastrow_are_exactly_the_wavefunction(self):
        matrices = self.orbitals(self.params, self.positions, self.ions)[0]
        self.assertEqual(matrices.shape, (2, 3, 3))
        raw = so3_l0.orbital_matrices(
            self.params, self.positions, self.ions, self.types, self.spins, 2,
            dataclasses.replace(self.config, use_kato_jastrow=False))
        log_j = so3_utils.kato_ee_jastrow(
            self.positions, self.spins, self.params["jastrow"], self.config.eps)
        np.testing.assert_allclose(matrices, raw * jnp.exp(log_j / 3), rtol=1e-12)
        sign, logabs = self.network(self.params, self.positions, self.ions)
        expected = jnp.sum(jnp.linalg.det(matrices))
        np.testing.assert_allclose(sign * jnp.exp(logabs), expected, rtol=1e-10)
        raw_sign, raw_log = so3_utils.phase_logsumexp(*jnp.linalg.slogdet(raw))
        self.assertEqual(float(sign), float(raw_sign))
        self.assertAlmostEqual(float(logabs - raw_log), float(log_j), places=9)
        self.assertEqual(self.dist(self.params, self.positions, self.ions).shape, (2,))

    def test_joint_rotation_translation_and_same_spin_exchange(self):
        # 通用轴角转动，三个轴都有混合；直接检查轨道，比只看最终和更强。
        axis = np.array((1., 2., -1.))
        axis /= np.linalg.norm(axis)
        skew = np.array(((0., -axis[2], axis[1]),
                         (axis[2], 0., -axis[0]),
                         (-axis[1], axis[0], 0.)))
        rotation = np.eye(3) + np.sin(.67) * skew + (1 - np.cos(.67)) * (skew @ skew)
        matrix = self.orbitals(self.params, self.positions, self.ions)[0]
        transformed = self.orbitals(
            self.params, self.positions @ rotation.T + 1.3,
            self.ions @ rotation.T + 1.3)[0]
        np.testing.assert_allclose(matrix, transformed, rtol=2e-10, atol=1e-10)
        perm = np.array((1, 0, 2))
        exchanged = self.orbitals(self.params, self.positions[perm], self.ions)[0]
        np.testing.assert_allclose(exchanged, matrix[:, perm, :], rtol=1e-11, atol=1e-11)
        s, l = self.network(self.params, self.positions, self.ions)
        sp, lp = self.network(self.params, self.positions[perm], self.ions)
        self.assertNotEqual(float(s), 0.)
        self.assertEqual(float(sp), -float(s))
        self.assertAlmostEqual(float(l), float(lp), places=9)

    def test_complex_backbones_still_return_real_zero_order_orbitals(self):
        for basis in ("complex", "complex_to_real"):
            cfg = dataclasses.replace(self.config, lmax=1, n_layers=1, basis=basis)
            p, network, _, orbitals, _ = so3_l0.make_network(
                jax.random.PRNGKey(9), self.charges, self.spins, 2, self.types, cfg)
            self.assertEqual(p["orbital"][0]["w"].shape, (8, 6))
            matrix = orbitals(p, self.positions, self.ions)[0]
            self.assertFalse(jnp.iscomplexobj(matrix))
            self.assertTrue(bool(jnp.all(jnp.isfinite(matrix))))
            _, logabs = network(p, self.positions, self.ions)
            self.assertTrue(bool(jnp.isfinite(logabs)))

    def test_empty_spin_group_and_lmax_zero(self):
        cfg = dataclasses.replace(self.config, lmax=0, n_layers=1)
        types = dp.ManyElectronSystem(self.charges, (1, 0), "full").get_part_features()
        p, network, _, orbitals, _ = so3_l0.make_network(
            jax.random.PRNGKey(8), self.charges, (1, 0), 3, types, cfg)
        self.assertEqual(len(p["orbital"]), 1)
        self.assertEqual(orbitals(p, self.positions[:1], self.ions)[0].shape, (3, 1, 1))
        self.assertTrue(bool(jnp.isfinite(network(p, self.positions[:1], self.ions)[1])))
        with self.assertRaisesRegex(ValueError, "n_layers"):
            so3_l0.init_backbone(jax.random.PRNGKey(0), 8,
                                 dataclasses.replace(cfg, n_layers=0))

    def test_finite_parameter_gradient_and_local_kinetic(self):
        # 动能需要坐标二阶导数；只检查前向有限无法发现反向或 Hessian 的 NaN。
        objective = lambda p: self.network(p, self.positions, self.ions)[1]
        gradient = jax.grad(objective)(self.params)
        self.assertTrue(all(bool(jnp.all(jnp.isfinite(x)))
                            for x in jax.tree_util.tree_leaves(gradient)))
        logpsi = lambda x: self.network(self.params, x, self.ions)[1]
        g = jax.grad(logpsi)(self.positions)
        h = jax.hessian(logpsi)(self.positions)
        lap = jnp.trace(h.reshape((9, 9)))
        kinetic = -.5 * (lap + jnp.sum(g * g))
        self.assertTrue(bool(jnp.isfinite(kinetic)))

    def _configs(self):
        teacher = get_default_config()
        teacher.dtype = "float64"
        teacher.gq.ndet = 2
        teacher.gq.h1, teacher.gq.h2, teacher.gq.nh = 4, 2, 2
        teacher.gq.wfn_depth = 2
        teacher.gq.jastrow_type = "null"
        teacher.gq.envelope_type = "iso"
        teacher.problem.ion_pos = tuple(map(tuple, np.asarray(self.ions)))
        teacher.problem.ion_charges = (2., 1.)
        teacher.problem.nspins = self.spins
        teacher.problem.single_nspins = ((1, 1), (1, 0))
        teacher.problem.atoms_symbol = ("He", "H")
        student = ConfigDict(teacher.to_dict())
        student.gq.type = "so3_l0"
        student.gq.so3_l0.channels = 4
        student.gq.so3_l0.lmax = 1
        student.gq.so3_l0.n_layers = 1
        student.gq.so3_l0.n_rbf = 2
        # 新网络不得读取旧 so3_params，也不对 student 做兼容性补全。
        del student.gq["so3_params"]
        return teacher, student

    def test_build_pretrain_adam_and_checkpoint_scale(self):
        teacher, student = self._configs()
        validate_orbital_types(teacher, student)
        molecule = validate_configs(teacher, student)
        p, network, orbitals = construct_orbital_network(
            student, molecule, jax.random.PRNGKey(41))
        teacher_params, _, teacher_orbitals = construct_orbital_network(
            teacher, molecule, jax.random.PRNGKey(42))
        targets = jnp.stack((
            teacher_orbitals(teacher_params, self.positions, self.ions),
            teacher_orbitals(teacher_params, self.positions + .03, self.ions)))
        self.assertEqual(p["orbital"][0]["w"].dtype, jnp.float64)
        points = jnp.stack((self.positions, self.positions + .03))
        for mode in ("l2", "fulldet"):
            with self.subTest(loss_mode=mode):
                optimizer, step, _ = make_train_step(
                    orbitals, self.ions, mode, 1e-4)
                updated, _, loss, diagnostics = step(
                    p, optimizer.init(p), points, targets)
                self.assertTrue(bool(jnp.isfinite(loss)))
                self.assertTrue(bool(diagnostics["gradient_finite"]))
                self.assertTrue(any(not np.array_equal(a, b) for a, b in zip(
                    jax.tree_util.tree_leaves(p), jax.tree_util.tree_leaves(updated))))
        # 完整波函数预训练/runner 的参数构造必须与轨道入口完全相同。
        other_p, other_network = construct_network(
            student, molecule, jax.random.PRNGKey(41))
        self.assertEqual(jax.tree_util.tree_structure(p),
                         jax.tree_util.tree_structure(other_p))
        np.testing.assert_allclose(
            network(p, self.positions, self.ions)[1],
            other_network(other_p, self.positions, self.ions)[1])
        scaled = _fold_student_log_scale(p, jnp.log(3.), student)
        self.assertAlmostEqual(float(
            network(scaled, self.positions, self.ions)[1]
            - network(p, self.positions, self.ions)[1]), float(jnp.log(3.)), places=9)
        legacy = ConfigDict(student.to_dict())
        legacy.gq.type = "so3"
        with self.assertRaisesRegex(NotImplementedError, "完整波函数"):
            validate_orbital_types(teacher, legacy)

    def tearDown(self):
        # 多种 JIT/高阶导数测试依次运行时释放编译缓存，降低 CPU 测试峰值内存。
        jax.clear_caches()


if __name__ == "__main__":
    unittest.main()
