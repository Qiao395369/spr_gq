"""原版 SO(3) 主干的逐电子 MLP 读出消融。

检查的重点是：不再有 A/rank，layers=1 确实只有线性映射，各角分量
都进入轨道；失去旋转约束后，电子交换反对称性与平移不变性仍保持。
还实际经过轨道 Adam 更新、VMC 构建和坐标二阶导数，避免只测形状。
"""
import contextlib
import dataclasses
import io
import unittest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict

from vmcnet.gaoqiao import dp, so3_utils
from vmcnet.train.default_config import get_default_config
from vmcnet.train.pretrain_from_trained_net import (
    _fold_student_log_scale, construct_network, validate_configs)
from vmcnet.train.pretrain_teacher_orbitals import (
    construct_orbital_network, make_train_step, validate_orbital_types)
from so3_test._helpers import ROTATION, representation_matrix


class MLPReadoutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ions = jnp.array(((0., 0., -.8), (.2, .1, .8)))
        cls.charges = jnp.array((2., 1.))
        cls.spins = (2, 1)
        cls.electrons = jnp.array(((.3, .4, -.4), (-.5, .2, .1), (.4, -.3, .9)))
        cls.types = dp.ManyElectronSystem(cls.charges, cls.spins, "full").get_part_features()
        cls.config = so3_utils.SO3Config(
            readout_mode="mlp", readout_mlp_layers=1,
            channels=3, lmax=2, n_layers=1, n_rbf=2, wbn_layers=1,
            regularize_angular_origin=True, float_dtype=jnp.float64)
        cls.params, net, dist, orb, _ = so3_utils.make_mlp_network(
            jax.random.PRNGKey(73), cls.charges, cls.spins, 2, cls.types, cls.config)
        cls.network, cls.dist, cls.orbitals = map(staticmethod, (net, dist, orb))

    def test_single_layer_is_exactly_flat_features_times_weight(self):
        # 三阶块分别有 C、3C、5C 个分量；最终总共 9C 个输入。
        self.assertEqual(self.params["orbital"][0]["w"].shape, (27, 6))
        self.assertEqual(self.params["readout_hidden"], ())
        self.assertNotIn("readout", self.params["so3"])
        self.assertNotIn("so3_det", self.params)
        cfg = dataclasses.replace(self.config, use_kato_jastrow=False)
        out = so3_utils.evaluate_wavefunction(
            self.params, self.electrons, self.ions, self.types, self.spins, 2, cfg)
        self.assertNotIn("A", out["backbone"])
        self.assertNotIn("multiplet_phase", out)
        expected_flat = jnp.concatenate([
            out["backbone"]["y"][l][:3].reshape((3, -1)) for l in range(3)], axis=-1)
        np.testing.assert_array_equal(out["features"], expected_flat)
        # 两组自旋独立轨道头，同组电子共享；包络只在最终输出上相乘。
        blocks = []
        for x, pos, head, env in zip(
                jnp.split(expected_flat, (2,)), jnp.split(self.electrons, (2,)),
                self.params["orbital"], self.params["envelope"]):
            raw = (x @ head["w"]).reshape((len(x), 2, 3))
            env_values = so3_utils.apply_orbital_envelope(
                pos, self.ions, env, cfg.envelope_sigma_floor, cfg.eps)
            blocks.append((raw * env_values).transpose((1, 0, 2)))
        np.testing.assert_allclose(out["orbitals"][0], jnp.concatenate(blocks, axis=1))

    def test_jastrow_and_determinant_count(self):
        matrix = self.orbitals(self.params, self.electrons, self.ions)[0]
        self.assertEqual(matrix.shape, (2, 3, 3))  # ndet，没有 rank 维。
        value = jnp.sum(jnp.linalg.det(matrix))
        sign, logabs = self.network(self.params, self.electrons, self.ions)
        np.testing.assert_allclose(sign * jnp.exp(logabs), value, rtol=1e-10)
        without = so3_utils.evaluate_wavefunction(
            self.params, self.electrons, self.ions, self.types, self.spins, 2,
            dataclasses.replace(self.config, use_kato_jastrow=False))
        log_j = so3_utils.kato_ee_jastrow(
            self.electrons, self.spins, self.params["so3_jastrow"], self.config.eps)
        np.testing.assert_allclose(matrix, without["orbitals"][0] * jnp.exp(log_j / 3))
        self.assertAlmostEqual(float(logabs - without["logabs"]), float(log_j), places=10)
        self.assertEqual(self.dist(self.params, self.electrons, self.ions).shape, (2,))
        # 这些旧读出选项均不应改变新模式的结构或数值，也不校验它们的旧规则。
        ignored = dataclasses.replace(self.config, det_rank=0, target_ls=(),
                                      det_bank_mode="unused", output_projection="unused",
                                      use_invariant_orbital_features=False)
        p = so3_utils.init_wavefunction_params(
            jax.random.PRNGKey(73), self.types.shape[-1], self.spins, 2, 2, ignored)
        for a, b in zip(jax.tree_util.tree_leaves(self.params), jax.tree_util.tree_leaves(p)):
            np.testing.assert_array_equal(a, b)

    def test_translation_exchange_and_rotation_ablation(self):
        matrix = self.orbitals(self.params, self.electrons, self.ions)[0]
        shifted = self.orbitals(self.params, self.electrons + 1.3, self.ions + 1.3)[0]
        np.testing.assert_allclose(shifted, matrix, rtol=1e-10, atol=1e-10)
        permutation = np.array((1, 0, 2))
        swapped = self.orbitals(self.params, self.electrons[permutation], self.ions)[0]
        np.testing.assert_allclose(swapped, matrix[:, permutation], rtol=1e-10, atol=1e-10)
        s, log = self.network(self.params, self.electrons, self.ions)
        sp, logp = self.network(self.params, self.electrons[permutation], self.ions)
        self.assertNotEqual(float(s), 0.)
        self.assertEqual(float(sp), -float(s))
        self.assertAlmostEqual(float(log), float(logp), places=9)
        rotated = self.orbitals(self.params, self.electrons @ ROTATION.T,
                                self.ions @ ROTATION.T)[0]
        # 固定的通用参数应确实对方向敏感，不能悄悄退回 RMS 或 l=0。
        self.assertGreater(float(jnp.max(jnp.abs(rotated - matrix))), 1e-6)
        positions = jnp.concatenate((self.electrons, self.ions))
        original_h = so3_utils.apply_so3_network(
            self.params["so3"], positions, self.types, self.config)["h"]
        rotated_h = so3_utils.apply_so3_network(
            self.params["so3"], positions @ ROTATION.T, self.types, self.config)["h"]
        for l in range(3):
            d = representation_matrix(l, "real")
            np.testing.assert_allclose(rotated_h[l], original_h[l] @ d.T,
                                       rtol=1e-10, atol=1e-10)

    def test_layer_count_and_all_basis_modes(self):
        # layers=3 意味着两个非线性隐藏层加一个轨道输出层。
        for basis in ("real", "complex", "complex_to_real"):
            cfg = dataclasses.replace(self.config, lmax=1, basis=basis,
                                      readout_mlp_layers=3, readout_mlp_width=5)
            p, net, _, orb, _ = so3_utils.make_mlp_network(
                jax.random.PRNGKey(9), self.charges, self.spins, 2, self.types, cfg)
            f = 12 if basis == "real" else 24
            self.assertEqual(len(p["readout_hidden"]), 2)
            self.assertEqual(p["readout_hidden"][0]["w"].shape, (f, 5))
            self.assertEqual(p["orbital"][0]["w"].shape, (5, 6))
            values = orb(p, self.electrons, self.ions)[0]
            self.assertFalse(jnp.iscomplexobj(values))
            self.assertTrue(bool(jnp.isfinite(net(p, self.electrons, self.ions)[1])))
            self.assertNotIn("readout", p["so3"])
        with self.assertRaisesRegex(ValueError, "readout_mlp_layers"):
            so3_utils.readout_mode(dataclasses.replace(self.config, readout_mlp_layers=0))
        with self.assertRaisesRegex(ValueError, "readout_mode"):
            so3_utils.readout_mode(dataclasses.replace(self.config, readout_mode="typo"))

    def test_lmax_zero_empty_spin_and_spin_factored_determinants(self):
        cfg = dataclasses.replace(self.config, lmax=0)
        types = dp.ManyElectronSystem(self.charges, (1, 0), "full").get_part_features()
        p = so3_utils.init_wavefunction_params(
            jax.random.PRNGKey(4), types.shape[-1], (1, 0), 2, 3, cfg)
        out = so3_utils.evaluate_wavefunction(
            p, self.electrons[:1], self.ions, types, (1, 0), 3, cfg)
        self.assertEqual(out["orbitals"][0].shape, (3, 1, 1))
        self.assertTrue(bool(jnp.isfinite(out["logabs"])))
        p = so3_utils.init_wavefunction_params(
            jax.random.PRNGKey(5), self.types.shape[-1], self.spins, 2, 2,
            self.config, full_det=False)
        out = so3_utils.evaluate_wavefunction(
            p, self.electrons, self.ions, self.types, self.spins, 2,
            self.config, full_det=False)
        value = jnp.sum(jnp.linalg.det(out["orbitals"][0])
                        * jnp.linalg.det(out["orbitals"][1]))
        np.testing.assert_allclose(out["phase"] * jnp.exp(out["logabs"]), value, rtol=1e-10)

    def test_parameter_gradient_and_coordinate_second_derivative(self):
        logpsi = lambda p, x: self.network(p, x, self.ions)[1]
        g = jax.grad(logpsi)(self.params, self.electrons)
        self.assertTrue(all(bool(jnp.all(jnp.isfinite(a)))
                            for a in jax.tree_util.tree_leaves(g)))
        gx = jax.grad(logpsi, argnums=1)(self.params, self.electrons)
        h = jax.hessian(lambda x: logpsi(self.params, x))(self.electrons)
        kinetic = -.5 * (jnp.trace(h.reshape((9, 9))) + jnp.sum(gx * gx))
        self.assertTrue(bool(jnp.isfinite(kinetic)))

    def test_orbital_pretrain_scale_save_and_vmc_entry(self):
        teacher = get_default_config()
        teacher.dtype = "float64"
        teacher.gq.ndet, teacher.gq.wfn_depth = 2, 2
        teacher.gq.h1, teacher.gq.h2, teacher.gq.nh = 4, 2, 2
        teacher.gq.type, teacher.gq.jastrow_type = "ef_shrd_sym", "null"
        teacher.gq.envelope_type = "iso"
        teacher.problem.ion_pos = tuple(map(tuple, np.asarray(self.ions)))
        teacher.problem.ion_charges = (2., 1.)
        teacher.problem.nspins = self.spins
        teacher.problem.single_nspins = ((1, 1), (1, 0))
        teacher.problem.atoms_symbol = ("He", "H")
        student = ConfigDict(teacher.to_dict())
        student.gq.type = "so3"
        cfg = student.gq.so3_params
        cfg.readout_mode, cfg.basis = "mlp", "complex"
        cfg.channels, cfg.lmax, cfg.n_layers, cfg.n_rbf, cfg.wbn_layers = 2, 1, 1, 2, 1
        cfg.readout_mlp_layers, cfg.readout_mlp_width = 2, 4
        cfg.regularize_angular_origin = True
        # rank=0 证明新模式没有借用旧 rank；RHF 被忽略，尺度必须落在 orbital。
        cfg.det_rank, student.gq.RHF = 0, True
        validate_orbital_types(teacher, student)
        molecule = validate_configs(teacher, student)
        key = jax.random.PRNGKey(41)
        p, net, orb = construct_orbital_network(student, molecule, key)
        tp, _, torb = construct_orbital_network(teacher, molecule, jax.random.PRNGKey(42))
        points = jnp.stack((self.electrons, self.electrons + .03))
        targets = jax.vmap(torb, in_axes=(None, 0, None))(tp, points, self.ions)
        for loss_mode in ("l2", "fulldet"):
            optimizer, step, _ = make_train_step(orb, self.ions, loss_mode, 1e-4)
            updated, _, loss, diagnostics = step(p, optimizer.init(p), points, targets)
            self.assertTrue(bool(jnp.isfinite(loss)))
            self.assertTrue(bool(diagnostics["gradient_finite"]))
            self.assertTrue(any(not np.array_equal(a, b) for a, b in zip(
                jax.tree_util.tree_leaves(p), jax.tree_util.tree_leaves(updated))))
        other_p, other_net = construct_network(student, molecule, key)
        np.testing.assert_allclose(net(p, self.electrons, self.ions)[1],
                                   other_net(other_p, self.electrons, self.ions)[1])
        scaled = _fold_student_log_scale(p, jnp.log(3.), student)
        for a, b in zip(jax.tree_util.tree_leaves(p["readout_hidden"]),
                        jax.tree_util.tree_leaves(scaled["readout_hidden"])):
            np.testing.assert_array_equal(a, b)
        self.assertAlmostEqual(float(net(scaled, self.electrons, self.ions)[1]
                                     - net(p, self.electrons, self.ions)[1]),
                               float(jnp.log(3.)), places=9)
        from vmcnet.train.runners import _get_gaoqiao_model
        with contextlib.redirect_stdout(io.StringIO()):
            built = _get_gaoqiao_model(student.gq, jnp.float64, "gaoqiao", 3,
                                       self.charges, self.spins, key, False)
        self.assertEqual(jax.tree_util.tree_structure(built[6]), jax.tree_util.tree_structure(p))
        self.assertTrue(bool(jnp.isfinite(built[2](built[6], self.ions, self.electrons))))

    def tearDown(self):
        jax.clear_caches()


if __name__ == "__main__":
    unittest.main()
