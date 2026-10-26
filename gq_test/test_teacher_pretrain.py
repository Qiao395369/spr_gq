"""Teacher→student 预训练的损失和历史配置兼容性回归测试。"""

import unittest
import csv
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import optax
from ml_collections import ConfigDict
from absl import flags

from vmcnet.train.config_compat import complete_config
from vmcnet.train.default_config import get_default_config
from vmcnet.train import runners
from vmcnet.mcmc import dynamic_width_position_amplitude as dwpa
from vmcnet.train.pretrain_from_trained_net import (
    Molecule, _fold_so3_log_scale, _fold_student_log_scale, construct_network, initial_teacher_positions,
    check_checkpoint_amplitudes, evaluate_student_energy, make_energy_components,
    validate_problem_match, make_metropolis_sampler, signed_amplitude_loss, validate_configs,
    make_training_step, _check_adam_step)


class TeacherPretrainTest(unittest.TestCase):
    def test_standard_adam_keeps_update_when_current_batch_loss_rises(self):
        def network(params, positions, ions):
            return jnp.asarray(1.), params["w"]

        optimizer = optax.chain(optax.clip_by_global_norm(1.), optax.adam(0.01))
        step, objective, _ = make_training_step(
            network, network, jnp.zeros((1, 3)), "signed_mse", optimizer)
        trainable = {"network": {"w": jnp.asarray(0.)}, "log_scale": jnp.asarray(0.)}
        state = optimizer.init(trainable)
        positions = jnp.zeros((2, 1, 3))
        targets = (jnp.ones(2), jnp.ones(2))
        # 先积累指向增大幅度的动量，再换一批要求略微减小幅度的目标。
        for i in range(10):
            trainable, state, loss, _, diagnostics = step(trainable, state, positions, targets)
            _check_adam_step(loss, diagnostics, i + 1)
        target_log = trainable["network"]["w"] + trainable["log_scale"] - 0.001
        targets = (jnp.ones(2), jnp.full((2,), target_log))
        proposed, new_state, loss, _, diagnostics = step(trainable, state, positions, targets)
        after_loss, _ = objective(proposed, positions, targets)
        _check_adam_step(loss, diagnostics, 11)  # 损失上升不是数值异常。
        self.assertGreater(float(after_loss), float(loss))
        self.assertNotEqual(float(proposed["network"]["w"]), float(trainable["network"]["w"]))
        self.assertEqual(int(new_state[1][0].count), 11)

    def test_nonfinite_gradient_stops_even_if_forward_loss_is_finite(self):
        def network(params, positions, ions):
            return jnp.asarray(1.), jnp.sqrt(params["w"])

        optimizer = optax.chain(optax.clip_by_global_norm(1.), optax.adam(0.01))
        step, _, _ = make_training_step(
            network, network, jnp.zeros((1, 3)), "signed_mse", optimizer)
        trainable = {"network": {"w": jnp.asarray(0.)}, "log_scale": jnp.asarray(0.)}
        _, _, loss, _, diagnostics = step(
            trainable, optimizer.init(trainable), jnp.zeros((2, 1, 3)),
            (jnp.ones(2), jnp.ones(2)))
        self.assertTrue(bool(jnp.isfinite(loss)))
        with self.assertRaisesRegex(FloatingPointError, "step 3: 学生梯度"):
            _check_adam_step(loss, diagnostics, 3)

    def test_non_so3_students_keep_scale_and_checkpoint_structure(self):
        teacher = get_default_config()
        teacher.gq.type = "ef_shrd_sym"
        teacher.problem.nspins = (1, 1)
        teacher.problem.single_nspins = ((1, 0), (0, 1))
        teacher.problem.ion_charges = (1., 1.)
        teacher.problem.atoms_symbol = ("H", "H")
        student = ConfigDict(teacher.to_dict())
        del student.gq["so3_params"]  # 普通 student 不应依赖 SO(3) 配置。
        student.gq.h1, student.gq.h2, student.gq.nh = 4, 2, 2
        student.gq.wfn_depth, student.gq.ndet = 2, 2
        student.gq.jastrow_type = "null"
        student.gq.envelope_type = "iso"
        student.gq.dp_type = "original"
        positions = jnp.asarray(((0.3, 0.2, 0.7), (-0.5, 0.4, -0.8)))
        for kind, rhf in (("ef_shrd_sym", False),
                          ("ef_shrd_sym_proj", False),
                          ("ef_shrd_sym_ablation", False), ("ef_shrd_sym", True)):
            with self.subTest(kind=kind, rhf=rhf):
                student.gq.type, student.gq.RHF = kind, rhf
                molecule = validate_configs(teacher, student)
                params, network = construct_network(student, molecule, jax.random.PRNGKey(7))
                folded = _fold_student_log_scale(params, jnp.log(3.), student)
                sign, logabs = network(params, positions, molecule.ions)
                new_sign, new_logabs = network(folded, positions, molecule.ions)
                self.assertTrue(bool(jnp.isfinite(logabs)))
                self.assertEqual(float(sign), float(new_sign))
                self.assertAlmostEqual(float(new_logabs - logabs), float(jnp.log(3.)), places=4)
                self.assertEqual(jax.tree_util.tree_structure(params),
                                 jax.tree_util.tree_structure(folded))
                runners._checked_pretrained_params(params, folded, apply_pmap=False)

    def test_reload_reads_saved_config_without_filling_fields(self):
        parser = runners.train.parse_config_flags
        reload = SimpleNamespace(logdir="/old/run", config_relative_file_path="config.json")
        saved = ConfigDict({"logdir": "/old/run", "base_logdir": "/old"})
        with patch.object(parser, "load_config_dict", return_value=saved) as read, \
             patch.object(parser.sys, "argv", ["test"]):
            result = parser._get_config_from_reload(reload, flags.FlagValues())
        read.assert_called_once_with("/old/run", "config.json")
        self.assertEqual(result.to_dict(), {"logdir": "/old", "base_logdir": "/old"})
        # 原始读取逻辑缺 base_logdir 会报错，不补默认值或拿 reload 路径回退。
        with patch.object(parser, "load_config_dict", return_value=ConfigDict({"logdir": "/old/run"})):
            with self.assertRaises(AttributeError):
                parser._get_config_from_reload(reload, flags.FlagValues())

    def test_pretrain_completes_only_teacher_not_student(self):
        from vmcnet.train import pretrain_from_trained_net as pretrain
        teacher = get_default_config()
        teacher.gq.type = "ef_shrd_sym"
        student = get_default_config()
        student.gq.type = "so3"
        del student.gq["so3_params"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            teacher_path, student_path = root / "teacher.json", root / "student.json"
            teacher_path.write_text(teacher.to_json())
            student_path.write_text(student.to_json())
            checkpoint = root / "teacher.npz"
            checkpoint.touch()
            # 在构建网络前检查实际传入的 student，避免运行预训练。
            with patch.object(pretrain, "complete_config", wraps=complete_config) as complete, \
                 patch.object(pretrain, "validate_configs", side_effect=RuntimeError("stop-before-network")) as validate:
                with self.assertRaisesRegex(RuntimeError, "stop-before-network"):
                    pretrain.main(["--teacher-config", str(teacher_path),
                                   "--student-config", str(student_path),
                                   "--teacher-checkpoint", str(checkpoint),
                                   "--output-dir", str(root / "output")])
            complete.assert_called_once()
            self.assertEqual(complete.call_args.kwargs["source"], str(teacher_path))
            self.assertNotIn("so3_params", validate.call_args.args[1].gq)

    def test_problem_validation_preserves_student_and_needs_no_network_defaults(self):
        teacher = get_default_config()
        student = get_default_config()
        # 模拟旧 JSON：没有 gq/so3 等字段，也能提前验证物理体系。
        teacher = ConfigDict({"problem": teacher.problem.to_dict(), "dtype": teacher.dtype})
        teacher.problem.ion_pos = np.asarray(teacher.problem.ion_pos)[None].tolist()
        before = student.to_dict()
        student.lock()
        validate_problem_match(teacher, student)
        self.assertEqual(student.to_dict(), before)
        self.assertNotIn("gq", teacher)
        for field, value in (("ei_softening", 0.1), ("atoms_symbol", ("C", "H")),
                             ("nspins", (1, 1))):
            with self.subTest(field=field):
                changed = ConfigDict(teacher.to_dict())
                changed.problem[field] = value
                with self.assertRaisesRegex(ValueError, "problem." + field):
                    validate_problem_match(changed, student)
        del teacher.problem["ee_softening"]
        with self.assertRaisesRegex(ValueError, "ee_softening.*缺失"):
            validate_problem_match(teacher, student)

    def test_sampler_matches_main_vmc_moves_and_preserves_width_state(self):
        ions = jnp.zeros((1, 3))
        positions = jnp.zeros((3, 2, 3))
        key = jax.random.PRNGKey(18)

        def network(params, x, atoms):
            return jnp.asarray(1.), jnp.asarray(0., dtype=x.dtype)

        sampler = make_metropolis_sampler(network, ions, 1, 0.2, 2)
        step = dwpa.make_dynamic_pos_amp_gaussian_step(
            lambda p, a, x: jnp.zeros(x.shape[0], dtype=x.dtype), 2,
            dwpa.make_threshold_adjust_std_move(0.5, 0.05, 0.1))
        data = dwpa.make_dynamic_width_position_amplitude_data(
            ions[None], positions[None], jnp.zeros((1, 3)), jnp.asarray([0.2]),
            jnp.zeros(1), jnp.full((1,), 0))
        reference_key = key[None]
        metadata = None
        for _ in range(2):
            positions, key, _, metadata = sampler({}, positions, key, metadata)
            _, data, reference_key = step({}, data, reference_key)
            np.testing.assert_allclose(positions, data["walker_data"]["elec_position"][0],
                                       rtol=1e-6, atol=1e-7)
            np.testing.assert_array_equal(key, reference_key[0])
            self.assertEqual(float(metadata["std_move"][0]),
                             float(data["move_metadata"]["std_move"][0]))
        self.assertTrue(np.all(np.asarray(positions) != 0.))
        self.assertGreater(float(metadata["std_move"][0]), 0.2)
        self.assertEqual(int(metadata["moves_since_update"][0]), 0)

    def test_runner_rejects_problem_mismatch_before_completion_or_output(self):
        student = get_default_config()
        student.pretrain.teacher.enabled = True
        student.pretrain.teacher.config = "/teacher/config.json"
        teacher = ConfigDict({"problem": student.problem.to_dict(), "dtype": student.dtype})
        teacher.problem.ei_softening = 0.1
        from vmcnet.train.default_config import get_default_reload_config
        with patch.object(runners.train.parse_config_flags, "parse_flags",
                          return_value=(get_default_reload_config(), student)), \
             patch.object(runners.utils.io, "load_config_dict", return_value=teacher), \
             patch("vmcnet.train.pretrain_from_trained_net.complete_config") as complete, \
             patch.object(runners, "_get_logdir_and_save_config") as save:
            with self.assertRaisesRegex(ValueError, "problem.ei_softening"):
                runners.run_molecule()
        complete.assert_not_called()
        save.assert_not_called()

    def test_legacy_config_fills_missing_so3_fields_without_overwriting(self):
        legacy = ConfigDict({
            "dtype": "float64",
            "gq": {"type": "so3", "so3_params": {"lmax": 1,
                                                "target_ls": (0, 1)}},
            "model": {"type": "ferminet"},
        })
        completed, added = complete_config(legacy)
        self.assertEqual(completed.gq.so3_params.lmax, 1)
        self.assertEqual(tuple(completed.gq.so3_params.target_ls), (0, 1))
        self.assertIn("gq.so3_params.regularize_angular_origin", added)
        self.assertFalse(completed.gq.so3_params.regularize_angular_origin)
        # Reloaded model configs are narrowed; do not insert all other model types.
        self.assertNotIn("psiformer", completed.model)

    def test_legacy_gaoqiao_defaults_preserve_old_network_choices(self):
        legacy = ConfigDict({
            "gq": {"type": "ef_shrd_sym", "activation_type": "silu",
                   "h1": 16},
            "pretrain": {"iterations": 0},
        })
        completed, added = complete_config(legacy)
        self.assertEqual(completed.gq.dp_type, "original")
        self.assertEqual(completed.gq.jastrow_activation_type, "silu")
        self.assertEqual(completed.gq.h1, 16)
        self.assertIn("gq.dp_type", added)
        self.assertIn("gq.jastrow_activation_type", added)
        self.assertIn("gq.so3_params", added)
        self.assertIn("pretrain.teacher", added)

    def test_config_completion_logs_each_filled_value(self):
        legacy = ConfigDict({
            "gq": {"type": "ef_shrd_sym", "activation_type": "silu"},
            "pretrain": {"iterations": 0},
        })
        with self.assertLogs(level="WARNING") as captured:
            _, added = complete_config(legacy, source="old_config.json")

        message = "\n".join(captured.output)
        self.assertEqual(len(captured.output), 1)
        self.assertIn("old_config.json", message)
        self.assertIn("gq.dp_type = 'original'", message)
        self.assertIn("gq.jastrow_activation_type = 'silu'", message)
        self.assertIn("pretrain.teacher.enabled = False", message)
        self.assertIn("pretrain.teacher", added)

        with patch("vmcnet.train.config_compat.logging.warning") as warning:
            _, added = complete_config(get_default_config())
        self.assertEqual(added, ())
        warning.assert_not_called()

    def test_saved_gaoqiao_choices_override_historical_fallbacks(self):
        legacy = ConfigDict({"gq": {
            "type": "ef_shrd_sym", "activation_type": "silu",
            "jastrow_activation_type": "tanh", "dp_type": "charge"}})
        completed, added = complete_config(legacy)
        self.assertEqual(completed.gq.dp_type, "charge")
        self.assertEqual(completed.gq.jastrow_activation_type, "tanh")
        self.assertNotIn("gq.dp_type", added)
        self.assertNotIn("gq.jastrow_activation_type", added)

    def test_so3_missing_fields_do_not_use_pre_so3_fallbacks(self):
        completed, _ = complete_config(ConfigDict({"gq": {"type": "so3"}}))
        self.assertEqual(completed.gq.dp_type, get_default_config().gq.dp_type)
        self.assertEqual(completed.gq.jastrow_activation_type, "tanh")

    def test_checkpoint_amplitude_check_detects_changed_network_meaning(self):
        molecule = Molecule(jnp.zeros((1, 3)), jnp.ones((1,)), (1, 0),
                            jnp.asarray(((1, 0),)))
        positions = np.asarray([[[1., 0., 0.]], [[2., 0., 0.]]])
        data = {"walker_data": {"elec_position": positions,
                                "amplitude": np.asarray([-1., -2.])}}

        def original_network(params, x, ions):
            del params
            return jnp.asarray(1.), -jnp.linalg.norm(x[0] - ions[0])

        self.assertTrue(check_checkpoint_amplitudes(
            original_network, {}, data, molecule))

        def changed_network(params, x, ions):
            sign, logabs = original_network(params, x, ions)
            return sign, logabs + 0.1

        with self.assertRaisesRegex(ValueError, "原波函数定义"):
            check_checkpoint_amplitudes(changed_network, {}, data, molecule)

    def test_signed_mse_sees_global_sign_and_alignment_removes_it(self):
        target_sign = jnp.asarray((1., -1.))
        student_sign = -target_sign
        logs = jnp.zeros((2,))
        raw, _ = signed_amplitude_loss(
            target_sign, logs, student_sign, logs, "signed_mse")
        aligned, (scale, _) = signed_amplitude_loss(
            target_sign, logs, student_sign, logs, "scale_aligned")
        self.assertAlmostEqual(float(raw), 4.0)
        self.assertAlmostEqual(float(aligned), 0.0)
        self.assertAlmostEqual(float(scale), -1.0)

    def test_signed_mse_fits_scale_but_aligned_loss_does_not(self):
        signs = jnp.asarray((1., -1.))
        target_log = jnp.zeros((2,))
        student_log = jnp.full((2,), jnp.log(2.0))
        raw, _ = signed_amplitude_loss(signs, target_log, signs, student_log,
                                       "signed_mse")
        aligned, _ = signed_amplitude_loss(signs, target_log, signs, student_log,
                                           "scale_aligned")
        self.assertGreater(float(raw), 0.0)
        self.assertAlmostEqual(float(aligned), 0.0)

    def test_log_scale_can_be_folded_into_all_so3_sectors(self):
        params = {"so3": {}, "so3_det": {"alpha": {
            "0": jnp.ones((1, 2)), "1": -jnp.ones((1, 2))}}}
        folded = _fold_so3_log_scale(params, jnp.log(3.0))
        self.assertAlmostEqual(float(folded["so3_det"]["alpha"]["0"][0, 0]), 3.0)
        self.assertAlmostEqual(float(folded["so3_det"]["alpha"]["1"][0, 0]), -3.0)
        self.assertAlmostEqual(float(params["so3_det"]["alpha"]["0"][0, 0]), 1.0)

    def test_aligned_loss_handles_large_initial_log_scale_gap(self):
        signs = jnp.asarray((1., -1.))
        teacher_log = jnp.zeros((2,))
        student_log = jnp.full((2,), -100.0)
        loss, (alignment, _) = signed_amplitude_loss(
            signs, teacher_log, signs, student_log, "scale_aligned")
        self.assertAlmostEqual(float(loss), 0.0)
        self.assertAlmostEqual(float(alignment), 1.0)

    def test_teacher_sampler_starts_from_checkpoint_walkers(self):
        ions = jnp.asarray(((0., 0., 1.), (0., 0., -1.)))
        molecule = Molecule(ions, jnp.asarray((1., 1.)), (1, 1),
                            jnp.asarray(((1, 0), (0, 1))))
        walkers = np.arange(24., dtype=float).reshape(1, 4, 2, 3)
        data = {"atoms_position": np.asarray(ions)[None],
                "walker_data": {"elec_position": walkers}}
        _, positions, source = initial_teacher_positions(
            data, None, molecule, 2, jax.random.PRNGKey(8))
        self.assertEqual(source, "checkpoint")
        self.assertEqual(positions.shape, (2, 2, 3))
        self.assertTrue(any(np.array_equal(np.asarray(positions[0]), old)
                            for old in walkers[0]))

    def test_energy_diagnostic_uses_coulomb_local_energy(self):
        """For the hydrogen 1s wavefunction, T + V is exactly -1/2 Ha."""
        molecule = Molecule(jnp.zeros((1, 3)), jnp.ones((1,)), (1, 0),
                            jnp.asarray(((1, 0),)))
        config = SimpleNamespace(
            vmc=SimpleNamespace(fwdlap_inner_size=0, nmoves_per_width_update=100),
            problem=SimpleNamespace(ei_softening=0.0, ee_softening=0.0))

        def hydrogen_network(params, x, ions):
            del params
            return jnp.asarray(1.0), -jnp.linalg.norm(x[0] - ions[0])

        components = make_energy_components(hydrogen_network, molecule, config)
        terms = np.asarray(components({}, jnp.asarray(((1.3, 0.2, -0.4),))))
        self.assertEqual(terms.shape, (4,))
        self.assertAlmostEqual(float(np.sum(terms)), -0.5, places=5)
        self.assertAlmostEqual(float(terms[2]), 0.0)
        self.assertAlmostEqual(float(terms[3]), 0.0)

    def test_optional_energy_diagnostic_writes_samples_and_summary(self):
        molecule = Molecule(jnp.zeros((1, 3)), jnp.ones((1,)), (1, 0),
                            jnp.asarray(((1, 0),)))
        config = SimpleNamespace(
            vmc=SimpleNamespace(fwdlap_inner_size=0, nmoves_per_width_update=100),
            problem=SimpleNamespace(ei_softening=0.0, ee_softening=0.0))
        args = SimpleNamespace(energy_samples=3, energy_burn_in=2,
                               mcmc_steps=2, proposal_std=0.2)

        def hydrogen_network(params, x, ions):
            del params
            return jnp.asarray(1.0), -jnp.linalg.norm(x[0] - ions[0])

        starting_points = jnp.asarray((((1., 0., 0.),),
                                       ((0., 1., 0.),)))
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            evaluate_student_energy(hydrogen_network, {}, molecule, config,
                                    starting_points, jax.random.PRNGKey(4),
                                    args, output)
            with (output / "energy_summary.json").open() as file:
                summary = json.load(file)
            with (output / "energy_samples.csv").open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(summary["sample_count"], 3)
            self.assertEqual(summary["nonfinite_count"], 0)
            self.assertEqual(len(rows), 3)
            self.assertAlmostEqual(summary["mean_hartree"], -0.5, places=5)

    def test_runner_passes_teacher_settings_and_uses_saved_student(self):
        config = get_default_config()
        teacher = config.pretrain.teacher
        teacher.enabled = True
        teacher.checkpoint = "/teacher/model.npz"
        teacher.config = "/teacher/config.json"
        teacher.steps = 12
        teacher.test_energy = True
        saved = {"weight": np.asarray((2.0,))}
        with patch("vmcnet.train.pretrain_from_trained_net.main",
                   return_value=Path("/run/teacher_pretrain")) as train_main:
            with patch.object(runners.utils.io, "reload_vmc_state",
                              return_value=(0, None, saved, None, None)):
                params = runners._run_teacher_pretraining(config, "/run")
        self.assertIs(params, saved)
        arguments = train_main.call_args.args[0]
        self.assertEqual(arguments[arguments.index("--student-config") + 1],
                         "/run/config.json")
        self.assertEqual(arguments[arguments.index("--steps") + 1], "12")
        self.assertIn("--test-energy", arguments)
        self.assertNotIn("--mix-start-step", arguments)

    def test_runner_checks_student_weights_before_vmc_initialization(self):
        initialized = {"weight": jnp.zeros((2, 3))}
        trained = {"weight": np.ones((2, 3))}
        selected = runners._checked_pretrained_params(
            initialized, trained, apply_pmap=False)
        np.testing.assert_array_equal(selected["weight"], trained["weight"])
        with self.assertRaisesRegex(ValueError, "形状"):
            runners._checked_pretrained_params(
                initialized, {"weight": np.ones((3, 2))}, apply_pmap=False)


if __name__ == "__main__":
    unittest.main()
