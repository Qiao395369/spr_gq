#!/usr/bin/env bash
# 在项目根目录运行。每个测试文件单独显示结果；即使其中一个失败，也继续跑完。
set -u

python_bin="${PYTHON_BIN:-python}"
status=0
"$python_bin" -m unittest -v gq_test.test_wavefunction_values || status=1
"$python_bin" -m unittest -v gq_test.test_translation_invariance || status=1
"$python_bin" -m unittest -v gq_test.test_exchange_symmetry || status=1
"$python_bin" -m unittest -v gq_test.test_derivatives || status=1
"$python_bin" -m unittest -v gq_test.test_teacher_pretrain || status=1
exit "$status"
