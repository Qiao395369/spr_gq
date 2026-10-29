#!/usr/bin/env bash

# 从项目根目录运行；任一测试失败就停止。
set -e

python -m unittest -v so3_test.test_rotation_l0
python -m unittest -v so3_test.test_rotation_l1
python -m unittest -v so3_test.test_rotation_l2
python -m unittest -v so3_test.test_representation_matrix
python -m unittest -v so3_test.test_coalescence
python -m unittest -v so3_test.test_scalar_wavefunction_rotation
python -m unittest -v so3_test.test_exchange_antisymmetry

# 独立零阶 orbital-like 网络及其轨道预训练接入。
python -m unittest -v so3_test.test_so3_l0

# 原版主干的逐电子展开 + 可选 MLP，保留置换性质并测试 VMC 接口。
python -m unittest -v so3_test.test_mlp_readout
