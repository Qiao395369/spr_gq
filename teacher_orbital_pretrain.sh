#!/usr/bin/env bash
# CH4：用已训练 ef_shrd_sym teacher 的完整轨道，预训练新的 ef_shrd_sym student。
# student 也可选 so3_l0；未指定的 config.gq.so3_l0.* 使用默认值。
# 训练时固定 teacher 坐标和标签；结束后从 student 分布重新采样测试能量。
# 不启动 VMC 能量优化，能量测试不更新参数。
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_dir"

# 使用你运行前已激活的环境中的 python；Mac 与服务器采用同一调用方式。
teacher_checkpoint="$repo_dir/teacher_net/CH4/150000_101158.npz"
teacher_config="$repo_dir/teacher_net/CH4/config_101158.json"
output_dir="${OUTPUT_DIR:-$repo_dir/../results/CH4_teacher_orbital_pretrain}"

# 这里只设置物理体系和网络构造参数；未写出的配置直接使用 default。
# 不设置 VMC 优化器或训练参数；能量诊断使用下方独立选项。
# teacher/student 的分子与精度必须一致，不会从 teacher 自动覆盖 student。
ARGS=(
  --config.dtype=float64
  --config.wfn_type=gaoqiao
  '--config.problem.ion_pos=((0.,0.,0.),(1.18886,1.18886,1.18886),(-1.18886,-1.18886,1.18886),(1.18886,-1.18886,-1.18886),(-1.18886,1.18886,-1.18886))'
  '--config.problem.ion_charges=(6.,1.,1.,1.,1.)'
  '--config.problem.atoms_symbol=("C","H","H","H","H")'
  '--config.problem.nspins=(5,5)'
  '--config.problem.single_nspins=((4,2),(0,1),(0,1),(0,1),(1,0))'

  --config.gq.type=ef_shrd_sym                 # 可改为 so3_l0，使用 config.gq.so3_l0.* 配置。
  --config.gq.so3_l0.channels=16 \
  --config.gq.so3_l0.lmax=2 \
  --config.gq.so3_l0.n_layers=3
  --config.gq.dp_type=full                    # 与当前 teacher 的粒子类型编码一致。
  --config.gq.det_mode=det                    # 完整 N×N Slater 行列式。
  --config.gq.ndet=16                         # 必须与 teacher 相同，按序号匹配每个行列式。
  --config.gq.wfn_depth=4                     # student 消息更新层数。
  --config.gq.h1=32                           # 单电子特征宽度。
  --config.gq.h2=8                            # 双电子特征宽度。
  --config.gq.nh=16                           # 核包络网络的隐藏宽度。
  --config.gq.do_attn=False
  --config.gq.envelope_type=ds_hz_deg          # student 可与 teacher 使用不同的网络容量或包络。
  --config.gq.jastrow_type=mlp_res             # 正常轨道接口已包含 Jastrow 乘子，不再额外乘一次。
  --config.gq.jastrow_mlp_nlayer=4
  --config.gq.jastrow_mlp_ndim=32
  --config.gq.rescale_input=all
  --config.gq.activation_type=silu
  --config.gq.jastrow_activation_type=tanh
)

# 以下都是本次独立预训练的设置，与 config.vmc 的优化参数无关。
PRETRAIN_ARGS=(
  --teacher-checkpoint "$teacher_checkpoint"
  --teacher-config "$teacher_config"
  --output-dir "$output_dir"
  --loss-mode "${LOSS_MODE:-l2}"              # l2：逐元素 MSE；fulldet：按行列式比较 QR 列空间。
  --sample-count 4096                         # 固定训练构型总数，全部用于训练，不划验证集。
  --sampler-chains 1024                        # teacher 并行采样链数，与训练 batch-size 分开。
  --batch-size 1024                            # 每次 Adam 更新使用的固定构型数。
  --steps 10000                                # 预训练 Adam 更新次数。
  --learning-rate 3e-4                        # 沿用 pretrain_demo.py 的 Adam 学习率。
  --save-every 2000                            # 日志输出与 student checkpoint 保存间隔。
  --seed 0
  --test-energy                              # 预训练结束后，用最终 student 测试能量。
  --energy-samples 2048                       # 重新采集的能量测试构型总数。
  --energy-burn-in 20000
  # student 能量测试热化步数默认读取 config.eval.nburn。
  # 可用 --energy-burn-in 覆盖；--no-test-energy 可关闭测试。
  # 热化、批次间采样间隔及初始步宽未显式设置，读取默认配置中的相应值。
)

# 默认测试 l2；测试 fulldet 可运行 LOSS_MODE=fulldet ./teacher_orbital_pretrain.sh。
# "$@" 允许临时覆盖，如 --steps 20 --sample-count 128 --burn-in 100。
python -m vmcnet.train.pretrain_teacher_orbitals "${PRETRAIN_ARGS[@]}" "${ARGS[@]}" "$@"
