#!/usr/bin/env bash
# CH4: 用已收敛的 ef_shrd_sym teacher 单独预训练 gaoqiao student。
# 只运行蒸馏脚本；不会启动 runners、VMC 优化或最终评估。
# -e：命令失败就退出；-u：使用未定义变量时报错；pipefail：管道内任一命令失败都视为失败。
set -euo pipefail

# BASH_SOURCE[0] 是脚本自身的路径；转成绝对目录后，无论从哪里调用都能找到项目文件。
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 后续 Python 模块导入和相对路径均以项目根目录为起点。
cd "$repo_dir"

# 使用调用者事先激活的环境中的 python。
# 这两个文件必须属于同一次 CH4 teacher 训练：前者是参数与 walker，后者定义网络和分子。
teacher_checkpoint="$repo_dir/teacher_net/CH4/150000_101158.npz"
teacher_config="$repo_dir/teacher_net/CH4/config_101158.json"
# 可通过环境变量 OUTPUT_DIR 改结果目录；它不受下面 --config.logdir 控制。
output_dir="${OUTPUT_DIR:-$repo_dir/../results/CH4_teacher_pretrain}"

# 依次检查 teacher 参数文件和配置文件；-f 要求是普通文件。
for input_file in "$teacher_checkpoint" "$teacher_config"; do
  if [[ ! -f "$input_file" ]]; then
    echo "缺少输入文件: $input_file" >&2
    exit 1
  fi
done

# 独立预训练入口只能接收 student JSON；mktemp 的 XXXXXX 会生成不冲突的临时文件名。
student_config="$(mktemp "${TMPDIR:-/tmp}/CH4_so3_student.XXXXXX")"
# 不论正常完成还是中途报错，脚本退出时都删除临时文件；结果目录会另存配置副本。
trap 'rm -f "$student_config"' EXIT

# student 配置采用与常规 runners 脚本相同的 --config.* 参数格式。
# student 显式指定 CH4；后续仅核对 teacher 的物理体系，不从 teacher 覆盖。
ARGS=(
  --config.logdir=../results                         # 保存到 student 配置的初始日志根目录。
  --config.save_to_current_datetime_subfolder=False  # 不在日志目录下额外创建时间戳子目录。
  --config.subfolder_name=CH4                         # student 配置中的子目录名；独立预训练另用 --output-dir。
  --config.dtype=float64                              # 请求双精度；必须与 teacher 一致。
  --config.distribute=False                           # student 配置记录为非分布式运行。
  '--config.problem.ion_pos=((0.,0.,0.),(1.18886,1.18886,1.18886),(-1.18886,-1.18886,1.18886),(1.18886,-1.18886,-1.18886),(-1.18886,1.18886,-1.18886))'
  '--config.problem.ion_charges=(6.,1.,1.,1.,1.)'
  '--config.problem.atoms_symbol=("C","H","H","H","H")'
  '--config.problem.nspins=(5,5)'
  '--config.problem.single_nspins=((4,2),(0,1),(0,1),(0,1),(1,0))'
  '--config.eval.ion_pos=((0.,0.,0.),(1.18886,1.18886,1.18886),(-1.18886,-1.18886,1.18886),(1.18886,-1.18886,-1.18886),(-1.18886,1.18886,-1.18886))'
  '--config.eval.ion_charges=(6.,1.,1.,1.,1.)'
  '--config.eval.nspins=(5,5)'
  '--config.eval.single_nspins=((4,2),(0,1),(0,1),(0,1),(1,0))'

  # 这些常规 VMC 设置主要保存在配置中供日后 reload；本脚本不执行 VMC。
  --config.pretrain.iterations=0                       # 不安排常规 Hartree--Fock 预训练。
  --config.vmc.init_width=0.5                         # 日后新建 VMC walker 时的坐标扰动宽度。
  --config.vmc.nchains=16                             # 日后 VMC 的链数；独立预训练的批量由 --batch-size 控制。
  --config.vmc.nburn=10                               # 日后 VMC 的热化步数，不是下方 --burn-in。
  --config.vmc.nepochs=20                             # 日后 VMC 的训练轮数，不是下方 --steps。
  --config.vmc.nsteps_per_param_update=40             # 日后 VMC 每次参数更新之间的采样步数。
  --config.vmc.fwdlap_inner_size=30                   # 可选 --test-energy 诊断计算动能时也会用到。
  --config.vmc.check_for_nans=True                    # 日后 VMC 的 NaN 检查选项。
  --config.vmc.optimizer_type=spring                  # 日后 VMC 使用 Spring；本次预训练使用下方 Adam 学习率。
  --config.vmc.optimizer.spring.learning_rate=5e-3   # 日后 Spring 的学习率。
  --config.vmc.optimizer.spring.norm_constraint=0.001  # 日后 Spring 的更新范数约束。
  --config.vmc.optimizer.spring.type=2                # 日后 Spring 的算法类型。
  --config.vmc.optimizer.spring.det_regularization_scale=0  # 日后行列式正则项的系数。

  --config.wfn_type=gaoqiao                            # 使用项目中的 gaoqiao 波函数入口。
  --config.gq.type=ef_shrd_sym                          # student 网络类型；也可改为 so3 等分支。
  --config.gq.dp_type=full                             # 粒子类型特征采用 full 编码。
  --config.gq.det_mode=det                             # 使用行列式输出头。
  --config.gq.ndet=16                                 # 每个 rank 的行列式数 K。
  --config.gq.wfn_depth=4 \
  --config.gq.h1=32 \
  --config.gq.h2=8 \
  --config.gq.envelope_type=ds_hz_deg \

  --config.gq.so3_params.basis=real                     # 主干、读出和行列式均使用实数表示。
  --config.gq.so3_params.lmax=2                        # 主干保留 l=0、1、2 阶隐藏特征。
  '--config.gq.so3_params.target_ls=(0,)'             # 最终仅构造 l=0 sector；引号保护括号不被 shell 解析。
  --config.gq.so3_params.channels=8                   # 每个 l 阶的 channel 数。
  --config.gq.so3_params.n_layers=2                   # SO(3) 消息传递层数。
  --config.gq.so3_params.n_rbf=8                      # 边距离的径向基函数数目。
  --config.gq.so3_params.wbn_layers=1                 # WBN 张量积更新块数。
  --config.gq.so3_params.init_l1=True                 # 初始化 l=1 特征。
  --config.gq.so3_params.regularize_angular_origin=True  # 高阶边角向项在 r→0 时衰减。

  --config.gq.so3_params.det_rank=2                    # 行列式与 SO(3) 读出的 rank 数 R。
  --config.gq.so3_params.det_bank_mode=shared         # 各目标 l 共享行列式库；当前只有 l=0。
  --config.gq.so3_params.use_invariant_orbital_features=True  # 高阶特征的范数也输入标量轨道头。
  --config.gq.so3_params.output_projection=q0        # 取每个目标 l 的 q=0 分量；这里只剩 l=0。

  --config.gq.so3_params.use_envelope=True             # 在行列式轨道上使用各向同性核包络。
  --config.gq.so3_params.envelope_sigma_init=1.0      # 核包络衰减参数的初始值。
  --config.gq.so3_params.envelope_pi_init=1.0         # 核包络系数的初始值。
  --config.gq.so3_params.use_kato_jastrow=True        # 加入电子对 Kato Jastrow。
  --config.gq.so3_params.jastrow_b_init=1.0           # Kato Jastrow 范围参数的初始值。

  --config.eval.nburn=10                              # 日后评估链的热化步数。
  --config.eval.nepochs=20                            # 日后评估的轮数。
  --config.eval.nchains=12                            # 日后评估使用的链数。
  --config.eval.use_data_from_training=False          # 日后评估重新初始化 walker。
  --config.eval.init_width=0.5                        # 日后评估初始坐标扰动宽度。
  --config.eval.nsteps_per_param_update=40            # 日后评估每轮的采样步数。
)

# 独立预训练入口读取 JSON，而不是 --config.*；先用项目自身的 flags 解析器生成 JSON。
# 两个路径仅作为这次 Python 子进程的环境变量；"${ARGS[@]}" 保留数组中每个参数的边界。
# Python 的 "-" 表示从标准输入读取下方代码；<<'PY' 禁止 shell 展开其中的变量。
STUDENT_CONFIG_PATH="$student_config" TEACHER_CONFIG_PATH="$teacher_config" \
  python - "${ARGS[@]}" <<'PY'
# 读取 shell 传入的环境变量与 --config.* 命令行参数。
import os
import sys
# 用 Path 写出供独立预训练模块读取的临时 JSON。
from pathlib import Path

# 根据 student 的双精度设置启用 JAX x64。
import jax
# 使用与 runner 相同的 absl / ml_collections 配置参数解析机制。
from absl import flags
from ml_collections.config_flags import config_flags

# 此处只读原始 teacher 配置；网络重构前才执行兼容性补全。
from vmcnet.utils.io import load_config_dict
# 默认配置提供未在 ARGS 中写出的 student 参数；model 只保留当前类型的子配置。
from vmcnet.train.default_config import (
    choose_model_type_in_model_config, get_default_config)
# 只比较原始物理体系和精度，不依赖旧配置缺少的网络字段。
from vmcnet.train.pretrain_from_trained_net import (
    validate_problem_match)

# 新建独立的 FlagValues，避免借用已注册过其他 flag 的全局解析器。
flag_values = flags.FlagValues()
# 先装入完整默认配置；sys_argv 让嵌套的 --config.xxx 覆盖项得到注册。
config_flags.DEFINE_config_dict(
    "config", get_default_config(), flag_values=flag_values,
    lock_config=False, sys_argv=sys.argv)
# 将 shell 数组 ARGS 中的覆盖项写到默认配置上。
flag_values(sys.argv)
student = flag_values.config
# 默认 model 配置包含多个模型分支，保存前只留下当前选中的一个。
student.model = choose_model_type_in_model_config(student.model)

# 核对原始 teacher 配置；任何不一致都中止脚本。
teacher = load_config_dict("", os.environ["TEACHER_CONFIG_PATH"])
validate_problem_match(teacher, student)
# 如果 student dtype 是 float64，允许 JAX 创建真正的双精度数组。
jax.config.update("jax_enable_x64", student.dtype == "float64")
# 网络类型与实数模式在预训练重构网络前校验。

# 从环境变量取得 mktemp 生成的临时路径。
path = Path(os.environ["STUDENT_CONFIG_PATH"])
# 把解析后的完整 student 配置写成 UTF-8 JSON，供下一条命令读取。
path.write_text(student.to_json(indent=2), encoding="utf-8")
print("已按 ARGS 生成 CH4 student 配置；预训练会将其保存到结果目录。")
PY

# 只调用独立预训练脚本。若结果目录已存在，它会自动使用 _1、_2 等新目录。
PRETRAIN_ARGS=(
  --teacher-checkpoint "$teacher_checkpoint"  # 已训练 teacher 的参数及采样状态。
  --teacher-config "$teacher_config"          # 与 checkpoint 配对的原始 teacher 配置。
  --student-config "$student_config"          # 上面临时生成的完整 student JSON。
  --output-dir "$output_dir"                  # 蒸馏结果的目标目录。
  --loss-mode scale_aligned                    # 可选 signed_mse、scale_aligned、relative_mse。
  --sampling-strategy fixed_teacher            # 固定 teacher 点训练；dynamic_mixed 保留原动态策略。
  --fixed-sample-count 0                       # 0 冻结 batch-size 个点；设更大值则先采满再训练，所有点都训练。
  --steps 2000                                # student 参数更新次数。
  --batch-size 16                             # 每次损失计算使用的电子构型数。
  --mix-start-step 500                        # 仅 dynamic_mixed：开始混入 student 样本的步数。
  --student-fraction 0.5                      # 仅 dynamic_mixed：student 样本比例。
  --burn-in 200                               # 初始 teacher 采样链的热化步数。
  --student-burn-in 100                       # 仅 dynamic_mixed：首次启动 student 链时的热化步数。
  --mcmc-steps 10                             # 动态模式的更新间采样步数；固定模式的采集批次间隔。
  --proposal-std 0.25                         # 全电子高斯提议的初始步宽；之后按接受率自动调节。
  --learning-rate 1e-5                        # 本次预训练 Adam 的学习率，非 VMC Spring 学习率。
  --gradient-clip 1.0                         # 预训练梯度的全局范数裁剪阈值。
  --save-every 200                            # 每 200 次更新保存一次 student checkpoint。
  --seed 0                                    # JAX 随机数生成器的初始种子。
  --test-energy
  --energy-samples 16 
  --energy-burn-in 100
)
# "${PRETRAIN_ARGS[@]}" 传入固定参数；"$@" 额外转发脚本调用者给出的参数。
python -m vmcnet.train.pretrain_from_trained_net "${PRETRAIN_ARGS[@]}" "$@"

