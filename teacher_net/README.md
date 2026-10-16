# 已训练 teacher 网络存放处

按分子放置原始 checkpoint 和对应运行的配置文件，文件名可以自定：

```text
teacher_net/
├── N2/    # 例如 best_checkpoint.npz、config2026.json
├── CH4/
└── NH3/
```

预训练脚本使用显式路径，因此不会猜测同一目录里哪个 `.npz` 对应哪个
`config*.json`。两者必须来自同一次已收敛训练。每次预训练都会把实际使用的
原始文件复制到新结果目录的 `teacher_source/`，并另存补齐旧字段后的配置。

现在也可以在一次 `runners` 调用中完成 teacher 预训练和后续 VMC。设置
学生 SO(3) 网络及优化参数，再额外传入：

```bash
python -m vmcnet.train.runners \
  --config.wfn_type=gaoqiao \
  --config.gq.type=so3 \
  --config.pretrain.teacher.enabled=True \
  --config.pretrain.teacher.checkpoint=teacher_net/N2/best_checkpoint.npz \
  --config.pretrain.teacher.config=teacher_net/N2/config2026.json \
  --config.pretrain.teacher.loss_mode=signed_mse \
  --config.pretrain.teacher.steps=2000 \
  --config.pretrain.teacher.test_energy=True
```

runner 将预训练文件放在本次 VMC 结果目录的 `teacher_pretrain/`，从该目录的
`pretrained_student.npz` 读取学生参数，再初始化 VMC walker 的波函数值和
优化器状态。学生训练的核坐标、核顺序、核电荷、原子符号、自旋电子数及
`dtype` 直接继承 teacher 配置；评估使用同一分子，但保留学生配置中的评估
步数和链数。无需在命令中重复设置学生的分子参数。启用 teacher 模式时会跳过 HF 预训练；已有 VMC checkpoint 的
续训不会重新做 teacher 预训练。能量诊断默认关闭，打开后结果位于
`teacher_pretrain/energy_samples.csv` 和 `energy_summary.json`。其余采样、损失、
步数和能量诊断选项与独立脚本同名，前缀改为 `--config.pretrain.teacher.`。
当前 teacher 预训练阶段要求单主机；同一主机上的多设备 VMC 可以在预训练后继续运行。

示例（学生配置提供准备训练的 SO(3) 网络和优化设置，分子信息由 teacher 覆盖）：

```bash
python -m vmcnet.train.pretrain_from_trained_net \
  --teacher-checkpoint teacher_net/N2/best_checkpoint.npz \
  --teacher-config teacher_net/N2/config2026.json \
  --student-config path/to/so3_config.json \
  --output-dir ../results/N2_teacher_pretrain \
  --loss-mode signed_mse
```

将 `--loss-mode` 改为 `scale_aligned` 并使用另一个 `--output-dir`，即可在
同一 teacher/student 配置上比较两种损失。脚本的 `--help` 列出采样和优化选项。
先按 teacher 的 `|ψ|²` 采样；默认从总步数的四分之一处起，按 1:1 合并
teacher 和 student 两组独立链的样本。
优先从 teacher checkpoint 的 walker 坐标开始，再按 teacher 分布热化；
如果 checkpoint 没保存可用坐标，则从核附近初始化。默认批次为 16，
可根据收敛 checkpoint 的链数和计算资源调整。

`signed_mse` 比较带符号的概率幅并拟合整体尺度；`scale_aligned` 每批为
student 求一个最佳整体乘数，再比较带符号概率幅。两个模式都使用同一批
预训练定义，只改变是否要求匹配全局尺度。

结果目录包含 `teacher_source/teacher.npz`、原始和补齐后的 teacher 配置、
`student_config_original.json`、可供 VMC 加载且已继承 teacher 分子信息的 `config.json`、
`pretrain_settings.json`、`pretrain_metrics.csv` 和
`pretrained_student.npz`。`signed_mse` 另有可训练的整体对数尺度，避免随机
SO(3) 初值与 teacher 幅度差很多时梯度消失；保存 checkpoint 时把它并入
SO(3) 行列式系数，并在 `calibration.json` 记录其最终值。
默认使用较小的 Adam 学习率，并在每步更新后用同一批样本检查损失；若损失
升高，就缩小更新，最多尝试 `--max-backtracks` 次。训练记录包含更新前后
的损失和实际使用的步长比例。

学生 checkpoint 沿用项目的 VMC 格式，但预训练 Adam 状态不会冒充 Spring
状态；后续恢复 VMC 必须设置 `--reload.new_optimizer_state=True`，例如：

```bash
python -m vmcnet.train.runners \
  --reload.logdir=../results/N2_teacher_pretrain \
  --reload.checkpoint_relative_file_path=pretrained_student.npz \
  --reload.new_optimizer_state=True \
  --reload.reburn=True
```

存下来的学生 `config.json` 默认把后续 VMC 结果写入与预训练目录并列的
`N2_teacher_pretrain_vmc` 目录；原始学生配置另存，便于比较。

这些流程只支持单个固定分子几何、实值 `ef_shrd_sym → so3`。学生训练与评估
自动沿用 teacher 的核顺序和电子自旋分块。原有 teacher 参数按 checkpoint 读取，输入配置缺少的新字段
只从当前默认配置补齐；补齐项会记录在 `pretrain_settings.json`，不能将补齐
误认为历史训练当时显式使用的参数。对早于 `dp_type` 和独立 Jastrow 激活
选项的普通网络配置，补全会使用当时的粒子类型编码和激活选择。预训练开始前
还会核对参数结构，并在 checkpoint 含 walker 幅值时比较保存的与重算的
`log|ψ|`；不一致则停止。若旧代码本身改变过层结构，仅补配置无法兼容，
需要原训练版本的网络实现。
