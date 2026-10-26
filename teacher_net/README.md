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
学生网络及优化参数，再额外传入：

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
`dtype` 必须与 teacher 一致，不会从 teacher 覆盖；请显式设置学生的物理体系。
启用 teacher 模式时会跳过 HF 预训练；已有 VMC checkpoint 的
续训不会重新做 teacher 预训练。能量诊断默认关闭，打开后结果位于
`teacher_pretrain/energy_samples.csv` 和 `energy_summary.json`。其余采样、损失、
步数和能量诊断选项与独立脚本同名，前缀改为 `--config.pretrain.teacher.`。
当前 teacher 预训练阶段要求单主机；同一主机上的多设备 VMC 可以在预训练后继续运行。

示例（学生配置提供网络、分子与优化设置，物理体系须与 teacher 一致）：

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
采样策略是独立于损失的选项：

- `--sampling-strategy dynamic_mixed`：保留原策略。训练时继续采样，默认从
  总步数的四分之一处起，按 1:1 合并 teacher 和 student 两组独立链的样本。
- `--sampling-strategy fixed_teacher`：teacher 热化后冻结训练点，训练中不再
  推进采样链，也不重新计算 teacher 标签。没有验证集，所有固定点全部训练。
  `--fixed-sample-count 0` 使用 `batch-size` 个点，每步使用整批固定点；设为
  大于 batch-size 的数量时，会先继续 teacher 采样收满，再打乱遍历全部点。
  尾批不丢弃，单点尾批并入前一批；训练步数必须足以覆盖整个数据集至少一次。
  数据保存为 `fixed_teacher_samples.npz`，含坐标、teacher 符号和 log|psi|。

例如固定 8192 个点，batch-size=512，所有点均用于训练：

```bash
# 在独立预训练命令中增加或替换这些选项
--sampling-strategy fixed_teacher --fixed-sample-count 8192 \
--batch-size 512 --loss-mode scale_aligned
```

一体化入口对应 `--config.pretrain.teacher.sampling_strategy` 和
`--config.pretrain.teacher.fixed_sample_count`。库默认仍为 dynamic_mixed；
`teacher_pretrain.sh` 已设为 fixed_teacher + scale_aligned，命令尾部仍可覆盖。
固定模式忽略 student_fraction、mix_start_step、student_burn_in；CSV 中
teacher_accept/student_accept 为 NaN，表示训练期间没有执行采样，而非数值错误。
能量诊断仍重新按最终 student 的 |psi|² 采样，不能把 teacher 固定点直接当作能量分布。

优先从 teacher checkpoint 的 walker 坐标开始，再按 teacher 分布热化；
如果 checkpoint 没保存可用坐标，则从核附近初始化。默认批次为 16，
可根据收敛 checkpoint 的链数和计算资源调整。

`signed_mse` 比较带符号的概率幅并拟合整体尺度；`scale_aligned` 每批为
student 求一个最佳整体乘数，再比较带符号概率幅。两个模式都使用同一批
预训练定义，只改变是否要求匹配全局尺度。

第三个选项 `--loss-mode relative_mse` 直接比较带符号比值：
`mean((student / teacher - 1)**2)`，保留可训练的 log_scale 来匹配整体幅度。
点来自 teacher 的 |psi|² 分布时，其期望为
`integral((student - teacher)**2) / integral(teacher**2)`，消除了原幅度 MSE
对高幅值 teacher 点的额外加权。该等价关系不适用于混入 student 样本后的分布。
建议与 fixed_teacher 配合；一体化入口设置
`--config.pretrain.teacher.loss_mode=relative_mse`。现有两种损失和默认值保留。
例如 `./teacher_pretrain.sh --loss-mode relative_mse` 会覆盖脚本中的损失设置。

比值用符号和 log|psi| 差计算，并用 expm1 保持接近匹配时的精度；
不加分母 eps、不裁剪比值、不剔除样本。teacher 精确为零时相对误差未定义，
或者近节点的比值导致数值溢出时，训练按现有非有限检查明确停止。
relative_mse 对整体符号相反的 student 仍有惩罚，区别于 scale_aligned。

结果目录包含 `teacher_source/teacher.npz`、原始和补齐后的 teacher 配置、
`student_config_original.json`、可供 VMC 加载且与 teacher 物理体系已核对一致的 `config.json`、
`pretrain_settings.json`、`pretrain_metrics.csv` 和
`pretrained_student.npz`。`signed_mse` 另有可训练的整体对数尺度，避免随机
student 初值与 teacher 幅度差很多时梯度消失；保存 checkpoint 时把它并入
student 输出参数，并在 `calibration.json` 记录其最终值。
预训练使用标准随机 Adam，每步正常推进参数和优化器状态，保留梯度范数裁剪。
同批更新后的损失仅用于监测，允许上升；不再执行更新拒绝或步长回退，
旧的 `--max-backtracks` 参数已移除。梯度、参数、Adam 状态或损失出现
NaN/Inf 时立即报错停止。训练记录包含 `loss_before`、`loss_after` 和裁剪前的
`gradient_norm`，不再包含 `update_scale`。

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

这些流程支持单个固定分子几何、实值波函数。teacher 仍为 `ef_shrd_sym`，
student 可选择 `wfn_type=gaoqiao` 下由 `build_network` 支持的 `gq.type`，
例如 `so3`、`ef_shrd_sym`、`ef_shrd_sym_proj`、`ef_shrd_sym_ablation`。一体化运行通过
`--config.gq.type=ef_shrd_sym` 等选项选择；独立运行在 student JSON 中选择。
普通 student 使用 `wfn_depth`、`h1`、`h2`、`nh` 等结构参数，不需要 `so3_params`。
其他顶层 `wfn_type`（例如 `psiformer`）不属于此入口的支持范围。
预训练入口沿用现有网络构建器，网络类型与 envelope 等配置必须能在原构建器中正常运行。
现有 `fermi` 分支存在 feature 接口参数不匹配，`ef_shrd` 在小体系测试中存在
轨道维度不匹配；放开 student 类型限制不会修复这些网络自身的问题。

teacher/student 的完整 `problem` 和 `dtype` 必须一致，否则停止；不会继承或覆盖
student 配置。仅在预训练重建 teacher 时补全其旧配置，student 和普通 reload
不自动补全。原有 teacher 参数按 checkpoint 读取，teacher 配置缺少的新字段
只从当前默认配置补齐；补齐项会记录在 `pretrain_settings.json`，不能将补齐
误认为历史训练当时显式使用的参数。对早于 `dp_type` 和独立 Jastrow 激活
选项的普通网络配置，补全会使用当时的粒子类型编码和激活选择。预训练开始前
还会核对参数结构，并在 checkpoint 含 walker 幅值时比较保存的与重算的
`log|ψ|`；不一致则停止。若旧代码本身改变过层结构，仅补配置无法兼容，
需要原训练版本的网络实现。

保存时，SO(3) 的整体幅度写入 `so3_det.alpha`；普通 gaoqiao student 将轨道
输出层乘以 `exp(log_scale / N)`（N 为电子数），使行列式整体乘以 `exp(log_scale)`。
RHF 对共享的 `RHF_orbital` 处理，其他网络对 `orbital` 处理。参数树结构不变，
保存后的 checkpoint 可直接由同配置的 VMC 网络读取。
