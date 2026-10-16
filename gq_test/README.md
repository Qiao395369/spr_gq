# gaoqiao 网络的 N₂ 性质测试

本目录直接调用 `vmcnet.gaoqiao.build.build_network`，不启动采样、预训练或 VMC 优化。测试物理系统与 `train_N2_single.sh` 一致：两颗氮核位于 `(0,0,±1.034)`，核电荷各为 7，14 个电子分成 `(7,7)` 两个自旋块。`single_nspins=((5,2),(2,5))` 也与总自旋数一致。

网络分支保留训练配置的 `ef_shrd_sym`、`dp_type=full`、`ds_hz_old` 包络、`mlp_res` Jastrow、`rescale_input=all`、`silu/tanh` 激活及 `float64`。为了快速测试，改为 `ndet=2`、`wfn_depth=2`、`h1=8`、`h2=4`、`nh=4`、Jastrow `2×8`；约 2276 个参数。**这不是训练配置的能量或收敛测试。**

在项目根目录、安装了项目依赖的 Python 环境中运行：

```bash
bash gq_test/run_all_test.sh
```

也可以指定解释器，例如 `PYTHON_BIN=/path/to/python bash gq_test/run_all_test.sh`。脚本按文件枚举测试，逐个报告结果，最终退出码反映是否存在失败。

测试范围：有效构型上的有限实值输出与 `float64`、整体平移不变性、同自旋电子奇/偶置换的符号、同种氮核换序不变性，以及 `log|ψ|` 对电子坐标的前向和反向一阶导数。这些性质对固定随机初始化进行检查，不代表训练后的能量质量。普通 `ef_shrd_sym` MLP 没有被构造成严格的旋转不变波函数，因此这里不把任意旋转不变性作为必过测试。

## 当前已发现的问题

对于训练脚本实际使用的 `rescale_input=all`，网络值与前向坐标导数有限，但反向坐标梯度包含 NaN，因此 `test_reverse_coordinate_gradient_is_finite` 当前会失败。这不是电子重合导致的：测试构型中所有电子坐标不同。`vmcnet/gaoqiao/open_feature_layer.py` 的 `factor=jnp.where(log_r_pp!=0, log_r_pp/r_pp, 0.0)` 在自身配对的 `r_pp=0` 处计算了 `0/0`；`where` 虽然把前向值选成 0，反向自动微分仍可经过该无效分支传播 NaN。相同网络在已有 `all_eps` 缩放分支下，反向梯度有限。这里保留失败测试，避免把生产配置悄悄改成另一种输入特征。

训练脚本中还有一些与该网络分支无关或目前不生效的选项：`psiformer_multi`、`ferminet_multi`、`lapnet_multi` 及 Psiformer/LapNet 尺寸属于其他波函数分支；`do_attn=False` 时注意力尺寸不使用；选择 Spring 优化器后 KFAC 学习率不使用；`pretrain.iterations=0` 时预训练步数与优化设置不生效；`repeat_single_mol=False` 时相应 walker 重复数不生效；`down_sample_num=0` 时下采样模式不生效。`fwdlap_inner_size=42` 与 `14×3` 个电子坐标一致。

`test_teacher_pretrain.py` 另检验 teacher→student 的两种带符号损失、旧配置补缺、整体幅度折叠，以及优先使用 checkpoint walker。
