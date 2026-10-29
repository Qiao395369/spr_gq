# SO(3) 物理性质测试

从项目根目录执行 `bash so3_test/run_all_test.sh`。脚本按文件逐项运行测试，遇到失败立即停止。也可以手动运行：

```bash
python -m unittest discover -s so3_test -p 'test_*.py' -v
```

`test_rotation_l0.py`、`test_rotation_l1.py`、`test_rotation_l2.py` 分别检查主干特征和读出在实基、复基及末端转实基模式下的旋转规律。网络测试使用球谐在独立探针方向上的取值求出预期矩阵；为了避免只检验代码与自身一致，另从笛卡尔向量（l=1）和对称无迹二阶张量（l=2）独立推导旋转矩阵，逐项核对实基与复基结果。l=2 还检查矩阵的酉性。

`test_representation_matrix.py` 对单位旋转、两个单轴旋转及其复合旋转逐项核对表示矩阵，并检验 `D_l(R_z R_y)=D_l(R_z)D_l(R_y)`。这些检查仍是有限样本的数值测试，不能代替对所有旋转的数学证明。

`test_coalescence.py` 在 `regularize_angular_origin=True` 时检查电子从两个方向趋近同一原子核，完整波函数的方向差随距离缩小而消失。旧配置默认关闭该选项，以保持已有 checkpoint 的含义；新实验可显式开启。

`test_scalar_wavefunction_rotation.py` 检查仅选择 `target_ls=(0,)` 时，完整波函数在电子和原子核共同旋转下不变。`test_exchange_antisymmetry.py` 分别交换两组同自旋电子，检查整体相位变号、对数绝对值不变；不对异自旋交换作错误的变号要求。

测试使用 JAX 双精度。交换反对称性检查 `(0,1,2)` 下的 `shared` 行列式库、`per_l` 实基库，以及仅有 `(0,)` 的标量输出。`target_ls=(0,1,2)` 配合固定 `q0` 投影一般不是旋转不变标量，因此完整波函数旋转测试特意使用 `(0,)`。

## 独立 orbital-like 网络 so3_l0

选择 `--config.gq.type=so3_l0`，结构参数来自独立的
`--config.gq.so3_l0.*`，例如 `channels=16`、`lmax=2`、`n_layers=3`。
精度仍取顶层 `--config.dtype`；行列式总数直接取 `--config.gq.ndet`。

每层都是 Conv → WBN；最后的 Conv 保留各隐藏阶，最后的 WBN 仅输出
l=0。末层仍保留 l×l→0 耦合。没有 A 读出、rank 或 target_ls 配置，
也不再读取高阶 RMS：轨道头直接读取末层零阶特征。
real、complex、complex_to_real 都可作为主干模式；两种复主干的零阶
实/虚部作为两组实特征送入实轨道头，最终始终是实波函数。

Kato Jastrow 通过 exp(J/Ne) 乘每个轨道元素。轨道接口与普通网络一致，
返回一个只含 (ndet,Ne,Ne) 矩阵的列表，矩阵已包含核包络和 Jastrow；
波函数直接对这些矩阵的行列式求和，不再乘第二次 Jastrow。

`test_so3_l0.py` 核对层的顺序与裁剪、Jastrow 恒等式、旋转/平移和
同自旋反对称性、实复主干、空自旋组、参数梯度与坐标 Hessian，
并检查 teacher 轨道预训练的 Adam 更新与 checkpoint 尺度写回。
`pretrain_teacher_orbitals` 已支持 ef_shrd_sym teacher → so3_l0 student，
旧 so3 的带 A/rank 轨道库仍由完整波函数预训练入口处理。

例如沿用现有 CH4 轨道脚本，仅覆盖 student 结构：

```bash
./teacher_orbital_pretrain.sh \
  --config.gq.type=so3_l0 \
  --config.gq.so3_l0.channels=16 \
  --config.gq.so3_l0.lmax=2 \
  --config.gq.so3_l0.n_layers=3
```


## 原版 so3 的无约束轨道读出

仍选择 `gq.type=so3`，设置 `gq.so3_params.readout_mode=mlp`。
主干的卷积和 WBN 顺序保持原样；对每个电子分别展开 l=0,...,lmax 的
全部 channel 和角分量，按 l 拼成一个向量，不求范数、不作全局粒子求和。
实基输入维数为 C*(lmax+1)^2；复主干保留实/虚部，为两倍维数。

`readout_mlp_layers` 包含输出线性层：1 表示直接映射到 ndet*Ne，
2 表示一层隐藏层加输出层，以此类推；隐藏层宽度用 `readout_mlp_width`
设置，激活用 `readout_mlp_activation`（silu 或 tanh）。同自旋电子
共享轨道输出头，隐藏层在所有粒子间共享；不再有 A、rank、alpha 或 sector 库。
总行列式数直接为 `gq.ndet`，轨道接口与 sym 一样返回 `[Phi]`。
旧的 target_ls、det_rank、det_bank_mode、use_invariant_orbital_features、
output_projection 对此模式均无效。包络继续使用 so3 的核指数包络，
Kato Jastrow 以 exp(J/Ne) 乘入轨道，保证轨道监督与波函数完全一致。

```bash
--config.gq.type=so3
--config.gq.ndet=16
--config.gq.so3_params.readout_mode=mlp
--config.gq.so3_params.readout_mlp_layers=1
```

设置 `readout_mode=equivariant` 恢复旧读出（默认值），切换模式须重新
初始化网络。`so3_l0` 独立实现与配置不变。新模式保持主干等变性、整体
平移不变性和同自旋交换反对称性，但不保证轨道和波函数的旋转规律。
`pretrain_teacher_orbitals` 支持 sym teacher → MLP 读出的 so3 student。

`test_mlp_readout.py` 检查展开顺序、直接线性映射、MLP 层数、三个 basis、
Jastrow 只乘一次、ndet 数目、忽略旧读出参数、平移/交换性质、主干旋转规律、
轨道的方向敏感性、参数梯度与坐标二阶导数，以及 Adam/尺度保存/VMC 接入。
