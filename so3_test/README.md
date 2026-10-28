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
