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
