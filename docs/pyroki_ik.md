# PyRoki 离线 IK

当前双臂与 arm7 pipeline 使用的 `ArmModel.inverse()` 已从自写 SciPy 最小二乘切换到 PyRoki。参考 [PyRoki 官方 IK 示例](https://github.com/chungmin99/pyroki/blob/388e43e1fc0d0ee382968d3dd72970fd62a0450c/examples/pyroki_snippets/_solve_ik.py)：使用解析姿态雅可比、jaxls 最小二乘及关节限位约束，另外加入很小的前一关节解正则项以维持连续性。

安装到现有环境：

```bash
/home/sja/miniconda3/envs/cali/bin/python -m pip install -r requirements-ik.txt
```

本次环境：Python 3.10、CPU JAX/jaxlib 0.6.2，PyRoki commit `388e43e1fc0d0ee382968d3dd72970fd62a0450c`，jaxls commit `50a58be88c5ef74532f09e3f55268b4f02c490e3`。PyRoki 的上游依赖包含浮动的 jaxls Git 地址；复现本次 jaxls 版本可在安装后运行：

```bash
/home/sja/miniconda3/envs/cali/bin/python -m pip install --no-deps 'jaxls @ git+https://github.com/brentyi/jaxls.git@50a58be88c5ef74532f09e3f55268b4f02c490e3'
```

适配器仅提取原 URDF 的 `link_base` 到物理法兰链，追加当前 SDK TCP 的六维固定偏移，不把夹爪驱动关节作为 IK 自由度。显式按关节名映射顺序；公共接口仍为基座坐标 mm/deg，PyRoki 内部使用 m/rad、双精度。模型按需创建，JIT 结果在同一模型的后续调用中复用。

先以前一关节解初始化；失败时再尝试 Home 和限位中点，并以原始 seed 作为正则参考。不存在自动回退到旧 SciPy 的生产路径。结果必须通过独立检查：有限值、关节范围、原 yourdfpy FK 的位置误差不超过 0.5 mm、姿态误差不超过 0.2°。IK 合格不代表路径无碰撞，后续原有碰撞/停止范围验证保持不变。历史动画类的 `solve()` 未改，当前 pipeline 不调用这些类。

离线验证和对照：

```bash
/home/sja/miniconda3/envs/cali/bin/python -m pytest -q tests/test_pyroki_ik.py
/home/sja/miniconda3/envs/cali/bin/python scripts/benchmark_pyroki_ik.py
```

脚本不会连接相机或机械臂。`results/pyroki_ik_migration/comparison.json` 保存旧实现与 PyRoki 的逐目标对照。本次两臂各 5 个 Home 附近抬升目标，热调用中位耗时约 SciPy 28.7 ms、PyRoki 2.14 ms；首次调用另有 JIT 编译开销，初步测试约每臂 1.4–1.9 秒。该小样本不代表所有目标或整体规划耗时。

已保存的失败目标 `loop_20261009T125018455816Z/round_001` 在两种求解器下均被拒绝：旧实现为 5.727 mm / 0.092°，PyRoki 为 2.094 mm / 0.837°。二者目标函数权重不同，不能只比较位置误差来宣称已解决。切换 IK 不会自动修复不合适的终点姿态，也不会消除停止包络区间证明的耗时。
