# 连续抓取姿态优化与双臂轨迹规划

阶段 4 的连续 13DoF IK、阶段 5 的离线路径与时间参数化，以及阶段 7 的 `DualArmPlanner` 接口已实现。抓布流程目前仍提供离线执行预检；Home 已通过 `motion/home.py` 接入现有位置模式执行器。450 mm 已由用户确认为错误计算并撤销，不再阻塞开发。

运动规划位于 `cloth_agent/dual_arm/motion/`。`gripper_home()` 通过 Home 专用适配器调用 `PathPlanner` 和时间参数化；抓布 API 不会连接硬件。Home 使用当前 `dual_arm.local.json` 中保存的模型、基座、附件包络、工作空间和停止界限，不加载演示 FCL 场景或重新要求配置附件尺寸。

### Home 执行接入

运行 `/home/sja/miniconda3/envs/cali/bin/python scripts/gripper_home.py` 会读取两臂当前关节位置，用新规划器搜索依次回位路径，再交给现有位置执行器。距离较近时先规划短分离前缀；直接路径不通过时用 OMPL 搜索；若两种顺序均失败，即使起始间距已超过近距离阈值，仍会补试短分离前缀。整条路线和所有下发指令通过检查后才开始移动，最后打开两个夹爪。`--simulate` 使用模拟连接，`--preflight-only` 仅读取状态和规划。

Home 路径搜索使用当前运行时碰撞与停止包络适配器，而不是演示 FCL 盒体。平滑多项式的逐轴极值也会检查；实际下发仍是 `set_servo_angle(wait=True, radius=-1)`，每条指令独立验证整个关节区间。不会把每个时间采样点变成一次阻塞指令。控制器指令边界仍会停稳，因此参考曲线的连续速度不代表实际位置指令之间无停顿。

`plan.json` 明确记录 `planner`、`collision_backend`、`reference_timing`、`controller_interpolation` 和 `controller_segments`。真实姿态能否回位取决于当次完整规划结果；Home 接入不等于任意抓布轨迹已支持真机执行。

## 文件与分阶段实现

| 阶段 | 文件 | 技术方案 |
| --- | --- | --- |
| 4 | `ik.py` | SciPy least-squares 多初值生成 + SLSQP 连续约束优化，13 个关节均为变量；FCL 距离为不等式约束 |
| 5 | `path.py`、`validation.py` | OMPL RRTConnect + 自定义连续区间检查；笛卡尔路径使用固定抓取朝向的连续 IK |
| 5 | `trajectory.py` | 双臂统一时间轴；分段五次曲线、解析速度/加速度上界；时间参数化后重新检查 |
| 5 | `viewer.py` | 复用碰撞 Viser，增加播放/时间滑块、阶段与距离显示 |
| 6（离线部分） | `preflight.py` | 校验轨迹与模型摘要、时效、双臂采样时间差、起始状态及轨迹内容；明确拒绝真实执行 |
| 7 | `planner.py`、`__init__.py` | 抓取点到完整阶段轨迹的公开 API；失败返回阶段和原因，不返回部分可执行轨迹 |
| 工具与测试 | `__main__.py`、`requirements-motion.txt`、`tests/test_dual_arm_motion.py` | 离线命令和实际 FK/FCL/OMPL 回归 |

碰撞模块新增了可选 `check(..., details=True)`，返回稳定的各对距离/间隙供优化器和区间检查使用。原默认返回格式兼容。

本次优先复用已有 FK/FCL，因此选用 SciPy 连续约束优化，没有再建立 Drake/Pinocchio 的平行机器人模型。它支持连续旋转、限位和碰撞约束，但依然是非凸局部搜索；找不到结果表示在预算内未找到可行解，不是数学上的不可达证明。未来可在同一接口下加入 Drake 后端。

## 阶段 4：约束与目标

所有位置使用 World（xArm6 `link_base`）中的米，关节角使用弧度。两个 TCP 位置必须落在默认 1 mm 容差内；TCP 的局部 +Z 为接近轴，默认朝向 World -Z，允许 15° 锥角。局部 +X 作为夹爪抓取方向；必须在实机验收中核实这些轴与安装指尖一致。

`edge_a/edge_b` 可提供衣物边缘方向。先投影到接近方向的垂面，然后约束夹爪 +X 与其平行，允许夹爪的 180° 对称性，默认误差不超过 25°。如果未提供边缘方向，结果会标记 `edge_direction_supplied=false`；两个点本身无法确定实际衣物边缘方向或抓取稳定性。LLM 不参与碰撞判定。

夹爪偏航角没有预设角度表，xArm7 的肘部姿态随七个关节一起连续优化。关节初值使用当前角度及连续随机扰动，不是 0°/90° 抓取姿态枚举。

优化目标为：

```text
J = motion_weight * 归一化关节位移平方和
  + clearance_weight * 小于 preferred_margin 的各对间隙惩罚
  + grasp_weight * 接近方向与衣物边缘对齐误差
  + joint_weight * 偏离关节区间中心的四次惩罚
```

默认权重依次为 `1.0 / 0.3 / 0.2 / 0.05`。间隙惩罚在达到默认 40 mm 剩余间隙后归零，不会无限追求最大间距。硬约束仍要求全部碰撞对满足 scene 安全间隙和误差余量，另留默认 1 mm 优化缓冲。通过 `IKOptions` 调整；没有把这些权重宣称为衣物任务的最终最优设置。

默认 5 个初值，每个最多 120 次迭代，总求解预算 30 s。无论优化器是否宣称收敛，都要独立核对 FK、方向和碰撞约束。接触/重叠距离为 0，深度碰撞区的距离梯度可能不足，因此多初值仍可能失败；此时明确拒绝。

```python
from cloth_agent.dual_arm.collision import CollisionScene
from cloth_agent.dual_arm.motion import DualArmPlanner
from cloth_agent.dual_arm.motion.ik import IKOptions

scene = CollisionScene.load("measured_collision_scene.json")
planner = DualArmPlanner(scene, ik_options=IKOptions(starts=5))
pose = planner.solve_grasp_pose(
    point_a=point_a_world_m,
    point_b=point_b_world_m,
    current_q_a=q_a_rad,  # 6 个关节角
    current_q_b=q_b_rad,  # 7 个关节角
    edge_a=edge_a_world,  # 可选方向向量
    edge_b=edge_b_world,
)
```

成功返回 `q_a/q_b`、4×4 的 World TCP `grasp_pose_a/b`、位置和接近角误差、目标值、最小间隙与多初值诊断。成功仍不授权真实运动。

## 阶段 5：路径、时间轴与连续检查

默认阶段为：

1. 联合优化抓取姿态。
2. 从抓取姿态向 World +Z 连续求解预抓取路径，再反向作为下降段，保持已选抓取朝向。
3. 当前状态到预抓取：先证明直线路径；不通过时用 OMPL RRTConnect 寻找绕行路径。
4. 按下降段同时到达抓取点，添加闭合夹爪、确认抓住的事件。
5. 双臂同时抬升，再沿两点的水平连线向外展开。`spread_m` 是两点间距的总增量，两臂各移动一半。
6. 可选 `return_after_release=True`：先添加释放确认事件，再规划返回初始关节状态；未确认释放的真机不能执行这段。

`order=auto` 对前往预抓取的路径依次尝试同时运动、左先右后、右先左后。顺序路径固定未运动臂的全部关节。抓取、抬升、展开始终同时进行。自动顺序是依次尝试可行方案，不是全局最优排序；某个抓取解后续路径失败时不会伪造安全轨迹。

笛卡尔路径默认每 5 mm 求解一次完整 TCP 姿态，限制相邻 IK 分支跳变。每个关节插值段另外证明其 TCP 在目标路径的 1 mm 位置误差和 1° 朝向误差内。没有依赖“端点 IK 正确，所以中间也正确”的假设。

连续碰撞检查以关节区间中心的 FCL 实体距离为基础，按运动学链的三角不等式估计每个几何体相对各祖先关节的最大位移。如果距离减去两物体可能位移仍大于所需间隙，整个区间才通过；否则递归细分。真实相交中点、节点/时间预算耗尽或无法证明的区间均拒绝。OMPL 的 `MotionValidator` 也使用同一检查，不接收近似终点解。

时间参数化使用共享节点速度的五次 Hermite 曲线：同一阶段内部通过 PCHIP 导数确定连续关节速度，节点加速度为零；阶段边界、静止段及活动机械臂切换处停稳。两臂共享时间轴，不再在每个 5 mm 细分点停车。解析求解多项式导数的极值后，对整个阶段统一缩放时间，满足关节速度和加速度上限。默认上限与当前 Home 配置一致，为 5°/s、10°/s²（API 仍使用弧度）。这不是笛卡尔恒速保证，也未接入真机执行。

时间参数化后重新认证所有段。用多项式极值计算曲线偏离原关节直线的逐轴上界，将它加入区间包络后重新检查碰撞和笛卡尔误差；无法证明时拒绝，不沿用原直线证书。`minimum_clearance` 是全轨迹连续证明得到的保守距离下界，单位米，通常低于离散采样得到的最小值。它只约束名义机器人几何，不包含实际执行偏差、停止距离、衣物张力/形变或衣物与机器人的碰撞。

```python
result = planner.plan(
    grasp_a=point_a_world_m,
    grasp_b=point_b_world_m,
    current_q_a=q_a_rad,
    current_q_b=q_b_rad,
    task="dual_grasp",
    approach_m=0.05,
    lift_m=0.05,
    spread_m=0.05,
    order="auto",
)
```

`task=center_pair` 是同一运动原语的别名，输入已经是两个确定的 3D 点；图像中心选择、像素深度 grounding、中心邻域核验仍由已有 ClothAgent 层负责。`pin_pull` 保留在原有流程，新规划器暂拒绝它，因为压住衣物需要接触控制。

成功返回所需的 `grasp_pose_a/b`、`trajectory_a/b`、`minimum_clearance`、`execution_order`，还包含五次插值段、阶段、事件、模型和内容摘要。两条 trajectory 都有相同 `times_s`，并提供角度、速度、加速度。`events` 只是离线标注，不能把列表直接作为无人值守执行队列。

## 阶段 6：当前可做与尚未接通的部分

`preflight` 接收调用方已采集的双臂关节状态，检查：轨迹生成后默认 120 s 时效；当前角度与起点相差不超过 0.005 rad；两臂反馈不老于 100 ms、采样差不超过 20 ms；模型、标定和几何摘要一致；轨迹内容未改动；公共时间轴、解析动态上限和连续碰撞证明有效；测量状态到计划起点的微小偏差也通过检查。

时间戳为同一主机的 Unix 秒。离线核验信任输入时间戳的真实性，不能代替 SDK 数据采集。结果包含 `execution_authorized=false`，`real=True` 明确拒绝；不会自动连接机械臂。

原有 `DualArmCoordinator` 已包含同步读取、跟踪误差检测、异常时停止两臂，以及 servo/停止包络检查。本次没有绕过它另建真机执行器。新轨迹接入它之前仍需：

- 补齐实测相机、支架、夹爪、桌面，以及 FK/基座精度验收。
- 验证 SDK 控制模式实际轨迹与五次轨迹或发送目标的关系，不能把这些点交给内部位置规划后宣称检查过真实轨迹。
- 将跟踪误差、通信/调度延迟、两臂时间差、控制器看门狗和制动位移纳入新 FCL 区间边界，并验证最坏检查耗时。
- 把夹爪/抓住确认事件接入已有观测与停止逻辑；先完成无电机运动的配对状态演练，再在明确授权后低速测试。

因此本轮不宣称阶段 6 真机验收完成。

## CLI、Viser 与自动化验证

可选依赖安装到已有项目环境：

```bash
python -m pip install -r requirements-motion.txt
```

当前依赖安装于 `cali` 环境，可把命令中的 `python` 替换为：

```bash
/home/sja/miniconda3/envs/cali/bin/python
```

请求 JSON 包含 `units: "m_rad"` 及上述 API 的参数，可选 `ik_options`。`solve` 的参数名为 `point_a/point_b`，`plan` 为 `grasp_a/grasp_b`。输出文件不覆盖已有内容。真实 scene 不完整时直接报错，不会自动替换成演示几何。

```bash
python -m cloth_agent.dual_arm.motion solve --scene scene.json --request grasp_request.json --output grasp_pose.json
python -m cloth_agent.dual_arm.motion plan --scene scene.json --request plan_request.json --output trajectory.json
python -m cloth_agent.dual_arm.motion view --scene scene.json --plan trajectory.json --port 8767
python -m cloth_agent.dual_arm.motion preflight --scene scene.json --plan trajectory.json --state paired_state.json --output preflight.json
```

`paired_state.json` 字段为 `units: "rad_s"`、`current_q_a`、`current_q_b`、`sampled_at_a`、`sampled_at_b`。真实记录不要伪造时间戳。Viser 可以回放过期轨迹，但会重新核对模型和轨迹内容，执行预检仍拒绝过期轨迹。

本机演示产物在 `/tmp/cloth_motion_scene.json`、`/tmp/cloth_motion_plan.json`，请求示例为 `/tmp/cloth_motion_request.json`，使用合成附件和桌面。打开 <http://127.0.0.1:8767>，点击 `Play` 或拖动 `Time (s)`，检查各阶段、相机随动、最近对和距离；官方视觉网格与碰撞网格都可隐藏。关节手动滑块在轨迹模式禁用，避免把手动姿态与计划回放混淆。夹爪事件和衣物动力学没有动画模拟。

```bash
python -m pytest -q tests/test_dual_arm_motion.py tests/test_dual_arm_collision_scene.py
```

运动规划新增 19 项通过：连续 IK 与非预设边缘方向、不可行返回、端点安全但中间碰撞、真实 OMPL 绕障、两种顺序路径、同时抬升展开、释放后返回、公共时间轴与解析动态峰值、笛卡尔段内部误差、预检正常与过期/失步/状态改变/内容改变/真机请求拒绝、重新时间参数化与采样损坏、模型改变拒绝。实际 OMPL 与 FCL 后端运行，非模拟算法替身。

连同已有双臂 Home、停止包络、跟踪误差、工作空间、runtime 和安全模式检查，共 197 项回归通过。Ruff 检查通过，规划 CLI 完成实际场景求解，轨迹 Viser 启动且 HTTP 返回 200；尚未完成浏览器中的人工几何验收。预检中的反馈新鲜度是对输入记录的入口检查，离线轨迹复核可能超过反馈有效期；未来真实下发前必须重新采集双臂状态。

尚未解决：真机执行、衣物抓取稳定性与张力约束、深碰撞初值的局部搜索失败、不同抓取解之间的后续路径可行性联合优化。当前采取显式失败，不强行执行。

当前 xArm7 默认采用 `assets/robots/xarm7/xarm7_controller_fit.urdf`，与现有 Home 使用的控制器拟合运动学一致；碰撞网格仍来自原模型。

Home 搜索边可以递归分成多个已校验的关节直线区间；这不代表整个端点盒已通过。平滑曲线另用多项式极值包络递归检查，实际控制器指令仍做完整关节盒校验。工作空间、高度和走廊的快速检查先于昂贵的停止包络证明。RRT 失败会记录状态/边检查次数及主要拒绝原因，预算耗尽不表示物理上无路可走。
