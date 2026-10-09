# 双臂模型审计与静态碰撞检测（阶段 1～3）

本模块复用现有 `ArmModel`、yourdfpy FK、官方 URDF 和 Viser，新增 FCL 实体碰撞与距离查询。不连接机械臂，也不调用 SDK。后续连续 IK、轨迹规划与离线执行预检见 [运动规划说明](dual_arm_motion.md)。

**当前完成的是离线软件实现和测试，尚未完成实机几何与标定验收。** `safe=true` 仅表示输入关节状态在配置的静态几何模型中满足间隙，不证明真实运动、制动距离或标定精度。所有结果都包含 `execution_authorized=false`。现有 runtime 的胶囊体、停止包络检查与执行入口保持不变。

## 新增文件与技术选择

| 阶段 | 文件 | 实现与验证 |
| --- | --- | --- |
| 1 | `cloth_agent/dual_arm/collision/audit.py`、`calibration.py` | 审计官方关节顺序、限位、collision mesh、历史 Home FK、标定来源和基座变换；支持导入离线控制器与距离测量记录 |
| 2 | `scene.py`、`config/dual_arm.collision.example.json` | 通过独立 JSON 添加相机、夹爪、支架、电缆包络和桌面；相机光学系与外壳中心分开配置 |
| 3 | `checker.py`、`viewer.py` | python-fcl 检测实体接触/重叠和最近距离；Viser 显示官方模型、碰撞几何和最近点连线 |
| 工具 | `__init__.py`、`__main__.py`、`requirements-collision.txt` | Python API 和离线命令行；可选依赖独立安装 |
| 测试 | `tests/test_dual_arm_collision_scene.py` | 实际 URDF/FCL 查询、异常输入、标定记录与 Viser 更新测试 |

本阶段使用已有 FK 加 FCL，无需引入 Pinocchio 或 Drake 的第二套运动学模型。每个官方 collision mesh 转为独立凸包，保留 mesh scale 与 collision origin。凸包包围原网格，可能保守地多报碰撞，但不会把完全包含于实体内的小物体误判为无碰撞。后续连续优化可以复用这一碰撞接口；本次不实现优化器。

## 环境与运行

在已经安装项目依赖的 Python 环境中安装可选依赖：

```bash
python -m pip install -r requirements-collision.txt
```

当前使用 `cali` 环境，FCL 与 OMPL 直接安装到该环境。以下命令中的 `python` 可替换为：

```bash
/home/sja/miniconda3/envs/cali/bin/python
```

生成审计和明确标为 synthetic 的演示场景：

```bash
python -m cloth_agent.dual_arm.collision audit --output /tmp/dual_collision_audit.json
python -m cloth_agent.dual_arm.collision init --synthetic --output /tmp/dual_collision_demo.json
python -m cloth_agent.dual_arm.collision check --scene /tmp/dual_collision_demo.json --output /tmp/dual_collision_check.json
python -m cloth_agent.dual_arm.collision view --scene /tmp/dual_collision_demo.json --port 8766
```

输出文件使用独占创建，不覆盖已有记录；重复运行时指定新的路径。`check` 的退出码：0 表示模型间隙通过，2 表示碰撞或间隙不足，1 表示配置/输入/后端错误。`audit` 正常生成报告时退出 0，必须另看 `physical_acceptance`，它不是实机验收命令。

打开 <http://127.0.0.1:8766>。此入口只监听回环地址，可通过 SSH 端口转发访问。界面有 13 个关节滑块，显示单位为度、内部为弧度；xArm7 原生夹爪的额外驱动关节仅用于显示，不计入 13DoF 输入。

Viser 验证步骤：

1. 检查两侧官方模型和碰撞网格是否重合，使用 `Show official visual meshes`、`Show collision geometry` 分别开关。
2. 改动两侧末端关节，观察各自相机盒、夹爪盒随腕部运动。
3. 确认最近两个几何体标红，非重叠时显示最近点连线和米制距离。重叠距离为 0，最近点连线不显示。
4. 构造靠近或相交姿态，查看 `UNSAFE`、`collision` 和间隙变化。演示场景的相机外壳、夹爪和桌面为合成尺寸，不能用于实机验证。
5. 补齐实测配置后，用同一时刻记录的真实关节角替代 `initial_q_rad`，重新打开查看。必须另外核验真实位置，视觉重合不能代替测量。

实际 Viser 服务已完成启动和 HTTP 200 冒烟验证；自动化测试还检查了双臂关节向量、相机跟随和最近对状态更新。尚未完成浏览器中的人工几何验收。

## 阶段 1：模型与标定发现

输入顺序固定为 `left/joint1..6`（Arm A/xArm6）和 `right/joint1..7`（Arm B/xArm7）。限位直接读取 URDF；非有限值、错误长度和越限状态被拒绝。内部适配边界将原 `ArmModel` 的毫米/度转换为米/弧度。

官方 xArm6 的基座及六个连杆、xArm7 的基座及七个连杆均有 collision mesh，资源存在。xArm6 模型没有原生夹爪；xArm7 原生夹爪有 collision mesh，但部分指尖 link 无独立碰撞体。本模块对两侧统一使用固定包络盒覆盖夹爪全部开合状态。右侧被替换的七个 collision link 会在结果 `gripper_meshes_replaced_by_envelopes` 中列出；额外未覆盖的 collision link 会导致加载失败。

标定读取 `config/calibration/dual_arm_working_20261008/`，核对 `import_manifest.json` 的 SHA256，并检查矩阵和米制单位。当前约定：

```text
World = xArm6 link_base
p_world = X_Base1Base2 @ p_xarm7_base
X_Base1Base2 的平移 = [1.1260379512, -0.0194824079, 0.0165630688] m
```

| 证据 | 结果 | 含义 |
| --- | --- | --- |
| 基座原点距离 | 1.126328 m | 由导入的平移向量范数计算 |
| 历史 Home 的 TCP 间距 | 0.242668 m | 使用当前 FK 和基座矩阵计算，不是实测 |
| xArm6 官方 FK 对历史 TCP | 约 0.000463 mm、0.000419° | 通过 2 mm / 1° 的单样本数值阈值 |
| xArm7 官方 FK 对历史 TCP | 约 4.367 mm、1.909° | 未通过该阈值 |
| xArm7 controller-fit 模型 | 约 0.0142 mm、0.00845° | 当前默认使用，与 Home 运行时一致；此行是历史 Home 样本的数值结果 |

历史记录来自 `data/robot/dual_arm_home.json`，时间为 `2026-09-26T05:40:08.682481+00:00`，不是本次同步采集的真实状态。标定状态为 `applied_coordinate_correction_pending_live_validation`。用户已确认此前的 450 mm 来自错误计算，该项撤销，不再作为验收条件或后续开发阻塞项。

## 阶段 2：补齐实测碰撞几何

`config/dual_arm.collision.example.json` 是**故意不完整**的实测模板。相机外壳中心变换、夹爪尺寸/位姿和桌面参数为 `null`，直接执行 `check` 或 `view` 会失败。不要把演示值复制到真实环境中当作测量值。

相机盒的变换链为：

```text
world_from_housing(q) = world_from_base
                       @ base_from_mount_link(q)
                       @ link_from_optical
                       @ optical_from_housing
```

`link_from_optical_m` 来自已有 `X_CammountCam`；源标定没有明确 URDF mount link，目前模板假定为 `link_eef` 并在 provenance 标记待确认。`optical_from_housing_m` 必须给出盒子坐标系到光学系的旋转和平移。不能把相机光学原点当成外壳几何中心。模板中的 D435 尺寸 `[0.09, 0.025, 0.025]` m 也是标注为待核验的名义示例。

测量支架、接头和电缆的实际外伸部分，扩大包络或增加 `kind=bracket/cable` 附件。普通附件使用 `link_from_box_m` 与 `size_m`；相机使用上述两个变换。刚体盒不能代表任意摆动电缆，包络必须覆盖计划使用范围。夹爪盒必须覆盖所有开合状态和安装指尖，而非只覆盖当前闭合外观。

桌面采用 World 中有厚度的 Box。只允许为固定 `link_base` 配置有理由的 `base_mount_contacts` 安装接触豁免。夹爪、相机和活动连杆与桌面的碰撞一直检测；后续衣物接触任务需要专门设计接触约束，不能在这里关闭桌面检测。

## 阶段 3：API 与碰撞对

```python
from cloth_agent.dual_arm.collision import CollisionChecker, CollisionScene

scene = CollisionScene.load("/path/to/measured_collision_scene.json")
checker = CollisionChecker(scene)
result = checker.check(q_a, q_b)  # 6 和 7 个关节角，单位 rad
```

`collision` 表示实体接触或重叠；`min_distance` 是所有有效对的最小无符号实体距离，重叠时为 0；`closest_pair` 与 `nearest_points_m` 用于可视化。所需间隙为：

```text
safety_distance_m + geometry_uncertainty_m[owner_a] + geometry_uncertainty_m[owner_b]
```

所有对必须无碰撞且距离严格大于所需间隙才返回 `safe=true`（1e-9 m 数值保护）。`minimum_margin_m` 和 `limiting_pair` 反映扣除间隙要求后的最危险对；设置不同误差余量时，它可以不同于几何距离最近对。零误差余量只是默认演示设置，真实余量需由 FK、基座标定、附件测量误差决定。

检查跨臂所有模型实体、相机、夹爪；检查非相邻自碰撞以及所有机器人实体与环境。只排除同 link、直接 URDF 关节邻居、附件与其刚性宿主 link、明确声明的基座安装接触。相机与同侧夹爪仍检测，不接受任意 `disabled_pairs`，跨臂对不能豁免。

只检查给定静态状态；相邻 link 的局部穿插、未建模物体、衣物形变、实机模型误差和两状态之间的运动都不在这个结果的保证范围。没有轨迹采样、停止包络或执行授权。界面显示的盒子不额外膨胀，误差余量在距离判定中体现。

## 离线测量记录核验

`validate-evidence` 读取两台控制器已有的、同一静止状态下的关节/TCP记录，以及有明确端点的独立实测距离。输入结构如下，尖括号均为待填写内容，不能直接作为 JSON 运行：

```text
{
  "units": "m_rad",
  "capture_id": "<独立采集编号>",
  "captured_at": "<带时区的采集时间>",
  "arms": {
    "left": {
      "serial": "AC130308A40050",
      "q_rad": <6 个真实关节角>,
      "tcp_base_m_rad": <控制器报告的本臂基座系 xyz/rpy，6 个值>
    },
    "right": {
      "serial": "AC130412A4001B",
      "q_rad": <7 个真实关节角>,
      "tcp_base_m_rad": <控制器报告的本臂基座系 xyz/rpy，6 个值>
    }
  },
  "distances": [{
    "label": "<实际测量的两个可复现端点>",
    "a": {"arm": "left", "frame": "tcp", "point_m": [0, 0, 0]},
    "b": {"arm": "right", "frame": "tcp", "point_m": [0, 0, 0]},
    "measured_m": <实测距离，不能用 FK 结果代填>,
    "tolerance_m": <大于 0 且不超过 0.01 的测量容差>
  }]
}
```

端点的 `frame` 可以是 `tcp` 或对应臂中的 URDF link，`point_m` 是该坐标系内的局部点；应选能实际测量的位置。输入 TCP 必须使用与 scene 一致的控制器 TCP offset，不能把基座坐标误当成 World。此工具信任记录中声明的采集身份与时间，**不验证同时性、真实性或状态新鲜度**；仅用于静止记录的离线数值比较，不是执行前状态检查。

```bash
python -m cloth_agent.dual_arm.collision validate-evidence --scene /path/to/measured_collision_scene.json --evidence /path/to/paired_evidence.json --output /tmp/dual_evidence_report.json
```

至少一个距离测量及两臂 FK 都通过才有 `numerical_consistency=true`，但 `physical_acceptance` 始终为 false。实际验收需多姿态独立样本和附件尺寸核验。

## 自动化验证与剩余验收

```bash
python -m pytest -q tests/test_dual_arm_collision_scene.py
```

26 项测试通过：实际 13DoF/mesh 审计；相机光学/外壳变换和腕部跟随；安全、接近、重叠相机对；夹爪互撞、跨臂连杆互撞、相机对另一臂和桌面；实体完全包含；固定折叠姿态的真实连杆自碰撞；间隙和误差余量；漏配、单位、关节限位和未覆盖 link 拒绝；历史与独立端点记录检查；Viser 数据更新。

连同现有双臂 Home、路径分离、mesh narrowphase、停止扫掠、跟踪误差、工作空间、包络编辑、runtime 与安全模式回归，本次共 178 项通过。新增代码的 Ruff 检查通过。真实 SDK 与手动硬件测试未运行。

真实运动前仍需解决（不阻塞离线规划开发）：

- 用独立测量验证最新基座修正。
- 获取新的双臂静止关节/TCP记录，解释官方 xArm7 FK 偏差；若采用拟合模型，必须另用多姿态样本验收。
- 确认相机安装 link、外壳中心及姿态、支架/线缆、完整夹爪包络与桌面几何；设置有测量依据的误差余量。
- 在 Viser 中用上述实测状态对照真实几何，完成人工检查。本模块未下发任何真实运动。

阶段 4～5 已新增独立离线实现，见 [运动规划说明](dual_arm_motion.md)。优化权重仍可根据衣物抓取效果调整。

当前 xArm7 默认采用 `assets/robots/xarm7/xarm7_controller_fit.urdf`，与现有 Home 使用的控制器拟合运动学一致；碰撞网格仍来自原模型。
