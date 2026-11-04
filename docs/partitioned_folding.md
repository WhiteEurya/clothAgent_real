# 折叠 Graph：左右分工，每步一个夹爪

第一版实现位于 `cloth_agent/experiments/partitioned_folding.py`。
沿用冻结 Graph 的 9 个节点、8 条边及状态转移，每次将一条边转换为一个选定机械臂的
单夹爪动作提案。左臂和右臂使用同一套生成逻辑，不增加同步动作或中途交接。
目前只输出预览，机器人接入继续留待下一阶段。

自动视觉提案已接到[俯视图主流程与原图对照](topdown_folding.md)：俯视图负责 Graph 识别、
动作选择、中心线与锚点，映射回本模块的 `plan_step()`；原图仅保留独立对照。
下文的手工标注入口继续可用。

## 分配规则

1. 使用当前观测的中心线、抓取点和落点；通过深度转换到右臂底座公共坐标系。
2. 将中心线投影到已记录的桌面平面。左右区域按两台实际底座所在侧定向，
   不使用图像中点，也不按 Graph 中“左袖”这样的文字直接选臂。
3. 抓点落在左臂一侧，整步交给左臂；右侧同理。即使落点跨线，仍由原臂完成。
4. 距中心线不超过 15 mm 的区域要求显式选择 `center_arm`；未选择则返回
   `CENTER_ARM_REQUIRED`，动作列表为空。不会凭距离假定某一臂可达。
5. 所有夹爪和 TCP 操作必须属于同一臂。未选中的臂本步不接收操作；它的实际姿态和
   占用空间仍需在之后的机器人预检中检查。

两端标注顺序不影响归属。衣服若旋转到中心线无法将两个底座分到两侧，会拒绝预览，
要求核对衣物朝向/中心线；不会静默交换物理左右臂。
中心线与锚点绑定观测 ID 和 RGB 哈希，换图后必须重新标注或重新估计。

Graph 的衣物左右按“领口朝上时的观察者左右”定义；机器人的左右是物理设备身份。
例如当前照片领口朝下，Graph 的左袖可能在照片右侧，此时应由物理右臂承担该动作。

## 运行

```bash
conda activate cloth-real
cd ~/agentAuto/exp/clothAgent_real

# 验证冻结 Graph、当前标定和分工策略，不连接相机/机器人/模型
python -m cloth_agent.experiments.partitioned_folding

# 在已保存观测上打开标注窗口，生成离线预览
python -m cloth_agent.experiments.partitioned_folding \
  --observation deployment-artifacts/zed_site_live_observation \
  --annotate --output outputs/partition_preview_001
```

窗口中选择当前适用的边，依次点选中心线两端、抓取点、落点，核对源节点及动作语义后生成。
上述照片是之前的历史采图；需要当前场景时先用 `python -m cloth_agent.site_observation --live-camera`
采集，再将其输出目录作为 `--observation`。拍照不会连接机器人。

也可用 JSON 输入，便于之后连接现有 Graph 状态识别和锚点定位模块：

```json
{
  "observation_id": "从 observation.json 读取",
  "rgb_sha256": "从 observation.json 的 sha256.rgb.png 读取",
  "current_node": "n_initial",
  "edge_id": "e_initial_left_sleeve",
  "centerline_px": [[620, 300], [625, 640]],
  "grasp_px": [965, 525],
  "target_px": [790, 460],
  "confidence": 1.0,
  "anchor_source": "manual",
  "center_arm": null
}
```

上面坐标仅是本次历史照片的示例，不应复制到新图片。

```bash
python -m cloth_agent.experiments.partitioned_folding \
  --observation PATH_TO_OBSERVATION --request request.json \
  --output outputs/partition_preview_002
```

输出 `request.json`、`plan.json` 和 `preview.png`。输出目录必须是新目录。
`plan.json` 记录选臂理由、距中心线距离、是否跨线、另一臂身份、Graph 原始示范归属、
当前位姿提案及验证范围。每个 TCP 目标同时给出公共系与所选臂基座系的毫米/度位姿，
并保留对应七轴设备身份。它不是现有运动执行器可以直接加载的计划文件。
中央未指定臂时也保存可审查预览，退出码为 2。

## 动作模板

同一套单夹爪序列用于两臂：

```text
张开 → 接近 → 到抓取点 → 闭合 → 抬升 → 搬运 → 放下 → 张开 → 退离
```

选定臂的姿态来自当前现场 `grasp_rpy_deg`；完整旋转矩阵随坐标系转换。
高度沿记录的桌面法向调整，参数见 `config/folding.partitioned.json`。
当前仅用于预览：接近 100 mm、搬运净空 80 mm、抓取表面偏移 0 mm、放置偏移 5 mm。
没有沿用旧六轴示范的负抓取偏移、旧工具姿态和部分释放脉冲；新参数也尚未经过抓布实测。
这保留了示范的抓点/落点关系和抓放阶段，不声称精确复现历史机器人轨迹。

## Graph 的连接点与尚未完成的部分

`plan_step()` 接收已编译 Graph、当前节点、锚点、中心线和现场标定，返回本步提案。
它核对所选边确实从当前节点出发，不自动将预期目标节点当成已到达节点。
接入真实循环时应当是：

```text
新观测 → 现有节点识别 → 选择出边 → 当前图锚点/中心线
→ plan_step 选臂 → 该臂 IK/路径及双臂占用空间检查
→ 单臂执行完成并退离 → 新观测、重新识别 → 下一步
```

自动中心线识别和模型调用见 `topdown_folding.py`；机器人执行循环尚未接入，也没有假定另一臂已在 Home。
到接近点的路径、退离后的停车位置、完整任务范围和碰撞检查都留在机器人接入阶段。
当前 ZED/xArm7 执行配置只保留每条基座坐标系的最小 Z 作为工作区约束；X/Y 和最大 Z
不再套用旧标定中的局部矩形。`within_imported_local_target_bounds` 现在只表示抓点和落点
高于该最小 Z，真正的可达性由选定臂的 IK、关节限制、路径证明和另一条机械臂的静态
碰撞模型共同决定。

旧 Graph 和旧单臂执行器未修改。这个入口没有 `--execute`，所有结果均标记
`execution_allowed=false`，不输出可直接调用 SDK 的代码。

## 本次验证与示例

新增分工测试覆盖左右归属、中心区域、跨线搬运、翻转中心线端点、衣服旋转、
过期锚点、无效深度、Graph 源节点匹配及完整姿态转换。
与原 Graph、ZED 接入测试合计 53 项通过。

当前红色衣服照片的人工示例保存在
`deployment-artifacts/partitioned_folding_example/preview/`。
示例抓点位于物理右臂一侧，结果为 `selected_arm=right`，9 个操作都属于右臂。
它仅演示分工和坐标变换，没有独立验证衣物节点识别、锚点对应关系、可达性或折叠效果。
