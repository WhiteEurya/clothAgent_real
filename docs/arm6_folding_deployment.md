# Arm6：只运行折叠 Graph

使用已经随仓库发布的 `data/graphs/arm6_replay_goal_n007/graph.json`。
这是 2026-10-18 的 Arm6 / Camera A 折叠示范图，协议为
`arm6-traj-experience-graph-v1`。共 **9 个节点、8 条边**；从展开状态开始的
主路径有 **7 个状态、6 个动作**，目标是 `n007`。

```text
n_initial → n000 → n001 → n004 → n005 → n006 → n007
  左袖内收   右袖内收   左侧内折   右下内收   右中段内收   下摆上折
```

另外保留 `n002 → n003 → n004` 两条左侧重排/收拢示范边作为原图分支。
图中没有新增展开探索、机械随机 reset 或在线扩图流程。普通 replay 根据当前观测
匹配已有状态，不保证总从 `n_initial` 开始；测试完整流程时由操作者先将衣服展开。
源图、证据标签和历史用户修订均保持原样。

## 时间线与选择依据

| 时间 / 原版本目录 | 节点 / 边 | 为什么选或不选 |
|---|---:|---|
| 10-17 `arm6_traj_graph_20261017` | 8 / 6 | 初始示范整理，缺少首个左袖步骤和中间连接 |
| 10-18 `arm6_traj_graph_20261018_without_last` | 7 / 5 | 删去了最后的下摆动作和终点，数量最少但不完整 |
| 10-18 `arm6_traj_graph_20261018_with_first` | 8 / 6 | 补回左袖动作，末步和图中断点仍未补全 |
| 10-18 `arm6_traj_graph_20261018_n001_to_n004` | 8 / 7 | 接通两袖之后的左侧内折，仍没有最终下摆动作 |
| **10-18 `arm6_traj_graph_20261018_goal_n007`** | **9 / 8** | **补回末步，指定 n007 目标；仓库中的便携包对应这一版** |
| 10-18 `arm6_graph_extended_20261018` | 18 / 21 | 后来的手动证据扩展图，本入口不使用 |
| 10-19 `arm6_archive_graph_20261019` | 81 / 84 | 后来的大型归档图，本入口不使用 |

节点数用于定位版本；实际选择依据还包括动作内容、连通主路径和目标节点。

## 在另一台机器上运行

先按 [安装与现场配置](deployment.md) 安装 `requirements-experiments.txt`，配置
`config/dual_arm.local.json`、相机 A、Arm6、TCP/外参、Home/观察位与现场工作空间。
这个入口进入 Arm6 replay 分支，只连接 Arm6 / Camera A；双臂配置格式和原有静态
几何/记录仍被底层代码使用，不能简单删掉配置中的另一臂字段。
不需要旧 Graph Release 数据包，也不需要本机的大型 `reports/` 或 `runs/`。
便携包的 119 个文件（约 133 MiB）包含 RGB、XYZ、示范动作和原始控制器证据。

从仓库根目录，在安装依赖的 Python 环境中执行离线检查：

```bash
bash scripts/run_arm6_folding_graph.sh --dry-run
```

不带参数也只做离线检查。预期 `DRY_RUN_OK`，8 条 compiled edges，
`robot_calls=0`、`model_calls=0`。这不会连接相机或机械臂，也不会执行动作。
本次在独立代码副本中验证了该结果。

完成新现场配置和核对后，以下命令会实际运行 Arm6：

```bash
bash scripts/run_arm6_folding_graph.sh --execute
```

输出为新的 `reports/arm6_folding_<时间>/`，只读图页面端口为 8090。
可通过 `--config /path/to/config.json`、`--output /path/to/new_trial` 覆盖默认值。
脚本默认不启用 expanded graph、恢复规划、online growth 或 retrieval matcher。

## 多件衣服的折叠泛化测试

固定这份 Graph，分别用新的衣服和相同初态协议测试，每次使用独立结果目录，例如：

```bash
bash scripts/run_arm6_folding_graph.sh --execute \
  --output reports/folding_generalization/shirt_001/trial_001
```

该图的参考来自紫色短袖上衣，节点文字也含紫色、印花和袖口形状描述。
换颜色、图案或尺寸正是需要验证的泛化条件，不能事先声称已支持；裤子、裙子等不同
拓扑不等价于这份 T 恤折叠图。与 RGB-only 展开实验不同，本入口依靠图的状态匹配和
示范动作定位，不需要另传展开目标图或 GGO 七点标注。

分别记录初始匹配、执行边、动作后匹配、停止原因与人工判定的折叠质量。
`GOAL_REACHED` 表示匹配到历史目标 n007，不等于边缘整齐或达到通用折叠标准。
图中 e001 保留原拒绝示范及后续用户指定修订的证据，不应把所有边视为独立成功样本。

沿用既有 CollectionRuntime 的运行策略：路径/几何碰撞验证关闭，限位、IK、反馈和
控制器故障检查保留；本次只选择并验证折叠图，没有改变该执行策略或做实机试验。
