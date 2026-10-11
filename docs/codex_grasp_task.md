# 抓点与持布动作接口

每次启动会依据本次加载的 URDF、基座和 TCP 配置生成 `robot_capabilities.md/json`，初次生成和 IK 代码修正都会收到同一份能力说明。当前本机示例见 [自动生成的能力说明](robot_capabilities.generated.md)。这是供 GPT 选点的参考，不增加执行限制，也不替代既有验证。可运行 `scripts/robot_capabilities.py --output docs/robot_capabilities.generated.md` 更新文档；该工具还支持单个 World 目标的离线 IK 查询。

双臂 `scripts/codex_astra_loop.py` 和单臂 `scripts/codex_arm7_pipeline.py` 的新提示词都使用以下分工：GPT 声明抓点及夹爪姿态，只生成抓住衣物之后、到最终放置位置之间的动作列表；宿主规划抓取前和放置后的移动及夹爪开合。

格式示例（参数为格式占位，不是可执行坐标）：

```python
motion.grasp(left=[x,y,z,roll,pitch,yaw], right=[x,y,z,roll,pitch,yaw])
motion.move_tcp(left=[...], right=[...])
motion.move_tcp(left=[...], right=[...])
```

单臂只包含 right。`grasp` 必须为第一条且只有一次，其后可以有 `move_tcp` 和 `move_transfer`，每个参与抓取的机械臂都必须至少有一个持布/放置目标；最后的位置表示可以松开的位置。GPT 不再输出开闭夹爪、接近点、空手撤离或 Home 指令。完整任务已经完成时仍单独输出 `motion.complete()`，缺少依据则单独输出 `motion.blocked(reason=...)`。

单位沿用 World 的米/弧度。宿主只将 `grasp` 声明中的 Z 降低 10 mm，保存在 `effective_motion.py`；中间动作位置不施加这个偏移。原始代码仍保存在 `motion.py`，旧格式保留供已有工具兼容。

宿主在任何下发前规划完整程序：

1. 验证真实抓点 IK，生成开爪事件。
2. 每臂尝试抓点上方 60、40、20、10 mm 的接近位置；任一高度 IK/路径验证失败就尝试下一个。竖直候选都失败时，再尝试向该臂基座水平收回 20、40、60 mm，同时抬高 20 或 10 mm 的斜向接近点。到接近位置走现有 OMPL 关节路径规划，再验证从候选点到真实抓点的 TCP 插值段，不使用任意绕行替换贴近衣物的下降段。
3. 双臂均到抓点后生成闭合事件。若一种双臂先后顺序失败，从原始配置重新尝试另一种顺序；未验证前不会实际移动或闭合。
4. `move_tcp` 保持笛卡尔插值，失败标注 `gpt_cloth_motion`。新增 `move_transfer` 使用终点 IK 和关节直达／OMPL 绕行，允许中间姿态变化，失败标注 `gpt_cloth_transfer`、终点 IK 标注 `transfer_goal_ik`；终点位置和姿态不被宿主自动修改。两臂仍顺序执行。
5. 两臂到最终位置后生成松开事件；尝试 60、40、20、10 mm 的向上撤离，若竖直候选失败则尝试相同的斜向收回候选，再通过关节路径规划回 Home。双臂先后顺序也可更换。
6. 合并允许连续衔接的段，保持夹爪事件边界，经完整验证后显示 Viser 并按当前执行模式下发。

候选接近点的数量有限；全部尝试失败仍会拒绝，不声称穷尽了所有可能路径。现有碰撞包络、相机碰撞、停止范围和 IK 验收标准没有削弱。IK 失败反馈中保留原 GPT 指令序号以及 `task_phase`、宿主生成目标、尝试高度/顺序，用于区分宿主接近失败和模型中间动作失败。代码修正请求不重新选择任务策略。

测试：`tests/test_codex_grasp_task.py` 覆盖任务边界、只偏移抓点、接近高度与顺序回退、闭合顺序、自动释放回 Home，以及中间/撤离失败时前序动作零下发。保存抓点的真实 URDF IK 对照在 `results/pyroki_ik_migration/host_approach_ik.json`；它仅测试 IK，不代表整条路径已经通过实机验证。

## 关键路径与自由转移

GPT 仍生成完整的持布动作代码，可混用两类段：

```python
motion.grasp(right=[x, y, z, roll, pitch, yaw])
motion.move_transfer(right=[inward_x, inward_y, lifted_z, tilted_roll, tilted_pitch, yaw])
motion.move_tcp(right=[fold_x, fold_y, fold_z, roll, pitch, yaw])
```

以上仅为格式占位。`move_transfer` 用于允许自由转移的动作，不保证衣物形状、张力或两爪相对距离；需要这些约束时不能使用它代替关键折叠段。工作区边缘的转移终点由 GPT 根据能力说明选择，可同时向内收回和抬升、改变终点倾角。终点本身不可达时仍返回 IK 失败，不把 RRT 当作不可达终点的补救。宿主接近／撤离候选保持原端点姿态，仅改变辅助点位置。

所有新路线继续通过原有停止范围、碰撞及下发段验证；预览识别两类目标。没有更改真实控制器执行方式。
