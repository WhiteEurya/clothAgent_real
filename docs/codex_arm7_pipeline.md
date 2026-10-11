# Arm7 单臂折衣 pipeline

当前动作接口已更新为 **GPT 声明抓点＋持布动作列表，宿主规划前后接近、抓取、释放与回 Home**，详见 [抓点与持布动作接口](codex_grasp_task.md)。下文旧 `move_tcp`/`gripper` 流程仍兼容历史代码；新模型输出遵守上述分工。

在项目根目录启动：

```bash
/home/sja/miniconda3/envs/cali/bin/python scripts/codex_arm7_pipeline.py
```

沿用本机 Codex（gpt-6-astra/high）、五张顺序人类示范图、当前 RGB-D 点云和动作代码接口；每轮只有一次模型规划，不调用额外反思，不注入历史经验。

IK 失败且未执行时，下一次规划尝试复用同一 obs，将具体失败步骤和完整报告交回 GPT，只修正失败代码，保留既定抓点和折叠顺序，不重新看图规划任务；不重复 Home/拍照。与双臂共用这一失败反馈逻辑，执行故障不自动重试。

流程：仅打开 arm7 夹爪、arm7 回 Home、arm7 到自己的已保存 perception（整段先验证、显示 Viser 后自动执行）→ 到位后采集 obs → GPT 输出仅含 right 的动作 → IK/RRTConnect 与严格碰撞验证 → Viser 预览后自动执行 → 下一轮。模型从当前 obs 判断完成时输出 `motion.complete()` 结束。

预览默认 `http://127.0.0.1:8088`，可用 `--viser-port` 修改。其他参数和双臂入口一致，例如 `--max-rounds 1`、`--task`、`--reference`、`--output`。默认记录目录为 `/mnt/newssd/sja/clothAgent_real/runs/codex_astra/loop_arm7_时间戳/`。

拍照准备阶段与 GPT 生成的折衣代码都只允许 arm7/right。模型动作解析阶段拒绝任何 left 命令；下发前再次验证折衣操作属于 right，且 arm6 关节保持不变。与双臂入口共用硬件进程锁。

单臂入口使用 arm7 的 `data/robot/xarm7_perception_pose.json`，Camera B 序列号 `233622079809`，外参取自 `config/calibration/dual_arm_working_20261008/camB_extrinsics.yaml`（平移从米换成毫米，挂载于 right/link_eef）。内参由实际 RealSense 流读取。配置只在单臂进程中派生，不修改双臂配置文件；不继承 arm6 的 1° 倾角修正。

Camera B 深度曝光固定为 **4000 µs（4 ms）**、增益 **16**，折叠和展开入口共用；RGB 曝光不随之修改。该设置来自布料 IR 过曝排查和实机验证，避免重新使用自动深度曝光时产生已观察到的深度变化。观测文件记录曝光策略。当前相机外参按实测 **32.5 mm** 黑色标签边长重新拟合，左右两处标签中心的目测对位及约 50 mm 指尖间隙已通过验证；不叠加排查期间的 16 mm／9 mm 手动 XY 偏移。单次褶皱抓取成功不等于所有平铺布边都已验证，抓取下降策略仍按下文执行。

arm6 在准备、采集和折衣期间均保持原位，不使能其运动、不操作其夹爪。arm7 专用入口完全不连接 arm6；碰撞检查使用 `data/robot/arm6_parked_for_arm7.json` 保存的固定关节位置。每次采集前（包括首次启动），若 arm7 已在保存的 perception 位置允许的关节误差内，则复核到位后直接采集，不回 Home、不操作夹爪；未到位时仍按上述准备流程执行。发生故障时停止已连接的 arm7，arm6 固定障碍适配器不发送停止命令。

原双臂启动入口不变。离线测试及实机测试结果分别记录在测试套件和 `results/arm7_run_check/`。

抓取高度：arm7 闭合前最后一条显式 `move_tcp` 的 World Z 由宿主额外减去 15 mm。接近、抬升、折叠运输、松开目标保持原值；重复闭合不重复修改同一目标。提示词要求 GPT 输出原始观测高度，避免模型与宿主重复偏移。实际动作保存在 `effective_motion.py`，偏移记录在 `grasp_height_policy.json`；IK、碰撞验证和 Viser 均使用修改后的目标，若降低后碰撞仍拒绝执行。

## 自动执行与计时

arm7 单臂入口现在默认在严格规划验证通过、生成 Viser 预览、确认起始关节状态后自动执行，不再等待按钮。失败或 UNKNOWN 仍拒绝下发。需要人工预览暂停时加 `--manual-execute`；`--auto-execute` 可显式启用自动模式。双臂入口现也默认自动执行，可用 `--manual-execute` 恢复等待点击。

每轮打印耗时并写入 `round_NNN/timing.json`：`observation_s` 包含 Home/perception 准备及 RGB-D 采集；`gpt_planning_s` 为模型调用；`trajectory_planning_s` 为 IK/RRT/严格验证；`preview_s` 为预览生成；`preview_wait_s` 为人工等待（自动模式接近零）；`pre_dispatch_check_s` 为执行前状态复核；`execution_s` 为动作执行；`round_total_s` 为全轮总时间。`motion_pipeline_s` 是运动阶段总计，包含上述规划/预览/执行子项，不应重复相加。会话总时间存于 `result.json.elapsed_s`。失败和中断也记录已花费时间；未开始的阶段不伪造为零耗时。

GPT 附加指令：尽可能和human demonstration相似。此要求与 arm6 同步。

## 腕部旋转与相机线缆

观察方向按“从机械臂朝夹爪看”：当前模型 joint7 的负向旋转为逆时针。arm7 抓取接近阶段会比较 joint7 相差 360° 的等价解，在关节限位内先选择较近解；若较近解仍需顺时针转动超过 90°，则优先采用逆时针等价解。例：从 0.3° 到 185.9° 改为到 −174.1°，TCP 位置和姿态均保持不变。仅更改关节角的圈数表示，不自动将夹爪方向翻转 180°。

下降与持布的连续 IK 选择最接近前一关节角的等价解，避免 ±180° 表示切换造成额外绕圈。回 Home/perception 保留原关节目标，允许反向退回。所有新路径仍须经过现有碰撞验证；此规则是抓取目标转向偏好，不是线缆几何模型，也不保证避碰绕行中每个小段都逆时针。若关节限位不允许逆时针等价解，则保留限位内较近解。

## 队列下发

arm7 折衣阶段使用与 arm6 相同的队列下发：相邻同臂移动最多 32 段一批，中间指令 `wait=False`，队尾等待完成，减少逐段 SDK 等待和到位复核产生的软件间隔。夹爪动作以及准备阶段的 Home/perception 边界保留。下发前逐段检查规划中的 arm6 固定关节不变；执行中监控 arm7 实时反馈，arm6 不读取实时反馈。

原路点、速度、加速度和 `radius=-1` 保持不变；队列减少软件等待，但控制器仍可能在路点处减速或停止，并非连续速度插补。arm7 专用入口关闭 right 的额外碰撞包络（停止距离、跟踪角误差、静止角容差及位置误差膨胀），与 arm6 专用入口的策略对应；仍保留名义几何和完整指令运动范围的碰撞检查、执行反馈容差及停止处理。此设置仅作用于当前运行时，不修改共享标定文件或双臂入口。`native_plan.json` 中的 `queued_segments` 记录原始路点。


## 展开复原与跨轮记忆

新增入口将揉乱的衣服恢复到 human demonstration 的最初展开状态：

```bash
/home/sja/miniconda3/envs/cali/bin/python scripts/codex_arm7_unfold_pipeline.py
```

默认目标图为 `data/reference/fold_states/active/state_00_unfolded.png`，只使用这一张目标图。可通过 `--reference /absolute/path/initial.png` 替换；此模式只接受一张参考图。默认 arm7 单臂自动执行，可加 `--manual-execute` 或 `--max-rounds N`。结果目录前缀为 `/mnt/newssd/sja/clothAgent_real/runs/codex_astra/unfold_arm7_`。展开复原模式固定额外下降 15 mm，闭合完成后继续原定搬运动作，不根据夹爪机械信号判空或自动加深。普通折叠模式仍为 15 mm。腕部转向偏好、队列和名义碰撞检查继续生效。

该入口默认开启 `--remember-iterations`：GPT 每轮返回包含 `code` 和 `memory` 的 JSON。宿主先验证记忆结构及动作语法，再执行动作；GPT 仍通过只读工具分析，不直接写文件或连接机器人。记忆包括当前状态、上一动作效果、不确定性、经验和下一步。当前动作尚未执行，效果只能写为待验证；下一轮结合新观测及上一轮动作前图进行核对。IK 修正轮复用观测，不虚构一次新动作或新观测。

每个 run 独立保存记忆，不自动继承其他衣物或旧 run：

- `iteration_memory.json`：最新模型记忆、动作代码、宿主执行结果及观测目录。
- `iteration_history.jsonl`：每轮完整记忆记录。
- `round_NNN/memory_input.json`：当轮收到的历史记忆。
- `round_NNN/model_memory.json`：当轮 GPT 写出的记忆；即使随后的执行中断，也保留该份规划期记录。
- `round_NNN/planner/answer.txt` 保留原始 JSON，`motion.py` 只保存提取并验证的动作代码。

普通折叠入口的默认行为不变；显式传 `--remember-iterations` 也可启用记忆。旧的 `--memory` 参数仍保持兼容空操作，不用于跨 run 恢复。展开流程在当前图像显示两袖、衣身、下摆展开且接近参考形状时，由模型返回 `motion.complete()`；执行成功状态本身不代表复原完成。


### unfold 抓取反馈

arm7 固定额外下降 15 mm。夹爪闭合完成后继续搬运，到原定释放点才开爪；不根据接近完全闭合的位置或机械抓取信号判定抓空、中止搬运、复用旧抓点或累计下降。夹爪原始反馈仍写入日志，动作效果由后续图像判断。执行错误和原有规划检查仍会阻止执行。

默认输出遵循 `config/run_storage.json` 的 `runs_root`，所有轮次图像、深度、视频和 planner 日志都保存在该输出目录下。配置要求的数据盘未挂载时会报错，不回退到系统盘。显式 `--output` 仍覆盖默认位置，请勿指定到已满的系统盘。修改仅影响新启动的 run，已有输出不自动迁移。
