# 双臂末端小范围测试

脚本：`scripts/dual_arm_micro_test.py`。默认只读取 home 文件并输出计划，不连接机器人。
实机使用带 xArm SDK 的 Python，例如 `/home/sja/miniconda3/envs/robo/bin/python`。

## 设置对称 home

两台分别记录自己的 TCP 位姿、关节角、TCP offset、轴数和控制箱序列号。
当前两台控制箱报告 6 轴和 7 轴，不可以用同一组关节角定义对称。
机器人报告的机械臂序列号曾出现重复，因此这里用控制箱序列号识别硬件。

先在 UFactory Studio 中分别把两台摆到所需的对称 home：以桌面中线为对称面，
两末端到中线距离相同、高度相同，夹爪朝向按实际任务对称；根据实际安装方向判断。
这是人工示教，不是脚本自动计算的标定结果。未知两底座坐标变换时，不能简单给某个坐标取负。
记录前两台应静止，夹爪空载，短程上移和回程空间均无障碍。脚本不会开合夹爪。

```bash
/home/sja/miniconda3/envs/robo/bin/python scripts/dual_arm_micro_test.py --capture-home
```

默认 IP 按顺序为 `192.168.2.232`、`192.168.1.195`，可用 `--ips IP_A IP_B` 修改。
默认文件 `data/robot/dual_arm_home.json` 不覆盖；重新示教请指定新的 `--home` 文件。
此操作只读取并保存当前姿态，不会把机械臂移动到任何位置。

## 预览和执行

```bash
python3 scripts/dual_arm_micro_test.py
/home/sja/miniconda3/envs/robo/bin/python scripts/dual_arm_micro_test.py --execute
```

执行前必须已经位于记录的 home（TCP 误差不超过 0.5 mm / 0.5°，关节差不超过 1°）。
脚本不负责从任意位置规划到 home。两台需预先处于位置控制模式、无错误和警告、电机已使能；
程序不会清除故障或使能电机。

动作顺序：两台分别沿自己的 base +Z 方向直线上移 3 mm，速度 2 mm/s、加速度 5 mm/s²；
等待两台都完成并核对 TCP，再同时返回各自记录的 home TCP。这里的 home 指末端位姿，
7 轴机械臂的冗余关节不保证精确恢复同一角度。若底座倾斜或倒装，base +Z 不代表世界向上，
必须先确认实际路径。可用 `--delta-mm`、`--speed-mm-s` 调整，范围均为 (0, 5]。

两条命令通过线程屏障并发提交；打印的是主机发出调用的时间差，不是实测机械同步精度。
这不是硬件同步，不保证同时起动或到达。

两台完成状态、身份、TCP offset 和路径采样 IK/限位检查后才发出任何运动。
执行故障或中断时尝试对两台发送 `set_state(4)`，不再自动回 home。
若网络失联，软件停止可能无法送达。IK 检查不是双臂碰撞检测：本项目尚无两底座共同坐标系
和双臂碰撞模型，实际路径间距与周围障碍仍需在示教时确认。

## 验证

`tests/test_dual_arm_micro_test.py` 使用模拟控制器，验证并发、各自回位、任一台预检失败时
禁止全部动作、运动失败时停止两台且不再回位、以及非法 home 输入。
编写程序时未执行实机运动，也未生成假定的 home 坐标。
