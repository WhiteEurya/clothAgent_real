# 手动专项测试

这里集中保存按需运行的相机、机械臂和远程接口专项测试，原先位于 `scripts/` 顶层。`legacy/` 保存原项目根目录的旧版远程测试；完整目录说明见 [测试目录索引](../README.md)。

从项目根目录运行 `python tests/manual/<脚本名>.py`，参数保持不变。可先加 `--help` 查看参数。

| 脚本 | 用途 |
| --- | --- |
| `test_camera_exposure_sweep.py` | RealSense RGB 曝光扫描 |
| `test_camera_white_balance_sweep.py` | RealSense 白平衡扫描 |
| `test_heightmap_exposure_sweep.py` | 不同曝光下的高度图对比 |
| `test_height_map_pipeline.py` | 高度图完整管线采集与离线回放 |
| `test_wrist_camera_board.py` | 腕部相机 AprilTag 标定板验证 |
| `test_claude_single_sleeve_grasp.py` | Claude 单袖口抓取与竖直抬升验证 |
| `dual_arm_micro_test.py` | 双臂 3 mm 微动与回位验证 |
| `replay_gripper_test.py` | 已记录夹爪轨迹的模拟或真实回放 |
| `xarm_shake_open_test.py` | 独立抖动展开动作验证 |
| `remote_planner_test.py` | 远程规划接口冒烟测试，支持 `--mock` |
| `remote_image_tools_test.py` | 图像工具接口验证，支持 `--offline` |
| `remote_fold_smoke.py` | 已保存感知数据上的远程折叠规划验证 |

这些脚本按原有参数连接相机、机器人或模型服务，不由默认的 `python -m pytest` 自动执行。需要真实硬件的脚本继续使用各自的执行确认参数。

`legacy/remote_claude_test.py` 验证 SSH Claude 文本调用；`legacy/remote_image_test.py` 验证 SCP 图片上传；`legacy/remote_planner_test.py` 验证旧版 HTTPS 图片中转与远程规划。这些旧脚本会直接调用网络服务。
