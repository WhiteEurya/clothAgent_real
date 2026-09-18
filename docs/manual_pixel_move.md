# 手动点选验证相机落点

在连接机器人和腕部相机的机器上，从仓库目录运行（需要桌面显示和 Python tkinter）：

```bash
python scripts/manual_pixel_move.py --real --confirm-real
```

1. 脚本经 Home 回到配置中的 perception 观察位置，拍摄新的对齐 RGB-D。
2. 窗口显示原始 RGB（没有 upright 旋转）。点击目标，显示原始像素、深度、测量 XYZ 和目标 TCP XYZ；可以重新选点。
3. 按 Enter 或点击 Move。完成实时 TCP offset 和控制器 IK 检查后，经 Home 在高处移到目标上方，再垂直下降，最后停住供观察。窗口关闭，终端显示运动结果。

默认停在测量表面上方 **30 mm**，高度沿机器人基座 Z 轴。需要到达测量点本身时运行：

```bash
python scripts/manual_pixel_move.py --real --confirm-real --clearance-mm 0
```

这是 TCP 的目标坐标；夹指实际接触点是否重合仍取决于 TCP 和标定精度。先用默认悬停高度检查偏差，再决定是否降低。没有自动闭爪、抓取高度补偿或在线桌面 Z 偏置修正，也不调用 Claude/Molmo。

选点阶段 Esc 或关闭窗口会取消后续运动，机器人留在观察位。每次启动只执行一个点，再次测试请重新运行，以免腕部运动后使用旧图。运行期间不要用其他程序控制机器人或移动被测物体。IK/工作空间检查不等于障碍物碰撞检测；观察位置往返沿现有配置的 Home 路线。运动开始后 GUI 已关闭，硬件急停仍使用机器人的急停按钮，退出程序不保证中断控制器中的运动。

可选参数：`--camera A`、`--robot-config PATH`、`--perception-config PATH`、`--output-dir PATH`。默认输出到 run storage 的 `manual_pixel_move` 辅助目录，每次创建新的时间戳子目录，保存：

- `rgb.png`、`depth_m.npy`、`camera_A_base_xyz_mm.npy`：本次拍照及逐像素 XYZ。
- `result.json`：内参、实际使用的外参、机器人配置、观察位反馈。
- `selected_pixel.png`：选中像素。
- `execution.json`：点击坐标、目标、IK/TCP 校验、真实运动反馈和错误。

拍摄使用现有静止时序聚合逻辑；记录观察位反馈，未逐帧同步关节与深度。保存结果也可交给离线校验脚本：

```bash
python scripts/check_camera_calibration.py --perception /path/to/result.json --pixel 359 551
```
