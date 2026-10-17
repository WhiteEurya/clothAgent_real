# Camera B registration_field 操作说明

第一版只采 L-M、C-M、R-M 三点，每点 10 个有效 repeat。只连接 Camera B；
不连接 arm7/arm6，不运动机械臂，不写入 intrinsics、extrinsics 或 RealSense calibration。

## 启动

在可打开桌面窗口的终端运行这一条：

    /home/sja/miniconda3/envs/cali/bin/python -m cloth_agent.experiments.registration_field run

窗口需要桌面 DISPLAY；使用 Tk + PIL，不用 OpenCV/Qt GUI。
没有从零启动机器人、回 Home 或回 perception 的动作。相机/机械臂应由操作者事先保持固定。
其他占用相机的程序需先结束；本程序不会停止其他程序。

## 摆板与按键

1. 只留下 **一张** 4×4 tag36h11、ID 0–15 的板，板平整，移走尺子等遮挡。
2. 开始前先把板在桌面内旋转约 **30°**，让图案边不平行于画面水平线。板仍平放，**不是抬起倾斜30°**。之后整个 session 保持相机固定、板平放在**同一桌面**，三处只平移，不再旋转板。
3. 当前绿色框表示“整组 tag 图案的**中心**应该落入的区域”，不是板的边界。黄色圆圈是检测到的板中心。
4. 先摆到 **L-M**：原始 RGB 画面的左侧、中排。整张板需完整可见；把手移开，然后在窗口按 **Enter**，也可点击开始按钮。
5. 自动采 10 个有效 repeat。期间不要移动板；投点会切换，这是正常的 RGB/IR 测量步骤。
6. 窗口提示 **C-M** 后，再沿桌面把板移到中央；按 Enter。随后 R-M 同理。
7. 完成后自动生成报告；点“打开结果目录”查看。要停止，按 **Esc** 或点停止，等待设置恢复完成再关窗口。

L/C/R 指 **RGB 图像**方向，不是机器人左右手。
M 指画面中排，并不保证三个点光轴 Z 完全相等；每个角点实际 native depth Z 都会保存。
中心参考位置是图像归一化坐标 L-M=(0.30,0.55)、C-M=(0.50,0.55)、R-M=(0.72,0.55)；
中心允许框宽为图宽的 13%，高为图高的 20%。完整 tag 图案可以超出绿色中心框，但不能超出相机画面。

## 自动采集与重采

每个 repeat：
- emitter-on：稳定 30 帧，再采 15 帧 native depth；
- emitter-off：稳定 30 帧，再采 RGB/IR1/IR2；
- finally 恢复 session 原 emitter 值并读回核验，重新拍 RGB；
- 比较 on 起始/结束、off 起始/结束、恢复后的 RGB 角点；
- 任意运动 RMS > **1 px** 时拒绝并重采，不增加有效 repeat 编号。

RMS 使用二维角点位移 sqrt(mean(du²+dv²))；共同 tag 少于 4 个时拒绝，不把缺少检测当成静止。
至少 12 个 RGB tag，至少 24 个两方法共同有效角点才接受一个 repeat。
板离开指定位置、相对首个有效姿态旋转/倾斜超过 10°，也会拒绝。
窗口红字保留上次拒绝原因，方便修正摆放。

默认每处最多 40 次尝试；达到上限、相机断连或设置恢复失败时停止 session，并保留部分结果。
不会跳过设备异常继续试，不会无限重采。
两条方法不根据被测 RGB residual 大小筛选样本。
5×5 native depth 沿用原诊断的至少 18 个有效像素中位数；记录 spread，不引入 grasp 门控。

### 为什么推荐斜放

2026-10-16 的曝光、投射器功率、板旋转往返对照表明：本次板边近水平时角点深度大量缺失；
同样曝光4 ms、gain16、投射器功率150，桌面内旋转约30°后深度明显恢复。
共有13个tag的52个角点：旋转前18/52，旋转后连续三次52/52，转回后21–22/52。
这支持横向/重复纹理导致双目匹配歧义；不是通过修改内外参或填补深度修复。
保留原曝光、功率与全部测量门槛，先斜放再在三处平移。
该修复针对角点深度空洞，不表示此前RGB↔IR残差已经消失。
完整原始证据位于 `reports/camb_exposure_audit/`。

## 输出

默认位置：runs/registration_field/<UTC时间>/

- session.json：状态、每次 attempt 的接受/拒绝原因和计数。
- camera.json：本次读取的 factory profiles、depth scale、内部外参、原 emitter 和采集曝光设置。
- all_samples.csv：所有角点，包括 rejected attempts；两种方法分别有 valid 标记。
- repeat_summary.csv：每个有效 repeat 的统计。
- summary.json / summary.md：每处 median/P95 error、median du/dv、实际 Z 和 repeat 波动范围。
- residual_field.png：统一 RGB 坐标中的残差场，箭头放大 5 倍。
- residual_overlays.png：实际 RGB 上的两种方法残差，各 placement 分开显示。
- median_vectors.png：L/C/R 三个位置中位残差向量对比，单位为真实像素。
- L-M、C-M、R-M/attempt_XXX/：原生深度压缩数组、on/off/恢复后 RGB、IR1/IR2、角点数据、时间戳及 emitter 恢复记录。
- on_first_ir1/ir2.png、on_last_ir1/ir2.png：深度采集阶段同步IR，供核对饱和与空洞；ir1.png/ir2.png仍为emitter-off角点检测图。
- final_emitter_restoration.json：退出时恢复核验。

du/dv = 投影到 RGB 的像素 − RGB 实际角点。正 du 向右，正 dv 向下。
同一角点的 10 次重复不当成 10 个独立空间位置；报告不会自动拟合 compensation 或部署修正。

## 离线重新生成图表

    /home/sja/miniconda3/envs/cali/bin/python -m cloth_agent.experiments.registration_field review SESSION目录

review 不连接相机或机器人；保留原始 CSV，仅重新生成统计和图。

## 九点扩展

当前明确只启用三点。先实际完成 30 个有效 repeat，人工检查摆放、恢复记录、拒绝原因、图表方向及原始证据。
三点实测验证通过后，再扩展 LEFT/CENTER/RIGHT × NEAR/MID/FAR；当前离线测试通过不等于已完成三点实测验证。

## 已完成的验证

- 数学/判定/恢复单元测试与模拟完整三点 session（含一次拒绝重采）。
- 使用先前保存的 RGB/IR/native-depth 记录回放：旧方法的 38 个 direct 角点全部复现，du/dv 最大差约 1.1e-13 px；stereo 中位误差保持 4.291369 px。
- 新日志同时保留只满足单条方法的角点，因此 direct 总有效角点可能多于旧的三流共同 tag 子集；报告给出各方法的样本数。
- 没有自动启动真实 registration_field session。

示意图与离线验证记录：reports/registration_field_implementation/
