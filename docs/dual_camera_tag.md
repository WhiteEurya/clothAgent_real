# 双夹爪相机同 Tag 对齐诊断

入口 `scripts/compare_dual_camera_tag.py`，只连接两台 RealSense RGB，不导入机器人 SDK、不运动、不改标定。
默认 A=317222073552，B=233622079809，tag36h11。先关闭占用相机的预览程序。
两台相机必须看到**同一个物理 Tag 的同一正面**；不能各放一张同 ID 图案。
测量黑色方框的外边长，不是纸张、白边或图案内侧边长。缩放打印后不能沿用文件名中的尺寸。

```bash
# 仅当实测黑框边长为 48 mm、ID=0 时使用这个例子
/home/sja/miniconda3/envs/cali/bin/python scripts/compare_dual_camera_tag.py \
  --tag-id 0 --tag-size-mm 48
```

启动并列预览。两台机械臂和相机保持不动，Tag 在每次采集时也应静止。
程序自动收集前 10 对合格观测拟合 Camera B -> Camera A，固定变换后收集后续 20 对作独立验证。
默认间隔 0.5 秒，总时限 180 秒；不合格检测记录原因但不计入样本。Q/Esc 退出。
仅原地重复采集衡量重复性；要检验工作区一致性，可分开两次运行，在第二次使用已保存变换，重新摆放同一个 Tag；相机不能动。

```bash
/home/sja/miniconda3/envs/cali/bin/python scripts/compare_dual_camera_tag.py \
  --tag-id 0 --tag-size-mm 48 \
  --transform results/上一轮目录/camera_transform.json
```

坐标定义：T_A_tag 将 Tag 坐标变成 A 光学坐标，T_B_tag 对 B 同理。
每对拟合候选 `T_A_B = T_A_tag @ inverse(T_B_tag)`；旋转平均、平移平均。
验证时固定 T_A_B，比较 T_A_tag 与 T_A_B @ T_B_tag，报告 B转换后减A 的 XYZ 差、三维位置误差和朝向角误差。
**不直接相减两个相机各自坐标系下的 XYZ，不用本帧算出的变换评价本帧。**

输出 `results/dual_tag_时间/`：
- `settings.json`、`intrinsics.json`：相机身份、实测流内参/畸变和参数。
- `observations.jsonl`：Tag 位姿、FIT/VALIDATION 分组、像素角点、重投影误差、解码置信信息、拒绝原因、帧号及时间戳。
- `*_A_raw.png`、`*_B_raw.png`：每对被接受观测的原始 RGB。检测在去畸变图上进行，角点坐标也属于去畸变图。
- `camera_transform.json`：本次拟合的 B->A，平移单位 mm（外部变换模式在 summary 中记录输入来源）。
- `summary.json`：独立验证的位置 RMSE、均值、P95、最大值、XYZ 系统差和朝向误差。没有验证样本时是 UNKNOWN。

局限：没有硬件曝光同步；host_receive_skew 不是曝光时间差，不同设备时间戳不能无条件相减。
平面 Tag 可能出现双解，重投影质量差/明显歧义时拒绝。两个相机共同的尺度错误不会被相互对齐检验发现。
此结果仅针对当前固定相机姿态，不是绝对精度、手眼或机器人底座标定。机械臂移动后，B->A 通常变化，不能继续沿用旧变换。
若要验证 450 mm 底座假设，应另外通过各臂 FK 和手眼标定独立计算 A->B，再用本程序的观测对比；不能将同 Tag 拟合结果宣称为该底座假设的验证。
