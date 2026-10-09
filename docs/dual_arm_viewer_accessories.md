# Viser 腕部附件（仅可视化）

`cloth_agent/dual_arm/viewer_accessories.py` 给只读快照 Viser 补齐 arm6 夹爪、两台 D435 和安装支架。
arm6 原始 URDF 为 wo_ee，不含夹爪；补全复用 xarm7 URDF 的 link_eef 下夹爪子树，使用默认开合角，未采集实际夹爪状态。
TCP 坐标继续由 FK 和配置工具偏移得到，不因显示修改；加大原点球/坐标轴并设置标签不做深度遮挡。

相机光学坐标分别取 config/extrinsics_A.yaml 与 config/calibration/dual_arm_working_20261008/camB_extrinsics.yaml 的 X_CammountCam。
A 明确标注 link_eef；B 文件没有 mount 字段，沿用该工作标定的 link_eef 挂载解释，需与标定源确认。
这些变换平移为米；ArmModel 输出毫米，因此只在转换到 Viser 世界坐标时除以 1000。

D435 壳体采用 90×25×25 mm，来源 https://www.realsenseai.com/products/stereo-depth-camera-d435/ 。
手眼标定给的是光学坐标，不是外壳中心；当前壳体在光学坐标后方放置，光心到壳体中心的真实偏置未建模，镜头外观为示意。
打印件按照片构造法兰环、连接条和相机端连接点；没有实测尺寸/CAD，不能用于碰撞认证。
附件只存在于 Viser 场景，尚未加入 collision_capsules，净空数字仍仅针对原先胶囊模型。
