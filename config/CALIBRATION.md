# Current device calibration

- Robot IP: `192.168.2.232`.
- Camera A: wrist-mounted RealSense D435, serial `317222073552`.
- Color stream: 1280 × 720 at 30 FPS.
- The optional separate Camera C is disabled by default.

`extrinsics_wrist.yaml` is an unchanged copy of
`/home/sja/RobotCamCalib/outputs/extrinsics_wrist.yaml`, updated there on
2026-09-14. Its `X_CammountCam` maps camera coordinates into `link_eef`,
not directly into the robot base frame.

`extrinsics_A.yaml` contains that calibration plus the robot connection,
mount frame, and model reference. `load_extrinsics()` reads the current joint
angles and composes `X_base_eef @ X_eef_camera`. The associated
`calibration_xarm6.urdf` is copied from the same calibration project's
`assets/robots/xarm6/xarm6_wo_ee.urdf`; it is loaded without meshes for forward
kinematics. Capture assumes the robot stays still during image acquisition.
This model is used for camera calibration, not to replace the project's
motion-planning robot model.

`intrinsics_A.yaml` records the connected device's factory color intrinsics
for the configured stream. The calibration project's current wrist workflow
reads these from RealSense rather than loading its older intrinsics YAMLs.
Runtime RGB-D capture likewise reads the active stream's intrinsics from
RealSense; this YAML is a reference export, not a runtime override.

Home and observation remain separate: `data/robot/xarm_init_pose.json` and
`data/robot/xarm_perception_pose.json`, respectively.
