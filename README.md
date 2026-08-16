# Minimal real-xArm Agent experimentation loop

This repository now implements only the first requested loop:

```text
Agent intent
  -> Claude Code writes a visible experiment_*.py
  -> AST/static safety preflight
  -> print every planned action and xyz/yaw
  -> explicit one-rollout confirmation
  -> restricted RobotAPI executes on xArm
  -> result/stdout/trace are saved
  -> manual result + memory-based stop/modify decision
```

It does not implement folding, learned policies, automatic success
classification, or automatic retry. Perception is deliberately limited to one
calibrated two-camera RGB-D fusion pass.

The perception extension transforms both calibrated A/B RealSense clouds into
the robot base frame, voxel-fuses them, fits the table plane, and extracts the
largest garment-height component. It reports only cloth `x/y` and observed
surface `z`; Claude then chooses grasp/approach/lift/transfer/release heights
and yaw from the saved views.

## Existing robot interfaces and the new wrapper

The original repository contained:

- `scripts/record_xarm_boundaries.py`: `XArmAPI(ip)`, `motion_enable`,
  `set_mode`, `set_state`, `get_position`, and `get_servo_angle`.
- `xarm_boundaries.json`: measured base-frame bounds. It currently lacks
  `x_max`, which is allowed: upper-X reachability is delegated to read-only
  controller inverse kinematics and the final controller motion command.
- `data/robot/xarm_init_pose.json`: the existing home/observation joint pose
  and TCP pose.

The thin real-robot adapter is `cloth_agent/robot_api.py`:

- initialization: `XArmBackend.__init__`
- Cartesian motion: `XArmBackend.move` -> `XArmAPI.set_position(...)`
- gripper initialization: `set_gripper_mode(0)`,
  `set_gripper_enable(True)`, `set_gripper_speed(...)`
- gripper open/close: `set_gripper_position(...)`
- home/observation pose: `set_servo_angle(...)` using
  `data/robot/xarm_init_pose.json`

Generated code never receives `XArmBackend` or `XArmAPI`. It receives only:

```python
move(x, y, z, yaw)
open_gripper()
close_gripper()
home()
```

Every move is checked against the measured bounds, the local z lower bound,
hard low-speed/acceleration caps, and read-only controller IK before the robot
is enabled. The configured workspace margin is currently `0 mm`; the first
failure aborts the script immediately.

## Agent code tools

`cloth_agent/session.py::AgentSession` exposes the requested tools:

- `inspect_file(path)` reads project/current-run files only.
- `invoke_claude_code(prompt)` asks Claude Code to create or modify an
  experiment in the current workspace. The complete prompt, raw stdout/stderr,
  command metadata, and return code are saved under `results/claude/`.
- `run_experiment(path)` performs validation, prints the full plan, and runs
  either the simulator or one explicitly confirmed physical rollout.
- `inspect_result(experiment)` reads the saved result, stdout, requested
  actions, actual EE poses (when available), gripper results, and errors.
- `locate_cloth_center()` captures both calibrated RGB-D cameras, performs dense
  A/B fusion in the robot base frame, and returns a validated fused garment
  center plus the observed surface height. It does not generate motion
  waypoints; Claude chooses the approach, grasp, lift, transfer, release, and
  yaw actions from the saved views.

`record_manual_result` accepts only `SUCCESS`, `FAILED_GRASP`, `FAILED_LIFT`,
or `OTHER_FAILURE`. `update_memory` records the hypothesis and why the next
experiment changes, not only the parameter history.

## Claude Code confinement

`cloth_agent/claude.py` runs the installed `claude` CLI with:

- working directory and `--add-dir` set to `runs/<run_id>/workspace/`;
- only the `Read`, `Edit`, and `Write` tools (no Bash/shell tool);
- safe mode and an explicit workspace-only system prompt;
- no shell interpolation (`subprocess.run(..., shell=False)`).

The core project is not added as a writable Claude directory. The workspace
contains `ROBOT_API.md`, the experiment configuration, robot configuration,
and `memory.md`, so Claude does not need to edit the core project. If it finds
an infrastructure problem, its contract requires an `ENGINEERING_ISSUE:`
report instead of a core edit.

Before execution, `cloth_agent/experiment.py` parses generated code as Python
AST. It rejects all imports (including `xarm`), attribute access, filesystem or
shell access, unknown calls, loops/retries, exception handling, dynamic code,
and anything other than one `run()` function containing the four allowed robot
calls and simple numeric variables/arithmetic. The script is then executed
with empty builtins and only the four bound RobotAPI functions.

## Dense two-camera RGB-D fusion

The runtime uses this data flow:

```text
camera A aligned RGB-D -----------> A point cloud in robot base frame
camera B aligned RGB-D -----------> B point cloud in robot base frame
                                      -> voxel-fuse A+B points
                                      -> fit the table plane
                                      -> select garment points above the table
                                      -> choose the fused garment grasp point
                                      -> save fused points and a top-down height map
```

The perception result is rejected before code generation if the two-camera
cloud cannot produce enough valid points, the fitted table is unstable, or no
connected garment-height component can be found.

Each dense-fusion result also saves visual height-map diagnostics:

- `camera_A/B_height_above_table_mm.npy`: per-pixel surface height above the
  fitted table plane, in millimeters;
- `camera_A/B_height_map_heatmap.png`: garment-focused heatmap of that height
  difference; brighter colors mean a larger garment/table height difference;
- `camera_A/B_height_map_heatmap_global.png`: the same height map normalized over
  the whole valid camera image;
- `camera_A/B_height_map_boundary.png`: the focused heatmap with the fused
  garment boundary overlaid in white;
- `camera_A/B_height_gradient_edges.png`: internal height-gradient/occlusion
  evidence overlaid in cyan;
- `fused_height_map_heatmap.png` and `fused_height_map_boundary.png`: the
  top-down A+B fused height-above-table map and its boundary;
  `fused_fold_edges.png` remains as a compatibility filename for the fused
  height-gradient diagnostic.

For every valid camera pixel and fused point, the runtime transforms the RGB-D
measurement into the robot base frame, evaluates the fitted table plane at that
same `x/y`, and stores `surface_z_mm - table_z_mm`. The heatmaps colorize this
height-above-table map directly; there is no longer a second conversion to fold
depth below the garment upper surface. These images are diagnostic overlays only;
the segmented fused point cloud remains the source used for perception validation.

No scene coordinate needs to be entered manually. Perception writes a fused
center/reference surface height (the legacy `grasp_z` field is used as
`surface_z_mm` for compatibility) plus per-camera calibrated coordinate guides.
The center is not a mandatory grasp target. Uniform cyan `Rxxx` references in
`camera_A/B_coordinate_overlay.png` map through
`camera_A/B_coordinate_guide.json` to measured robot-base XYZ; they are not
ranked grasp candidates. The full per-pixel mapping is saved in
`camera_A/B_base_xyz_mm.npy`. Claude chooses the visual/geometric region and all
motion waypoints. The generated action program is then checked against robot
bounds, static preflight, controller IK, and animation before execution.

## Anchor-discovery free exploration

Claude free/automatic exploration searches for a usable garment lifting
anchor rather than optimizing every action for immediate visible coverage. A
small lift, drag, repositioning, or test grasp may be used to gather evidence;
there is no hard-coded grasp-candidate selector or probe/verify state machine.
Previous automatic-loop proposals and before/after evaluations are passed into
the next planning prompt.

The reusable `laydown` skill is procedural prompt guidance, not a hidden robot
trajectory. When Claude believes a grasp supports a useful hanging
configuration it may invoke `laydown`, then it must still emit every concrete
`move`/gripper action itself. The intended maneuver is a quasi-static retreat
and descent followed by controlled release, not a fling or high drop.

The xArm controller already defines its TCP at the installed gripper tool
point. A read-only hardware check on 2026-08-11 reported
`tcp_offset=[0, 0, 172, 0, 0, 0]`. Real execution verifies this value before
enabling motion and aborts if the controller tool frame changed.

The xArm gripper URDF uses `drive_joint=0.0` for open and `0.85` for closed.
The measured lower TCP boundary may be used for the grasp descent with the
configured `0 mm` workspace margin, and no command is allowed below the
recorded `z_min`.

## What must be configured

The workspace measurement is already present for the current machine. If the
robot/table layout changes, rerun this manual/free-drive measurement procedure;
it does not run an experiment:

```bash
/home/CNS2026330003/miniconda3/envs/cali/bin/python \
  scripts/record_xarm_boundaries.py \
  --ip 192.168.1.200 \
  --output xarm_boundaries.json
```

After re-measuring, review the generated file and ensure these five local limits exist: `x_min`,
`y_min`, `y_max`, `z_min`, and `z_max`. `x_max` is optional and is checked by
the xArm controller/SDK instead.

For the current machine, no per-run experiment configuration is required.
`config/experiment.example.json` deliberately contains six `null` values;
perception fills the center reference and observed surface height while saving
the coordinate guides above. Claude selects the interaction region and supplies
the motion values in the generated action program.

Only one-time hardware facts remain:

- measured bounds in `xarm_boundaries.json` (already present; `x_max` is
  intentionally optional and delegated to the xArm SDK);
- home/observation pose in `data/robot/xarm_init_pose.json` (already present);
- controller TCP offset `[0, 0, 172, 0, 0, 0]` and gripper pulse values
  (already recorded in `config/robot.example.json`);
- camera serials and calibrated extrinsics (already recorded in
  `config/perception.example.json`).

Re-measure these only when hardware is physically moved, a camera is remounted,
or the gripper/tool frame is changed. Every explicit Claude move is checked
against the robot bounds; `require_ready()` is used only for manually supplied
complete plans.

### Very-slow motion profile

The default real-robot profile is intentionally slow so the operator has time
to use the emergency stop:

- Cartesian moves: `15 mm/s`, acceleration `30 mm/s^2`
- `home()` joint motion: `5 deg/s`, acceleration `10 deg/s^2`
- gripper speed: `500`

The safety layer also rejects Cartesian speeds above `30 mm/s`, Cartesian
acceleration above `60 mm/s^2`, home speed above `10 deg/s`, or home
acceleration above `20 deg/s^2`. Generated experiment code cannot change any
of these values.

After every explicitly confirmed real experiment launched through
`AgentSession.run_experiment`, the runtime makes a separate best-effort
`home()` call in a `finally` path, regardless of whether the Claude rollout
succeeded or failed. The Home attempt has its own result JSON and is also
summarized under `results/mandatory_return_home/`; a hardware/controller fault
can still prevent physical return, but the attempt and error are never hidden.

`config/perception.example.json` is pre-populated with the serials and A/B
extrinsic paths found in the current calibration project. Verify that camera A
is still serial `243722070226`, camera B is `261822074715`, and that the two
YAML files match the physical camera mounts before use.

Both cameras must provide enough calibrated depth for the fused cloud. The
runtime fits a robust table plane, selects points above that plane, keeps the
largest connected garment component, and blocks execution if that component is
too small or the table fit is unstable. Single-camera perception is disabled.

Reachability note: the current saved camera-A plan around
`x=784.6 mm, y=-85.4 mm` passes simple Cartesian bounds, but the controller
returns IK code 10 for its approach, grasp, and lift poses. The Viser
console therefore displays it as unreachable and keeps animation/physical
execution disabled. Move the garment closer to the robot or correct the camera
calibration before trying to execute that same target.

## Start one Agent session

### 1. View only the fused RGB-D result

This creates a run and performs perception only. It never connects to xArm:

```bash
/home/CNS2026330003/miniconda3/envs/cali/bin/python -m cloth_agent create \
  --run-id preview_center \
  --goal "locate the cloth center"

/home/CNS2026330003/miniconda3/envs/cali/bin/python -m cloth_agent perceive \
  --run-dir runs/preview_center \
  --perception-config config/perception.example.json
```

Look in `runs/preview_center/results/perception/` for the original A/B images,
aligned depth arrays, fused base-frame points, source masks, and height map.

### 2. View the complete Agent loop without robot motion

This captures both cameras, calculates the fused center/surface observation,
asks Claude to choose all motion waypoints and write the experiment, prints the source and full action sequence, then executes
only in the simulator:

```bash
/home/CNS2026330003/miniconda3/envs/cali/bin/python -m cloth_agent session \
  --goal "grasp cloth center, lift, release, return to observation pose" \
  --intent "Inspect the fused A/B garment views and choose a cautious approach, grasp, lift, transfer, release, yaw, and return-home sequence. Emit explicit move coordinates; use experiment_config.json only for the fused center and surface observation." \
  --detect-center
```

The outer process uses the existing `cali` environment for RealSense and xArm;
no Molmo process or GPU model is launched.

Without `--detect-center`, all six experiment values must be supplied manually;
that mode is retained only for diagnostics.

Full real-session command:

```bash
/home/CNS2026330003/miniconda3/envs/cali/bin/python -m cloth_agent session \
  --goal "grasp cloth center, lift, release, return to observation pose" \
  --intent "Inspect the fused A/B garment views and choose a cautious approach, grasp, lift, transfer, release, yaw, and return-home sequence. Emit explicit move coordinates; use experiment_config.json only for the fused center and surface observation." \
  --detect-center \
  --real
```

The command first creates and prints the run workspace, calls Claude Code,
prints the complete experiment source, and prints the exact action sequence.
No real robot command has happened at that point. Physical movement can begin
only after the operator types exactly `EXECUTE`. The first physical command in
the requested sequence is normally `home()`; this is where
`XArmBackend.set_servo_angle` actually starts robot motion.

There is no automatic retry. After a rollout the operator supplies the manual
result. A modification requires explicitly choosing `MODIFY_EXPERIMENT` and
entering the reason. Physical execution has no per-run count limit, but every
rollout still requires a fresh validated plan and explicit confirmation.

## Complete Viser operation console

The Viser application now owns the complete interactive workflow. Start a new
preview-only run with one command:

```bash
python -m cloth_agent viewer \
  --run-id viser_grasp_01
```

Open `http://127.0.0.1:8080`. All subsequent steps are buttons inside Viser:

0. when the server was started with `--enable-real`, optionally click
   `Init: Return arm to Home` to send exactly one low-speed `home()` action;
1. capture aligned A/B RealSense RGB-D and inspect the photographs/3D point cloud;
2. run dense A/B fusion and inspect the fused base-frame cloud/height map;
3. choose the standard center-grasp path or generate a separate garment
   randomization path, then inspect every named path point and restricted source;
5. ask the xArm controller for read-only IK for every Cartesian target;
6. load the copied xArm7 + xArm gripper URDF and play the complete arm/gripper
   animation with a frame slider, play, pause, reset, and loop controls;
7. optionally authorize exactly one physical rollout.

The URDF and required visual/collision meshes are stored under
`assets/robots/xarm7/`. Its `joint_tcp` is 172 mm, matching the controller's
saved TCP tool offset.

To allow the final physical button, start the same console with:

```bash
python -m cloth_agent viewer \
  --run-id viser_grasp_real_01 \
  --enable-real
```

Real authority is accepted only on a loopback-bound server. The physical button
remains disabled until perception, static validation, controller TCP/IK checks,
and URDF animation generation pass, and the source must remain unchanged.
The separate red Init button is available immediately in real mode because it
contains only the configured `home()` action. It still performs the live TCP
tool-offset, controller-state, workspace, and Home configuration checks before
moving, and uses the configured low Home joint speed.
After those gates pass, clicking the single red
`Confirm and execute one physical rollout` button is the explicit confirmation
and starts one physical rollout immediately; no token or extra checkbox is
required.

Controller IK is a hard validation gate. A target that passes simple Cartesian
bounds but has no inverse-kinematics solution is shown as a hard failure, and
animation/execution stay disabled. There is no automatic retry.

### Single-gripper garment randomization

After perception, click `Generate garment randomization path`. Each click uses
a recorded random seed and produces a deterministic, reviewable sequence:

```text
approach -> grasp -> lift -> inward drag -> twist while moving to drop point
         -> low-air release -> retreat -> home
```

The Viser panel prints every path point as `x/y/z/yaw`, draws the complete TCP
polyline in the point cloud, prints the restricted RobotAPI source, and keeps
physical execution disabled until static preflight, controller IK, and URDF
animation all pass. `Use standard center-grasp path` switches back without
requiring a new camera capture.

An existing run can also be reopened:

```bash
python -m cloth_agent viewer \
  --run-dir runs/preview_a_01
```

## Saved run layout

```text
runs/<run_id>/
  run_metadata.json
  workspace/
    ROBOT_API.md
    robot_config.json
    experiment_config.json
    experiment_001_grasp_lift_drop.py
    experiment_002.py                 # only after an explicit modification
    memory.md
    results/claude/*.json             # Claude invocation records
  results/
    perception/center_<timestamp>/
      camera_0_A.png
      camera_0_A_depth_m.npy
      camera_A_height_above_table_mm.npy
      camera_A_height_map_heatmap.png
      camera_A_base_xyz_mm.npy
      camera_A_coordinate_guide.json
      camera_A_coordinate_overlay.png
      camera_1_B.png
      camera_1_B_depth_m.npy
      camera_B_height_above_table_mm.npy
      camera_B_height_map_heatmap.png
      camera_B_base_xyz_mm.npy
      camera_B_coordinate_guide.json
      camera_B_coordinate_overlay.png
      fused_points_base_mm.npy
      fused_colors_rgb.npy
      fused_source_mask.npy
      fused_height_above_table_mm.npy
      fused_height_map_mm.npy
      fused_height_map_preview.png
      result.json
    experiment_001_grasp_lift_drop.json
    experiment_001_grasp_lift_drop.source.py
    experiment_001_grasp_lift_drop.stdout.txt
    experiment_001_grasp_lift_drop.trace.json
```

For automation, the same flow is available as separate `create`, `perceive`,
`generate`, `preflight`, `run`, `inspect`, `label`, and `memory` subcommands.
`run --real` also requires `--confirm-real`; call `preflight` and review its
output first.
