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
MolmoPoint cloth-center query over two already calibrated cameras.

The perception extension uses the existing MolmoPoint model and calibrated A/B
RealSense pair to derive the complete per-run grasp plan: cloth `x/y`, surface
`z`, grasp/approach/lift heights, and yaw. It still does not add folding or
policy learning.

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

Every move is checked against the measured bounds plus a safety margin, the
local z lower bound, hard low-speed/acceleration caps, and read-only controller
IK before the robot is enabled. The first failure aborts the script immediately.

## Agent code tools

`cloth_agent/session.py::AgentSession` exposes the requested tools:

- `inspect_file(path)` reads project/current-run files only.
- `invoke_claude_code(prompt)` asks Claude Code to create or modify an
  experiment in the current workspace.
- `run_experiment(path)` performs validation, prints the full plan, and runs
  either the simulator or one explicitly confirmed physical rollout.
- `inspect_result(experiment)` reads the saved result, stdout, requested
  actions, actual EE poses (when available), gripper results, and errors.
- `invoke_molmo(image_paths, prompt)` invokes MolmoPoint on two images from the
  current run.
- `locate_cloth_center()` captures both calibrated RGB-D cameras, invokes
  MolmoPoint, and returns a validated base-frame cloth center plus the complete
  automatic grasp plan.

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

## Two-camera Molmo cloth center

MolmoPoint accepts RGB images rather than raw RealSense point clouds. The
runtime therefore uses this data flow:

```text
camera A RGB --------------------> Molmo semantic pixel A
camera A aligned depth ----------> diagnostic depth at pixel A
camera B aligned RGB-D ----------> B point cloud in base frame
                                      -> reproject B cloud into camera A
                                      -> select points near Molmo pixel A
                                      -> choose a stable local depth cluster
                                      -> place final 3D point on camera A's semantic ray
```

The model integration follows the existing
`molmo2/manual_tests/test_cloth_points.py` interface. The model is loaded once,
and only the primary camera A image is queried for the garment point. Camera B
is intentionally not asked to find its own garment center because its view may
contain only a partial garment. Instead, B supplies calibrated depth for the
same physical region selected in A. `extract_image_points` returns
`(object_id, image_num, pixel_x, pixel_y)` records.

The perception result is rejected before code generation if:

- Molmo does not return exactly one point for primary camera A;
- the A point is outside its image or has no valid diagnostic depth;
- B is unavailable and A's wider local depth patch has too few valid samples;
- B is unavailable and A's local depth crosses an unstable fold/edge;
- A and B depths for the same A-selected point differ by more than the
  configured threshold;
- the resulting grasp/approach/lift targets fail robot workspace validation.

No scene coordinate needs to be entered manually. After depth selection:

- `grasp_z = detected_surface_z`, with no fixed above-surface clearance, and
  clamped to the measured TCP lower limit;
- `approach_z = grasp_z + 80 mm`;
- `lift_z = grasp_z + 160 mm`, capped by the safe upper-Z limit;
- `yaw` comes from the recorded home/observation pose.

These clearances are fixed safety policy, not scene measurements. The xArm
controller already defines its TCP at the installed gripper tool point. A
read-only hardware check on 2026-08-11 reported
`tcp_offset=[0, 0, 172, 0, 0, 0]`. Real execution verifies this value before
enabling motion and aborts if the controller tool frame changed.

The xArm gripper URDF uses `drive_joint=0.0` for open and `0.85` for closed.
The measured lower TCP boundary may be used for the grasp descent without the
general 10 mm workspace margin; X/Y and upper-Z targets retain that margin, and
no command is allowed below the recorded `z_min`.

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
perception replaces all six before Claude is allowed to write motion code.

Only one-time hardware facts remain:

- measured bounds in `xarm_boundaries.json` (already present; `x_max` is
  intentionally optional and delegated to the xArm SDK);
- home/observation pose in `data/robot/xarm_init_pose.json` (already present);
- controller TCP offset `[0, 0, 172, 0, 0, 0]` and gripper pulse values
  (already recorded in `config/robot.example.json`);
- camera serials and calibrated extrinsics (already recorded in
  `config/perception.example.json`).

Re-measure these only when hardware is physically moved, a camera is remounted,
or the gripper/tool frame is changed. The runtime enforces
`grasp_z <= approach_z <= lift_z` and checks every move against the robot
bounds.

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

`config/perception.example.json` is pre-populated with the serials and A/B
extrinsic paths found in the current calibration project. Verify that camera A
is still serial `243722070226`, camera B is `261822074715`, and that the two
YAML files match the physical camera mounts before use.

Camera A must see enough of the garment for Molmo to select the intended point.
Camera B may see only a partial garment, but the physical region selected by A
must appear in B's point cloud if B is to supply depth. If a fold self-occludes
that region or it falls outside B's image, the runtime validates a wider A depth
patch. Stable A depth is accepted with status
`VALIDATED_PRIMARY_DEPTH_FALLBACK`; an unstable A patch still blocks execution.
The runtime never substitutes B's visible-region center for the hidden point.
Single-camera A remains an explicit diagnostic mode and is marked
`VALIDATED_SINGLE_VIEW` in the saved result.

Reachability note: the current saved camera-A plan around
`x=784.6 mm, y=-85.4 mm` passes simple Cartesian bounds, but the controller
returns IK code 10 for its approach, grasp, and lift poses. The Viser
console therefore displays it as unreachable and keeps animation/physical
execution disabled. Move the garment closer to the robot or correct the camera
calibration before trying to execute that same target.

## Start one Agent session

### 1. View only the camera/Molmo result

This creates a run and performs perception only. It never connects to xArm:

```bash
/home/CNS2026330003/miniconda3/envs/cali/bin/python -m cloth_agent create \
  --run-id preview_center \
  --goal "locate the cloth center"

/home/CNS2026330003/miniconda3/envs/cali/bin/python -m cloth_agent perceive \
  --run-dir runs/preview_center \
  --single-camera A
```

Look in `runs/preview_center/results/perception/` for the original images,
annotated Molmo points, aligned depth arrays, model output, and fused result.

### 2. View the complete Agent loop without robot motion

This captures both cameras, calculates all six scene values, asks Claude to
write the experiment, prints the source and full action sequence, then executes
only in the simulator:

```bash
/home/CNS2026330003/miniconda3/envs/cali/bin/python -m cloth_agent session \
  --goal "grasp cloth center, lift, release, return to observation pose" \
  --intent "Call home, open_gripper, move to center at approach_z, move to grasp_z, close_gripper, move to lift_z, open_gripper, move back to approach_z, then home. Copy the numeric values from experiment_config.json." \
  --detect-center \
  --single-camera A
```

The outer process uses the existing `cali` environment for RealSense and xArm.
It automatically launches MolmoPoint with the separate existing `molmo`
environment configured in `config/perception.example.json`.

Without `--detect-center`, all six experiment values must be supplied manually;
that mode is retained only for diagnostics.

Full real-session command:

```bash
/home/CNS2026330003/miniconda3/envs/cali/bin/python -m cloth_agent session \
  --goal "grasp cloth center, lift, release, return to observation pose" \
  --intent "Call home, open_gripper, move to center at approach_z, move to grasp_z, close_gripper, move to lift_z, open_gripper, move back to approach_z, then home. Copy the numeric values from experiment_config.json." \
  --detect-center \
  --single-camera A \
  --real
```

The command first creates and prints the run workspace, calls Claude Code,
prints the complete experiment source, and prints the exact action sequence.
No real robot command has happened at that point. Physical movement can begin
only after the operator types exactly `EXECUTE_SINGLE_VIEW` for a single-camera
plan (`EXECUTE` remains the dual-camera token). The first physical command in
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
1. choose `Camera A only` or `Camera A primary + B auxiliary depth`;
2. capture aligned RealSense RGB-D and inspect the photographs/3D point cloud;
3. run Molmo, display the marked image center, and mark the same base-frame
   point in the point cloud;
4. choose the standard center-grasp path or generate a separate garment
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
      camera_0_A_annotated.png
      camera_1_B.png
      camera_1_B_depth_m.npy
      camera_1_B_annotated.png
      molmo_output.json
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
