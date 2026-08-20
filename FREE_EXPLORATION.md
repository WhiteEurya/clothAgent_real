# Claude garment free exploration

This is a standalone Viser console. It does not replace or patch the existing
center-grasp dashboard in `cloth_agent.viewer`.

Start a preview-only console in a new deferred run:

```bash
python -m cloth_agent.free_exploration --run-id claude_explore_01
```

Or reopen a run that already contains a saved RGB-D/Molmo result:

```bash
python -m cloth_agent.free_exploration \
  --run-dir runs/viser_grasp_01 \
  --port 8081
```

The workflow is:

1. Capture and validate the garment views with the existing RGB-D/Molmo
   perception pipeline.
2. Ask Claude to describe the visible folds and occlusions and return one
   short action proposal that advances the goal of opening and spreading the
   garment as much as safely possible.
3. Inspect Claude's observation, confidence, expected observation, safety
   notes, and restricted `RobotAPI` action table in Viser.
4. Run static preflight, workspace checks, read-only controller IK, and URDF
   animation before any physical authority is exposed.
5. Play or scrub the animation. Physical execution is available only when the
   server is loopback-bound and started with `--enable-real`; clicking the red
   button is the explicit confirmation for one rollout.

Claude is invoked in read-only mode. Its response must be a strict JSON object
with garment reasoning plus actions using only `move`, `open_gripper`,
`close_gripper`, and `home`. Invalid schema, unknown actions, non-finite values,
or a path with no Cartesian move are rejected before source generation.

The legacy Viser automatic planner is split into two independent Claude processes:

1. **Visual planning:** the original `safe-mode + plan + Read` flow, with no
   MCP configuration. Claude inspects the images/overlays and returns one final
   Camera A/B Rxxx plus its opening and motion intent.
2. **Final grounding/run generation:** a short context containing only the
   stage-one decision and robot constraints. This process exposes only
   `lookup_reference(camera, reference_id)`, calls the already selected Rxxx
   exactly once, and emits final numeric actions.

Between these stages, the runtime validates the selected Rxxx against the saved
calibration and robot XY workspace. If it is missing or outside the safe bounds,
Stage 2 is not started: the rejected Camera/Rxxx is logged, excluded, and the
same current images return to Stage 1 for a different visual selection. The
number of reference reselections is bounded by `--max-replans`.

The MCP server rejects a second successful lookup. The returned measurement
grounds the selected grasp XY; Claude still decides approach, grasp TCP height,
lift, retreat, laydown, release, and yaw. A saved-coordinate verification checks
that the final grasp uses the selected Rxxx before preflight. The server cannot
write files, run commands, select candidates, or control the robot.

The standalone console defaults to
`config/perception.free_exploration.json`. Its grasp TCP target is the validated
garment surface (`0 mm` contact clearance), with approach and lift targets a further 80 mm
and 160 mm above the grasp target. Re-run capture/perception after changing
this configuration; an existing run keeps its previously derived heights
until a new perception result is produced.

## Automatic real loop

The manual console is still available. For an explicit automatic loop with a
live CamA preview inside Viser, start the separate module:

```bash
python -m cloth_agent.auto_exploration \
  --run-id claude_auto_01 \
  --max-iterations 0 \
  --settle-s 2 \
  --record-rollouts \
  --enable-real
```

With `--record-rollouts`, the loop starts the standalone Camera A/B recorder
only after perception, Claude planning, static preflight, and read-only
controller IK have passed. It records the physical rollout plus mandatory
return-home, closes both cameras, and only then starts the after-observation.
Each iteration saves RGB/depth MP4, a four-panel composite, timestamps, and
native `.db3` recordings under
`results/auto_exploration/<stamp>/iteration_NNN/rollout_recording_<stamp>/`. A recorder
startup failure is a hard pre-execution failure and sends no robot command.
The MP4 files are finalized as H.264/AVC with `yuv420p` and fast-start for
compatibility with browsers, phones, and common desktop players.

After Claude proposes a plan, the dashboard defines each grasp target as the
last finite `move(x,y,z,yaw)` immediately preceding `close_gripper()`. It shows
the Base-frame XYZ/yaw in a dedicated panel, a red 3-D sphere plus yaw axis in
Viser, and projected crosshairs on the Camera A/B RGB captures. Validated
overlays are saved as `grasp_target_camera_A.png`,
`grasp_target_camera_B.png`, and `grasp_target_visualization.json` in the
iteration directory. A plan without a grounded grasp displays `unknown`.

Open `http://127.0.0.1:8082`. The Viser page contains the live CamA RGB image,
normalized depth image, and a calibrated CamA point cloud in the robot base
frame. After each synchronized capture it also keeps the latest A/B RGB images
and point clouds in the same page. The preview pauses while A/B capture runs,
so the RealSense devices are never opened by two competing processes. The loop
starts automatically when the module launches and continues until Claude judges
that it should stop, the operator clicks `Stop after current phase`, or a hard
failure occurs. The `Restart automatic exploration` button starts a fresh loop.

Use `--max-iterations N` to cap the run; the default `--max-iterations 0` means
continuous opening until Claude judges the garment reasonably maximally spread
or safe grounded continuation is no longer possible.

The headless path defaults to `claude_global`. Claude receives the complete
Camera A/B RGB scenes, garment-only images, height maps, boundaries, gradients,
calibration context, and prior physical outcomes. It summarizes the current
state, decides the next experiment, and chooses any Camera A/B pixel. The
runtime does not generate or rank Sxxx/Rxxx candidates.

After the visual decision, a read-only MCP server exposes only one
`sample_local_surface(camera, x_px, y_px, radius_px=3)` call. The returned local
median Base X/Y must match the grasp move within 2 mm. Workspace, action schema,
controller IK, and trajectory validation remain hard execution gates, but they
never select the interaction point. A failed gate is returned once to Claude
with the exact reason. Real rollouts append complete before/after `keep/change`
evaluations to `workspace/global_experience.jsonl`.

Use the terminal-only entry point:

```bash
/home/CNS2026330003/miniconda3/envs/cali/bin/python \
  -m cloth_agent.molmo_keypoint_cli \
  --project-root . \
  --run-id claude_global_cli_01 \
  --planning-policy claude_global \
  --perception-config config/perception.free_exploration.json \
  --max-iterations 1
```

Add `--enable-real --max-iterations 0` only for continuous physical execution.
The CLI prints global planning, selected pixel/measurement, IK, execution, and
before/after evaluation results under
`runs/<run-id>/results/molmo_keypoint_cli/<timestamp>/`. The older
Molmo/Sxxx/local-Rxxx flow is retained only as explicit
`--planning-policy semantic_local`; only that compatibility mode loads Molmo or
uses the `19000 MiB` GPU gate.

The legacy Viser console still shows a live `Claude stage timer` panel with the active stage, Stage-1
reference-attempt count, elapsed time/timeout, and completed duration for each stage. The same values are saved
under `planning_timing` in the iteration record, while the stage-one result is
saved separately as `claude_visual_plan.json`.

If Claude proposes a path that fails static validation or read-only controller
IK, the exact error and rejected actions are sent back to Claude for a new
proposal. This is limited to two replans by default and can be changed with
`--max-replans N`. No replan is attempted after physical motion has started.

```text
Viser CamA RGB + depth + point cloud preview
  -> synchronized A/B capture and center/depth validation
  -> Claude action proposal
  -> static preflight + read-only controller IK
  -> exactly one physical rollout
  -> settle and a fresh CamA/B capture
  -> Claude before/after usefulness judgement
  -> continue with Claude's next objective, or stop
```

`Stop after current phase` never interrupts a command already sent to the arm;
it takes effect at the next safe phase boundary. Any malformed Claude output,
failed perception, incomplete rollout, or other hard runtime error stops the
loop without a physical retry. Pre-execution schema/IK failures use the bounded
Claude replan path described above. Every iteration is saved under
`runs/<run_id>/results/auto_exploration/`.

Each iteration directory contains the captured perception metadata, before/after
image paths, Claude's raw planning and evaluation output, validated proposal,
generated RobotAPI source, static preflight, controller IK result, and physical
execution result. The top-level `iteration_###.json` is the consolidated record;
failed iterations are saved as `iteration_###_failed.json`.

The motion heights are not substituted by a fixed template. Claude must output
every `move(x,y,z,yaw)` waypoint itself, including approach, grasp, lift,
transfer, and release height. The runtime validates those exact coordinates and
the controller IK before any physical execution.

## Standalone Camera A RealSense monitor

Camera A can be opened as a separate OpenCV stream for diagnostics. It uses the
same manual RGB controls as the A/B perception configuration and must not run
at the same time as the automatic A/B loop, because both processes would own
the same device.

For Camera A (`261722071490`):

```bash
/home/CNS2026330003/miniconda3/envs/cali/bin/python -m cloth_agent.camera_window \
  --project-root . \
  --serial 261722071490 \
  --label A \
  --width 640 \
  --height 480 \
  --fps 30
```

The window is named `CamA live monitor (261722071490)`. Close it with `q`,
`Esc`, or the window close button. Keep this process separate from the
automatic A/B loop because it competes with CamA ownership.
