"""Camera A point selection -> assisted Arm6 demonstration collection."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time

import numpy as np

from .dual_arm.geometry import DualArmError, matrix_pose, pose_matrix
from .gamepad_teleop import Gamepad, InputGuard, Intent, Mapper, list_gamepads, control_status


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def tcp_command(pose):
    pose = np.asarray(pose, dtype=float)
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise ValueError('invalid World TCP pose')
    return f'motion.move_tcp(left={np.r_[pose[:3]/1000, np.radians(pose[3:])].tolist()})\n'


def transfer_program(lifted, start, target, *, clearance, place_offset):
    """Preserve manually chosen orientation, translate aloft, then lower."""
    lifted, start, target = map(lambda x: np.asarray(x, dtype=float), (lifted, start, target))
    if lifted.shape != (6,) or start.shape != (3,) or target.shape != (3,):
        raise ValueError('invalid transfer geometry')
    if not all(np.isfinite(x).all() for x in (lifted, start, target)):
        raise ValueError('non-finite transfer geometry')
    if not np.isfinite(clearance) or clearance <= 0 or not np.isfinite(place_offset) or place_offset < 0:
        raise ValueError('invalid transfer heights')
    if lifted[2] < target[2] + place_offset:
        raise DualArmError('当前高度低于放置高度，请用手柄调整后再按 START')
    across = lifted.copy(); across[:2] = target[:2]
    placed = across.copy(); placed[2] = target[2] + place_offset
    retreat = placed.copy(); retreat[2] += clearance
    return (tcp_command(across) + tcp_command(placed)
            + 'motion.gripper(arm="left", action="open")\n' + tcp_command(retreat))


def automatic_program(runtime, start, target, rpy, *, approach_height, grasp_offset,
                      lift_height, clearance, place_offset):
    """Plan the complete point-to-point pickup before dispatching any command."""
    start, target, rpy = (np.asarray(x, dtype=float) for x in (start, target, rpy))
    if any(x.shape != (3,) or not np.isfinite(x).all() for x in (start, target, rpy)):
        raise ValueError('invalid automatic pickup geometry')
    if not np.isfinite([approach_height, grasp_offset, lift_height, clearance, place_offset]).all():
        raise ValueError('invalid automatic pickup heights')
    if not (-30 <= grasp_offset <= 10 and 0 < approach_height <= 200
            and 0 < lift_height <= 200 and approach_height > grasp_offset
            and 0 < clearance <= 200 and 0 <= place_offset <= 50
            and lift_height > max(grasp_offset, place_offset)):
        raise ValueError('invalid automatic pickup heights')
    if np.linalg.norm(target[:2]-start[:2]) < 10:
        raise DualArmError('start 与 target 水平距离至少需要 10 mm')
    approach = np.r_[start+[0, 0, approach_height], rpy]
    grasp = np.r_[start+[0, 0, grasp_offset], rpy]
    lifted = grasp.copy(); lifted[2] = max(start[2], target[2])+lift_height
    source = (approach_program(runtime, approach) + tcp_command(grasp)
              + 'motion.gripper(arm="left", action="close")\n' + tcp_command(lifted)
              + transfer_program(lifted, start, target, clearance=clearance, place_offset=place_offset))
    return source


def approach_program(runtime, approach):
    """Solve the approach endpoint; validate direct joint interpolation."""
    seed = np.asarray(runtime.coordinator.snapshots()['left']['joints'])
    base = matrix_pose(np.linalg.inv(runtime.config.arms['left'].world_from_base)
                       @ pose_matrix(approach))
    runtime.planning_context = dict(stage='approach_endpoint_ik')
    joints = runtime._inverse_target('left', base, seed, 'host_approach_ik')
    return ('motion.gripper(arm="left", action="open")\n'
            + f'motion.move_joints(left={np.radians(joints).tolist()})\n')


def select_points(observation, max_spread, *, single=False):
    """Display the saved image at a known scale; reject unsupported depth."""
    import tkinter as tk
    from PIL import Image, ImageTk
    count = 1 if single else 2
    instruction = ('Click START. Enter: confirm; R: reset; Esc: abort.' if single else
                   'Click START, then TARGET. Enter: confirm; R: reset; Esc: abort.')
    root = tk.Tk(); root.title('Arm6: select START' if single else 'Arm6: select START, then TARGET')
    image = Image.open(observation.image).convert('RGB')
    scale = min(1., 1200/image.width, 750/image.height)
    size = (round(image.width*scale), round(image.height*scale))
    photo = ImageTk.PhotoImage(image.resize(size))
    canvas = tk.Canvas(root, width=size[0], height=size[1]); canvas.pack()
    canvas.create_image(0, 0, image=photo, anchor='nw')
    text = tk.StringVar(value=instruction)
    tk.Label(root, textvariable=text).pack()
    points = []; accepted = []

    def reset(event=None):
        points.clear(); canvas.delete('point')
        text.set(instruction)

    def click(event):
        if len(points) == count:
            return
        pixel = [round(event.x*image.width/size[0]), round(event.y*image.height/size[1])]
        try:
            xyz = observation.sample(pixel, max_spread).tolist()
        except DualArmError as exc:
            text.set(str(exc)); return
        label = ('start', 'target')[len(points)]
        points.append(dict(name=label, pixel_xy=pixel, world_xyz_mm=xyz))
        canvas.create_oval(event.x-5, event.y-5, event.x+5, event.y+5,
                           outline='lime' if label == 'start' else 'red', width=3, tags='point')
        canvas.create_text(event.x+8, event.y, text=label, fill='yellow', anchor='w', tags='point')
        text.set(f'{label}: World mm {np.round(xyz, 1)}. Enter confirms {count} point(s).')

    def confirm(event=None):
        if len(points) == count:
            accepted.extend(points); root.destroy()

    canvas.bind('<Button-1>', click)
    root.bind('<Return>', confirm); root.bind('r', reset)
    root.bind('<Escape>', lambda event: root.destroy())
    root.mainloop()
    if not accepted:
        raise KeyboardInterrupt('point selection cancelled')
    return accepted


class TrajectoryRecorder:
    """Timestamped measured joints/TCP and gripper pulses, nominally 10 Hz."""
    def __init__(self, runtime, folder):
        self.runtime = runtime; self.folder = folder
        self.phase = 'capture'; self.error = None; self.samples = 0
        self.started = time.monotonic(); self.done = threading.Event()
        self.lock = threading.Lock(); self.thread = None; self.api = None
        self.events = open(folder/'events.jsonl', 'w', buffering=1)

    def event(self, kind, **data):
        with self.lock:
            self.events.write(json.dumps(dict(t_s=time.monotonic()-self.started,
                phase=self.phase, kind=kind, **data), ensure_ascii=False, allow_nan=False)+'\n')

    def stage(self, phase):
        self.phase = phase; self.event('stage')
        print(f'traj: {phase}', flush=True)

    def start(self):
        from xarm.wrapper import XArmAPI
        self.api = XArmAPI(self.runtime.config.arms['left'].ip, is_radian=False)
        self.thread = threading.Thread(target=self._run, name='traj-recorder', daemon=True)
        self.thread.start()

    def _read_feedback(self, connection):
        with connection._report_lock:
            report = connection._latest_report
            # Timestamp after reading the report: a callback may have delivered
            # a new sample since this recorder iteration began.
            observed = time.monotonic()
        if report is None:
            raise DualArmError('trajectory feedback missing')
        joints, sdk_pose, sampled = report
        q = np.asarray(joints[:6], dtype=float)
        pose = np.asarray(sdk_pose, dtype=float)
        if q.shape != (6,) or pose.shape != (6,) or not np.isfinite(q).all() or not np.isfinite(pose).all():
            raise DualArmError('trajectory feedback has invalid joints or TCP pose')
        age = observed-sampled
        limit = self.runtime.config.raw['safety']['max_feedback_age_s']
        if not 0 <= age <= limit:
            raise DualArmError(f'trajectory feedback age out of range: age_s={age:.6f}, '
                               f'limit_s={limit:.6f}, sampled={sampled:.6f}, observed={observed:.6f}')
        return q, pose, sampled, observed

    def _run(self):
        from .gamepad_continuous import checked
        r = self.runtime; connection = r.coordinator.connections['left']
        try:
            with open(self.folder/'states.jsonl', 'w', buffering=1) as stream:
                while not self.done.is_set():
                    began = time.monotonic()
                    q, sdk_pose, sampled, observed = self._read_feedback(connection)
                    grip = checked(self.api.get_gripper_position(), 'record gripper')
                    world = matrix_pose(r.config.arms['left'].world_from_base @ pose_matrix(r.models['left'].forward(q)))
                    stream.write(json.dumps(dict(t_s=observed-self.started, feedback_t_s=sampled-self.started,
                        phase=self.phase, joints_deg=q.tolist(), world_tcp_mm_deg=world.tolist(),
                        sdk_tcp_mm_deg=list(sdk_pose), gripper_pulse=float(grip)), allow_nan=False)+'\n')
                    self.samples += 1
                    self.done.wait(max(0., .1-(time.monotonic()-began)))
        except BaseException as exc:
            self.error = f'{type(exc).__name__}: {exc}'
            r.cancel.set(); r.stop()

    def check(self):
        if self.error:
            raise DualArmError(f'trajectory recording failed: {self.error}')

    def close(self):
        self.done.set()
        if self.thread:
            self.thread.join(timeout=3)
            if self.thread.is_alive():
                self.error = self.error or 'trajectory recorder did not stop'
        try:
            if self.api:
                self.api.disconnect()
        except Exception as exc:
            self.error = self.error or f'recorder disconnect: {exc}'
        finally:
            self.events.close()


def world_pose(runtime):
    snapshot = runtime.coordinator.snapshots()
    q = np.asarray(snapshot['left']['joints'])
    return matrix_pose(runtime.config.arms['left'].world_from_base @ pose_matrix(runtime.models['left'].forward(q)))


def execute_program(runtime, recorder, folder, stage, source):
    recorder.check(); recorder.stage(stage)
    directory = folder/stage; directory.mkdir()
    (directory/'motion.py').write_text(source)
    initial = {k: np.asarray(v['joints']) for k, v in runtime.coordinator.snapshots().items()}
    operations = runtime.plan(source, initial, preserve_command_boundaries=True)
    if stage == 'transfer_place':
        operations = staged_place_release(operations)
    elif stage == 'automatic_pick_place':
        # The first open prepares the grasp; only the placement open is partial.
        openings = [i for i, (kind, data) in enumerate(operations)
                    if kind == 'gripper' and data['action'] == 'open']
        if len(openings) != 2:
            raise DualArmError('automatic pickup requires preparation and placement opens')
        release = openings[-1]
        operations = operations[:release] + staged_place_release(operations[release:])
    recorder.check(); runtime.arrival(initial)
    result = runtime._execute_operations(operations, directory)
    save_json(directory/'result.json', result)
    if result['status'] != 'EXECUTED':
        raise DualArmError(f'{stage} failed: {result}')
    recorder.check()


def staged_place_release(operations):
    """Open slightly at contact; retain that opening throughout the retreat."""
    openings = [i for i, (kind, data) in enumerate(operations)
                if kind == 'gripper' and data['arm'] == 'left' and data['action'] == 'open']
    if (len(openings) != 1
            or any(kind == 'gripper' for kind, _ in operations[openings[0]+1:])
            or not any(kind == 'move' for kind, _ in operations[openings[0]+1:])):
        raise DualArmError('placement requires partial release followed by retreat only')
    result = list(operations)
    index = openings[0]
    result[index] = ('gripper', dict(operations[index][1], action='release_partial',
                                   opening_delta_pulse=180., settle_s=.5))
    return result


def manual_stage(runtime, recorder, folder, pad, args, start=None, target=None):
    from .gamepad_continuous import ContinuousArm6
    mapper, guard = Mapper(), InputGuard()
    backend = ContinuousArm6(args.config, folder/'manual', guard, args.speed_mm_s,
        args.yaw_speed_deg_s, args.gripper_rate, runtime=runtime)
    handed_off = False; neutral = False; previous = None; wait_release = False
    previous_status = None
    free_manual = getattr(args, 'manual_only', False) or getattr(args, 'single_point', False)
    recorder.stage('manual')
    print('手柄控制：按住 LB 操作；松开全部控件，再单独按 START '+
          ('结束记录并确认保存。' if free_manual else '交回自动搬运。')+'Ctrl+C 中止。', flush=True)
    try:
        while True:
            raw = pad.sample(); intent = mapper.update(*raw)
            recorder.check(); backend.check_health()
            if wait_release:
                guard.update(Intent())
                if not raw[3] and not any(raw[:3]):
                    wait_release = False; neutral = True
                time.sleep(.02)
                continue
            if intent.stop:
                # START only hands off after a neutral frame and with no LB,
                # sticks or gripper buttons held. Other combinations abort.
                if not neutral or raw[3] != {'START'} or any(raw[:3]):
                    raise KeyboardInterrupt('START pressed with active controls')
                guard.update(Intent())
                # Read live feedback without position-mode snapshot constraints.
                q = backend._feedback()[0]
                lifted = matrix_pose(runtime.config.arms['left'].world_from_base @ pose_matrix(runtime.models['left'].forward(q)))
                try:
                    if not free_manual:
                        transfer_program(lifted, start, target, clearance=args.clearance_mm,
                                         place_offset=args.place_offset_mm)
                except DualArmError as exc:
                    print(str(exc), flush=True); neutral = False; wait_release = True
                    time.sleep(.02); continue
                backend.handoff(); handed_off = True
                recorder.event('manual_recording_finished' if free_manual else 'manual_confirmed',
                               world_tcp_mm_deg=lifted.tolist())
                return
            guard.update(intent)
            neutral = not raw[3] and not any(raw[:3])
            current = asdict(intent)
            if current != previous:
                recorder.event('gamepad', intent=current); previous = current
            status = backend.control_status(control_status(mapper.ready, intent))
            if status != previous_status:
                print(status, flush=True); previous_status = status
            time.sleep(.02)
    finally:
        if not handed_off:
            guard.update(Intent(stop=True)); backend.close()


def assisted_pickup(runtime, recorder, folder, args, start, target, rpy):
    """Acquire a gamepad before approach; let the operator choose grasp depth."""
    from types import SimpleNamespace
    path=getattr(args,'device',None)
    if path is None:
        devices=list_gamepads()
        if len(devices)!=1:raise DualArmError('手柄抓取需要唯一手柄，或通过 --device 指定')
        path=devices[0]['path']
    pad=Gamepad(path)
    try:
        approach=np.r_[start+[0,0,args.approach_height_mm],rpy]
        execute_program(runtime,recorder,folder,'approach',approach_program(runtime,approach))
        fine_args=SimpleNamespace(**vars(args))
        fine_args.speed_mm_s=getattr(args,'grasp_adjust_speed_mm_s',5.)
        print('已到起点上方。手柄微调抓取：LB + Y/A 升降，X/B 开合；抓好并抬升后松开所有控件，再按 START 自动搬运。',flush=True)
        manual_stage(runtime,recorder,folder,pad,fine_args,start,target)
    finally:
        pad.close()


def run_iteration(runtime, pad, args, folder):
    folder.mkdir(parents=True, exist_ok=False)
    coordinator = getattr(runtime, 'coordinator', None)
    if coordinator is not None:
        # Connections persist across iterations, but their event log belongs
        # to the current iteration, including observation/preparation events.
        with coordinator.log_lock:
            directory = folder/'controller'
            directory.mkdir()
            coordinator.directory = directory
    recorder = TrajectoryRecorder(runtime, folder)
    manifest = dict(schema_version=1, status='IN_PROGRESS', arm='left', camera='A',
        collection_mode=('automatic_pick_place' if getattr(args, 'auto_pick_place', False) else
                         'manual_only' if getattr(args, 'manual_only', False) else
                         'single_point_manual' if getattr(args, 'single_point', False) else 'assisted'),
        started_at=datetime.now(timezone.utc).isoformat(), calibration_id=runtime.calibration_id,
        parameters={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    if getattr(args, 'random_reset', False) or getattr(args, 'auto_pick_place', False):
        manifest.update(episode_id=getattr(args, 'reset_episode', 0),
                        preceding_reset=getattr(args, 'preceding_reset', None))
    save_json(folder/'traj.json', manifest)
    try:
        # The initial preparation is outside the demonstrated trajectory; each
        # successful iteration already ended at the same perception pose.
        observation = runtime.observe(folder/'observation')
        recorder.start()
        observation.validate_for(runtime.config, live=True, check_age=False)
        if getattr(args, 'manual_only', False):
            manual_stage(runtime, recorder, folder, pad, args)
            save_json(folder/'final_state.json', dict(world_tcp_mm_deg=world_pose(runtime).tolist()))
            recorder.check()
            manifest['status'] = 'PENDING_CONFIRMATION'
        else:
            recorder.stage('select_points')
            single = getattr(args, 'single_point', False)
            rpy = args.approach_rpy_deg
            if rpy is None:
                rpy = matrix_pose(runtime.config.arms['left'].world_from_base @
                    pose_matrix(runtime.models['left'].forward(runtime.config.arms['left'].home_joints)))[3:]
            manual_grasp = False
            if getattr(args, 'auto_pick_place', False):
                from .grasp_angle import select_points_and_angle
                points, rpy, manual_grasp = select_points_and_angle(runtime, observation, rpy, folder/'grasp_angle')
            else:
                points = select_points(observation, runtime.config.limits['max_surface_spread_mm'], single=single)
            if manual_grasp == 'random_reset':
                from .random_fold_reset import run_reset
                manifest.update(collection_mode='random_reset',not_a_demonstration=True)
                save_json(folder/'traj.json',manifest)
                run_reset(runtime,folder/'random_reset',radius_mm=args.reset_radius_mm,
                          lift_height=args.reset_lift_height_mm,grasp_offset=args.grasp_offset_mm,
                          recorder=recorder, **reset_options(args))
            elif manual_grasp == 'codex_smooth':
                from .cloth_smoothing import smooth_cloth
                manifest.update(collection_mode='codex_smoothing',grasp_control='codex')
                save_json(folder/'traj.json',manifest)
                result=smooth_cloth(runtime,recorder,observation,folder/'smoothing',rpy,
                    max_steps=getattr(args,'smoothing_max_steps',3),grasp_offset=args.grasp_offset_mm)
                manifest.update(smoothing_status=result['status'],smoothing_actions=result['completed_actions'])
            else:
                save_json(folder/'points.json', points)
                start = np.asarray(points[0]['world_xyz_mm'])
                target = None if single else np.asarray(points[1]['world_xyz_mm'])
                if manual_grasp:
                    manifest.update(collection_mode='assisted',grasp_control='gamepad')
                    save_json(folder/'traj.json',manifest)
                    assisted_pickup(runtime,recorder,folder,args,start,target,rpy)
                elif getattr(args, 'auto_pick_place', False):
                    source = automatic_program(runtime, start, target, rpy,
                        approach_height=args.approach_height_mm, grasp_offset=args.grasp_offset_mm,
                        lift_height=args.lift_height_mm, clearance=args.clearance_mm,
                        place_offset=args.place_offset_mm)
                    execute_program(runtime, recorder, folder, 'automatic_pick_place', source)
                else:
                    approach = np.r_[start + [0, 0, args.approach_height_mm], rpy]
                    source = approach_program(runtime, approach)
                    execute_program(runtime, recorder, folder, 'approach', source)
                    manual_stage(runtime, recorder, folder, pad, args, start, target)
                if single:
                    save_json(folder/'final_state.json', dict(world_tcp_mm_deg=world_pose(runtime).tolist()))
                else:
                    if manual_grasp or not getattr(args, 'auto_pick_place', False):
                        lifted = world_pose(runtime)
                        source = transfer_program(lifted, start, target, clearance=args.clearance_mm,
                                                  place_offset=args.place_offset_mm)
                        execute_program(runtime, recorder, folder, 'transfer_place', source)
                    recorder.stage('return_perception')
                    runtime.open_at_home_on_return = True
                    try:
                        runtime.move_to_perception(folder/'return_perception')
                    finally:
                        runtime.open_at_home_on_return = False
                    if getattr(args, 'auto_pick_place', False):
                        recorder.stage('after_observation')
                        runtime.capture_current(folder/'after')
        recorder.check()
        manifest['status'] = 'RESET_COMPLETE' if manifest['collection_mode']=='random_reset' else 'PENDING_CONFIRMATION'
    except BaseException as exc:
        manifest.update(status='ABORTED' if isinstance(exc, KeyboardInterrupt) else 'FAILED',
                        error=f'{type(exc).__name__}: {exc}')
        runtime.cancel.set(); runtime.stop()
        raise
    finally:
        recorder.close()
        if recorder.error:
            manifest.update(status='FAILED', recording_error=recorder.error)
        manifest.update(samples=recorder.samples, elapsed_s=time.monotonic()-recorder.started,
                        finished_at=datetime.now(timezone.utc).isoformat())
        save_json(folder/'traj.json', manifest)
    recorder.check()


def confirm_save_iteration(folder, destination, coordinator=None):
    """Publish a recorded iteration only after explicit operator acceptance."""
    from .collection_review import confirm_images

    manifest = json.loads((folder/'traj.json').read_text())
    if manifest['status'] != 'PENDING_CONFIRMATION':
        raise ValueError('only a successful iteration can be confirmed')
    accepted = confirm_images(folder)
    manifest.update(status='COMPLETE' if accepted else 'REJECTED',
                    operator_accepted=bool(accepted),
                    reviewed_at=datetime.now(timezone.utc).isoformat())
    if accepted:
        if destination.exists():
            raise FileExistsError(destination)
        # Coordinate the rename with event writers so no worker can open the
        # removed pending path, even during shutdown before the next iteration.
        with coordinator.log_lock if coordinator is not None else nullcontext():
            folder.rename(destination)
            if coordinator is not None:
                coordinator.directory = destination/'controller'
        save_json(destination/'traj.json', manifest)
    else:
        save_json(folder/'traj.json', manifest)
    return accepted


def reset_options(args):
    return dict(height_jitter=getattr(args,'reset_height_jitter_mm',50.),
                yaw_range=getattr(args,'reset_yaw_range_deg',45.),
                drop_radius=getattr(args,'reset_drop_radius_mm',40.),
                max_retries=getattr(args,'reset_max_retries',1))


def main(argv=None):
    from .collection_graph_check import DEFAULT_GRAPH, DEFAULT_MEMORY
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--real', action='store_true', help='execute on Arm6')
    parser.add_argument('--config', type=Path, default=Path('config/dual_arm.local.json'))
    parser.add_argument('--output', type=Path, default=Path('data/demonstrations'))
    parser.add_argument('--graph', type=Path, default=DEFAULT_GRAPH,
                        help='graph used by the read-only state-check button')
    parser.add_argument('--graph-memory', type=Path, default=DEFAULT_MEMORY,
                        help='precomputed retrieval memory paired with --graph')
    parser.add_argument('--device')
    parser.add_argument('--random-reset', action='store_true',
                        help='offer an independent unfold-complete button for Codex-guided random drop')
    parser.add_argument('--reset-radius-mm', type=float, default=80.)
    parser.add_argument('--reset-lift-height-mm', type=float, default=300.)
    parser.add_argument('--reset-height-jitter-mm', type=float, default=50.)
    parser.add_argument('--reset-yaw-range-deg', type=float, default=45.)
    parser.add_argument('--reset-drop-radius-mm', type=float, default=40.)
    parser.add_argument('--reset-max-retries', type=int, default=1)
    parser.add_argument('--iterations', type=int, default=1)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--auto-pick-place', action='store_true',
                       help='select start and target, then automatically grasp and transfer without gamepad')
    parser.add_argument('--grasp-offset-mm', type=float, default=-15.,
                        help='automatic grasp TCP Z relative to clicked surface, default -15 mm')
    parser.add_argument('--lift-height-mm', type=float, default=30.,
                        help='automatic transport height above the higher clicked surface, default 30 mm')
    modes.add_argument('--manual-only', action='store_true',
                        help='record full manual motion; START finishes without automatic transfer or return')
    modes.add_argument('--single-point', action='store_true',
                       help='select one start point, approach above it, then record free manual motion')
    parser.add_argument('--approach-height-mm', type=float, default=50.)
    parser.add_argument('--clearance-mm', type=float, default=30.)
    parser.add_argument('--place-offset-mm', type=float, default=5.)
    parser.add_argument('--approach-rpy-deg', type=float, nargs=3)
    parser.add_argument('--speed-mm-s', type=float, default=20.)
    parser.add_argument('--smoothing-max-steps',type=int,default=3,
                        help='maximum Codex smoothing actions per selection, 1..6')
    parser.add_argument('--grasp-adjust-speed-mm-s', type=float, default=5.,
                        help='gamepad fine grasp linear speed, default 5 mm/s')
    parser.add_argument('--yaw-speed-deg-s', type=float, default=5.)
    parser.add_argument('--gripper-speed', '--gripper-rate', dest='gripper_rate', type=float,
                        default=2000., help='Arm6 gripper SDK speed, (0,5000]')
    args = parser.parse_args(argv)
    if not args.real:
        parser.error('实机采集需要显式指定 --real；--help 不连接硬件')
    if (not 1 <= args.smoothing_max_steps <= 6 or args.iterations < 1 or not 0 < args.approach_height_mm <= 200
            or not 0 < args.clearance_mm <= 200 or not 0 <= args.place_offset_mm <= 50
            or not 0 < args.grasp_adjust_speed_mm_s <= 30
            or not 0 < args.speed_mm_s <= 30 or not 0 < args.yaw_speed_deg_s <= 5
            or not 0 < args.gripper_rate <= 5000
            or (args.approach_rpy_deg is not None and not np.isfinite(args.approach_rpy_deg).all())):
        parser.error('invalid iteration count, heights, orientation or speeds')
    if args.auto_pick_place and not (-30 <= args.grasp_offset_mm <= 10
            and max(args.grasp_offset_mm, args.place_offset_mm) < args.lift_height_mm <= 200
            and args.approach_height_mm > args.grasp_offset_mm):
        parser.error('invalid automatic grasp offset or lift height')
    if ((args.random_reset and not args.auto_pick_place)
            or not 0 < args.reset_radius_mm <= 80 or not 50 < args.reset_lift_height_mm <= 300):
        parser.error('--random-reset requires --auto-pick-place, radius (0,80] and lift height (50,300] mm')
    from .random_fold_reset import validate_randomization
    try:validate_randomization(args.reset_lift_height_mm,**reset_options(args))
    except ValueError as exc:parser.error(str(exc))
    # Detect UI and gamepad problems before connecting or moving the robot.
    import tkinter as tk
    root = tk.Tk(); root.withdraw(); root.destroy()
    pad = None; runtime = None
    if not args.auto_pick_place:
        path = args.device
        if path is None:
            devices = list_gamepads()
            if len(devices) != 1:
                parser.error('需要唯一手柄，或使用 --device 指定')
            path = devices[0]['path']
        pad = Gamepad(path)
    try:
        from .gamepad_collect_runtime import CollectionRuntime

        runtime = CollectionRuntime(args.config)
        runtime.collection_graph = args.graph
        runtime.collection_graph_memory = args.graph_memory
        runtime.config.home_recover_stopped = True
        runtime.config.arms['left'].gripper.gripper_speed = args.gripper_rate
        session = args.output/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        args.reset_episode = 0
        args.preceding_reset = None
        for index in range(args.iterations):
            folder = session/'pending'/f'iter_{index:04d}'
            print(f'开始 iter {index}: {folder}', flush=True)
            run_iteration(runtime, pad, args, folder)
            if json.loads((folder/'traj.json').read_text()).get('collection_mode')=='random_reset':
                destination=session/'resets'/f'ui_iter_{index:04d}'
                destination.parent.mkdir(parents=True,exist_ok=True)
                with runtime.coordinator.log_lock:
                    folder.rename(destination)
                    runtime.coordinator.directory=destination/'controller'
                args.reset_episode+=1
                args.preceding_reset=str(destination/'random_reset')
                print(f'随机重置已完成，记录：{destination}（不保存为展开 traj）',flush=True)
                continue
            destination = session/f'traj_{index:04d}'
            if confirm_save_iteration(folder, destination, runtime.coordinator):
                print(f'traj 已确认保存：{destination}', flush=True)
            else:
                print(f'本次未保存为 traj，临时数据：{folder}', flush=True)
            if args.random_reset:
                from .random_fold_reset import reset_choice, run_reset
                choice = reset_choice()
                if choice == 'finish':break
                if choice == 'reset':
                    reset_folder=session/'resets'/f'after_iter_{index:04d}'
                    print(f'随机重置：{reset_folder}',flush=True)
                    run_reset(runtime,reset_folder,radius_mm=args.reset_radius_mm,
                              lift_height=args.reset_lift_height_mm,grasp_offset=args.grasp_offset_mm,
                              **reset_options(args))
                    args.reset_episode += 1
                    args.preceding_reset = str(reset_folder)
    finally:
        if pad is not None:
            pad.close()
        if runtime is not None:
            runtime.close()
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
