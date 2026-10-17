"""Fixed-world-XY batch harness for grasp_acquire_baseline; no outcome inference."""
from __future__ import annotations

import hashlib
import csv
import json
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import grasp_acquire_baseline as base

OFFSETS = [0., -2., -4., -6.]
COMPARISON_OFFSETS = [0., -2.]
XY_TOLERANCE_MM = 2.


def now():
    return datetime.now(timezone.utc).isoformat()


def schedule(kind, seed, shakedown=None):
    if kind in ('compare6', 'compare9'):
        values = np.repeat(COMPARISON_OFFSETS if kind == 'compare6' else [0., -2., -4.], 3)
        np.random.default_rng(seed).shuffle(values)
        return values.tolist()
    if kind == 'shakedown':
        return OFFSETS.copy()
    if kind != 'balanced12':
        raise ValueError(f'Unknown batch schedule: {kind}')
    if shakedown is None:
        raise ValueError('balanced12 requires --shakedown with an operator-reviewed PASS')
    root = Path(shakedown)
    data = json.loads((root/'session.json').read_text())
    review = json.loads((root/'shakedown_review.json').read_text())
    validate_shakedown(root, data)
    digest = hashlib.sha256((root/'session.json').read_bytes()).hexdigest()
    if review.get('result') != 'PASS' or review.get('session_sha256') != digest:
        raise ValueError('shakedown has no matching manual PASS review')
    values = np.repeat(OFFSETS, 3)
    np.random.default_rng(seed).shuffle(values)
    return values.tolist()


def validate_shakedown(root, data):
    if data.get('schedule') != 'shakedown' or data.get('offsets_mm') != OFFSETS or data.get('status') != 'COMPLETED':
        raise ValueError('Requires a completed four-trial shakedown')
    if len(data.get('trials', [])) != 4:
        raise ValueError('Incomplete shakedown')
    for row in data['trials']:
        trial = json.loads((root/row['trial']/'trial.json').read_text())
        if trial.get('status') != 'COMPLETED' or not trial.get('returned_safely'):
            raise ValueError('Shakedown contains incomplete trial')


def review_shakedown(root, result):
    root = Path(root).resolve()
    data = json.loads((root/'session.json').read_text())
    validate_shakedown(root, data)
    if (root/'shakedown_review.json').exists():
        raise ValueError('Manual shakedown review already exists')
    base.save(root/'shakedown_review.json', dict(result=result, reviewed_at=now(),
        source='operator', session_sha256=hashlib.sha256((root/'session.json').read_bytes()).hexdigest(),
        meaning='Operator reviewed motion, reset and recording; not an automatic HELD/MISS judgement'))


class SessionCamera:
    """One camera-reading thread owns B; consumers receive distinct fresh frames.

    Recording never races capture() for the RealSense pipeline. Session video is
    constant-FPS; frame_times.jsonl contains actual frame wall times and stages.
    """
    def __init__(self, camera, folder, fps, cancel):
        self.camera = camera
        self.folder = Path(folder)
        self.fps = float(fps)
        self.cancel = cancel
        self.condition = threading.Condition()
        self.stop_event = threading.Event()
        self.started = False
        self.closed = False
        self.error = None
        self.latest = None
        self.sequence = 0
        self.consumed = 0
        self.stage = 'initialization'
        self.thread = None
        self.frames = 0

    @property
    def intrinsics(self):
        return self.camera.intrinsics

    def start(self):
        if self.started:
            return
        self.camera.start()
        self.started = True
        self.clock = time.monotonic()
        self.thread = threading.Thread(target=self._run, name='grasp-batch-camera', daemon=True)
        self.thread.start()

    def check(self):
        if self.error:
            raise RuntimeError('B camera/session recording failed: '+self.error)
        if self.cancel.is_set():
            raise RuntimeError('Batch cancelled')

    def _run(self):
        import cv2
        writer = None
        try:
            with (self.folder/'frame_times.jsonl').open('w') as times:
                while not self.stop_event.is_set():
                    rgb, depth = self.camera.read()
                    if self.stop_event.is_set():
                        break
                    rgb, depth = np.array(rgb, copy=True), np.array(depth, copy=True)
                    if writer is None:
                        h, w = rgb.shape[:2]
                        writer = cv2.VideoWriter(str(self.folder/'session.mp4'),
                            cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (w, h))
                        if not writer.isOpened():
                            raise RuntimeError('Cannot open session video writer')
                    writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                    elapsed = time.monotonic()-self.clock
                    times.write(json.dumps(dict(frame=self.frames, elapsed_s=elapsed,
                        unix_s=time.time(), stage=self.stage))+'\n')
                    times.flush()
                    self.frames += 1
                    with self.condition:
                        self.latest = (rgb, depth)
                        self.sequence += 1
                        self.condition.notify_all()
        except Exception as exc:
            if not self.stop_event.is_set():
                self.error = f'{type(exc).__name__}: {exc}'
                self.cancel.set()
        finally:
            if writer is not None:
                writer.release()
            with self.condition:
                self.condition.notify_all()

    def read(self):
        if not self.started:
            self.start()
        with self.condition:
            # Wait for a frame acquired AFTER this read request, not an old held frame.
            previous = max(self.consumed, self.sequence)
            ready = self.condition.wait_for(lambda: self.sequence > previous or self.error
                or self.stop_event.is_set() or self.cancel.is_set(), timeout=6.)
            self.check()
            if not ready or self.stop_event.is_set():
                raise RuntimeError('No fresh B RGB-D frame')
            self.consumed = self.sequence
            return tuple(a.copy() for a in self.latest)

    def stop(self):
        if self.closed:
            return
        self.stop_event.set()
        with self.condition:
            self.condition.notify_all()
        if self.thread is not None:
            self.thread.join(timeout=6.)
        self.camera.stop()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=6.)
            if self.thread.is_alive():
                raise RuntimeError('B camera reader did not terminate')
        self.closed = True
        base.save(self.folder/'recording.json', dict(camera='B', path='session.mp4', frames=self.frames,
            fps=self.fps, status='FAILED' if self.error else 'RECORDED', error=self.error,
            timing='constant FPS video; actual timestamps in frame_times.jsonl'))


def fixed_xy_pixel(observation, xy, tolerance_mm=XY_TOLERANCE_MM):
    """Locate fixed world XY in the freshly measured, already-corrected B cloud."""
    cloud = observation.xyz
    distances = np.linalg.norm(cloud[:, :, :2]-np.asarray(xy), axis=2)
    valid = np.isfinite(cloud).all(axis=2)
    valid[:2] = False; valid[-2:] = False; valid[:, :2] = False; valid[:, -2:] = False
    distances[~valid] = np.inf
    index = np.unravel_index(np.argmin(distances), distances.shape)
    error = float(distances[index])
    if not np.isfinite(error) or error > tolerance_mm:
        raise ValueError(f'Fixed world XY has no measured B point within {tolerance_mm} mm')
    return [int(index[1]), int(index[0])], error


def target_xyz(measurement, xy, offset):
    if measurement['sample_error']:
        raise ValueError('Existing depth sampler rejected fixed XY: '+measurement['sample_error'])
    sampled = np.asarray(measurement['surface_xyz_mm'], dtype=float)
    delta = float(np.linalg.norm(sampled[:2]-xy))
    measurement['sampled_xy_deviation_mm'] = delta
    if not np.isfinite(sampled).all() or delta > XY_TOLERANCE_MM:
        raise ValueError('Depth patch does not represent the fixed world XY')
    return [float(xy[0]), float(xy[1]), float(sampled[2]+offset)]


def prepare_batch_source(runtime, before, measurement, xy, offset, yaw_deg, folder):
    xyz = target_xyz(measurement, xy, offset)
    source, grasp, lifted = base.prepare_source(runtime, before, xyz, yaw_deg, folder)
    # Never silently stack an existing grasp lowering policy on this experiment.
    if not np.allclose(np.asarray(grasp[:3])*1000, xyz, atol=1e-6, rtol=0):
        raise ValueError('Existing height policy changed requested XY/Z; batch stopped (no policy bypass)')
    measurement.update(fixed_world_xy_mm=list(xy), z_offset_mm=offset,
        requested_command_z_mm=xyz[2], command_grasp_z_mm=grasp[2]*1000,
        command_grasp_world_m_rad=grasp, command_lift_world_m_rad=lifted,
        existing_grasp_offset_mm=runtime.grasp_offset_mm)
    return source, grasp


def confirm_before_release(runtime, folder, report):
    runtime.camera.stage = f'{folder.name}/waiting_release_confirmation'
    record = dict(status='WAITING', started_at=now(), checkpoint='after_test_lift_before_replace')
    report['release_confirmation'] = record
    base.save(folder/'trial.json', report)
    started = time.monotonic()
    try:
        while True:
            runtime.camera.check()
            answer = input('已试抬 25 mm 并保持拍照。请目视判断抓取结果，不要触碰机械臂或衣物。'
                '按 Enter 放回、松爪并撤离；输入 q 停止：').strip().lower()
            runtime.camera.check()
            if answer == 'q':
                raise KeyboardInterrupt('Operator stopped before release; no automatic put-back')
            if answer == '':
                record.update(status='CONFIRMED', confirmed_at=now(), confirmation='ENTER_RELEASE')
                return
            print('按 Enter 允许放回松爪，或输入 q 停止。', flush=True)
    except BaseException as exc:
        record.update(status='CANCELLED', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        record['elapsed_s'] = time.monotonic()-started
        base.save(folder/'trial.json', report)


def dispatch_batch(runtime, phases, folder, report, photo_fn=base.hold_photo):
    """Fail closed on ANY phase or checkpoint failure. No retry or outcome label."""
    for name, operations in phases:
        runtime.camera.check()
        runtime.camera.stage = f'{folder.name}/{name}'
        directory = folder/name
        directory.mkdir()
        started = time.monotonic()
        if name == 'return_home':
            with base.home_speed(runtime):
                result = runtime._execute_operations(operations, directory)
        else:
            result = runtime._execute_operations(operations, directory)
        report['phases'].append(dict(phase=name, result=result, elapsed_s=time.monotonic()-started))
        base.save(folder/'trial.json', report)
        if result.get('status') != 'EXECUTED':
            raise RuntimeError(f'{name}: {result.get("error", result.get("status"))}')
        runtime.camera.check()
        report['phases'][-1]['snapshot'] = runtime.coordinator.snapshots()
        if name in ('close', 'lift'):
            photo_fn(runtime, folder/f'{name}_observation', 2. if name == 'lift' else 0.)
            shutil.copyfile(folder/f'{name}_observation/rgb.png', folder/f'{name}.png')
        if name == 'lift':
            confirm_before_release(runtime, folder, report)
        if name == 'return_home':
            report['returned_safely'] = True
        base.save(folder/'trial.json', report)


def draw_pa(observation, pa, folder):
    shutil.copyfile(observation.image, folder/'before.png')
    image = Image.open(observation.image).convert('RGB')
    draw = ImageDraw.Draw(image)
    x, y = pa
    draw.line((x-10,y,x+10,y), fill='red', width=2)
    draw.line((x,y-10,x,y+10), fill='red', width=2)
    image.save(folder/'before_pa.png')


def one_trial(runtime, args, session, xy, offset, folder, report):
    runtime.camera.stage = f'{folder.name}/before'
    before = runtime.observe(folder/'before')
    np.save(folder/'before/intrinsics.npy', runtime.camera.intrinsics, allow_pickle=False)
    shutil.copyfile(before.image, folder/'before.png')
    pa, pixel_error = fixed_xy_pixel(before, xy)
    report['Pa'] = pa
    draw_pa(before, pa, folder)
    measurement = base.measure_pa(before, pa, runtime.config.limits['max_surface_spread_mm'])
    measurement.update(fixed_world_xy_mm=list(xy), z_offset_mm=offset,
        nearest_pixel_xy_deviation_mm=pixel_error, xy_sampling_tolerance_mm=XY_TOLERANCE_MM)
    try:
        measurement['A'] = base.capture_a(runtime, args.config, before, pa, folder/'camera_A')
    except Exception as exc:
        measurement['A'].update(reason=f'{type(exc).__name__}: {exc}', used_for_command=False)
    report['measurement'] = measurement
    base.save(folder/'measurement.json', measurement)
    source, grasp = prepare_batch_source(runtime, before, measurement, xy, offset, args.yaw_deg, folder)
    base.save(folder/'measurement.json', measurement)
    before.validate_for(runtime.config, live=True, check_age=False)
    phases = base.plan_trial(runtime, before, source, grasp)
    base.save(folder/'certified_phases.json', [dict(phase=name, operations=[dict(kind=k,
        data=d.to_dict() if k=='move' else d) for k,d in ops]) for name,ops in phases])
    runtime.arrival({k:np.asarray(v) for k,v in before.meta['joints_deg'].items()})
    report['status'] = 'EXECUTING'
    base.save(folder/'trial.json', report)
    dispatch_batch(runtime, phases, folder, report)
    # Return to the original B perception pose for comparable after/reset imagery.
    # Native observe certifies/executes this move; it is never called while holding.
    runtime.camera.stage = f'{folder.name}/reset_wait'
    if runtime.cancel.wait(2.):
        raise RuntimeError('Batch cancelled during reset wait')
    runtime.camera.check()
    runtime.camera.stage = f'{folder.name}/after_reset'
    after = runtime.observe(folder/'after_reset_observation')
    shutil.copyfile(after.image, folder/'after_reset.png')
    report['status'] = 'COMPLETED'


def wait_for_manual_reset(runtime, folder, session, report, next_offset):
    """No motion/capture preparation until the operator confirms hands clear."""
    if report.get('status') != 'COMPLETED' or not report.get('returned_safely'):
        raise RuntimeError('Cannot continue after an incomplete or unsafe trial')
    runtime.camera.check()
    runtime.camera.stage = f'{report["trial"]}/waiting_manual_reset'
    session['status'] = 'WAITING_FOR_MANUAL_RESET'
    record = dict(after_trial=report['trial'], next_offset_mm=next_offset,
        started_at=now(), status='WAITING', confirmation=None)
    session.setdefault('manual_resets', []).append(record)
    base.save(folder/'session.json', session)
    started = time.monotonic()
    try:
        while True:
            answer = input(f'本轮已安全放回并撤离。请手动复原衣服，移开手后按 Enter 继续 '
                f'(下一轮 offset={next_offset:+g} mm)；输入 q 结束：').strip().lower()
            runtime.camera.check()
            if answer == 'q':
                raise KeyboardInterrupt('Operator ended batch during manual reset')
            if answer == '':
                record.update(status='CONFIRMED', confirmation='ENTER_HANDS_CLEAR', confirmed_at=now())
                session['status'] = 'RUNNING'
                break
            print('复原并移开手后按 Enter；或输入 q 结束。', flush=True)
    except BaseException as exc:
        record.update(status='CANCELLED', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        record['elapsed_s'] = time.monotonic()-started
        base.save(folder/'session.json', session)


def run_trials(runtime, args, folder, session, trial_fn=one_trial):
    """Allocate once per offset, persist failed attempt and propagate immediately."""
    for index, offset in enumerate(session['offsets_mm']):
        runtime.camera.check()
        trial = base.allocate_trial(folder)
        report = dict(protocol='grasp-acquire-batch-v1', trial=trial.name, index=index,
            status='STARTED', started_at=now(), z_offset_mm=offset,
            fixed_world_xy_mm=session['fixed_world_xy_mm'], phases=[],
            operator_label=None, automatic_retry=False, folding=False,
            lift_mm=25., hold_s=2., reset_wait_s=2., returned_safely=False)
        session['trials'].append(dict(trial=trial.name, z_offset_mm=offset, status='STARTED'))
        base.save(folder/'session.json', session)
        base.save(trial/'trial.json', report)
        start = time.monotonic()
        print(f'Batch {index+1}/{len(session["offsets_mm"])}: {trial.name}, offset={offset:+g} mm', flush=True)
        try:
            trial_fn(runtime, args, session, session['fixed_world_xy_mm'], offset, trial, report)
        except BaseException as exc:
            report.update(status='INTERRUPTED' if isinstance(exc, KeyboardInterrupt) else 'FAILED',
                error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            report.update(elapsed_s=time.monotonic()-start, finished_at=now(),
                gripper_feedback_files=[str(p.relative_to(trial)) for p in trial.glob('*/gripper_*.json')])
            if 'measurement' in report:
                base.save(trial/'measurement.json', report['measurement'])
            base.save(trial/'trial.json', report)
            session['trials'][-1]['status'] = report['status']
            base.save(folder/'session.json', session)
        if index+1 < len(session['offsets_mm']):
            wait_for_manual_reset(runtime, folder, session, report, session['offsets_mm'][index+1])


def make_review(folder, session):
    """Offline checkpoint slideshow; not a substitute for continuous session video."""
    import cv2
    writer = cv2.VideoWriter(str(folder/'review.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 10., (960,600))
    if not writer.isOpened():
        raise RuntimeError('Cannot open review video writer')
    try:
        for row in session['trials']:
            for checkpoint in ('before', 'close', 'lift', 'after_reset'):
                path = folder/row['trial']/f'{checkpoint}.png'
                canvas = Image.new('RGB', (960,600), 'black')
                if path.exists():
                    with Image.open(path) as source:
                        frame = source.convert('RGB');frame.thumbnail((960,540))
                        canvas.paste(frame, ((960-frame.width)//2,60))
                title = f'{row["trial"]}  offset {row["z_offset_mm"]:+g} mm  {checkpoint}  {row["status"]}'
                ImageDraw.Draw(canvas).text((15,15), title, fill='white')
                if not path.exists():
                    ImageDraw.Draw(canvas).text((15,40), 'NOT CAPTURED', fill='red')
                frame = cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)
                for _ in range(20):
                    writer.write(frame)
    finally:
        writer.release()


def label_worksheet(folder, session):
    with (folder/'labels_to_fill.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['trial', 'z_offset_mm', 'status', 'label_HELD_MISS_UNCERTAIN',
            'before', 'close', 'lift', 'after_reset'])
        for row in session['trials']:
            writer.writerow([row['trial'], row['z_offset_mm'], row['status'], '',
                *[f'{row["trial"]}/{stage}.png' for stage in ('before', 'close', 'lift', 'after_reset')]])


def run(args):
    from cloth_agent.codex_arm7_runtime import Arm7Runtime
    from cloth_agent.dual_arm.observation import create_camera
    offsets = schedule(args.schedule, args.random_seed, args.shakedown)
    folder = Path(args.output).resolve()/datetime.now(timezone.utc).strftime('batch_%Y%m%dT%H%M%S_%fZ')
    folder.mkdir(parents=True, exist_ok=False)
    session = dict(protocol='grasp-acquire-batch-v1', status='STARTED', started_at=now(),
        schedule=args.schedule, offsets_mm=offsets, random_seed=args.random_seed,
        shakedown_source=str(args.shakedown) if args.shakedown else None,
        automatic_retry=False, online_outcome_judgement=False,
        reset_mode='OPERATOR_BETWEEN_TRIALS', release_confirmation_required=True,
        manual_resets=[], trials=[])
    base.save(folder/'session.json', session)
    print(f'Session: {folder}', flush=True)
    runtime = None
    start = time.monotonic()
    try:
        runtime = Arm7Runtime(args.config)
        runtime.auto_grasp_debug = False;runtime.auto_grasp_branch = None
        session['speed'] = base.low_speed(runtime)
        base.save(folder/'runtime_config.json', runtime.config.raw)
        runtime.camera = SessionCamera(create_camera(runtime.config), folder,
            runtime.config.raw['camera']['fps'], runtime.cancel)
        runtime.camera.start()
        runtime.camera.read()  # Verify camera/encoder before the initial perception move.
        before = runtime.observe(folder/'selection_observation')
        pa = base.select_pa(before.image)
        measurement = base.measure_pa(before, pa, runtime.config.limits['max_surface_spread_mm'])
        base.save(folder/'selection_measurement.json', measurement)
        if measurement['sample_error']:
            raise ValueError('Initial Pa rejected: '+measurement['sample_error'])
        xy = measurement['surface_xyz_mm'][:2]
        session.update(initial_Pa=pa, fixed_world_xy_mm=xy)
        base.save(folder/'session.json', session)
        print(f'固定 world XY = {xy} mm；Z offsets = {offsets} mm。每轮试抬 25 mm、保持 2 秒、放回。', flush=True)
        print('每轮试抬后等你按 Enter 才放回松爪；撤离停止后，再手动复原并按 Enter 开始下一轮。', flush=True)
        answer = input('确认序列安全，手已离开工作区，允许开始第一轮？输入 YES 开始：').strip().upper()
        if answer != 'YES':
            raise KeyboardInterrupt('Operator did not confirm batch')
        session['safety_confirmation'] = dict(source='operator', confirmed_at=now(), response='YES')
        session['status'] = 'RUNNING'
        base.save(folder/'session.json', session)
        run_trials(runtime, args, folder, session)
        runtime.camera.check()
        session['status'] = 'COMPLETED'
    except BaseException as exc:
        session.update(status='INTERRUPTED' if isinstance(exc, KeyboardInterrupt) else 'FAILED',
            error=f'{type(exc).__name__}: {exc}')
        if runtime is not None:
            session['stop_results'] = runtime.stop()
    finally:
        if runtime is not None:
            camera = getattr(runtime, 'camera', None)
            try:
                runtime.close()
            except Exception as exc:
                session.update(status='FAILED', cleanup_error=f'{type(exc).__name__}: {exc}')
            if getattr(camera, 'error', None):
                session.update(status='FAILED', recording_error=camera.error)
        session.update(elapsed_s=time.monotonic()-start, finished_at=now())
        try:
            if session['trials']:
                make_review(folder, session)
            label_worksheet(folder, session)
        except Exception as exc:
            session.update(status='FAILED', review_error=f'{type(exc).__name__}: {exc}')
        base.save(folder/'session.json', session)
    print(f'{session["status"]}: {folder}', flush=True)
    for key in ('error', 'cleanup_error', 'recording_error', 'review_error'):
        if session.get(key):
            print('停止原因：'+session[key], flush=True)
    print('不在线判定抓取结果。请结束后查看 session.mp4 / review.mp4，再用 label 子命令统一补录。', flush=True)
    return 0 if session['status']=='COMPLETED' else 1
