"""Camera A view tour; only the request worker runs concurrently with motion."""
from concurrent.futures import Future, TimeoutError
from pathlib import Path
import threading
import time

import numpy as np

from .dual_arm.geometry import DualArmError
from .experiments.one_shot_goal_planner_probe_v1 import read, save

RIGHT_VIEW = Path('data/robot/views/right_view.json')
VIEW_SPEED_DEG_S = 10.
VIEW_ACCEL_DEG_S2 = 20.


class ViewError(DualArmError):
    """Observation motion errors must never enter model-error recovery."""


def view_target(runtime, record):
    camera = runtime.config.raw['camera']
    arm = runtime.config.arms['left']
    if (runtime.arm_key != 'left' or camera['mount_arm'] != 'left'
            or record.get('name') != 'right_view' or record.get('mount_arm') != 'left'
            or record.get('camera_label') != 'A' or record.get('robot_ip') != arm.ip
            or record.get('axis') != arm.axis or record.get('control_box_sn') != arm.serial
            or str(record.get('camera_serial')) != str(camera['serial'])):
        raise ViewError('right view camera/robot identity mismatch')
    q = np.asarray(record['joint_angles_deg'], dtype=float)
    offset = np.asarray(record['tcp_offset_mm_deg'], dtype=float)
    if q.shape not in ((arm.axis,), (7,)) or not np.isfinite(q).all():
        raise ViewError('invalid right view joint vector')
    if (offset.shape != (6,) or not np.isfinite(offset).all()
            or np.max(np.abs(offset-arm.tcp_offset)) > .5):
        raise ViewError('right view TCP calibration mismatch')
    return q[:arm.axis].copy()


class ViewTour:
    def __init__(self, runtime, before, folder, record):
        self.runtime, self.folder = runtime, Path(folder)
        before.validate_for(runtime.config, live=True, check_age=False)
        self.initial = {k: np.asarray(q, dtype=float) for k, q in before.meta['joints_deg'].items()}
        owner, perception = runtime.perception_target()
        if (owner != 'left' or np.max(np.abs(self.initial[owner]-perception)) >
                runtime.config.limits['stationary_tolerance_deg']):
            raise ViewError('first observation is not at perception')
        self.side = {k: q.copy() for k, q in self.initial.items()}
        self.side[owner] = view_target(runtime, record)
        runtime.arrival(self.initial)
        # Plan both directions before moving. Preserve existing motion checks;
        # only reduce dynamics, and leave the manipulation limits untouched.
        limits = runtime.config.limits
        keys = ('joint_speed_deg_s', 'joint_accel_deg_s2')
        old = {k: limits[k] for k in keys}
        try:
            limits.update({k: min(old[k], cap) for k, cap in
                           zip(keys, (VIEW_SPEED_DEG_S, VIEW_ACCEL_DEG_S2))})
            self.outbound = self._plan('to_right_view', self.initial, self.side)
            self.inbound = self._plan('view_return_perception', self.side, self.initial)
        finally:
            limits.update(old)

    def _plan(self, name, start, end):
        folder = self.folder/name
        folder.mkdir()
        source = f'motion.move_joints(left={np.radians(end["left"]).tolist()})\n'
        (folder/'motion.py').write_text(source)
        operations = self.runtime._plan_preparation(source, start)
        return folder, start, end, operations

    def _move(self, prepared):
        folder, start, end, operations = prepared
        if self.runtime.cancel.is_set():
            raise KeyboardInterrupt('view tour cancelled')
        from .trajectory_preview import record_preview
        record_preview(self.runtime, (folder/'motion.py').read_text(), start, operations, None, folder)
        preview = getattr(self.runtime, 'preview', None)
        if preview is not None:
            preview.await_execution(self.runtime.cancel)
        self.runtime.arrival(start)
        result = self.runtime._execute_preparation(operations, folder)
        save(folder/'result.json', result)
        if result['status'] != 'EXECUTED':
            raise ViewError('view transition failed: '+str(result))
        self.runtime.arrival(end)

    def capture(self):
        self._move(self.outbound)
        if self.runtime.cancel.wait(.35):
            raise KeyboardInterrupt('view capture cancelled')
        self.runtime.arrival(self.side)
        return self.runtime.capture_current(self.folder/'right_view')

    def return_to_perception(self):
        self._move(self.inbound)


def capture_and_return(runtime, root, folder, before, timing):
    """Recovery's tool-using Agent stays synchronous; give it the same views."""
    started = time.monotonic()
    try:
        tour = ViewTour(runtime, before, folder, read(root/'right_view_pose.json'))
        tour.capture()
        timing['view_observation_s'] = time.monotonic()-started
        started = time.monotonic()
        tour.return_to_perception()
        timing['view_return_s'] = time.monotonic()-started
        save(folder/'view_planning.json', dict(status='RETURNED_BEFORE_RECOVERY',
            return_verified=True, coordinate_image='before/rgb.png',
            right_view_image='right_view/rgb.png', overlap=False))
    except Exception as exc:
        raise ViewError('recovery view tour failed: '+str(exc)) from exc


def plan_with_right_view(runtime, root, folder, before, plan_once, timing):
    """Return a proposal only after both request and verified return succeed.

    A daemon worker owns no runtime/camera and cannot dispatch an action. On a
    motion failure/interruption the caller can stop hardware immediately, even
    if HTTP is still waiting; any late model files are explicitly discarded.
    """
    evidence = dict(status='OBSERVING', model_result_usable=False,
                    coordinate_image='before/rgb.png', right_view_image='right_view/rgb.png')
    save(folder/'view_planning.json', evidence)
    tour_started = time.monotonic()
    try:
        tour = ViewTour(runtime, before, folder, read(root/'right_view_pose.json'))
        tour.capture()
    except BaseException as exc:
        evidence.update(status='DISCARDED', error=f'{type(exc).__name__}: {exc}')
        save(folder/'view_planning.json', evidence)
        if not isinstance(exc, Exception):
            raise
        raise ViewError('right view observation failed: '+str(exc)) from exc
    finally:
        timing['view_observation_s'] = time.monotonic()-tour_started

    future = Future()
    worker_started = threading.Event()
    stamps = {}
    def request():
        stamps['start'] = time.monotonic()
        worker_started.set()
        try:
            value = plan_once()
        except BaseException as exc:
            stamps['end'] = time.monotonic()
            future.set_exception(exc)
        else:
            stamps['end'] = time.monotonic()
            future.set_result(value)

    thread = threading.Thread(target=request, name='one-shot-model', daemon=True)
    thread.start()
    return_start = None
    try:
        worker_started.wait()
        evidence['status'] = 'MODEL_AND_RETURN'
        save(folder/'view_planning.json', evidence)
        return_start = time.monotonic()
        try:
            tour.return_to_perception()
        except Exception as exc:
            raise ViewError('perception return failed: '+str(exc)) from exc
        finally:
            return_end = time.monotonic()
            timing['view_return_s'] = return_end-return_start
        evidence['return_verified'] = True
        wait_start = time.monotonic()
        try:
            while True:
                if runtime.cancel.is_set():
                    raise KeyboardInterrupt('planning cancelled')
                try:
                    result = future.result(timeout=.1)
                    break
                except TimeoutError:
                    # A model-raised TimeoutError is a completed failure too.
                    if future.done():
                        result = future.result()
                        break
        finally:
            timing['model_wait_after_return_s'] = time.monotonic()-wait_start
        evidence.update(status='READY', model_result_usable=True)
        return result
    except BaseException as exc:
        evidence.update(status='DISCARDED', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if return_start is not None:
            timing['planner_return_overlap_s'] = max(0., min(
                stamps.get('end', return_end), return_end)-max(stamps['start'], return_start))
        evidence['model_finished_at_close'] = future.done()
        save(folder/'view_planning.json', evidence)
