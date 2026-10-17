"""Continuous, expiring joint velocity commands with independently checked bounds.

No controller motion is issued by construction. The input thread, velocity
thread, collision validator and gripper worker never wait for one another's
motion completion. Only an explicitly held, fresh input can move anything.
"""
from dataclasses import dataclass
from collections import deque
import json
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .gamepad_arm7 import Arm7Jogger
from .gamepad_teleop import Intent
from .dual_arm.geometry import DualArmError, pose_error, pose_matrix


PERIOD = .02
VELOCITY_DURATION = .15  # Firmware returns velocity to zero if publishing stops.
BOX_RADIUS_DEG = 2.
BOX_RADII_DEG = (2., 1., .5, .25, .125)


def checked(result, label):
    code = result[0] if isinstance(result, (tuple, list)) else result
    if code != 0:
        raise DualArmError(f'{label}: SDK code {code}')
    return result[1] if isinstance(result, (tuple, list)) else result


def world_twist(intent, speed_mm_s, yaw_deg_s):
    if not intent.enabled or intent.stop:
        return np.zeros(6)
    linear = np.array([intent.forward, -intent.right, intent.up], dtype=float)
    linear *= speed_mm_s / max(1., np.linalg.norm(linear))
    return np.r_[linear, 0., 0., intent.yaw * yaw_deg_s]


def velocity_ik(model, joints, world_from_base, twist, max_speed):
    """Damped differential IK; mm/degree Jacobian, no discrete pose search."""
    joints = np.asarray(joints, dtype=float)
    if not np.any(twist):
        return np.zeros_like(joints)
    base = world_from_base @ pose_matrix(model.forward(joints))
    jac = np.empty((6, len(joints)))
    epsilon = .001
    for index in range(len(joints)):
        q = joints.copy(); q[index] += epsilon
        frame = world_from_base @ pose_matrix(model.forward(q))
        jac[:3, index] = (frame[:3, 3] - base[:3, 3]) / epsilon
        jac[3:, index] = np.degrees(Rotation.from_matrix(
            frame[:3, :3] @ base[:3, :3].T).as_rotvec()) / epsilon
    weights = np.array([1., 1., 1., 2., 2., 2.])
    a, b = weights[:, None] * jac, weights * twist
    velocity = a.T @ np.linalg.solve(a @ a.T + .05**2 * np.eye(6), b)
    if not np.isfinite(velocity).all():
        raise DualArmError('non-finite differential IK')
    if np.linalg.norm(a @ velocity - b) > max(.1, .15 * np.linalg.norm(b)):
        raise DualArmError('requested TCP velocity is not achievable near this singularity')
    # Scale every joint together to preserve the requested TCP direction.
    velocity *= min(1., max_speed / max(float(np.max(np.abs(velocity))), 1e-12))
    return velocity


def slew(previous, target, dt, acceleration):
    if not 0 < dt <= .1:
        raise DualArmError('velocity update deadline exceeded')
    delta = target - previous
    factor = min(1., acceleration * dt / max(float(np.max(np.abs(delta))), 1e-12))
    return previous + factor * delta


@dataclass(frozen=True)
class CertifiedBox:
    lower: np.ndarray
    upper: np.ndarray
    checked_at: float
    speed_limit: float = float('inf')

    def permits(self, joints, sampled_at, now, max_speed, max_age, latency):
        age = now - sampled_at
        if not 0 <= age <= max_age or not 0 <= now - self.checked_at <= .5:
            return False
        # A full direction-independent reachable interval until the command
        # expires, including delayed feedback and command delivery. Collision
        # certification additionally pads this box by measured stop excursions.
        reach = max_speed * (age + VELOCITY_DURATION + latency)
        return bool(np.all(joints - reach >= self.lower) and np.all(joints + reach <= self.upper))


def certify_adaptive_box(q, static, config, models, max_speed, max_age, latency, certify):
    """Use the pipeline proof and workspace, reducing the candidate on failure."""
    for radius in BOX_RADII_DEG:
        low = {'left': static.copy(), 'right': q - radius}
        high = {'left': static.copy(), 'right': q + radius}
        try:
            certify(low, high, config, models)
        except DualArmError:
            if radius == BOX_RADII_DEG[-1]:
                raise
            continue
        # Reserve room for validator refresh as well as a whole command lease.
        speed = min(max_speed, radius / (max_age + VELOCITY_DURATION + latency + .15))
        return CertifiedBox(low['right'], high['right'], time.monotonic(), speed)


class HeldGripper:
    """Bounded lookahead; no full-open/full-close request on a button edge."""
    def __init__(self, closed, opened, rate):
        if not 0 <= closed < opened <= 850 or not 0 < rate <= 200:
            raise ValueError('invalid gripper bounds/rate')
        self.closed, self.opened, self.rate = closed, opened, rate
        self.active = False

    def target(self, position, direction):
        if not np.isfinite(position) or not self.closed - 15 <= position <= self.opened + 15:
            raise DualArmError('gripper position is outside configured range')
        position = float(np.clip(position, self.closed, self.opened))
        if direction is None:
            if not self.active:
                return None
            self.active = False
            return position  # Hold measured opening on release; never disable.
        if direction not in ('open', 'close'):
            raise ValueError('invalid held gripper direction')
        self.active = True
        distance = self.rate * .12
        return float(np.clip(position + (distance if direction == 'open' else -distance),
                             self.closed, self.opened))


class ContinuousArm7(Arm7Jogger):
    def _watch(self):
        # The legacy one-step watchdog ends a session on LB release. Continuous
        # control instead publishes zero and permits a fresh LB press to resume.
        pass

    def __init__(self, config, output, guard, speed_mm_s=10., yaw_deg_s=5., gripper_rate=100., *, start_workers=True):
        if not (0 < speed_mm_s <= 15 and 0 < yaw_deg_s <= 5 and 0 < gripper_rate <= 200):
            raise ValueError('invalid continuous teleoperation rates')
        self.workers = []
        self.gripper_api = None
        self.gripper = None
        self.mode_entered = False
        self.transitioning = threading.Event()
        self.activation_lock = threading.Lock()
        self.failure = None
        self.failure_lock = threading.Lock()
        self.zone_lock = threading.Lock()
        self.zone = None
        self.velocity_history = deque()
        self.control_heartbeat = time.monotonic()
        self.gripper_heartbeat = time.monotonic()
        self.velocity_active = False
        self.status = None
        self.bounds_available = False
        super().__init__(config, output, guard)
        try:
            from .dual_arm.kinematics import ArmModel
            from xarm.wrapper import XArmAPI
            self.connection = self.runtime.coordinator.connections['right']
            self.arm = self.connection.arm
            self.fast = self.connection._report_arm
            if tuple(self.arm.version_number) < (1, 8, 0):
                raise DualArmError('continuous control requires firmware with expiring velocity commands (>=1.8.0)')
            self.model = ArmModel(self.runtime.config.arms['right'])
            self.speed_mm_s = min(speed_mm_s, self.runtime.config.limits['cartesian_speed_mm_s'])
            self.yaw_deg_s = yaw_deg_s
            self.max_speed = min(self.runtime.config.limits['joint_speed_deg_s'],
                self.runtime.config.raw['safety']['arms']['right']['max_joint_speed_deg_s'])
            self.acceleration = self.runtime.config.limits['joint_accel_deg_s2']
            self.max_age = self.runtime.config.raw['safety']['max_feedback_age_s']
            self.latency = self.runtime.config.raw['safety']['stop_command_latency_s']
            self.static = np.asarray(self.runtime.config.static_arm_obstacles['left']['joint_angles_deg'])
            self.gripper_api = XArmAPI(self.runtime.config.arms['right'].ip, is_radian=False)
            grip = self.runtime.config.arms['right'].gripper
            self.gripper = HeldGripper(grip.gripper_close, grip.gripper_open, gripper_rate)
            (self.output / 'continuous_config.json').write_text(json.dumps(dict(
                command_period_s=PERIOD, velocity_duration_s=VELOCITY_DURATION,
                speed_mm_s=self.speed_mm_s, yaw_deg_s=yaw_deg_s,
                joint_speed_deg_s=self.max_speed, gripper_rate=gripper_rate,
                box_radius_deg=BOX_RADIUS_DEG, adaptive_box_radii_deg=BOX_RADII_DEG,
                commissioning_verified=False), indent=2))
            self.control_heartbeat = self.gripper_heartbeat = time.monotonic()
            jobs = [('bounds', self._validate_loop), ('velocity', self._control_loop),
                    ('gripper', self._gripper_loop), ('watchdog', self._watch_loop)]
            for name, target in jobs if start_workers else []:
                thread = threading.Thread(target=self._worker, args=(name, target), name=f'gamepad-{name}', daemon=True)
                self.workers.append(thread)
                thread.start()
        except BaseException:
            self.close()
            raise

    def _worker(self, name, target):
        try:
            target()
        except BaseException as exc:
            if not self.closed.is_set():
                self._fail(f'{name}: {type(exc).__name__}: {exc}')

    def _fail(self, reason):
        with self.failure_lock:
            if self.failure is not None:
                return
            self.failure = reason
        print('连续遥操停止：' + reason, flush=True)
        self.runtime.cancel.set()
        # Independent transport: a blocked gripper/velocity request must not
        # prevent attempting the arm STOP. Firmware expiry remains a fallback.
        try:
            with self.activation_lock:
                if self.mode_entered:
                    checked(self.fast.set_state(4), 'emergency stop')
        except Exception as exc:
            print(f'停止请求未确认：{exc}', flush=True)
        self.runtime.coordinator.event('teleop_failure', reason=reason)

    def check_health(self):
        if self.failure:
            raise DualArmError(self.failure)

    def control_status(self, input_status):
        if self.failure:
            return '连续遥操停止：' + self.failure
        if self.status:
            return self.status
        if not self.bounds_available:
            return '运动暂停：等待有效的运动范围验证（当前不发送运动指令）'
        return input_status

    def _input(self):
        intent, timestamp, generation = self.guard.snapshot()
        if time.monotonic() - timestamp > .25:
            if timestamp:
                raise DualArmError('gamepad input heartbeat expired')
            return Intent(), generation
        return intent, generation

    def _feedback(self):
        if not self.arm.connected or not self.fast.connected:
            raise DualArmError('Arm7 disconnected')
        # Fast packets contain state/joints/TCP, but error/warning fields come
        # from the normal report channel (not the real-time packet format).
        if self.arm.error_code or self.arm.warn_code or (self.fast.state not in (0, 1, 2)
                                                         and not self.transitioning.is_set()):
            raise DualArmError(f'Arm7 fault/state {self.fast.state}, {self.arm.error_code}, {self.arm.warn_code}')
        with self.connection._report_lock:
            report = self.connection._latest_report
            if report is None:
                raise DualArmError('no coherent feedback')
            joints, pose, sampled = report
            q, pose = np.asarray(joints[:7], dtype=float), np.asarray(pose, dtype=float)
        if not np.isfinite(q).all() or not 0 <= time.monotonic() - sampled <= self.max_age:
            raise DualArmError('stale or invalid Arm7 feedback')
        return q, pose, sampled

    def _validate_loop(self):
        from .dual_arm.sequential_home import certify_segment
        from .dual_arm.stop_sweep import search_proof_budget
        while not self.closed.is_set() and not self.runtime.cancel.is_set():
            q, _, _ = self._feedback()
            try:
                # Native box certification is a proof over ALL configurations
                # in this interval, plus stop excursions; not a path sample.
                with search_proof_budget(.15):
                    zone = certify_adaptive_box(q, self.static, self.runtime.config,
                        self.runtime.models, self.max_speed, self.max_age, self.latency,
                        certify_segment)
            except DualArmError as exc:
                zone = None
                message = f'等待可验证的运动范围：{str(exc)[:180]}'
                if self.status != message:
                    print(message, flush=True)
                    self.runtime.coordinator.event('teleop_bounds_blocked', reason=str(exc),
                                                   joints_deg=q.tolist())
                    self.status = message
            with self.zone_lock:
                self.zone = zone
            if zone is not None:
                if self.status is not None:
                    self.runtime.coordinator.event('teleop_bounds_restored')
                self.status = None
            self.closed.wait(.08)

    def _lease_speed(self, zone):
        """Keep previous commands in the reach bound until expiry and feedback.

        A newly smaller box cannot assume an older faster command has stopped.
        Zero commands are sent until that old lease has elapsed if necessary.
        """
        now = time.monotonic()
        horizon = VELOCITY_DURATION + self.latency + self.max_age
        with self.zone_lock:
            while self.velocity_history and now - self.velocity_history[0][0] > horizon:
                self.velocity_history.popleft()
            previous = max((v for _, v in self.velocity_history), default=0.)
        return max(min(self.max_speed, zone.speed_limit), previous)

    def _enter_mode(self):
        if self.mode_entered:
            return
        # Only from an already ready, stationary controller. No fault clearing
        # or motor enabling. set_state(0) completes the explicit mode switch.
        if checked(self.arm.get_state(), 'state') not in (0, 2):
            raise DualArmError('Arm7 must be ready before entering velocity mode')
        if list(checked(self.arm.get_err_warn_code(), 'errors')) != [0, 0]:
            raise DualArmError('cannot enter velocity mode with faults')
        if self.arm.get_is_moving() or checked(self.arm.get_cmdnum(), 'queue') != 0:
            raise DualArmError('controller has another motion/queued command')
        with self.activation_lock:
            self.transitioning.set()
            try:
                if self.runtime.cancel.is_set() or self.closed.is_set():
                    raise DualArmError('cancelled before velocity mode switch')
                self.mode_entered = True  # even a failed switch must be cleaned up
                checked(self.arm.set_mode(4), 'joint velocity mode')
                if self.runtime.cancel.is_set() or self.closed.is_set():
                    raise DualArmError('cancelled during velocity mode switch')
                checked(self.arm.set_state(0), 'activate velocity mode')
                checked(self.arm.vc_set_joint_velocity([0.]*7, is_radian=False, duration=VELOCITY_DURATION), 'initial zero velocity')
                deadline = time.monotonic() + .2
                while self.fast.mode != 4 or self.fast.state not in (0, 1, 2):
                    if time.monotonic() > deadline or self.runtime.cancel.is_set():
                        raise DualArmError('velocity mode transition not confirmed')
                    time.sleep(.005)
            finally:
                self.transitioning.clear()

    def _control_loop(self):
        previous = np.zeros(7)
        last = time.monotonic()
        logged = last
        while not self.closed.is_set() and not self.runtime.cancel.is_set():
            began = time.monotonic()
            intent, generation = self._input()
            q, pose, sampled = self._feedback()
            if self.mode_entered and self.fast.mode != 4:
                raise DualArmError('controller mode changed externally')
            distance, angle = pose_error(pose, self.model.forward(q))
            if distance > 2 or angle > 1:
                raise DualArmError('controller/model TCP mismatch')
            with self.zone_lock:
                zone = self.zone
            lease_speed = self._lease_speed(zone) if zone is not None else self.max_speed
            allowed = zone is not None and zone.permits(q, sampled, time.monotonic(),
                lease_speed, self.max_age, self.latency)
            self.bounds_available = allowed
            dt = min(began - last, .05)
            target = np.zeros(7)
            if allowed and intent.enabled and intent.moving and not intent.stop:
                twist = world_twist(intent, self.speed_mm_s, self.yaw_deg_s)
                target = velocity_ik(self.model, q, self.runtime.config.arms['right'].world_from_base,
                                     twist, min(self.max_speed, zone.speed_limit))
                target = slew(previous, target, max(.001, dt), self.acceleration)
                peak = float(np.max(np.abs(target)))
                target *= min(1., zone.speed_limit / max(peak, 1e-12))
                self._enter_mode()
            # A fresh input check after computation prevents an old direction
            # from being issued after LB was released/repressed meanwhile.
            live, token = self._input()
            if token != generation or not live.enabled or live.stop:
                target[:] = 0
            if intent.moving != live.moving or any(abs(getattr(intent, k)-getattr(live, k)) > .15
                    for k in ('forward', 'right', 'up', 'yaw')):
                target[:] = 0
            if self.mode_entered:
                if self.runtime.cancel.is_set() or self.closed.is_set():
                    break
                # Recheck validity at dispatch, including compute/mode latency.
                if zone is None or not zone.permits(q, sampled, time.monotonic(),
                        lease_speed, self.max_age, self.latency):
                    target[:] = 0
                checked(self.arm.vc_set_joint_velocity(target.tolist(), is_radian=False,
                            is_sync=True, duration=VELOCITY_DURATION), 'velocity command')
                if np.any(target):
                    with self.zone_lock:
                        self.velocity_history.append((time.monotonic(), float(np.max(np.abs(target)))))
            previous = target
            self.velocity_active = bool(np.any(target))
            self.control_heartbeat = time.monotonic()
            if self.control_heartbeat - logged >= 1:
                self.runtime.coordinator.event('teleop_velocity', velocity_deg_s=target.tolist(),
                    control_iteration_s=self.control_heartbeat-began, bounds_available=allowed,
                    zone_speed_limit_deg_s=zone.speed_limit if zone is not None else None)
                logged = self.control_heartbeat
            last = began
            self.closed.wait(max(0., PERIOD-(time.monotonic()-began)))

    def _gripper_loop(self):
        initialized = False
        next_update = 0.
        last_direction = None
        api = self.gripper_api
        while not self.closed.is_set():
            intent, generation = self._input()
            direction = intent.gripper if intent.enabled and not intent.stop and not self.runtime.cancel.is_set() else None
            if direction is not None:
                q, _, sampled = self._feedback()
                with self.zone_lock:
                    zone = self.zone
                if zone is None or not zone.permits(q, sampled, time.monotonic(), self._lease_speed(zone), self.max_age, self.latency):
                    direction = None
            now = time.monotonic()
            if direction is not None or self.gripper.active:
                if direction != last_direction or now >= next_update:
                    if not initialized and direction:
                        checked(api.set_gripper_mode(0), 'gripper mode')
                        checked(api.set_gripper_enable(True), 'gripper enable')
                        checked(api.set_gripper_speed(self.runtime.config.arms['right'].gripper.gripper_speed), 'gripper speed')
                        initialized = True
                    position = checked(api.get_gripper_position(), 'gripper position')
                    live, token = self._input()
                    if token != generation or not live.enabled or live.stop or self.runtime.cancel.is_set():
                        direction = None
                    elif direction is not None:
                        direction = live.gripper
                    target = self.gripper.target(position, direction)
                    if target is not None:
                        checked(api.set_gripper_position(target, wait=False, wait_motion=False,
                                                         auto_enable=False), 'held gripper target')
                    next_update = time.monotonic() + .06
                    self.runtime.coordinator.event('teleop_gripper', direction=direction,
                                                   position=position, target=target)
            last_direction = direction
            self.gripper_heartbeat = time.monotonic()
            self.closed.wait(.02)

    def _watch_loop(self):
        while not self.closed.wait(.02) and not self.runtime.cancel.is_set():
            now = time.monotonic()
            if self.mode_entered and not self.transitioning.is_set() and now - self.control_heartbeat > .25:
                raise DualArmError('velocity worker heartbeat expired')
            if self.gripper.active and now - self.gripper_heartbeat > .5:
                raise DualArmError('gripper worker heartbeat expired')
            intent, timestamp, _ = self.guard.snapshot()
            if timestamp and now - timestamp > .25:
                raise DualArmError('input reader heartbeat expired')

    def stop(self):
        # Main input has already published stop=True. Keep the gripper worker
        # alive until close so it can replace its short lookahead by a hold.
        if hasattr(self, 'runtime'):
            self.runtime.cancel.set()
        if self.mode_entered:
            try:
                with self.activation_lock:
                    checked(self.fast.set_state(4), 'stop Arm7')
            except Exception as exc:
                print(f'停止请求未确认：{exc}', flush=True)

    def close(self):
        self.stop()
        self.closed.set()
        for worker in self.workers:
            if worker is not threading.current_thread():
                worker.join(timeout=1)
        try:
            if self.gripper_api is not None:
                try:
                    if self.gripper is not None and self.gripper.active:
                        position = checked(self.gripper_api.get_gripper_position(), 'final gripper position')
                        checked(self.gripper_api.set_gripper_position(position, wait=False, wait_motion=False), 'final gripper hold')
                except Exception as exc:
                    print(f'夹爪最终保持请求未确认：{exc}', flush=True)
                finally:
                    self.gripper_api.disconnect()
                    self.gripper_api = None
            if self.mode_entered and self.arm.connected:
                checked(self.arm.set_mode(0), 'restore position mode')
        finally:
            super().close()
