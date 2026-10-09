"""Persistent paired connections, shared servo clock, cancellation and audit."""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np

from .geometry import DualArmError, finite, matrix_pose, pose_error, pose_matrix
from .model import require_hold
from .planning import Motion, validate_sample
from .safety import validate_sweep
from .single_arm_backend import XArmBackend


class XArmConnection(XArmBackend):
    """Use the copied feedback gate without the single-arm auto-enable/Home flow."""

    simulated = False

    def __init__(self, config, cancel, sdk_factory=None, *, recover_stopped=False):
        use_fast_reports = sdk_factory is None
        if sdk_factory is None:
            from xarm.wrapper import XArmAPI

            sdk_factory = XArmAPI
        self.config, self.cancel = config, cancel
        self.lock = threading.RLock()
        self.arm = sdk_factory(config.ip, is_radian=False)
        self.original_mode = None
        self.position_command_active = False
        self.recover_stopped = recover_stopped
        self._report_lock = threading.Lock()
        self._report_received = threading.Event()
        self._latest_report = None
        self._report_callback = None
        self._report_arm = self.arm
        try:
            # SDK cached TCP/mode fields may still contain defaults at connect.
            # Wait for two reports before validating them; no controller writes.
            if hasattr(self.arm, 'register_report_callback'):
                ready = threading.Event()
                count = [0]
                def received(report):
                    if 'joints' not in report or 'cartesian' not in report:
                        return
                    with self._report_lock:
                        self._latest_report = (list(report['joints']),
                                               list(report['cartesian']), time.monotonic())
                    self._report_received.set()
                    count[0] += 1
                    if count[0] >= 2:
                        ready.set()
                self._report_callback = received
                self.arm.register_report_callback(received)
                deadline = time.monotonic() + 2.0
                while not ready.wait(.02):
                    if cancel.is_set() or time.monotonic() >= deadline:
                        raise DualArmError('controller initial coherent reports unavailable')
            self._check("get_robot_sn", self.arm.get_robot_sn())
            stopped = recover_stopped and self.checked_value(self.arm.get_state(), 'state') == 4
            if (
                not self.arm.connected
                or self.arm.axis != config.axis
                or self.arm.control_box_sn != config.serial
            ):
                raise DualArmError(
                    f"{config.arm_id}: controller identity/axis mismatch"
                )
            self.original_mode = int(self.arm.mode)
            if self.original_mode != 0:
                raise DualArmError("start both arms in stationary position mode (0)")
            self.snapshot()
            actual_tcp = finite(self.arm.tcp_offset, (6,), "live TCP")
            if np.max(np.abs(actual_tcp - config.tcp_offset)) > 0.5:
                raise DualArmError(f"{config.arm_id}: TCP calibration changed")
            if not stopped and (
                not all(self.arm.motor_enable_states[: config.axis])
                or not all(self.arm.motor_brake_states[: config.axis])
                or len(self.arm.motor_enable_states) < config.axis
                or len(self.arm.motor_brake_states) < config.axis
            ):
                raise DualArmError(
                    "motors must already be enabled with brakes released"
                )
            if use_fast_reports:
                # Rich/normal reports on these controllers arrive at 5 Hz.
                # Retain rich metadata for identity/TCP/brakes, and read paired
                # joints/TCP from the independent real-time report connection.
                fast = sdk_factory(config.ip, is_radian=False, report_type='real')
                self.arm.release_report_callback(received)
                self._report_arm = fast
                ready.clear()
                count[0] = 0
                self._latest_report = None
                fast.register_report_callback(received)
                if not ready.wait(2.) or cancel.is_set():
                    raise DualArmError('real-time controller reports unavailable')
        except BaseException:
            self.disconnect()
            raise

    def checked_value(self, result, name):
        self._check(name, result)
        return result[1]

    def snapshot(self):
        with self.lock:
            sampled_started = time.monotonic()
            report_received = getattr(self, '_report_received', None)
            if report_received is not None:
                report_received.clear()
            if not self.arm.connected:
                raise DualArmError(f"{self.config.arm_id}: disconnected")
            state = self.checked_value(self.arm.get_state(), "state")
            errors = self.checked_value(self.arm.get_err_warn_code(), "errors")
            allowed_states = (0, 1, 2) if self.position_command_active else (0, 2)
            if getattr(self, 'recover_stopped', False):
                allowed_states += (4,)
            if state not in allowed_states or list(errors) != [0, 0]:
                raise DualArmError(
                    f"{self.config.arm_id}: controller fault/state {state}, {errors}"
                )
            feedback = self._checked_gripper_feedback()
            if not feedback["usable_for_completion"]:
                raise DualArmError(f"{self.config.arm_id}: gripper feedback unavailable")
            if getattr(self, '_report_callback', None) is not None:
                # Planning can hold the GIL long enough to leave a cached frame
                # old. Yield for a new report; never relabel an old timestamp.
                if report_received is not None:
                    report_received.wait(.06)
                with self._report_lock:
                    sample = self._latest_report
                if sample is None:
                    raise DualArmError('coherent controller report unavailable')
                raw, raw_pose, report_time = sample
                sampled_started = min(sampled_started, report_time)
            else:
                # Compatibility with adapters without a report channel. Real
                # xArm SDK connections above always retain their report callback.
                raw = self.checked_value(
                    self.arm.get_servo_angle(is_radian=False, is_real=True), 'joints')
                raw_pose = self.checked_value(self.arm.get_position(is_radian=False), 'pose')
                report_time = time.monotonic()
            if (
                not isinstance(raw, (list, tuple))
                or not self.config.axis <= len(raw) <= 7
            ):
                raise DualArmError("unexpected SDK joint vector")
            joints = finite(raw[: self.config.axis], (self.config.axis,), "live joints")
            pose = finite(
                raw_pose,
                (6,),
                "live pose",
            )
            return {
                "joints": joints.tolist(),
                "pose": pose.tolist(),
                "gripper": feedback,
                "sampled_monotonic_s": report_time,
                "sample_started_monotonic_s": sampled_started,
            }

    def inverse(self, pose, seed):
        with self.lock:
            result = self.arm.get_inverse_kinematics(
                list(pose),
                input_is_radian=False,
                return_is_radian=False,
                limited=True,
                ref_angles=list(seed),
            )
            raw = self.checked_value(result, "IK")
            if (
                not isinstance(raw, (list, tuple))
                or not self.config.axis <= len(raw) <= 7
            ):
                raise DualArmError("controller returned malformed IK")
            q = finite(raw[: self.config.axis], (self.config.axis,), "controller IK")
            if self.checked_value(
                self.arm.is_joint_limit(q.tolist(), is_radian=False), "joint limits"
            ):
                raise DualArmError("controller joint limit")
            return q

    def prepare(self):
        with self.lock:
            if self.cancel.is_set():
                raise DualArmError("paired task cancelled")
            # No fault clearing, motor enabling or automatic return to Home.
            self._check("set_gripper_mode", self.arm.set_gripper_mode(0))
            self._check("set_gripper_enable", self.arm.set_gripper_enable(True))
            self._check("set_mode(servo)", self.arm.set_mode(1))
            # Never send state=0 after another worker may have stopped both arms.
            # Firmware that does not preserve ready state here must be commissioned
            # before this path is enabled; snapshot() will reject a paused state.
            self.snapshot()

    def servo(self, joints):
        with self.lock:
            if self.cancel.is_set():
                raise DualArmError("paired task cancelled")
            self._check(
                "set_servo_angle_j",
                self.arm.set_servo_angle_j(list(joints), is_radian=False),
            )

    def prepare_position(self):
        if self.cancel.is_set() or self.arm.mode != 0:
            raise DualArmError('native Home requires ready position mode (0)')
        self.snapshot()
        if self.arm.get_is_moving() or self.checked_value(self.arm.get_cmdnum(), 'command queue') != 0:
            raise DualArmError('native Home requires stationary arms with empty command queues')
        if getattr(self, 'recover_stopped', False):
            state = self.checked_value(self.arm.get_state(), 'state before recovery')
            if state == 4:
                if self.cancel.is_set() or list(self.checked_value(self.arm.get_err_warn_code(), 'recovery faults')) != [0, 0]:
                    raise DualArmError('cannot recover a cancelled or faulted controller')
                self._check('motion_enable', self.arm.motion_enable(enable=True))
                self._check('set_state(normal)', self.arm.set_state(0))
                deadline = time.monotonic() + 2.
                while self.checked_value(self.arm.get_state(), 'recovered state') not in (0, 2):
                    if self.cancel.wait(.02) or time.monotonic() >= deadline:
                        raise DualArmError('controller did not leave STOP')
            self.recover_stopped = False
            self.snapshot()
        self._check('set_gripper_mode', self.arm.set_gripper_mode(0))
        self._check('set_gripper_enable', self.arm.set_gripper_enable(True))

    def move_joint_position(self, joints, speed, acceleration, timeout):
        if self.cancel.is_set() or self.arm.mode != 0:
            raise DualArmError('native Home cancelled or controller mode changed')
        self.position_command_active = True
        try:
            self._check('set_servo_angle(position)', self.arm.set_servo_angle(
                angle=list(joints), speed=speed, mvacc=acceleration,
                is_radian=False, wait=True, timeout=timeout, radius=-1))
        finally:
            self.position_command_active = False

    def check_gripper_wait(self, started):
        if self.cancel.is_set():
            raise DualArmError("paired gripper wait cancelled")
        if (
            time.monotonic() - started
            > self.config.gripper.gripper_completion_timeout_s
        ):
            raise DualArmError("gripper completion deadline exceeded")
        if not self.arm.connected:
            raise DualArmError("gripper connection lost")
        state = self.checked_value(self.arm.get_state(), "state during gripper wait")
        errors = self.checked_value(
            self.arm.get_err_warn_code(), "faults during gripper wait"
        )
        if state not in (0, 2) or list(errors) != [0, 0]:
            raise DualArmError("controller fault during gripper wait")

    def gripper(self, target):
        # SDK requests have their own transport lock. Do not hold the arm lock
        # for the entire feedback wait: the coordinator must keep reading both
        # arms and stop if a supposedly stationary holder drifts.
        return self._command_gripper(self.config.gripper, target=target)[0]

    def stop(self):
        # Do not wait on a possibly blocked motion/gripper call's lock.
        self._check("set_state(stop)", self.arm.set_state(4))

    def finish(self):
        with self.lock:
            self._check("set_mode(position)", self.arm.set_mode(0))

    def disconnect(self):
        callback = getattr(self, '_report_callback', None)
        if callback is not None:
            getattr(self, '_report_arm', self.arm).release_report_callback(callback)
            self._report_callback = None
        report_arm = getattr(self, '_report_arm', self.arm)
        if report_arm is not self.arm:
            report_arm.disconnect()
        self.arm.disconnect()


class SimulatedConnection:
    simulated = True

    def __init__(self, config, model, cancel):
        self.config, self.model, self.cancel = config, model, cancel
        self.joints = config.home_joints.copy()
        self.position = config.gripper.gripper_open
        self.stopped = False
        self.commands = []

    def snapshot(self):
        if self.stopped:
            raise DualArmError("simulated arm stopped")
        return {
            "joints": self.joints.tolist(),
            "pose": self.model.forward(self.joints).tolist(),
            "gripper": {
                "position_pulse": self.position,
                "state": "stop",
                "error_code": 0,
                "usable_for_completion": True,
            },
            "sampled_monotonic_s": time.monotonic(),
        }

    def inverse(self, pose, seed):
        return self.model.inverse(pose, seed)

    def prepare(self):
        if self.cancel.is_set():
            raise DualArmError("cancelled")

    def prepare_position(self):
        self.prepare()

    def move_joint_position(self, joints, speed, acceleration, timeout):
        if self.cancel.is_set() or self.stopped:
            raise DualArmError('cancelled')
        self.joints = np.array(joints).copy()
        self.commands.append(('position', self.joints.tolist()))

    def servo(self, joints):
        if self.cancel.is_set() or self.stopped:
            raise DualArmError("cancelled")
        self.joints = np.array(joints).copy()
        self.commands.append(("servo", self.joints.tolist()))

    def gripper(self, target):
        if self.cancel.is_set():
            raise DualArmError("cancelled")
        self.position = (
            self.config.gripper.gripper_open
            if target == "open"
            else self.config.gripper.gripper_close + 20
        )
        self.commands.append(("gripper", target))
        return {
            "feedback": self.snapshot()["gripper"],
            "completion": {"status": "COMPLETED", "synthetic": True},
        }

    def stop(self):
        self.stopped = True

    def finish(self):
        pass

    def disconnect(self):
        pass


class DualArmCoordinator:
    def __init__(
        self, config, models, connections, cancel, directory, *, realtime=True
    ):
        self.config, self.models, self.connections = config, models, connections
        self.cancel = cancel
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="dual-arm")
        self.vision_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="dual-vision"
        )
        self.realtime = realtime
        self.simulated = all(c.simulated for c in connections.values())
        if not self.simulated and any(c.simulated for c in connections.values()):
            raise DualArmError("mixing real and simulated arms is forbidden")
        if not self.simulated and not realtime:
            raise DualArmError("real execution requires a real clock")
        self.holding_positions = None
        self.pinned_reference = {}
        self.sequence = 0
        self.log_lock = threading.Lock()

    def event(self, name, **data):
        with self.log_lock:
            self.sequence += 1
            with (self.directory / "events.jsonl").open("a") as out:
                out.write(
                    json.dumps(
                        {
                            "sequence": self.sequence,
                            "event": name,
                            "monotonic_s": time.monotonic(),
                            **data,
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )

    def paired(self, operation, *, timeout=2.0, check_skew=False, on_wait=None):
        barrier = threading.Barrier(2)
        starts = {}

        def call(k):
            barrier.wait(timeout=timeout)
            if self.cancel.is_set():
                raise DualArmError("paired task cancelled")
            starts[k] = time.monotonic()
            return operation(k, self.connections[k])

        jobs = {k: self.pool.submit(call, k) for k in self.connections}
        began = time.monotonic()
        try:
            while True:
                done, pending = wait(
                    jobs.values(),
                    timeout=min(0.02, timeout) if on_wait else timeout,
                    return_when=FIRST_EXCEPTION,
                )
                if (
                    not pending
                    or any(f.exception() is not None for f in done)
                    or time.monotonic() - began >= timeout
                ):
                    break
                if on_wait:
                    on_wait()
                else:
                    break
        except BaseException:
            self.stop_all()
            raise
        failure = next((f.exception() for f in done if f.exception() is not None), None)
        if failure is not None or pending:
            self.cancel.set()
            barrier.abort()
            self.stop_all()
            if failure is not None:
                raise failure
            raise DualArmError("paired operation deadline exceeded")
        skew = max(starts.values()) - min(starts.values())
        if check_skew and skew > self.config.limits["dispatch_skew_s"]:
            raise DualArmError(f"host dispatch skew exceeded: {skew:.4f}s")
        return {k: job.result() for k, job in jobs.items()}, skew

    def snapshots(self):
        state = self.paired(
            lambda k, c: c.snapshot(),
            timeout=self.config.raw["safety"]["max_feedback_age_s"],
        )[0]
        self.validate_fresh(state)
        return state

    def validate_fresh(self, state):
        now = time.monotonic()
        for k, row in state.items():
            timestamp = row.get(
                "sample_started_monotonic_s", row["sampled_monotonic_s"]
            )
            if (
                not 0
                <= now - timestamp
                <= self.config.raw["safety"]["max_feedback_age_s"]
            ):
                raise DualArmError(f"{k}: feedback is stale ({now-timestamp:.3f}s > "
                                   f"{self.config.raw['safety']['max_feedback_age_s']:.3f}s); refusing the next command")

    def stop_all(self):
        self.cancel.set()
        # A separate pool keeps stop attempts independent of blocked workers.
        pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="dual-stop")
        jobs = {k: pool.submit(c.stop) for k, c in self.connections.items()}
        done, _ = wait(jobs.values(), timeout=2)
        status = {
            k: (
                "sent"
                if f in done and f.exception() is None
                else str(f.exception())
                if f in done
                else "stop delivery unconfirmed"
            )
            for k, f in jobs.items()
        }
        pool.shutdown(wait=False, cancel_futures=True)
        self.event("stop_both", results=status)
        return status

    def monitor(
        self,
        expected_joints,
        expected_poses,
        *,
        holding,
        arrival=False,
        tool_contact=False,
    ):
        state = self.snapshots()
        joints = {k: np.array(v["joints"]) for k, v in state.items()}
        # Use controller-measured TCP, then cross-check independent FK.
        for k, arm in self.config.arms.items():
            if (
                k in self.pinned_reference
                and np.max(np.abs(joints[k] - self.pinned_reference[k]))
                > self.config.limits["stationary_tolerance_deg"]
            ):
                raise DualArmError(f"{k}: pinned arm moved")
            base = state[k]["pose"]
            fk_error = pose_error(base, self.models[k].forward(joints[k]))
            if fk_error[0] > 2 or fk_error[1] > 1:
                raise DualArmError(f"{k}: live controller/model FK mismatch")
            actual_world = matrix_pose(arm.world_from_base @ pose_matrix(base))
            position, angle = pose_error(actual_world, expected_poses[k])
            limit = (
                self.config.limits["arrival_error_mm"]
                if arrival
                else self.config.limits["tracking_error_mm"]
            )
            if position > limit or angle > self.config.limits["tracking_error_deg"]:
                raise DualArmError(
                    f"{k}: TCP tracking error {position:.3f} mm, {angle:.3f} deg"
                )
            if (
                np.max(np.abs(joints[k] - expected_joints[k]))
                > self.config.limits["tracking_error_deg"]
            ):
                raise DualArmError(f"{k}: joint tracking error")
            if (
                holding
                and self.holding_positions is not None
                and k in self.holding_positions
            ):
                g = state[k]["gripper"]
                if (
                    g["state"] == "moving"
                    or abs(g["position_pulse"] - self.holding_positions[k]) > 5
                ):
                    raise DualArmError(f"{k}: gripper position changed while holding")
        validate_sample(
            self.config, self.models, joints, holding=holding, tool_contact=tool_contact
        )
        return state

    def observe(self, callback, phase, expected_joints):
        # Keep monitoring both holders while capture/model evaluation runs.
        job = self.vision_pool.submit(callback, phase)
        started = time.monotonic()
        while not job.done():
            if time.monotonic() - started > self.config.limits["max_vision_wait_s"]:
                raise DualArmError("paired hold vision deadline exceeded")
            state = self.snapshots()
            for k in state:
                if (
                    np.max(np.abs(np.asarray(state[k]["joints"]) - expected_joints[k]))
                    > self.config.limits["stationary_tolerance_deg"]
                ):
                    raise DualArmError("arm moved during hold observation")
                if (
                    self.holding_positions is not None
                    and k in self.holding_positions
                    and abs(
                        state[k]["gripper"]["position_pulse"]
                        - self.holding_positions[k]
                    )
                    > 5
                ):
                    raise DualArmError("gripper changed during hold observation")
            validate_sample(
                self.config,
                self.models,
                {k: np.asarray(v["joints"]) for k, v in state.items()},
                holding=phase.holding,
                tool_contact=phase.tool_contact,
            )
            if self.cancel.wait(0.05):
                raise DualArmError("cancelled during observation")
        result = job.result()
        self.event("hold_evaluation", phase=phase.name, result=result)
        require_hold(result, phase)

    def execute(
        self,
        program,
        initial_joints,
        observation,
        checkpoint,
        *,
        confirmed=False,
        commissioning=False,
        homing=False,
    ):
        result = {
            "schema_version": 1,
            "physical_execution": not self.simulated,
            "status": "RUNNING",
            "completed_phases": [],
            "automatic_release_on_failure": False,
            "automatic_home_on_failure": False,
        }
        expected_joints = {k: np.array(q).copy() for k, q in initial_joints.items()}
        try:
            if self.config.execution_mode != 'servo_stream':
                raise DualArmError('streaming executor cannot use controller-sequential envelopes')
            if homing:
                from .homing import validate_home_program
                validate_home_program(program, self.config)
            if not self.simulated:
                if not confirmed:
                    raise DualArmError("--real requires --confirm-real")
                self.config.require_real(commissioning=commissioning, homing=homing)
                if any(
                    (i.phase if isinstance(i, Motion) else i).pin_arm
                    or (i.phase if isinstance(i, Motion) else i).name == "pin_descend"
                    for i in program
                ):
                    raise DualArmError(
                        "pin_pull physical execution requires a force-limited contact controller; currently simulation/preflight only"
                    )
                if not commissioning and not homing:
                    observation.validate_for(self.config, live=True)
            expected_poses = validate_sample(
                self.config, self.models, expected_joints, holding=False
            )
            state = self.monitor(
                expected_joints, expected_poses, holding=False, arrival=True
            )
            self.paired(lambda k, c: c.prepare())
            for item in program:
                if self.cancel.is_set():
                    raise DualArmError("task cancelled")
                phase = item.phase if isinstance(item, Motion) else item
                if phase.pin_arm:
                    self.pinned_reference.setdefault(
                        phase.pin_arm, expected_joints[phase.pin_arm].copy()
                    )
                else:
                    self.pinned_reference.clear()
                self.event("phase_start", phase=phase.name, kind=phase.kind)
                if isinstance(item, Motion):
                    state = self.monitor(
                        expected_joints,
                        expected_poses,
                        holding=phase.holding,
                        arrival=True,
                        tool_contact=phase.tool_contact,
                    )
                    began = time.monotonic()
                    for index, t in enumerate(item.times[1:], 1):
                        targets = {k: q[index] for k, q in item.joints.items()}
                        # Check the actual measured-to-command interval before
                        # either arm receives it, not only after motion occurs.
                        validate_sweep(
                            self.config,
                            self.models,
                            {k: np.asarray(v["joints"]) for k, v in state.items()},
                            targets,
                            tool_contact=phase.tool_contact,
                        )
                        if self.realtime:
                            remaining = began + float(t) - time.monotonic()
                            if remaining > 0 and self.cancel.wait(remaining):
                                raise DualArmError("cancelled")
                            if (
                                time.monotonic() - (began + float(t))
                                > self.config.limits["max_tick_lateness_s"]
                            ):
                                raise DualArmError(
                                    "servo clock deadline missed; no catch-up jumps"
                                )
                        self.validate_fresh(state)
                        expected_joints = targets
                        expected_poses = {k: p[index] for k, p in item.poses.items()}
                        _, skew = self.paired(
                            lambda k, c, targets=expected_joints: c.servo(targets[k]),
                            check_skew=True,
                            timeout=self.config.raw["safety"]["max_feedback_age_s"],
                        )
                        state = self.monitor(
                            expected_joints,
                            expected_poses,
                            holding=phase.holding,
                            tool_contact=phase.tool_contact,
                        )
                        self.event(
                            "servo_tick",
                            phase=phase.name,
                            index=index,
                            planned_time_s=float(t),
                            dispatch_skew_s=skew,
                            targets={k: q.tolist() for k, q in expected_joints.items()},
                            state=state,
                        )
                    # Allow the final servo point to settle while monitoring.
                    if self.realtime and self.cancel.wait(0.15):
                        raise DualArmError("cancelled")
                    state = self.monitor(
                        expected_joints,
                        expected_poses,
                        holding=phase.holding,
                        arrival=True,
                        tool_contact=phase.tool_contact,
                    )
                elif phase.kind in {"open", "close"}:
                    replies, _ = self.paired(
                        lambda k, c, target=phase.kind, arms=phase.gripper_arms: (
                            c.gripper(target) if k in arms else None
                        ),
                        timeout=max(
                            a.gripper.gripper_completion_timeout_s
                            for a in self.config.arms.values()
                        )
                        + 1,
                        on_wait=lambda q=expected_joints, p=expected_poses, contact=phase.tool_contact: (
                            self.monitor(
                                q, p, holding=False, arrival=True, tool_contact=contact
                            )
                        ),
                    )
                    self.event("gripper_completed", phase=phase.name, results=replies)
                    positions = dict(self.holding_positions or {})
                    for k in phase.gripper_arms:
                        if phase.kind == "close":
                            positions[k] = replies[k]["feedback"]["position_pulse"]
                        else:
                            positions.pop(k, None)
                    self.holding_positions = positions or None
                    state = self.monitor(
                        expected_joints,
                        expected_poses,
                        holding=phase.holding,
                        arrival=True,
                        tool_contact=phase.tool_contact,
                    )
                elif phase.kind == "observe":
                    self.observe(checkpoint, phase, expected_joints)
                    state = self.monitor(
                        expected_joints,
                        expected_poses,
                        holding=phase.holding,
                        arrival=True,
                        tool_contact=phase.tool_contact,
                    )
                else:
                    raise DualArmError(f"unknown phase kind {phase.kind}")
                result["completed_phases"].append(phase.name)
                self.event("phase_complete", phase=phase.name)
            self.paired(lambda k, c: c.finish())
            result["status"] = "COMPLETED"
        except BaseException as exc:  # noqa: BLE001 -- cancellation must stop both devices and persist evidence
            result["status"] = (
                "INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAILED"
            )
            result["error"] = f"{type(exc).__name__}: {exc}"
            result["stop_results"] = self.stop_all()
            self.event("task_failed", error=result["error"])
        finally:
            (self.directory / "execution.json").write_text(
                json.dumps(result, indent=2, allow_nan=False) + "\n"
            )
        return result

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)
        self.vision_pool.shutdown(wait=False, cancel_futures=True)
        for connection in self.connections.values():
            connection.disconnect()
