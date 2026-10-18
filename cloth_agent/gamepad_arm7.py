"""Bounded jog backend; reuse native full-path validation without detours."""
import fcntl
import json
from pathlib import Path
import threading
import time
from datetime import datetime, timezone

import numpy as np

from .codex_native_runtime import NativeRuntime
from .dual_arm.geometry import matrix_pose, pose_matrix, DualArmError
from .gamepad_teleop import motion_source, DEFAULT_YAW_STEP_DEG


class JogRuntime(NativeRuntime):
    active_arms = ('right',)

    def load_config(self, config_path):
        config = super().load_config(config_path)
        # Reuse the Arm7-only pipeline's parked Arm6 geometry, without its
        # camera setup or changes to collision envelopes.
        parked_file = config.root / 'data/robot/arm6_parked_for_arm7.json'
        parked = json.loads(parked_file.read_text())
        if parked.get('arm') != 'left':
            raise DualArmError('arm7 static obstacle must be arm6/left')
        joints = np.asarray(parked['joint_angles_deg'], dtype=float)
        if joints.shape != (config.arms['left'].axis,) or not np.isfinite(joints).all():
            raise DualArmError('invalid arm6 static obstacle joints')
        parked['reference_file'] = str(parked_file)
        config.static_arm_obstacles = {'left': parked}
        return config

    def _tcp_segments(self, k, pose, q, *, allow_detour=False):
        if k != 'right':
            raise DualArmError('gamepad only controls Arm7')
        return super()._tcp_segments(k, pose, q, allow_detour=False)


class Arm7Jogger:
    recover_stopped_on_start = False

    def _create_runtime(self, config):
        return JogRuntime(config)

    def __init__(self, config, output, guard, step_mm=1., yaw_step_deg=DEFAULT_YAW_STEP_DEG, *, runtime=None):
        from .dual_arm.cli import connect
        from .dual_arm.execution import DualArmCoordinator
        from .run_storage import runs_root
        from .codex_astra_pipeline import ROOT
        self.guard, self.step_mm, self.yaw_step_deg = guard, step_mm, yaw_step_deg
        self.output = Path(output) if output else runs_root(ROOT) / datetime.now(timezone.utc).strftime('gamepad_%Y%m%dT%H%M%S%fZ')
        self.output.mkdir(parents=True, exist_ok=False)
        self.owns_runtime = runtime is None
        self.runtime = self._create_runtime(config) if runtime is None else runtime
        # No automatic controller STOP recovery, Home, perception, or camera use.
        if self.owns_runtime:
            self.runtime.config.home_recover_stopped = self.recover_stopped_on_start
        self.executing = threading.Event()
        self.execution_generation = None
        self.closed = threading.Event()
        self.index = 0
        self.watchdog = None
        if not self.owns_runtime:
            if self.runtime.coordinator is None or self.runtime.cancel.is_set():
                raise DualArmError('shared runtime must be connected and ready')
            return
        try:
            r = self.runtime
            r._lease = open('/tmp/clothagent_codex_dual_arm.lock', 'a')
            fcntl.flock(r._lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            connections = connect(r.config, r.models, r.cancel, True)
            try:
                r.coordinator = DualArmCoordinator(r.config, r.models, connections, r.cancel,
                                                   self.output / 'controller', realtime=True)
            except BaseException:
                for connection in connections.values():
                    connection.disconnect()
                raise
            self.watchdog = threading.Thread(target=self._watch, daemon=True)
            self.watchdog.start()
        except BaseException:
            self.close()
            raise

    def _watch(self):
        while not self.closed.wait(.02):
            if self.executing.is_set() and not self.guard.permits(generation=self.execution_generation):
                self.runtime.stop()
                return

    def step(self, intent, generation):
        r = self.runtime
        if r.cancel.is_set() or not self.guard.permits(intent, generation):
            return dict(status='INPUT_EXPIRED')
        self.index += 1
        folder = self.output / f'step_{self.index:06d}'
        folder.mkdir()
        snapshot = r.coordinator.snapshots()
        initial = {k: np.asarray(v['joints']) for k, v in snapshot.items()}
        world = matrix_pose(r.config.arms['right'].world_from_base @
                            pose_matrix(r.models['right'].forward(initial['right'])))
        source = motion_source(intent, world, self.step_mm, self.yaw_step_deg)
        if source is None:
            return dict(status='IDLE')
        (folder / 'motion.py').write_text(source)
        started = time.monotonic()
        try:
            operations = r.plan(source, initial, preserve_command_boundaries=True)
        except DualArmError as exc:
            result = dict(status='REJECTED', reason=str(exc))
        else:
            # Never dispatch a direction chosen before a long validation pause.
            if time.monotonic() - started > 2 or not self.guard.permits(intent, generation):
                result = dict(status='INPUT_EXPIRED')
            else:
                # Independent dispatch check keeps the other arm stationary.
                for kind, data in operations:
                    if kind == 'move':
                        if data.arm != 'right' or not np.allclose(data.start['left'], data.end['left'], atol=1e-9, rtol=0):
                            raise DualArmError('teleoperation cannot move Arm6')
                    elif data['arm'] != 'right':
                        raise DualArmError('teleoperation cannot operate Arm6 gripper')
                self.execution_generation = generation
                self.executing.set()
                try:
                    if not self.guard.permits(intent, generation):
                        result = dict(status='INPUT_EXPIRED')
                    else:
                        r.arrival(initial)
                        result = r._execute_operations(operations, folder)
                finally:
                    self.executing.clear()
        (folder / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
        return result

    def stop(self):
        # Cancel pending planning; stop active controller motion through the
        # existing safety stop (Arm6's static adapter sends no command).
        # A stopped session never auto-resumes.
        if self.executing.is_set():
            return self.runtime.stop()
        self.runtime.cancel.set()
        return {}

    def close(self):
        self.closed.set()
        if self.watchdog is not None:
            self.watchdog.join(timeout=3)
        if self.owns_runtime:
            self.runtime.close()
