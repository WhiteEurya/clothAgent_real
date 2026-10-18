"""Arm6 collection runtime; Arm7 is offline geometry, never a connection."""
import json

import numpy as np

from .single_arm_runtime import SingleArmRuntime
from .dual_arm.geometry import DualArmError


def _arm6_text(text):
    return (text.replace('arm7', '__ACTIVE__').replace('arm6', 'arm7')
            .replace('__ACTIVE__', 'arm6').replace('right', '__SIDE__')
            .replace('left', 'right').replace('__SIDE__', 'left')
            .replace('Camera B', 'Camera A'))


class CollectionRuntime(SingleArmRuntime):
    arm_key = 'left'
    other_key = 'right'
    active_arms = ('left',)
    robot_name = 'arm6'
    other_robot_name = 'arm7'
    wrist_joint = 'joint6'
    camera_label = 'A'
    planning_policy = _arm6_text(SingleArmRuntime.planning_policy)
    preparation_description = _arm6_text(SingleArmRuntime.preparation_description)
    motion_contract = _arm6_text(SingleArmRuntime.motion_contract)

    direct_joint_moves = True

    def load_config(self, config_path):
        config = super().load_config(config_path)
        config.path_validation_enabled = False
        # Automatic replay/collection uses _command_gripper, which reads this
        # value on every position command. Match the Arm6 gamepad SDK speed.
        config.arms['left'].gripper.gripper_speed = 2000.
        camera = config.raw['camera']
        if camera['mount_arm'] != 'left' or str(camera['serial']) != '317222073552':
            raise DualArmError('Arm6 collection requires calibrated Camera A')
        # Match Arm7's active-arm collision policy exactly, with Arm6 geometry.
        config.collision_envelopes_disabled_arms = frozenset({'left'})
        # Camera-specific exposure/calibration are hardware data, not a planner.
        camera.update(depth_exposure_us=8500., depth_gain=16.,
                      depth_stability={'warmup_frames':30, 'max_frames':90,
                                       'window_frames':5, 'stable_windows':3,
                                       'max_range_mm':3.})
        # Retain the existing paired geometry interface without contacting the
        # inactive controller. This saved pose is explicitly not live feedback.
        path = config.root / 'data/robot/xarm7_perception_pose.json'
        record = json.loads(path.read_text())
        arm = config.arms['right']
        if (record.get('robot_ip') != arm.ip or record.get('axis') != arm.axis
                or record.get('control_box_sn') != arm.serial):
            raise DualArmError('saved Arm7 geometry identity mismatch')
        q = np.asarray(record['joint_angles_deg'], dtype=float)
        if q.shape != (arm.axis,) or not np.isfinite(q).all():
            raise DualArmError('invalid saved Arm7 geometry')
        config.static_arm_obstacles = {'right': dict(
            arm='right', joint_angles_deg=q.tolist(), reference_file=str(path),
            recorded_at=record.get('recorded_at'), assumption='saved_perception_pose',
            live_feedback=False)}
        return config

    def _tcp_segments(self, k, pose, q, *, allow_detour=False):
        from .queued_motion import CartesianPositionSegment
        from .dual_arm.geometry import matrix_pose, pose_matrix
        segments, final = super()._tcp_segments(k, pose, q, allow_detour=False)
        if not segments:
            return segments, final
        # Retain host IK samples for limits and live feedback monitoring, but
        # send one straight Cartesian command per source move to the controller.
        speed = self.config.limits['cartesian_speed_mm_s']
        distance = 0.
        for segment in segments:
            a = self.models[k].forward(segment.start[k])
            b = self.models[k].forward(segment.end[k])
            step = float(np.linalg.norm(np.asarray(b[:3])-a[:3]))
            distance += step
            joint_delta = float(np.max(np.abs(segment.end[k]-segment.start[k])))
            if step > 1e-6 and joint_delta > 1e-6:
                speed = min(speed, step / joint_delta * segment.speed)
        target = matrix_pose(np.linalg.inv(self.config.arms[k].world_from_base) @
                            pose_matrix(np.r_[np.asarray(pose[:3])*1000,np.degrees(pose[3:])]))
        acceleration = self.config.limits['cartesian_accel_mm_s2']
        timeout = max(sum(s.timeout for s in segments), distance / speed + 10.)
        first, last = segments[0], segments[-1]
        command = CartesianPositionSegment(k, first.start, last.end,
            min(s.speed for s in segments), min(s.acceleration for s in segments),
            timeout, tuple(segments), target.tolist(), speed, acceleration)
        return [command], final

    def move_to_perception(self, directory):
        # On a newly started session, recover software STOP before any pose
        # comparison (including ALREADY_AT_PERCEPTION). Faults stay latched.
        if not self.prepared:
            self.coordinator.connections[self.arm_key].prepare_position()
            self.prepared = True
        return super().move_to_perception(directory)

    def _preparation_program(self, initial, owner, target):
        final, source = super()._preparation_program(initial, owner, target)
        if getattr(self, 'open_at_home_on_return', False):
            lines = source.splitlines()
            # Wait for Home arrival before fully opening, then go to perception.
            source = '\n'.join([lines[1], lines[0], *lines[2:]]) + '\n'
        return final, source

    def _plan_preparation(self, source, initial):
        """Home/perception: direct commands without swept-path certification."""
        from .codex_motion import parse_code
        from .dual_arm.sequential_home import PositionSegment, command_limits

        self.preview_candidates = []
        self.preview_failures = []
        self.planning_context = dict(stage='direct_home_perception')
        self.plan_stats = dict(path_validation='disabled_by_user',
                               scope='home_perception', rrt=False)
        q = {k: np.asarray(v, dtype=float).copy() for k, v in initial.items()}
        operations = []
        model = self.models[self.arm_key]
        for name, args in parse_code(source):
            if self.cancel.is_set():
                raise DualArmError('Home/perception cancelled')
            if name == 'gripper' and args['arm'] == self.arm_key:
                operations.append(('gripper', dict(**args,
                    joints={k: v.tolist() for k, v in q.items()})))
            elif name == 'move_joints' and set(args) == {self.arm_key}:
                radians = np.asarray(args[self.arm_key], dtype=float)
                if (radians.shape != (self.config.arms[self.arm_key].axis,)
                        or not np.isfinite(radians).all()
                        or np.any(radians < model.lower) or np.any(radians > model.upper)):
                    raise DualArmError('Home/perception target exceeds joint limits')
                end = {k: v.copy() for k, v in q.items()}
                end[self.arm_key] = np.degrees(radians)
                if not np.allclose(q[self.arm_key], end[self.arm_key], atol=1e-9, rtol=0):
                    speed, acceleration, timeout = command_limits(
                        self.config, model, self.config.arms[self.arm_key],
                        q[self.arm_key], end[self.arm_key])
                    operations.append(('move', PositionSegment(
                        self.arm_key, q, end, speed, acceleration, timeout)))
                q = end
            else:
                raise DualArmError('Home/perception only accepts active-arm joint and gripper commands')
        return operations
