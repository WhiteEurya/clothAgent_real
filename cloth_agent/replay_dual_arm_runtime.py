"""Route the legacy logical Arm7 grasp to one of two live xArm7 devices."""
import json
import threading
from pathlib import Path

import numpy as np

from .codex_motion import parse_code
from .dual_arm.geometry import DualArmError, pose_matrix


def route_grasp(source, tcp_world_mm):
    """Bisect live TCP XY positions; preserve every world-frame target pose."""
    calls = parse_code(source, allow_grasp=True)
    if calls[0][0] != 'grasp' or any(set(args) != {'right'} for _, args in calls):
        raise DualArmError('partitioned replay requires one logical right grasp task')
    left, right = (np.asarray(tcp_world_mm[k], dtype=float) for k in ('left', 'right'))
    if any(p.shape != (3,) or not np.isfinite(p).all() for p in (left, right)):
        raise DualArmError('invalid live TCP positions')
    axis = right[:2] - left[:2]
    length = np.linalg.norm(axis)
    if length < 1.:
        raise DualArmError('TCP centers are indistinguishable in the table plane')
    axis /= length
    midpoint = (left + right) / 2.
    grasp_mm = np.asarray(calls[0][1]['right'][:3]) * 1000.
    distance = float((grasp_mm[:2] - midpoint[:2]) @ axis)
    arm = 'left' if distance < 0. else 'right'
    routed = '\n'.join(f'motion.{name}({arm}={args["right"]!r})' for name, args in calls) + '\n'
    return routed, dict(active_arm=arm, inactive_arm='right' if arm == 'left' else 'left',
                       tcp_world_mm={k: np.asarray(v).tolist() for k, v in tcp_world_mm.items()},
                       midpoint_world_mm=midpoint.tolist(), right_axis_xy=axis.tolist(),
                       grasp_world_mm=grasp_mm.tolist(), signed_distance_mm=distance,
                       center_tie_arm='right', rule='live_tcp_xy_perpendicular_bisector')


class PartitionedReplayRuntime:
    """Graph/observation adapter to ExperimentRunner, with no motion backend."""
    active_arms = ('left', 'right')
    grasp_offset_mm = 0.
    preview = None
    planning_policy = '输出逻辑 right 的单夹爪 grasp 和搬运；宿主按两夹爪中点选择物理臂。'
    motion_contract = '只允许一个 motion.grasp(right=[x,y,z,roll,pitch,yaw])，随后使用 right 的 move_tcp，保持 roll/pitch；单位 m/rad。'
    preparation_description = '启动时依次张开两夹爪并回 Home；每次采图前左臂到记录的 perception 位置并确认静止。'

    def __init__(self, config_path, *, planning_only=False):
        from .dual_arm.config import DualConfig
        from .dual_arm.kinematics import ArmModel
        from .site_observation import ROOT, load_site
        self.config_path = Path(config_path).resolve()
        self.config = DualConfig.load_geometry(config_path, root=ROOT)
        if any(a.axis != 7 for a in self.config.arms.values()):
            raise DualArmError('partitioned replay requires two xArm7 robots')
        if self.config.raw['camera'].get('backend') != 'zed' or self.config.raw['camera'].get('mount_arm') is not None:
            raise DualArmError('partitioned replay requires the calibrated fixed ZED')
        self.models = {k: ArmModel(a) for k, a in self.config.arms.items()}
        self.profile = load_site()
        self.task_policy = json.loads((ROOT / 'config/folding.partitioned.json').read_text())
        self.calibration_id = self.config.raw['calibration_id']
        self.planning_only = planning_only
        self.physical = not planning_only
        self._routing_lock = threading.Lock()
        self._lease = None
        self._startup_checked = False
        self._startup_home_complete = False
        self.perception_position = self.config.raw['arms']['left'].get('perception_position')
        if self.perception_position is not None:
            from .dual_arm.geometry import finite
            saved = self.perception_position
            finite(saved['joints_deg'], (7,), 'perception joints')
            finite(saved['pose_base_mm_deg'], (6,), 'perception TCP pose')
            if (saved['control_box_sn'] != self.config.arms['left'].serial
                    or saved['robot_sn'] != self.config.raw['arms']['left']['robot_sn']):
                raise DualArmError('perception position belongs to a different left arm')

    def _open_startup(self, directory, before, snapshots):
        """Open both tools, sequentially and with feedback, before any Home."""
        from .experiment import ExperimentRunner
        from .experiments.full_grasp_adapter import build_robot_config
        from .robot_collision import RobotCollision
        from .experiments import unfold_console as console
        if self.planning_only:
            raise DualArmError('planning-only adapter cannot execute motion')
        for key in ('left', 'right'):
            initial = {k: np.asarray(v['joints']) for k,v in before.items()}
            pose = self.models[key].forward(initial[key])
            config = build_robot_config(self.config.raw, {}, key, *pose[3:5], config_path=self.config_path)
            checker = RobotCollision(self.config_path, initial, key, root=self.config.root)
            checker.require_execution(config, [{'name':'open_gripper','args':{}}])
            runner = ExperimentRunner(Path(directory)/key, config, collision_checker=checker)
            path = runner.workspace/'open_gripper.py'
            path.write_text('def run():\n    open_gripper()\n',encoding='utf-8')
            with console.stage(('左' if key=='left' else '右')+'夹爪张开，等待位置反馈','startup'):
                result = runner.run_experiment(path,real=True,confirmed=True)
            if not result['execution_completed']:
                raise DualArmError(f'{key}: startup gripper open failed; Home not started')
            after = snapshots()
            for arm in initial:
                if np.max(np.abs(np.asarray(after[arm]['joints'])-initial[arm])) > self.config.limits['stationary_tolerance_deg']:
                    raise DualArmError(f'{arm}: arm moved while opening startup grippers')
            before = after
        return before

    def _home_startup(self, directory, before, snapshots):
        from .experiment import ExperimentRunner
        from .experiments.full_grasp_adapter import build_robot_config
        from .robot_collision import RobotCollision
        from .experiments import unfold_console as console
        if self.planning_only:
            raise DualArmError('planning-only adapter cannot execute motion')
        tolerance = self.config.limits['stationary_tolerance_deg']
        predicted = {k: np.asarray(v['joints']).copy() for k, v in before.items()}
        pending = []
        # Validate both sequential legs before dispatching either arm.
        for key in ('left', 'right'):
            home = self.config.arms[key].home_joints
            if np.max(np.abs(predicted[key] - home)) <= tolerance:
                continue
            pose = self.models[key].forward(home)
            config = build_robot_config(self.config.raw, {}, key, *pose[3:5], config_path=self.config_path)
            checker = RobotCollision(self.config_path, predicted, key, root=self.config.root)
            checker.require_execution(config, [{'name': 'home', 'args': {}}])
            pending.append((key, config))
            predicted[key] = home.copy()
        for key, config in pending:
            print(f'Startup Home: {key}', flush=True)
            initial = {k: v['joints'] for k, v in before.items()}
            checker = RobotCollision(self.config_path, initial, key, root=self.config.root)
            runner = ExperimentRunner(Path(directory) / key, config, collision_checker=checker)
            path = runner.workspace / 'startup_home.py'
            path.write_text('def run():\n    home()\n', encoding='utf-8')
            with console.stage(('左' if key=='left' else '右')+'臂返回 Home','startup'):
                result = runner.run_experiment(path, real=True, confirmed=True)
            if not result['execution_completed']:
                raise DualArmError(f'{key}: startup Home failed; graph replay not started')
            after = snapshots()
            if np.max(np.abs(np.asarray(after[key]['joints']) - self.config.arms[key].home_joints)) > tolerance:
                raise DualArmError(f'{key}: startup Home not reached')
            other = 'right' if key == 'left' else 'left'
            if np.max(np.abs(np.asarray(after[other]['joints']) - before[other]['joints'])) > tolerance:
                raise DualArmError(f'{other}: inactive arm moved during startup Home')
            before = after
        return before

    def _perception_before_capture(self, directory, before, snapshots, *, arm="left", saved=None):
        """Existing checked joint executor; no image is acquired before arrival."""
        from dataclasses import replace
        from .experiments.full_grasp_adapter import build_robot_config
        from .robot_collision import RobotCollision
        from .robot_api import XArmBackend
        from .experiments import unfold_console as console
        if self.planning_only:
            raise DualArmError('planning-only adapter cannot execute motion')
        saved = self.perception_position if saved is None else saved
        if saved is None:
            return before
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        target = np.asarray(saved['joints_deg'])
        tolerance = self.config.limits['stationary_tolerance_deg']
        record = dict(name='perception', active_arm=arm, target_joints_deg=target.tolist(),
                      snapshot_before=before, status='PENDING', motion_dispatched=False)
        backend = None
        try:
            if np.max(np.abs(np.asarray(before[arm]['joints']) - target)) <= tolerance:
                record.update(status='ALREADY_AT_PERCEPTION', snapshot_after=before)
                return before
            initial = {k: v['joints'] for k,v in before.items()}
            pose = self.models[arm].forward(target)
            config = build_robot_config(self.config.raw, {}, arm, *pose[3:5], config_path=self.config_path)
            config = replace(config, perception_joints_deg=tuple(target),
                             perception_pose_mm_deg=tuple(saved['pose_base_mm_deg']))
            checker = RobotCollision(self.config_path, initial, arm, root=self.config.root)
            with console.stage(f'{arm} 前往 perception 位置，另一臂保持静止', 'motion'):
                record['checked_plans'] = checker.require_execution(config, [{'name':'perception_position','args':{}}])
                backend = XArmBackend(config, collision_checker=checker)
                record['motion_dispatched'] = True
                actual_pose, state = backend.perception_position(config)
                record.update(actual_tcp_pose_mm_deg=actual_pose, robot_state=state)
                after = snapshots()
                if np.max(np.abs(np.asarray(after[arm]['joints']) - target)) > tolerance:
                    raise DualArmError('left: perception position not reached; capture not started')
                if np.max(np.abs(np.asarray(after['right' if arm=='left' else 'left']['joints']) - before['right' if arm=='left' else 'left']['joints'])) > tolerance:
                    raise DualArmError('right: inactive arm moved during perception positioning; capture not started')
                record.update(status='REACHED_PERCEPTION', snapshot_after=after)
                return after
        except BaseException as exc:
            record.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
            if backend is not None:
                try:
                    backend.stop()
                except BaseException as stop_error:
                    record['stop_error'] = str(stop_error)
            raise
        finally:
            try:
                if backend is not None:
                    backend.close()
            finally:
                (directory/'result.json').write_text(json.dumps(record, ensure_ascii=False, indent=2,
                    default=lambda value: value.tolist(), allow_nan=False), encoding='utf-8')

    def observe(self, directory, *, reset_zed=False):
        if self.planning_only:
            raise DualArmError('planning-only adapter cannot connect hardware')
        import fcntl
        from .site_robot_check import read_arm
        from .dual_arm.observation import save_observation
        from .perception import PerceptionConfig, capture_two_view_rgbd
        from .site_observation import ROOT, world_xyz_mm
        if self._lease is None:
            self._lease = open('/tmp/clothagent_codex_dual_arm.lock', 'w')
            fcntl.flock(self._lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def snapshots(*, startup=False):
            result = {}
            for key in self.models:
                row = read_arm(self.profile, key, **({'recover_stopped': True} if startup else {}))
                if row['status'] != 'READ_ONLY_CHECKS_PASS':
                    detail = row.get('error') or row.get('checks') or row
                    raise DualArmError(f'{key}: controller not stationary/ready ({row["status"]}): {detail}')
                if row['control_box_sn'] != self.config.arms[key].serial:
                    raise DualArmError(f'{key}: controller identity mismatch')
                result[key] = dict(joints=row['joints_deg'])
            return result
        # Fixed camera, sequential observe/execute: query both arms before and
        # after capture, not by reconnecting controllers for every video frame.
        perception_path = ROOT / 'config/perception.zed_dual_xarm7_20261103.json'
        perception = None if getattr(self,'dual_wrist_config',None) and not reset_zed else PerceptionConfig.load(ROOT, perception_path)
        startup = not self._startup_checked
        self._startup_checked = True  # No recovery retry within this run.
        if not startup and not self._startup_home_complete:
            raise DualArmError('startup Home failed; start a new run after resolving the failure')
        before = snapshots(startup=startup)
        if startup:
            before = self._open_startup(Path(directory).parent / 'startup_open', before, snapshots)
            before = self._home_startup(Path(directory).parent / 'startup_home', before, snapshots)
            self._startup_home_complete = True
        if getattr(self,'dual_wrist_config',None) and not reset_zed:
            from .dual_wrist_observation import capture_pair
            return capture_pair(self,directory,before,snapshots)
        position_log = Path(directory).with_name(Path(directory).name + '_perception_position')
        if not reset_zed:
            before = self._perception_before_capture(position_log, before, snapshots)
        elif not startup:
            before = self._home_startup(Path(directory).parent / 'reset_home', before, snapshots)
        recorder = getattr(self, 'video_recorder', None)
        if reset_zed and recorder is not None and getattr(recorder.camera,'rgb_only',False):
            raise DualArmError('ZED reset requires measured depth; RGB-only recorder cannot supply it')
        frames = (capture_two_view_rgbd(perception, camera_factory=recorder.camera_factory)
                  if recorder is not None else capture_two_view_rgbd(perception))
        fusion = None if reset_zed else getattr(self, 'dual_cloud_config', None)
        secondary = None
        if fusion is not None:
            from .dual_cloud_observation import validate_primary, capture_secondary
            if len(frames) != 1:
                raise DualArmError('dual fusion expects exactly one primary ZED frame')
            if self.config.raw['camera'].get('world_tilt_correction') or self.config.raw['camera'].get('support_height_patch'):
                raise DualArmError('fusion cannot combine unrelated world height corrections')
            if any(c.depth_mode != fusion['zed_mode'] for c in perception.cameras if c.label in perception.active_camera_labels):
                raise DualArmError('fusion depth scale requires the calibrated ZED mode')
            validate_primary(frames[0], fusion)
            secondary = (capture_secondary(fusion, wrist_joints=before[fusion['realsense_mount']['arm']]['joints'])
                         if fusion.get('realsense_mount') else capture_secondary(fusion))
        after = snapshots()
        for key in self.models:
            if np.max(np.abs(np.asarray(before[key]['joints']) - after[key]['joints'])) > self.config.limits['stationary_tolerance_deg']:
                raise DualArmError('arm moved during RGB-D capture; observation discarded')
        if len(frames) != 1 or str(frames[0].serial) != str(self.config.raw['camera']['serial']):
            raise DualArmError('replay requires the configured fixed ZED observation')
        frame = frames[0]
        world_camera = np.array(frame.X_base_camera, copy=True)
        world_camera[:3, 3] *= 1000.
        if not np.allclose(world_camera, self.config.raw['camera']['world_from_camera_mm'], atol=1e-6, rtol=0):
            raise DualArmError('perception and replay camera calibration differ')
        raw_xyz = world_xyz_mm(frame)
        xyz = raw_xyz
        if fusion is not None:
            from .dual_cloud_observation import scale_world
            xyz = scale_world(raw_xyz, world_camera, fusion['depth_scale']['zed'])
        observation = save_observation(directory, frame.rgb, xyz, self.config,
                                {k: v['joints'] for k, v in before.items()}, extra={
            'common_frame': self.profile['common_frame'],
            'world_from_camera_mm': world_camera.tolist(),
            'camera_serial': frame.serial,
            'intrinsics': frame.intrinsics.tolist(),
            'snapshot_before': before, 'snapshot_after': after,
            'reset_zed_at_home': reset_zed,
            'depth_capture_policy': {
                'config': str(perception_path.relative_to(ROOT)),
                'warmup_frames': perception.warmup_frames,
                'temporal_median_frames': perception.temporal_median_frames,
            },
        })
        if fusion is not None:
            from .dual_cloud_observation import attach_secondary
            np.save(observation.directory/'raw_world_xyz_mm.npy', raw_xyz, allow_pickle=False)
            observation.meta['depth_scale'] = fusion['depth_scale']['zed']
            return attach_secondary(observation, secondary, fusion, self.config)
        return observation

    def compile(self, source, observation, *, press_depth_mm=None, post_release_descent_mm=0.):
        result = self._compile_for_arm(source, observation, press_depth_mm=press_depth_mm,
                                       post_release_descent_mm=post_release_descent_mm)
        if not getattr(self, 'workspace_arm_fallback', False) or press_depth_mm is not None or post_release_descent_mm:
            return result
        from .config import SafetyError
        import ast
        def check(candidate):
            config, program, _ = candidate
            for node in ast.walk(ast.parse(program)):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'move':
                    config.validate_workspace_pose(*(ast.literal_eval(a) for a in node.args))
        config, program, assignment = result
        preferred = assignment['active_arm']
        checks = {}
        try:
            check(result)
            checks[preferred] = {'status': 'PASS'}
        except SafetyError as exc:
            checks[preferred] = {'status': 'REJECTED', 'error': str(exc)}
            other = assignment['inactive_arm']
            result = PartitionedReplayRuntime._alternate_candidate(self, source, observation, preferred, other)
            try:
                check(result)
                checks[other] = {'status': 'PASS'}
            except SafetyError as other_error:
                checks[other] = {'status': 'REJECTED', 'error': str(other_error)}
                raise DualArmError(f'Neither arm satisfies workspace: {checks}') from other_error
        result[2].update(preferred_arm=preferred, workspace_candidates=checks,
                         rule='partition_preferred_workspace_fallback',
                         path_collision_validation='required before execution')
        return result

    def _alternate_candidate(self, source, observation, preferred, other):
        """Rebuild the same world points using the other arm's Home attitude."""
        from .dual_arm.geometry import matrix_pose, pose_error
        home = {k: matrix_pose(a.world_from_base @ pose_matrix(self.models[k].forward(a.home_joints)))
                for k, a in self.config.arms.items()}
        calls = parse_code(source, allow_grasp=True)
        from scipy.spatial.transform import Rotation
        rotations = [Rotation.from_euler('xyz', args['right'][3:]) for _, args in calls]
        first = rotations[0]
        if any((r * first.inv()).magnitude() > 1e-6 for r in rotations):
            raise DualArmError('alternate-arm routing requires constant grasp orientation')
        preferred_home = Rotation.from_euler('xyz', home[preferred][3:], degrees=True)
        # Only transfer the supported world-Z grasp rotation, never an arbitrary tilt.
        delta = first * preferred_home.inv()
        if np.linalg.norm(delta.apply([0., 0., 1.]) - [0., 0., 1.]) > 1e-5:
            raise DualArmError('alternate-arm routing cannot transfer non-yaw grasp tilt')
        other_home = Rotation.from_euler('xyz', home[other][3:], degrees=True)
        target_rotation = other_home
        if delta.magnitude() > 1e-6:
            def opening_axis(arm):
                return Rotation.from_euler('xyz', self.config.arms[arm].tcp_offset[3:], degrees=True).inv().apply([0., 1., 0.])
            desired = first.apply(opening_axis(preferred))
            current = other_home.apply(opening_axis(other))
            if min(np.linalg.norm(desired[:2]), np.linalg.norm(current[:2])) < 1e-6:
                raise DualArmError('alternate-arm opening axis has no horizontal direction')
            yaw = np.arctan2(desired[1], desired[0]) - np.arctan2(current[1], current[0])
            yaw = (yaw + np.pi/2) % np.pi - np.pi/2
            target_rotation = Rotation.from_euler('z', yaw) * other_home
        rpy = target_rotation.as_euler('xyz').tolist()
        alternate = '\n'.join(f'motion.{name}(right={list(args["right"][:3])+rpy!r})'
                              for name, args in calls)+'\n'
        return self._compile_for_arm(alternate, observation, active_arm=other)

    def _compile_for_arm(self, source, observation, *, press_depth_mm=None, post_release_descent_mm=0., active_arm=None):
        from .experiments.full_grasp_adapter import build_robot_config
        from .site_observation import convert_pose_mm_deg
        from .dual_arm.geometry import pose_error
        tcp = {k: (a.world_from_base @ pose_matrix(self.models[k].forward(observation.meta['joints_deg'][k])))[:3, 3]
               for k, a in self.config.arms.items()}
        routed, assignment = route_grasp(source, tcp)
        if active_arm is not None:
            if active_arm not in self.config.arms:
                raise DualArmError('unknown active arm')
            routed = '\n'.join(f'motion.{name}({active_arm}={args["right"]!r})'
                               for name,args in parse_code(source,allow_grasp=True))+'\n'
            assignment.update(active_arm=active_arm,inactive_arm='right' if active_arm=='left' else 'left')
        active = assignment['active_arm']
        calls = parse_code(routed, allow_grasp=True)
        if any(name not in {'grasp', 'move_tcp'} for name, _ in calls):
            raise DualArmError('RobotAPI does not reproduce Native move_transfer detours')
        poses = [np.r_[np.array(kw[active][:3])*1000, np.degrees(kw[active][3:])] for _, kw in calls]
        if press_depth_mm is not None:
            if not np.isfinite(press_depth_mm) or not 0 < press_depth_mm <= 15:
                raise DualArmError('press depth must be in (0,15] mm')
            if (len(poses) != 2 or not np.allclose(poses[0][[0,1,3,4,5]],poses[1][[0,1,3,4,5]])
                    or poses[1][2] < poses[0][2]+30):
                raise DualArmError('press requires one vertical descent and retraction')
        poses = [convert_pose_mm_deg(self.profile, p, 'common', active) for p in poses]
        state = {'arms': {k: {'pose_base_mm_deg': self.models[k].forward(a.home_joints).tolist()}
                          for k, a in self.config.arms.items()}}
        config = build_robot_config(self.config.raw, state, active, poses[0][3], poses[0][4], config_path=self.config_path)
        # Wrist-camera observations are captured from fixed perception poses.
        # Return the selected arm through a fully checked Home path before any
        # Cartesian task motion, so grasp planning always starts at Home.
        lines = ['def run():', '    home()',
                 '    close_gripper()' if press_depth_mm is not None else '    open_gripper()']
        def move(pose):
            yaw = config.relative_yaw_from_absolute_deg(float(pose[5]))
            represented = [*pose[:3], config.orientation_roll_deg, config.orientation_pitch_deg, config.command_yaw_deg(yaw)]
            if pose_error(pose, represented)[1] > 1e-6:
                raise DualArmError('RobotAPI fixed roll/pitch cannot represent this graph action')
            args = [float(v) for v in pose[:3]] + [yaw]
            lines.append('    move(' + ', '.join(repr(v) for v in args) + ')')
        up = self.config.arms[active].world_from_base[:3, :3].T @ np.array([0., 0., 1.])
        hover = poses[0].copy(); hover[:3] += self.task_policy['approach_clearance_mm'] * up
        # Apply the site contact offset once, along common/world Z. Keep the
        # graph's approach, transport and release targets unchanged.
        offset = -float(press_depth_mm) if press_depth_mm is not None else float(self.task_policy.get('grasp_surface_offset_mm', 0.))
        if not np.isfinite(offset) or not -30. <= offset <= 10.:
            raise DualArmError('invalid grasp surface offset')
        # Far-table correction is shared by both arms in the common world frame.
        # Reset (post-release descent) and pressing retain their existing policy.
        far_y = self.task_policy.get('far_grasp_world_y_max_mm')
        if press_depth_mm is None and not post_release_descent_mm and far_y is not None:
            raise_mm = float(self.task_policy.get('far_grasp_raise_mm', 0.))
            if not np.isfinite(far_y) or not np.isfinite(raise_mm) or not 0 <= raise_mm <= 2.:
                raise DualArmError('invalid far-table grasp correction')
            applied = raise_mm if assignment['grasp_world_mm'][1] <= far_y else 0.
            offset += applied
            assignment['far_grasp_height_correction'] = dict(
                world_y_max_mm=far_y, applied_raise_mm=applied)
        requested_offset = offset
        if press_depth_mm is None and offset < 0 and config.boundaries.z_min is not None:
            floor = float(config.boundaries.z_min + config.lower_z_margin_mm)
            if up[2] <= 0:
                raise DualArmError('world vertical descent is incompatible with base Z floor')
            available = (float(poses[0][2]) - floor) / float(up[2])
            if available < 0:
                raise DualArmError('measured cloth surface is below configured minimum TCP height')
            offset = -min(-offset, available)
            assignment['grasp_height_limit'] = dict(
                minimum_base_z_mm=floor, requested_descent_mm=-requested_offset,
                applied_descent_mm=-offset, limited=offset != requested_offset,
                policy='min(requested_descent, distance_to_configured_z_floor)')
        contact = poses[0].copy(); contact[:3] += offset * up
        assignment['grasp_surface_offset_mm'] = offset
        assignment['contact_base_mm_deg'] = contact.tolist()
        move(hover); move(contact)
        if press_depth_mm is None:lines.append('    close_gripper()')
        else:assignment['motion_kind']='single_vertical_press'
        for pose in poses[1:]: move(pose)
        lines.append('    open_gripper()')
        if not np.isfinite(post_release_descent_mm) or not 0 <= post_release_descent_mm <= 100:
            raise DualArmError('invalid post-release descent')
        if press_depth_mm is not None and post_release_descent_mm:
            raise DualArmError('post-release descent is only for grasp/drop actions')
        retreat = poses[-1].copy()
        clearance = -post_release_descent_mm if post_release_descent_mm else self.task_policy['retreat_clearance_mm']
        retreat[:3] += clearance * up
        assignment['post_release_descent_mm'] = post_release_descent_mm
        move(retreat); lines.append('    home()')
        return config, '\n'.join(lines) + '\n', assignment

    def execute(self, source, observation, folder, *, press_depth_mm=None, post_release_descent_mm=0.):
        if self.planning_only:
            raise DualArmError('planning-only adapter cannot execute motion')
        if not self._routing_lock.acquire(blocking=False):
            raise DualArmError('a partitioned replay action is already running')
        path_checks = {}
        try:
            from .experiment import ExperimentRunner
            from .robot_collision import RobotCollision
            observation.validate_for(self.config, live=True, check_age=False)
            config, program, assignment = (self.compile(source, observation) if press_depth_mm is None and not post_release_descent_mm
                                           else self.compile(source, observation, press_depth_mm=press_depth_mm,
                                                             post_release_descent_mm=post_release_descent_mm))
            folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
            # Select using the complete checked trajectory, before constructing any backend.
            for candidate_index in range(2):
                arm = assignment['active_arm']
                candidate_folder = folder if candidate_index == 0 else folder / 'alternate_arm'
                collision = RobotCollision(self.config_path, observation.meta['joints_deg'], arm, root=self.config.root)
                runner = ExperimentRunner(candidate_folder / 'experiment', config, collision_checker=collision)
                path = runner.workspace / 'graph_action.py'; path.write_text(program)
                try:
                    preview = runner.preflight(path)
                    if preview.error:
                        raise ValueError(preview.error)
                    collision.require_execution(config, preview.actions)
                except (ValueError, RuntimeError) as exc:
                    path_checks[arm] = dict(status='REJECTED', error=f'{type(exc).__name__}: {exc}')
                    if hasattr(exc, 'planning_failure'):
                        path_checks[arm]['planning_failure'] = exc.planning_failure
                    (folder / 'arm_path_checks.json').write_text(json.dumps(path_checks, indent=2))
                    if candidate_index or not getattr(self, 'workspace_arm_fallback', False) or press_depth_mm is not None or post_release_descent_mm:
                        raise
                    other = assignment['inactive_arm']
                    try:
                        config, program, assignment = self._alternate_candidate(source, observation, arm, other)
                    except (ValueError, RuntimeError) as alternate_error:
                        path_checks[other] = dict(status='REJECTED', stage='candidate_generation',
                            error=f'{type(alternate_error).__name__}: {alternate_error}')
                        (folder / 'arm_path_checks.json').write_text(json.dumps(path_checks, indent=2))
                        raise
                    assignment.update(preferred_arm=arm, rule='partition_preferred_complete_path_fallback')
                    continue
                path_checks[arm] = dict(status='PASS')
                break
            assignment['complete_path_candidates'] = path_checks
            (folder / 'arm_path_checks.json').write_text(json.dumps(path_checks, indent=2))
            (folder / 'arm_assignment.json').write_text(json.dumps(assignment, indent=2)+'\n')
            (folder / 'routed_motion.py').write_text(program)
            result = runner.run_experiment(path, real=True, confirmed=True)
            return dict(status='EXECUTED' if result['execution_completed'] else 'EXECUTION_FAILED',
                        physical_execution=result['physical_execution'], arm_assignment=assignment, experiment=result,
                        error=None if result['execution_completed'] else
                            ('; '.join(result.get('robot_errors', [])) or result.get('notes') or 'robot execution did not complete'))
        except (ValueError, RuntimeError) as exc:
            result = dict(status='PLANNING_FAILED', physical_execution=False, error=f'{type(exc).__name__}: {exc}')
            if hasattr(exc, 'planning_failure'):
                result['planning_failure'] = exc.planning_failure
            if path_checks and all(check['status'] == 'REJECTED' for check in path_checks.values()):
                result['arm_path_checks'] = path_checks
                result['planning_failure'] = dict(
                    kind='ARM_CANDIDATES_REJECTED',
                    stage='complete_pre_execution_path_planning',
                    arm_candidates=path_checks, physical_execution=False,
                    explanation='Recorded candidates failed at their reported stages; candidate generation failure is not an IK or collision verdict. Failed IK search does not prove unreachability.')
                result['error'] = 'No accepted arm path: ' + '; '.join(
                    f'{arm}: {check["error"]}' for arm, check in path_checks.items()
                    if check['status'] == 'REJECTED')
            return result
        finally:
            self._routing_lock.release()

    def stop(self):
        # Motion lifecycle and exception-stop belong to RobotAPI. This adapter
        # has only short-lived read-only controller connections, never motors.
        pass

    def close(self):
        recorder = getattr(self, 'video_recorder', None)
        if recorder is not None:
            recorder.close(); self.video_recorder = None
        if self._lease is not None:
            self._lease.close(); self._lease = None
