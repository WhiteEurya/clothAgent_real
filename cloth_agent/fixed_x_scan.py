"""Operator-triggered elevated approach followed by a Y sweep at X=420 mm."""
from dataclasses import asdict, dataclass, replace
import json
import math
import time

from .robot_api import _controller_trajectory_with_arm


@dataclass(frozen=True)
class ScanSettings:
    y_start: float
    y_end: float
    z: float
    speed: float
    x_tolerance: float = .5

    def validate(self, config):
        if not all(math.isfinite(v) for v in asdict(self).values()):
            raise ValueError('Scan parameters must be finite')
        if self.y_start == self.y_end:
            raise ValueError('Scan Y start and end must differ')
        if not 0 < self.speed <= min(config.speed_mm_s, config.MAX_SAFE_SPEED_MM_S):
            raise ValueError(f'Scan speed must be positive and <= configured {config.speed_mm_s} mm/s')
        if not 0 < self.x_tolerance <= .5:
            raise ValueError('X tolerance must be >0 and <=0.5 mm')


def require_ready(arm, moving=False):
    code, state = arm.get_state()
    if code != 0 or state not in ((0, 1, 2) if moving else (0, 2)) or arm.mode != 0:
        raise RuntimeError('Select position mode and ready state in vendor controls before scanning')
    code, errors = arm.get_err_warn_code()
    if code != 0 or errors[0] != 0:
        raise RuntimeError('Controller error or unreadable status')


def require_start(pose, settings):
    if (abs(pose[0]-420.) > settings.x_tolerance
            or abs(pose[1]-settings.y_start) > .5 or abs(pose[2]-settings.z) > .5):
        raise RuntimeError(f'Manually position TCP at X=420, Y={settings.y_start}, Z={settings.z} first; actual={pose[:3]}')


def prepare(arm, config, settings, read_pose):
    settings.validate(config)
    require_ready(arm)
    if arm.get_is_moving():
        raise RuntimeError('Robot must be stopped at the start point')
    pose = read_pose(arm, config)
    require_start(pose, settings)
    code, joints = arm.get_servo_angle(is_radian=False)
    if code != 0 or len(joints) != 7 or not all(math.isfinite(v) for v in joints):
        raise RuntimeError('Invalid start joints')
    # Existing IK checker starts at config.init_joints. Substitute measured
    # starting joints, not saved Home. No home command is ever executed here.
    local = replace(config, init_joints_deg=tuple(joints), init_pose_mm_deg=tuple(pose),
                    orientation_roll_deg=pose[3], orientation_pitch_deg=pose[4])
    action = {'name': 'move', 'args': {'x': 420., 'y': settings.y_end, 'z': settings.z,
                                    'yaw': local.relative_yaw_from_absolute_deg(pose[5])}}
    validation = _controller_trajectory_with_arm(arm, local, [action])
    fresh = read_pose(arm, config)
    require_start(fresh, settings)
    if (math.dist(pose[:3], fresh[:3]) > .2
            or max(abs((a-b+180)%360-180) for a, b in zip(pose[3:], fresh[3:])) > .2
            or arm.get_is_moving()):
        raise RuntimeError('Robot moved during preflight; scan not started')
    require_ready(arm)
    return pose, validation


def approach_start(arm, config, settings, stop, report, directory, poll_sample,
                   projection, latest, publish, read_pose):
    settings.validate(config)
    require_ready(arm)
    if arm.get_is_moving():
        raise RuntimeError('Robot must be stationary before automatic approach')
    pose = list(read_pose(arm, config))
    try:
        require_start(pose, settings)
        return
    except RuntimeError:
        pass
    code, joints = arm.get_servo_angle(is_radian=False)
    if code != 0 or len(joints) != 7 or not all(math.isfinite(v) for v in joints):
        raise RuntimeError('Invalid approach joints')
    local = replace(config, init_joints_deg=tuple(joints), init_pose_mm_deg=tuple(pose),
                    orientation_roll_deg=pose[3], orientation_pitch_deg=pose[4])
    # Raise if necessary, translate above the start, then descend vertically.
    height = max(pose[2], config.init_pose_mm_deg[2], settings.z + 80.)
    points = [(pose[0], pose[1], height), (420., settings.y_start, height),
              (420., settings.y_start, settings.z)]
    yaw = local.relative_yaw_from_absolute_deg(pose[5])
    actions = [{'name': 'move', 'args': dict(x=x, y=y, z=z, yaw=yaw)}
               for x, y, z in points + [(420., settings.y_end, settings.z)]]
    report['approach_ik'] = asdict(_controller_trajectory_with_arm(arm, local, actions))
    fresh = read_pose(arm, config)
    if (math.dist(pose[:3], fresh[:3]) > .2
            or max(abs((a-b+180)%360-180) for a, b in zip(pose[3:], fresh[3:])) > .2
            or arm.get_is_moving()):
        raise RuntimeError('Robot moved during approach preflight')
    require_ready(arm)
    report.update(status='APPROACHING', approach_waypoints_mm=points,
                  approach_orientation_deg=pose[3:])
    (directory / 'scan.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    armed = False
    previous = pose[:3]
    try:
        with (directory / 'approach_samples.jsonl').open('w', encoding='utf-8') as log:
            for target in points:
                if math.dist(previous, target) <= .2:
                    continue
                if stop.is_set():
                    raise RuntimeError('Approach cancelled')
                armed = True
                started = time.monotonic()
                duration = math.dist(previous, target)/settings.speed * 2 + 15
                code = arm.set_position(x=target[0], y=target[1], z=target[2],
                                        roll=pose[3], pitch=pose[4], yaw=pose[5],
                                        speed=settings.speed, mvacc=config.acceleration_mm_s2,
                                        radius=-1, wait=False, is_radian=False)
                if code != 0:
                    raise RuntimeError(f'Approach command failed: {code}')
                while True:
                    if stop.is_set():
                        raise RuntimeError('Operator cancelled approach')
                    sample = poll_sample(arm, config, projection)
                    sample['scan_status'] = 'APPROACHING'
                    log.write(json.dumps(sample, allow_nan=False)+'\n')
                    log.flush()
                    publish(latest, sample)
                    if sample['status'] != 'OK' or sample['read_duration_s'] > .5:
                        raise RuntimeError('Approach feedback failed or took >0.5 s')
                    require_ready(arm, moving=True)
                    xyz = sample['tcp_pose_mm_deg'][:3]
                    # Stay within a narrow tube around the validated segment.
                    delta = [b-a for a, b in zip(previous, target)]
                    length2 = sum(v*v for v in delta)
                    t = max(0., min(1., sum((v-a)*d for v, a, d in zip(xyz, previous, delta))/length2))
                    if math.dist(xyz, [a+t*d for a, d in zip(previous, delta)]) > 1.:
                        raise RuntimeError('Approach left validated segment by >1 mm')
                    if math.dist(xyz, target) <= .2 and not arm.get_is_moving():
                        break
                    if time.monotonic()-started > duration:
                        raise RuntimeError('Approach timed out')
                    stop.wait(.05)
                previous = list(target)
        require_start(read_pose(arm, config), settings)
        armed = False
    finally:
        if armed:
            report['approach_stop_return_code'] = arm.set_state(4)


def run_scan(arm, config, settings, projection, latest, stop, directory, poll_sample, publish, read_pose):
    report = {'status': 'PREFLIGHT', 'fixed_base_x_mm': 420., 'settings': asdict(settings),
              'profile': 'single Cartesian segment; acceleration/deceleration at endpoints', 'samples': 0}
    armed = False
    try:
        approach_start(arm, config, settings, stop, report, directory, poll_sample,
                       projection, latest, publish, read_pose)
        pose, validation = prepare(arm, config, settings, read_pose)
        report['controller_ik'] = asdict(validation)
        report['start_pose'] = pose
        command = dict(x=420., y=settings.y_end, z=settings.z,
                       roll=pose[3], pitch=pose[4], yaw=pose[5], speed=settings.speed,
                       mvacc=config.acceleration_mm_s2, radius=-1, wait=False, is_radian=False)
        report['command'] = command
        # Persist before motion; logging must be available for the whole sweep.
        report_path = directory / 'scan.json'
        report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
        with (directory / 'scan_samples.jsonl').open('w', encoding='utf-8') as log:
            if stop.is_set():
                raise RuntimeError('Scan cancelled before motion')
            armed = True  # Even a failed command may have reached the controller.
            started = time.monotonic()
            code = arm.set_position(**command)
            if code != 0:
                raise RuntimeError(f'Scan command failed: {code}')
            report['status'] = 'SCANNING'
            duration_limit = abs(settings.y_end-settings.y_start)/settings.speed * 2 + 15
            previous = None
            max_x_error = 0.
            while True:
                if stop.is_set():
                    raise RuntimeError('Operator cancelled scan')
                sample = poll_sample(arm, config, projection)
                sample['scan_status'] = 'SCANNING'
                sample['scan_elapsed_s'] = time.monotonic()-started
                if sample['status'] == 'OK':
                    xyz = sample['tcp_pose_mm_deg'][:3]
                    sample['x_error_mm'] = xyz[0]-420.
                    sample['z_error_mm'] = xyz[2]-settings.z
                    max_x_error = max(max_x_error, abs(sample['x_error_mm']))
                    if previous is not None:
                        dt = sample['sample_monotonic']-previous['sample_monotonic']
                        if dt > 0:
                            sample['measured_y_speed_mm_s'] = (xyz[1]-previous['tcp_pose_mm_deg'][1])/dt
                    previous = sample
                log.write(json.dumps(sample, allow_nan=False)+'\n')
                log.flush()
                publish(latest, sample)
                report.update(samples=report['samples']+1, max_abs_x_error_mm=max_x_error)
                if sample['status'] != 'OK' or sample['read_duration_s'] > .5:
                    raise RuntimeError('Scan feedback failed or took >0.5 s')
                if abs(sample['x_error_mm']) > settings.x_tolerance:
                    raise RuntimeError(f'X tracking exceeded tolerance: {sample["x_error_mm"]:.3f} mm')
                if abs(sample['z_error_mm']) > .5:
                    raise RuntimeError(f'Z tracking exceeded tolerance: {sample["z_error_mm"]:.3f} mm')
                require_ready(arm, moving=True)
                y = sample['tcp_pose_mm_deg'][1]
                if not min(settings.y_start, settings.y_end)-.5 <= y <= max(settings.y_start, settings.y_end)+.5:
                    raise RuntimeError('Y left the scan segment')
                if abs(y-settings.y_end) <= .2 and not arm.get_is_moving():
                    report.update(status='COMPLETE', final_pose=sample['tcp_pose_mm_deg'])
                    armed = False
                    break
                if sample['scan_elapsed_s'] > duration_limit:
                    raise RuntimeError('Scan timed out')
                stop.wait(.05)
    except Exception as exc:
        report.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
    finally:
        if armed:
            try:
                report['stop_return_code'] = arm.set_state(4)
            except Exception as exc:
                report['stop_error'] = str(exc)
        (directory / 'scan.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        print(f'Scan {report["status"]}: {report.get("error", "held at endpoint")} | {directory / "scan.json"}', flush=True)
    return report


def scan_monitor(arm, config, projection, latest, stop, begin, settings, directory, poll_sample, publish, read_pose):
    attempted = False
    scan_result = 'WAITING_FOR_B'
    with (directory / 'tcp_samples.jsonl').open('a', encoding='utf-8') as log:
        while not stop.is_set():
            if begin.is_set() and not attempted:
                attempted = True
                report = run_scan(arm, config, settings, projection, latest, stop, directory, poll_sample, publish, read_pose)
                scan_result = f'{report["status"]}: {report.get("error", "held at endpoint")}'
            sample = poll_sample(arm, config, projection)
            sample['scan_status'] = scan_result
            log.write(json.dumps(sample, allow_nan=False)+'\n')
            log.flush()
            publish(latest, sample)
            stop.wait(.1)
