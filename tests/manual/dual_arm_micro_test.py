#!/usr/bin/env python3
"""Preview, record home, or execute a paired 3 mm TCP lift and return.

Home is taught independently for each arm; no shared joint-angle assumption.
No motor enabling, fault clearing, mode changes, or gripper commands are issued.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
from pathlib import Path
import threading
import time


def checked(result, label):
    code, value = result
    if code != 0:
        raise RuntimeError(f'{label}: controller code {code}')
    return value


def vector(values, size, label):
    if len(values) != size or not all(math.isfinite(float(x)) for x in values):
        raise ValueError(f'{label}: expected {size} finite numbers')
    return [float(x) for x in values]


def mirror_facing_home(pose):
    """Mirror identical tool frames for level, opposing (+X facing) bases.

    In the two local base frames p_B = diag(1,-1,1) p_A.
    R_B = diag(1,-1,1) R_A diag(1,-1,1): reflect the tool's Y
    direction to preserve a right-handed frame. Requires equivalently mounted
    symmetric grippers and calibrated TCPs; this does not plan a collision-free
    transition from the current posture.
    """
    x, y, z, roll, pitch, yaw = vector(pose, 6, 'arm6 home')
    return [x, -y, z, -roll, pitch, -yaw]


def validate(doc):
    arms = doc['arms']
    if len(arms) != 2 or len({a['ip'] for a in arms}) != 2:
        raise ValueError('Exactly two distinct controller IPs are required')
    for a in arms:
        if a['axis'] not in (6, 7):
            raise ValueError('Expected a 6- or 7-axis controller')
        for key, size in [('home', 6), ('tcp_offset', 6), ('joints', a['axis'])]:
            vector(a[key], size, key)
        if not a['control_box_sn']:
            raise ValueError('Missing control box serial number')
    return arms


def poses_close(actual, expected, mm=0.5, deg=0.5):
    actual = vector(actual, 6, 'actual TCP')
    expected = vector(expected, 6, 'expected TCP')
    # Conservative Euler comparison: equivalent singular representations may fail.
    return (math.dist(actual[:3], expected[:3]) <= mm and
            all(abs((a-b+180) % 360-180) <= deg for a, b in zip(actual[3:], expected[3:])))


def ready(arm):
    state = checked(arm.get_state(), 'get_state')
    errors = checked(arm.get_err_warn_code(), 'get_err_warn_code')
    if state not in (0, 2) or arm.mode != 0 or list(errors) != [0, 0]:
        raise RuntimeError(f'Not ready: state={state}, mode={arm.mode}, errors={errors}')
    if (len(arm.motor_enable_states) < arm.axis or len(arm.motor_brake_states) < arm.axis
            or not all(arm.motor_enable_states[:arm.axis])
            or not all(arm.motor_brake_states[:arm.axis])):
        raise RuntimeError('All motors must already be enabled with brakes released')


def snapshot(arm, ip):
    checked(arm.get_robot_sn(), 'get_robot_sn')
    return dict(ip=ip, axis=int(arm.axis), control_box_sn=arm.control_box_sn,
                home=checked(arm.get_position(is_radian=False), 'get_position'),
                joints=checked(arm.get_servo_angle(is_radian=False), 'get_servo_angle')[:arm.axis],
                tcp_offset=list(arm.tcp_offset))


def preflight(arms, configs, delta):
    for arm, cfg in zip(arms, configs):
        ready(arm)
        current = snapshot(arm, cfg['ip'])
        vector(current['joints'], cfg['axis'], 'current joints')
        if current['axis'] != cfg['axis'] or current['control_box_sn'] != cfg['control_box_sn']:
            raise RuntimeError('Controller identity changed; record new home')
        if not poses_close(current['tcp_offset'], cfg['tcp_offset'], mm=0.1, deg=0.1):
            raise RuntimeError('TCP offset changed; record new home')
        if not poses_close(current['home'], cfg['home']):
            raise RuntimeError('Arm must start at its taught home (within 0.5 mm / 0.5 deg)')
        if max(abs(a-b) for a, b in zip(current['joints'], cfg['joints'])) > 1:
            raise RuntimeError('Joint configuration differs from taught home')
        # Sample the short straight path in each independent base frame.
        for step in range(6):
            pose = list(cfg['home']); pose[2] += delta * step / 5
            if checked(arm.is_tcp_limit(pose, is_radian=False), 'TCP limits') != False:
                raise RuntimeError('TCP outside controller limits')
            joints = checked(arm.get_inverse_kinematics(
                pose, input_is_radian=False, return_is_radian=False), 'IK')
            vector(joints[:cfg['axis']], cfg['axis'], 'IK joints')
            if max(abs(a-b) for a,b in zip(joints, cfg['joints'])) > 5:
                raise RuntimeError('IK changes a joint by more than 5 degrees')
            if checked(arm.is_joint_limit(joints, is_radian=False), 'joint limits') != False:
                raise RuntimeError('IK outside joint limits')


def stop_both(arms):
    # Never use SDK emergency_stop(), which may re-enable motion afterward.
    def stop(arm):
        try:
            code = arm.set_state(4)
            if code != 0: print(f'STOP FAILED: code {code}', flush=True)
        except Exception as exc:
            print(f'STOP FAILED: {exc}', flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(stop, arms))


def phase(arms, targets, speed):
    barrier = threading.Barrier(2)
    starts = [None, None]
    cancelled = threading.Event()
    def move(i):
        barrier.wait(timeout=5)
        if cancelled.is_set():
            raise RuntimeError('Paired motion cancelled')
        starts[i] = time.monotonic()
        code = arms[i].set_position(*targets[i], speed=speed, mvacc=5,
                                   is_radian=False, relative=False, radius=-1,
                                   wait=True, timeout=20, motion_type=0)
        if code != 0: raise RuntimeError(f'Arm {i}: motion failed ({code})')
        ready(arms[i])
        actual = checked(arms[i].get_position(is_radian=False), 'verify TCP')
        if not poses_close(actual, targets[i]):
            raise RuntimeError(f'Arm {i}: target not reached')
    pool = ThreadPoolExecutor(max_workers=2)
    try:
        jobs = [pool.submit(move, i) for i in range(2)]
        for job in as_completed(jobs): job.result()
    except BaseException:
        cancelled.set()
        barrier.abort()
        stop_both(arms)
        raise
    finally:
        pool.shutdown(wait=True)
    print(f'Host dispatch skew: {abs(starts[0]-starts[1])*1000:.2f} ms', flush=True)


def execute(arms, configs, delta, speed):
    # Complete checks on BOTH arms before submitting ANY movement.
    preflight(arms, configs, delta)
    targets = [list(c['home']) for c in configs]
    for p in targets: p[2] += delta
    try:
        print('Paired lift', flush=True)
        phase(arms, targets, speed)
        for arm in arms: ready(arm)
        print('Paired return to taught home', flush=True)
        phase(arms, [c['home'] for c in configs], speed)
    except BaseException:
        stop_both(arms)
        raise  # Never attempt automatic home after a failed phase.


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--home', type=Path, default=Path('data/robot/dual_arm_home.json'))
    group = p.add_mutually_exclusive_group()
    group.add_argument('--capture-home', action='store_true', help='Record manually taught symmetric poses; no motion')
    group.add_argument('--execute', action='store_true', help='Execute the previewed test on hardware')
    p.add_argument('--ips', nargs=2, default=['192.168.2.232', '192.168.1.195'])
    p.add_argument('--delta-mm', type=float, default=3)
    p.add_argument('--speed-mm-s', type=float, default=2)
    args = p.parse_args()
    if not 0 < args.delta_mm <= 5 or not 0 < args.speed_mm_s <= 5:
        p.error('delta and speed must be finite, positive, and at most 5')
    configs = None
    if not args.capture_home:
        configs = validate(json.loads(args.home.read_text()))
        print(json.dumps({'arms': configs, 'lift_base_z_mm': args.delta_mm,
                          'speed_mm_s': args.speed_mm_s, 'return': 'each taught home'}, indent=2))
        if not args.execute:
            print('PREVIEW ONLY. No controller connection or motion.'); return
    elif len(set(args.ips)) != 2 or args.home.exists():
        p.error('Use two distinct IPs and a new home file (existing files are not overwritten)')
    from xarm.wrapper import XArmAPI
    arms = []
    try:
        for ip in args.ips if args.capture_home else [c['ip'] for c in configs]:
            arm = XArmAPI(ip, is_radian=False)
            arms.append(arm)
            if not arm.connected: raise RuntimeError(f'Cannot connect: {ip}')
        if args.capture_home:
            first = [snapshot(a, ip) for a, ip in zip(arms, args.ips)]
            time.sleep(0.5)
            second = [snapshot(a, ip) for a, ip in zip(arms, args.ips)]
            if any(not poses_close(a['home'], b['home'], .1, .1) or
                   max(abs(x-y) for x,y in zip(a['joints'], b['joints'])) > .1
                   for a,b in zip(first, second)):
                raise RuntimeError('Arms must be stationary while capturing home')
            doc = {'schema': 1, 'home_method': 'independently taught symmetric TCP poses', 'arms': second}
            validate(doc)
            args.home.parent.mkdir(parents=True, exist_ok=True)
            with args.home.open('x') as f: json.dump(doc, f, indent=2)
            print(f'Home saved: {args.home}')
        else:
            execute(arms, configs, args.delta_mm, args.speed_mm_s)
            print('Both TCPs returned to taught home.')
    finally:
        for arm in arms: arm.disconnect()


if __name__ == '__main__':
    main()
