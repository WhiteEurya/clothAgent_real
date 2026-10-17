"""Linux gamepad input and continuous Arm7 control (preview by default)."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
import threading
import time


DEFAULT_YAW_STEP_DEG = 2.
MAX_YAW_STEP_DEG = 5.


@dataclass(frozen=True)
class Intent:
    forward: float = 0.
    right: float = 0.
    up: float = 0.
    yaw: float = 0.
    gripper: str | None = None
    enabled: bool = False
    stop: bool = False

    @property
    def moving(self):
        return any((self.forward, self.right, self.up, self.yaw))


def normalize(value, minimum, maximum, deadzone=.15):
    if not all(math.isfinite(x) for x in (value, minimum, maximum, deadzone)):
        raise ValueError('non-finite gamepad axis')
    if maximum <= minimum or not 0 <= deadzone < 1:
        raise ValueError('invalid axis range/deadzone')
    v = max(-1., min(1., 2 * (value - minimum) / (maximum - minimum) - 1))
    return math.copysign(max(0., (abs(v) - deadzone) / (1 - deadzone)), v)


class Mapper:
    """Semantic Xbox-layout names; require neutral controls before enabling."""
    def __init__(self):
        self.ready = False
        self.previous = set()
        self.blocked_gripper = set()

    def update(self, lx, ly, rx, buttons):
        buttons = set(buttons)
        if not all(math.isfinite(v) and -1 <= v <= 1 for v in (lx, ly, rx)):
            raise ValueError('axes must be normalized to [-1, 1]')
        if not buttons and not any((lx, ly, rx)):
            self.ready = True
        enabled = self.ready and 'LB' in buttons and 'START' not in buttons
        gripper = None
        self.blocked_gripper.intersection_update(buttons)
        if not enabled:
            self.blocked_gripper.update(buttons & {'X', 'B'})
        # A button held before enabling must first be released. Once enabled,
        # emit a held intent every frame, rather than one full-travel command.
        if enabled and not {'X', 'B'} <= buttons:
            if 'X' in buttons - self.blocked_gripper:
                gripper = 'open'
            elif 'B' in buttons - self.blocked_gripper:
                gripper = 'close'
        self.previous = buttons
        return Intent(-ly if enabled else 0., lx if enabled else 0.,
                      float(('Y' in buttons) - ('A' in buttons)) if enabled else 0.,
                      -rx if enabled else 0., gripper, enabled, 'START' in buttons)


class Gamepad:
    """Poll kernel state, rather than replaying a backlog of input events."""
    def __init__(self, path, deadzone=.15, yaw_axis='ABS_RX', profile=None):
        from evdev import InputDevice, ecodes
        self.device = InputDevice(path)
        self.deadzone = deadzone
        self.axes = (ecodes.ABS_X, ecodes.ABS_Y, getattr(ecodes, yaw_axis))
        self.signs = (1, 1, 1)
        # Xbox-compatible reports use BTN_X (307) / BTN_Y (308). The
        # WEST/NORTH aliases do not match the printed X/Y on this controller.
        self.keys = {ecodes.BTN_SOUTH: 'A', ecodes.BTN_EAST: 'B',
                     ecodes.BTN_X: 'X', ecodes.BTN_Y: 'Y',
                     ecodes.BTN_TL: 'LB', ecodes.BTN_START: 'START'}
        try:
            from .gamepad_profile import load_profile, validate_profile
            profile = profile if profile is not None else load_profile(self.device)
            if profile is not None:
                validate_profile(profile, self.device)
                self.axes = tuple(item['code'] for item in profile['axes'])
                self.signs = tuple(item['sign'] for item in profile['axes'])
                self.keys = {code: name for name, code in profile['buttons'].items()}
            caps = self.device.capabilities(absinfo=False)
            if not set(self.axes) <= set(caps.get(ecodes.EV_ABS, [])):
                raise ValueError('device lacks selected stick axes; run bash scripts/calibrate_gamepad.sh')
            if not set(self.keys) <= set(caps.get(ecodes.EV_KEY, [])):
                raise ValueError('device needs button mapping; run bash scripts/calibrate_gamepad.sh')
        except BaseException:
            self.device.close()
            raise

    def sample(self):
        values = []
        for code, sign in zip(self.axes, self.signs):
            info = self.device.absinfo(code)
            values.append(sign * normalize(info.value, info.min, info.max, self.deadzone))
        keys = self.device.active_keys()
        return (*values, {name for code, name in self.keys.items() if code in keys})

    def close(self):
        self.device.close()


def list_gamepads():
    from evdev import InputDevice, list_devices, ecodes
    rows = []
    for path in list_devices():
        dev = InputDevice(path)
        try:
            caps = dev.capabilities(absinfo=False)
            keys = caps.get(ecodes.EV_KEY, [])
            if ecodes.BTN_GAMEPAD in keys or (ecodes.BTN_JOYSTICK in keys
                    and {ecodes.ABS_X, ecodes.ABS_Y} <= set(caps.get(ecodes.EV_ABS, []))):
                rows.append(dict(path=path, name=dev.name, vendor=hex(dev.info.vendor),
                                 product=hex(dev.info.product), capabilities=caps))
        finally:
            dev.close()
    return rows


def control_status(ready, intent, busy=False):
    if not ready:
        return '等待初始化：请松开全部按键（包括 LB），并让两个摇杆回中。'
    if busy:
        return '正在规划或执行本次小步；结果将显示在终端。'
    if not intent.enabled:
        return '手柄已就绪：请按住左肩键 LB（不是 LT），再操作摇杆或 ABXY。'
    if intent.moving or intent.gripper:
        return '连续控制中：松开摇杆停止平移/旋转，松开 X/B 停止夹爪，松开 LB 停止两者。'
    return 'LB 已按住：左摇杆平移，右摇杆 yaw；Y/A 升降，X/B 开合。'


def motion_source(intent, world_pose, step_mm=1., yaw_step_deg=DEFAULT_YAW_STEP_DEG):
    """World +X forward, -Y right, +Z up; positive yaw about world +Z."""
    import numpy as np
    from .dual_arm.geometry import matrix_pose, pose_matrix
    from scipy.spatial.transform import Rotation
    if not (0 < step_mm <= 2 and 0 < yaw_step_deg <= MAX_YAW_STEP_DEG):
        raise ValueError(f'jog step must be (0,2] mm / (0,{MAX_YAW_STEP_DEG:g}] degrees')
    if not intent.enabled or intent.stop:
        return None
    if intent.gripper:
        if intent.gripper not in ('open', 'close'):
            raise ValueError('invalid gripper command')
        return f'motion.gripper(arm="right", action="{intent.gripper}")\n'
    if not intent.moving:
        return None
    delta = np.array([intent.forward, -intent.right, intent.up], dtype=float)
    delta /= max(1., float(np.linalg.norm(delta)))
    frame = pose_matrix(world_pose)
    frame[:3, 3] += delta * step_mm
    frame[:3, :3] = Rotation.from_euler('z', intent.yaw * yaw_step_deg, degrees=True).as_matrix() @ frame[:3, :3]
    pose = matrix_pose(frame)
    target = np.r_[pose[:3] / 1000, np.radians(pose[3:])]
    return f'motion.move_tcp(right={target.tolist()})\n'


class InputGuard:
    """Shared latest input, not a queue of movement commands."""
    def __init__(self):
        self.lock = threading.Lock()
        self.intent = Intent()
        self.updated = 0.
        self.generation = 0

    def update(self, intent):
        with self.lock:
            # A release/repress must invalidate an older pending action.
            if not intent.enabled or intent.stop:
                self.generation += 1
            self.intent, self.updated = intent, time.monotonic()

    def token(self):
        with self.lock:
            return self.generation

    def snapshot(self):
        with self.lock:
            return self.intent, self.updated, self.generation

    def permits(self, candidate=None, generation=None):
        with self.lock:
            live = self.intent
            if (time.monotonic() - self.updated > .25 or not live.enabled or live.stop
                    or (generation is not None and self.generation != generation)):
                return False
            if candidate is None or candidate.gripper:
                return True
            return all(abs(getattr(live, k) - getattr(candidate, k)) < .15
                       for k in ('forward', 'right', 'up', 'yaw')) and live.moving


def run(args):
    rows = list_gamepads()
    if args.list:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0
    path = args.device
    if path is None:
        if len(rows) != 1:
            raise RuntimeError(f'found {len(rows)} accessible gamepads; connect the USB controller, then use --list / --device')
        path = rows[0]['path']
    pad = Gamepad(path, args.deadzone, args.yaw_axis)
    backend = None
    mapper, guard = Mapper(), InputGuard()
    try:
        print(f'{pad.device.name}: Y up / A down / X open / B close; hold LB; START exits', file=sys.stderr)
        if args.real:
            from .gamepad_continuous import ContinuousArm7
            backend = ContinuousArm7(args.config, args.output, guard,
                                     args.speed_mm_s, args.yaw_speed_deg_s, args.gripper_rate,
                                     start_workers=not args.connect_only)
            if args.connect_only:
                print('连接检查通过：仅连接 Arm7，未切换运动模式，未发送运动或夹爪指令。', flush=True)
                return 0
        previous = None
        previous_status = None
        while True:
            raw = pad.sample()
            intent = mapper.update(*raw)
            guard.update(intent)
            if intent.stop:
                break
            if backend is None:
                record = dict(intent=asdict(intent), ready=mapper.ready,
                              axes=dict(zip(('left_x', 'left_y', 'right_x'), raw[:3])),
                              buttons=sorted(raw[3]))
                if record != previous:
                    print(json.dumps(record, ensure_ascii=False), flush=True)
                    previous = record
            else:
                backend.check_health()
                status = control_status(mapper.ready, intent)
                status = backend.control_status(status)
                if status != previous_status:
                    print(status, flush=True)
                    previous_status = status
            time.sleep(.02)
    finally:
        guard.update(Intent(stop=True))
        if backend is not None:
            backend.stop()
        if backend is not None:
            backend.close()
        pad.close()
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--device', help='/dev/input/eventN or stable by-id path')
    parser.add_argument('--yaw-axis', choices=('ABS_RX', 'ABS_Z'), default='ABS_RX')
    parser.add_argument('--deadzone', type=float, default=.15)
    parser.add_argument('--real', action='store_true', help='enable continuous Arm7 control')
    parser.add_argument('--connect-only', action='store_true', help='with --real: check connections without controller writes')
    parser.add_argument('--speed-mm-s', type=float, default=10.)
    parser.add_argument('--yaw-speed-deg-s', type=float, default=5.)
    parser.add_argument('--gripper-rate', type=float, default=100., help='gripper target pulse rate while held')
    parser.add_argument('--config', type=Path, default=Path('config/dual_arm.local.json'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if not 0 <= args.deadzone < .8:
        parser.error('deadzone must be in [0,0.8)')
    if args.connect_only and not args.real:
        parser.error('--connect-only requires --real')
    if not (0 < args.speed_mm_s <= 15 and 0 < args.yaw_speed_deg_s <= 5 and 0 < args.gripper_rate <= 200):
        parser.error('speed must be (0,15] mm/s, yaw (0,5] deg/s, gripper rate (0,200] pulse/s')
    try:
        return run(args)
    except KeyboardInterrupt:
        return 130
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        print(f'gamepad: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
