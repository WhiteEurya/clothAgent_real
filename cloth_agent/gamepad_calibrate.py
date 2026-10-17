"""Learn physical stick/button assignments, then run input-only verification."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

from .gamepad_teleop import Gamepad, list_gamepads, normalize
from .gamepad_profile import BUTTONS, identity, save_profile, validate_profile
from .gamepad_check import Check


def axis_candidate(axes, keys, used=()):
    moved = [code for code, value in axes.items() if abs(value) > .25]
    if keys or len(moved) != 1:
        return None
    code = moved[0]
    if code in used or abs(axes[code]) < .8:
        return None
    return (code, 1 if axes[code] > 0 else -1)


def button_candidate(axes, keys, used=()):
    if any(abs(v) > .15 for v in axes.values()) or len(keys) != 1:
        return None
    code = next(iter(keys))
    return code if code not in used else None


class Learner:
    def __init__(self, device, timeout):
        self.device, self.timeout = device, timeout
        self.axes = [code for code, info in device.capabilities().get(3, [])
                     if info.max - info.min > 2]
        if len(self.axes) < 3:
            raise ValueError('至少需要三个模拟轴，请确认手柄处于双摇杆/模拟模式')

    def sample(self):
        values = {}
        for code in self.axes:
            info = self.device.absinfo(code)
            values[code] = normalize(info.value, info.min, info.max, deadzone=0)
        return values, set(self.device.active_keys())

    def wait(self, prompt, predicate):
        print('\n' + prompt, flush=True)
        end, since, previous = time.monotonic() + self.timeout, None, None
        while time.monotonic() < end:
            value = predicate(*self.sample())
            if value is not None:
                if value != previous or since is None:
                    since = time.monotonic()
                if time.monotonic() - since >= .35:
                    print(f'  ✓ 已识别 {value}', flush=True)
                    return value
            else:
                since = None
            previous = value
            time.sleep(.02)
        raise RuntimeError('本项超时，未保存映射；请重新运行并按提示操作')

    def neutral(self):
        self.wait('松开所有按键，两个摇杆回中。',
                  lambda axes, keys: True if not keys and all(abs(v) < .15 for v in axes.values()) else None)

    def learn(self):
        self.neutral()
        axes, buttons = [], {}
        for positive, negative in (('左摇杆向右推到底', '左摇杆向左推到底'),
                                   ('左摇杆向后/下拉到底', '左摇杆向前/上推到底'),
                                   ('右摇杆向右推到底', '右摇杆向左推到底')):
            used = [a['code'] for a in axes]
            code, sign = self.wait(positive + '，保持半秒；其他控件不动。',
                                   lambda a, k: axis_candidate(a, k, used))
            self.neutral()
            self.wait(negative + '，保持半秒（验证反方向）。',
                      lambda a, k: True if axis_candidate(a, k) == (code, -sign) else None)
            axes.append(dict(code=code, sign=sign))
            self.neutral()
        for name in BUTTONS:
            label = 'LB / L1（左肩键，不是扳机）' if name == 'LB' else name
            buttons[name] = self.wait(f'只按住实体 {label} 键，保持半秒。',
                lambda a, k: button_candidate(a, k, buttons.values()))
            self.neutral()
        profile = dict(version=1, device=identity(self.device), name=self.device.name,
                       axes=axes, buttons=buttons)
        validate_profile(profile, self.device)
        return profile


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device')
    parser.add_argument('--timeout', type=float, default=60.)
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('--timeout must be finite and positive')
    device, pad, check = None, None, None
    try:
        from evdev import InputDevice
        path = args.device
        if path is None:
            rows = list_gamepads()
            if len(rows) != 1:
                raise RuntimeError(f'发现 {len(rows)} 个手柄，请通过 --device 选择')
            path = rows[0]['path']
        device = InputDevice(path)
        print(f'设备：{device.name}，{identity(device)}\n只读取输入，不连接机器人。\n'
              '全程按屏幕提示操作，自动进入下一项，不用回车。Ctrl+C 退出。', flush=True)
        profile = Learner(device, args.timeout).learn()
        output = Path(__file__).resolve().parents[1] / 'reports/gamepad_check' / datetime.now(timezone.utc).strftime('learn_%Y%m%dT%H%M%S%fZ')
        output.mkdir(parents=True)
        (output / 'candidate_profile.json').write_text(json.dumps(profile, ensure_ascii=False, indent=2))
        pad = Gamepad(path, profile=profile)
        print('\n映射已识别。自动进入功能核定，通过后才保存为默认配置。', flush=True)
        check = Check(pad, output, args.timeout)
        check.report['candidate_profile'] = profile
        check.run()
        if check.report['status'] != 'PASS':
            print('核定未通过，不修改默认配置。', flush=True)
            return 1
        destination = save_profile(profile, device)
        check.report['active_profile'] = str(destination)
        print(f'\n映射已保存：{destination}\n后续预览与遥操将自动使用这份映射。', flush=True)
        return 0
    except KeyboardInterrupt:
        if check:
            check.report.update(status='INTERRUPTED', reason='user interrupted')
        print('已退出；没有启动机器人。')
        return 130
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        if check:
            check.report.update(status='ERROR', reason=str(exc))
        print(f'测试未完成：{exc}', flush=True)
        return 1
    finally:
        if check:
            check.finish()
        if pad:
            pad.close()
        if device:
            device.close()


if __name__ == '__main__':
    raise SystemExit(main())
