"""One-shot guided controller verification. Never imports the robot backend."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from .gamepad_teleop import Gamepad, Mapper, InputGuard, Intent, list_gamepads


@dataclass(frozen=True)
class Step:
    name: str
    prompt: str
    keys: frozenset[str]
    axis: int | None = None
    sign: int = 0
    effect: str | None = None
    value: object = None


STEPS = (
    Step('disabled', '不要按 LB，左摇杆向前推到底', frozenset(), 1, -1, 'enabled', False),
    Step('forward', '按住 LB，左摇杆向前推到底 → 前移', frozenset({'LB'}), 1, -1, 'forward', 1),
    Step('backward', '按住 LB，左摇杆向后拉到底 → 后移', frozenset({'LB'}), 1, 1, 'forward', -1),
    Step('left', '按住 LB，左摇杆向左推到底 → 左移', frozenset({'LB'}), 0, -1, 'right', -1),
    Step('right', '按住 LB，左摇杆向右推到底 → 右移', frozenset({'LB'}), 0, 1, 'right', 1),
    Step('yaw_left', '按住 LB，右摇杆向左推到底 → yaw 正方向', frozenset({'LB'}), 2, -1, 'yaw', 1),
    Step('yaw_right', '按住 LB，右摇杆向右推到底 → yaw 负方向', frozenset({'LB'}), 2, 1, 'yaw', -1),
    Step('up', '按住 LB，再按住 Y → 上升', frozenset({'LB', 'Y'}), effect='up', value=1),
    Step('down', '按住 LB，再按住 A → 下降', frozenset({'LB', 'A'}), effect='up', value=-1),
    Step('open', '按住 LB，再按住 X → 开爪', frozenset({'LB', 'X'}), effect='gripper', value='open'),
    Step('close', '按住 LB，再按住 B → 合爪', frozenset({'LB', 'B'}), effect='gripper', value='close'),
    Step('vertical_conflict', '按住 LB，同时按住 Y 和 A → 不升降', frozenset({'LB', 'Y', 'A'}), effect='up', value=0),
    Step('gripper_conflict', '按住 LB，同时按住 X 和 B → 不新增开合指令', frozenset({'LB', 'X', 'B'}), effect='gripper', value=None),
    Step('release_prepare', '按住 LB，左摇杆向前推，并准备保持摇杆位置', frozenset({'LB'}), 1, -1, 'forward', 1),
    Step('release_lb', '保持左摇杆向前，只松开 LB → 输入立即归零', frozenset(), 1, -1, 'enabled', False),
    Step('stop', '按住 START → 停止标记（本核定程序会正常结束）', frozenset({'START'}), effect='stop', value=True),
)


def matches(step, axes, keys):
    if set(keys) != set(step.keys):
        return False
    for i, value in enumerate(axes):
        if i == step.axis:
            if value * step.sign < .75:
                return False
        elif abs(value) > .15:
            return False
    return True


def effect_matches(step, intent):
    if step.effect == 'enabled' and step.value is False:
        return not intent.enabled and not intent.moving and intent.gripper is None
    actual = getattr(intent, step.effect)
    if step.axis is not None:
        return actual * step.value >= .75
    return actual == step.value


class Check:
    def __init__(self, pad, output, timeout=45., hold=.3):
        self.pad, self.output, self.timeout, self.hold = pad, output, timeout, hold
        self.mapper = Mapper()
        self.guard = InputGuard()
        self.previous_time = None
        self.report = dict(status='INCOMPLETE', robot_connected=False,
                           scope='controller input and software mapping only; no physical robot verification',
                           device=dict(path=pad.device.path, name=pad.device.name,
                                       info=pad.device.info._asdict()), steps=[], samples=0,
                           max_poll_gap_s=0., started_at=datetime.now(timezone.utc).isoformat())
        self.trace = (output / 'samples.jsonl').open('w')

    def sample(self):
        now = time.monotonic()
        axes_and_keys = self.pad.sample()
        axes, keys = axes_and_keys[:3], axes_and_keys[3]
        # Include unmapped buttons so accidental RB/trigger-stick clicks do
        # not silently satisfy a prompt asking for a different physical key.
        all_keys = self.pad.device.active_keys()
        keys = set(keys) | {f'KEY_{code}' for code in all_keys if code not in self.pad.keys}
        intent = self.mapper.update(*axes, keys)
        self.guard.update(intent)
        if self.previous_time is not None:
            self.report['max_poll_gap_s'] = max(self.report['max_poll_gap_s'], now - self.previous_time)
        self.previous_time = now
        self.report['samples'] += 1
        self.trace.write(json.dumps(dict(t=now, axes=axes, buttons=sorted(keys), intent=asdict(intent))) + '\n')
        return now, axes, keys, intent

    def neutral(self, allow_lb=True):
        print('  松开操作键、摇杆回中' + ('；LB 可以继续按住。' if allow_lb else '，LB 也松开。'), flush=True)
        deadline, since = time.monotonic() + self.timeout, None
        while time.monotonic() < deadline:
            now, axes, keys, _ = self.sample()
            neutral = not any(axes) and keys <= ({'LB'} if allow_lb else set())
            since = (now if since is None else since) if neutral else None
            if since is not None and now - since >= self.hold:
                return True
            time.sleep(.02)
        return False

    def step(self, step):
        deadline, since, effect_seen = time.monotonic() + self.timeout, None, False
        last_display = 0.
        evidence = None
        # release_lb must invalidate the token captured while LB was held.
        previous_token = self.guard.token()
        while time.monotonic() < deadline:
            now, axes, keys, intent = self.sample()
            if matches(step, axes, keys):
                if since is None:
                    since, effect_seen = now, False
                effect_seen |= effect_matches(step, intent)
                if evidence is None or effect_matches(step, intent):
                    evidence = dict(axes=axes, buttons=sorted(keys), intent=asdict(intent))
                if now - since >= self.hold and effect_seen:
                    if step.name == 'release_lb' and self.guard.permits(generation=previous_token):
                        return dict(name=step.name, status='FAIL', reason='LB release did not invalidate permission')
                    return dict(name=step.name, status='PASS', evidence=evidence)
            else:
                since, effect_seen, evidence = None, False, None
            if now - last_display > 3:
                print(f'  当前摇杆 {[round(v, 2) for v in axes]}，按键 {sorted(keys)}', flush=True)
                last_display = now
            time.sleep(.02)
        return dict(name=step.name, status='FAIL', reason='timeout: required input/effect not held',
                    evidence=evidence)

    def run(self):
        print(f'设备：{self.pad.device.name}\n共 {len(STEPS)} 项；照提示操作即可，自动进入下一项，无需回车。\n'
              '每次保持约半秒；超时记录失败后继续。Ctrl+C 可中止。\n'
              '只核定输入，机器人不会运动。', flush=True)
        if not self.neutral(allow_lb=False):
            self.report['reason'] = 'startup controls did not return to neutral'
            return
        for i, step in enumerate(STEPS, 1):
            print(f'\n[{i}/{len(STEPS)}] {step.prompt}', flush=True)
            row = self.step(step)
            self.report['steps'].append(row)
            print('  ✓ 通过' if row['status'] == 'PASS' else '  ✗ 未通过（已记录，继续下一项）', flush=True)
            # Preserve the stick and LB for the release test, then let the
            # next prompt explicitly request releasing only LB.
            if step.name != 'release_prepare' and i != len(STEPS):
                if not self.neutral():
                    self.report['reason'] = f'controls did not reset after {step.name}'
                    return
        self.report['status'] = 'PASS' if all(s['status'] == 'PASS' for s in self.report['steps']) else 'FAIL'

    def finish(self):
        self.trace.close()
        self.report['finished_at'] = datetime.now(timezone.utc).isoformat()
        (self.output / 'report.json').write_text(json.dumps(self.report, ensure_ascii=False, indent=2) + '\n')
        print(f'\n核定结果：{self.report["status"]}\n报告：{self.output / "report.json"}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--timeout', type=float, default=45., help='seconds per prompt')
    parser.add_argument('--yaw-axis', choices=('ABS_RX', 'ABS_Z'), default='ABS_RX')
    parser.add_argument('--probe-seconds', type=float, help='read-only connection check; does not certify controls')
    args = parser.parse_args(argv)
    import math
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('--timeout must be finite and positive')
    if args.probe_seconds is not None and (not math.isfinite(args.probe_seconds) or not 0 < args.probe_seconds <= 60):
        parser.error('--probe-seconds must be in (0,60]')
    pad, check = None, None
    try:
        path = args.device
        if path is None:
            devices = list_gamepads()
            if len(devices) != 1:
                raise RuntimeError(f'发现 {len(devices)} 个可读手柄，请连接设备或用 --device 明确选择')
            path = devices[0]['path']
        pad = Gamepad(path, yaw_axis=args.yaw_axis)
        output = args.output or Path(__file__).resolve().parents[1] / 'reports' / 'gamepad_check' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        output.mkdir(parents=True, exist_ok=False)
        check = Check(pad, output, args.timeout)
        if args.probe_seconds is not None:
            deadline = time.monotonic() + args.probe_seconds
            while time.monotonic() < deadline:
                check.sample()
                time.sleep(.02)
            check.report['status'] = 'CONNECTION_ONLY'
        else:
            check.run()
        return 0 if check.report['status'] in ('PASS', 'CONNECTION_ONLY') else 1
    except KeyboardInterrupt:
        if check is not None:
            check.report.update(status='INTERRUPTED', reason='user interrupted')
        return 130
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        if check is not None:
            check.report.update(status='ERROR', reason=f'{type(exc).__name__}: {exc}')
        print(f'核定未完成：{exc}', flush=True)
        return 1
    finally:
        if check is not None:
            check.finish()
        if pad is not None:
            pad.close()


if __name__ == '__main__':
    raise SystemExit(main())
