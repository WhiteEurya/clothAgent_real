#!/usr/bin/env python3
"""Sequential 3 mm lift, close/open gripper, and return to saved home."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.manual.dual_arm_micro_test import checked, validate, preflight, ready, poses_close, stop_both


def run(arms, configs, events):
    preflight(arms, configs, 3)
    for a in arms:
        if checked(a.get_gripper_err_code(), 'gripper error') != 0:
            raise RuntimeError('Gripper fault; not clearing it automatically')
        checked(a.get_gripper_position(), 'initial gripper position')

    def record(cfg, phase, **kw):
        row = dict(ip=cfg['ip'], phase=phase, timestamp=datetime.now(timezone.utc).isoformat(), **kw)
        events.append(row)
        print(json.dumps(row), flush=True)

    def move(a, cfg, pose, phase):
        ready(a)
        code = a.set_position(*pose, speed=2, mvacc=5, radius=-1,
                              is_radian=False, relative=False, motion_type=0,
                              wait=True, timeout=20)
        if code != 0: raise RuntimeError(f'{phase}: motion code {code}')
        ready(a)
        actual = checked(a.get_position(is_radian=False), 'verify pose')
        if not poses_close(actual, pose): raise RuntimeError(f'{phase}: TCP target not reached')
        record(cfg, phase, tcp=actual)

    try:
        for a, cfg in zip(arms, configs):
            target = list(cfg['home']); target[2] += 3
            move(a, cfg, target, 'lifted_3mm')
            for label, pulse in [('closed', 5), ('opened', 850)]:
                ready(a)
                code = a.set_gripper_position(pulse, wait=True, speed=200,
                                              auto_enable=True, timeout=15)
                if code != 0: raise RuntimeError(f'{label}: gripper code {code}')
                end = time.monotonic() + 3
                while True:
                    pos = checked(a.get_gripper_position(), 'verify gripper')
                    status = checked(a.get_gripper_status(), 'gripper status')
                    error = checked(a.get_gripper_err_code(), 'gripper error')
                    if error: raise RuntimeError(f'Gripper error {error}')
                    if abs(pos-pulse) <= 15 and status & 3 != 1: break
                    if time.monotonic() >= end:
                        raise RuntimeError(f'{label}: target {pulse}, actual {pos}, status {status}')
                    time.sleep(.1)
                record(cfg, label, target_pulse=pulse, actual_pulse=pos, status=status)
                time.sleep(1)
            move(a, cfg, cfg['home'], 'returned_home')
        for a,cfg in zip(arms, configs):
            ready(a)
            if not poses_close(checked(a.get_position(is_radian=False),'final pose'),cfg['home']):
                raise RuntimeError('Final home check failed')
    except BaseException:
        stop_both(arms)
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--home',type=Path,default=Path('data/robot/dual_arm_home.json'))
    args=p.parse_args()
    cfg=validate(json.loads(args.home.read_text()))
    print('Sequence:',[c['ip'] for c in cfg], 'lift 3 mm -> close -> open -> home')
    if not args.execute: return
    from xarm.wrapper import XArmAPI
    arms=[];report={'status':'started','events':[]}
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    path=Path('results/dual_arm_startup') / f'gripper_sequence_{stamp}.json'
    try:
        for c in cfg:
            a=XArmAPI(c['ip'],is_radian=False);arms.append(a);time.sleep(.3)
        run(arms,cfg,report['events'])
        report['status']='completed'
    except BaseException as exc:
        report.update(status='failed',error=str(exc))
        raise
    finally:
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2)+'\n')
        print('Report:',path,flush=True)
        for a in arms:a.disconnect()


if __name__=='__main__': main()
