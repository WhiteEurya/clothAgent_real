"""Check a far-side translation after lifting to 450 mm; no hardware IO."""
import ast
import argparse
import json
import sys
from pathlib import Path
from copy import deepcopy
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cloth_agent.experiments.partitioned_unfold import ROOT, configure_runtime
from cloth_agent.replay_dual_arm_runtime import PartitionedReplayRuntime
from cloth_agent.dual_arm.observation import Observation
from cloth_agent.robot_collision import RobotCollision
from cloth_agent.codex_motion import parse_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'reports/reset_point_trial_20261110_214829_343737')
    parser.add_argument('--output',type=Path,default=ROOT/'reports/reset_far_probe_20261110')
    parser.add_argument('--fine-step-mm',type=int,choices=(1,5),default=5)
    parser.add_argument('--heights',type=float,nargs='+',default=[450.])
    parser.add_argument('--distances',type=float,nargs='+',help='signed mm: positive toward image top, negative toward bottom')
    args = parser.parse_args()
    original = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(exist_ok=True)
    runtime = configure_runtime(PartitionedReplayRuntime(
        ROOT/'config/dual_arm.dual_wrist_xarm7_20261110.json', planning_only=True))
    before = Observation.load(original/'before')
    before.meta = deepcopy(before.meta)
    before.meta['joints_deg'] = json.loads((original/'post_parking_joints.json').read_text())
    calls = parse_code((original/'motion.py').read_text(), allow_grasp=True)
    # The fourth command, if present, is the previous trial's far translation.
    # Always measure a new distance from the third (midline) waypoint.
    if len(calls) not in (3,4) or [n for n,_ in calls[:3]] != ['grasp','move_tcp','move_tcp']:
        raise ValueError('unexpected reset source structure')
    calls = calls[:3]
    rows = []
    def check(distance, height=450.):
        lines = []
        for name, args in calls:
            pose = list(args['right'])
            if name != 'grasp': pose[2] = height/1000.
            lines.append(f'motion.{name}(right={pose!r})')
        if distance:
            pose = list(calls[-1][1]['right'])
            pose[2] = height/1000.
            pose[1] -= distance/1000.
            lines.append(f'motion.move_tcp(right={pose!r})')
        source = '\n'.join(lines)+'\n'
        row = dict(far_translation_mm=distance, world_z_mm=height,
                   release_world_mm=[v*1000 for v in pose[:3]],physical_execution=False)
        try:
            robot, program, assignment = runtime.compile(source, before, post_release_descent_mm=50.)
            actions = []
            for node in ast.parse(program).body[0].body:
                call = node.value
                actions.append(dict(name=call.func.id,args=dict(zip(
                    ('x','y','z','yaw'),[ast.literal_eval(a) for a in call.args]))))
            checker = RobotCollision(runtime.config_path, before.meta['joints_deg'],
                                     assignment['active_arm'], root=ROOT)
            checker.require_execution(robot, actions)
            row['status'] = 'COMPLETE_PATH_PASS'
            (output/f'z_{height}_far_{distance}_motion.py').write_text(source)
        except (ValueError, RuntimeError) as exc:
            row.update(status='REJECTED',error=f'{type(exc).__name__}: {exc}')
        rows.append(row)
        (output/'results.json').write_text(json.dumps(dict(source=str(original),
            hardware_calls=0, direction_world=[0,-1,0], grasp_yaw_offset_deg=90, left_retreat_mm=30,
            results=rows),ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(row,ensure_ascii=False),flush=True)
        return row['status']=='COMPLETE_PATH_PASS'
    if args.distances is not None:
        for height in args.heights:
            for distance in args.distances: check(distance,height)
        return
    for h in range(0, 301, 25): check(h)
    passing = [r['far_translation_mm'] for r in rows if r['status']=='COMPLETE_PATH_PASS']
    if passing:
        highest = max(passing)
        for h in range(highest+5, min(highest+25,301),5): check(h)
    if passing and args.fine_step_mm == 1:
        highest = max(r['far_translation_mm'] for r in rows if r['status']=='COMPLETE_PATH_PASS')
        for h in range(highest+1,min(highest+5,301)): check(h)


if __name__ == '__main__': main()
