"""Check saved reset geometry at alternative absolute heights; no hardware IO."""
import ast
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
    original = ROOT/'reports/reset_point_trial_20261110_214048_034003'
    output = ROOT/'reports/reset_height_probe_20261110'
    output.mkdir(exist_ok=True)
    runtime = configure_runtime(PartitionedReplayRuntime(
        ROOT/'config/dual_arm.dual_wrist_xarm7_20261110.json', planning_only=True))
    before = Observation.load(original/'before')
    before.meta = deepcopy(before.meta)
    before.meta['joints_deg'] = json.loads((original/'post_parking_joints.json').read_text())
    calls = parse_code((original/'motion.py').read_text(), allow_grasp=True)
    rows = []
    def check(height):
        lines = []
        for name, args in calls:
            pose = list(args['right'])
            if name != 'grasp': pose[2] = height/1000.
            lines.append(f'motion.{name}(right={pose!r})')
        source = '\n'.join(lines)+'\n'
        row = dict(world_z_mm=height, physical_execution=False)
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
            (output/f'z_{height}_motion.py').write_text(source)
        except (ValueError, RuntimeError) as exc:
            row.update(status='REJECTED',error=f'{type(exc).__name__}: {exc}')
        rows.append(row)
        (output/'results.json').write_text(json.dumps(dict(source=str(original),
            hardware_calls=0, grasp_yaw_offset_deg=90, left_retreat_mm=30,
            results=rows),ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(row,ensure_ascii=False),flush=True)
        return row['status']=='COMPLETE_PATH_PASS'
    for h in range(325, 776, 50): check(h)
    passing = [r['world_z_mm'] for r in rows if r['status']=='COMPLETE_PATH_PASS']
    if passing:
        highest = max(passing)
        for h in range(highest+10, min(highest+50,776),10): check(h)


if __name__ == '__main__': main()
