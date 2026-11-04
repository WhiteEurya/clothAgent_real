"""Check a saved top-down fold through the native planner without hardware."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from cloth_agent.codex_dual_runtime import ZedSingleArmRuntime
from cloth_agent.experiments.prepare_partitioned_trial import staged_operations, validate_source
from cloth_agent.site_observation import load_site, convert_pose_mm_deg


def planning_assumptions(config):
    return {
        'scope': 'Conditional model check, not measured physical safety performance',
        'physical_measurements_established_by_this_run': {
            'maximum_stop_excursion_deg': 'UNKNOWN',
            'maximum_joint_tracking_error_deg': 'UNKNOWN',
            'communication_loss_stop_response_s': 'UNKNOWN',
        },
        'configured_model_inputs': {
            'stop_excursion_deg': {
                arm: row['stop_excursion_deg']
                for arm, row in config.raw['safety']['arms'].items()
            },
            'tracking_error_deg': config.limits['tracking_error_deg'],
        },
        'physical_acceptance_inferred': False,
    }


def motion_source(plan, start_pose, profile):
    lines=[]
    for operation in staged_operations(plan,start_pose):
        arm=operation['arm']
        if operation['kind']=='gripper':
            lines.append(f'motion.gripper(arm={arm!r}, action={operation["action"]!r})')
        else:
            pose=convert_pose_mm_deg(profile,operation['pose_base_mm_deg'],arm,'common')
            target=np.r_[pose[:3]/1000,np.radians(pose[3:])].tolist()
            lines.append(f'motion.move_tcp({arm}={target})')
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proposals',type=Path,required=True)
    parser.add_argument('--robot-state',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=Path('config/dual_arm.zed_dual_xarm7_20261103.json'))
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(status='STARTED',physical_execution=False,execution_allowed=False,robot_calls=0)
    runtime=None
    try:
        profile=load_site()
        plan=validate_source(args.proposals,profile)
        state=json.loads(args.robot_state.read_text())
        active=plan['selected_arm']
        runtime=ZedSingleArmRuntime(args.config,active,planning_only=True)
        report['planning_assumptions'] = planning_assumptions(runtime.config)
        initial={arm:np.asarray(row['joints_deg']) for arm,row in state['arms'].items()}
        for arm,record in state['arms'].items():
            expected=runtime.config.arms[arm]
            if record['status']!='READ_ONLY_CHECKS_PASS' or record['control_box_sn']!=expected.serial:
                raise ValueError('Saved robot identity/state mismatch')
        inactive=plan['inactive_arm']
        reference=runtime.config.static_arm_obstacles[inactive]['joint_angles_deg']
        if not np.allclose(initial[inactive],reference,atol=.1,rtol=0):
            raise ValueError('Inactive arm differs from the static model')
        source=motion_source(plan,state['arms'][active]['pose_base_mm_deg'],profile)
        (args.output/'motion.py').write_text(source)
        report.update(active_arm=active,grasp_height_adjustment_mm=0,
                      note='Offline native planning of exact proposed anchors; no legacy grasp-height lowering.')
        operations=runtime.plan(source,initial,preserve_command_boundaries=True)
        serialized=[dict(kind=kind,data=data.to_dict() if kind=='move' else data)
                    for kind,data in operations]
        artifact=dict(config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
            config_path=str(args.config.resolve()),active_arm=active,
            source_proposals=str(args.proposals.resolve()),source_robot_state=str(args.robot_state.resolve()),
            physical_execution=False,execution_allowed=False,
            initial_joints_deg={arm:values.tolist() for arm,values in initial.items()},
            operations=serialized)
        (args.output/'operations.json').write_text(json.dumps(artifact,indent=2,allow_nan=False)+'\n')
        stats=dict(runtime.plan_stats)
        if 'dispatched_segments' in stats:
            stats['planned_dispatch_segments']=stats.pop('dispatched_segments')
        report.update(status='OFFLINE_PATH_PASS',operation_count=len(operations),stats=stats,
                      physical_motion_commands=0,operations_file='operations.json')
    except Exception as exc:
        report.update(status='BLOCKED',error=f'{type(exc).__name__}: {exc}')
        if runtime is not None:
            report['planning_context']=getattr(runtime,'planning_context',{})
    finally:
        if runtime is not None:
            runtime.close()
        (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2))
    return 0 if report['status']=='OFFLINE_PATH_PASS' else 2


if __name__=='__main__':
    raise SystemExit(main())
