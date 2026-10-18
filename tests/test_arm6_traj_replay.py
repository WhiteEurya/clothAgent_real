import numpy as np
import pytest
from cloth_agent.experiments.arm6_traj_replay import instantiate_poses, compile_templates


def test_rejected_demo_requires_specific_authorization():
    from cloth_agent.experiments.arm6_traj_replay import validate_demonstration_authorization
    action = dict(manifest=dict(path='/record/traj.json', sha256='original'))
    manifest = dict(status='REJECTED', operator_accepted=False)
    with pytest.raises(ValueError):
        validate_demonstration_authorization(action, manifest)
    action['replay_authorization'] = dict(kind='USER_REQUESTED_REJECTED_DEMONSTRATION',
        manifest=action['manifest'].copy(), reason='User requested this iteration')
    validate_demonstration_authorization(action, manifest)
    with pytest.raises(ValueError):
        validate_demonstration_authorization(action, dict(status='ABORTED', operator_accepted=False))
    action['replay_authorization']['manifest']['sha256'] = 'another-record'
    with pytest.raises(ValueError):
        validate_demonstration_authorization(action, manifest)


def template():
    return dict(seed_start_mm=[100,100,30],seed_target_mm=[200,100,30],
        grasp_offset_mm=[1,2,-15], measured_grasp_mm_deg=[101,102,15,180,0,0],
        lift_above_surfaces_mm=30, approach_height_mm=50,place_offset_mm=5,clearance_mm=30)


def test_same_anchors_reproduce_measured_grasp_and_lift():
    poses=instantiate_poses(template(),[100,100,30],[200,100,30])
    assert poses['grasp'][:3]==pytest.approx([101,102,15])
    assert poses['lifted'][:3]==pytest.approx([101,102,60])
    assert poses['placed'][:3]==pytest.approx([200,100,35])
    assert poses['retreat'][2]==65


def test_live_anchors_rotate_offsets_and_use_current_depth():
    poses=instantiate_poses(template(),[300,200,40],[300,300,45])
    assert poses['grasp'][:3]==pytest.approx([298,201,25])
    assert poses['lifted'][2]==75
    assert poses['placed'][:3]==pytest.approx([300,300,50])
    from scipy.spatial.transform import Rotation
    assert Rotation.from_euler('xyz',poses['grasp'][3:],degrees=True).as_matrix()==pytest.approx(
        (Rotation.from_euler('z',90,degrees=True)*Rotation.from_euler('xyz',[180,0,0],degrees=True)).as_matrix())


def test_bad_anchor_or_insufficient_lift_rejected():
    with pytest.raises(ValueError):instantiate_poses(template(),[0,0,30],[0,1,30])
    with pytest.raises(ValueError):instantiate_poses(template(),[float('nan'),0,30],[0,100,30])
    value=template();value['lift_above_surfaces_mm']=0
    with pytest.raises(ValueError):instantiate_poses(value,[0,0,30],[0,100,30])


def test_arm6_cli_routes_without_calling_arm7_runtime(tmp_path,monkeypatch):
    from cloth_agent.experiments import graph_policy_replay as cli
    from cloth_agent.experiments import arm6_traj_replay as arm6
    monkeypatch.setattr(cli,'load_frozen',lambda _:dict(protocol=arm6.PROTOCOL))
    seen=[]
    monkeypatch.setattr(arm6,'run_cli',lambda args,parser:seen.append(args.execute) or dict(status='DRY_RUN_OK'))
    assert cli.main(['--graph',str(tmp_path/'graph.json'),'--dry-run'])==0
    assert seen==[False]
