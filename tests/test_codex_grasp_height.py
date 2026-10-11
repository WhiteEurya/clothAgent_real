import pytest

from cloth_agent.codex_grasp_height import share_arm7_grasp_z
from cloth_agent.codex_motion import parse_code


@pytest.mark.parametrize('order', [('left', 'right'), ('right', 'left')])
def test_shared_height_only_changes_left_grasp_z(order):
    source = '\n'.join([
        'motion.move_tcp(left=[.4,.2,.1,1,2,3])',
        'motion.move_tcp(left=[.4,.2,.03,1,2,3],right=[.7,.25,.035,3,2,1])',
        *(f'motion.gripper(arm="{arm}",action="close")' for arm in order),
        'motion.move_tcp(left=[.4,.2,.15,1,2,3])',
    ])
    effective, changes = share_arm7_grasp_z(source)
    before, after = parse_code(source), parse_code(effective)
    assert after[1][1]['left'] == [.4,.2,.035,1,2,3]
    before[1][1]['left'][2] = .035
    assert after == before
    assert changes[0]['original_left_z_mm'] == 30
    assert changes[0]['arm7_reference_z_mm'] == 35


@pytest.mark.parametrize('source', [
    'motion.blocked(reason="no image")',
    'motion.gripper(arm="left",action="open")',
])
def test_no_grasp_unchanged(source):
    assert share_arm7_grasp_z(source) == (source, [])


@pytest.mark.parametrize('source', [
    'motion.gripper(arm="left",action="close")',
    'motion.gripper(arm="left",action="close")\nmotion.gripper(arm="right",action="close")',
    'motion.move_joints(left=[0,0,0,0,0,0],right=[0,0,0,0,0,0,0])\nmotion.gripper(arm="left",action="close")\nmotion.gripper(arm="right",action="close")',
    'motion.move_tcp(left=[0,0,.03,0,0,0],right=[0,0,.04,0,0,0])\nmotion.gripper(arm="left",action="close")\nmotion.move_tcp(left=[0,0,.1,0,0,0])\nmotion.gripper(arm="right",action="close")',
])
def test_ambiguous_grasps_rejected(source):
    with pytest.raises(ValueError):
        share_arm7_grasp_z(source)


def test_multiple_grasps_use_their_own_arm7_reference():
    source = '\n'.join(
        f'motion.move_tcp(left=[.4,.2,.03,0,0,0],right=[.7,.2,{z},0,0,0])\n'
        'motion.gripper(arm="left",action="close")\n'
        'motion.gripper(arm="right",action="close")\n'
        'motion.gripper(arm="left",action="open")\n'
        'motion.gripper(arm="right",action="open")' for z in [.035,.04])
    effective, changes = share_arm7_grasp_z(source)
    assert [args['left'][2] for name,args in parse_code(effective) if name=='move_tcp'] == [.035,.04]
    assert len(changes) == 2


def test_lower_only_grasp_target_and_not_repeated_close_or_other_arm():
    from cloth_agent.codex_grasp_height import lower_grasp_z
    from cloth_agent.codex_motion import parse_code
    source='''motion.move_tcp(right=[.4,0,.15,0,0,0])
motion.move_tcp(right=[.4,0,.04,0,0,0])
motion.gripper(arm="right",action="close")
motion.gripper(arm="right",action="close")
motion.move_tcp(right=[.4,0,.10,0,0,0])
motion.move_tcp(right=[.5,0,.04,0,0,0])
motion.gripper(arm="right",action="open")
motion.move_tcp(left=[.3,0,.03,0,0,0])
motion.gripper(arm="left",action="close")
'''
    effective,changes=lower_grasp_z(source,arm='right',offset_mm=10)
    before=parse_code(source);after=parse_code(effective)
    assert len(changes)==1
    assert abs(after[1][1]['right'][2]-.03)<1e-12
    assert before[:1]==after[:1] and before[2:]==after[2:]
