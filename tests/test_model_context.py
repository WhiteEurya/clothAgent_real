from copy import deepcopy
from cloth_agent.model_context import compact_context


def test_preserves_commands_pose_and_outcome_without_raw_gripper_traces():
    action = dict(name='close_gripper', args={}, actual_ee_pose=[1, 2, 3], success=False,
                  error='timeout', robot_state={'samples': ['x'] * 10000},
                  gripper_result={'samples': ['x'] * 10000})
    raw = {'executed_actions': [action], 'evaluation': {'status': 'FAILURE'},
           'geometry': {'lower_z_mm': 20, 'height_retry': {'retry_index': 2, 'attempt_history': [action]}},
           'earlier_attempts': [{'executed_actions': [action]}]}
    original = deepcopy(raw)
    result = compact_context(raw)
    assert raw == original
    assert result['executed_actions'][0] == dict(name='close_gripper', args={}, actual_ee_pose=[1,2,3], success=False, error='timeout')
    assert result['evaluation'] == raw['evaluation']
    assert result['geometry'] == {'lower_z_mm': 20, 'height_retry': {'retry_index': 2}}
    assert result['earlier_attempts'][0]['executed_actions'] == result['executed_actions']
