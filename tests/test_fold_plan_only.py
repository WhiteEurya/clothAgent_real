import json
from types import SimpleNamespace

import pytest

from cloth_agent.fold_exploration_pipeline import FoldExplorationPipeline, build_parser


def test_plan_only_guard_prevents_execution_before_any_setup(tmp_path):
    pipe=FoldExplorationPipeline.__new__(FoldExplorationPipeline)
    pipe.stop_after_plan=True
    # No session/config exists: the guard must reject before touching hardware.
    with pytest.raises(PermissionError,match='prohibits'):
        pipe._execute(tmp_path/'motion.py',None,tmp_path,label='test')


def test_plan_only_preserves_code_points_and_does_not_record_execution(tmp_path):
    pipe=FoldExplorationPipeline.__new__(FoldExplorationPipeline)
    moves=[{'target':'grasp','base_xyz_mm':[400,0,20]},
           {'target':'pixel','base_xyz_mm':[450,50,80]}]
    pipe.client=SimpleNamespace(last_grounding_verification={'workspace_trace':{'moves':moves}})
    pipe._debug=lambda *args,**kwargs:None
    pipe._active_iteration=(tmp_path,{})
    source=tmp_path/'source.py';source.write_text('def run(robot):\n    robot.close_gripper()\n')
    iteration=tmp_path/'iteration_001';iteration.mkdir()
    proposal=SimpleNamespace(as_dict=lambda:{'actions':[{'name':'close_gripper','args':{}}]})
    summary={'iterations':[]}
    result=pipe._finish_plan_only(output=tmp_path,iteration_dir=iteration,iteration=1,
        step='left_side',mode='FOLD',source_path=source,model_proposal=proposal,
        execution_proposal=proposal,host_compilation={},summary=summary)
    assert result['status']=='PLAN_GENERATED' and result['manipulation_executed'] is False
    record=json.loads((iteration/'planning_only.json').read_text())
    assert record['grasp']==moves[0] and record['transport_destinations']==moves[1:]
    assert record['controller_ik']==record['preflight']=='NOT_RUN'
    assert record['execution']['status']=='NOT_EXECUTED'
    assert (iteration/'generated_motion.py').read_text()==source.read_text()
    assert pipe._active_iteration is None


def test_plan_only_flag_is_opt_in_and_retains_real_confirmation():
    parser=build_parser()
    assert parser.parse_args([]).stop_after_plan is False
    args=parser.parse_args(['--real','--confirm-real','--stop-after-plan'])
    assert args.real and args.confirm_real and args.stop_after_plan


def test_plan_only_failure_is_saved_without_learning(tmp_path):
    pipe=FoldExplorationPipeline.__new__(FoldExplorationPipeline)
    pipe.stop_after_plan=True
    pipe._active_iteration=(tmp_path,{'iteration':1})
    pipe._last_operational_stage='supervisor'
    pipe.experiences=SimpleNamespace(history=lambda **kwargs:[])
    pipe._save_iteration_learning=lambda *args:pytest.fail('must not learn an unexecuted test')
    result=pipe._save_interrupted_iteration(RuntimeError('backend unavailable'))
    assert result['status']=='FAILED' and result['experience_update']=='NOT_RUN'
    assert (tmp_path/'planning_only_failure.json').exists()
