"""Offline tests: no camera, robot or real model requests."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch,Mock
import pytest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('reason_runner',ROOT/'experiments/live_skill_discovery_01/run.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
OBSERVED={'regions':{k:{'before':'有折线','after':'折线仍可见','visible_change':'边缘位置改变',
    'correspondence':'UNKNOWN','image_evidence':'印花部分可见','unknowns':'无法对应布层'} for k in r.REGION_NAMES},
    'whole_garment_motion':'整体向右移动','local_relative_changes':'不确定','observation_limits':'遮挡'}
def judgment(label):return {'label':label,'evidence':'具体图像证据','observation_keys':['central_body'],'uncertainty':'部分遮挡'}
RESULT={'original_target':'局部卷折','target_completed':judgment('NO'),'local_progress':judgment('YES'),
    'adverse_change':judgment('UNKNOWN'),'undetermined_results':['隐藏布层'],'observation_conflicts':'无'}

def test_delegates_existing_episode_without_new_loop(tmp_path):
    from cloth_agent.experiments import one_shot_goal_full_episode_v1 as ep
    original=ep.EpisodeCollectionRuntime;validate=ep.planner.validate_plan;prompt=ep.SINGLE_VIEW_PROMPT
    def main():
        argv=sys.argv
        assert '--single-view' in argv and '--same-prompt-retry' in argv and '--retry-plan-feedback' in argv
        assert '--visible-sleeve-hem' not in argv and '--automatic-top5' not in argv
        assert argv[argv.index('--max-rounds')+1]=='6'
        assert '--no-reset' not in argv and '--review-initialization' not in argv
        assert issubclass(ep.EpisodeCollectionRuntime,original)
        assert ep.SINGLE_VIEW_PROMPT==prompt
        assert 20<=float(argv[argv.index('--initialize-center-radius-mm')+1])<=40
        plan={'decision':'ACTION','grasp_px':[1,1],'target_px':[2,2],'motion_type':'lift_pull_place','grasp_angle_deg':None,'short_reason':'第一句。第二句。'}
        ep.planner.validate_plan(plan,[10,10],require_grasp_rotation=True)
        with pytest.raises(ValueError):ep.planner.validate_plan({**plan,'grasp_px':[100,100]},[10,10])
        return 0
    args=SimpleNamespace(output=tmp_path/'run',config=tmp_path/'config',max_rounds=6,goal_image=None)
    with patch.object(ep,'main',side_effect=main) as called:
        assert r.execute(args)==0
    called.assert_called_once()
    assert ep.EpisodeCollectionRuntime is original and ep.planner.validate_plan is validate and ep.SINGLE_VIEW_PROMPT==prompt

@pytest.mark.parametrize('status',['EXECUTED','DISPATCHED','FAILED'])
def test_analyzes_only_completed_actions(tmp_path,status):
    r.io.save(tmp_path/'execution.json',{'status':status})
    r.io.save(tmp_path/'plan.json',{'short_reason':'test'})
    with patch.object(r.io,'request',side_effect=[OBSERVED,RESULT]) as request:
        r.analyze_round(tmp_path)
    assert request.call_count==2*int(status=='EXECUTED')
    if status=='EXECUTED':
        result=r.io.read(tmp_path/'reason_analysis.json')
        assert result['target_completed']['label']=='NO' and result['local_progress']['label']=='YES'
        assert result['adverse_change']['label']=='UNKNOWN' and not result['planner_feedback']
        assert 'outcome' not in result
        assert 'short_reason' not in request.call_args_list[0].args[2]
        assert (tmp_path/'physical_observation.json').exists()
        assert request.call_args.args[3]==[tmp_path/'before/rgb.png',tmp_path/'after/rgb.png']

def test_analysis_error_does_not_stop_episode_but_interrupt_does(tmp_path):
    r.io.save(tmp_path/'execution.json',{'status':'EXECUTED'});r.io.save(tmp_path/'plan.json',{})
    with patch.object(r.io,'request',side_effect=RuntimeError('model failure')):r.analyze_round(tmp_path)
    assert (tmp_path/'reason_analysis_error.json').exists()
    with patch.object(r.io,'request',side_effect=KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):r.analyze_round(tmp_path)

def test_dry_run_has_no_execution():
    with patch.object(r,'execute',side_effect=AssertionError('hardware')):
        assert r.main(['--dry-run'])==0


def test_second_layer_failure_preserves_observation_without_fake_labels(tmp_path):
    r.io.save(tmp_path/'execution.json',{'status':'EXECUTED'});r.io.save(tmp_path/'plan.json',{})
    with patch.object(r.io,'request',side_effect=[OBSERVED,RuntimeError('timeout')]):r.analyze_round(tmp_path)
    assert r.io.read(tmp_path/'physical_observation.json')['regions']==OBSERVED['regions']
    assert not (tmp_path/'reason_analysis.json').exists()
    assert r.io.read(tmp_path/'reason_analysis_error.json')['stage']=='interpretation'

def test_schema_keeps_independent_labels_and_rejects_old_success_verdict():
    import jsonschema
    jsonschema.validate(OBSERVED,r.OBSERVATION_SCHEMA)
    jsonschema.validate(RESULT,r.ANALYSIS_SCHEMA)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({**RESULT,'outcome':'FAILURE'},r.ANALYSIS_SCHEMA)


def test_unlimited_rounds_and_retry_have_no_artificial_cap():
    from itertools import islice
    from cloth_agent.experiments import one_shot_goal_full_episode_v1 as ep
    bounded=SimpleNamespace(max_rounds=6,unlimited_rounds=False)
    unlimited=SimpleNamespace(max_rounds=12,unlimited_rounds=True)
    assert list(ep.episode_round_indices(bounded))==list(range(6))
    assert not ep.another_round_available(bounded,5)
    assert list(islice(ep.episode_round_indices(unlimited),101))[-1]==100
    assert ep.another_round_available(unlimited,1000000)


def test_resume_is_no_reset_with_original_goal_and_no_round_limit(tmp_path):
    from cloth_agent.experiments import one_shot_goal_full_episode_v1 as ep
    source=tmp_path/'previous';source.mkdir()
    (source/'prompt.txt').write_text(ep.SINGLE_VIEW_PROMPT+'\n本实验Pa到Pb的桌面XY牵拉距离限定为20–40毫米，由Host按标定坐标核验。')
    args=SimpleNamespace(output=tmp_path/'next',config=tmp_path/'config',max_rounds=0,goal_image=None,resume_from=source)
    def main():
        assert '--unlimited-rounds' in sys.argv and '--max-rounds' not in sys.argv
        assert '--no-reset' in sys.argv
        assert sys.argv[sys.argv.index('--goal-image')+1]==str(source/'goal/target.png')
        args.output.mkdir()
        return 0
    with patch.object(ep,'main',side_effect=main):assert r.execute(args)==0
    protocol=r.io.read(args.output/'reason_analysis_protocol.json')
    assert protocol['max_rounds'] is None and not protocol['startup_reset']
    assert protocol['resume_from']==str(source)
    assert protocol['source_prompt_corrected']
    assert protocol['pull_distance_xy_mm']==[10,150]
    assert protocol['reset_radius_bound_mm'] is None


def test_resume_prompt_mismatch_never_starts_runtime(tmp_path):
    from cloth_agent.experiments import one_shot_goal_full_episode_v1 as ep
    source=tmp_path/'previous';source.mkdir();(source/'prompt.txt').write_text('different')
    args=SimpleNamespace(output=tmp_path/'next',config=tmp_path/'config',max_rounds=0,goal_image=None,resume_from=source)
    with patch.object(ep,'main',side_effect=AssertionError('must not connect')):
        with pytest.raises(ValueError,match='Prompt'):r.execute(args)


@pytest.mark.parametrize('radius',[20.,40.])
def test_reset_radius_only_and_no_pull_override(tmp_path,radius):
    from cloth_agent.experiments import one_shot_goal_full_episode_v1 as ep
    prepare=ep.prepare_action;prompt=ep.SINGLE_VIEW_PROMPT
    args=SimpleNamespace(output=tmp_path/'new',config=tmp_path/'config',max_rounds=6,goal_image=None)
    def main():
        assert ep.prepare_action is prepare
        assert ep.SINGLE_VIEW_PROMPT==prompt
        assert float(sys.argv[sys.argv.index('--initialize-center-radius-mm')+1])==radius
        args.output.mkdir()
        return 0
    with patch('random.SystemRandom.uniform',return_value=radius),patch.object(ep,'main',side_effect=main):
        assert r.execute(args)==0
    saved=r.io.read(args.output/'reason_analysis_protocol.json')
    assert saved['reset_radius_bound_mm']==radius and saved['pull_distance_xy_mm']==[10,150]
