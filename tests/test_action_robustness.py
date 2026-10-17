import json
from types import SimpleNamespace

import numpy as np
import pytest

from cloth_agent.action_robustness import CODE_PROMPT, strip_code_evidence, strip_grounding_evidence
from cloth_agent.codex_loop import loop
from cloth_agent.codex_motion import parse_code
from cloth_agent.experiments import graph_minimal_grounding as mg
from cloth_agent.experiments import graph_policy_replay as replay
from cloth_agent.experiments.robustness_planner_audit import camera_mapping, project
from tests.test_codex_loop import FakeRuntime, model
from tests.test_graph_policy_replay import policy, candidate


EVIDENCE = dict(tags=['clear_edge', 'supported_target'], confidence=.75,
                reason='Exposed flap and broad receiving fabric')
COMMENT = '# ACTION_ROBUSTNESS ' + json.dumps(EVIDENCE) + '\n'
MOTION = ('motion.grasp(right=[.5,0,.06,3.14,0,0])\n'
          'motion.move_tcp(right=[.5,.1,.12,3.14,0,0])\n')


@pytest.mark.parametrize('evidence',[EVIDENCE,dict(tags=['supported_target'],confidence=0,
    reason='Adjacent layers may be grasped together and can carry the target cloth region')])
def test_loop_logs_evidence_but_dispatch_history_and_motion_exclude_it(tmp_path,evidence):
    runtime = FakeRuntime();prompts = []
    root = tmp_path/'run'
    comment='# ACTION_ROBUSTNESS '+json.dumps(evidence)+'\n'
    result = loop('fold', runtime, root, max_rounds=1,
                  model_call=model(runtime, prompts, [comment+MOTION]))
    assert result['rounds'] == 1
    assert CODE_PROMPT in prompts[0]
    assert runtime.executions[0][0] == MOTION
    assert 'ACTION_ROBUSTNESS' not in (root/'iteration_history.jsonl').read_text()
    assert 'clear_edge' not in (root/'round_001/motion.py').read_text()
    log = json.loads((root/'round_001/action_robustness.json').read_text())
    assert log['tags'] == evidence['tags'] and log['host_validated'] is False
    assert log['confidence'] == evidence['confidence']
    assert parse_code(comment+MOTION, allow_grasp=True) == parse_code(MOTION, allow_grasp=True)


def test_no_robust_handle_never_dispatches(tmp_path):
    runtime = FakeRuntime()
    answer = '# ACTION_ROBUSTNESS {"tags":[],"confidence":0,"reason":"occluded handle"}\n'
    answer += 'motion.blocked(reason="NO_ROBUST_ACTION: missing reliable grasp handle")'
    result = loop('fold', runtime, tmp_path/'run', max_rounds=1,
                  model_call=model(runtime, [], [answer]))
    assert result['rounds'] == 1 and not runtime.executions
    outcome = json.loads((tmp_path/'run/round_001/outcome.json').read_text())
    assert outcome['status'] == 'BLOCKED' and outcome['physical_execution'] is False


@pytest.mark.parametrize('comment,status', [('', 'MISSING'), ('# ACTION_ROBUSTNESS nope\n','INVALID'),
    ('# ACTION_ROBUSTNESS {"tags":["invented"],"confidence":1,"reason":"x"}\n','INVALID'),
    (COMMENT+COMMENT,'INVALID'),
    ('# ACTION_ROBUSTNESS {"tags":[],"confidence":NaN,"reason":"x"}\n','INVALID')])
def test_bad_metadata_does_not_change_motion_or_manufacture_evidence(tmp_path,comment,status):
    assert strip_code_evidence(comment+MOTION,tmp_path) == MOTION
    log = json.loads((tmp_path/'action_robustness.json').read_text())
    assert log['status'] == status and 'tags' not in log


@pytest.mark.parametrize('evidence',[EVIDENCE,dict(tags=[],confidence=0,
    reason='Layer identity uncertain; available cloth handle can move the target region')])
def test_full_grounding_strips_diagnostics_before_graph_consumers(tmp_path,evidence):
    _, graph = policy(tmp_path)
    value = dict(candidate(), robustness=evidence)
    replay.validate_grounding(value)
    class Requests:
        def ask(self,*args,**kwargs):
            kwargs['validator'](value)
            return value
    obs = SimpleNamespace(image=tmp_path/'rgb.png', xyz=np.zeros((20,20,3)), meta={})
    result = replay.selected_edge_grounding(Requests(),graph,graph['edges'][0],obs,tmp_path)
    assert result == candidate() and 'robustness' in value  # no input mutation
    assert json.loads((tmp_path/'action_robustness.json').read_text())['status'] == 'RECORDED'


def test_minimal_uncertainty_logged_before_existing_rejection(tmp_path):
    grounder = object.__new__(mg.MinimalGrounder)
    grounder.template = lambda edge: dict(spec={},image_size=[100,100], confidence_min=.8)
    value = dict(grasp_px=None,target_px=None,confidence=0,robustness=dict(
        tags=[], confidence=0, reason='NO_ROBUST_ACTION: occluded grasp handle'))
    class Requests:
        def ask(self,*args,**kwargs):
            kwargs['validator'](value)
            return value
    with pytest.raises(mg.GroundingUncertain):
        grounder.ground(Requests(),dict(edge_id='e001'),SimpleNamespace(image='rgb.png'),tmp_path)
    assert 'robustness' not in json.loads((tmp_path/'minimal_anchors.json').read_text())
    assert json.loads((tmp_path/'action_robustness.json').read_text())['confidence'] == 0


def test_recovery_full_grounding_diagnostics_do_not_enter_candidate(tmp_path,monkeypatch):
    from tests.test_edge_applicability import setup_case
    manager,edge,obs,calls,graph,requests = setup_case(tmp_path,monkeypatch,uncertain=True)
    monkeypatch.setattr(replay,'grounding_payload',lambda *args:{})
    original = requests.ask
    def ask(kind,*args,**kwargs):
        value = original(kind,*args,**kwargs)
        if kind == 'ground_edge':value['robustness'] = EVIDENCE
        return value
    requests.ask = ask
    clean = manager.prepare(edge,obs,tmp_path/'step')
    assert clean == candidate()
    assert 'robustness' not in manager.active['candidate']


def test_projection_inverts_nonrigid_support_height_correction():
    h,w = 200,300
    y,x = np.mgrid[:h,:w]
    z = 600 + 20*np.sin(x/40)
    camera = np.stack(((x-150)*z/900,(y-100)*z/910,z),axis=-1)
    affine = np.array([[0,-1,0,600],[-1,0,0,20],[.03,.02,-1,680],[0,0,0,1.]])
    world = (np.c_[camera.reshape(-1,3),np.ones(h*w)]@affine.T)[:,:3].reshape(h,w,3)
    obs = SimpleNamespace(xyz=world,meta=dict(world_from_camera_mm=np.eye(4).tolist(),
                                            corrected_world_from_camera_affine_mm=affine.tolist()))
    mapping = camera_mapping(obs)
    np.testing.assert_allclose(project([world[80,50]/1000,world[150,200]/1000],mapping),
                               [[50,80],[200,150]],atol=1e-7)
