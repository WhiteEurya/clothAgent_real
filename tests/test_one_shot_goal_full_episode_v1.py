import json
from pathlib import Path
from types import SimpleNamespace
import threading
import numpy as np

from PIL import Image
import pytest

pytestmark = pytest.mark.usefixtures("fake_one_shot_view_hardware")
from cloth_agent.experiments import one_shot_goal_full_episode_v1 as m
from cloth_agent.codex_motion import parse_code


def action(decision='ACTION'):
    return dict(decision=decision, grasp_px=[20, 30] if decision == 'ACTION' else None,
        target_px=[70, 40] if decision == 'ACTION' else None,
        motion_type='lift_pull_place' if decision == 'ACTION' else None, grasp_angle_deg=None, short_reason='拉开可见折边。')


def test_initialization_release_keeps_home_y_and_yaw_for_varying_grasps(monkeypatch):
    monkeypatch.setattr(m.reset, 'approach_program',
                        lambda *a: 'motion.gripper(arm="left", action="open")\n')
    base = np.eye(4); base[:3, 3] = [10, 20, 30]
    arm = SimpleNamespace(world_from_base=base, home_joints=np.zeros(6))
    model = SimpleNamespace(forward=lambda q: [400, 50, 500, 180, 0, 23])
    runtime = SimpleNamespace(config=SimpleNamespace(arms={'left': arm}), models={'left': model})
    home = m.matrix_pose(base @ m.pose_matrix(model.forward(arm.home_joints)))
    for seed in range(20):
        release = m.sample_initialization_release(seed)
        assert release == m.sample_initialization_release(seed)
        assert release['yaw_delta'] == 0 and release['drop_offset_xy'][1] == 0
        assert abs(release['drop_offset_xy'][0]) <= 40 and 250 <= release['lift_height'] <= 300
        for grasp in ([100, 200, 40], [180, -100, 55]):
            calls = parse_code(m.reset.reset_program(runtime, grasp, home[3:], **release))
            assert len(calls) == 6  # No extra yaw-rotation command.
            assert calls[-1][1]['action'] == 'open'
            drop = calls[-2][1]['left']
            assert drop[1] == pytest.approx(home[1]/1000)
            assert drop[5] == pytest.approx(np.radians(home[5]))
            assert drop[0] == pytest.approx((home[0]+release['drop_offset_xy'][0])/1000)
            assert drop[2] == pytest.approx((grasp[2]+release['lift_height'])/1000)


@pytest.fixture(autouse=True)
def offline_gate_transport(monkeypatch):
    """Existing runtime tests never dispatch the newly added shadow request."""
    real = m.PreActionGate
    def factory(folder, headers, request_once, timeout_s):
        def fake(*args):
            return {'decision': 'TRUST'}, {'wall_latency_s': .01}
        return real(folder, headers, fake, timeout_s)
    monkeypatch.setattr(m, 'PreActionGate', factory)


def test_transport_no_retry_persistent_attempt(tmp_path, monkeypatch):
    calls = []
    class Response:
        status_code = 503
        text = 'synthetic unavailable'
        def __enter__(self): return self
        def __exit__(self, *args): pass
    class Session:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def mount(self, prefix, adapter): assert adapter.max_retries.total == 0
        def post(self, *args, **kwargs):
            calls.append(kwargs)
            assert kwargs['allow_redirects'] is False
            return Response()
    monkeypatch.setattr(m.planner.requests, 'Session', Session)
    with pytest.raises(RuntimeError): m.direct_once(tmp_path, {}, {}, 2)
    with pytest.raises(FileExistsError): m.direct_once(tmp_path, {}, {}, 2)
    assert len(calls) == 1
    assert m.read(tmp_path/'model_metrics.json')['status'] == 'FAILED'


def test_multiview_prompt_and_three_rgb_no_history(tmp_path, monkeypatch):
    (tmp_path/'goal').mkdir(); (tmp_path/'before').mkdir(); (tmp_path/'right_view').mkdir()
    for path, size, color in (('goal/target.png', (100, 80), 'red'),
                              ('before/rgb.png', (100, 80), 'green'),
                              ('right_view/rgb.png', (200, 160), 'blue')):
        Image.new('RGB', size, color).save(tmp_path/path)
    (tmp_path/'prompt.txt').write_text(m.planner.ROTATING_MULTIVIEW_PROMPT)
    def request(folder, body, headers, timeout):
        assert body['instructions'] == m.planner.ROTATING_MULTIVIEW_PROMPT
        assert body['tools'] == [] and body['tool_choice'] == 'none'
        assert body['reasoning'] == {'effort': 'high'}
        assert len(body['input']) == 1 and len(body['input'][0]['content']) == 4
        content = body['input'][0]['content']
        assert [part['image_url'] for part in content[1:]] == [m.planner.image_data(tmp_path/path)
            for path in ('before/rgb.png', 'right_view/rgb.png', 'goal/target.png')]
        assert '100×80' in content[0]['text']
        assert '边缘或角点' in body['instructions']
        assert '必须全部使用第一张 perception 原图的像素坐标' in body['instructions']
        assert body['text']['format']['schema'] == m.planner.ROTATION_SCHEMA
        assert 'previous_response_id' not in body
        return action(), dict(wall_latency_s=1)
    monkeypatch.setattr(m, 'direct_once', request)
    m.plan_round(tmp_path, tmp_path, SimpleNamespace(image=tmp_path/'before/rgb.png'), {}, 2)
    assert set(m.read(tmp_path/'plan.json')) == set(action())
    manifest = m.read(tmp_path/'request_manifest.json')
    assert manifest['input_image_count'] == 3
    assert manifest['right_view_sha256'] == m.sha(tmp_path/'right_view/rgb.png')


@pytest.mark.parametrize('mode', ['done', 'abstain', 'reject', 'model_error', 'execution_error', 'limit'])
def test_closed_loop_stops_without_repair_or_extra_action(tmp_path, monkeypatch, mode):
    calls = []
    class Runtime:
        def __init__(self, config): self.cancel = threading.Event()
        def capture_current(self, path):
            calls.append('capture')
            path.mkdir()
            Image.new('RGB', (100, 80), 'purple').save(path/'rgb.png')
            return SimpleNamespace(image=path/'rgb.png')
        def arrival(self, initial): pass
        def _execute_operations(self, operations, folder):
            calls.append('dispatch')
            return dict(status='FAILED' if mode == 'execution_error' else 'EXECUTED')
        def move_to_perception(self, folder): calls.append('return')
        def stop(self): calls.append('stop')
        def close(self): calls.append('close')
    class Recorder:
        def __init__(self, *args): pass
        def start(self): pass
        def check(self): pass
        def stage(self, *args): pass
        def close(self): pass
    def init(runtime, folder, headers, timeout):
        calls.append('init'); folder.mkdir()
        m.save(folder/'result.json', dict(physical_actions=1, robot_execution_s=1))
    def plan(root, folder, before, headers, timeout):
        calls.append('model')
        m.save(folder/'attempt.json', {})
        m.save(folder/'model_metrics.json', dict(wall_latency_s=1))
        if mode == 'model_error': raise ValueError('invalid model output')
        decision = 'ACTION'
        if calls.count('model') > 1:
            decision = 'ABSTAIN' if mode == 'abstain' else 'DONE'
        p = action(decision); m.save(folder/'plan.json', p)
        return p, dict(wall_latency_s=1)
    def prepare(*args):
        calls.append('checks')
        if mode == 'reject': raise m.DualArmError('depth rejected')
        return [], {}
    monkeypatch.setattr(m, 'EpisodeCollectionRuntime', Runtime)
    monkeypatch.setattr(m, 'TrajectoryRecorder', Recorder)
    monkeypatch.setattr(m, 'initialize', init)
    monkeypatch.setattr(m, 'plan_round', plan)
    monkeypatch.setattr(m, 'prepare_action', prepare)
    monkeypatch.setattr(m.planner, 'auth_headers', lambda: {})
    goal = tmp_path/'goal.png'; Image.new('RGB', (100, 80)).save(goal)
    root = tmp_path/'output'
    args = SimpleNamespace(output=root, goal_image=goal, config=Path('unused'),
                           max_rounds=1 if mode == 'limit' else 12, timeout_s=2)
    m.run(args)
    s = m.read(root/'episode_summary.json')
    assert calls.count('init') == 1
    assert calls.count('model') == (2 if mode in ('done', 'abstain') else 1)
    assert calls.count('dispatch') == (0 if mode in ('reject', 'model_error') else 1)
    assert s['termination_reason'] == dict(done='DONE', abstain='ABSTAIN', limit='ROUND_LIMIT').get(mode, 'SAFETY_STOP')
    assert s['safety_rejections'] == int(mode == 'reject')
    assert s['human_fold_ready'] is None and m.read(root/'human_review.json')['choice'] is None
    assert s['total_planning_s'] == calls.count('model')
    assert (root/'report.md').exists() and 'report_error' not in s
    if mode in ('reject', 'model_error', 'execution_error'):
        assert calls.count('stop') == 1
        assert (root/'round_000/after/unavailable.json').exists()
    else: assert (root/'round_000/after/rgb.png').exists()


@pytest.mark.parametrize('decision,matching', [('ACCEPT', True), ('REJECT', True), ('ACCEPT', False)])
def test_initialization_review_binds_images_and_fails_closed(tmp_path, decision, matching):
    folder=tmp_path/'initialization'
    for phase in ('before','after'):
        (folder/phase).mkdir(parents=True)
        Image.new('RGB',(100,80),'purple').save(folder/phase/'rgb.png')
    m.save(folder/'review_decision.json',dict(decision=decision,reason='test',
        before_sha256=m.sha(folder/'before/rgb.png'),
        after_sha256=m.sha(folder/'after/rgb.png') if matching else 'stale'))
    runtime=SimpleNamespace(cancel=threading.Event())
    if decision=='ACCEPT' and matching:
        m.wait_initialization_review(runtime,tmp_path)
        assert (folder/'review_result.json').exists()
    else:
        with pytest.raises(m.DualArmError):m.wait_initialization_review(runtime,tmp_path)
        assert not (folder/'review_result.json').exists()


def test_authorized_continue_fresh_observation_no_init_no_history_to_model(tmp_path,monkeypatch):
    src=tmp_path/'previous';(src/'goal').mkdir(parents=True)
    Image.new('RGB',(100,80),'purple').save(src/'goal/target.png')
    m.save(src/'episode_summary.json',dict(experiment=m.ROOT.name,termination_reason='SAFETY_STOP',
        total_rounds=7,physical_actions=6,safety_rejections=1,total_planning_s=147.,total_robot_execution_s=390.,rounds=[]))
    m.save(src/'protocol.json',dict(prompt=m.planner.ROTATING_MULTIVIEW_PROMPT,schema=m.planner.ROTATION_SCHEMA,model=m.planner.MODEL,effort=m.planner.EFFORT))
    calls=[]
    class Runtime:
        def __init__(self,*args):self.cancel=threading.Event()
        def capture_current(self,path):
            calls.append('capture');path.mkdir();Image.new('RGB',(100,80),'purple').save(path/'rgb.png')
            return SimpleNamespace(image=path/'rgb.png')
        def observe(self,path):calls.append('observe');return self.capture_current(path)
        def move_to_perception(self,path):calls.append('return')
        def close(self):calls.append('close')
    def plan(root,folder,before,headers,timeout_s):
        calls.append('model');assert (root/'startup_observation/rgb.png').exists()
        p=action('DONE');m.save(folder/'plan.json',p);m.save(folder/'attempt.json',{})
        return p,dict(wall_latency_s=1.)
    def forbidden(*args,**kwargs):raise AssertionError('must not initialize')
    monkeypatch.setattr(m,'EpisodeCollectionRuntime',Runtime)
    monkeypatch.setattr(m,'initialize',forbidden)
    monkeypatch.setattr(m,'plan_round',plan)
    monkeypatch.setattr(m.planner,'auth_headers',lambda:{})
    args=SimpleNamespace(continue_from=src,output=tmp_path/'continued',goal_image=Path('unused'),
        config=Path('unused'),max_rounds=5,timeout_s=2)
    m.run(args)
    s=m.read(args.output/'episode_summary.json')
    assert s['termination_reason']=='DONE' and s['initialization_physical_actions']==0
    assert s['cumulative_rounds']==8 and s['cumulative_physical_actions']==6
    assert s['continuation_source_unchanged'] and calls.count('model')==1
    args.output=tmp_path/'excess';args.max_rounds=6
    with pytest.raises(ValueError,match='remaining'):m.run(args)
    assert not args.output.exists()
