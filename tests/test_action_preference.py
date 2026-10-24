import threading
from types import SimpleNamespace

import pytest
from PIL import Image

from cloth_agent import action_preference as p
from cloth_agent import preference_one_shot as scoring


def rows():
    return [dict(id=f'P{i:02d}R',point_id=f'P{i:02d}',grasp_px=[20+i*3,30],target_px=[70+i*3,40],
                 grasp_angle_deg=0,Ps=100-i*5,Pt=i*8,point_reason='可见边缘。',trajectory_reason='向外展开折边。') for i in range(1,11)]


def test_preference_updates_follow_tradeoff_and_lower_loss():
    candidates=rows()
    higher_pt=p.learn(candidates,candidates[-1]['id'],1.)
    lower_pt=p.learn(candidates,candidates[0]['id'],1.)
    assert higher_pt['alpha_after']>1 and lower_pt['alpha_after']<1
    for update in (higher_pt,lower_pt):
        assert update['loss_after']<update['loss_before']
        assert abs(update['alpha_after']-1)<=.25
    assert p.learn(candidates,candidates[-1]['id'],5)['alpha_after']<=5
    assert p.learn(candidates,candidates[0]['id'],0)['alpha_after']>=0


def test_unidentifiable_or_dominated_choice_is_not_fabricated():
    candidates=rows()
    for c in candidates:c['Pt']=50
    event=p.learn(candidates,candidates[-1]['id'],1.)
    assert event['alpha_after']==pytest.approx(1)
    assert not event['chosen_can_be_top1_with_scalar']
    assert p.ranked(candidates,event['alpha_after'])[0]['id']!=candidates[-1]['id']


def test_feedback_persistent_idempotent_and_stale_rejected(tmp_path):
    path=tmp_path/'alpha.json';state=p.load_state(path)
    manifest=dict(review_id='one',displayed=rows(),state_revision=0,alpha=1.)
    event=p.apply_feedback(path,manifest,'P10R')
    assert p.load_state(path)['revision']==1
    assert p.apply_feedback(path,manifest,'P10R')==event
    assert p.load_state(path)['revision']==1
    with pytest.raises(ValueError,match='another choice'):p.apply_feedback(path,manifest,'P01R')
    with pytest.raises(ValueError,match='stale'):p.apply_feedback(path,dict(manifest,review_id='two'),'P10R')
    with pytest.raises(ValueError,match='Top10'):p.apply_feedback(path,manifest,'not_shown')
    assert state['alpha']==1.


def test_choice_binds_scene_top10_and_review_token():
    manifest=dict(review_id='a',token='b',displayed=rows())
    body=dict(review_id='a',token='b',decision='SELECT',candidate_id='P10R',scene_unchanged=True)
    assert p.validate_choice(body,manifest)['candidate_id']=='P10R'
    for change in ({'token':'old'},{'review_id':'old'},{'candidate_id':'absent'},{'scene_unchanged':False},{'decision':'AUTO'}):
        with pytest.raises(ValueError):p.validate_choice(dict(body,**change),manifest)
    assert p.validate_choice(dict(body,decision='STOP'),manifest)['decision']=='STOP'


def test_choose_does_not_replace_choice_when_alpha_changes(tmp_path,monkeypatch):
    root=tmp_path;folder=root/'round_000';folder.mkdir();(root/'goal').mkdir()
    before=SimpleNamespace(image=root/'current.png')
    for target in (before.image,root/'goal/target.png'):Image.new('RGB',(150,100),'purple').save(target)
    candidates=rows();chosen=candidates[0]
    def wait(directory,manifest,cancel,**kwargs):
        assert len(manifest['displayed'])==10
        return dict(decision='SELECT',candidate_id=chosen['id']),12.
    monkeypatch.setattr(p,'wait_for_choice',wait)
    timing={};state=root/'state.json'
    plan=scoring.choose_plan(root,folder,before,dict(candidates=candidates),SimpleNamespace(cancel=threading.Event()),timing,state,0,1)
    assert plan['grasp_px']==chosen['grasp_px']
    assert p.load_state(state)['alpha']<1
    assert timing['human_review_wait_s']==12
    assert p.ranked(candidates,p.load_state(state)['alpha'])[0]['id']!=chosen['id']


def test_stop_does_not_learn(tmp_path,monkeypatch):
    folder=tmp_path/'round';folder.mkdir();(tmp_path/'goal').mkdir()
    image=tmp_path/'current.png'
    for target in (image,tmp_path/'goal/target.png'):Image.new('RGB',(150,100)).save(target)
    monkeypatch.setattr(p,'wait_for_choice',lambda *a,**kw:(dict(decision='STOP'),1.))
    state=tmp_path/'state.json'
    with pytest.raises(KeyboardInterrupt):
        scoring.choose_plan(tmp_path,folder,SimpleNamespace(image=image),dict(candidates=rows()),
                            SimpleNamespace(cancel=threading.Event()),{},state,0,1)
    assert p.load_state(state)['revision']==0


def test_scene_change_rejects_local_movement(tmp_path):
    a=tmp_path/'before.png';b=tmp_path/'fresh.png'
    Image.new('RGB',(1280,720),'purple').save(a)
    im=Image.open(a);im.save(b)
    before=SimpleNamespace(image=a,meta={'calibration_id':'c'})
    fresh=SimpleNamespace(image=b,meta={'calibration_id':'c'})
    scoring.verify_scene(before,fresh,dict(grasp_px=[700,500]),tmp_path)
    from PIL import ImageDraw
    ImageDraw.Draw(im).rectangle((680,480,720,520),fill='white');im.save(b)
    with pytest.raises(Exception,match='scene changed'):scoring.verify_scene(before,fresh,dict(grasp_px=[700,500]),tmp_path)


def test_ps_and_pt_are_independent_requests_with_clean_images(tmp_path,monkeypatch):
    folder=tmp_path/'round';folder.mkdir();(tmp_path/'goal').mkdir()
    image=tmp_path/'current.png'
    for target in (image,tmp_path/'goal/target.png'):Image.new('RGB',(300,200),'purple').save(target)
    before=SimpleNamespace(image=image,validate_for=lambda *a,**kw:None)
    points=dict(decision='ACTION',short_reason='提出可见抓点。',points=[dict(id=f'P{i:02d}',grasp_px=[20+i*3,30],Ps=70,
                 grasp_angle_deg=None,reason='PS_PRIVATE_REASON') for i in range(1,11)])
    candidates=rows();calls=[]
    def request(path,body,headers,timeout):
        calls.append(body)
        images=[c['image_url'] for c in body['input'][0]['content'] if c['type']=='input_image']
        assert images==[scoring.io.image_data(image),scoring.io.image_data(tmp_path/'goal/target.png')]
        assert 'right view' not in body['instructions'] and '右视角' not in body['instructions']
        if len(calls)==1:
            assert body['instructions']==scoring.PS_PROMPT;return points,dict(wall_latency_s=2)
        assert body['instructions']==scoring.PT_PROMPT
        import json
        extra=body['input'][0]['content'][-1]['text']
        assert 'PS_PRIVATE_REASON' not in extra
        assert all('Ps' not in c and 'point_reason' not in c for c in json.loads(extra))
        return dict(scores=[dict(id=r['id'],Pt=r['Pt'],reason=r['trajectory_reason']) for r in candidates]),dict(wall_latency_s=3)
    monkeypatch.setattr(scoring,'generate_candidates',lambda *a:candidates)
    bundle,metrics=scoring.score_round(tmp_path,folder,before,{},2,object(),request)
    assert len(calls)==2 and len(bundle['candidates'])==10 and metrics['wall_latency_s']==5
    assert not (folder/'plan.json').exists()  # Generation alone cannot choose an executable action.


def test_invalid_scores_and_duplicate_points_rejected():
    for score in (float('nan'),float('inf'),-1,101,True):
        bad=rows();bad[0]['Ps']=score
        with pytest.raises(ValueError):p.ranked(bad,1)
    output=dict(decision='ACTION',short_reason='抓取。',points=[dict(id=f'P{i:02d}',grasp_px=[20,30],Ps=70,
                 grasp_angle_deg=None,reason='边缘') for i in range(1,11)])
    with pytest.raises(ValueError,match='duplicate'):scoring.validate_points(output,[100,100])


def test_candidate_filter_creates_depth_audits_and_never_moves_pixels(tmp_path,monkeypatch):
    import numpy as np
    image=tmp_path/'current.png';Image.new('RGB',(300,200)).save(image)
    before=SimpleNamespace(image=image,sample=lambda pixel,spread:np.array([pixel[0]/2,pixel[1]/2,10.]))
    arm=SimpleNamespace(world_from_base=np.eye(4),validate_point=lambda point:None)
    config=SimpleNamespace(arms={'left':arm},limits={'max_surface_spread_mm':8})
    def target(obs,pb,pa,spread,path):
        path.write_text('{}')  # Requires the real generator to create its parent directory.
        return before.sample(pb,spread)
    monkeypatch.setattr(scoring.table_target_depth,'sample_target',target)
    points=[dict(id=f'P{i:02d}',grasp_px=[80+3*i,90],Ps=70,grasp_angle_deg=None,reason='边缘') for i in range(1,11)]
    actions=scoring.generate_candidates(points,before,config,tmp_path)
    assert len(actions)==10 and len(list((tmp_path/'candidate_depth').glob('*.json')))==10
    for action,point in zip(actions,points):
        assert action['grasp_px']==point['grasp_px']
        assert action['target_px']==[point['grasp_px'][0]+120,90]
    assert len(p.io.read(tmp_path/'candidate_filter.json')['rejected'])==30


@pytest.mark.parametrize('mode',['select','stop','timeout','changed_scene'])
@pytest.mark.parametrize('correction',[False,True])
def test_episode_requires_human_choice_and_preserves_selected_action(tmp_path,monkeypatch,mode,correction):
    from cloth_agent.experiments import one_shot_goal_full_episode_v1 as episode
    from cloth_agent import residual_features, ranking_residual
    def forbidden_view(*a,**kw):raise AssertionError('preference rollout must not use right view')
    monkeypatch.setattr(episode.one_shot_views,'view_target',forbidden_view)
    monkeypatch.setattr(episode.one_shot_views,'plan_with_right_view',forbidden_view)
    monkeypatch.setattr(episode.one_shot_views,'RIGHT_VIEW',tmp_path/'missing_right_view.json')
    calls=[];candidates=rows();chosen=candidates[0]
    class Runtime:
        def __init__(self,config):self.cancel=threading.Event();self.config=object()
        def capture_current(self,path):
            calls.append('capture:'+path.name);path.mkdir()
            color='green' if mode=='changed_scene' and path.name=='after_human_review' else 'purple'
            Image.new('RGB',(150,100),color).save(path/'rgb.png')
            return SimpleNamespace(image=path/'rgb.png',meta={'calibration_id':'test'})
        def move_to_perception(self,path):calls.append('returned')
        def arrival(self,*a):pass
        def _execute_operations(self,*args):calls.append('dispatch');return dict(status='EXECUTED')
        def stop(self):calls.append('stop')
        def close(self):pass
    class Recorder:
        def __init__(self,*a):pass
        def start(self):pass
        def stage(self,*a):pass
        def check(self):pass
        def close(self):pass
    class Gate:
        def __init__(self,*a):self.metrics=dict(gate_wall_s=0.,status='VALID')
        def evaluate(self,*a):return dict(decision='TRUST')
    def init(runtime,folder,*a):
        folder.mkdir();episode.save(folder/'result.json',dict(physical_actions=0,robot_execution_s=0.))
    def score(root,folder,*a):
        episode.save(folder/'attempt.json',{})
        episode.save(folder/'model_metrics.json',dict(wall_latency_s=5))
        return dict(candidates=candidates),dict(wall_latency_s=5)
    def wait(directory,manifest,*a,**kw):
        assert 'returned' not in calls and 'dispatch' not in calls
        calls.append('human')
        if mode=='timeout':raise TimeoutError('no human response')
        return (dict(decision='STOP') if mode=='stop' else dict(decision='SELECT',candidate_id=chosen['id'],
            human_order=[r['id'] for r in manifest['displayed']][::-1])),8.
    def prepare(runtime,before,plan,folder):
        assert before.image.parent.name=='after_human_review'
        assert plan['grasp_px']==chosen['grasp_px'] and plan['target_px']==chosen['target_px']
        calls.append('checks');return [],{}
    monkeypatch.setattr(episode,'EpisodeCollectionRuntime',Runtime)
    monkeypatch.setattr(episode,'TrajectoryRecorder',Recorder)
    monkeypatch.setattr(episode,'PreActionGate',Gate)
    monkeypatch.setattr(episode,'initialize',init)
    monkeypatch.setattr(episode,'prepare_action',prepare)
    monkeypatch.setattr(episode.planner,'auth_headers',lambda:{})
    monkeypatch.setattr(scoring,'score_round',score)
    monkeypatch.setattr(p,'wait_for_choice',wait)
    monkeypatch.setattr(residual_features,'extract',lambda image,*a:dict(wall_s=.1,image_sha256=p.io.sha(image),
        identity=dict(version=residual_features.VERSION,dimension=residual_features.DIM),
        features={r['id']:[0.]*residual_features.DIM for r in candidates}))
    goal=tmp_path/'goal.png';Image.new('RGB',(150,100),'purple').save(goal)
    out=tmp_path/'episode';state=tmp_path/'alpha.json'
    args=SimpleNamespace(output=out,goal_image=goal,config=tmp_path/'unused',max_rounds=1,timeout_s=1,
                         human_preference=True,alpha_state=state,preference_port=0,preference_timeout_s=1,
                         ranking_correction=correction,residual_state=tmp_path/'residual.json')
    episode.run(args)
    summary=episode.read(out/'episode_summary.json')
    assert calls.count('dispatch')==int(mode=='select')
    assert p.load_state(state)['revision']==int(not correction and mode in ('select','changed_scene'))
    if correction:
        assert ranking_residual.load_state(args.residual_state,1.)['revision']==int(mode in ('select','changed_scene'))
    assert summary['total_planning_s']==5
    protocol=episode.read(out/'protocol.json')
    assert protocol['observation_views']==['perception'] and protocol['input_image_count']==2
    assert protocol['model_overlaps_view_return'] is False
    assert not (out/'right_view_pose.json').exists() and not (out/'round_000/right_view').exists()
    assert not (out/'round_000/preference_review/right.png').exists()
    review_html=(out/'round_000/preference_review/index.html').read_text()
    assert 'right.png' not in review_html and '右视角' not in review_html
    assert 'report_error' not in summary
    if mode=='select':
        assert calls.index('human')<calls.index('checks')<calls.index('dispatch')
        assert summary['rounds'][0]['timing']['human_review_wait_s']==8
    if mode=='changed_scene':assert 'scene changed' in summary['termination_detail']
