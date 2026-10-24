import threading
from types import SimpleNamespace
import pytest
from PIL import Image
from cloth_agent import automatic_top5 as auto, preference_one_shot as scoring
from cloth_agent.experiments import automatic_top5_batch as batch


def rows():
    return [dict(id=f'P{i:02d}R',point_id=f'P{i:02d}',grasp_px=[20+i*3,30],target_px=[70+i*3,40],
        grasp_angle_deg=0,Ps=100-i*5,Pt=i*8,point_reason='可见边缘。',trajectory_reason='展开折边。') for i in range(1,11)]


def selection():
    return dict(selected_id='P01R',selection_reason='该处可见边缘更可靠。',pa_identity='折边。',
        expected_change='打开左侧覆盖。',uncertainty='布层不明。',failure_risk='整体拖动。',
        completion_check=dict(torso_not_covered=False,both_sleeves_out=False,
                              hem_open=False,no_major_fold=False,evidence='衣身仍大面积折叠。'))


def test_auto_uses_explicit_choice_not_pt_or_learned_scores(tmp_path,monkeypatch):
    im=tmp_path/'current.png';Image.new('RGB',(200,100),'purple').save(im)
    def forbidden(*a,**kw):raise AssertionError('learned state must not be read')
    monkeypatch.setattr(scoring.pref,'load_state',forbidden)
    output=auto.choose(tmp_path,SimpleNamespace(image=im),dict(candidates=rows(),selection=selection()),threading.Event())
    assert output['grasp_px']==rows()[0]['grasp_px']
    assert auto.io.read(tmp_path/'automatic_selection.json')['Pt_rank']==10
    top=auto.io.read(tmp_path/'trajectories_top5.json')['actions']
    assert len(top)==5 and top[0]['id']=='P10R' and all(r['id']!='P01R' for r in top)
    bad=selection();bad['selected_id']='absent'
    with pytest.raises(ValueError):auto.choose(tmp_path,SimpleNamespace(image=im),dict(candidates=rows(),selection=bad),threading.Event())
    cancel=threading.Event();cancel.set()
    with pytest.raises(KeyboardInterrupt):auto.choose(tmp_path,SimpleNamespace(image=im),dict(candidates=rows(),selection=selection()),cancel)


def test_auto_can_stop_after_scoring_without_dispatch(tmp_path):
    im=tmp_path/'current.png';Image.new('RGB',(200,100),'purple').save(im)
    answer=dict(selection(),selected_id='DONE',selection_reason='主要结构完全展开，局部小皱褶可接受。',
                completion_check=dict(torso_not_covered=True,both_sleeves_out=True,
                                      hem_open=True,no_major_fold=True,evidence='四项结构条件均可见。'))
    plan=auto.choose(tmp_path,SimpleNamespace(image=im),dict(candidates=rows(),selection=answer),threading.Event())
    assert plan['decision']=='DONE' and plan['grasp_px'] is None
    assert auto.io.read(tmp_path/'automatic_selection.json')['selected_id']=='DONE'
    assert len(auto.io.read(tmp_path/'trajectories_top5.json')['actions'])==5
    assert 'DONE' in auto.trajectory_schema([r['id'] for r in rows()])['properties']['selected_id']['enum']
    with pytest.raises(ValueError,match='selected_id conflicts'):
        auto.choose(tmp_path,SimpleNamespace(image=im),
                    dict(candidates=rows(),selection=dict(selection(),selected_id='DONE')),
                    threading.Event())


def test_completion_check_blocks_conflicting_done_and_action():
    ready=dict(torso_not_covered=True,both_sleeves_out=True,hem_open=True,no_major_fold=True,
               evidence='衣身、袖子和下摆均展开。')
    not_ready=dict(ready,hem_open=False,evidence='下摆仍分成两片。')
    done=dict(decision='DONE',points=[],short_reason='已展开。',completion_check=ready)
    assert scoring.validate_points(done,[200,100],automatic=True)['decision']=='DONE'
    with pytest.raises(ValueError,match='DONE conflicts'):
        scoring.validate_points(dict(done,completion_check=not_ready),[200,100],automatic=True)
    action=dict(decision='ACTION',short_reason='下摆仍分成两片。',
                points=[dict(id=f'P{i:02d}',grasp_px=[20+i*3,30],Ps=50,
                             grasp_angle_deg=0,reason='可见边缘。') for i in range(1,11)],
                completion_check=not_ready)
    assert scoring.validate_points(action,[200,100],automatic=True) is None
    with pytest.raises(ValueError,match='ACTION conflicts'):
        scoring.validate_points(dict(action,completion_check=ready),[200,100],automatic=True)


def test_auto_scoring_two_requests_and_points_saved_before_trajectory(tmp_path,monkeypatch):
    folder=tmp_path/'round';folder.mkdir();(tmp_path/'goal').mkdir()
    im=tmp_path/'current.png'
    for p in (im,tmp_path/'goal/target.png'):Image.new('RGB',(200,100),'purple').save(p)
    points=dict(decision='ACTION',short_reason='选择可见点。',points=[dict(id=r['point_id'],grasp_px=r['grasp_px'],Ps=r['Ps'],grasp_angle_deg=0,reason=r['point_reason']) for r in rows()],
                completion_check=dict(torso_not_covered=False,both_sleeves_out=False,
                                      hem_open=False,no_major_fold=False,evidence='仍有大块覆盖。'))
    calls=[]
    def request(path,body,*a):
        calls.append(body)
        assert sum(v['type']=='input_image' for v in body['input'][0]['content'])==2
        if len(calls)==1:return points,dict(wall_latency_s=1)
        assert (folder/'points_top5.json').exists()
        assert body['instructions']==auto.PROMPT
        assert 'selected_id' in body['text']['format']['schema']['required']
        return dict(selection(),scores=[dict(id=r['id'],Pt=r['Pt'],reason=r['trajectory_reason']) for r in rows()]),dict(wall_latency_s=2)
    monkeypatch.setattr(scoring,'generate_candidates',lambda *a:rows())
    bundle,metrics=scoring.score_round(tmp_path,folder,SimpleNamespace(image=im,validate_for=lambda *a,**kw:None),{},1,object(),request,automatic=True)
    assert len(calls)==2 and metrics['wall_latency_s']==3 and bundle['selection']['selected_id']=='P01R'
    assert len(auto.io.read(folder/'points_top5.json')['points'])==5


def test_controls_stop_current_separately_and_require_fresh_ready(tmp_path,monkeypatch):
    c=batch.Control(tmp_path,3);calls=[]
    monkeypatch.setattr(c,'forward_stop',lambda:calls.append('stop'))
    c.transition('RUNNING',1)
    old=dict(token=c.token,episode=1)
    with pytest.raises(ValueError):c.command(dict(old,command='next'))
    c.command(dict(old,command='stop_current'))
    assert c.phase=='STOPPING' and c.episode_stop and not c.cancel.is_set() and calls==['stop']
    c.transition('WAITING_READY')
    with pytest.raises(ValueError):c.command(dict(old,command='next'))
    body=dict(token=c.token,episode=1,command='next')
    c.command(body)
    assert c.ready.is_set() and c.phase=='STARTING_NEXT'
    with pytest.raises(ValueError):c.command(body)
    c.command(dict(token=c.token,episode=1,command='stop_all'))
    assert c.cancel.is_set()


@pytest.mark.parametrize('wait_first',[False,True])
def test_batch_waits_for_operator_after_stopped_episode(tmp_path,monkeypatch,wait_first):
    import time
    controls=[];launches=[];operator_errors=[]
    class Server:
        def shutdown(self):pass
        def server_close(self):pass
    monkeypatch.setattr(batch,'serve',lambda c,p:(controls.append(c) or Server()))
    def launch(command,log,c):
        folder=__import__('pathlib').Path(command[command.index('--output')+1]);folder.mkdir()
        launches.append(folder)
        assert '--automatic-top5' in command and '--human-preference' not in command
        assert float(command[command.index('--initialize-center-radius-mm')+1])==20
        assert '--press-after-drop' in command
        batch.base.save(folder/'episode_summary.json',dict(termination_reason='USER_STOP' if len(launches)==1 else 'DONE',
            finished_at='test',physical_actions=1,total_rounds=1))
        return 0
    monkeypatch.setattr(batch.base,'launch_episode',launch)
    def operator():
        try:
            deadline=time.monotonic()+5
            while (not controls or controls[0].phase!='WAITING_READY') and time.monotonic()<deadline:time.sleep(.005)
            assert controls and controls[0].phase=='WAITING_READY'
            if wait_first:
                c=controls[0];assert c.episode==0
                time.sleep(.05);assert launches==[]
                c.command(dict(command='next',token=c.token,episode=0))
                while c.phase!='WAITING_READY' and time.monotonic()<deadline:time.sleep(.005)
                assert c.phase=='WAITING_READY' and c.episode==1
            time.sleep(.05);assert len(launches)==1
            c=controls[0];c.command(dict(command='next',token=c.token,episode=1))
        except BaseException as exc:
            operator_errors.append(exc)
            if controls:controls[0].request_stop()
    thread=threading.Thread(target=operator,daemon=True);thread.start()
    goal=tmp_path/'goal.png';Image.new('RGB',(100,100)).save(goal)
    config=tmp_path/'config.json';config.write_text('{}')
    out=tmp_path/'batch'
    batch.run(SimpleNamespace(output=out,episodes=2,max_rounds=12,port=8769,goal_image=goal,config=config,
                              wait_before_first=wait_first,initialize_center_radius_mm=20.,press_after_drop=True))
    thread.join(1)
    assert not operator_errors and len(launches)==2 and controls[0].phase=='FINISHED'


def test_batch_continues_first_episode_without_reset_or_press(tmp_path,monkeypatch):
    class Server:
        def shutdown(self):pass
        def server_close(self):pass
    monkeypatch.setattr(batch,'serve',lambda *a:Server())
    source=tmp_path/'stopped';(source/'goal').mkdir(parents=True)
    Image.new('RGB',(100,100)).save(source/'goal/target.png')
    batch.base.save(source/'episode_summary.json',dict(termination_reason='SAFETY_STOP'))
    commands=[]
    def launch(command,log,session):
        commands.append(command)
        assert '--stateful-context' in command
        folder=__import__('pathlib').Path(command[command.index('--output')+1]);folder.mkdir()
        batch.base.save(folder/'episode_summary.json',dict(termination_reason='DONE',finished_at='test',
                                                          physical_actions=1,total_rounds=1))
        return 0
    monkeypatch.setattr(batch.base,'launch_episode',launch)
    goal=tmp_path/'unused_goal.png';Image.new('RGB',(100,100)).save(goal)
    config=tmp_path/'config.json';config.write_text('{}')
    out=tmp_path/'batch'
    batch.run(SimpleNamespace(output=out,episodes=1,max_rounds=12,port=8770,goal_image=goal,config=config,
                              resume_first_from=source,initialize_center_radius_mm=20.,press_after_drop=False,
                              stateful_context=True))
    assert commands[0][commands[0].index('--continue-from')+1]==str(source)
    assert '--press-after-drop' not in commands[0]
    assert batch.base.json.loads((out/'protocol.json').read_text())['resume_first_from']==str(source)
    assert batch.base.json.loads((out/'protocol.json').read_text())['stateful_context'] is True


def test_contextual_scoring_uses_previous_clean_image_and_records_effect(tmp_path,monkeypatch):
    from cloth_agent.experiments import one_shot_goal_planner_probe_v1 as io
    (tmp_path/'goal').mkdir()
    current=tmp_path/'current.png';previous=tmp_path/'previous.png'
    for path,color in ((current,'purple'),(previous,'blue'),(tmp_path/'goal/target.png','white')):
        Image.new('RGB',(200,100),color).save(path)
    folder=tmp_path/'round_001';folder.mkdir()
    points=dict(decision='ACTION',short_reason='当前仍需展开。',
        points=[dict(id=f'P{i:02d}',grasp_px=[20+i*3,30],Ps=100-i*5,
                     grasp_angle_deg=0,reason='可见边缘。') for i in range(1,11)],
        previous_effect=dict(verdict='IMPROVED',changed_relation='左下折边被拉开。',evidence='露出的衣片变宽。'),
        completion_check=dict(torso_not_covered=False,both_sleeves_out=False,
                              hem_open=False,no_major_fold=False,evidence='仍有大块覆盖。'))
    monkeypatch.setattr(scoring,'generate_candidates',lambda *a:rows())
    calls=[]
    def request(path,body,*a):
        calls.append(body)
        content=body['input'][0]['content']
        images=[v for v in content if v['type']=='input_image']
        assert len(images)==3 and images[2]['image_url']==io.image_data(previous)
        assert 'Pa=[40, 50]' in str(content) and 'Pb=[40, 70]' in str(content)
        if len(calls)==1:
            assert body['text']['format']['schema']==scoring.AUTO_PS_CONTEXT_SCHEMA
            return points,dict(wall_latency_s=1)
        assert '左下折边被拉开' in str(content)
        return dict(selection(),scores=[dict(id=r['id'],Pt=r['Pt'],reason=r['trajectory_reason']) for r in rows()]),dict(wall_latency_s=2)
    context=dict(source_round='round_000',before_image=previous,after_image=current,
                 plan=dict(grasp_px=[40,50],target_px=[40,70],grasp_angle_deg=90))
    bundle,metrics=scoring.score_round(tmp_path,folder,
        SimpleNamespace(image=current,validate_for=lambda *a,**kw:None),{},1,object(),request,
        automatic=True,previous_context=context)
    assert len(calls)==2 and metrics['wall_latency_s']==3
    assert bundle['previous_effect']['verdict']=='IMPROVED'
    assert io.read(folder/'previous_action_context.json')['input_images']==[
        'current RGB','goal RGB','previous action before RGB']
    assert io.read(folder/'previous_effect.json')['changed_relation']=='左下折边被拉开。'


@pytest.mark.parametrize('mode',['execute','scene_change','cancel'])
def test_automatic_episode_no_human_or_learning_and_scene_guard(tmp_path,monkeypatch,mode):
    from cloth_agent.experiments import one_shot_goal_full_episode_v1 as episode
    from cloth_agent import ranking_residual
    calls=[]
    def forbidden(*a,**kw):raise AssertionError('unexpected learned state/human/right view')
    for obj,name in [(scoring.pref,'load_state'),(scoring.pref,'wait_for_choice'),(ranking_residual,'load_state'),
                     (episode.one_shot_views,'view_target'),(episode.one_shot_views,'plan_with_right_view')]:monkeypatch.setattr(obj,name,forbidden)
    class Runtime:
        def __init__(self,*a):self.cancel=threading.Event();self.config=object()
        def observe(self,path):return self.capture_current(path)
        def capture_current(self,path):
            path.mkdir();Image.new('RGB',(200,100),'green' if mode=='scene_change' and path.name=='pre_execution_observation' else 'purple').save(path/'rgb.png')
            return SimpleNamespace(image=path/'rgb.png',meta={'calibration_id':'test'})
        def arrival(self,*a):pass
        def _execute_operations(self,*a):calls.append('dispatch');return dict(status='EXECUTED')
        def move_to_perception(self,*a):pass
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
        if mode=='cancel':runtime.cancel.set()
    def score(root,folder,before,*a,**kw):
        assert kw==dict(automatic=True)
        episode.save(folder/'attempt.json',{});episode.save(folder/'model_metrics.json',dict(wall_latency_s=2))
        return dict(candidates=rows(),selection=selection()),dict(wall_latency_s=2)
    def prepare(runtime,before,plan,folder):
        assert before.image.parent.name=='pre_execution_observation' and plan['grasp_px']==rows()[0]['grasp_px']
        calls.append('checks');return [],{}
    for name,val in [('EpisodeCollectionRuntime',Runtime),('TrajectoryRecorder',Recorder),('PreActionGate',Gate),('initialize',init),('prepare_action',prepare)]:monkeypatch.setattr(episode,name,val)
    monkeypatch.setattr(episode.planner,'auth_headers',lambda:{})
    monkeypatch.setattr(scoring,'score_round',score)
    goal=tmp_path/'goal.png';Image.new('RGB',(200,100),'purple').save(goal)
    out=tmp_path/'episode'
    episode.run(SimpleNamespace(output=out,goal_image=goal,config=tmp_path/'unused',max_rounds=1,timeout_s=1,automatic_top5=True))
    s=episode.read(out/'episode_summary.json');protocol=episode.read(out/'protocol.json')
    assert calls.count('dispatch')==int(mode=='execute')
    assert protocol['observation_views']==['perception'] and not protocol['human_review_required']
    assert not s['human_preference_enabled'] and not s['ranking_correction_enabled']
    assert 'report_error' not in s
    assert (out/'round_000/outcome_evidence.json').exists()
    assert s['termination_reason']==dict(execute='ROUND_LIMIT',scene_change='SAFETY_STOP',cancel='USER_STOP')[mode]
    if mode=='scene_change':
        continued=tmp_path/'continued'
        episode.run(SimpleNamespace(output=continued,goal_image=goal,config=tmp_path/'unused',max_rounds=1,
                                    timeout_s=1,automatic_top5=True,continue_from=out))
        assert (continued/'startup_observation/rgb.png').exists()
        assert not (continued/'initialization').exists()
        assert episode.read(continued/'continuation.json')['source_episode']==str(out)


def test_stateful_episode_uses_only_last_executed_action_on_next_round(tmp_path,monkeypatch):
    from cloth_agent.experiments import one_shot_goal_full_episode_v1 as episode
    class Runtime:
        def __init__(self,*a):self.cancel=threading.Event();self.config=object()
        def capture_current(self,path):
            path.mkdir();Image.new('RGB',(200,100),'purple').save(path/'rgb.png')
            return SimpleNamespace(image=path/'rgb.png',meta={'calibration_id':'test'})
        def arrival(self,*a):pass
        def _execute_operations(self,*a):return dict(status='EXECUTED')
        def move_to_perception(self,*a):pass
        def stop(self):pass
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
    def init(runtime,folder,*a,**kw):
        folder.mkdir();episode.save(folder/'result.json',dict(physical_actions=0,robot_execution_s=0.))
    contexts=[]
    def score(root,folder,before,*a,**kw):
        contexts.append(kw.get('previous_context'))
        episode.save(folder/'attempt.json',{})
        episode.save(folder/'model_metrics.json',dict(wall_latency_s=1))
        if len(contexts)==1:
            assert kw==dict(automatic=True)
            return dict(candidates=rows(),selection=selection()),dict(wall_latency_s=1)
        assert kw['automatic'] is True and kw['previous_context']['source_round']=='round_000'
        return dict(stop_plan=dict(decision='DONE',grasp_px=None,target_px=None,
                                   motion_type=None,grasp_angle_deg=None,short_reason='展开完成。')),dict(wall_latency_s=1)
    for name,val in [('EpisodeCollectionRuntime',Runtime),('TrajectoryRecorder',Recorder),
                     ('PreActionGate',Gate),('initialize',init),
                     ('prepare_action',lambda *a:([],{}))]:monkeypatch.setattr(episode,name,val)
    monkeypatch.setattr(episode.planner,'auth_headers',lambda:{})
    monkeypatch.setattr(scoring,'score_round',score)
    goal=tmp_path/'goal.png';Image.new('RGB',(200,100),'purple').save(goal)
    out=tmp_path/'episode'
    episode.run(SimpleNamespace(output=out,goal_image=goal,config=tmp_path/'unused',max_rounds=2,
                                timeout_s=1,automatic_top5=True,stateful_context=True))
    assert len(contexts)==2 and contexts[0] is None
    assert contexts[1]['plan']['grasp_px']==rows()[0]['grasp_px']
    assert contexts[1]['before_image']==out/'round_000/before/rgb.png'
    assert episode.read(out/'round_001/previous_context_continuity.json')['passed'] is True
    assert episode.read(out/'episode_summary.json')['termination_reason']=='DONE'
