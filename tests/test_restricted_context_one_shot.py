import copy
from types import SimpleNamespace
import pytest
from PIL import Image
from cloth_agent import restricted_context_one_shot as r


def output():
    return dict(decision='ACTION',grasp_px=[20,30],target_px=[50,30],motion_type='lift_pull_place',
                grasp_angle_deg=90,short_reason='拉开衣袖',is_smoothing=False)


def test_compact_contract_and_simple_done():
    assert set(r.SCHEMA['properties'])=={'decision','grasp_px','target_px','motion_type','grasp_angle_deg','short_reason','is_smoothing'}
    assert r.validate(output(),(100,100),True)['decision']=='ACTION'
    d=output();d['decision']='DONE'
    for k in ('grasp_px','target_px','motion_type','grasp_angle_deg'):d[k]=None
    assert r.validate(d,(100,100),False)['decision']=='DONE'
    d['primary_structure']={}
    with pytest.raises(Exception):r.validate(d,(100,100),False)


@pytest.mark.parametrize('contextual',[False,True])
def test_compact_request_retains_images_and_context(tmp_path,contextual):
    (tmp_path/'goal').mkdir();folder=tmp_path/'round';(folder/'before').mkdir(parents=True)
    Image.new('RGB',(100,100)).save(tmp_path/'goal/target.png')
    Image.new('RGB',(100,100)).save(folder/'before/rgb.png')
    before=SimpleNamespace(image=folder/'before/rgb.png');calls=[]
    context=dict(source_round='previous',before_image=before.image,after_image=before.image,
                 plan=output(),completed_commands=7) if contextual else None
    def direct(folder,body,headers,timeout):
        calls.append(body);return output(),dict(wall_latency_s=1)
    r.plan_round(tmp_path,folder,before,{},10,direct,context)
    assert len(calls)==1
    assert 'Geometry Report' in str(calls[0])
    assert (folder/'geometry_report.json').exists()
    assert r.io.read(folder/'request_manifest.json')['geometry_report']=='geometry_report.json'
    assert '请先判断上一动作的实际局部效果' not in str(calls[0])
    assert len([c for c in calls[0]['input'][0]['content'] if c['type']=='input_image'])==(3 if contextual else 2)
    assert (folder/'decision.html').exists()
    assert 'previous_effect' not in calls[0]['instructions']
    assert '只能抓当前图中可见的衣袖或者下摆' in calls[0]['instructions']


def test_target_rejection_is_supplied_as_unexecuted_feedback(tmp_path):
    (tmp_path/'goal').mkdir();folder=tmp_path/'round';folder.mkdir()
    Image.new('RGB',(100,100)).save(tmp_path/'goal/target.png')
    Image.new('RGB',(100,100)).save(folder/'rgb.png')
    feedback=dict(error='target outside bare-table ROI',rejected_plan={'target_px':[99,50]},physical_execution=False)
    calls=[]
    def direct(folder,body,headers,timeout):
        calls.append(body);return output(),dict(wall_latency_s=1)
    r.plan_round(tmp_path,folder,SimpleNamespace(image=folder/'rgb.png'),{},10,direct,rejection_feedback=feedback)
    text=calls[0]['input'][-1]['content'][0]['text']
    assert feedback['error'] in text and '没有执行' in text and 'previous_effect' not in text
    assert r.io.read(folder/'rejection_feedback.json')==feedback


def test_replan_uses_distinct_http_attempt_directory(tmp_path):
    (tmp_path/'goal').mkdir()
    Image.new('RGB',(100,100)).save(tmp_path/'goal/target.png')
    folder=tmp_path/'round';folder.mkdir()
    (folder/'attempt.json').write_text('{"original":true}')
    observation=folder/'target_replan'/'observation';observation.mkdir(parents=True)
    Image.new('RGB',(100,100)).save(observation/'rgb.png')
    before=SimpleNamespace(directory=observation,image=observation/'rgb.png')
    calls=[]
    def direct(attempt,body,headers,timeout):
        with (attempt/'attempt.json').open('x') as stream:stream.write('{"retry":true}')
        calls.append(attempt)
        return output(),dict(wall_latency_s=1)
    plan,_=r.replan_rejected(tmp_path,folder,before,{},10,direct,None,{'error':'outside ROI'})
    assert calls==[folder/'target_replan'/'retry_01']
    assert plan['decision']=='ACTION'
    assert (calls[0]/'before/rgb.png').exists()
    assert r.io.read(folder/'attempt.json')=={'retry':True}
    with pytest.raises(FileExistsError):
        r.replan_rejected(tmp_path,folder,before,{},10,direct,None,{})
    assert len(calls)==1


def test_failed_replan_preserves_original_plan(tmp_path):
    folder=tmp_path/'round';folder.mkdir()
    r.io.save(folder/'plan.json',{'original':True})
    observation=tmp_path/'observation';observation.mkdir()
    before=SimpleNamespace(directory=observation)
    def fail(*args,**kwargs):raise RuntimeError('model failed')
    from unittest.mock import patch
    with patch.object(r,'plan_round',side_effect=fail):
        with pytest.raises(RuntimeError,match='model failed'):
            r.replan_rejected(tmp_path,folder,before,{},10,None,None,{})
    assert r.io.read(folder/'plan.json')=={'original':True}
