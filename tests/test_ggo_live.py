import copy
from types import SimpleNamespace
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from cloth_agent.ggo import live


def annotation():
    return dict(reference={k:[20,20] for k in live.LANDMARKS},current={'N':[20,20],'AL':None,'AR':[30,30]},reference_rgb_sha256='r',current_rgb_sha256='c')


def test_annotation_bound_to_frozen_rgb_and_hidden_is_not_imputed():
    a=annotation();assert live.validate_annotations(a,'r','c',(100,100),(100,100))['current']['AL'] is None
    with pytest.raises(ValueError):live.validate_annotations(a,'r','new',(100,100),(100,100))
    a['current']['N']=[100,20]
    with pytest.raises(ValueError):live.validate_annotations(a,'r','c',(100,100),(100,100))


@pytest.mark.parametrize('contextual',[False,True])
@pytest.mark.parametrize('automatic',[False,True])
def test_live_geometry_uses_one_action_call_no_structure_model(tmp_path,monkeypatch,automatic,contextual):
    root=tmp_path;folder=root/'round_000';(folder/'before').mkdir(parents=True);(root/'goal').mkdir()
    for p in [root/'goal/target.png',folder/'before/rgb.png']:Image.new('RGB',(100,100),'purple').save(p)
    obs=SimpleNamespace(directory=folder/'before',image=folder/'before/rgb.png',xyz=np.zeros((100,100,3)),meta={'calibration_id':'test','sha256':{}})
    live.io.save(root/'ggo_live_config.json',dict(reference=str(folder/'before'),port=8781,automatic_anchors=automatic));live.io.save(root/'ggo_reference_seed.json',dict(annotation(),reference_rgb_sha256=live.io.sha(obs.image)))
    monkeypatch.setattr(live.Observation,'load',lambda _:obs)
    def review(*args):
        assert not automatic, 'automatic mode must not wait for human review'
        return annotation()
    monkeypatch.setattr(live,'review',review)
    monkeypatch.setattr(live,'observe',lambda *args:({'registration':{'status':'UNKNOWN'},'deformation':{'status':'UNKNOWN'},'annotation_provenance':'test','corners':{'right_hem':{'status':'UNVERIFIED','point':None}}},{}))
    context=dict(plan=dict(grasp_px=[10,10],target_px=[20,20]),before_image=obs.image) if contextual else None
    calls=[]
    def direct(path,body,*args):
        calls.append(body)
        if body['text']['format']['name']=='ggo_upper_anchors':
            return {'N':[20,20],'AL':None,'AR':[30,30]},dict(wall_latency_s=2.)
        assert 'Geometry Observation:' in str(body) and 'UNKNOWN' in str(body)
        assert sum(c['type']=='input_image' for c in body['input'][0]['content'])==(3 if contextual else 2)
        if contextual:assert '上一轮实际执行状态：EXECUTED' in str(body)
        return dict(decision='ABSTAIN',grasp_px=None,target_px=None,motion_type=None,grasp_angle_deg=None,is_smoothing=False,short_reason='No reliable action',action_purpose=None,expected_stop_state=None),dict(wall_latency_s=1.)
    plan,metrics=live.plan_round(root,folder,obs,{},10,direct,previous_context=context)
    assert len(calls)==(2 if automatic else 1) and plan['decision']=='ABSTAIN'
    assert live.io.read(folder/'request_manifest.json')['geometry_model_calls']==int(automatic)
    assert metrics['wall_latency_s']==(3. if automatic else 1.)
    if automatic:
        assert metrics['annotation_wait_s']==0
        assert live.io.read(folder/'ggo_annotation/annotations.json')['current']['AL'] is None
    assert (folder/'ggo/geometry_assessment.txt').exists()
