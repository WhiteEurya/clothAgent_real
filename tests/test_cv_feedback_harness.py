import copy
from dataclasses import asdict
import cv2
import numpy as np
import pytest
from cloth_agent.cloth_harness.runtime.interfaces import Observation,Action
from cloth_agent.cloth_harness.cv_harness.motion_observer import fit_rigid_matches,register
from cloth_agent.cloth_harness.cv_harness.outcome_observer import observe,execution_status
from cloth_agent.cloth_harness.cv_harness.feedback_generator import compress
from cloth_agent.cloth_harness.evaluation.evaluator import classification_metrics,validate_label,LABELS
from cloth_agent.cloth_harness.runtime.gpt_policy import payload


def obs(shift=0):
    rgb=np.zeros((240,400,3),dtype='uint8');rgb[50:190,130+shift:270+shift]=[180,130,200]
    return Observation(rgb,None,1.,'camera',provenance={'joints_deg':{'camera_arm':[0,0,0]}})


def test_completed_motor_does_not_prove_grasp():
    r=execution_status({'status':'EXECUTED','completed_commands':9,'acquisition':'SUCCESS'})
    assert r['completed'] and r['grasp_status']=='UNKNOWN'
    assert execution_status({'status':'FAILED','completed_commands':3})['status']=='PARTIAL_EXECUTION'


def test_area_uncalibrated_and_no_semantic_progress():
    a,b=obs(),obs(15);o,_,_=observe(a,Action((150,80),(165,80),0),b,{'status':'EXECUTED'})
    assert o.coverage['status']=='UNCALIBRATED'
    assert o.coverage['delta_ratio']==0
    assert o.local_change['status']=='UNKNOWN'
    assert o.motion['status']=='UNKNOWN' # untextured rectangle must not be confidently aligned
    assert 'cloth grasp=UNKNOWN' in compress(o,Action((150,80),(165,80),0))


def test_changed_camera_abstains():
    a,b=obs(),obs();b.provenance['joints_deg']['camera_arm'][0]=2
    o,_,_=observe(a,Action(None,None,None),b,{'status':'EXECUTED'})
    assert o.coverage['delta_ratio'] is None and o.motion['status']=='UNKNOWN'


def test_clipping_abstains():
    a,b=obs(),obs();b.rgb[:100,130:270]=[180,130,200]
    o,_,_=observe(a,Action(None,None,None),b,{})
    assert o.quality['segmentation_after']['clipped'] and o.coverage['status']=='UNKNOWN'


def test_spread_matches_known_rigid_rotation():
    x,y=np.meshgrid(np.linspace(20,180,5),np.linspace(20,180,5));a=np.c_[x.ravel(),y.ravel()]
    angle=.13;r=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]);b=a@r.T+[20,-10]
    result=fit_rigid_matches(a,b,np.ones(len(a)),np.ones((200,200),bool))
    assert result['status']=='OK';np.testing.assert_allclose(result['matrix'],np.column_stack((r,[20,-10])),atol=1e-6)


def test_localized_logo_not_whole_garment_evidence():
    rng=np.random.default_rng(9);a=rng.uniform(50,60,(30,2));b=a+[20,5]
    assert fit_rigid_matches(a,b,np.ones(30),np.ones((200,200),bool))['status']=='UNKNOWN'


def test_rigid_residual_rejects_shape_deformation():
    rng=np.random.default_rng(4);a=rng.uniform(10,190,(30,2));b=a.copy();b[:,0]*=1.3
    assert fit_rigid_matches(a,b,np.ones(30),np.ones((200,200),bool))['status']=='UNKNOWN'


def test_accuracy_and_coverage_are_separate():
    r=classification_metrics({'a':'X','b':'Y','c':'UNKNOWN'},{'a':'X','b':'UNKNOWN','c':'Z'})
    assert r['coverage']==.5 and r['accuracy_when_confident']==1
    r=classification_metrics({'a':'X'},{'a':'UNKNOWN'})
    assert r['accuracy_when_confident'] is None and r['coverage']==0
    r=classification_metrics({'a':'X'},{'a':'Y'})
    assert r['false_confident_rate']==1


def test_oracle_rejects_action_advice_and_missing_evidence():
    r={'confirmed_by':'human','evidence':['after.png'],'labels':{k:'UNKNOWN' for k in LABELS}}
    validate_label(r)
    r['labels']['motion_type']='grab right sleeve'
    with pytest.raises(ValueError):validate_label(r)
    r['labels']['motion_type']='UNKNOWN';r['evidence']=[]
    with pytest.raises(ValueError):validate_label(r)


def test_payload_same_policy_and_goal_no_image_history_leak(tmp_path):
    from PIL import Image
    from pathlib import Path
    for side in ['before','after']:
        (tmp_path/side).mkdir();Image.fromarray(obs().rgb).save(tmp_path/side/'rgb.png')
    goal=tmp_path/'goal.png';Image.fromarray(obs().rgb).save(goal)
    pair={'before':str(tmp_path/'before'),'after':str(tmp_path/'after'),'goal':str(goal),'plan':{'grasp_px':[150,80],'target_px':[180,80],'grasp_angle_deg':0},'execution':{'status':'EXECUTED'}}
    a=payload(pair,'A');b=payload(pair,'B');c=payload(pair,'C','test feedback')
    assert a['instructions']==b['instructions']==c['instructions']
    def images(p):return [x for x in p['input'][0]['content'] if x['type']=='input_image']
    assert len(images(a))==len(images(c))==2 and len(images(b))==3
    assert images(a)==images(c)==images(b)[:2]
    assert a['text']['format']['schema']==c['text']['format']['schema']
    with pytest.raises(ValueError):payload(pair,'D')


def test_unlabelled_d_creates_protocol_without_model_call(tmp_path):
    from argparse import Namespace
    from cloth_agent.cloth_harness.experiment import save,run,read
    save(tmp_path/'pairs.json',[{'id':'case','execution':{'status':'EXECUTED'}}])
    save(tmp_path/'cv_results.json',{'case':{'feedback':'none'}})
    save(tmp_path/'oracle_labels.json',{})
    run(Namespace(output=tmp_path,labels=None,limit=6,conditions=['D']))
    assert read(tmp_path/'abcd/summary.json')['completed']==0
    assert len(read(tmp_path/'abcd/summary.json')['skipped'])==1


def test_geometry_depth_discontinuity_is_diagnostic_not_rejection(tmp_path):
    from PIL import Image
    from types import SimpleNamespace
    from cloth_agent.cloth_harness.cv_harness.geometry_check import check
    image=tmp_path/'rgb.png';Image.fromarray(obs().rgb).save(image)
    xyz=np.zeros((240,400,3));xyz[:,:,2]=20;xyz[79:82,149:152,2]=60
    ob=SimpleNamespace(image=image,xyz=xyz)
    r=check(ob,{'grasp_px':[150,80]})
    assert r['status']=='ACCEPTED' and r['checks']['depth_z_range_mm']==40
    assert r['robot_feasibility']=='PENDING_EXISTING_HOST'
    xyz[80,150]=np.nan
    assert check(ob,{'grasp_px':[150,80]})['status']=='UNVERIFIED'
    assert check(ob,{'grasp_px':[1000,80]})['status']=='REJECTED'
    assert check(ob,{'grasp_px':[100,80]})['status']=='UNVERIFIED'


def test_live_policy_only_one_call_and_no_region_or_previous_image(tmp_path):
    from PIL import Image
    from types import SimpleNamespace
    from cloth_agent.cloth_harness.runtime import live
    (tmp_path/'goal').mkdir();Image.fromarray(obs().rgb).save(tmp_path/'goal/target.png')
    previous=tmp_path/'round_000';previous.mkdir();(previous/'cv_feedback.txt').write_text('Previous observed area +1%; grasp UNKNOWN.')
    folder=tmp_path/'round_001';folder.mkdir();image=folder/'rgb.png';Image.fromarray(obs().rgb).save(image)
    calls=[]
    def direct(f,body,h,t):
        calls.append(body)
        return {'decision':'DONE','grasp_px':None,'target_px':None,'grasp_angle_deg':None,'motion_type':None,'is_smoothing':False,'action_purpose':None,'expected_stop_state':None,'short_reason':'done'}, {'wall_latency_s':1}
    live.plan_round(tmp_path,folder,SimpleNamespace(image=image),{},10,direct,{'source_round':'round_000'})
    assert len(calls)==1
    content=calls[0]['input'][0]['content']
    assert sum(c['type']=='input_image' for c in content)==2
    assert any('Previous observed area' in c.get('text','') for c in content)
    # The episode runner reads this even when previous-effect assessment is off:
    # max_actions still enables progress/termination bookkeeping.
    import json
    evidence=json.loads((folder/'decision_evidence.json').read_text())
    assert evidence['decision']=='DONE'
    assert evidence.get('previous_effect') is None


def test_live_host_render_available_on_rejection(tmp_path):
    from cloth_agent.cloth_harness.runtime import live
    (tmp_path/'cv_geometry_check.json').write_text('{"status":"UNVERIFIED"}')
    live.render_round(tmp_path)
    assert 'UNVERIFIED' in (tmp_path/'decision.html').read_text()
