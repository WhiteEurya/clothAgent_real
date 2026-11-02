import copy,json
import pytest
from PIL import Image
from cloth_agent.experiments.semantic_corner_oracle.core import PARTS,save,build_request,validate_annotation,endpoint_match
from cloth_agent.experiments.semantic_corner_oracle.review import Store
from cloth_agent.cloth_harness.runtime.gpt_policy import PROMPT
from cloth_agent.restricted_context_one_shot import SCHEMA
from cloth_agent.experiments import one_shot_goal_planner_probe_v1 as io


def fixture(tmp_path):
    (tmp_path/'images').mkdir();(tmp_path/'frozen').mkdir();Image.new('RGB',(100,100)).save(tmp_path/'images/S01.png');Image.new('RGB',(100,100)).save(tmp_path/'goal.png')
    save(tmp_path/'frozen/policy.json',{'prompt':PROMPT,'schema':SCHEMA,'model':io.MODEL,'effort':io.EFFORT,'constraint':'FOUR_ENDPOINT_ONLY'})
    case={'id':'S01','size':[100,100],'rgb_sha256':'abc'};save(tmp_path/'cases.json',[case]);save(tmp_path/'annotations.json',{})
    annotation={'confirmed_by':'tester','points':{k:{'visibility':'HIDDEN','px':None} for k in PARTS}};annotation['points']['LS']={'visibility':'VISIBLE','px':[20,25]}
    return case,annotation


def test_hidden_coordinates_never_imputed(tmp_path):
    case,a=fixture(tmp_path);validate_annotation(a,case['size']);a['points']['RH']['px']=[90,90]
    with pytest.raises(ValueError):validate_annotation(a,case['size'])


def test_all_conditions_same_images_model_schema_only_a2_gets_coordinates(tmp_path):
    c,a=fixture(tmp_path);bodies=[build_request(tmp_path,c,k,a) for k in ['A0','A1','A2']]
    def images(body):return [x for x in body['input'][0]['content'] if x['type']=='input_image']
    assert images(bodies[0])==images(bodies[1])==images(bodies[2])
    assert all(b['model']==io.MODEL and b['text']['format']['schema']==SCHEMA for b in bodies)
    assert bodies[0]['instructions']==PROMPT
    assert bodies[1]['input']==bodies[0]['input']
    extra=bodies[2]['input'][0]['content'][-1]['text'];assert '20, 25' in extra and '左袖口边缘中点' in extra
    assert 'RH' not in extra


def test_outside_oracle_radius_does_not_mean_false_fold_corner(tmp_path):
    c,a=fixture(tmp_path);r=endpoint_match({'decision':'ACTION','grasp_px':[80,80]},a)
    assert r['status']=='OUTSIDE_RADIUS_NOT_A_FALSE_CORNER_LABEL' and r['part'] is None
    assert endpoint_match({'decision':'DONE'},a)['status']=='DONE'


def test_confirmation_freeze_and_blind_condition_isolation(tmp_path):
    c,a=fixture(tmp_path);s=Store(tmp_path);body={'id':'S01','points':a['points'],'reviewer':'tester','confirm':True}
    assert s.save(body)['confirmed']['S01'];body['confirm']=False;assert s.save(body)['confirmed']['S01']
    body['points']=copy.deepcopy(body['points']);body['points']['LS']['px']=[21,25];assert not s.save(body)['confirmed']
    save(tmp_path/'frozen/oracle_annotations.json',{})
    with pytest.raises(ValueError):s.save(body)
    save(tmp_path/'blind_manifest_private.json',[{'review_id':'R001','case':'S01','condition':'A2','pick_px':[20,25]}])
    b=Store(tmp_path,'blind').data();assert 'A2' not in json.dumps(b) and 'condition' not in json.dumps(b)
