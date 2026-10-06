import copy
import json
import pytest
from PIL import Image
from cloth_agent.harness.observation_code_trial import run
from cloth_agent.harness.executors.restricted import RestrictedProgram
from cloth_agent.harness.policy import PolicyError
from cloth_agent.harness.candidate_patch import _fixtures, _request
from cloth_agent.harness.skills import builtin_registry


class Model:
    def __init__(self,*answers):self.answers=list(answers);self.calls=[];self.requests=[]
    def invoke(self,**kw):
        self.requests.append(kw);self.calls.append({'response_received':True})
        return copy.deepcopy(self.answers.pop(0))


def setup(tmp_path):
    obs,images=_fixtures(tmp_path/'input',[100,80])
    evidence={'schema_version':1,'observation_id':'synthetic','fold_goal':'Inspect the visible boundary',
              'images':[{**i,'path':str(p),'status':'AVAILABLE'} for i,p in zip(obs['images'],images)],
              'candidate_registry':{'observation_id':'synthetic','binding':'RAW_RGB_HASH_VERIFIED',
                'raw_size':[100,80],'to_raw':[1,0,0,0,1,0],
                'candidates':[{'camera':'A','candidate_id':'R001','pixel_xy':[20,30],'raw_pixel_xy':[20,30]}]}}
    skill=builtin_registry().get('local_boundary')
    tests=[]
    for roi in ([0,0,1,1],[.2,.2,.8,.8]):
        req=_request('local_boundary',roi=roi);req['enlarge']=False
        tests.append({'request':req,'source_size':[100,80],
                      'expected_recipe':{'roles':['clean'],'operations':[{'op':'crop','roi':roi}],'reuse_existing':False}})
    return evidence,{'specification':skill.specification,'source':skill.source,'tests':tests}


def judgment(source,*,last=None,done=False):
    req=_request('local_boundary',roi=[.2,.2,.8,.8]);req.update(gap_id='boundary',source_image_id=source,enlarge=False)
    return {'decision':'DONE' if done else 'EXECUTE','reason':'inspect boundary',
        'information':[{'id':'boundary','need':'locate edge','status':'KNOWN' if done else 'UNKNOWN',
                        'finding':'edge readable' if done else '', 'missing_information':'' if done else 'detail',
                        'source_image_ids':[source]}],
        'last_result':last,'request':None if done else req}


def assessment(outcome,ref):return {'outcome':outcome,'reason':'visible result','source_image_ids':[ref]}


def test_before_after_code_loop_preserves_module_and_intermediate_lineage(tmp_path):
    evidence,module=setup(tmp_path)
    feedback={'summary':'synthetic','information_obtained':'edge','remaining_unknowns':'physical',
              'code_assessment':'worked','binding_assessment':'dynamic','next_revision':'none',
              'evidence':[{'trace_index':3,'lesson':'second crop was executed'}]}
    text=Model(module,feedback)
    vision=Model(judgment('image_0'),judgment('image_2',last=assessment('INSUFFICIENT','image_2')),
                 judgment('image_3',last=assessment('SUFFICIENT','image_3'),done=True))
    result=run(evidence,{'lessons':[]},'draft only',tmp_path/'trial',vision,text)
    assert result['status']=='COMPLETED',result.get('error')
    assert result['outcome']=='DONE' and result['code_executions']==2
    assert [x['stage'] for x in result['trace']]==['observe','code','observe','code','observe']
    assert len(vision.requests[0]['images'])==2
    assert len(vision.requests[1]['images'])==3
    assert len(vision.requests[2]['images'])==4
    assert all(not r['images'] for r in text.requests)
    hashes=[x['module_hash'] for x in result['trace'] if x['stage']=='code']
    assert hashes==[result['module_hash']]*2
    assert result['trace'][3]['execution']['request']['source_image_id']=='image_2'
    assert result['trace'][3]['execution']['lineage'][-1]['parent_image_id']=='image_2'
    with Image.open(vision.requests[2]['images'][-1]) as im:assert im.size==(36,30)
    assert result['robot_actions']==0 and result['automatic_activation'] is False


def test_illegal_generated_code_never_reaches_image_execution(tmp_path):
    evidence,module=setup(tmp_path);module['source']='import os\ndef prepare(request,source,available):\n return {}'
    vision=Model();result=run(evidence,{'lessons':[]},'',tmp_path/'trial',vision,Model(module))
    assert result['status']=='ERROR' and result['host_image_ops']==0
    assert not vision.requests


def test_execution_success_does_not_override_unknown_result(tmp_path):
    evidence,module=setup(tmp_path)
    follow=judgment('image_2',last=assessment('UNKNOWN','image_2'),done=True)
    vision=Model(judgment('image_0'),follow)
    result=run(evidence,{'lessons':[]},'',tmp_path/'trial',vision,Model(module))
    assert result['status']=='ERROR'
    assert 'DONE contradicts' in result['error']


def test_reused_module_is_repaired_once_before_vision(tmp_path):
    evidence,module=setup(tmp_path)
    broken=copy.deepcopy(module)
    broken['source']='def prepare(request, source, available):\n return len(available["cached_views"])'
    path=tmp_path/'previous.json';path.write_text(json.dumps(broken))
    first=judgment('image_0',done=True)
    feedback={'summary':'already readable','information_obtained':'edge',
              'remaining_unknowns':'none','code_assessment':'not exercised',
              'binding_assessment':'unneeded','next_revision':'none','evidence':[]}
    text=Model(module,feedback)
    result=run(evidence,{'lessons':[]},'',tmp_path/'trial',Model(first),text,module_from=path)
    assert result['status']=='COMPLETED',result.get('error')
    assert result['outcome']=='NOT_EXERCISED'
    assert [r['stage'] for r in text.requests]==['compile_repair','feedback']


def test_recipe_list_concatenation_is_bounded():
    source='def prepare(request, source, available):\n ops = []\n ops = ops + [{"op":"rotate","degrees":90}]\n return ops'
    assert RestrictedProgram(source).run({}, {}, {})==[{'op':'rotate','degrees':90}]
    growing='def prepare(request, source, available):\n items = [0]\n'+' items = items + items\n'*8+' return items'
    with pytest.raises(PolicyError,match='collection budget'):
        RestrictedProgram(growing).run({}, {}, {})
    with pytest.raises(PolicyError,match='numeric'):
        RestrictedProgram('def prepare(request, source, available):\n return [0] * 1000000').run({}, {}, {})


def test_five_execution_trial_observes_every_result(tmp_path):
    evidence,module=setup(tmp_path)
    answers=[judgment('image_0')]
    for image_id in range(2,7):
        answers.append(judgment(f'image_{image_id}',
            last=assessment('SUFFICIENT' if image_id==6 else 'INSUFFICIENT',f'image_{image_id}'),
            done=image_id==6))
    feedback={'summary':'ok','information_obtained':'edge','remaining_unknowns':'none',
              'code_assessment':'ok','binding_assessment':'ok','next_revision':'none','evidence':[]}
    result=run(evidence,{'lessons':[]},'',tmp_path/'long',Model(*answers),Model(module,feedback),
               max_executions=5,max_ops=24)
    assert result['status']=='COMPLETED',result.get('error')
    assert result['code_executions']==5 and len(result['trace'])==11


def test_initial_observation_reuse_checks_context_and_normalizes_empty_unknown(tmp_path):
    evidence,module=setup(tmp_path)
    first=judgment('image_0',last=assessment('UNKNOWN','image_0'))
    first['last_result']['source_image_ids']=[]
    previous=tmp_path/'previous'
    result=run(evidence,{'lessons':[]},'',previous,Model(first),Model(module))
    assert result['host_image_ops']==1
    assert result['trace'][0]['judgment']['last_result'] is None
    feedback={'summary':'ok','information_obtained':'edge','remaining_unknowns':'none',
              'code_assessment':'ok','binding_assessment':'ok','next_revision':'none','evidence':[]}
    vision=Model(judgment('image_2',last=assessment('SUFFICIENT','image_2'),done=True))
    result=run(evidence,{'lessons':[]},'',tmp_path/'resumed',vision,Model(feedback),
               module_from=previous/'module.json',initial_observation_from=previous)
    assert result['status']=='COMPLETED',result.get('error')
    assert [r['stage'] for r in vision.requests]==['observe_01']
    changed=copy.deepcopy(evidence);changed['fold_goal']='Different information task'
    vision=Model()
    result=run(changed,{'lessons':[]},'',tmp_path/'mismatch',vision,Model(),
               module_from=previous/'module.json',initial_observation_from=previous)
    assert 'context differs' in result['error'] and not vision.requests


def test_stop_suggestion_is_not_executed_and_all_observations_can_be_reused(tmp_path):
    evidence,module=setup(tmp_path)
    stop=judgment('image_2',last=assessment('INSUFFICIENT','image_2'))
    stop['decision']='UNKNOWN'
    previous=tmp_path/'previous'
    result=run(evidence,{'lessons':[]},'',previous,Model(judgment('image_0'),stop),Model(module))
    assert result['host_image_ops']==1
    assert result['trace'][-1]['unexecuted_suggestion']==stop['request']
    assert result['trace'][-1]['judgment']['request'] is None
    feedback={'summary':'unknown','information_obtained':'partial','remaining_unknowns':'edge',
              'code_assessment':'ok','binding_assessment':'partial','next_revision':'none','evidence':[]}
    vision=Model()
    result=run(evidence,{'lessons':[]},'',tmp_path/'resumed',vision,Model(feedback),
               module_from=previous/'module.json',observations_from=previous)
    assert result['status']=='COMPLETED',result.get('error')
    assert result['outcome']=='UNKNOWN' and result['code_executions']==1
    assert not vision.requests and len(result['observations_reused'])==2
    assert result['metrics']['host_image_ops']==1
