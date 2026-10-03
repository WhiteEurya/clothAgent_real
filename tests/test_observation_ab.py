import copy
import json
import shutil
from types import SimpleNamespace

import pytest

from test_information_flow import setup_case, plan, request
from cloth_agent.harness.common import read_json, write_json
from cloth_agent.harness.observation_ab import run_arm, render
from cloth_agent.image_tools_mcp import ImageTools


class Model:
    configuration={'model':'same','tools':[], 'image_delivery':'direct'}
    def __init__(self, result):self.result=result;self.calls=[];self.requests=[]
    def invoke(self,**kw):
        self.calls.append({'stage':kw['stage']});self.requests.append(kw)
        return copy.deepcopy(self.result)


class Backend:
    def __init__(self):self.requests=[]
    def invoke(self,**kw):
        self.requests.append(kw)
        trace=kw['debug_dir'];image_dir=trace/'images';image_dir.mkdir(parents=True)
        for i,p in enumerate(kw['image_paths']):shutil.copyfile(p,image_dir/f'image_{i}.png')
        tools=ImageTools(image_dir,len(kw['image_paths']),edit_limit=6)
        v=tools.call('crop_image',{'image_id':'image_0','box':[4,12,28,48]})
        write_json(trace/'image_debug.json',{'views':list(tools.views.values())})
        stdout=json.dumps({'type':'result','structured_output':{'selected_image_ids':[v['image_id']]}})
        return SimpleNamespace(stdout=stdout,stderr='',timings={})


def test_only_observation_changes_and_common_reasoning_is_identical(tmp_path,setup_case):
    case,images,artifact=setup_case
    a_reason,b_reason=Model(plan()),Model(plan())
    free=Backend()
    req=request('local_boundary',[.1,.2,.7,.8]);req['expected_information_gain']='SECRET_NOT_FOR_REASONING'
    bound=Model({'observation_requests':[req]})
    a=run_arm(case,images,artifact,tmp_path/'free',mode='free',observer_backend=free,
              observer_model=Model({}),reasoning_model=a_reason,model_name='same')
    b=run_arm(case,images,artifact,tmp_path/'skill',mode='skill',observer_backend=Backend(),
              observer_model=bound,reasoning_model=b_reason,model_name='same')
    assert a['status']==b['status']=='COMPLETED'
    assert a['reasoning_contract_hash']==b['reasoning_contract_hash']
    ar,br=a_reason.requests[0],b_reason.requests[0]
    # Fake observers happen to produce the same crop, so even full prompt bytes match.
    for key in ('prompt','schema','stage','timeout_s'):assert ar[key]==br[key]
    assert ar['timeout_s']==360
    assert len(ar['images'])==len(br['images'])==4
    assert 'SECRET_NOT_FOR_REASONING' not in br['prompt']
    assert 'skill_id' not in br['prompt'] and 'expected_information_gain' not in br['prompt']
    assert len(bound.calls)==1 and len(b_reason.calls)==1
    assert free.requests[0]['image_edit_limit']==6
    report={'arms':{'free':a,'skill':b}};render(tmp_path,report)
    assert (tmp_path/'index.html').exists()


def test_no_observation_edits_still_runs_same_reasoning(tmp_path,setup_case):
    case,images,artifact=setup_case;reason=Model(plan())
    r=run_arm(case,images,artifact,tmp_path/'skill',mode='skill',observer_backend=Backend(),
              observer_model=Model({'observation_requests':[]}),reasoning_model=reason,model_name='same')
    assert r['status']=='COMPLETED' and r['host_image_ops']==0
    assert len(reason.requests[0]['images'])==3


def test_invalid_observation_does_not_reach_reasoning(tmp_path,setup_case):
    case,images,artifact=setup_case;reason=Model(plan())
    req=request();req['source_image_id']='image_2'
    r=run_arm(case,images,artifact,tmp_path/'skill',mode='skill',observer_backend=Backend(),
              observer_model=Model({'observation_requests':[req]}),reasoning_model=reason,model_name='same')
    assert r['status']=='FAILED' and not reason.calls
    assert not (tmp_path/'skill/result.json').exists()


def test_reasoning_schema_failure_preserves_raw_response_without_valid_result(tmp_path,setup_case):
    case,images,artifact=setup_case
    r=run_arm(case,images,artifact,tmp_path/'skill',mode='skill',observer_backend=Backend(),
              observer_model=Model({'observation_requests':[]}),reasoning_model=Model({'selected_marker_id':'R001'}),model_name='same')
    assert r['status']=='FAILED'
    assert (tmp_path/'skill/raw_reasoning_result.json').exists()
    assert not (tmp_path/'skill/result.json').exists()
