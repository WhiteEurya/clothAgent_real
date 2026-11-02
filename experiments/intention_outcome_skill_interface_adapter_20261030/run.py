"""Repair only the VISION_MODEL response adapter, preserving generated policy."""
from pathlib import Path
import sys,json,time,hashlib,shutil
import numpy as np
import jsonschema
ROOT=Path(__file__).resolve().parents[2]
CODE=ROOT/'experiments/intention_outcome_skill_discovery_v0'
sys.path.insert(0,str(CODE))
from transport import call,read,save,sha
from policy import checked_compile,invoke,overlay
from contracts import obj,en,S,B,POINT,arr
OUT=ROOT/'reports/intention_outcome_skill_discovery_v0_20261030'
FIX=OUT/'interface_repair'

def nullable(schema):return {**schema,'type':[schema['type'],'null']}
def vision_schema(goal):
    strings=['target_identity_basis','target_occlusion_evidence','observable_geometry','grasp_region','direction_reason','expected_target_change','falsifier','uncertainty','abstain_reason']
    flags=['identity_confirmed','self_fold_evidence','body_to_root_connection','direction_opens_fold','projected_motion_clear']
    fields={'goal':en(goal),'decision':en('ACTION','ABSTAIN'),'target_state':en('OCCLUDED','SUSPECTED_OCCLUDED','VISIBLE','OUT_OF_VIEW','UNKNOWN')}
    fields.update({k:S for k in strings});fields.update({k:B for k in flags})
    fields.update({k:nullable(POINT) for k in ['grasp_px','sleeve_root_px','folded_cuff_px']})
    fields.update(fold_width_endpoints_px=nullable({'type':'array','items':POINT,'minItems':2,'maxItems':2}),
                  pull_direction_xy=nullable({'type':'array','items':{'type':'number'},'minItems':2,'maxItems':2}),
                  grasp_angle_deg={'type':['integer','null'],'minimum':0,'maximum':180})
    fields.update({k:nullable({'type':'array','items':POINT,'minItems':3,'maxItems':12}) for k in ['grasp_patch_px','motion_region_px']})
    return obj(fields)

def smoke_contract(code,prompt,goal,schema):
    payload={k:('synthetic interface fixture' if v['type']=='string' else False) for k,v in schema['properties'].items()}
    payload.update(goal=goal,decision='ACTION',target_state='SUSPECTED_OCCLUDED',abstain_reason='',
                   grasp_px=[600,400],sleeve_root_px=[600,300],folded_cuff_px=[500,300],
                   fold_width_endpoints_px=[[500,300],[600,300]],pull_direction_xy=[1,0],
                   grasp_patch_px=[[550,350],[650,350],[650,450],[550,450]],
                   motion_region_px=[[500,300],[750,300],[750,500],[500,500]],grasp_angle_deg=90)
    for k in ['identity_confirmed','self_fold_evidence','body_to_root_connection','direction_opens_fold','projected_motion_clear']:payload[k]=True
    jsonschema.validate(payload,schema)
    env={'LOCKED_GOAL':goal,'FROZEN_PROMPT':prompt,'VISION_MODEL':lambda rgb,p:payload}
    exec(checked_compile(code),env);out=env['generate_action'](np.zeros((720,1280,3),dtype=np.uint8),goal[7:-1])
    assert out['decision']=='ACTION' and out['goal']==goal and out['grasp_px']!=out['target_px']
    # The legacy final-action-only return cannot carry these required geometric fields.
    legacy={k:v for k,v in out.items()}
    env['VISION_MODEL']=lambda rgb,p:legacy
    blocked=env['generate_action'](np.zeros((720,1280,3),dtype=np.uint8),goal[7:-1])
    assert blocked['decision']=='ABSTAIN'
    return {'correct_intermediate_contract_allows_action':True,'legacy_final_only_contract_blocks_action':True,'fixture_is_synthetic_not_physical_evidence':True,'legacy_rejection':blocked['abstain_reason']}

def main():
    FIX.mkdir(exist_ok=True)
    original=read(OUT/'skill_freeze.json');goal=original['goal']
    code=(OUT/'candidate_skill.py').read_text();prompt=(OUT/'inference_prompt.txt').read_text()
    assert sha(OUT/'candidate_skill.py')==original['code_sha256'] and sha(OUT/'inference_prompt.txt')==original['prompt_sha256']
    schema=vision_schema(goal)
    save(FIX/'vision_schema.json',schema);save(FIX/'contract_test.json',smoke_contract(code,prompt,goal,schema))
    base=ROOT/'reports/hindsight_fast_test_20261029'
    split=read(base/'split.json');idx=read(base/'experience_index.json')
    previous=read(ROOT/'reports/hindsight_corner_continuation_20261030/input_manifest.json')['validation_before_only']
    old=read(OUT/'validation_inputs_before_only.json')
    used={x['before_sha256'] for x in split['validation']+previous+old}
    chosen=[]
    for run in split['heldout_complete_runs']:
        pool=[x for x in idx if x['run']==run and x['before_sha256'] not in used]
        row=min(pool,key=lambda x:hashlib.sha256(('schema-bridge-only-20261030|'+x['before_sha256']).encode()).hexdigest())
        chosen.append({'id':f'R{len(chosen)+1:02d}','run':run,'before':str(ROOT/row['before']),'before_sha256':row['before_sha256']})
    train=read(OUT/'experience_index.json')
    assert not ({x['run'] for x in chosen}&{x['run'] for x in train})
    assert not ({x['before_sha256'] for x in chosen}&{x[k+'_sha256'] for x in train for k in ['before','after']})
    save(FIX/'validation_inputs_before_only.json',chosen)
    freeze={'time':time.time(),'change':'VISION_MODEL response schema adapter only; generated code, prompt, selection and hypotheses unchanged','goal':goal,'skill_code_sha256':sha(OUT/'candidate_skill.py'),'skill_prompt_sha256':sha(OUT/'inference_prompt.txt'),'adapter_sha256':sha(__file__),'vision_schema_sha256':sha(FIX/'vision_schema.json'),'reason':'Generated Python consumes intermediate geometry but original host constrained model output to final Pa/Pb schema. ACTION branch was unreachable through that bridge.','test_images_seen_by_skill_induction':0,'previous_smoke_images_excluded':True,'prior_test_outputs_supplied_to_a_model_for_repair':False,'no_model_based_skill_revision':True}
    if (FIX/'freeze.json').exists():
        oldfreeze=read(FIX/'freeze.json')
        assert {k:v for k,v in oldfreeze.items() if k!='time'}=={k:v for k,v in freeze.items() if k!='time'}
        freeze=oldfreeze
    else:save(FIX/'freeze.json',freeze)
    for f in ['summary.json','review.html','validation_record.json']:
        dest=FIX/('initial_'+f)
        if not dest.exists():shutil.copyfile(OUT/f,dest)
    results=[]
    for i,row in enumerate(chosen):
        assert sha(row['before'])==row['before_sha256']
        def request(text,w,h):
            result=call(OUT,f'repair_smoke_{i:02d}',
                        '执行给定冻结的候选技能视觉提示。返回中间几何对象，由冻结Python生成最终Pa/Pb；这不是物理验证。ABSTAIN时无依据的几何填null、布尔项填false，不捏造几何。不要新增字段，不输出target_px。',
                        {'locked_goal':goal,'skill_prompt':text,'image_size':[w,h]},[Path(row['before'])],schema)
            if result['status']!='COMPLETE':raise RuntimeError(result.get('error'))
            return result['output']
        start=time.time();action,violations=invoke(code,prompt,goal,row['before'],request)
        d=FIX/row['id'];d.mkdir(exist_ok=True);shutil.copyfile(row['before'],d/'before.png');overlay(row['before'],action,d/'overlay.png')
        results.append({'id':row['id'],'action':action,'contract_violations':violations,'latency_s':time.time()-start,'physical_success':'NOT_TESTED','host_safety':'NOT_TESTED'})
        save(FIX/'results.json',results)
        print(json.dumps({'stage':'INTERMEDIATE_SCHEMA_REPAIR_TEST','completed':len(results),'total':3,'decision':action['decision']},ensure_ascii=False),flush=True)
    assert sha(OUT/'candidate_skill.py')==original['code_sha256'] and sha(OUT/'inference_prompt.txt')==original['prompt_sha256']
    save(FIX/'summary.json',{'status':'COMPLETE','actions':sum(x['action']['decision']=='ACTION' for x in results),'abstentions':sum(x['action']['decision']=='ABSTAIN' for x in results),'violations':sum(len(x['contract_violations']) for x in results),'skill_and_prompt_unchanged':True,'physical_actions':0,'test_scope':'software integration repair, not policy improvement or independent episode success'})
    print('REPAIR_COMPLETE',flush=True)
if __name__=='__main__':main()
