"""One offline RSI code update. This module has no robot execution path."""
from pathlib import Path
import argparse, ast, dataclasses, hashlib, importlib.util, json, math, sys
import cv2
import numpy as np
from PIL import Image

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
from cloth_agent.cloth_harness.runtime.dataset import load_observation
from cloth_agent.cloth_harness.runtime.interfaces import Action
from cloth_agent.cloth_harness.cv_harness.outcome_observer import observe
from cloth_agent.cloth_harness.cv_harness.segmentation import segment
ROOT=Path(__file__).parent
OUT=REPO/'reports/rsi_code_iteration_01_20261029'
def read(p):return json.loads(Path(p).read_text())
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
PROTECTED=[REPO/p for p in ['cloth_agent/geometry_report.py','cloth_agent/cloth_harness/cv_harness/segmentation.py','cloth_agent/cloth_harness/cv_harness/motion_observer.py','cloth_agent/cloth_harness/cv_harness/outcome_observer.py','cloth_agent/cloth_harness/runtime/interfaces.py']]
def assert_frozen():
    for p,h in read(OUT/'evaluator_freeze.json').items():assert sha(p)==h, 'Evaluator modified: '+p
def prepare():
    if (OUT/'inventory.json').exists():raise ValueError('Already prepared; do not overwrite frozen experiment')
    save(OUT/'evaluator_freeze.json',{str(p):sha(p) for p in PROTECTED})
    selections=[('S1','seed','154723_183805',0),('S2','seed','160504_473135',0),('S3','seed','161102_776945',0),
                ('P1','pool','163419_725630',3),('P2','pool','163419_725630',4),('P3','pool','163419_725630',5),
                ('H1','heldout','165427_789294',3),('H2','heldout','171553_732107',7)]
    inventory=[]
    for ident,split,run,idx in selections:
        folder=REPO/'reports'/('minimal_one_shot_6iter_20261028_'+run)/f'round_{idx:03d}'
        before=load_observation(folder/'before');after=load_observation(folder/'after')
        dispatched=read(folder/'dispatch_intent.json');log=read(folder/'execution.json');plan=dispatched['plan']
        assert log['status']=='EXECUTED' and log['physical_execution'] is True
        assert dispatched['observation_sha256']==sha(folder/'before/rgb.png')
        assert after.timestamp>before.timestamp
        action={k:plan[k] for k in ['grasp_px','target_px','motion_type','grasp_angle_deg']}
        action['measured_distance_mm']=read(folder/'geometry.json')['distance_mm']
        outcome,mb,ma=observe(before,Action(tuple(action['grasp_px']),tuple(action['target_px']),action['grasp_angle_deg']),after,log)
        dest=OUT/'data'/ident;dest.mkdir(parents=True)
        for side in ['before','after']:(dest/(side+'.png')).write_bytes((folder/side/'rgb.png').read_bytes())
        for side,mask in [('before',mb),('after',ma)]:Image.fromarray(mask.astype('uint8')*255).save(dest/(side+'_mask.png'))
        save(dest/'feedback.json',dataclasses.asdict(outcome))
        save(dest/'action.json',action)
        inventory.append({'id':ident,'split':split,'source':str(folder),'run':run,'action':action,'before_sha256':sha(dest/'before.png'),'after_sha256':sha(dest/'after.png')})
    save(OUT/'inventory.json',inventory)
    packet={'stage':'hypothesis_and_replay_selection','seed':[],'pool':[]}
    images=[]
    for row in inventory:
        if row['split']=='heldout':continue
        entry={'id':row['id'],'recorded_action':row['action']}
        images.append(str(OUT/'data'/row['id']/'before.png'));entry['before_image_index']=len(images)
        if row['split']=='seed':
            images.append(str(OUT/'data'/row['id']/'after.png'));entry['after_image_index']=len(images)
            entry['fixed_feedback']=read(OUT/'data'/row['id']/'feedback.json')
        packet['seed' if row['split']=='seed' else 'pool'].append(entry)
    save(OUT/'stage1_packet.json',packet);save(OUT/'stage1_images.json',images)
    save(OUT/'protocol.json',{'iteration_budget':1,'physical_actions':0,'mode':'historical_replay','source_selection':'first valid completed records for seed, three later states from another run for replay, two further runs for heldout; screened by existing segmentation quality, not outcome sign','seed_run_count':3,'pool_run_count':1,'heldout_run_count':2,'seed_rows_correlated':False,'evaluation':'Frozen existing outcome_observer; area proxy uncalibrated; registration/local tracking may be UNKNOWN. No new-policy outcome is fabricated.','policy_contract':'policy(observation)->action and predict_delta(observation, recorded_action)->float or None','learner_input':'RGB images, masks through pure code input; primitive contract and measured recorded actions. No old planner prose, skills, prompts or test outcomes.','heldout':'not provided to learner; evaluated after v1 frozen','generalization':'heldout recorded-action outcome prediction plus synthetic geometric equivariance; not new-policy physical success','physical_path':'not exposed; future actions must use existing prepare_action and native path/scene/workspace checks plus current explicit authorization'})
    save(OUT/'context_audit.json',{'root_context_exposed_to_human_skill':True,'root_role':'data/interface/evaluation orchestration only','discovery':'fresh native CLI context with custom neutral instructions, no tools/memory/plugins/project docs or skills, no history','limitation':'Isolation of supplied context, not a claim that the pretrained model has no prior knowledge.'})
    print('Prepared 3 seed records, 3 unrevealed replay candidates and 2 heldout states.')
def install(stage):
    result=read(OUT/f'stage{stage}'/'answer.json');code=result['policy_code']
    tree=ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node,(ast.Import,ast.ImportFrom)):
            modules=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module]
            assert all(m in ['numpy','cv2','math'] for m in modules),'Only numeric imports allowed'
        if isinstance(node,ast.Name):assert node.id not in ['open','exec','eval','compile','__import__','getattr','globals','locals','input','breakpoint']
        if isinstance(node,ast.Attribute):
            assert not node.attr.startswith('__') and node.attr not in ['load','save','loadtxt','savetxt','fromfile','tofile','imread','imwrite','VideoCapture','FileStorage','memmap','load_library']
    version=f'v{stage-1}';dest=OUT/'policies'/f'policy_{version}.py'
    if dest.exists():assert dest.read_text()==code
    else:dest.parent.mkdir(exist_ok=True);dest.write_text(code)
    save(OUT/'policies'/f'{version}_manifest.json',{'code_sha256':sha(dest),'generated_by':'isolated native discovery agent','stage':stage,'source':str(OUT/f'stage{stage}'/'answer.json')})
    print('Frozen',version,sha(dest))
def reveal():
    assert_frozen();a=read(OUT/'stage1/answer.json');assert a['selected_replay'] in ['P1','P2','P3']
    install(1);pick=a['selected_replay']
    baseline=[];m=module('v0')
    for row in read(OUT/'inventory.json'):
        if row['split']=='heldout':continue
        obs=observation(row['id']);plan=validate_action(m.policy(obs),obs)
        pred=m.predict_delta(obs,row['action'])
        baseline.append({'id':row['id'],'policy_action':plan,'recorded_action_prediction':float(pred) if pred is not None else None})
    save(OUT/'v0_pre_feedback_outputs.json',baseline)
    save(OUT/'selection_freeze.json',{'selected_replay':pick,'stage1_response_sha256':sha(OUT/'stage1/answer.json'),'policy_v0_sha256':sha(OUT/'policies/policy_v0.py'),'unselected_outcomes_exposed':False})
    packet={'stage':'one_feedback_and_one_code_update','frozen_prior':a,'selected_replay':pick,'feedback':read(OUT/'data'/pick/'feedback.json'),'recorded_action':read(OUT/'data'/pick/'action.json'),'image_order':['selected before','selected after']}
    save(OUT/'stage2_packet.json',packet);save(OUT/'stage2_images.json',[str(OUT/'data'/pick/(s+'.png')) for s in ['before','after']])
    print('Revealed exactly one recorded result:',pick)
def module(version):
    p=OUT/'policies'/f'policy_{version}.py'
    assert sha(p)==read(OUT/'policies'/f'{version}_manifest.json')['code_sha256'],'Frozen policy changed'
    spec=importlib.util.spec_from_file_location('rsi_'+version,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def observation(ident):
    rgb=np.asarray(Image.open(OUT/'data'/ident/'before.png').convert('RGB'));mask=np.asarray(Image.open(OUT/'data'/ident/'before_mask.png'))>0
    return {'rgb':rgb,'mask':mask}
def validate_action(a,obs):
    assert a['decision'] in ['ACTION','ABSTAIN','DONE']
    assert isinstance(a.get('reason'),str)
    if a['decision']=='ACTION':
        assert a['motion_type']=='lift_pull_place'
        h,w=obs['mask'].shape
        for k in ['grasp_px','target_px']:
            x,y=a[k];assert isinstance(x,(int,np.integer)) and isinstance(y,(int,np.integer)) and 0<=x<w and 0<=y<h
        x,y=a['grasp_px'];assert obs['mask'][y,x],'grasp not visible in mask'
        assert a.get('grasp_angle_deg') is None
    else:assert all(a.get(k) is None for k in ['grasp_px','target_px','motion_type'])
    return a
def evaluate():
    assert_frozen();install(2);rows=[];transforms=[]
    pick=read(OUT/'selection_freeze.json')['selected_replay']
    for version in ['v0','v1']:
        m=module(version)
        for row in read(OUT/'inventory.json'):
            obs=observation(row['id']);a=validate_action(m.policy(obs),obs)
            prediction=m.predict_delta(obs,row['action'])
            if prediction is not None:prediction=float(prediction);assert math.isfinite(prediction)
            actual=read(OUT/'data'/row['id']/'feedback.json')['coverage']['delta_ratio']
            exposure = ('seed' if row['split']=='seed' else 'selected_replay' if row['id']==pick else 'unseen_state' if row['split']=='heldout' else 'seen_input_unrevealed_outcome')
            rows.append({'version':version,'id':row['id'],'exposure':exposure,'split':row['split'],'generated_action':a,'predicted_recorded_action_area_delta':prediction,'actual_recorded_action_area_delta':actual,'absolute_error':abs(prediction-actual) if prediction is not None and actual is not None else None,'new_policy_physical_outcome':'UNOBSERVED'})
        for ident in ['S2','H1','H2']:
            obs=observation(ident);base=validate_action(m.policy(obs),obs);h,w=obs['mask'].shape
            # Rotate full image/mask 90 degrees; no resegmentation with camera-specific ROI.
            rot={'rgb':np.rot90(obs['rgb']).copy(),'mask':np.rot90(obs['mask']).copy()};ar=validate_action(m.policy(rot),rot)
            # Translate cloth/image together into a padded canvas, avoiding clipping.
            dx,dy=47,31
            shifted={'rgb':np.pad(obs['rgb'],((dy,17),(dx,23),(0,0))),'mask':np.pad(obs['mask'],((dy,17),(dx,23)))}
            at=validate_action(m.policy(shifted),shifted)
            for kind,other,mapping in [('rotation90',ar,lambda p:[p[1],w-1-p[0]]),('translation',at,lambda p:[p[0]+dx,p[1]+dy])]:
                result={'version':version,'id':ident,'transform':kind,'same_decision':base['decision']==other['decision'],'physical_trial':False}
                if base['decision']==other['decision']=='ACTION':
                    for k in ['grasp_px','target_px']:result[k+'_equivariance_error_px']=float(np.linalg.norm(np.asarray(mapping(base[k]))-other[k]))
                transforms.append(result)
    save(OUT/'evaluation.json',{'predictions':rows,'equivariance':transforms,'limitations':'Different generated actions have no observed outcome. Recorded-action prediction is not policy success. Synthetic rotation/translation test coordinate behavior, not cloth dynamics.'})
    save(OUT/'status.json',{'status':'COMPLETE_ONE_OFFLINE_ITERATION','policy_versions':2,'physical_actions':0,'replayed_results_revealed':1})
    print(json.dumps({'rows':len(rows),'equivariance_checks':len(transforms)},ensure_ascii=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','reveal','evaluate']);a=p.parse_args();globals()[a.stage]()
