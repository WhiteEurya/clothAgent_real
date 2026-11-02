"""Stage 6: current input only; existing robot adapter unchanged; one action cap."""
from pathlib import Path
import sys,argparse,importlib.util,time,json,subprocess
import numpy as np
from PIL import Image
from transport import read,save,sha,call
from contracts import EFFECT_SCHEMA,OBS_PROMPT
from policy import generate
CODE=Path(__file__).resolve().parent;REPO=CODE.parents[1]

def module(path,name):
 sys.path.insert(0,str(path.parent));sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prepare-live',action='store_true');p.add_argument('--execute',action='store_true');p.add_argument('--onsite-confirmed',action='store_true');p.add_argument('--operator');args=p.parse_args();root=args.root.resolve();freeze=read(root/'skill_freeze.json')
 assert sha(root/'skill_spec.json')==freeze['skill_spec_sha256'] and sha(root/'candidate_skill.py')==freeze['candidate_code_sha256']
 record={'status':'PENDING','physical_actions':0,'maximum_actions':1,'skill_frozen':freeze,'prediction_saved_before_after':False}
 if not args.prepare_live:save(root/'validation_record.json',record);return
 live=root/'validation';live.mkdir(exist_ok=True)
 if (live/'dispatch_intent.json').exists():raise RuntimeError('This experiment already dispatched; never repeat')
 try:
  # Read-only capture adapter already used in the project. No reset/perception motion.
  if not (live/'before_capture/observation/rgb.png').exists():
   cp=REPO/'experiments/hindsight_skill_discovery_20261029_01/capture.py'
   subprocess.run([sys.executable,str(cp),'--output',str(live/'before_capture')],cwd=REPO,check=True)
  rgb=live/'before_capture/observation/rgb.png'
  skill=read(root/'skill_spec.json')
  if skill.get('status')!='CANDIDATE':record.update(status='PENDING_NO_EXECUTABLE_SKILL');return
  action=generate(rgb,'AUTO_SELECT_CURRENT_HIDDEN_SEMANTIC_ENDPOINT',root,live/'action_request')
  record['generated_action']=action
  # Near-identical historical input is explicitly disallowed as a new-state trial.
  current=np.asarray(Image.open(rgb).convert('RGB').resize((160,90)),float)[9:81,16:144]
  nearest=[]
  for row in read(root/'experience_index.json'):
   for key in ['before','after']:
    q=np.asarray(Image.open(row[key]).convert('RGB').resize((160,90)),float)[9:81,16:144]
    delta=float(np.quantile(np.abs(current-q),.9));nearest.append((delta,row['id'],key))
  nearest.sort();record['nearest_historical_views']=nearest[:5]
  if action.get('decision')!='ACTION':record.update(status='PENDING_ABSTAIN',reason=action.get('abstain_reason'));return
  proposal={**action,'decision':'EXPLORATORY_ACTION','reason':action['visible_evidence'],'purpose':action['expected_change']}
  save(live/'prediction_before_execution.json',{'proposal':proposal,'saved_at_unix':time.time(),'before_sha256':sha(rgb),'skill_sha256':freeze['candidate_code_sha256']});record['prediction_saved_before_after']=True
  adapter=module(REPO/'experiments/hindsight_skill_discovery_20261029_01/plan.py','validation_plan_adapter');adapter.BASE=live;adapter.OUT=live/'planned_action_01'
  if not (adapter.OUT/'result.json').exists():
   old=sys.argv;sys.argv=['plan.py','--observation',str(live/'before_capture/observation'),'--output',str(adapter.OUT)]
   try:adapter.main()
   finally:sys.argv=old
  host=read(adapter.OUT/'result.json');record['host']=host
  if host.get('execution_permission')!='YES':record.update(status='PENDING_HOST_REJECTED');return
  if nearest and nearest[0][0]<10:record.update(status='PENDING_NEW_STATE',reason='Current scene near historical state under frozen image novelty screen; no independent physical validation');return
  if not(args.execute and args.onsite_confirmed and args.operator):record.update(status='PENDING_EXISTING_ONSITE_PERMISSION',reason='Complete action and host plan saved; execute only with existing onsite-confirmed run interface');return
  runner=module(REPO/'experiments/hindsight_skill_discovery_20261029_01/execute_once.py','validation_execution_adapter');runner.R=live;runner.P=live/'planned_action_01';old=sys.argv;sys.argv=['execute_once.py','--execute']
  try:runner.main()
  finally:sys.argv=old
  execution=read(live/'live_execution/status.json');record['execution']=execution;record['physical_actions']=execution['physical_actions']
  if execution['status']!='EXECUTED':record.update(status='EXECUTION_STOPPED');return
  after=live/'live_execution/after/rgb.png'
  # Blind outcome observer deliberately receives NO goal, prediction or skill.
  observed=call(root,'validation_blind_observer',OBS_PROMPT,{'image_order':['VALIDATION Before','VALIDATION After'],'records':[{'id':'VALIDATION','execution_status':'EXECUTED'}]},[rgb,after],EFFECT_SCHEMA)
  record['blind_observed_effect']=observed
  if observed['status']=='COMPLETE':
   v=observed['output']['effects'][0];record['target_reveal_match']=v['classification']=='CONFIRMED_SEMANTIC_REVEAL' and v['endpoint']==action['target_corner']
   record['status']='VALIDATED_ONE_TRIAL'
   # Refinement remains an unvalidated proposal, never promoted automatically.
   from contracts import HYP_SCHEMA,HYP_PROMPT
   refin=call(root,'candidate_refinement',HYP_PROMPT,{'frozen_skill':skill,'prediction':action,'blind_observed_effect':v,'actual_execution':execution,'instruction':'Only propose candidate refinement; no new execution, no claim of improvement.'},[rgb,after],HYP_SCHEMA)
   save(root/'candidate_refinement_v1.json',refin);record['refinement_deployed']=False
  else:record.update(status='EXECUTED_EVALUATION_FAILED')
 except Exception as e:record.update(status='PENDING_OR_STOPPED',error=f'{type(e).__name__}: {e}')
 finally:
  save(root/'validation_record.json',record);print(json.dumps(record,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
