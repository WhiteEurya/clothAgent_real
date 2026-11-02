"""Read-only historical mining. No policy imports or robot execution."""
from pathlib import Path
import sys,json,hashlib,datetime,subprocess,collections,dataclasses,re
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from cloth_agent.cloth_harness.runtime.dataset import load_observation
from cloth_agent.cloth_harness.runtime.interfaces import Action
from cloth_agent.cloth_harness.cv_harness.outcome_observer import observe
OUT=ROOT/'reports/offline_failure_mining_20261029';OUT.mkdir(exist_ok=True)
def read(p,default=None):
 try:return json.loads(Path(p).read_text())
 except (OSError,ValueError):return {} if default is None else default
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rel(p):return str(Path(p).relative_to(ROOT))
def main():
 protocol=OUT/'protocol.json'
 if protocol.exists():cfg=read(protocol)
 else:
  end=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)));start=end-datetime.timedelta(days=7)
  cfg={'start':start.isoformat(),'end':end.isoformat(),'start_unix':start.timestamp(),'end_unix':end.timestamp(),'window':'rolling 7 days, Asia/Singapore','physical_actions':0,'policy_changes':False,'screen':'3 consecutive completed, provenance-valid, same-view, usable-segmentation actions: all delta<=2% => no-positive-progress candidate; all abs(delta)<=2% => plateau candidate. 1%, 2%, 3% sensitivity. Not calibrated success labels.','dedup':'before/after RGB hash, capture time, dispatched coordinates and execution status; copied records excluded; initialization actions indexed separately','limits':'Completed robot commands do not establish effective cloth grasp. Area is visible pixel proxy. Unknown local tracking stays unknown.'};save(protocol,cfg)
 files=subprocess.check_output(['rg','--files','--hidden','--no-ignore','-g','!**/.git/**','-g','!**/node_modules/**','-g','!**/source_snapshot/**','-g','execution.json','-g','episode_summary.json','-g','observation.json'],cwd=ROOT,text=True).splitlines()
 starts=set();unhandled=[]
 for f in files:
  p=ROOT/f
  if p.name=='observation.json' and p.parent.name in ['before','pre_execution_observation']:
   m=read(p);t=m.get('captured_unix_s',0)
   if cfg['start_unix']<=t<=cfg['end_unix'] and m.get('synthetic') is False:starts.add(p.parent.parent)
  if p.name=='execution.json' and not (p.parent/'before/observation.json').exists():unhandled.append(f)
 runs={};duplicates=[];seen={};excluded=[];cache=OUT/'steps';cache.mkdir(exist_ok=True)
 # Original dispatch directories before derived archives; deterministic.
 starts=sorted(starts,key=lambda p:(not (p/'dispatch_intent.json').exists(),'archive' in str(p) or 'replay' in str(p),len(p.parts),str(p)))
 for folder in starts:
  run=folder.parent if re.match(r'(round|step)_\d+',folder.name) else folder
  if folder.name=='execution' and folder.parent.name.startswith('iter_'):run=folder.parent.parent
  is_init=any(x=='initialization' or re.fullmatch(r'reset_\d+',x) for x in folder.relative_to(ROOT).parts)
  before_dir=folder/('before' if (folder/'before/observation.json').exists() else 'pre_execution_observation')
  meta=read(before_dir/'observation.json');log=read(folder/'execution.json');intent=read(folder/'dispatch_intent.json');plan=intent.get('plan',{})
  if not plan:plan=read(folder/'plan.json')
  action={k:plan.get(k) for k in ['decision','grasp_px','target_px','motion_type','grasp_angle_deg']}
  geometry=read(folder/'geometry.json');primitive=geometry.get('primitive',{})
  action.update({k:geometry.get(k) for k in ['distance_mm','rpy_world_deg']});action['lift_height_mm']=primitive.get('lift_height_mm')
  before=before_dir/'rgb.png';after=folder/'after/rgb.png';am=read(folder/'after/observation.json')
  bh=sha(before) if before.exists() else None;ah=sha(after) if after.exists() else None
  key=json.dumps([meta.get('captured_unix_s'),bh,ah,log.get('status')],sort_keys=True)
  if key in seen:duplicates.append({'duplicate':rel(folder),'canonical':seen[key]});continue
  seen[key]=rel(folder)
  rid=rel(run);sid=hashlib.sha256(str(folder).encode()).hexdigest()[:16]
  r=runs.setdefault(rid,{'path':rid,'steps':[],'summary':{k:v for k,v in read(run/'episode_summary.json').items() if k in ['started_at','termination_reason','termination_detail','total_rounds','physical_actions']},'status':read(run/'status.json'),'human_review':read(run/'human_review.json'),'initialization':is_init})
  row={'id':sid,'source':rel(folder),'step':folder.name,'timestamp':meta['captured_unix_s'],'captured_at':meta.get('captured_at'),'before':rel(before) if before.exists() else None,'after':rel(after) if after.exists() else None,'before_sha256':bh,'after_sha256':ah,'execution':log,'action':action,'recorded_evaluation_files':[rel(p) for pattern in ['*evaluation*.json','*feedback*.json','outcome.json','*score*.json','*result*.json'] for p in folder.glob(pattern)],'valid_pair':False,'evaluation':None}
  reason=None
  if not log.get('physical_execution') or log.get('completed_commands',0)<=0:reason='no physical command evidence'
  elif log.get('status')!='EXECUTED':reason='partial/interrupted execution; not a completed action pair'
  elif not before.exists() or not after.exists():reason='missing before or after RGB'
  elif am.get('synthetic') is not False or am.get('captured_unix_s',0)<=meta['captured_unix_s']:reason='nonreal/nonchronological after'
  elif (intent.get('observation_sha256') or intent.get('before_sha256') or intent.get('binding',{}).get('before'))!=bh or plan.get('decision')!='ACTION':reason='missing/mismatched dispatched action provenance'
  elif intent.get('plan_sha256') and intent['plan_sha256']!=sha(folder/'plan.json'):reason='plan hash mismatch'
  elif am.get('sha256',{}).get('rgb.png')!=ah or meta.get('sha256',{}).get('rgb.png')!=bh:reason='saved RGB hash mismatch'
  if reason is None:
   try:
    outpath=cache/(sid+'.json')
    if outpath.exists():ev=read(outpath)
    else:
     out,mb,ma=observe(load_observation(before_dir),Action(tuple(plan['grasp_px']),tuple(plan['target_px']),plan.get('grasp_angle_deg')),load_observation(folder/'after'),log)
     ev=dataclasses.asdict(out);save(outpath,ev)
    row.update(valid_pair=True,evaluation=ev)
   except Exception as e:reason=type(e).__name__+': '+str(e)
  if reason:row['exclusion']=reason;excluded.append({'source':rel(folder),'reason':reason})
  r['steps'].append(row)
  if len(seen)%50==0:print('indexed',len(seen),flush=True)
 for r in runs.values():
  r['steps'].sort(key=lambda s:s['timestamp']);term=r['summary'].get('termination_reason','UNKNOWN')
  statuses=[s['execution'].get('status','') for s in r['steps']]
  term=term or 'UNKNOWN'
  r['termination']=term;r['human_interrupted']=term in ['USER_STOP','USER_INTERRUPTED','KEYBOARD_INTERRUPT'] or 'USER_INTERRUPTED' in statuses
  r['done_recorded']=term=='DONE' or any(s['action'].get('decision')=='DONE' for s in r['steps'])
  r['physical_command_steps']=sum(s['execution'].get('physical_execution') is True and s['execution'].get('completed_commands',0)>0 for s in r['steps'])
  r['valid_pairs']=sum(s['valid_pair'] for s in r['steps']);r['numeric_pairs']=sum(s['evaluation'] is not None and s['evaluation']['coverage']['delta_ratio'] is not None for s in r['steps'])
  r['screen']={}
  for threshold in [.01,.02,.03]:
   windows=[]
   for i in range(len(r['steps'])-2):
    seq=r['steps'][i:i+3];ev=[s['evaluation'] for s in seq]
    if any(e is None or e['coverage']['delta_ratio'] is None for e in ev):continue
    nums=[int(re.search(r'(\d+)$',s['step']).group(1)) if re.search(r'(\d+)$',s['step']) else -999 for s in seq]
    if nums[1]!=nums[0]+1 or nums[2]!=nums[1]+1:continue
    ds=[e['coverage']['delta_ratio'] for e in ev]
    if max(ds)<=threshold:windows.append({'start':seq[0]['step'],'end':seq[-1]['step'],'ids':[s['id'] for s in seq],'deltas':ds,'plateau':all(abs(x)<=threshold for x in ds)})
   r['screen'][str(threshold)]={'first_nonprogress':next(iter(windows),None),'first_plateau':next((w for w in windows if w['plateau']),None),'windows':windows}
  r['difficult_candidate']=r['human_interrupted'] or bool(r['screen']['0.02']['windows']) or not r['done_recorded']
 auxiliary=[r for r in runs.values() if r['initialization'] or not (r['physical_command_steps'] or r['summary'] or r['status'])]
 save(OUT/'auxiliary_observations.json',auxiliary)
 runs={k:r for k,r in runs.items() if r not in auxiliary}
 physical=[r for r in runs.values() if r['physical_command_steps']>0]
 summary={'runs':len(runs),'runs_with_physical_commands':len(physical),'human_interrupted_runs':sum(r['human_interrupted'] for r in physical),'no_done_runs':sum(not r['done_recorded'] for r in physical),'nonprogress_runs':sum(bool(r['screen']['0.02']['windows']) for r in physical),'plateau_runs':sum(r['screen']['0.02']['first_plateau'] is not None for r in physical),'valid_pairs':sum(r['valid_pairs'] for r in runs.values()),'numeric_pairs':sum(r['numeric_pairs'] for r in runs.values()),'deduplicated_copies':len(duplicates),'termination_counts':dict(collections.Counter(r['termination'] for r in physical)),'auxiliary_capture_groups':len(auxiliary),'filter_counts_scope':'runs with physical command evidence; nonexecuted attempts indexed separately in same table'}
 save(OUT/'index.json',{'summary':summary,'runs':list(runs.values())});save(OUT/'excluded.json',excluded);save(OUT/'duplicates.json',duplicates);save(OUT/'unhandled_execution_paths.json',unhandled)
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
