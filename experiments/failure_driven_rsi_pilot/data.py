"""Original-episode inventory, prospectively fixed split and factual packets."""
from common import *

def scan():
 save(OUT/'protocol.json',{'at':utc(),'experiment':'failure_driven_rsi_pilot','budget':{'planned':15,'maximum':16,'discovery':3,'prediction':3,'evaluation':3,'planner':6},'offline_only':True,'model':'codex exec -p api; gpt-6-astra/high','input_boundary':'No Expert prompt, prior Rule Mining, prior Policy Evolution conclusions or old Planner reasons. Only raw imagery, numeric actions, factual execution. Generic CLI harness metadata unavoidable; actual tool use invalidates response.','selection':'Scan reports/*/episode_*/round_*. Middle complete executed manipulation per episode. Group-disjoint heldout: hash-sort run groups, take first 3. Discovery screening: next 9 different groups, one hash-first episode each. Review original before/after pairs for no clear gain, local unfolding, mixed effects. Keep every screened pair and rejection reason. No skill/content-driven case selection.','holdout_blinding':'Do not decode heldout before/after/process until unique rule frozen; inventory and path/metadata-only checks allowed. All 3 predictions frozen before any heldout process/after packet is made.','teacher_stages':'Observe without editing strategy; choose one failure with <=3 competing physical explanations and concrete alternative; compress to one skill without changing hypothesis; freeze exact output.','planner':'Fresh independent A/B requests on same current before RGB, neutral task/interface/current primitive only. No heldout outcomes or old planner. A/B order AB,BA,AB. No previous-action images supplied; rules requiring temporal evidence may legitimately be UNKNOWN/not triggered.','static_checks':'schema, pixels, coarse cloth mask, recorded depth/support-plane target, metric distance, archived Cartesian endpoint bounds. No robot, live Host, IK or collision execution.','meaningful_action_change':'Pa or Pb delta >15px or direction change >15 degrees. Single sample per arm; changes not automatically attributable to rule rather than sampling.','outcomes':'Supported/contradicted/inconclusive for frozen prediction; not applicable when trigger absent. No historical alternative execution implies no efficacy proof.'})
 protected=[p for top in ['cloth_agent','config'] for p in (ROOT/top).rglob('*') if p.is_file() and p.suffix in ['.py','.json','.yaml','.yml','.toml','.txt','.md']]
 save(OUT/'protected_before.json',{str(p.relative_to(ROOT)):sha(p) for p in protected})
 inventory=[];eligible=[]
 needed=['before/rgb.png','before/observation.json','after/rgb.png','after/observation.json','points.json','geometry.json','motion.py','states.jsonl']
 for ep in sorted((ROOT/'reports').glob('*/episode_*')):
  if not ep.is_dir():continue
  rs=[]
  for r in sorted(ep.glob('round_*')):
   e=read(r/'execution.json') if (r/'execution.json').exists() else {};miss=[x for x in needed if not (r/x).exists()]
   real=(read(r/'before/observation.json').get('synthetic') is False) if (r/'before/observation.json').exists() else False
   ok=not miss and real and e.get('physical_execution') is True and e.get('status')=='EXECUTED'
   frames=sum(1 for _ in (r/'execution_images/frames.jsonl').open()) if (r/'execution_images/frames.jsonl').exists() else 0
   inventory.append({'round':str(r.relative_to(ROOT)),'status':e.get('status'),'physical_execution':e.get('physical_execution'),'missing':miss,'real':real,'process_frames':frames,'eligible':ok})
   if ok:rs.append(r)
  if rs:eligible.append({'episode':str(ep.relative_to(ROOT)),'group':ep.parent.name,'round':str(rs[len(rs)//2].relative_to(ROOT)),'complete_rounds':[r.name for r in rs]})
 groups=sorted({e['group'] for e in eligible},key=key);held=[];pool=[]
 for i,g in enumerate(groups[:12]):
  c=dict(min([e for e in eligible if e['group']==g],key=lambda e:key(e['episode'])))
  c['id']=f'H{i+1}' if i<3 else f'C{i-2}';(held if i<3 else pool).append(c)
 save(OUT/'inventory.json',inventory);save(OUT/'eligible_episodes.json',eligible);save(OUT/'heldout_reserved.json',held);save(OUT/'discovery_screening_pool.json',pool)
 freeze('reserved_split',[OUT/f for f in ['protocol.json','inventory.json','eligible_episodes.json','heldout_reserved.json','discovery_screening_pool.json']])
 for c in pool:
  r=ROOT/c['round'];im=Image.new('RGB',(1280,384),'#111');d=ImageDraw.Draw(im)
  for j,n in enumerate(['before/rgb.png','after/rgb.png']):im.paste(Image.open(r/n).convert('RGB').resize((640,360)),(j*640,24));d.text((j*640+5,5),c['id']+' '+n,fill='white')
  dest=OUT/'screening'/f'{c["id"]}.jpg';dest.parent.mkdir(exist_ok=True);im.save(dest,quality=92)
 # Same model metadata cache, contains no project/cloth data; no auth copied.
 cache=read(Path.home()/'.codex/models_cache.json');m=next(m for m in cache['models'] if m['slug']=='gpt-6-astra');save(OUT/'local_model_catalog.json',{'models':[m]})
 cfg=read(ROOT/'config/dual_arm.local.json');save(OUT/'static_environment.json',{'source':'config/dual_arm.local.json','source_sha256':sha(ROOT/'config/dual_arm.local.json'),'frame_id':cfg['frame_id'],'world_from_base_mm':cfg['arms']['left']['world_from_base_mm'],'workspace':cfg['arms']['left']['workspace']})
 print(json.dumps({'eligible_episodes':len(eligible),'heldout':held,'screening':pool},ensure_ascii=False,indent=2))

def select():
 judgments=read(OUT/'screening_judgments.json');pool=read(OUT/'discovery_screening_pool.json')+read(OUT/'screening_extension_pool.json');chosen=[]
 for ix,j in enumerate([j for j in judgments if j['selected']]):
  c=dict(next(c for c in pool if c['id']==j['id']));c['screening_id']=c['id'];c['id']=f'D{ix+1}';c['selection_reason']=j;chosen.append(c)
 assert len(chosen)==3
 held=read(OUT/'heldout_reserved.json');allcases=chosen+held;assert len({c['group'] for c in allcases})==6
 for c in allcases:
  ep=ROOT/c['episode'];c['policy_source_hash']=sha(ep/'prompt.txt') if (ep/'prompt.txt').exists() else None;c['policy_semantics_read']=False
  c['history_demonstration_influence']='Historical controller may have used human rules; these are not expert-free demonstrations.'
 save(OUT/'data_manifest.json',{'at':utc(),'cases':allcases,'scope':'Original per-action observations; process files remain source-linked. Discovery outcome selection is raw-image review by experiment operator, not a Teacher conclusion. Heldout not selected by outcome.','heldout_outcome_decoded':False})
 freeze('selection',[OUT/'data_manifest.json',OUT/'screening_judgments.json'])
 for c in chosen:packet(c,True)

def packet(c,reveal):
 r=ROOT/c['round'];folder=OUT/'cases'/c['id'];folder.mkdir(parents=True,exist_ok=True)
 if not (folder/'before.png').exists():shutil.copyfile(r/'before/rgb.png',folder/'before.png')
 if not (folder/'pre.json').exists():
  pts=read(r/'points.json');geo=read(r/'geometry.json');pre={'id':c['id'],'image_size_px':[1280,720],'Pa':pts[0]['pixel_xy'],'Pb':pts[1]['pixel_xy'],'points':pts,'geometry':geo,'before_at':read(r/'before/observation.json')['captured_at'],'reliable_local_intention':'UNKNOWN: no trusted intent label supplied; old Planner reasons excluded.'}
  save(folder/'pre.json',pre);annotate(folder/'before.png',pre['Pa'],pre['Pb'],folder/'actual_action.png')
 if not reveal or (folder/'facts.json').exists():return
 pre=read(folder/'pre.json');shutil.copyfile(r/'after/rgb.png',folder/'after.png');feedback=[]
 for f in sorted(r.glob('gripper_*.json')):
  d=read(f);fb=d.get('feedback',{});s=fb.get('feedback',{});feedback.append({'command':d.get('action'),'command_result':fb.get('command_result'),'sample':{k:s.get(k) for k in ['sampled_at','state','position_pulse','mechanical_grasp_detected','error_code']}})
 native=read(r/'native_plan.json') if (r/'native_plan.json').exists() else []
 facts={**pre,'execution':read(r/'execution.json'),'after_at':read(r/'after/observation.json')['captured_at'],'gripper_feedback':feedback,'commanded_sequence':[{'kind':n['kind'],'data':{k:v for k,v in n['data'].items() if k in ['action','target_pose_mm_deg','cartesian_speed_mm_s','opening_delta_pulse','settle_s']}} for n in native],'process_warning':'Moving wrist camera. Sampled chronological frames cannot establish occluded contact or continuous layer motion. Feedback completion times are not exact onset times.'}
 fs=[];counts={}
 for phase,n in [('execution_images',10),('return_perception/execution_images',2)]:
  fp=r/phase/'frames.jsonl'
  if not fp.exists():counts[phase]=0;continue
  rr=[json.loads(l) for l in fp.read_text().splitlines() if l.strip()];counts[phase]=len(rr);idx={round(i*(len(rr)-1)/(n-1)) for i in range(n)}
  if phase=='execution_images':
   for g in feedback:
    s=g['sample'].get('sampled_at')
    if s:
     ts=datetime.fromisoformat(s).timestamp();idx.add(min(range(len(rr)),key=lambda j:abs(rr[j]['captured_unix_s']-ts)))
  for ix in sorted(idx):fs.append({**rr[ix],'source':str((r/phase/rr[ix]['file']).relative_to(ROOT)),'phase':phase})
 fs.sort(key=lambda f:f['captured_unix_s']);facts['available_process_counts']=counts;facts['sampled_process_frames']=[{k:v for k,v in f.items() if k!='source'} for f in fs]
 if fs:
  canvas=Image.new('RGB',(1280,math.ceil(len(fs)/4)*202),'#111');draw=ImageDraw.Draw(canvas)
  for j,f in enumerate(fs):
   x=j%4*320;y=j//4*202;src=ROOT/f['source'];canvas.paste(Image.open(src).convert('RGB').resize((320,180)),(x,y+22));draw.text((x+3,y+4),str(j+1)+' '+datetime.fromtimestamp(f['captured_unix_s'],timezone.utc).strftime('%H:%M:%S.%f')[:-3],fill='white')
   dest=folder/'raw_frames'/f'{j+1:02d}.jpg';dest.parent.mkdir(exist_ok=True);shutil.copyfile(src,dest)
  canvas.save(folder/'process.jpg',quality=94)
 else:facts['missing_process']='No process imagery; grasp/lift/pull/release mechanisms UNKNOWN.'
 save(folder/'facts.json',facts)
 paths=['before/rgb.png','after/rgb.png','before/observation.json','after/observation.json','points.json','geometry.json','execution.json','motion.py','states.jsonl','native_plan.json','execution_images/frames.jsonl','return_perception/execution_images/frames.jsonl']
 save(folder/'provenance.json',{'episode':c['episode'],'round':c['round'],'source_files':{n:sha(r/n) for n in paths if (r/n).exists()},'frames':[{**f,'sha256':sha(ROOT/f['source'])} for f in fs]})
 b=read(r/'before/observation.json')['captured_unix_s'];a=read(r/'after/observation.json')['captured_unix_s'];save(folder/'data_check.json',{'real':read(r/'before/observation.json').get('synthetic') is False,'all_frames_within_pre_post':all(b<f['captured_unix_s']<a for f in fs),'frame_timestamps_ordered':all(fs[i]['captured_unix_s']<fs[i+1]['captured_unix_s'] for i in range(len(fs)-1)),'before_after_s':a-b,'process_frames_sampled':len(fs),'nine_command_completion_is_not_acquisition_proof':True})
 freeze('packet_'+c['id'],[f for f in folder.rglob('*') if f.is_file()])

def cases(discovery):return [c for c in read(OUT/'data_manifest.json')['cases'] if c['id'].startswith('D' if discovery else 'H')]

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['scan','select']);a=p.parse_args();{'scan':scan,'select':select}[a.stage]()
