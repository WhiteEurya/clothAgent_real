"""Frozen six-stage End-to-End Hindsight Discovery v0. Resumable raw-data run."""
from pathlib import Path
import argparse,json,hashlib,concurrent.futures,traceback,time,subprocess,sys,html,shutil
from transport import read,save,sha,call
from contracts import *
from mining import scan
CODE=Path(__file__).resolve().parent;REPO=CODE.parents[1]
VERSION='end_to_end_hindsight_v0.0.1'

def protocol(root):
 spec={'version':VERSION,'stages':['Experience Mining','Effect Discovery','Hindsight Goal Relabeling','Skill Hypothesis Induction','Executable Skill Synthesis','Active Validation and Refinement'],'observer_batch':8,'mismatch_batch':32,'workers':2,'review_limit':12,'selected_limit':2,'counterexamples_per_case':2,'ranking':['confirmed before suspected','unplanned before ambiguous before planned','confidence HIGH before MEDIUM before LOW','stable content hash ascending'],'selection':'top reviewed candidates from distinct episodes; no manually specified IDs','llm':{'profile':'api','model':'gpt-6-astra','effort':'high','timeout_s':600,'attempts':2},'independent_observer':'no original intent; mismatch text-only afterward','validation_observer':'no prediction or requested outcome; compare afterward','isolation_limit':'orchestrator has historical conversation; discovery model sessions receive only frozen generic prompts and allowlisted raw inputs. Same provider/model prior knowledge cannot be removed.','evidence_limit':'model-confirmed is not independently measured material ground truth','online_limit':1,'code_sha256':{p.name:sha(p) for p in CODE.glob('*.py') if not p.name.startswith('test_')}}
 f=root/'protocol.json'
 if f.exists():
  old=read(f)
  if old!=spec:raise RuntimeError('Frozen protocol/code changed; use a new experiment root')
 else:save(f,spec)
 return spec

def progress(root,stage,**kw):
 x={'stage':stage,'updated_unix':time.time(),**kw};save(root/'progress.json',x);print(json.dumps(x),flush=True)

def compact(r,intent=False):
 x={'id':r['id'],'action':r['action'],'execution':r['execution'],'partial_execution':r['is_partial_execution']}
 if intent:x['original_intention']=r['intent']
 return x

def stage2(root,records,cfg):
 effects={};errors=[];batches=[records[i:i+cfg['observer_batch']] for i in range(0,len(records),cfg['observer_batch'])]
 def task(item):
  j,rows=item;payload={'image_order':[f'{r["id"]} {phase}' for r in rows for phase in ['Before','After']],'records':[compact(r) for r in rows]}
  x=call(root,f'observe_{j:04d}',OBS_PROMPT,payload,[Path(r[k]) for r in rows for k in ['before','after']],EFFECT_SCHEMA)
  return rows,x
 with concurrent.futures.ThreadPoolExecutor(max_workers=cfg['workers']) as pool:
  for rows,result in pool.map(task,enumerate(batches)):
   if result['status']=='COMPLETE':
    out=result['output']['effects'];ids=[r['id'] for r in out]
    if len(set(ids))==len(ids) and set(ids)=={r['id'] for r in rows}:effects.update({r['id']:r for r in out})
    else:errors.append({'ids':[r['id'] for r in rows],'error':'ID coverage mismatch'})
   else:errors.append({'ids':[r['id'] for r in rows],'error':result.get('error')})
   save(root/'observed_effects.json',list(effects.values()));save(root/'stage2_errors.json',errors);progress(root,'2_effect_observation',completed=len(effects),total=len(records),failed_batches=len(errors))
 mismatches={};valid=[r for r in records if r['id'] in effects];batches=[valid[i:i+32] for i in range(0,len(valid),32)]
 def compare(item):
  j,rows=item;x=call(root,f'mismatch_{j:04d}',MISMATCH_PROMPT,{'records':[{'id':r['id'],'original_intention':r['intent'],'observed_effect':effects[r['id']]} for r in rows]},[],MISMATCH_SCHEMA);return rows,x
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  for rows,x in pool.map(compare,enumerate(batches)):
   if x['status']=='COMPLETE':
    out=x['output']['comparisons']
    if {z['id'] for z in out}=={z['id'] for z in rows}:mismatches.update({z['id']:z for z in out})
   progress(root,'2_intention_mismatch',completed=len(mismatches),total=len(records))
 save(root/'intention_comparisons.json',list(mismatches.values()))
 candidates=[]
 for r in valid:
  e=effects[r['id']];m=mismatches.get(r['id'])
  if e['classification'] in ['CONFIRMED_SEMANTIC_REVEAL','SUSPECTED_REVEAL']:
   candidates.append({'id':r['id'],'episode':r['episode'],'effect':e,'comparison':m,'relabel_goal':'Reveal('+e['endpoint']+')','status':'CANDIDATE_MODEL_LABEL_NOT_PHYSICAL_GT'})
 save(root/'effect_candidates.json',candidates)
 events=[c for c in candidates if c['comparison'] and c['comparison']['mismatch'] in ['UNPLANNED','AMBIGUOUS']]
 save(root/'hindsight_events.json',events);return effects,mismatches,events

def rank(c):
 return (0 if c['effect']['classification']=='CONFIRMED_SEMANTIC_REVEAL' else 1,{'UNPLANNED':0,'AMBIGUOUS':1,'PLANNED':2,'NO_BENEFIT':3}.get((c.get('comparison') or {}).get('mismatch'),4),{'HIGH':0,'MEDIUM':1,'LOW':2}[c['effect']['confidence']],c['id'])

def stage4(root,records,effects,events,cfg):
 byid={r['id']:r for r in records};ranked=sorted(events,key=rank);save(root/'ranked_candidates.json',ranked)
 reviews=[]
 for c in ranked[:cfg['review_limit']]:
  r=byid[c['id']];images=[Path(r['before']),Path(r['after'])];payload={'image_order':['Before','After'],'record':compact(r,True)}
  x=call(root,'review_'+r['id'],REVIEW_PROMPT,payload,images,REVIEW_SCHEMA)
  if x['status']=='COMPLETE':reviews.extend(z for z in x['output']['reviews'] if z['id']==r['id'])
  save(root/'independent_reviews.json',reviews);progress(root,'4_independent_review',completed=len(reviews),total=min(len(ranked),cfg['review_limit']))
 good={r['id'] for r in reviews if r['reveal_verified'] and r['unexpected_verified'] and r['endpoint'] not in ['NONE','SEMANTIC_SIDE_UNKNOWN']}
 choices=[c for c in ranked if c['id'] in good];kind='REVIEWED_MODEL_CONFIRMED'
 if not choices:choices=ranked[:2];kind='UNCERTAIN_EXPLORATORY_ONLY'
 selected=[];episodes=set()
 for c in choices:
  if c['episode'] in episodes:continue
  selected.append(c);episodes.add(c['episode'])
  if len(selected)>=cfg['selected_limit']:break
 selection={'rule':cfg['ranking'],'selected':selected,'evidence_status':kind,'not_manually_selected':True}
 save(root/'selected_cases.json',selection)
 if not selected:
  out={'status':'DATA_INSUFFICIENT','hypotheses':[],'reason':'No unplanned/ambiguous reveal candidate under frozen protocol. Transport missingness is reported separately.','intention_actual_effect_analysis':'No selected cases'};save(root/'skill_hypotheses.json',out);return out
 # Fixed nearest-Before histogram retrieval of ordinary/no-reveal contrast cases,
 # not a directional or skill-based selection. Distinct episodes only.
 import cv2,numpy as np
 def descriptor(r):
  im=cv2.imread(r['before']);h=cv2.calcHist([cv2.cvtColor(im,cv2.COLOR_BGR2HSV)],[0,1],None,[16,16],[0,180,0,256]);return cv2.normalize(h,h).flatten()
 pool=[r for r in records if r['id'] in effects and effects[r['id']]['classification'] in ['NO_REVEAL','ORDINARY_DEFORMATION'] and r['episode'] not in episodes]
 desc={r['id']:descriptor(r) for r in pool};contrast=[]
 for c in selected:
  d=descriptor(byid[c['id']]);near=sorted(pool,key=lambda r:(float(np.linalg.norm(d-desc[r['id']])),r['id']))[:cfg['counterexamples_per_case']]
  contrast.extend(near)
 chosen=[byid[c['id']] for c in selected];chosen+=list({r['id']:r for r in contrast}.values());save(root/'contrast_cases.json',[r['id'] for r in contrast])
 payload={'selected_ids':[c['id'] for c in selected],'evidence_status':kind,'image_order':[f'{r["id"]} {p}' for r in chosen for p in ['Before','After']],'records':[{**compact(r,True),'observed_effect':effects[r['id']]} for r in chosen]}
 x=call(root,'hypothesis_induction',HYP_PROMPT,payload,[Path(r[k]) for r in chosen for k in ['before','after']],HYP_SCHEMA)
 out=x['output'] if x['status']=='COMPLETE' else {'status':'API_FAILED','hypotheses':[],'reason':x.get('error')};save(root/'skill_hypotheses.json',out);return out

def stage5(root,hyp):
 if hyp.get('status')!='HYPOTHESES' or len(hyp.get('hypotheses',[]))<2:
  skill={'status':'NOT_YET_EXECUTABLE','goal':'Corner Reveal','procedure':[],'observable_prerequisites':[],'abstention_conditions':['No sufficient competing testable hypotheses'],'competing_hypotheses':[],'reason':hyp.get('reason','Insufficient hypotheses')}
 else:
  x=call(root,'skill_synthesis',SYNTH_PROMPT,{'hypotheses':hyp,'interface':'generate_action(rgb,target_corner); one lift_pull_place Pa/Pb image pixels; 10..150mm XY host checked; gripper angle image degrees; nominal lift20mm; no robot access by discovery model'},[],SKILL_SCHEMA)
  skill=x['output'] if x['status']=='COMPLETE' else {'status':'NOT_YET_EXECUTABLE','procedure':[],'reason':x.get('error')}
 save(root/'skill_spec.json',skill)
 code='''"""Frozen candidate; no historical coordinates, IDs or After input."""\nfrom pathlib import Path\nimport sys\nsys.path.insert(0, CODE_DIR)\nfrom policy import generate\nARTIFACT=Path(__file__).resolve().parent\ndef generate_action(rgb,target_corner,request_dir):\n    return generate(rgb,target_corner,ARTIFACT,request_dir)\n'''.replace('CODE_DIR',repr(str(CODE)))
 (root/'candidate_skill.py').write_text(code);compile(code,'candidate_skill.py','exec')
 save(root/'skill_freeze.json',{'skill_spec_sha256':sha(root/'skill_spec.json'),'candidate_code_sha256':sha(root/'candidate_skill.py'),'generator_sha256':sha(CODE/'policy.py'),'time':time.time(),'validated_skill':False});return skill

def stage6(root,skill,args):
 # Default unattended/offline run prepares a deterministic held-out input contract.
 # Robot dispatch requires current onsite permission, never inherited from hours-old presence.
 record={'status':'PENDING','physical_actions':0,'maximum_actions':1,'reason':'New independent garment state and current existing onsite execution permission required after long offline run. No robot connection in offline default.','candidate_skill':str(root/'candidate_skill.py'),'expected_inputs':['fresh RGB-D','target semantic endpoint','existing host configuration and live stationary/scene checks'],'unchanged_host_required':True,'observer_must_not_read_prediction':True,'next_command':f'PYTHONPATH=. /home/sja/miniconda3/envs/cali/bin/python {CODE}/validate.py --root {root} --prepare-live'}
 if skill.get('status')!='CANDIDATE':record.update(status='PENDING_NO_EXECUTABLE_SKILL',reason=skill.get('reason'))
 save(root/'validation_record.json',record)
 if args.prepare_live and skill.get('status')=='CANDIDATE':subprocess.run([sys.executable,str(CODE/'validate.py'),'--root',str(root),'--prepare-live'],check=True)
 return record

def report(root,records):
 def optional(p,default):return read(root/p) if (root/p).exists() else default
 effects=optional('observed_effects.json',[]);cand=optional('effect_candidates.json',[]);events=optional('hindsight_events.json',[]);sel=optional('selected_cases.json',{});hyp=optional('skill_hypotheses.json',{});skill=optional('skill_spec.json',{});val=optional('validation_record.json',{})
 from collections import Counter
 summary={'experiences':len(records),'independent_episodes':len({r['episode'] for r in records}),'analyzed_experiences':len(effects),'effect_counts':dict(Counter(r['classification'] for r in effects)),'beneficial_candidates':len(cand),'unplanned_candidates':sum((c.get('comparison') or {}).get('mismatch')=='UNPLANNED' for c in cand),'ambiguous_candidates':sum((c.get('comparison') or {}).get('mismatch')=='AMBIGUOUS' for c in cand),'selected':sel,'hypotheses':hyp,'candidate_skill':skill,'validation':val,'contamination':'Discovery sessions did not receive previous discovery reports or expert prompts. Orchestrator knew prior conversation. Raw historical decisions/trajectories may themselves reflect expert influence; this is not expert-free data.','conclusion_scope':'Automated visual evidence and exploratory hypothesis generation; no unexecuted action is success.'}
 calls=[read(p) for p in (root/'calls').glob('*/result.json')];summary['calls']={'completed':sum(x['status']=='COMPLETE' for x in calls),'failed':sum(x['status']!='COMPLETE' for x in calls),'summed_latency_s':sum(x.get('latency_s',0) for x in calls)};save(root/'summary.json',summary)
 body='<h1>End-to-End Hindsight-Driven Skill Discovery v0</h1><p>固定六阶段，原始记录全量审计，独立效果观察与意图比较。模型确认不等于独立物理真值。</p><pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre>'
 byid={r['id']:r for r in records}
 for c in sel.get('selected',[]):
  r=byid[c['id']];d=root/'evidence'/r['id'];d.mkdir(parents=True,exist_ok=True)
  for k in ['before','after']:
   if not (d/(k+'.png')).exists():shutil.copyfile(r[k],d/(k+'.png'))
  body+='<h2>'+r['id']+'</h2><p>'+html.escape(r['source'])+'</p><div class="pair"><img src="evidence/'+r['id']+'/before.png"><img src="evidence/'+r['id']+'/after.png"></div><pre>'+html.escape(json.dumps({'original_intention':r['intent'],'action':r['action'],'discovery':c},ensure_ascii=False,indent=2))+'</pre>'
 body+='<h2>候选程序</h2><pre>'+html.escape((root/'candidate_skill.py').read_text() if (root/'candidate_skill.py').exists() else '')+'</pre><p><a href="experience_index.json">全量经验索引</a> · <a href="source_audit.json">排除与去重审计</a> · <a href="protocol.json">冻结协议</a> · <a href="skill_hypotheses.json">假设</a> · <a href="validation_record.json">验证状态</a></p>'
 (root/'review.html').write_text('<!doctype html><meta charset="utf-8"><title>End-to-End Hindsight v0</title><style>body{max-width:1400px;margin:30px auto;font:18px sans-serif}pre{white-space:pre-wrap;background:#eee;padding:20px}.pair{display:flex}.pair img{width:50%}</style>'+body)
 (root/'report.md').write_text('# End-to-End Hindsight v0\n\n'+json.dumps(summary,ensure_ascii=False,indent=2))

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prepare-live',action='store_true');args=p.parse_args();root=args.root.resolve();root.mkdir(parents=True,exist_ok=True)
 cfg=protocol(root);progress(root,'1_mining');records=scan(REPO,root);progress(root,'1_mining_complete',experiences=len(records));effects,mismatches,events=stage2(root,records,cfg);progress(root,'3_relabel',events=len(events));hyp=stage4(root,records,effects,events,cfg);progress(root,'5_synthesis');skill=stage5(root,hyp);stage6(root,skill,args);report(root,records);progress(root,'COMPLETE_OFFLINE',experiences=len(records),analyzed=len(effects),validation=read(root/'validation_record.json')['status'])
if __name__=='__main__':main()
