"""Resource-limited six-stage smoke test. No robot imports or execution paths."""
from pathlib import Path
import sys,json,time,hashlib,concurrent.futures,shutil,html
CODE=Path(__file__).resolve().parent;REPO=CODE.parents[1];FULL=REPO/'experiments/end_to_end_hindsight_v0'
sys.path.insert(1,str(FULL))
from transport import call,save,read,sha
from contracts import *
from cv_select import select,run_id
ROOT=REPO/'reports/hindsight_smoke_test'
INDEX=REPO/'reports/end_to_end_hindsight_v0_20261029/experience_index.json'

def progress(stage,**kw):
 x={'stage':stage,'time':time.time(),**kw};save(ROOT/'progress.json',x);print(json.dumps(x),flush=True)
def compact(r):return {'id':r['id'],'action':r['action'],'execution':r['execution']}
def rank(c):return (0 if c['effect']['classification']=='CONFIRMED_SEMANTIC_REVEAL' else 1,0 if c['comparison']['mismatch']=='UNPLANNED' else 1,{'HIGH':0,'MEDIUM':1,'LOW':2}[c['effect']['confidence']],c['id'])
def response(x,key,ids):
 if x['status']!='COMPLETE':return []
 rows=x['output'][key]
 if len(rows)!=len(ids) or {r['id'] for r in rows}!=set(ids):return []
 return rows

def report(index,selected,summary):
 byid={r['id']:r for r in index};effects={x['id']:x for x in read(ROOT/'observed_effects.json')};mismatch={x['id']:x for x in read(ROOT/'intention_comparisons.json')};calls=[read(p) for p in (ROOT/'calls').glob('*/result.json')]
 summary['api']={'attempts':len(list((ROOT/'calls').glob('*/attempt_*/started.json'))),'completed':sum(x['status']=='COMPLETE' for x in calls),'failed':sum(x['status']!='COMPLETE' for x in calls),'summed_latency_s':sum(x.get('latency_s',0) for x in calls),'usage':[x.get('usage') for x in calls]};summary['physical_actions']=0;summary['isolation']='Independent ephemeral model sessions; no old skill/discovery reports or prior full-run model outputs read. Original index/raw intentions allowed. Orchestrator retains chat context. Historical actions may contain expert-policy influence.'
 save(ROOT/'summary.json',summary)
 body='<h1>Hindsight Skill Discovery v0 — Smoke Test</h1><p>一次固定预算运行；CV只筛选，不是成功标签。模型视觉确认不是独立材料真值。未连接机器人。</p><pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre>'
 for z in selected:
  r=byid[z['id']];d=ROOT/'evidence'/r['id'];d.mkdir(parents=True,exist_ok=True)
  for k in ['before','after']:shutil.copyfile(r[k],d/(k+'.png'))
  body+='<h2>'+r['id']+' — '+z['selection_stratum']+'</h2><p>'+html.escape(r['source'])+'</p><div class="pair"><a href="evidence/'+r['id']+'/before.png"><img src="evidence/'+r['id']+'/before.png"></a><a href="evidence/'+r['id']+'/after.png"><img src="evidence/'+r['id']+'/after.png"></a></div><pre>'+html.escape(json.dumps({'cv_only':z,'original_intention':r['intent'],'observed_effect':effects.get(r['id'],'API/OUTPUT MISSING'),'intention_comparison':mismatch.get(r['id'],'API/OUTPUT MISSING')},ensure_ascii=False,indent=2))+'</pre>'
 for n,t in [('selected_cases.json','自动选择及独立审核'),('skill_hypotheses.json','技能假设'),('skill_spec.json','候选策略'),('validation_results.json','新RGB Dry-run')]:
  if (ROOT/n).exists():body+='<h2>'+t+'</h2><pre>'+html.escape(json.dumps(read(ROOT/n),ensure_ascii=False,indent=2))+'</pre>'
 for p in sorted((ROOT/'validation').glob('*/overlay.png')):body+='<a href="'+str(p.relative_to(ROOT))+'"><img width="100%" src="'+str(p.relative_to(ROOT))+'"></a>'
 body+='<p><a href="protocol.json">冻结协议</a> · <a href="cv_metrics.json">全量CV筛选</a> · <a href="source_manifest.json">来源与隔离审计</a></p>'
 (ROOT/'review.html').write_text('<!doctype html><meta charset="utf-8"><title>Hindsight Smoke Test</title><style>body{max-width:1400px;margin:30px auto;font:18px sans-serif}pre{white-space:pre-wrap;background:#eee;padding:15px}.pair{display:flex}.pair a{width:50%}.pair img{width:100%}</style>'+body)
 (ROOT/'report.md').write_text('# Smoke Test\n\n'+json.dumps(summary,ensure_ascii=False,indent=2))

def main():
 ROOT.mkdir(parents=True,exist_ok=True)
 protocol={'algorithm':'same six stages, resource-limited smoke v0','index_sha256':sha(INDEX),'cv_selection':{'seed':'smoke-v0-20261029','mask':'saturation>35 brightness>80 central 70%width; largest component; no semantic labels','registration':'ORB+RANSAC+rigid refit; centroid fallback flagged','ranking':'aligned contour symmetric difference 1-IoU','strata':'4 high / 3 medium / 3 low deformation; 2 seeded random controls','diversity':'different run for each pair; prefer unused orientation/aspect/solidity buckets','heldout':'4 different runs by seeded hash, locked BEFORE CV; only Before sent to dry-run; their CV/After never sent to learner'},'limits':{'visual_pairs':12,'deep_cases':2,'induction_rounds':1,'skills':1,'validation_images':4,'model_calls':20,'workers':2,'physical_actions':0},'candidate_ranking':'confirmed reveal > suspected; unplanned > ambiguous; confidence; content hash; independent review top2','no_discovery_gate':'No independently reviewed, semantically identified, unplanned reveal => NO_VALID_DISCOVERY; no hypothesis/action fabrication','files':{str(p):sha(p) for p in [*CODE.glob('*.py'),FULL/'contracts.py',FULL/'policy.py'] if not p.name.startswith('test_')}}
 if (ROOT/'protocol.json').exists():
  if read(ROOT/'protocol.json')!=protocol:raise RuntimeError('Frozen protocol changed')
 else:save(ROOT/'protocol.json',protocol)
 index=read(INDEX);save(ROOT/'experience_index.json',index);progress('1_CV_SCREEN',records=len(index));selected,val=select(ROOT,index);byid={r['id']:r for r in index};rows=[byid[z['id']] for z in selected]
 # No read of previous effect_candidates, reports, skill text or manual case lists.
 save(ROOT/'source_manifest.json',{'index':str(INDEX),'index_sha256':sha(INDEX),'model_deep_images':{r['id']:{k:{'path':r[k],'sha256':sha(r[k])} for k in ['before','after']} for r in rows},'validation_only_before':val,'prior_full_run_outputs_read':False,'selected_run_overlap_validation':sorted({run_id(r) for r in rows}&{r['run'] for r in val}),'intent_source':'original raw decision fields from provenance index','CV_all_pairs_read':True,'heldout_CV_results_used_for_selection_or_learning':False})
 assert not ({run_id(r) for r in rows}&{r['run'] for r in val});assert len(rows)<=12
 progress('2_EFFECT_DISCOVERY',pairs=len(rows));effects=[]
 batches=[rows[i:i+2] for i in range(0,len(rows),2)]
 def observe(item):
  j,rs=item;x=call(ROOT,f'observe_{j:02d}',OBS_PROMPT,{'image_order':[r['id']+' '+p for r in rs for p in ['Before','After']],'records':[compact(r) for r in rs]},[Path(r[k]) for r in rs for k in ['before','after']],EFFECT_SCHEMA);return response(x,'effects',[r['id'] for r in rs])
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  for batch in pool.map(observe,enumerate(batches)):
   effects+=batch;save(ROOT/'observed_effects.json',effects);progress('2_EFFECT_DISCOVERY',completed=len(effects),total=len(rows))
 if not (ROOT/'observed_effects.json').exists():save(ROOT/'observed_effects.json',effects)
 ef={x['id']:x for x in effects};comp=[]
 if effects:
  x=call(ROOT,'intention_mismatch',MISMATCH_PROMPT,{'records':[{'id':r['id'],'original_intention':r['intent'],'observed_effect':ef[r['id']]} for r in rows if r['id'] in ef]},[],MISMATCH_SCHEMA);comp=response(x,'comparisons',list(ef))
 save(ROOT/'intention_comparisons.json',comp);cm={x['id']:x for x in comp}
 candidates=[{'id':e['id'],'effect':e,'comparison':cm[e['id']],'goal':'Reveal('+e['endpoint']+')','confirmed_physical_success':False} for e in effects if e['classification'] in ['CONFIRMED_SEMANTIC_REVEAL','SUSPECTED_REVEAL'] and e['id'] in cm]
 events=[c for c in candidates if c['comparison']['mismatch'] in ['UNPLANNED','AMBIGUOUS']]
 save(ROOT/'effect_candidates.json',candidates);save(ROOT/'hindsight_events.json',events);progress('3_HINDSIGHT_RELABEL',candidates=len(events))
 chosen=sorted(events,key=rank)[:2];reviews=[]
 if chosen:
  rs=[byid[c['id']] for c in chosen];x=call(ROOT,'independent_review',REVIEW_PROMPT,{'image_order':[r['id']+' '+p for r in rs for p in ['Before','After']],'records':[{**compact(r),'original_intention':r['intent']} for r in rs]},[Path(r[k]) for r in rs for k in ['before','after']],REVIEW_SCHEMA);reviews=response(x,'reviews',[r['id'] for r in rs])
 valid={r['id'] for r in reviews if r['reveal_verified'] and r['unexpected_verified'] and r['endpoint'] not in ['NONE','SEMANTIC_SIDE_UNKNOWN']};chosen=[c for c in chosen if c['id'] in valid]
 save(ROOT/'selected_cases.json',{'ranking_frozen':protocol['candidate_ranking'],'independent_reviews':reviews,'selected':chosen,'manual_case_selection':False})
 summary={'historical_records_read':len(index),'cv_measureable':sum(x['cv_status']=='OK' for x in read(ROOT/'cv_metrics.json')),'deep_pairs_selected':len(rows),'deep_pairs_analyzed':len(effects),'beneficial_candidates':len(candidates),'unplanned_candidates':sum(c['comparison']['mismatch']=='UNPLANNED' for c in candidates),'reviewed_confirmed_unplanned_cases':len(chosen),'selected_ids':[c['id'] for c in chosen],'stages':{'1':'COMPLETED','2':'COMPLETED' if len(effects)==len(rows) and len(comp)==len(rows) else 'INCOMPLETE_API_OR_SCHEMA','3':'COMPLETED'}}
 if not chosen:
  summary.update(status='NO_VALID_DISCOVERY' if summary['stages']['2']=='COMPLETED' else 'INCOMPLETE_ANALYSIS_NO_VALID_DISCOVERY');summary['stages'].update({'4':'NOT_TRIGGERED: no independently reviewed unplanned semantic reveal','5':'NOT_TRIGGERED','6':'NOT_TRIGGERED'})
  save(ROOT/'skill_hypotheses.json',{'status':'NOT_TRIGGERED','reason':'No confirmed unexpected semantic reveal under frozen gate'});save(ROOT/'validation_results.json',{'status':'NOT_TRIGGERED','physical_actions':0});report(index,selected,summary);progress('COMPLETE',status=summary['status']);return
 progress('4_HYPOTHESIS_INDUCTION',cases=len(chosen));rs=[byid[c['id']] for c in chosen]
 x=call(ROOT,'hypothesis_induction',HYP_PROMPT,{'cases':[{**compact(r),'original_intention':r['intent'],'observed_effect':ef[r['id']]} for r in rs],'independent_review':reviews,'image_order':[r['id']+' '+p for r in rs for p in ['Before','After']]},[Path(r[k]) for r in rs for k in ['before','after']],HYP_SCHEMA)
 hyp=x.get('output',{'status':'API_FAILED','reason':x.get('error')});save(ROOT/'skill_hypotheses.json',hyp);summary['stages']['4']=hyp['status']
 if hyp.get('status')=='HYPOTHESES':
  x=call(ROOT,'skill_synthesis',SYNTH_PROMPT,{'hypotheses':hyp,'interface':'generate_action(rgb,target_corner); current image only; one Pa/Pb proposal, direction implied by displacement; no physical execution; no access to historical After'},[],SKILL_SCHEMA);skill=x.get('output',{'status':'API_FAILED','reason':x.get('error')})
 else:skill={'status':'NOT_YET_EXECUTABLE','reason':'Hypothesis induction did not establish candidate knowledge'}
 save(ROOT/'skill_spec.json',skill);summary['stages']['5']=skill['status'];validation=[]
 if skill.get('status')=='CANDIDATE':
  code='''from pathlib import Path\nimport sys\nsys.path.insert(0,SMOKE_DIR)\nsys.path.insert(1,FULL_DIR)\nfrom transport import call,read,sha\nfrom contracts import ACTION_PROMPT,ACTION_SCHEMA\nfrom PIL import Image\nARTIFACT=Path(__file__).resolve().parent\ndef generate_action(rgb,target_corner):\n    rgb=Path(rgb)\n    payload={'image_order':['CURRENT unseen Before only'],'target_corner':target_corner,'candidate_strategy':read(ARTIFACT/'skill_spec.json'),'image_size':Image.open(rgb).size}\n    r=call(ARTIFACT,'dry_'+sha(rgb)[:14],ACTION_PROMPT,payload,[rgb],ACTION_SCHEMA)\n    return r.get('output',{'decision':'ABSTAIN','abstain_reason':'INSUFFICIENT_OBSERVATION','error':r.get('error')})\n'''.replace('SMOKE_DIR',repr(str(CODE))).replace('FULL_DIR',repr(str(FULL)))
  (ROOT/'candidate_skill.py').write_text(code);compile(code,'candidate_skill.py','exec');save(ROOT/'skill_freeze.json',{'time':time.time(),'code_sha256':sha(ROOT/'candidate_skill.py'),'spec_sha256':sha(ROOT/'skill_spec.json')})
  # Use same global call root so the 20-call budget covers all stages.
  def dry(v):
   import importlib.util
   sp=importlib.util.spec_from_file_location('frozen_candidate_'+v['id'],ROOT/'candidate_skill.py');candidate=importlib.util.module_from_spec(sp);sp.loader.exec_module(candidate)
   candidate.generate_action(v['before'],'AUTO_SELECT_CURRENT_HIDDEN_ENDPOINT')
   x=read(ROOT/'calls'/('dry_'+sha(v['before'])[:14])/'result.json')
   a=x.get('output');result={'id':v['id'],'run':v['run'],'before':v['before'],'after_supplied':False,'historical_action_supplied':False,'result':x['status'],'action':a,'physical_success':'NOT_TESTED'}
   if a:
    from PIL import Image,ImageDraw
    import numpy as np
    im=Image.open(v['before']).convert('RGB');w,h=im.size;g=a['grasp_px'];t=a['target_px'];validpx=all(0<=p[0]<w and 0<=p[1]<h for p in [g,t]);result['format_and_pixels_valid']=bool(validpx);result['robot_feasibility']='NOT_CHECKED_NO_CALIBRATED_DRY_RUN';result['distance_px']=float(np.linalg.norm(np.asarray(t)-g))
    if a['decision']=='ACTION' and validpx:
     d=ImageDraw.Draw(im);d.line([tuple(g),tuple(t)],fill='yellow',width=4)
     for p,c in [(g,'lime'),(t,'red')]:d.ellipse((p[0]-6,p[1]-6,p[0]+6,p[1]+6),fill=c)
    out=ROOT/'validation'/v['id'];out.mkdir(parents=True,exist_ok=True);im.save(out/'overlay.png')
   return result
  progress('6_DRY_RUN',images=len(val))
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:validation=list(pool.map(dry,val))
  summary['stages']['6']='DRY_RUN_COMPLETED';summary['status']='SMOKE_COMPLETED_CANDIDATE_NOT_PHYSICALLY_VALIDATED'
 else:summary['stages']['6']='NOT_TRIGGERED_NO_EXECUTABLE_SKILL';summary['status']='HYPOTHESIS_ONLY'
 save(ROOT/'validation_results.json',validation);report(index,selected,summary);progress('COMPLETE',status=summary['status'])
if __name__=='__main__':main()
