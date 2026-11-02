"""Incremental, raw-record Intention–Outcome Mismatch Driven Skill Discovery v0."""
from pathlib import Path
import argparse,json,hashlib,time,concurrent.futures,shutil,sys
from transport import call,read,save,sha
from contracts import *
from policy import checked_compile,invoke,overlay
CODE=Path(__file__).resolve().parent
REPO=CODE.parents[1]
MD=REPO/'docs/experiments/intention_outcome_mismatch_driven_skill_discovery_20261030.md'
DEFAULT_SOURCE=REPO/'reports/hindsight_corner_continuation_20261030/input_manifest.json'
SEED='mismatch-v0-20261030'
VERSION='iom-v0.1'

def key(x):return hashlib.sha256((SEED+'|'+x).encode()).hexdigest()
def progress(root,stage,**kw):
    v={'stage':stage,'time':time.time(),**kw};save(root/'progress.json',v);print(json.dumps(v),flush=True)
def chunks(a,n=4):return [a[i:i+n] for i in range(0,len(a),n)]
def read_if(p,default):return read(p) if p.exists() else default

def collect(args):
    audit=[];rows=[]
    if args.episode:
        sources=[{'source':str(p.parent.relative_to(REPO))} for p in sorted(args.episode.resolve().rglob('native_plan.json'))]
    else:sources=read(args.source_manifest)['mapping']
    seen=set()
    for old in sources:
        source=REPO/old['source']
        try:
            execution=read(source/'execution.json');native=read(source/'native_plan.json');dispatch=read(source/'dispatch_intent.json')
            if execution.get('physical_execution') is not True:raise ValueError('not physical execution')
            if not any(o.get('kind')=='gripper' and o.get('data',{}).get('action')=='close' for o in native):raise ValueError('no manipulation in native plan')
            before=REPO/old['before'] if 'before' in old else source/'before/rgb.png'
            after=REPO/old['after'] if 'after' in old else source/'after/rgb.png'
            from PIL import Image
            for p in [before,after]:
                with Image.open(p) as im:im.verify()
            pair=sha(before)+sha(after)+sha(source/'native_plan.json')
            if pair in seen:continue
            seen.add(pair)
            plan=dispatch.get('plan',{})
            fields=['short_reason','reason','purpose','action_purpose','expected_effect','expected_outcome','expected_stop_state','expected_stop_condition','target_region']
            intention={k:plan[k] for k in fields if plan.get(k) is not None}
            action=old.get('action') or {k:plan.get(k) for k in ['grasp_px','target_px','motion_type','grasp_angle_deg']}
            episode=next((p for p in [source,*source.parents] if p.name.startswith('episode_')),source.parent)
            rawpaths=[before,after,source/'dispatch_intent.json',source/'execution.json',source/'native_plan.json']
            rows.append({'source':str(source.relative_to(REPO)),'episode':str(episode.relative_to(REPO)),'run':old.get('run',str(episode.parent.relative_to(REPO))),'round':source.name,'before':str(before),'after':str(after),'before_sha256':sha(before),'after_sha256':sha(after),'raw_intention':intention or {'status':'UNKNOWN'},'action':action,'execution':execution,'raw_hashes':{str(p):sha(p) for p in rawpaths},'experience_hash':key(pair),'intention_provenance':'Original dispatch_intent.json; producer pre-dispatch artifact, no explicit historical timestamp in this record','observation_hash_in_dispatch':dispatch.get('observation_sha256'),'source_mtime':(source/'dispatch_intent.json').stat().st_mtime})
        except Exception as e:audit.append({'source':str(source),'error':str(e)})
    rows.sort(key=lambda x:x['experience_hash'])
    for i,r in enumerate(rows):r['id']=f'M{i+1:02d}'
    return rows,audit

def tests_before_only(rows):
    base=REPO/'reports/hindsight_fast_test_20261029'
    split=read(base/'split.json');index=read(base/'experience_index.json')
    previous=read(DEFAULT_SOURCE)['validation_before_only']
    used={r['before_sha256'] for r in split['validation']+previous};tests=[]
    for run in split['heldout_complete_runs']:
        choices=[r for r in index if r['run']==run and r['before_sha256'] not in used]
        if choices:
            r=min(choices,key=lambda x:key(x['before_sha256']))
            tests.append({'id':f'T{len(tests)+1:02d}','run':run,'before':str(REPO/r['before']),'before_sha256':r['before_sha256']})
    assert not ({r['run'] for r in rows}&{r['run'] for r in tests})
    assert not ({r[k+'_sha256'] for r in rows for k in ['before','after']}&{r['before_sha256'] for r in tests})
    return tests

def event_eligible(corner,comparison):
    return (corner['goal']==comparison['goal'] and corner['classification'] in ['CONFIRMED','PROVISIONAL']
            and corner['before_state'] in ['OCCLUDED','SUSPECTED_OCCLUDED']
            and corner['after_state'] in ['EXPOSED','PARTIALLY_EXPOSED'] and corner['evidence_score']>=2
            and comparison['classification']=='UNEXPECTED_OBSERVED'
            and comparison['unexpected_evidence'] in ['SUPPORTED','PROVISIONAL']
            and comparison['benefit_related_to_goal'] and not comparison['absence_of_mention_only'])
def rank(event):
    c,m=event['observed'],event['comparison']
    return (0 if c['classification']=='CONFIRMED' else 1,0 if m['unexpected_evidence']=='SUPPORTED' else 1,-c['evidence_score'],key(event['id']+event['goal']))
def call_required(root,name,prompt,payload,images,schema):
    x=call(root,name,prompt,payload,[Path(p) for p in images],schema)
    if x['status']!='COMPLETE':raise RuntimeError(name+': '+str(x.get('error')))
    return x['output']
def check_ids(out,ids):
    assert len(out)==len(ids) and {x['id'] for x in out}==set(ids)
def check_corners(corners):
    assert len(corners)==4 and {x['goal'] for x in corners}==set(GOALS)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=REPO/'reports/intention_outcome_skill_discovery_v0_20261030')
    p.add_argument('--source-manifest',type=Path,default=DEFAULT_SOURCE)
    p.add_argument('--episode',type=Path,help='Process one new raw episode only; no full-history scan')
    args=p.parse_args();root=args.root.resolve();root.mkdir(parents=True,exist_ok=True)
    rows,audit=collect(args)
    if not rows:raise RuntimeError('No verifiable raw experience')
    if len(rows)>20:raise RuntimeError('This bounded iteration accepts at most 20 raw transitions; split into complete episode batches')
    tests=tests_before_only(rows)
    protocol={'version':VERSION,'seed':SEED,'specification':str(MD),'md_sha256':sha(MD),'code_hashes':{p.name:sha(p) for p in CODE.glob('*.py') if not p.name.startswith('test_')},
              'source':'one episode' if args.episode else 'existing17 queue RAW sources only; no prior model outputs',
              'sample_hashes':[r['experience_hash'] for r in rows],'stages':['Experience Collection','Intention–Outcome Comparison','Unexpected Benefit Discovery','Hindsight Goal Relabeling','Skill Hypothesis Induction','Active Validation & Skill Refinement'],
              'limits':{'workers':2,'intention_batch':4,'comparison_batch':4,'deep_cases':1,'induction_rounds':1,'skills':1,'offline_smoke_images':3,'model_call_cap':40,'maximum_physical_actions':1},
              'selection':'eligible unexpected Corner Reveal only; confirmed before provisional; supported mismatch before provisional; evidence score; seeded hash. INDETERMINATE or absence-of-mention-only never enter.',
              'independence':'intention: raw text+Before+Pa only, no After; observer: Before/After only, no intention or action; comparison: frozen outputs only; skill learners: selected current-round evidence only',
              'scope_limit':'known-page bootstrap is selected data, not unbiased discovery rate. Static skill smoke tests cannot replace active physical validation. Orchestrator has prior conversation; model contexts do not.'}
    if (root/'protocol.json').exists():assert read(root/'protocol.json')==protocol
    else:save(root/'protocol.json',protocol)
    save(root/'experience_index.json',rows);save(root/'source_audit.json',audit)
    save(root/'validation_inputs_before_only.json',tests)
    shutil.copyfile(MD,root/'specification.md')
    progress(root,'1_COLLECTION_COMPLETE',experiences=len(rows),excluded=len(audit))
    intentions={};observations={};failures=[]
    jobs=[]
    for i,batch in enumerate(chunks(rows)):
        jobs.append(('intent',i,batch))
    for i,r in enumerate(rows):jobs.append(('observe',i,[r]))
    def task(job):
        kind,i,batch=job
        if kind=='intent':
            payload={'image_order':[r['id']+' Before' for r in batch],'records':[{'id':r['id'],'original_pre_action_text':r['raw_intention'],'proposed_grasp_px':r['action'].get('grasp_px')} for r in batch]}
            result=call(root,f'intent_{i:02d}',INT_PROMPT,payload,[Path(r['before']) for r in batch],INT_SCHEMA)
        else:
            r=batch[0];result=call(root,f'observe_{r["id"]}',OBS_PROMPT,{'id':r['id'],'image_order':['Before','After'],'physical_action_record_exists':True},[Path(r['before']),Path(r['after'])],OBS_SCHEMA)
        return job,result
    progress(root,'2_INDEPENDENT_INTENTION_AND_OBSERVATION',total=len(jobs))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for fut in concurrent.futures.as_completed([pool.submit(task,j) for j in jobs]):
            (kind,i,batch),result=fut.result()
            try:
                if result['status']!='COMPLETE':raise RuntimeError(result.get('error'))
                if kind=='intent':
                    output=result['output']['records'];check_ids(output,[r['id'] for r in batch]);intentions.update({x['id']:x for x in output})
                else:
                    output=result['output'];assert output['id']==batch[0]['id'];check_corners(output['corners']);observations[output['id']]=output
            except Exception as e:failures.append({'phase':kind,'ids':[r['id'] for r in batch],'error':str(e)})
            save(root/'intended_effects.json',intentions);save(root/'observed_effects.json',observations);save(root/'failures.json',failures)
            progress(root,'2_INDEPENDENT_INTENTION_AND_OBSERVATION',intention_records=len(intentions),observed_records=len(observations),total_records=len(rows),failed=len(failures))
    comparisons={}
    valid=[r for r in rows if r['id'] in intentions and r['id'] in observations]
    def compare(task):
        i,batch=task
        x=call(root,f'compare_{i:02d}',CMP_PROMPT,{'records':[{'id':r['id'],'frozen_intention':intentions[r['id']],'frozen_observation':observations[r['id']]} for r in batch]},[],CMP_SCHEMA)
        return batch,x
    progress(root,'2_3_MISMATCH_COMPARISON',total=len(valid))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for fut in concurrent.futures.as_completed([pool.submit(compare,t) for t in enumerate(chunks(valid))]):
            batch,x=fut.result()
            try:
                if x['status']!='COMPLETE':raise RuntimeError(x.get('error'))
                output=x['output']['records'];check_ids(output,[r['id'] for r in batch])
                for o in output:check_corners(o['corner_comparisons'])
                comparisons.update({o['id']:o for o in output})
            except Exception as e:failures.append({'phase':'compare','ids':[r['id'] for r in batch],'error':str(e)})
            save(root/'mismatch_comparisons.json',comparisons);save(root/'failures.json',failures)
            progress(root,'2_3_MISMATCH_COMPARISON',completed=len(comparisons),total=len(rows))
    events=[];reveal_candidates=[];byid={r['id']:r for r in rows}
    for id,cmp in comparisons.items():
        obs={x['goal']:x for x in observations[id]['corners']}
        for m in cmp['corner_comparisons']:
            c=obs[m['goal']]
            if c['classification'] in ['CONFIRMED','PROVISIONAL']:
                candidate={'id':id,'goal':c['goal'],'observed':c,'comparison':m,'eligible':event_eligible(c,m)}
                reveal_candidates.append(candidate)
                if candidate['eligible']:events.append(candidate)
    events.sort(key=rank);save(root/'corner_reveal_candidates.json',reveal_candidates);save(root/'hindsight_events.json',events)
    save(root/'ranking.json',{'frozen_rule':protocol['selection'],'events':events,'time':time.time(),'manual_selection':False})
    summary={'experiences':len(rows),'intention_records':len(intentions),'observed_records':len(observations),'compared_records':len(comparisons),'corner_reveal_candidates':len(reveal_candidates),'unexpected_benefit_candidates':len(events),'failures':failures,'physical_actions':0,'levels':{'Discovery':'NOT_ESTABLISHED','Induction':'NOT_TRIGGERED','Acquisition':'NOT_TESTED'}}
    if failures:
        summary['status']='INCOMPLETE_ANALYSIS';finish(root,rows,summary);return
    if not events:
        summary['status']='NO_UNEXPECTED_CORNER_BENEFIT';finish(root,rows,summary);return
    selected=events[0];r=byid[selected['id']];goal=selected['goal'];save(root/'selected_case.json',selected)
    summary.update(selected=r['id'],goal=goal)
    summary['levels']['Discovery']='MODEL_SUPPORTED' if selected['observed']['classification']=='CONFIRMED' and selected['comparison']['unexpected_evidence']=='SUPPORTED' else 'PROVISIONAL'
    relabel={'experience_hash':r['experience_hash'],'original_intention':r['raw_intention'],'new_goal':goal,'action':r['action'],'observed_effect':observations[r['id']],'mismatch':comparisons[r['id']],'relabel_reason':selected['comparison']['evidence_for_mismatch'],'evidence_status':summary['levels']['Discovery'],'physical_success_ground_truth':False}
    save(root/'hindsight_relabel.json',relabel)
    source=Path(REPO/r['source']);native=read(source/'native_plan.json')
    compact=[{**o,'data':{k:v for k,v in o.get('data',{}).items() if k!='queued_segments'}} for o in native]
    frames=sorted((source/'execution_images').glob('*.jpg'));frames=[frames[i] for i in sorted({round((len(frames)-1)*q) for q in [.25,.5,.75]})] if frames else []
    save(root/'selected_execution_evidence.json',{'native_operations':compact,'process_frames':[str(p) for p in frames],'frame_phase':'uniform sequence samples; stage and material contact not assumed'})
    progress(root,'4_5_RELABEL_AND_INDUCE',id=r['id'],goal=goal)
    hyp=call_required(root,'induction',HYP_PROMPT,{'locked_goal':goal,'relabelled_experience':relabel,'frozen_intention_parse':intentions[r['id']],'execution_status':r['execution'],'native_operations':compact,'image_order':['Before','After']+['process sequence sample, unknown precise phase']*len(frames)},[r['before'],r['after'],*frames],hypothesis_schema(goal))
    save(root/'skill_hypotheses.json',hyp)
    if hyp['status']!='EXPERIMENTAL_HYPOTHESIS':
        summary['status']='DISCOVERY_NOT_SUFFICIENT_FOR_INDUCTION';finish(root,rows,summary);return
    assert len(hyp['hypotheses'])>=2
    summary['levels']['Induction']='EXPERIMENTAL_HYPOTHESIS'
    spec=call_required(root,'synthesis',SYNTH_PROMPT,{'locked_goal':goal,'hypotheses':hyp,'action_schema':action_schema(goal)},[],synthesis_schema(goal))
    save(root/'skill_synthesis.json',spec)
    if spec['status']!='EXPERIMENTAL_SKILL':
        summary['status']='HYPOTHESIS_ONLY';finish(root,rows,summary);return
    checked_compile(spec['python_code']);(root/'candidate_skill.py').write_text(spec['python_code']);(root/'inference_prompt.txt').write_text(spec['inference_prompt'])
    freeze={'time':time.time(),'goal':goal,'code_sha256':sha(root/'candidate_skill.py'),'prompt_sha256':sha(root/'inference_prompt.txt'),'test_images_seen':0}
    save(root/'skill_freeze.json',freeze);progress(root,'5_SKILL_FROZEN_6_OFFLINE_INTERFACE_SMOKE',total=len(tests))
    results=[]
    for i,t in enumerate(tests):
        assert sha(t['before'])==t['before_sha256']
        def request(prompt,w,h):
            return call_required(root,f'smoke_{i:02d}','当前RGB上的冻结候选技能推理；不读取After或原动作，不是物理验证。',{'goal':goal,'skill_prompt':prompt,'image_size':[w,h]},[t['before']],action_schema(goal))
        start=time.time();action,violations=invoke(spec['python_code'],spec['inference_prompt'],goal,t['before'],request)
        d=root/'smoke'/t['id'];d.mkdir(parents=True,exist_ok=True);shutil.copyfile(t['before'],d/'before.png');overlay(t['before'],action,d/'overlay.png')
        raw=read_if(root/'calls'/f'smoke_{i:02d}'/'result.json',{})
        results.append({'id':t['id'],'action':action,'contract_violations':violations,'model_call_status':raw.get('status','NOT_CALLED'),'latency_s':time.time()-start,'physical_success':'NOT_TESTED','host_safety':'NOT_TESTED'})
        save(root/'smoke_results.json',results);progress(root,'5_SKILL_FROZEN_6_OFFLINE_INTERFACE_SMOKE',completed=len(results),total=len(tests))
    assert sha(root/'candidate_skill.py')==freeze['code_sha256'] and sha(root/'inference_prompt.txt')==freeze['prompt_sha256']
    summary.update(status='CANDIDATE_READY_PENDING_ACTIVE_VALIDATION',smoke_actions=sum(x['action']['decision']=='ACTION' for x in results),smoke_abstentions=sum(x['action']['decision']=='ABSTAIN' for x in results))
    finish(root,rows,summary)

def finish(root,rows,summary):
    starts=[read(p)['time'] for p in (root/'calls').glob('*/attempt_*/started.json')]
    results=[read(p) for p in (root/'calls').glob('*/result.json')]
    usage={}
    for r in results:
        for k,v in (r.get('usage') or {}).items():
            if isinstance(v,(int,float)):usage[k]=usage.get(k,0)+v
    summary.update(model_calls=len(starts),failed_calls=sum(r['status']!='COMPLETE' for r in results),wall_time_s=time.time()-min(starts) if starts else 0,token_usage=usage)
    save(root/'summary.json',summary)
    pending={'status':'PENDING' if (root/'skill_freeze.json').exists() else 'NOT_TRIGGERED','physical_actions':0,'max_actions':1,'reason':'Fresh independent state, concrete frozen action, unchanged Host checks and current onsite execution conditions required.' if (root/'skill_freeze.json').exists() else 'No executable skill from this discovery branch.','contact_requirement':'Use existing verified Z=0 contact configuration; no repeat contact experiment. Legacy adapter uses -14.9 tool boundary and must not silently stand in for this requirement.','needed_inputs':['current RGB-D','target identity evidence','frozen action prediction','existing Host plan result'],'observer_isolation':'Post-action observer receives Before/After and execution status, never expected effect; compare frozen prediction afterward.','refinement':'at most one candidate revision after actual feedback, not auto-promoted','offline_smoke_is_active_validation':False}
    save(root/'validation_record.json',pending)
    ledger={'algorithm_version':VERSION,'specification_sha256':sha(MD),'processed_experiences':[{'hash':r['experience_hash'],'source':r['source']} for r in rows],'next_use':'--episode PATH processes a new rollout only; reuse this output root resumes cached calls; no historical full rescan'}
    save(root/'incremental_ledger.json',ledger)
    from report import render
    render(root)
    progress(root,'COMPLETE',status=summary['status'],levels=summary['levels'])
if __name__=='__main__':main()
