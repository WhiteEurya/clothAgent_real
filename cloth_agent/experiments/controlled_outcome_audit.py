"""Stage 3A: independently assess outcomes, then audit existing Stage 2 pairs."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import json
import math
from pathlib import Path
from PIL import Image
from .action_overlay_review import frozen
from .discovery_data import digest,read,save,file_hash
from .decision_state_discovery import Requests,montage

PRIORITY_Z=[('s_6444a0cab2ee1de6','s_0f4c38e58ccc2806'),
 ('s_783ad84a1555c45d','s_60c8d22cf05d7c24'),
 ('s_0933900f3bcb191a','s_dcae4a5ae7de4b0b'),
 ('s_dcae4a5ae7de4b0b','s_4df9f60fd1572842')]
TYPES=('GRASP_Z_PAIR','TARGET_PAIR','GRASP_SUBREGION_PAIR')
EVAL_FIELDS=('target_selection','grasp_acquisition','target_structure_acquired','transport','laydown',
 'task_progress','earliest_failure_stage','perception_comparison','grasp_execution_experience')


def action_difference(a,b):
    result={}
    for field in ('grasp','target'):
        x=a.get(field);y=b.get(field)
        result[field+'_difference_mm']=[v-u for u,v in zip(x,y)] if x and y else None
    ma=[x['args'] for x in a['raw'] if x['name']=='move'];mb=[x['args'] for x in b['raw'] if x['name']=='move']
    result.update(grasp_xy_distance_mm=math.dist(a['grasp'][:2],b['grasp'][:2]),
        target_xy_distance_mm=math.dist(a['target'][:2],b['target'][:2]),
        target_xyz_distance_mm=math.dist(a['target'],b['target']),
        move_sequences_a=ma,move_sequences_b=mb,
        action_names_a=[x['name'] for x in a['raw']],action_names_b=[x['name'] for x in b['raw']],
        warning='All differences are recorded, not assumed controlled. Same endpoints do not ensure same lifted path, yaw or execution.')
    return result


def select_pairs(action_manifest,review):
    rows={s['sample_id']:s for s in action_manifest['samples']}
    candidates={tuple(sorted(p['samples'])):p for p in review['appearance_similar_action_different']}
    ordered=list(PRIORITY_Z)+[tuple(p['samples']) for p in review['appearance_similar_action_different']]
    pairs=[];skipped=[];seen=set()
    for ids in ordered:
        key=tuple(sorted(ids))
        if key in seen:continue
        seen.add(key)
        if key not in candidates or any(s not in rows for s in ids):
            skipped.append(dict(samples=list(ids),reason='Not present in Stage2 candidate evidence'));continue
        a,b=[rows[s]['action'] for s in ids]
        if any(x.get(k) is None for x in (a,b) for k in ('grasp','target')):
            skipped.append(dict(samples=list(ids),reason='Missing commanded endpoints'));continue
        d=action_difference(a,b)
        if tuple(ids) in PRIORITY_Z:typ='GRASP_Z_PAIR'
        elif d['grasp_xy_distance_mm']<=2 and d['target_xyz_distance_mm']<=1 and abs(d['grasp_difference_mm'][2])>=.5:typ='GRASP_Z_PAIR'
        elif d['grasp_xy_distance_mm']<=2 and d['target_xy_distance_mm']>=10:typ='TARGET_PAIR'
        elif d['grasp_xy_distance_mm']>=10:typ='GRASP_SUBREGION_PAIR'
        else:
            skipped.append(dict(samples=list(ids),reason='Does not meet declared numeric candidate rules'));continue
        pairs.append(dict(pair_id='p_'+digest([typ,list(ids)])[:12],pair_type=typ,samples=list(ids),
            priority=tuple(ids) in PRIORITY_Z,stage2_evidence=candidates[key],action_variables=d,
            selection_note='Selected before reading outcomes. Candidate group is not proof of appearance similarity, isolation or subregion identity; GPT must recheck.'))
    return pairs,skipped


def clean_event(event):
    state=event.get('robot_state') or {};feedback=state.get('gripper_feedback') or {}
    if event.get('simulated') or state.get('state')=='simulated' or feedback.get('simulated'):return None
    if not isinstance(event,dict):return None
    return {**{k:event[k] for k in ('name','args','action_index','requested_at','completed_at','success','error','actual_ee_pose') if k in event},
        'gripper_feedback':{k:feedback[k] for k in ('available','state','position_pulse','mechanical_grasp_detected','error_code','read_errors') if k in feedback},
        'robot_state':{k:state[k] for k in ('state','error_warn') if k in state}}


def gather_sample(sample,action_row,stage_dir):
    sid=sample['sample_id'];root=Path(sample['source_directory']);sources=[];images=[];missing=[]
    def add(role,kind,content=None,path=None):
        entry=dict(source_id=sid+'.'+role,kind=kind,content=content)
        if path:entry['provenance']=dict(file=str(Path(path).resolve()),sha256=file_hash(path))
        sources.append(entry);return entry['source_id']
    def add_image(role,path,expected_hash=None):
        if not path or not Path(path).is_file():missing.append(role+': image unavailable');return
        if expected_hash and file_hash(path)!=expected_hash:raise ValueError(f'changed image {sid}/{role}')
        with Image.open(path) as im:im.verify()
        ident=add(role,'image',dict(role=role),path)
        target=stage_dir/'images'/f'{sid}_{role}.png';target.parent.mkdir(parents=True,exist_ok=True)
        with Image.open(path) as im:im.convert('RGB').save(target)
        images.append(dict(source_id=ident,role=role,path=str(target.resolve()),sha256=file_hash(target)))
    add('commanded_action','commanded_plan',action_row['action'])
    add_image('before',sample['before_rgb'],sample['image_hashes'].get('before'))
    add_image('after',sample.get('after_rgb'),sample['image_hashes'].get('after'))
    # Consume actual hardware events, never planned/simulated trajectory or preflight telemetry.
    ef=root/'execution.json';ex=read(ef)
    events=[clean_event(e) for e in ex.get('actual_robot_actions',[]) if isinstance(e,dict)]
    events=[e for e in events if e is not None]
    if ef.exists():add('actual_execution','controller_record',dict(
        **{k:ex[k] for k in ('physical_execution','execution_completed','robot_errors','started_at','completed_at') if k in ex},
        actual_robot_actions=events,excluded_simulated_events=len(ex.get('actual_robot_actions',[]))-len(events),
        warning='Controller completion/contact is not held-cloth proof; inspect sequence against planned target.'),ef)
    else:missing.append('actual_execution: unavailable')
    sf=root/'hold_check/grasp_snapshots.json';snapshots=read(sf)
    seen=set()
    def snapshot(role,item,path):
        if not isinstance(item,dict):return
        allowed={k:item[k] for k in ('status','captured_after','captured_before','action_index','requested_lift_mm','frame_utc','frame_monotonic_ns','asynchronous','note') if k in item}
        event=clean_event(item.get('action') or {})
        allowed['action_event']=event
        ident=add(role+'_record','capture_record',allowed,path)
        image_path=Path(item.get('image',''))
        if not image_path.is_absolute():image_path=root/image_path
        if item.get('status')=='CAPTURED' and image_path.is_file():
            h=file_hash(image_path)
            if h not in seen:
                seen.add(h);add_image(role,str(image_path));sources[-1]['content']['capture_record_id']=ident
        else:missing.append(role+': no confirmed captured image')
    for key,item in snapshots.items():snapshot('grasp_'+key,item,sf)
    rf=root/'recording.json';rec=read(rf)
    for i,item in enumerate(rec.get('lift_snapshots',[])):snapshot(f'additional_lift_{i}',item,rf)
    if not snapshots and not rec.get('lift_snapshots'):missing.append('grasp/lift snapshots unavailable')
    evf=root/'evaluation.json';ev=read(evf)
    if not ev:ev=(sample.get('outcome') or {}).get('evaluation') or {}
    add('historical_evaluator','fallible_evaluator',dict(
        judgments={k:ev[k] for k in EVAL_FIELDS if k in ev},
        warning='Historical evaluator may be policy-rewritten. UNCHANGED->ACQUISITION FAILURE is not direct physical evidence. Future advice omitted.'),evf if evf.exists() else None)
    sup=root/'supervisor_after.json'
    if sup.exists():
        v=read(sup);add('progress_ledger','fallible_host_or_supervisor',
            {k:v[k] for k in ('status','current_step','completed_steps','confidence','evidence','reason','fallback','local_deterministic','completion_ledger_source') if k in v},sup)
    else:missing.append('completion/progress ledger unavailable')
    cf=root/'perception_comparison.json'
    if cf.exists():add('perception_comparison','fallible_evaluator',read(cf),cf)
    add('availability','evidence_inventory',dict(missing=missing,image_roles=[x['role'] for x in images],
        warning='No video inference performed. Still images may miss transient acquisition/slip. Missing evidence is not failure.'))
    # Keep all captures, reducing only additional lift images to a labeled sheet when needed.
    if len(images)>6:
        fixed=[x for x in images if x['role'] in ('before','after')];extra=[x for x in images if x not in fixed]
        sheet=stage_dir/'images'/f'{sid}_lift_sheet.png'
        montage([x['source_id'] for x in extra],{x['source_id']:Path(x['path']) for x in extra},sheet)
        sheet_id=add('lift_sheet','image',dict(role='lift_contact_sheet',members=[x['source_id'] for x in extra]))
        images=fixed+[dict(source_id=sheet_id,role='lift_contact_sheet',path=str(sheet.resolve()),sha256=file_hash(sheet))]
    return dict(sample_id=sid,run_id=sample['run_id'],iteration_id=sample['iteration_id'],
        evidence=sources,images=images,missing=missing)


def model_sample(row):
    return dict(sample_id=row['sample_id'],evidence=[{k:e[k] for k in ('source_id','kind','content')} for e in row['evidence']],
        image_ids=[{k:i[k] for k in ('source_id','role')} for i in row['images']])


def prepare(source,output):
    source=Path(source).resolve();stage_dir=Path(output).resolve()/'stage3a'
    am=read(source/'manifest.json');review=read(source/'stage2/action_review.json')
    baseline=Path(am['source']);fm=frozen(baseline/'manifest.json',True)
    if digest(fm)!=am['source_manifest_hash']:raise ValueError('Stage2 dataset provenance mismatch')
    tax=frozen(baseline/'stage1/taxonomy_v0.json');frozen(baseline/'stage1/assignments.json')
    if digest(tax)!=am['frozen_taxonomy_hash'] or frozen(source/'frozen_taxonomy.json')!=tax:raise ValueError('Stage1 taxonomy changed')
    pairs,skipped=select_pairs(am,review)  # This happens before any outcome gathering.
    ids=list(dict.fromkeys(s for p in pairs for s in p['samples']));lookup={s['sample_id']:s for s in fm['samples']};actions={s['sample_id']:s for s in am['samples']}
    manifest=dict(protocol='controlled-outcome-pairs-v1',source=str(source),model=am['model'],task_goal=am['task_goal'],
        frozen_taxonomy_hash=digest(tax),stage2_review_sha256=file_hash(source/'stage2/action_review.json'),
        baseline_manifest_hash=digest(fm),selection_rules=dict(grasp_xy_tolerance_mm=2,release_xyz_tolerance_mm=1,
            min_grasp_z_difference_mm=.5,min_target_xy_difference_mm=10,min_subregion_xy_difference_mm=10,
            note='Numeric filters define candidate comparisons only. They do not prove visual comparability or single-variable control.'),
        pairs=pairs,skipped_pairs=skipped,samples=[gather_sample(lookup[s],actions[s],stage_dir) for s in ids])
    save(stage_dir/'manifest.json',manifest,immutable=True)
    save(stage_dir/'manifest.sha256.json',dict(sha256=digest(manifest)),immutable=True)
    return manifest


def run(manifest,output,requests,workers=2):
    d=Path(output)/'stage3a';rows={r['sample_id']:r for r in manifest['samples']}
    def ask_sample(row):
        sid=row['sample_id'];payload=dict(sample_ids=[sid],task_goal=manifest['task_goal'],sample=model_sample(row))
        v=requests.ask('outcome_pair','sample',payload,[(i['source_id'],Path(i['path'])) for i in row['images']],
            manifest['frozen_taxonomy_hash'],evidence_ids=[e['source_id'] for e in row['evidence']])
        save(d/'samples'/f'{sid}.json',v,immutable=True);return sid,v
    with ThreadPoolExecutor(max_workers=workers) as pool:
        assessed=dict(pool.map(ask_sample,rows.values()))
    def ask_pair(pair):
        ids=pair['samples'];subset=[rows[s] for s in ids];payload=dict(sample_ids=ids,pair=pair,task_goal=manifest['task_goal'],
            samples=[model_sample(r) for r in subset],independent_sample_assessments=[assessed[s] for s in ids])
        v=requests.ask('outcome_pair','pair',payload,[(i['source_id'],Path(i['path'])) for r in subset for i in r['images']],
            manifest['frozen_taxonomy_hash'],evidence_ids=[e['source_id'] for r in subset for e in r['evidence']])
        if v['pair_id']!=pair['pair_id'] or v['pair_type']!=pair['pair_type']:raise ValueError('pair identity mismatch')
        save(d/'pair_reviews'/f'{pair["pair_id"]}.json',v,immutable=True);return v
    with ThreadPoolExecutor(max_workers=workers) as pool:
        reviews=list(pool.map(ask_pair,manifest['pairs']))
    result=dict(protocol=manifest['protocol'],frozen_taxonomy_hash=manifest['frozen_taxonomy_hash'],
        source_stage2_review_sha256=manifest['stage2_review_sha256'],sample_assessments=assessed,
        pairs=[dict(selection=p,assessment=v) for p,v in zip(manifest['pairs'],reviews)],
        limitation='Retrospective matched-pair audit, not randomized causal identification. No taxonomy, policy or training output.')
    save(d/'pairs.json',result,immutable=True)
    return result


def report(manifest,output,error=None):
    d=Path(output)/'stage3a';result=read(d/'pairs.json');aud=d/'requests/calls.jsonl'
    calls=[json.loads(x) for x in aud.read_text().splitlines()] if aud.exists() else []
    entries=result.get('pairs',[])
    stats=dict(status='COMPLETED' if result else 'FAILED' if error else 'PREPARED',error=error,
        sample_count=len(manifest['samples']),pair_count=len(manifest['pairs']),
        groups=dict(Counter(p['pair_type'] for p in manifest['pairs'])),
        model_calls=sum(not x['cache_hit'] for x in calls),cache_hits=sum(x['cache_hit'] for x in calls),
        elapsed_model_s=sum(x['elapsed_s'] for x in calls),
        classifications=dict(Counter(x['assessment']['classification'] for x in entries)))
    save(d/'status.json',stats)
    lines=['# Stage 3A: controlled outcome pair audit','',json.dumps(stats,ensure_ascii=False),'',
        'Retrospective association only. Stage 1 taxonomy and Stage 2 are unchanged; no policy or training output.',
        'Every judgment cites source IDs. Source files and hashes: [manifest.json](manifest.json). Full assessments: [pairs.json](pairs.json).',
        '', '## Independent sample outcomes','', '| Sample | Acquisition | Progress | Completion | Side effect | Execution reach |', '|---|---|---|---|---|---|']
    for sid,a in result.get('sample_assessments',{}).items():
        lines.append('| '+f'[{sid}](samples/{sid}.json)'+' | '+' | '.join(a[k]['status'] for k in ('acquisition','task_progress','completion','side_effect','execution_reach'))+' |')
    rank={'STRONG':3,'MODERATE':2,'WEAK':1,'NONE':0}
    for typ,title in zip(TYPES,('grasp-depth','target-distance','grasp-subregion')):
        group=[x for x in entries if x['selection']['pair_type']==typ]
        lines+=['',f'## {typ}','',f'### Strongest evidence for {title} effects','']
        supported=[x for x in group if x['assessment']['parameter_effect']['status']=='SUPPORTED_ASSOCIATION']
        if supported:
            best=max(supported,key=lambda x:rank[x['assessment']['evidence_strength']]);lines+=[best['assessment']['pair_id']+': '+best['assessment']['conclusion']]
        else:lines+=['No supported parameter-effect association established in this group. This does not establish absence of an effect.']
        for item in group:
            a=item['assessment'];p=item['selection'];lines+=['',f'### {a["pair_id"]}: '+ ' vs '.join(p['samples']),'',
                f'Classification: **{a["classification"]}**; effect evidence: {a["evidence_strength"]}.',a['conclusion'],'',
                f'[Full evidence and confounders](pair_reviews/{a["pair_id"]}.json)','',
                '```json',json.dumps(a,ensure_ascii=False,indent=2),'```']
    for title,predicate in [
        ('Pairs where multiple different actions both appear feasible',lambda a:a['multiple_actions_feasible']['status']=='SUPPORTED'),
        ('Unresolved pairs',lambda a:a['classification']=='UNRESOLVED'),
        ('Possible RGB state-aliasing cases',lambda a:a['rgb_state_aliasing']['status']=='POSSIBLE')]:
        matches=[x['assessment'] for x in entries if predicate(x['assessment'])];lines+=['','## '+title,'']
        lines += [f'- [{a["pair_id"]}](pair_reviews/{a["pair_id"]}.json): {a["conclusion"]}' for a in matches] or ['None established by the available evidence.']
    if not result:lines+=['','Incomplete: no final cross-pair conclusions claimed. Completed individual calls are preserved.']
    (d/'report.md').write_text('\n'.join(lines)+'\n');return stats


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--source',type=Path,required=True);ap.add_argument('--output',type=Path)
    ap.add_argument('--execute',action='store_true');ap.add_argument('--max-calls',type=int,default=48)
    ap.add_argument('--workers',type=int,choices=(1,2),default=2);ap.add_argument('--no-cache',action='store_true')
    a=ap.parse_args(argv);output=a.output or a.source
    if a.max_calls<1:ap.error('max-calls must be positive')
    m=prepare(a.source,output);print(json.dumps(report(m,output),ensure_ascii=False),flush=True)
    if not a.execute:return 0
    from .discovery_model import CodexMultimodal
    requests=Requests(output/'stage3a',m['model'],CodexMultimodal(m['model']),a.no_cache,max_calls=a.max_calls)
    error=None
    try:run(m,output,requests,a.workers)
    except Exception as exc:error=f'{type(exc).__name__}: {exc}';raise
    finally:
        report(m,output,error)
        am=read(a.source/'manifest.json');baseline=Path(am['source'])
        if digest(frozen(baseline/'stage1/taxonomy_v0.json'))!=m['frozen_taxonomy_hash']:raise ValueError('Stage1 modified')
        frozen(baseline/'stage1/assignments.json')
        if file_hash(a.source/'stage2/action_review.json')!=m['stage2_review_sha256']:raise ValueError('Stage2 modified')
    return 0


if __name__=='__main__':raise SystemExit(main())
