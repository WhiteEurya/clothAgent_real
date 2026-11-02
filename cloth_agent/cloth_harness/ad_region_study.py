"""Frozen human region evaluation and A/D repeats; no robot dispatch capability."""
import argparse,copy,csv,hashlib,json,math,random,shutil,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
from PIL import Image
from .runtime.gpt_policy import payload
from .runtime.dataset import load_observation
from .experiment import trial,render_ab,save,read,sha
from .evaluation.evaluator import validate_label


def score(plan,label,size):
    """Partial region labels: outside both is UNLABELLED, not an incorrect pick."""
    expected=label['next_decision'];decision=plan['decision']
    if expected=='UNCERTAIN':return 'UNLABELLED'
    if decision=='ABSTAIN':return 'ABSTAIN'
    if expected=='DONE':return 'ACCEPTABLE' if decision=='DONE' else 'UNSUITABLE'
    if decision=='DONE':return 'UNSUITABLE'
    point=plan['grasp_px'];x,y=point[0]/size[0],point[1]/size[1]
    hits=[]
    for b in label['regions']:
        X,Y,W,H=b['rect']
        if X<=x<=X+W and Y<=y<=Y+H:hits.append(b['kind'])
    if 'allow' in hits and 'avoid' in hits:return 'UNLABELLED'
    if 'avoid' in hits:return 'UNSUITABLE'
    if 'allow' in hits:return 'ACCEPTABLE'
    return 'UNLABELLED'


def prepare(args):
    root=args.root.resolve();root.mkdir(parents=True,exist_ok=False);(root/'abcd').mkdir()
    chosen=read(args.selection);old=read(args.prior_labels) if args.prior_labels else {};pairs=[];oracle={};drafts={};manifest=[]
    for i,p in enumerate(chosen):
        key=f'S{i+1:02d}';folder=root/key;folder.mkdir();sourceid=p['id'];p=copy.deepcopy(p);p['id']=key
        # Validate RGB-D integrity, even though the repeated policy uses only RGB.
        load_observation(p['before']);load_observation(p['after'])
        for side in ('before','after'):shutil.copyfile(Path(p[side])/'rgb.png',folder/(side+'.png'))
        p['image_size']=list(Image.open(folder/'after.png').size);pairs.append(p)
        entry={'id':key,'source':p['source'],'source_case':sourceid,'before_sha256':sha(folder/'before.png'),'after_sha256':sha(folder/'after.png'),'goal_sha256':sha(p['goal']),
               'selection_reason':'previous human NO_CLEAR_PROGRESS' if i<3 else 'candidate area decrease plus manual visual inspection; not a failure label','selection_area_delta_proxy':p.get('area_delta')};manifest.append(entry)
        if i<3 and old.get(sourceid,{}).get('confirmed_by'):
            v=copy.deepcopy(old[sourceid]);v['original_label_source']=str(args.prior_labels);v['evidence']=[key+'/before.png',key+'/after.png'];oracle[key]=v
            drafts[key]={'labels':v['labels'],'reviewer':v['confirmed_by'],'note':v.get('note',''),'regions':[],'next_decision':''}
    save(root/'pairs.json',pairs);save(root/'selection_manifest.json',manifest);save(root/'oracle_labels.json',oracle);save(root/'review_drafts.json',drafts);save(root/'next_action_labels.json',{})
    save(root/'abcd/protocol_ABCD.json',{'selected_cases':[p['id'] for p in pairs],'purpose':'review UI compatibility only; model study is A/D'})
    save(root/'study_protocol.json',{'conditions':['A','D'],'repeats':3,'cases':len(pairs),'total_requests':len(pairs)*6,'seed':20261025,'workers':2,'robot_execution':False,
        'primary_evaluation':'partial human pick-region admissibility + DONE; not whole trajectory utility','unknown_handling':'outside all labelled regions is UNLABELLED; ABSTAIN separate',
        'evaluation_labels_never_in_model_input':True,'selection_bias':'purposively enriched development diagnostic sample, not held-out or unbiased failure rate',
        'prior_seen_cases':3,'notes':'No proven empty grasp/slip selected without independent evidence; human may mark unknown. Freeze all labels before any generation. Keep all selected cases regardless of labels; summarize progress strata separately.'})
    print(root,flush=True)


def run(args):
    root=args.root.resolve();pairs=read(root/'pairs.json');labels=read(root/'oracle_labels.json');regions=read(root/'next_action_labels.json')
    for p in pairs:
        key=p['id'];v=labels.get(key,{});g=regions.get(key,{})
        validate_label(v)
        if not v.get('confirmed_by') or not g.get('confirmed_by'):raise ValueError(key+': finish and confirm both outcome and next-action labels first')
    manifest=read(root/'selection_manifest.json')
    for p,e in zip(pairs,manifest):
        for side in ('before','after'):
            if sha(Path(p[side])/'rgb.png')!=e[side+'_sha256'] or sha(root/p['id']/(side+'.png'))!=e[side+'_sha256']:raise ValueError('review/model image hash mismatch')
        if sha(p['goal'])!=e['goal_sha256']:raise ValueError('goal changed')
    frozen=root/'frozen';frozen.mkdir(exist_ok=True)
    for name,value in [('oracle_labels.json',labels),('next_action_labels.json',regions),('pairs.json',pairs),('study_protocol.json',read(root/'study_protocol.json'))]:
        path=frozen/name
        if path.exists() and read(path)!=value:raise ValueError('frozen inputs changed: start a new study')
        save(path,value)
    for src,name in [(Path(__file__),'evaluator_and_runner.py'),(Path(__file__).parent/'runtime/gpt_policy.py','gpt_policy.py')]:
        dst=frozen/name
        if dst.exists() and sha(dst)!=sha(src):raise ValueError('frozen code changed')
        shutil.copyfile(src,dst)
    output=root/'results';output.mkdir(exist_ok=True);tasks=[]
    for pair in pairs:
        for rep in range(1,4):
            for condition in ('A','D'):
                f=output/pair['id']/f'{condition}{rep}';f.mkdir(parents=True,exist_ok=True)
                body=payload(pair,condition,oracle=labels[pair['id']] if condition=='D' else None)
                if (f/'request.json').exists() and read(f/'request.json')!=body:raise ValueError('request changed')
                save(f/'request.json',body)
                if (f/'result.json').exists():continue
                if (f/'model/attempt.json').exists():raise ValueError('unfinished dispatched request: no silent retry')
                tasks.append((str(f),pair,condition+str(rep)))
    random.Random(20261025).shuffle(tasks);save(frozen/'request_manifest.json',{'order':[(p['id'],c) for _,p,c in tasks],'label_sha256':sha(frozen/'oracle_labels.json'),'evaluation_sha256':sha(frozen/'next_action_labels.json')})
    started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(trial,t) for t in tasks]):
            value=future.result();print(json.dumps({k:value.get(k) for k in ('case','condition','status','model_s','error')},ensure_ascii=False),flush=True)
    rows=[read(p) for p in sorted(output.glob('*/*/result.json'))];render_ab(output,rows,[])
    scores=[]
    for row in rows:
        p=next(p for p in pairs if p['id']==row['case']);status=score(row['plan'],regions[row['case']],p['image_size']) if row['status']=='COMPLETE' else 'REQUEST_FAILED'
        scores.append({'case':row['case'],'condition':row['condition'][0],'repeat':int(row['condition'][1:]),'score':status,'model_s':row.get('model_s'),'outcome_stratum':labels[row['case']]['labels']['visible_unfolding_progress']})
    save(output/'scores.json',scores);summary={'elapsed_s':time.perf_counter()-started,'limitations':'3 repeats of same state are dependent; region labels do not evaluate direction/distance, actual benefit, or full trajectory safety.'}
    for c in 'AD':
        group=[s for s in scores if s['condition']==c];counts={k:sum(s['score']==k for s in group) for k in ('ACCEPTABLE','UNSUITABLE','UNLABELLED','ABSTAIN','REQUEST_FAILED')};known=counts['ACCEPTABLE']+counts['UNSUITABLE'];counts.update(total=len(group),label_coverage=known/len(group) if group else None,acceptable_among_labelled=counts['ACCEPTABLE']/known if known else None);summary[c]=counts
    save(output/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run']);p.add_argument('--root',type=Path,required=True);p.add_argument('--selection',type=Path);p.add_argument('--prior-labels',type=Path);a=p.parse_args()
    if a.command=='prepare' and not a.selection:p.error('prepare requires --selection')
    {'prepare':prepare,'run':run}[a.command](a)
