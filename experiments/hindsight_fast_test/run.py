"""One offline contact-sheet discovery -> provisional skill -> held-out RGB test.
No robot imports. Discovery models have no project filesystem or chat history.
"""
from pathlib import Path
import sys, json, time, hashlib, concurrent.futures, shutil, html, ast, math
from PIL import Image, ImageDraw
import numpy as np

CODE = Path(__file__).resolve().parent
REPO = CODE.parents[1]
sys.path.insert(0, str(REPO / 'experiments/hindsight_smoke_test'))
from transport import call, save, read, sha

ROOT = REPO / 'reports/hindsight_fast_test_20261029'
INDEX = REPO / 'reports/offline_failure_mining_20261029/index.json'
SEED = 'hindsight-fast-v0-20261029-independent-runs'
S = {'type': 'string'}
B = {'type': 'boolean'}
def enum(*x): return {'type': 'string', 'enum': list(x)}
def obj(d): return {'type': 'object', 'properties': d, 'required': list(d), 'additionalProperties': False}
def arr(d): return {'type': 'array', 'items': d}
RATING = {'type': 'integer', 'minimum': 0, 'maximum': 3}
POINT = {'type': 'array', 'items': {'type': 'integer'}, 'minItems': 2, 'maxItems': 2}
SCAN = obj({'candidates': arr(obj({'id': S, 'local_change': RATING, 'possible_reveal': RATING, 'evidence': S, 'uncertainty': S})), 'screening_limitations': S})
DETAIL = obj({'cases': arr(obj({'id': S, 'intended_object': S, 'intended_effect': S, 'actual_command': S, 'actual_material_identity': S, 'observed_local_change': S, 'observation_vs_explanation': S, 'reveal': enum('SUPPORTED','PROVISIONAL','NO_REVEAL','UNKNOWN'), 'unplanned_benefit': enum('SUPPORTED','PROVISIONAL','PLANNED','NONE','UNKNOWN'), 'target_endpoint': S, 'reveal_evidence': RATING, 'unexpected_evidence': RATING, 'research_value': RATING, 'limitations': S}))})
INDUCTION = obj({'status': enum('EXPERIMENTAL_HYPOTHESIS','INSUFFICIENT_EVIDENCE'), 'original_intention': S, 'actual_manipulation': S, 'observed_effect': S, 'intention_outcome_mismatch': S, 'hindsight_goal': S, 'hypotheses': arr(obj({'name': S, 'physical_explanation': S, 'observed_support': S, 'unobserved_assumptions': S, 'different_prediction': S, 'falsifier': S})), 'observable_geometry': arr(S), 'action_generation_rules': arr(S), 'applicability': S, 'uncertainty': S, 'why_worth_testing': S})
SYNTHESIS = obj({'status': enum('EXPERIMENTAL_SKILL','NOT_EXECUTABLE'), 'python_code': S, 'inference_prompt': S, 'algorithm_explanation': S, 'engineering_assumptions': arr(S), 'evidence_limits': S})
ACTION = obj({'decision': enum('ACTION','ABSTAIN'), 'target_corner': S, 'target_state': enum('OCCLUDED','VISIBLE','OUT_OF_VIEW','UNKNOWN'), 'observable_geometry': S, 'grasp_region': S, 'grasp_px': POINT, 'target_px': POINT, 'grasp_angle_deg': {'type':'integer','minimum':0,'maximum':180}, 'direction_reason': S, 'expected_change': S, 'falsifier': S, 'uncertainty': S, 'experimental_only': B})

SCAN_PROMPT = '''你是隔离的历史真机经验视觉初筛器。每个contact sheet单元是一对真实Before/After，左前右后，标有随机身份ID。快速浏览所给全部单元，从本批自主选最多3对值得放大研究的候选。重点是局部结构变化和可能的真实语义袖口/下摆显露，而非面积增长。其他区域卷曲不排除有价值的局部变化。普通褶尖不等于语义端点，低分辨率只允许候选判断。不要猜原始意图，不提出技能规则。不得按已知案例编号或任何先前报告选择。每个评分0无证据/1弱/2中/3强；没有可信线索允许少选或空。'''
DETAIL_PROMPT = '''独立审核所给最多5对原始Before/After及真实动作、原始决策字段。每对各一条。分开还原原始意图、实际命令、可确认接触材料、可观察结果。无法确认夹持材料写UNKNOWN；EXECUTED不是抓取成功。识别可能原意图之外的局部有益变化，尤其Corner Reveal。原始意图缺失写UNKNOWN，不事后补目标。衣服语义左右不是图像左右。分别给出语义显露、意外收益的证据等级0..3及研究价值0..3。可以保留PROVISIONAL，但不可将袖状新轮廓当确认真值。本阶段不归纳技能，不挑选正确运动方向。'''
INDUCT_PROMPT = '''你在独立会话中仅研究所给一个真实案例。先比较原始意图、实际牵动和观察效果，明确是否有意外有益变化。将值得研究的实际变化重标记为主动目标，并回答若事先追求它应怎样选动作。至少提出两个竞争性物理解释、不同预测和反证。再尝试归纳当前RGB中可观察、可计算的几何关系和动作生成规则。不得记忆案例坐标，不按图像固定方向，不把原始Agent语义或隐藏层序当事实，不套用外部专家规则。允许带明确适用条件的实验性假设，不要求单例证明可靠技能。若没有意外收益或连实验假设都不成立则说明不足。'''
SYNTH_PROMPT = '''根据提供的单轮自主归纳结果，写一个实验性Python Skill，不进行第二轮归纳。输出完整python_code以及冻结的inference_prompt。
函数必须是 generate_action(rgb, target_corner)。rgb是numpy RGB数组。可以调用一次注入的 VISION_MODEL(rgb, prompt) 返回动作字典；这是冻结的视觉模型，能读取当前图并按固定Schema输出目标状态、可见几何、Pa/Pb、角度、理由、预期及反证。函数应将inference_prompt和当前图像尺寸、target_corner结合传给它；默认target_corner='AUTO'时从当前图自主定位需要恢复的端点，无可靠目标可以ABSTAIN。只调用一次视觉模型，不用历史After或坐标。可选择加入简单numpy几何计算；不能捏造校准或安全通过。
代码仅允许 import numpy as np、import math；不允许文件IO、网络、路径、环境、反射、eval/exec、机器人、任意导入。全局可用 FROZEN_PROMPT（即inference_prompt）和 VISION_MODEL。保持函数简洁，返回视觉模型的原始动作字典，若无法成立保持ABSTAIN。推理规则由你的发现决定，不预先指定抓点或方向。清楚说明工程假设和证据边界。'''

def key(s): return hashlib.sha256((SEED+'|'+s).encode()).hexdigest()
def run_group(source):
    p=Path(source).parts
    return '/'.join(p[:2]) if p[0]=='reports' else str(Path(source).parent)
def progress(stage, **kw):
    d={'stage':stage,'time':time.time(),**kw};save(ROOT/'progress.json',d);print(json.dumps(d),flush=True)
def invoke(name, prompt, payload, images, schema):
    x=call(ROOT,name,prompt,payload,[Path(p) for p in images],schema)
    if x['status']!='COMPLETE':raise RuntimeError(name+': '+str(x.get('error')))
    return x['output']
def intention(r):
    folder=REPO/r['source'];p=folder/'dispatch_intent.json'
    d=read(p).get('plan',{}) if p.exists() else {}
    if not d and (folder/'plan.json').exists():d=read(folder/'plan.json')
    fields=['short_reason','purpose','expected_effect','expected_stop_condition','reason','rationale','target_region','is_smoothing']
    return {'original_fields':{k:d[k] for k in fields if k in d},'source':str(p),'missing_intention':not any(d.get(k) for k in fields[:6])}
def pair_images(rs):return [REPO/r[k] for r in rs for k in ['before','after']]
def packed(rs):return {'image_order':[r['id']+' '+k for r in rs for k in ['Before','After']], 'records':[{'id':r['id'],'original_intention':intention(r),'executed_action':r['action'],'execution_status':r['execution']} for r in rs]}

def prepare():
    raw=read(INDEX);rows=[]
    for run in raw['runs']:
        for s in run['steps']:
            if s['valid_pair']:
                rows.append({**{k:s[k] for k in ['source','before','after','before_sha256','after_sha256','action','execution','captured_at']},'run':run_group(s['source']),'id':'F'+hashlib.sha256(s['source'].encode()).hexdigest()[:10]})
    groups=sorted({r['run'] for r in rows},key=key);heldout=groups[:3]
    val=[sorted([r for r in rows if r['run']==g],key=lambda r:key(r['id']))[0] for g in heldout]
    train=sorted([r for r in rows if r['run'] not in heldout],key=lambda r:key(r['id']))
    # Split is persisted BEFORE loading even one image; validation is not opened here.
    save(ROOT/'split.json',{'seed':SEED,'heldout_complete_runs':heldout,'discovery_count':len(train),'excluded_heldout_pairs':len(rows)-len(train),'validation':[{'id':r['id'],'run':r['run'],'before':r['before'],'before_sha256':r['before_sha256']} for r in val]})
    save(ROOT/'experience_index.json',rows)
    sheets=[]
    for b in range(0,len(train),20):
        rs=train[b:b+20];im=Image.new('RGB',(1200,200*math.ceil(len(rs)/2)),'white');d=ImageDraw.Draw(im)
        for j,r in enumerate(rs):
            x=(j%2)*600;y=(j//2)*200;d.text((x+5,y+4),r['id']+' BEFORE | AFTER',fill='black')
            for side,k in enumerate(['before','after']):
                assert sha(REPO/r[k])==r[k+'_sha256']
                pic=Image.open(REPO/r[k]).convert('RGB');pic.thumbnail((298,170));im.paste(pic,(x+side*300,y+25))
        p=ROOT/'sheets'/f'page_{b//20:02d}.jpg';p.parent.mkdir(exist_ok=True);im.save(p,quality=93)
        sheets.append({'path':str(p),'ids':[r['id'] for r in rs],'sha256':sha(p)})
    save(ROOT/'contact_sheets.json',sheets)
    return rows,train,val,sheets

def validate_code(code):
    tree=ast.parse(code)
    forbidden={'open','exec','eval','compile','getattr','setattr','globals','locals','__import__','input','breakpoint','help','exit'}
    for n in ast.walk(tree):
        if isinstance(n,(ast.ImportFrom,ast.With,ast.AsyncWith,ast.ClassDef,ast.Global,ast.Nonlocal)):raise ValueError('unsupported code node '+type(n).__name__)
        if isinstance(n,ast.Import) and any(a.name not in ['numpy','math'] for a in n.names):raise ValueError('nonallowlisted import')
        if isinstance(n,ast.Name) and (n.id in forbidden or n.id.startswith('__')):raise ValueError('forbidden name')
        if isinstance(n,ast.Attribute) and (n.attr.startswith('_') or any(w in n.attr.lower() for w in ['load','save','file','ctypes','dump','frombuffer','memmap','fromregex'])):raise ValueError('forbidden attribute')
        if isinstance(n,(ast.While,ast.AsyncFunctionDef)):raise ValueError('unsupported unbounded code')
    fs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='generate_action']
    assert len(fs)==1 and [a.arg for a in fs[0].args.args]==['rgb','target_corner']
    return compile(tree,'candidate_skill.py','exec')

def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    protocol={'seed':SEED,'source_index':str(INDEX),'source_sha256':sha(INDEX),'code_sha256':sha(__file__),'transport_sha256':sha(REPO/'experiments/hindsight_smoke_test/transport.py'),'model':'gpt-6-astra','effort':'high','workers':2,'algorithm':'fixed contact-sheet scan; max3 per batch; max10 enlarge; one induction; one generated skill; max3 heldout; no robot','screen_ranking':'possible_reveal desc, local_change desc, seeded ID hash; prefer distinct complete run groups first, then fill','case_ranking':'eligible reveal SUPPORTED/PROVISIONAL and benefit SUPPORTED/PROVISIONAL/UNKNOWN; unexpected evidence desc, reveal evidence desc, research value desc, seeded hash; strongest provisional fallback allowed, never physical success','hard_limits':{'enlarged_pairs':10,'induction_cases':1,'induction_rounds':1,'skills':1,'heldout_images':3,'api_attempts':20,'physical_actions':0},'isolation':'No prior analysis/report/skill supplied to ephemeral models; only original intent allowed after screening. Orchestrator has prior chat context; selection deterministic from blind model outputs.'}
    if (ROOT/'protocol.json').exists():assert read(ROOT/'protocol.json')==protocol,'Frozen code/protocol changed'
    else:save(ROOT/'protocol.json',protocol)
    progress('1_SPLIT_AND_CONTACT_SHEETS');rows,train,val,sheets=prepare();byid={r['id']:r for r in rows}
    progress('1_CONTACT_SCAN',pairs=len(train),pages=len(sheets));batches=[sheets[i:i+3] for i in range(0,len(sheets),3)]
    def scan(z):
        i,ss=z;x=invoke(f'scan_{i:02d}',SCAN_PROMPT,{'pages':[{'image_index':j,'valid_ids':s['ids']} for j,s in enumerate(ss)],'max_candidates':3},[s['path'] for s in ss],SCAN)
        valid={i for s in ss for i in s['ids']};assert len(x['candidates'])<=3 and all(c['id'] in valid for c in x['candidates'])
        return {'batch':i,**x}
    scans=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for x in pool.map(scan,enumerate(batches)):
            scans.append(x);save(ROOT/'screening_results.json',scans);progress('1_CONTACT_SCAN',completed=len(scans),total=len(batches))
    cs=[c for x in scans for c in x['candidates']];cs=sorted(cs,key=lambda c:(-c['possible_reveal'],-c['local_change'],key(c['id'])))
    selected=[];used=set()
    for c in cs:
        if byid[c['id']]['run'] not in used and len(selected)<10:selected.append(c);used.add(byid[c['id']]['run'])
    for c in cs:
        if c['id'] not in {x['id'] for x in selected} and len(selected)<10:selected.append(c)
    save(ROOT/'shortlist.json',selected);progress('2_DETAIL',pairs=len(selected))
    rs=[byid[c['id']] for c in selected];details=[]
    for i in range(0,len(rs),5):
        batch=rs[i:i+5];x=invoke(f'detail_{i//5:02d}',DETAIL_PROMPT,packed(batch),pair_images(batch),DETAIL)
        assert {r['id'] for r in x['cases']}=={r['id'] for r in batch} and len(x['cases'])==len(batch)
        details+=x['cases'];save(ROOT/'detailed_effects.json',details);progress('2_DETAIL',completed=len(details),total=len(rs))
    eligible=[c for c in details if c['reveal'] in ['SUPPORTED','PROVISIONAL'] and c['unplanned_benefit'] in ['SUPPORTED','PROVISIONAL','UNKNOWN']]
    eligible.sort(key=lambda c:(-c['unexpected_evidence'],-c['reveal_evidence'],-c['research_value'],key(c['id'])))
    summary={'historical_pairs':len(rows),'discovery_pairs':len(train),'contact_pages':len(sheets),'screen_candidates':len(cs),'enlarged_pairs':len(rs),'test_runs':len(val),'physical_actions':0,'human_case_selection':False}
    if not eligible:
        save(ROOT/'selected_case.json',{'status':'NO_DEFENSIBLE_PROVISIONAL_CANDIDATE'});summary['status']='NO_VALID_DISCOVERY';finish(summary,rs);return
    chosen=eligible[0];r=byid[chosen['id']];save(ROOT/'selected_case.json',{'status':'PROVISIONAL','case':chosen,'source':r['source'],'selection_rule':protocol['case_ranking'],'physical_success_confirmed':False})
    progress('3_4_HINDSIGHT_AND_INDUCTION',selected=r['id']);hyp=invoke('induction',INDUCT_PROMPT,{**packed([r]),'prior_observation':chosen,'evidence_status':'PROVISIONAL, no independent material ground truth'},pair_images([r]),INDUCTION)
    save(ROOT/'skill_hypotheses.json',hyp);summary['selected_id']=r['id'];summary['induction']=hyp['status']
    if hyp['status']!='EXPERIMENTAL_HYPOTHESIS':summary['status']='INSUFFICIENT_HYPOTHESIS';finish(summary,rs);return
    assert len(hyp['hypotheses'])>=2
    progress('5_SKILL_SYNTHESIS');skill=invoke('skill_synthesis',SYNTH_PROMPT,{'hypothesis':hyp,'output_schema':ACTION,'interface':'numpy RGB -> action dict; supplied VISION_MODEL has current-image-only access','evidence_status':'PROVISIONAL'},[],SYNTHESIS)
    save(ROOT/'skill_synthesis.json',skill)
    if skill['status']!='EXPERIMENTAL_SKILL':summary['status']='HYPOTHESIS_ONLY';finish(summary,rs);return
    (ROOT/'candidate_skill.py').write_text(skill['python_code']);(ROOT/'inference_prompt.txt').write_text(skill['inference_prompt']);compiled=validate_code(skill['python_code'])
    freeze={'code_sha256':sha(ROOT/'candidate_skill.py'),'prompt_sha256':sha(ROOT/'inference_prompt.txt'),'time':time.time(),'validation_images_seen_before_freeze':0};save(ROOT/'skill_freeze.json',freeze)
    progress('6_HELDOUT_DRY_RUN',images=len(val));results=[]
    for i,v in enumerate(val):
        assert v['run'] not in {x['run'] for x in train};p=REPO/v['before'];assert sha(p)==v['before_sha256']
        rgb=np.asarray(Image.open(p).convert('RGB'));calls=[]
        def vision(image,prompt):
            assert not calls,'Only one VLM call per validation image';assert np.array_equal(image,rgb),'Skill changed validation RGB';calls.append(True)
            return invoke(f'validation_{i:02d}','仅依据当前RGB和冻结的实验性技能做一次离线动作推理。未见After，不读取历史动作。不是物理成功或安全证明。保留不确定性。',{'frozen_skill_instruction':prompt,'image_size':[rgb.shape[1],rgb.shape[0]],'target':'AUTO','physical_execution':False},[p],ACTION)
        env={'VISION_MODEL':vision,'FROZEN_PROMPT':skill['inference_prompt']};exec(compiled,env);start=time.time();a=env['generate_action'](rgb,'AUTO')
        import jsonschema
        jsonschema.validate(a,ACTION);assert len(calls)==1
        w,h=rgb.shape[1],rgb.shape[0];valid=all(0<=q[0]<w and 0<=q[1]<h for q in [a['grasp_px'],a['target_px']]);im=Image.fromarray(rgb);d=ImageDraw.Draw(im)
        if a['decision']=='ACTION' and valid:
            pa,pb=a['grasp_px'],a['target_px'];d.line([tuple(pa),tuple(pb)],fill='yellow',width=5)
            for q,c,t in [(pa,'lime','Pa'),(pb,'red','Pb')]:d.ellipse((q[0]-7,q[1]-7,q[0]+7,q[1]+7),fill=c);d.text((q[0]+9,q[1]),t,fill=c)
        out=ROOT/'validation'/v['id'];out.mkdir(parents=True,exist_ok=True);im.save(out/'overlay.png');shutil.copyfile(p,out/'before.png')
        results.append({'id':v['id'],'run':v['run'],'before':v['before'],'action':a,'pixels_in_image':valid,'distance_px':math.dist(a['grasp_px'],a['target_px']),'latency_s':time.time()-start,'host_feasibility':'NOT_TESTED_NO_ROBOT_OR_CALIBRATION','physical_success':'NOT_TESTED','after_or_recorded_action_supplied':False})
        save(ROOT/'validation_results.json',results);progress('6_HELDOUT_DRY_RUN',completed=len(results),total=len(val))
    assert freeze['code_sha256']==sha(ROOT/'candidate_skill.py') and freeze['prompt_sha256']==sha(ROOT/'inference_prompt.txt')
    summary.update(status='END_TO_END_PROVISIONAL_SKILL_TESTED',validation_images=len(results),actions=sum(v['action']['decision']=='ACTION' for v in results),abstentions=sum(v['action']['decision']=='ABSTAIN' for v in results));finish(summary,rs)

def finish(summary,rs):
    calls=[read(p) for p in (ROOT/'calls').glob('*/result.json')];starts=[read(p)['time'] for p in (ROOT/'calls').glob('*/attempt_*/started.json')]
    summary.update(model_calls=len(starts),api_failed=sum(c['status']!='COMPLETE' for c in calls),wall_time_s=time.time()-min(starts) if starts else 0,usage=[c.get('usage') for c in calls],isolation='Fresh ephemeral sessions, no old discovery reports/skills. Orchestrator retains chat context; no manual case override. Historical behavior may reflect expert policy. Validation complete runs excluded before image scanning.',claim='Pipeline execution and candidate behavior only, no demonstrated physical skill benefit.')
    save(ROOT/'summary.json',summary)
    body='<h1>Hindsight Skill Discovery — End-to-End Fast Test</h1><p>PROVISIONAL 实验性发现；无机器人执行，未验证物理技能收益。</p><pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre>'
    details={c['id']:c for c in read(ROOT/'detailed_effects.json')} if (ROOT/'detailed_effects.json').exists() else {}
    for r in rs:
        d=ROOT/'evidence'/r['id'];d.mkdir(parents=True,exist_ok=True)
        for k in ['before','after']:shutil.copyfile(REPO/r[k],d/(k+'.png'))
        body+='<h2>'+r['id']+'</h2><p>'+html.escape(r['source'])+'</p><div class="pair">'+''.join('<a href="evidence/'+r['id']+'/'+k+'.png"><img src="evidence/'+r['id']+'/'+k+'.png"></a>' for k in ['before','after'])+'</div><pre>'+html.escape(json.dumps({'original_intention':intention(r),'actual_action':r['action'],'analysis':details.get(r['id'])},ensure_ascii=False,indent=2))+'</pre>'
    for f,t in [('selected_case.json','自动选择'),('skill_hypotheses.json','目标重标记、竞争机制与几何规律'),('skill_synthesis.json','候选Skill及代码'),('validation_results.json','新图动作与理由')]:
        if (ROOT/f).exists():body+='<h2>'+t+'</h2><pre>'+html.escape(json.dumps(read(ROOT/f),ensure_ascii=False,indent=2))+'</pre>'
    for p in sorted((ROOT/'validation').glob('*/overlay.png')):body+='<h3>'+p.parent.name+'</h3><a href="'+str(p.relative_to(ROOT))+'"><img width="100%" src="'+str(p.relative_to(ROOT))+'"></a>'
    body+='<h2>证据边界</h2><p>人工没有指定案例、解释机制或修改候选。自动流程采用预先冻结规则。模型观察不是独立物理真值；无过程接触证据不能确定抓持布层。像素有效不等于Host可执行。Dry-run不能计为Corner Reveal成功。</p><p><a href="protocol.json">冻结协议</a> · <a href="split.json">整run隔离</a> · <a href="screening_results.json">全部初筛输出</a> · <a href="shortlist.json">放大候选选择</a> · <a href="calls/">原始模型请求</a></p>'
    (ROOT/'review.html').write_text('<!doctype html><meta charset="utf-8"><title>Hindsight Fast Test</title><style>body{font:18px/1.6 sans-serif;max-width:1400px;margin:30px auto;padding:20px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eee;padding:16px}.pair{display:flex}.pair a{width:50%}.pair img{width:100%}</style>'+body)
    progress('COMPLETE',status=summary['status'])

if __name__=='__main__':
    try:main()
    except Exception as e:
        progress('ERROR',error=type(e).__name__+': '+str(e));raise
