"""Queue-only, goal-locked offline continuation. No robot or production imports."""
from pathlib import Path
import sys,json,time,hashlib,concurrent.futures,ast,math,shutil,html
import numpy as np
from PIL import Image,ImageDraw
import jsonschema
CODE=Path(__file__).resolve().parent;REPO=CODE.parents[1];sys.path.insert(0,str(CODE))
from transport import call,save,read,sha
OUT=REPO/'reports/hindsight_corner_continuation_20261030'
QUEUE=REPO/'reports/effect_screen_page05_diagnostic_20261029/candidate_queue.json'
PAGE_GROUPS=QUEUE.parent/'groups.json'
DATA=REPO/'reports/hindsight_fast_test_20261029'
GOALS=['Reveal(LS)','Reveal(RS)','Reveal(LH)','Reveal(RH)']
SEED='corner-continuation-20261030-v0'
S={'type':'string'};B={'type':'boolean'};R={'type':'integer','minimum':0,'maximum':3}
def enum(*v):return {'type':'string','enum':list(v)}
def obj(v):return {'type':'object','properties':v,'required':list(v),'additionalProperties':False}
def arr(v):return {'type':'array','items':v}
POINT={'type':'array','items':{'type':'integer'},'minItems':2,'maxItems':2}
REVIEW=obj({'id':S,'status':enum('CONFIRMED','PROVISIONAL','UNCERTAIN','NO_REVEAL'),'goal':{'type':['string','null'],'enum':GOALS+[None]},'semantic_endpoint_basis':S,'semantic_side_basis':S,'before_state':enum('OCCLUDED','SUSPECTED_OCCLUDED','VISIBLE','OUT_OF_VIEW','UNKNOWN'),'after_state':enum('EXPOSED','PARTIALLY_EXPOSED','NOT_EXPOSED','UNKNOWN'),'before_evidence':S,'after_evidence':S,'correspondence':S,'counterevidence':S,'before_score':R,'identity_score':R,'transition_score':R,'semantic_side_score':R,'uncertainty':S,'review_reason':S})
INTENT=obj({'id':S,'intended_manipulation':S,'actual_executed_command':S,'actual_contact_and_material':S,'observed_effect':S,'expected_changes_supported':S,'expected_changes_not_established':S,'unplanned_benefit':enum('SUPPORTED','PROVISIONAL','NOT_ESTABLISHED','NONE'),'unplanned_benefit_evidence':S,'intention_outcome_mismatch':S,'observation_vs_explanation':S,'uncertainty':S})
def hypothesis_schema(goal):return obj({'status':enum('EXPERIMENTAL_HYPOTHESIS','INSUFFICIENT_EVIDENCE'),'goal':enum(goal),'goal_description':S,'success_criterion':S,'goal_relevance':S,'hypotheses':arr(obj({'name':S,'physical_explanation':S,'observed_support':S,'unobserved_assumptions':S,'different_prediction':S,'falsifier':S})),'observable_geometry':arr(S),'action_generation_rules':arr(S),'applicability':S,'abstention_conditions':arr(S),'engineering_assumptions':arr(S),'uncertainty':S})
def action_schema(goal):return obj({'decision':enum('ACTION','ABSTAIN'),'goal':enum(goal),'target_state':enum('OCCLUDED','SUSPECTED_OCCLUDED','VISIBLE','OUT_OF_VIEW','UNKNOWN'),'target_identity_basis':S,'target_occlusion_evidence':S,'observable_geometry':S,'grasp_region':S,'grasp_px':POINT,'target_px':POINT,'grasp_angle_deg':{'type':'integer','minimum':0,'maximum':180},'direction_reason':S,'expected_target_change':S,'success_criterion':S,'falsifier':S,'uncertainty':S,'abstain_reason':S,'experimental_only':B})
def synth_schema(goal):return obj({'status':enum('EXPERIMENTAL_SKILL','HYPOTHESIS_ONLY'),'goal':enum(goal),'python_code':S,'inference_prompt':S,'algorithm_explanation':S,'engineering_assumptions':arr(S),'limitations':S})
REVIEW_PROMPT='''你是隔离会话中的真实前后图审核者。只看这一对完整分辨率RGB，左输入Before、第二张After；没有原始意图、动作或先前筛选结论。查看四个真实语义端点是否从衣物自遮挡到显露：Reveal(LS/RS)为穿着者左/右袖口开口边缘中点；Reveal(LH/RH)为穿着者左/右真实下摆边与侧缝交角。左右按衣服语义结构，说明朝向依据，不直接按图像左右命名。
报告最有证据的一个目标；无法给出有依据的四类目标则goal=null。CONFIRMED表示这一对RGB支持较强视觉确认，不是独立材料真值；PROVISIONAL表示有具体遮挡和显露线索，但材料对应或侧别等仍有明确不确定性；UNCERTAIN表示线索不足；NO_REVEAL表示没有目标转换，或相关端点原本已经可见。不能把局部面积增加、普通轮廓扩张、缝边变长、标签出现单独当Corner Reveal。不要因语义身份尚不完全确定就否定有具体端点证据的暂定候选，但也不能任意指定身份。
分别提供Before自遮挡证据、After端点身份依据和前后对应。评分0=无、1=弱、2=有具体但暂定的支持、3=较强视觉支持。可进入归纳至少需要具体允许目标，Before为OCCLUDED或SUSPECTED_OCCLUDED、After为EXPOSED，遮挡/身份/转换/侧别评分均>=2。CONFIRMED应有四项3分，其余有意义候选可PROVISIONAL。不考虑是否意外，不推测原意图，不提出抓点规则或技能。中文简洁回答。'''
INTENT_PROMPT='''你在全新隔离会话中分析一个自动选择的真实案例。只依据提供的原始记录、Before/After及可选过程帧，严格区分Intended Manipulation、Actual Manipulation、Observed Effect。原始语义判断只是信念，不能当材料事实；意图缺失写UNKNOWN，不能从After补写意图。执行完成不等于夹持成功；实际接触、夹持层和材料身份看不到就UNKNOWN。过程帧按文件序列采样，可能有腕载相机视角变化，不推测未提供的精确阶段。说明符合预期、未证实预期、意外有益变化。即使效应属于计划内，仍可作为目标学习经验；不要为追求意外收益改写规划。此步只分析事实与意图，不提出技能或规则。'''
INDUCT_PROMPT='''从所给一个真实案例及原始意图/效果分析中归纳一个待验证的Corner Reveal技能。唯一目标已固定在payload.locked_goal；目标描述、成功判据与规则都必须始终针对这一真实语义端点从自遮挡到可辨认显露。不能替换成布片展开、缝边变长、标签显露或一般信息增加。若原来就是为了同一目标，则明确这是目标保持的经验，而非意外收益发现；这不阻止提出假设。
先解释实际效果，再提出至少两个竞争性物理解释，分清直接观察与隐藏层序等假设，并给出不同预测和反证。自主寻找当前RGB可观察的几何关系，形成抓取区域、方向及位移生成原则。不预设正确抓取侧、固定图像方向、几何算法或操作顺序；不复用历史坐标或run身份。
证据不足以证明可靠技能时，可以形成有明确适用条件的实验性规则；不必假装所有机制已区分。动作可以牵动可见布料间接影响目标，但必须说明目标端点预计怎样显露及什么结果推翻假设。若连有意义的目标假设都不成立则INSUFFICIENT_EVIDENCE，不更换目标。中文输出。'''
SYNTH_PROMPT='''只根据本轮提供的目标锁定假设，写一个候选Python函数 generate_action(rgb, target_corner)。rgb是numpy RGB数组，target_corner是指定的LS/RS/LH/RH；本轮只对locked_goal对应目标进行测试，其他目标可弃权。全局注入LOCKED_GOAL、FROZEN_PROMPT、VISION_MODEL(rgb,prompt)，视觉模型单次读取当前图，返回给定固定Schema动作字典。
函数可用numpy、math、cv2，最多调用一次VISION_MODEL；不得访问文件、网络、环境、历史数据或机器人，不允许反射、eval/exec。只允许import numpy as np、import math、import cv2。可以选择VLM驱动实现，不应伪称已实现未编写的CV分割/测量。保持代码简洁并校验目标和像素坐标。
冻结推理Prompt只能携带本轮归纳规则、适用条件和指定语义目标，不能携带原始规划意图作为当前目标，不能包含历史坐标、案例编号或原始案例定位信息。不得为了输出ACTION自动改选其他端点或替换为一般边界可见性。当前指定端点已可见应ABSTAIN，不将额外舒展当Reveal；证据不足或目标不适用可ABSTAIN并说明。存在合理探索动作而机制未确定时可输出实验性ACTION，但仍须有目标遮挡和材料身份的具体依据。
所有ACTION只为未来物理检验，experimental_only=true。推理必须输出目标状态、身份/遮挡证据、Pa/Pb、方向理由、预期目标显露、可见成功判据和反证；不要猜毫米标定，不声称通过Host安全检查。保持锁定目标不变，若不能实现就HYPOTHESIS_ONLY，不编造成功。'''

def key(s):return hashlib.sha256((SEED+'|'+s).encode()).hexdigest()
def progress(stage,**kw):
    x={'stage':stage,'time':time.time(),**kw};save(OUT/'progress.json',x);print(json.dumps(x),flush=True)
def require(name,prompt,payload,images,schema):
    x=call(OUT,name,prompt,payload,[Path(p) for p in images],schema)
    if x['status']!='COMPLETE':raise RuntimeError(name+': '+str(x.get('error')))
    return x['output']
def eligible(x):return x['status'] in ['CONFIRMED','PROVISIONAL'] and x['goal'] in GOALS and x['before_state'] in ['OCCLUDED','SUSPECTED_OCCLUDED'] and x['after_state']=='EXPOSED' and min(x[k] for k in ['before_score','identity_score','transition_score','semantic_side_score'])>=2
def rank(x):
    strong=x['status']=='CONFIRMED' and min(x[k] for k in ['before_score','identity_score','transition_score','semantic_side_score'])==3
    return (-int(strong),-min(x[k] for k in ['before_score','identity_score','transition_score','semantic_side_score']),-sum(x[k] for k in ['before_score','identity_score','transition_score','semantic_side_score']),key(x['id']))
def code_check(code):
    t=ast.parse(code);bad={'open','exec','eval','compile','getattr','setattr','globals','locals','__import__','input','breakpoint'}
    for n in ast.walk(t):
        if isinstance(n,(ast.ImportFrom,ast.With,ast.AsyncWith,ast.ClassDef,ast.Global,ast.Nonlocal,ast.While,ast.AsyncFunctionDef)):raise ValueError('disallowed syntax '+type(n).__name__)
        if isinstance(n,ast.Import) and any(a.name not in ['numpy','math','cv2'] for a in n.names):raise ValueError('nonallowlisted import')
        if isinstance(n,ast.Name) and (n.id in bad or n.id.startswith('__')):raise ValueError('disallowed name')
        if isinstance(n,ast.Attribute) and (n.attr.startswith('_') or any(w in n.attr.lower() for w in ['load','save','file','ctypes','dump','memmap','imread','imwrite','capture','writer'])):raise ValueError('disallowed attribute')
    fs=[n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='generate_action'];assert len(fs)==1 and [a.arg for a in fs[0].args.args]==['rgb','target_corner']
    return compile(t,'candidate_skill.py','exec')

def prepare():
    queue=read(QUEUE);want=set(queue['possible_reveal']+queue['uncertain_for_optional_enlargement']);assert len(want)==17
    order=[c['id'] for g in read(PAGE_GROUPS) for c in g['cells'] if c['id'] in want]
    idx=read(DATA/'experience_index.json');byid={x['id']:x for x in idx};rows=[{**byid[id],'original_id':id,'id':f'C{i+1:02d}'} for i,id in enumerate(order)]
    split=read(DATA/'split.json');old_hashes={x['before_sha256'] for x in split['validation']};validation=[]
    for run in split['heldout_complete_runs']:
        choices=[x for x in idx if x['run']==run and x['before_sha256'] not in old_hashes]
        r=sorted(choices,key=lambda x:key(x['before_sha256']))[0];validation.append({'id':f'T{len(validation)+1:02d}','run':run,'before':r['before'],'before_sha256':r['before_sha256']})
    assert not ({r['run'] for r in rows}&{v['run'] for v in validation})
    assert not ({r[k+'_sha256'] for r in rows for k in ['before','after']}&{v['before_sha256'] for v in validation})
    save(OUT/'input_manifest.json',{'queue_sha256':sha(QUEUE),'index_sha256':sha(DATA/'experience_index.json'),'mapping':rows,'validation_before_only':validation,'historical_case_aliases_sent':False,'old_reviews_or_skills_read':False,'previous_validation_images_excluded':True})
    return rows,validation

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    protocol={'seed':SEED,'algorithm':'review17 without intentions; fixed corner-evidence ranking; analyze1 intent/execution; induce1 locked goal; synthesize1; validate<=3 RGBs','limits':{'review_pairs':17,'selected_cases':1,'skills':1,'test_images':3,'normal_requests':23,'max_requests':26,'workers':2,'request_timeout_s':900,'robot_actions':0},'goals':GOALS,'ranking':'eligible CONFIRMED(all4 scores3) before PROVISIONAL; minimum evidence score; score sum; seeded hash tie-break. No intention/surprise preference.','test_selection':'fixed seed, one unused Before from each previous complete held-out run; exclude previous3 test images, no After/action input','confirmation_scope':'CONFIRMED is model visual confirmation only; no independent material ground truth','code_sha256':sha(__file__),'transport_sha256':sha(CODE/'transport.py'),'prompts_sha256':hashlib.sha256((REVIEW_PROMPT+INTENT_PROMPT+INDUCT_PROMPT+SYNTH_PROMPT).encode()).hexdigest(),'source_queue_sha256':sha(QUEUE),'no_new_scan':True,'no_production_changes':True}
    if (OUT/'protocol.json').exists():assert read(OUT/'protocol.json')==protocol
    else:save(OUT/'protocol.json',protocol)
    rows,validation=prepare();byid={r['id']:r for r in rows};reviews=[];failures=[];progress('1_REVIEW_FULL_RESOLUTION',total=17)
    def review(r):
        for k in ['before','after']:assert sha(REPO/r[k])==r[k+'_sha256']
        result=call(OUT,'review_'+r['id'],REVIEW_PROMPT,{'id':r['id'],'image_order':['Before','After'],'physical_transition_record':True},[REPO/r[k] for k in ['before','after']],REVIEW)
        return r['id'],result
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        fs=[pool.submit(review,r) for r in rows]
        for f in concurrent.futures.as_completed(fs):
            id,result=f.result()
            if result['status']=='COMPLETE':assert result['output']['id']==id;reviews.append(result['output'])
            else:failures.append({'id':id,'error':result.get('error')})
            save(OUT/'reviews.json',sorted(reviews,key=lambda x:x['id']));save(OUT/'failures.json',failures);progress('1_REVIEW_FULL_RESOLUTION',completed=len(reviews),failed=len(failures),total=17)
    # At most three transport retries, with identical inputs and new preserved call names.
    for failure in list(failures)[:3]:
        r=byid[failure['id']];result=call(OUT,'retry_review_'+r['id'],REVIEW_PROMPT,{'id':r['id'],'image_order':['Before','After'],'physical_transition_record':True},[REPO/r[k] for k in ['before','after']],REVIEW)
        if result['status']=='COMPLETE':assert result['output']['id']==r['id'];reviews.append(result['output']);failures.remove(failure)
    save(OUT/'reviews.json',sorted(reviews,key=lambda x:x['id']));save(OUT/'failures.json',failures)
    candidates=sorted([x for x in reviews if eligible(x)],key=rank);ranking={'candidates':candidates,'excluded_ids':[x['id'] for x in reviews if not eligible(x)],'unreviewed':failures,'frozen_at':time.time(),'selection_uses_intent':False};save(OUT/'ranking.json',ranking)
    summary={'reviewed':len(reviews),'requested':17,'review_status_counts':{s:sum(x['status']==s for x in reviews) for s in ['CONFIRMED','PROVISIONAL','UNCERTAIN','NO_REVEAL']},'eligible':len(candidates),'physical_actions':0,'skill_induction_rounds':0,'tests':0,'failures':failures}
    if failures or not candidates:
        summary['status']='INCOMPLETE_REVIEW' if failures else 'NO_VALID_CORNER_DISCOVERY';finish(rows,summary);return
    chosen=candidates[0];r=byid[chosen['id']];goal=chosen['goal'];save(OUT/'selected_case.json',{'review':chosen,'source_id':r['original_id'],'source':r['source'],'goal':goal,'rank':1,'selection_rule':protocol['ranking'],'physical_success_confirmed':False});summary.update(selected=r['id'],goal=goal)
    source=REPO/r['source'];plan=read(source/'dispatch_intent.json').get('plan',{})
    original={k:plan[k] for k in ['short_reason','purpose','expected_effect','expected_stop_condition','reason','rationale','target_region','is_smoothing'] if k in plan}
    frames=sorted((source/'execution_images').glob('*.jpg'));picked=[]
    if frames:
        picked=[frames[i] for i in sorted({round((len(frames)-1)*q) for q in [.25,.5,.75]})]
    native=read(source/'native_plan.json') if (source/'native_plan.json').exists() else []
    save(OUT/'selected_native_operations_full.json',native)
    compact_native=[{**op,'data':{**{k:v for k,v in op.get('data',{}).items() if k!='queued_segments'},'queued_segment_count':len(op.get('data',{}).get('queued_segments',[]))}} for op in native]
    evidence={'id':r['id'],'original_intention':original or 'UNKNOWN','dispatched_plan':{k:plan[k] for k in ['decision','grasp_px','target_px','motion_type','grasp_angle_deg'] if k in plan},'executed_action':r['action'],'execution_status':r['execution'],'native_operations':compact_native,'gripper_feedback':{p.name:read(p) for p in source.glob('gripper_*.json')},'image_order':['Before','After']+[f'Execution sequence sample {i+1}: {p.name}; no phase assumption' for i,p in enumerate(picked)]}
    save(OUT/'selected_raw_evidence.json',evidence);save(OUT/'process_frame_manifest.json',[{'source':str(p),'sha256':sha(p)} for p in picked]);progress('2_INTENTION_ACTUAL_EFFECT',case=r['id'])
    analysis=require('intention_analysis',INTENT_PROMPT,evidence,[REPO/r['before'],REPO/r['after'],*picked],INTENT);save(OUT/'intention_analysis.json',analysis)
    progress('3_4_GOAL_RELABEL_INDUCTION',goal=goal)
    hyp=require('induction',INDUCT_PROMPT,{'locked_goal':goal,'case_review':chosen,'intention_actual_effect':analysis,'executed_action':r['action'],'image_order':['Before','After'],'confirmation_scope':'model visual evidence, not independently labeled material truth'},[REPO/r['before'],REPO/r['after']],hypothesis_schema(goal));save(OUT/'skill_hypotheses.json',hyp);summary['skill_induction_rounds']=1
    if hyp['status']!='EXPERIMENTAL_HYPOTHESIS':summary['status']='GOAL_LOCKED_HYPOTHESIS_INSUFFICIENT';finish(rows,summary);return
    assert len(hyp['hypotheses'])>=2
    progress('5_SKILL_SYNTHESIS',goal=goal)
    spec=require('synthesis',SYNTH_PROMPT,{'locked_goal':goal,'hypothesis':hyp,'action_schema':action_schema(goal),'test_interface':'generate_action(numpy_rgb, target_corner); one injected VISION_MODEL call'},[],synth_schema(goal));save(OUT/'skill_synthesis.json',spec)
    if spec['status']!='EXPERIMENTAL_SKILL':summary['status']='GOAL_LOCKED_HYPOTHESIS_ONLY';finish(rows,summary);return
    code=spec['python_code'];compiled=code_check(code);(OUT/'candidate_skill.py').write_text(code);(OUT/'inference_prompt.txt').write_text(spec['inference_prompt'])
    freeze={'time':time.time(),'goal':goal,'code_sha256':sha(OUT/'candidate_skill.py'),'prompt_sha256':sha(OUT/'inference_prompt.txt'),'test_images_seen':0};save(OUT/'skill_freeze.json',freeze);progress('6_NEW_RGB_DRY_RUN',total=len(validation));results=[]
    for i,v in enumerate(validation):
        p=REPO/v['before'];assert sha(p)==v['before_sha256'];rgb=np.asarray(Image.open(p).convert('RGB'));count=[]
        def vision(image,prompt):
            assert not count and np.array_equal(image,rgb);count.append(True)
            return require(f'test_{i+1:02d}','只用当前RGB和冻结技能提出一次离线动作。保持指定语义端点恢复目标；不读取历史图片、After或原始动作。不是物理验证。',{'locked_goal':goal,'target_corner':goal[7:-1],'skill_prompt':prompt,'image_size':[rgb.shape[1],rgb.shape[0]]},[p],action_schema(goal))
        env={'VISION_MODEL':vision,'FROZEN_PROMPT':spec['inference_prompt'],'LOCKED_GOAL':goal};exec(compiled,env);start=time.time();a=env['generate_action'](rgb,goal[7:-1]);jsonschema.validate(a,action_schema(goal));assert len(count)<=1
        w,h=rgb.shape[1],rgb.shape[0];px=all(0<=q[0]<w and 0<=q[1]<h for q in [a['grasp_px'],a['target_px']]);violations=[]
        if a['decision']=='ACTION' and a['target_state'] not in ['OCCLUDED','SUSPECTED_OCCLUDED']:violations.append('ACTION without supported hidden-target state')
        if a['decision']=='ACTION' and (not px or a['grasp_px']==a['target_px']):violations.append('invalid or zero image displacement')
        if not a['experimental_only']:violations.append('experimental-only marker missing')
        d=OUT/'validation'/v['id'];d.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,d/'before.png');im=Image.fromarray(rgb);draw=ImageDraw.Draw(im)
        if a['decision']=='ACTION' and px:
            pa,pb=a['grasp_px'],a['target_px'];draw.line([tuple(pa),tuple(pb)],fill='yellow',width=5)
            angle=math.atan2(pb[1]-pa[1],pb[0]-pa[0]);draw.polygon([tuple(pb),(pb[0]-14*math.cos(angle-.45),pb[1]-14*math.sin(angle-.45)),(pb[0]-14*math.cos(angle+.45),pb[1]-14*math.sin(angle+.45))],fill='yellow')
            for q,c,t in [(pa,'lime','Pa'),(pb,'red','Pb')]:draw.ellipse((q[0]-7,q[1]-7,q[0]+7,q[1]+7),fill=c);draw.text((q[0]+9,q[1]),t,fill=c)
        im.save(d/'overlay.png');results.append({'id':v['id'],'run':v['run'],'before':v['before'],'action':a,'goal_contract_violations':violations,'pixels_valid':px,'latency_s':time.time()-start,'physical_success':'NOT_TESTED','host_safety':'NOT_TESTED','after_or_original_action_supplied':False});save(OUT/'validation_results.json',results);progress('6_NEW_RGB_DRY_RUN',completed=len(results),total=len(validation))
    assert freeze['code_sha256']==sha(OUT/'candidate_skill.py') and freeze['prompt_sha256']==sha(OUT/'inference_prompt.txt')
    summary.update(status='GOAL_LOCKED_PIPELINE_COMPLETED',tests=len(results),actions=sum(x['action']['decision']=='ACTION' for x in results),abstentions=sum(x['action']['decision']=='ABSTAIN' for x in results),goal_contract_violations=sum(len(x['goal_contract_violations']) for x in results));finish(rows,summary)

def finish(rows,summary):
    calls=[(p.parent.name,read(p)) for p in sorted((OUT/'calls').glob('*/result.json'))];starts=[read(p)['time'] for p in (OUT/'calls').glob('*/attempt_*/started.json')]
    summary.update(model_calls=len(starts),failed_requests=sum(c['status']!='COMPLETE' for _,c in calls),wall_time_s=time.time()-min(starts) if starts else 0,confirmation_scope='model visual confirmation only, no independent material truth or physical validation',isolation='Fresh model contexts; only queue membership, raw pairs, selected raw intentions/execution and current-round generated hypotheses. No prior skill reports or known-success identities. Orchestrator retains chat context. Known-page diagnostic continuation, not blind discovery or pooled discovery-rate evaluation.')
    save(OUT/'summary.json',summary)
    esc=lambda x:html.escape(str(x));dump=lambda x:'<pre>'+esc(json.dumps(x,ensure_ascii=False,indent=2))+'</pre>'
    body=['<h1>Hindsight Skill Discovery · 语义角点目标锁定续实验</h1><p>从现有17例队列继续；已知页面诊断延续，不与全量发现率混算。无机器人执行，所有生成动作仅为待验证假设。</p>',dump(summary)]
    reviews={x['id']:x for x in read(OUT/'reviews.json')}
    body.append('<h2>1. 全部17对完整分辨率审核</h2><table><tr><th>ID</th><th>状态</th><th>目标</th><th>证据</th></tr>')
    for r in rows:
        x=reviews.get(r['id'],{});body.append('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in [r['id'],x.get('status','API_FAILED'),x.get('goal'),x.get('review_reason')])+'</tr>')
    body.append('</table>')
    for r in rows:
        d=OUT/'evidence'/r['id'];d.mkdir(parents=True,exist_ok=True)
        for k in ['before','after']:shutil.copyfile(REPO/r[k],d/(k+'.png'))
        body.append('<details><summary>'+r['id']+' · '+esc(r['original_id'])+' · '+esc(r['source'])+'</summary><div class="pair">'+''.join('<figure><a href="evidence/'+r['id']+'/'+k+'.png"><img src="evidence/'+r['id']+'/'+k+'.png"></a><figcaption>'+k+'</figcaption></figure>' for k in ['before','after'])+'</div>'+dump(reviews.get(r['id']))+'</details>')
    for f,t in [('ranking.json','2. 自动排序'),('selected_case.json','3. 自动选择'),('selected_raw_evidence.json','4. 原始意图与执行记录'),('intention_analysis.json','5. Intended / Actual / Observed'),('skill_hypotheses.json','6. 目标重标记、机制与几何规则'),('skill_synthesis.json','7. 生成的Skill与冻结Prompt'),('validation_results.json','8. 预留新RGB测试')]:
        if (OUT/f).exists():body+=['<h2>'+t+'</h2>',dump(read(OUT/f))]
    for p in sorted((OUT/'validation').glob('*/overlay.png')):body+=['<h3>'+p.parent.name+'</h3><a href="'+str(p.relative_to(OUT))+'"><img class="full" src="'+str(p.relative_to(OUT))+'"></a>']
    body.append('<h2>调用成本与审计</h2><table><tr><th>调用</th><th>状态</th><th>秒</th><th>Token</th></tr>')
    for name,c in calls:body.append('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in [name,c['status'],round(c.get('latency_s',0),1),c.get('usage')])+'</tr>')
    body.append('</table><p><a href="protocol.json">冻结协议</a> · <a href="input_manifest.json">数据划分与来源</a> · <a href="reviews.json">完整审核JSON</a> · <a href="ranking.json">排名</a> · <a href="summary.json">总结</a></p>')
    if (OUT/'candidate_skill.py').exists():body.append('<p><a href="candidate_skill.py">Python Skill</a> · <a href="inference_prompt.txt">冻结Prompt</a> · <a href="skill_freeze.json">冻结记录</a></p>')
    (OUT/'review.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>Corner continuation</title><style>body{font:17px/1.6 system-ui;max-width:1450px;margin:30px auto;padding:20px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eef1f5;padding:16px}td,th{border:1px solid #ccc;padding:8px}table{border-collapse:collapse;width:100%}.pair{display:flex}.pair figure{width:50%;margin:8px}.pair img,.full{width:100%}details{padding:12px}summary{cursor:pointer}</style>'+''.join(body)+'</html>');progress('COMPLETE',status=summary['status'])
if __name__=='__main__':
    try:main()
    except Exception as e:progress('ERROR',error=type(e).__name__+': '+str(e));raise
