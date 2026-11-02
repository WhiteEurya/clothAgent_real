"""Re-audit exactly ten previously enlarged pairs. No skill or robot execution.
Coverage audit for the historical named event runs only after ranking is frozen.
"""
import sys,json,time,hashlib,concurrent.futures,html,shutil
from pathlib import Path
CODE=Path(__file__).resolve().parent;REPO=CODE.parents[1]
sys.path.insert(0,str(REPO/'experiments/hindsight_smoke_test'))
from transport import call,read,save,sha
SOURCE=REPO/'reports/hindsight_fast_test_20261029'
OUT=REPO/'reports/hindsight_corner_goal_audit_20261029'
GOALS=['Reveal(LS)','Reveal(RS)','Reveal(LH)','Reveal(RH)']
S={'type':'string'};B={'type':'boolean'};R={'type':'integer','minimum':0,'maximum':3}
def enum(*a):return {'type':'string','enum':list(a)}
def obj(d):return {'type':'object','properties':d,'required':list(d),'additionalProperties':False}
SCHEMA=obj({'id':S,'candidate_goal':{'type':['string','null'],'enum':GOALS+[None]},'corner_relevance':enum('SPECIFIC_ENDPOINT_TRANSITION','ENDPOINT_ALREADY_VISIBLE','GENERIC_CHANGE_ONLY','UNRESOLVED_IDENTITY'),'semantic_identity_basis':S,'semantic_left_right_basis':S,'before_state':enum('OCCLUDED','SUSPECTED_OCCLUDED','VISIBLE','OUT_OF_VIEW','UNKNOWN'),'before_occlusion_evidence':S,'after_state':enum('EXPOSED','PARTIALLY_EXPOSED','NOT_EXPOSED','UNKNOWN'),'after_endpoint_evidence':S,'material_correspondence':S,'counterevidence':S,'before_evidence_score':R,'after_identity_score':R,'transition_score':R,'evidence_tier':enum('STRONG_VISUAL','PROVISIONAL','WEAK_OR_NONE'),'original_intent':S,'observed_effect':S,'unexpectedness':enum('UNPLANNED','PLANNED','UNKNOWN'),'unexpectedness_reason':S,'hypothesis_value':R,'hypothesis_value_reason':S,'verdict':enum('ELIGIBLE_STRONG_VISUAL','ELIGIBLE_PROVISIONAL','INELIGIBLE','INSUFFICIENT_EVIDENCE'),'verdict_reason':S,'uncertainty':S})
PROMPT='''你是全新隔离会话中的语义端点显露证据审核者。只审核附带的一对真实Before/After、原始意图和执行动作，不生成动作或Skill。
唯一允许的候选技能目标是 Reveal(LS)、Reveal(RS)、Reveal(LH)、Reveal(RH)。LS/RS为穿着者左/右袖口开口边缘中点；LH/RH为穿着者左/右真实下摆边与侧缝交角。先观察四端部是否有具体显露线索，报告证据最强的一个；没有可辩护的目标时candidate_goal=null。不得将画面左右直接当衣服语义左右；说明朝向与左右身份依据，假设必须显式标注。
优先级严格为：是否针对具体语义端点的遮挡→显露；其视觉证据强弱；之后才看意外性和假设价值。计划内效果同样可以入选，不要求原动作失败或收益意外。
进入归纳候选至少要看到：Before有具体衣物自遮挡/折回线索，After有袖口开口或真实下摆交角的可辨认结构，并能说明对应关系。材料身份证据较强但无法完全确认时允许PROVISIONAL，必须具体写明哪一个允许目标及视觉依据。侧别仅为有依据的暂定判断可以PROVISIONAL，但不能强迫一个任意语义标签。
普通布片变宽、面积增加、一般轮廓扩张、缝边变长、标签显露，单独都不是Corner Reveal。Before已经可见的同一端点移位/转向，不是遮挡→显露。仅从After显露反推Before一定遮挡不成立。Out-of-view与自遮挡分开。原Agent语义判断只是原始信念，不是材料真值；缺失意图写UNKNOWN，不事后改写。
评分0=无，1=弱/无法排除普通变化，2=具体而暂定的支持，3=较强视觉支持。只有具体目标转换，Before遮挡证据、After端点身份、前后转换三个评分均至少2，才可ELIGIBLE_PROVISIONAL；三项均3且语义身份较强可ELIGIBLE_STRONG_VISUAL。STRONG_VISUAL仍不是人工或物理材料真值。若判断不满足入选条件，明确排除或证据不足。
不要为了流程继续而替换目标。对所有结论保留反证和不确定性。尽量简洁，中文输出，每个文字字段1–2句。不要给抓点规则、物理机制或完整Skill。'''

def progress(stage,**kw):
    x={'stage':stage,'time':time.time(),**kw};save(OUT/'progress.json',x);print(json.dumps(x),flush=True)
def raw_intention(r):
    p=REPO/r['source']/'dispatch_intent.json';d=read(p).get('plan',{})
    fields=['short_reason','purpose','expected_effect','expected_stop_condition','reason','rationale','target_region','is_smoothing']
    return {k:d[k] for k in fields if k in d}
def eligibility(x):
    reasons=[]
    if x['candidate_goal'] not in GOALS:reasons.append('没有可辩护的四端部目标')
    if x['corner_relevance']!='SPECIFIC_ENDPOINT_TRANSITION':reasons.append('未建立具体语义端点遮挡到显露的相关性')
    if x['before_state'] not in ['OCCLUDED','SUSPECTED_OCCLUDED']:reasons.append('Before缺少自遮挡状态依据')
    if x['after_state']!='EXPOSED':reasons.append('After未支持目标端点已显露')
    if min(x['before_evidence_score'],x['after_identity_score'],x['transition_score'])<2:reasons.append('至少一项关键证据低于暂定候选要求')
    if x['verdict'] not in ['ELIGIBLE_STRONG_VISUAL','ELIGIBLE_PROVISIONAL'] or x['evidence_tier']=='WEAK_OR_NONE':reasons.append('独立审核未判定具备归纳资格')
    return not reasons,reasons
def sortkey(x):
    tier=2 if x['verdict']=='ELIGIBLE_STRONG_VISUAL' and min(x['before_evidence_score'],x['after_identity_score'],x['transition_score'])==3 else 1
    return (-tier,-min(x['before_evidence_score'],x['after_identity_score'],x['transition_score']),-sum(x[k] for k in ['before_evidence_score','after_identity_score','transition_score']),-int(x['unexpectedness']=='UNPLANNED'),-x['hypothesis_value'],x['id'])

def posthoc_coverage():
    # Deliberately invoked only after all requests finish and ranking.json exists.
    assert (OUT/'ranking.json').exists()
    source='reports/controlled_ablation_20261026/robot_v2/runs/episode_0007/round_004'
    idx=read(SOURCE/'experience_index.json');matches=[x for x in idx if x['source']==source]
    sp=read(SOURCE/'split.json');sheets=read(SOURCE/'contact_sheets.json');short=read(SOURCE/'shortlist.json');result={'audit_time':time.time(),'ranking_sha256_before_audit':sha(OUT/'ranking.json'),'historical_label':'E353','source':source,'method':'metadata only: source-path match, frozen split, contact-sheet membership, original API result status; no image reads','discovery_model_informed':False,'ranking_changed':False}
    if len(matches)!=1:result.update(position='NOT_UNIQUELY_FOUND',matches=len(matches));return result
    r=matches[0];result.update(fast_test_id=r['id'],run_group=r['run'],before_sha256=r['before_sha256'],after_sha256=r['after_sha256'],in_ten_enlarged=any(x['id']==r['id'] for x in short))
    if r['run'] in sp['heldout_complete_runs']:result['position']='HELDOUT_RUN'
    else:
        pages=[i for i,s in enumerate(sheets) if r['id'] in s['ids']];assert len(pages)==1
        batch=f'scan_{pages[0]//3:02d}';p=SOURCE/'calls'/batch/'result.json';d=read(p) if p.exists() else {}
        result.update(contact_sheet=sheets[pages[0]]['path'],scan_batch=batch,scan_record=str(p),scan_status=d.get('status','MISSING'),scan_error=d.get('error'))
        result['position']='REVIEWED_DISCOVERY_SET' if d.get('status')=='COMPLETE' else ('TIMEOUT_UNREVIEWED_SET' if 'TimeoutExpired' in d.get('error','') else 'UNREVIEWED_OTHER_FAILURE')
        result['nominated_by_original_scan']=any(c['id']==r['id'] for c in d.get('output',{}).get('candidates',[]))
    assert result['ranking_sha256_before_audit']==sha(OUT/'ranking.json')
    return result

def report(rows,reviews,ranking,summary,coverage):
    esc=lambda x:html.escape(str(x));dump=lambda x:'<pre>'+esc(json.dumps(x,ensure_ascii=False,indent=2))+'</pre>'
    body=['<h1>Corner Reveal 目标锁定 · 10例最小修正审核</h1><p>只允许 Reveal(LS/RS/LH/RH)；不生成Skill，不运行机器人。STRONG_VISUAL/PROVISIONAL均不是独立材料真值。</p>',dump(summary),'<h2>修正后的候选排序</h2>',dump(ranking),'<h2>十个案例</h2><table><tr><th>ID</th><th>目标</th><th>Before→After</th><th>三项证据分</th><th>归纳资格</th><th>理由</th></tr>']
    byid={x['id']:x for x in reviews}
    for r in rows:
        x=byid.get(r['id'])
        if x:body.append('<tr>'+''.join('<td>'+esc(t)+'</td>' for t in [r['id'],x['candidate_goal'],x['before_state']+' → '+x['after_state'],[x[k] for k in ['before_evidence_score','after_identity_score','transition_score']],x['verdict'],x['verdict_reason']])+'</tr>')
        else:body.append('<tr><td>'+r['id']+'</td><td colspan="5">API失败，未得到审核；不能计为无证据。</td></tr>')
    body.append('</table>')
    for r in rows:
        folder=OUT/'evidence'/r['id'];folder.mkdir(parents=True,exist_ok=True)
        for k in ['before','after']:shutil.copyfile(REPO/r[k],folder/(k+'.png'))
        body+=['<details><summary>'+r['id']+' · '+esc(r['source'])+'</summary><div class="pair">'+''.join('<figure><a href="evidence/'+r['id']+'/'+k+'.png"><img src="evidence/'+r['id']+'/'+k+'.png"></a><figcaption>'+k.upper()+'</figcaption></figure>' for k in ['before','after'])+'</div>',dump({'original_intention':raw_intention(r),'executed_action':r['action'],'review':byid.get(r['id'],'MISSING')}),'</details>']
    body+=['<h2>原先选中的案例</h2>',dump(read(OUT/'previous_selection_assessment.json')),'<h2>实验结束后独立覆盖审计：E353</h2>',dump(coverage),'<p>上面的覆盖审计在候选排名冻结之后运行，没有将E353身份、位置或既有分析提供给审核模型。已审阅指Contact Sheet请求成功，不代表端点真值已确认。</p>','<p><a href="protocol.json">冻结协议</a> · <a href="reviews.json">10例原始审核</a> · <a href="ranking.json">排序</a> · <a href="coverage_audit.json">覆盖审计</a> · <a href="audit.json">输入隔离检查</a></p>']
    (OUT/'review.html').write_text('<!doctype html><meta charset="utf-8"><title>Corner Goal Audit</title><style>body{font:17px/1.6 system-ui;max-width:1450px;margin:30px auto;padding:20px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f0f2f5;padding:16px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:8px}.pair{display:flex}.pair figure{width:50%;margin:8px}.pair img{width:100%}details{padding:12px;border-bottom:1px solid #ddd}summary{cursor:pointer}</style>'+''.join(body))

def main():
    OUT.mkdir(parents=True,exist_ok=True);short=read(SOURCE/'shortlist.json');index={x['id']:x for x in read(SOURCE/'experience_index.json')};rows=[index[x['id']] for x in short];assert len(rows)==10
    protocol={'task':'semantic endpoint goal lock, re-audit only existing ten pairs','allowed_goals':GOALS,'models':'gpt-6-astra/high ephemeral independent per pair','limits':{'pairs':10,'calls':10,'workers':2,'timeout_s':480,'retries':0,'skills_generated':0,'robot_actions':0},'source_shortlist_sha256':sha(SOURCE/'shortlist.json'),'code_sha256':sha(__file__),'prompt_sha256':hashlib.sha256(PROMPT.encode()).hexdigest(),'priority':'Corner relevance mandatory; reveal reliability first; unexpectedness then hypothesis value only tie-breaks; planned benefits eligible','gate':'specific semantic transition, Before self-occluded/suspected, After exposed, all three evidence scores >=2, eligible model verdict; stronger requires all3 and strong verdict','models_receive':'one raw Before/After, original intention fields, executed action/status; no old rankings, reviews, selected-case marker or named historical audit information','posthoc_audit':'executed only after ranking is persisted; metadata only; never fed into models'}
    if (OUT/'protocol.json').exists():assert read(OUT/'protocol.json')==protocol
    else:save(OUT/'protocol.json',protocol)
    progress('REVIEW',total=10);reviews=[];failures=[]
    def one(r):
        imgs=[REPO/r[k] for k in ['before','after']]
        for k,p in zip(['before','after'],imgs):assert sha(p)==r[k+'_sha256']
        payload={'id':r['id'],'image_order':['Before','After'],'original_intention':raw_intention(r),'actual_executed_command':r['action'],'execution_status':r['execution']}
        x=call(OUT,r['id'],PROMPT,payload,imgs,SCHEMA)
        if x['status']=='COMPLETE':assert x['output']['id']==r['id']
        return r['id'],x
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for id,x in pool.map(one,rows):
            if x['status']=='COMPLETE':reviews.append(x['output'])
            else:failures.append({'id':id,'error':x.get('error')})
            save(OUT/'reviews.json',reviews);save(OUT/'failures.json',failures);progress('REVIEW',completed=len(reviews),failed=len(failures),total=10)
    eligible=[];excluded=[]
    for x in reviews:
        ok,why=eligibility(x)
        if ok:
            x=dict(x);x['host_tier']='STRONG_VISUAL' if x['verdict']=='ELIGIBLE_STRONG_VISUAL' and min(x[k] for k in ['before_evidence_score','after_identity_score','transition_score'])==3 else 'PROVISIONAL';eligible.append(x)
        else:excluded.append({'id':x['id'],'reasons':why})
    eligible.sort(key=sortkey)
    ranking={'ranked_candidates':[{'rank':i+1,'id':x['id'],'goal':x['candidate_goal'],'status':x['host_tier'],'scores':[x[k] for k in ['before_evidence_score','after_identity_score','transition_score']],'unexpectedness':x['unexpectedness'],'hypothesis_value':x['hypothesis_value'],'physical_success_confirmed':False} for i,x in enumerate(eligible)],'excluded':excluded,'api_failures':failures,'frozen_at':time.time()};save(OUT/'ranking.json',ranking)
    # Previous selection identifier is read only AFTER blind review and ranking.
    old=read(SOURCE/'selected_case.json')['case']['id'];oldrow=next((x for x in reviews if x['id']==old),None)
    ok,reasons=eligibility(oldrow) if oldrow else (False,['API审核未完成'])
    save(OUT/'previous_selection_assessment.json',{'id':old,'eligible_under_fixed_corner_goal':ok,'rejection_reasons':reasons,'new_review':oldrow,'old_review_or_goal_supplied_to_model':False})
    if failures:status='INCOMPLETE_CORNER_AUDIT'
    else:status='CORNER_INDUCTION_CANDIDATES_AVAILABLE' if eligible else 'NO_VALID_CORNER_DISCOVERY'
    summary={'status':status,'requested':10,'reviewed':len(reviews),'failures':len(failures),'eligible':len(eligible),'strong_visual':sum(x['host_tier']=='STRONG_VISUAL' for x in eligible),'provisional':sum(x['host_tier']=='PROVISIONAL' for x in eligible),'can_start_semantic_corner_induction':bool(eligible),'qualification':'可启动仅针对列明语义目标的待验证假设归纳；不是已验证技能或已确认物理成功。' if eligible else ('审核不完整，不能得出十例均无有效证据的结论。' if failures else '当前十例没有达到固定门槛的语义Corner Reveal候选；不降级目标，不生成Skill。'),'skill_induction_run':False,'physical_actions':0,'production_changes':False};save(OUT/'summary.json',summary)
    progress('RANKING_FROZEN_POSTHOC_METADATA_AUDIT');coverage=posthoc_coverage();save(OUT/'coverage_audit.json',coverage)
    allowed={r[k+'_sha256'] for r in rows for k in ['before','after']};manifests=list((OUT/'calls').glob('*/attempt_*/manifest.json'));assert len(manifests)==10
    for p in manifests:
        m=read(p);assert len(m['images'])==2 and all(im['sha256'] in allowed for im in m['images'])
        txt=(p.parent/'input.txt').read_text()
        assert 'E353' not in txt and 'previous_selection' not in txt
    save(OUT/'audit.json',{'status':'PASS','only_existing_ten_pairs':True,'old_conclusions_supplied':False,'named_historical_case_info_supplied':False,'ranking_frozen_before_coverage_audit':ranking['frozen_at']<=coverage['audit_time'],'ranking_unchanged_by_audit':sha(OUT/'ranking.json')==coverage['ranking_sha256_before_audit'],'model_manifests':len(manifests),'orchestrator_has_prior_chat':True})
    report(rows,reviews,ranking,summary,coverage);progress('COMPLETE',**summary)
if __name__=='__main__':main()
