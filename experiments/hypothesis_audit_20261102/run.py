"""Read structured mining artifacts only; no image decoding, models, robot or source mutation."""
from pathlib import Path
from collections import Counter, defaultdict
import json, hashlib, html, re, itertools, datetime

ROOT=Path('/mnt/newssd/sja/clothAgent_real/reports/intention_outcome_hypothesis_bank_v1')
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def populated(v): return v is not None and v not in ('',[],{})
def norm(s): return re.sub(r'\W+','',s).lower()
def flat(v):
    if isinstance(v,dict):return '\n'.join(flat(x) for x in v.values())
    if isinstance(v,list):return '\n'.join(flat(x) for x in v)
    return str(v)
def counter(v):return dict(Counter(v))
def esc(v):return html.escape(str(v))

required=['hypothesis_bank.json','experience_observations.json','experience_index.json']
missing=[str(ROOT/n) for n in required if not (ROOT/n).exists()]
if missing:
    result={'status':'MISSING_ARTIFACTS','missing':missing,'no_results_inferred':True}
    (ROOT/'hypothesis_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    (ROOT/'hypothesis_audit.html').write_text('<meta charset="utf-8"><pre>'+esc(json.dumps(result,ensure_ascii=False,indent=2))+'</pre>')
    raise SystemExit('Required artifacts missing; audit saved.')
source_names=required+['protocol.json','summary.json','data_audit.json','unprocessed.json','execution_state.json','progress.json']
source_paths=[ROOT/n for n in source_names if (ROOT/n).exists()]
source_hashes={str(p):sha(p) for p in source_paths}
bank=read(ROOT/required[0]); observations=read(ROOT/required[1]); index=read(ROOT/required[2])
obs_by={o['experience_id']:o for o in observations}; idx_by={x['experience_id']:x for x in index}
bank_by={h['hypothesis_id']:h for h in bank}
issues=[]; cases=[]; all_m={}; payloads={}; file_hash_cache={}
def filehash(p):
    if str(p) not in file_hash_cache:file_hash_cache[str(p)]=sha(p)
    return file_hash_cache[str(p)]
phase_fields={
 'intention_extraction':('intended_effect',['intended_object','intended_changes','explicit_expected_outcome','original_quotes','unspecified_elements','uncertainty']),
 'outcome_observation':('observed_effect',['whole_garment_changes','global_motion','relative_deformation','new_occlusions','view_confound','execution_evidence_limits','effect_determinacy','uncertainty'])}
for x in index:
    eid=x['experience_id']; item=obs_by.get(eid); ob=(item or {}).get('observation') or {}
    payload_path=Path(x['payload']); payload=read(payload_path) if payload_path.exists() else {};payloads[eid]=payload
    if payload_path.exists():source_hashes[str(payload_path)]=filehash(payload_path)
    pstatus={}
    for phase,(key,fields) in phase_fields.items():
        sub=ob.get(key); pstatus[phase]='COMPLETE' if isinstance(sub,dict) and all(f in sub for f in fields) else 'INCOMPLETE'
    pstatus['mismatch_analysis']='COMPLETE' if isinstance(ob.get('mismatches'),list) else 'INCOMPLETE'
    hs=ob.get('hypotheses');pstatus['hypothesis_generation']='COMPLETE' if isinstance(hs,list) and (len(hs)>0 or populated(ob.get('no_hypothesis_reason'))) else 'INCOMPLETE'
    quote_issues=[q for q in ob.get('intended_effect',{}).get('original_quotes',[]) if q not in flat(payload.get('original_intention',{}))]
    if quote_issues:issues.append({'type':'QUOTE_NOT_EXACT_SUBSTRING','experience_id':eid,'quotes':quote_issues})
    verified={}
    for k in ['before','after','payload']:
        path=Path(x[k]); expected=x.get(k+'_sha256'); exists=path.exists()
        verified[k]={'path':str(path),'exists':exists,'sha256_match':filehash(path)==expected if exists and expected else None}
        if not exists or verified[k]['sha256_match'] is False:issues.append({'type':'INPUT_EVIDENCE_FAILURE','experience_id':eid,'file':k})
    for m in ob.get('mismatches',[]):all_m[m['mismatch_id']]={'experience_id':eid,**m}
    for h in hs or []:
        bh=bank_by.get(h['hypothesis_id'])
        if bh is None or any(bh.get(k)!=v for k,v in h.items()):issues.append({'type':'BANK_OBSERVATION_DIFFERENCE','hypothesis_id':h['hypothesis_id']})
    # Eight reasons explicitly say no explanatory discrepancy; one says evidence too weak to attribute a mechanism.
    empty_class=None
    if hs==[]:empty_class='INSUFFICIENT_MECHANISM_EVIDENCE' if eid=='X0319' else 'NO_WORTH_EXPLAINING_MISMATCH_WITH_OBSERVATION_LIMITS'
    cases.append({'experience_id':eid,'source':x.get('source'),'run':x.get('run'),'observation_status':(item or {}).get('status','MISSING'),
      'phase_status':pstatus,'hypothesis_count':len(hs or []),'empty_result_class':empty_class,'original_empty_reason':ob.get('no_hypothesis_reason'),
      'effect_determinacy':ob.get('observed_effect',{}).get('effect_determinacy'),'original_intention':payload.get('original_intention'),
      'intention_has_explicit_expected_stop_state':populated(payload.get('original_intention',{}).get('expected_stop_state')),
      'evidence_checks':verified,'quote_issues':quote_issues,'observation_ref':f'cases/{eid}.html'})
nested_ids={h['hypothesis_id'] for o in observations for h in o.get('observation',{}).get('hypotheses',[])}
if nested_ids != set(bank_by):issues.append({'type':'BANK_ID_SET_MISMATCH','bank_only':sorted(set(bank_by)-nested_ids),'observation_only':sorted(nested_ids-set(bank_by))})
if len(bank_by)!=len(bank) or len(obs_by)!=len(observations):issues.append({'type':'DUPLICATE_IDS'})
# Valence review is of existing text only, not an image assessment or validation.
valence_review={
 'X0158-M002':('NEGATIVE_RELATIVE_TO_STATED_GOAL','明确预期横向展开，记录却描述横向投影缩小；材料是否压缩仍不确定。'),
 'X0160-M004':('NEGATIVE_RELATIVE_TO_STATED_GOAL','原意图避免向下缘牵拉，记录描述末态下缘继续向下；不推断执行轨迹。'),
 'X0162-M003':('OTHER_OR_AMBIGUOUS','原意图避免对侧被牵动，而该侧移动且变宽；不能由这些文字确定净收益。'),
 'X0340-M002':('MIXED','原记录同时描述局部平展和覆盖增加；不能归为单纯有益或单纯负面。'),
 'X0509-M001':('NEGATIVE_RELATIVE_TO_STATED_GOAL','暂定的小幅方向变化与明示左上展开相反；证据仍弱。'),
 'X0529-M002':('NEGATIVE_RELATIVE_TO_STATED_GOAL','明确消除堆折预期未成立，并描述新折叠与遮蔽。'),
 'X0532-M002':('NEGATIVE_RELATIVE_TO_STATED_GOAL','明示相邻衣身铺平，记录仍描述明显折叠和突出褶带。')}
required_h=['hypothesis_id','source_experience_id','intended_effect','observed_effect','mismatch','observable_conditions','hypothesized_mechanism','predicted_effect','falsification_condition','evidence_level','uncertainty','mismatch_ids','evidence']
hrows=[]; missing_counts=Counter(); linked_types=Counter(); goal_events=Counter()
for h in bank:
    hid=h['hypothesis_id']; eid=h['source_experience_id']; ms=[all_m[m] for m in h.get('mismatch_ids',[]) if m in all_m]
    missing_fields=[k for k in required_h if not populated(h.get(k))];missing_counts.update(missing_fields)
    bad_refs=[m for m in h.get('mismatch_ids',[]) if m not in all_m or all_m[m]['experience_id']!=eid or not all_m[m].get('worth_explaining') or all_m[m]['type']=='EXPECTED_OBSERVED']
    evidence_check={k:{'ref':v,'exists':(ROOT/v).is_file()} for k,v in h.get('evidence',{}).items()}
    absent_evidence=[k for k in ['before','after','original_record','full_observation'] if k not in evidence_check or not evidence_check[k]['exists']]
    if missing_fields or bad_refs or absent_evidence:issues.append({'type':'HYPOTHESIS_QUALITY_FAILURE','hypothesis_id':hid,'missing_fields':missing_fields,'bad_refs':bad_refs,'missing_evidence':absent_evidence})
    types=sorted(set(m['type'] for m in ms));linked_types.update(types)
    obs=obs_by.get(eid,{}).get('observation',{})
    corner_links=[c for c in obs.get('corners',[]) if c['goal'] in h.get('related_corner_goals',[])]
    positive=[c for c in corner_links if c['change'] in ['REVEAL','VISIBILITY_IMPROVED']]
    negative=[c for c in corner_links if c['change'] in ['NEW_OCCLUSION','VISIBILITY_REDUCED']]
    # Associations are same-case, same-corner; not proof the hypothesis describes the specific event.
    vc=[valence_review[m['mismatch_id']][0] for m in ms if m['mismatch_id'] in valence_review]
    relation_bucket='MIXED_EXPLICIT_LINKS' if len(set(vc))>1 else (vc[0] if vc else 'NO_EXPLICIT_UNEXPECTED_LINK')
    original=payloads.get(eid,{}).get('original_intention',{})
    orig=flat(original)
    inherited_cues=[k for k in ['对侧','对向','摩擦','间接','轮廓法线','向内法线','相对位移','上层','下层'] if k in orig]
    # Verbatim clause inclusion can flag copying; absence is not semantic novelty.
    clauses=[s.strip() for s in re.split('[，。；;\n]',orig) if len(norm(s))>=12]
    exact_inherited=[s for s in clauses if norm(s) in norm(h['hypothesized_mechanism'])]
    hrows.append({'hypothesis_id':hid,'source_experience_id':eid,'original_hypothesis':h,
      'quality':{'missing_or_empty_fields':missing_fields,'invalid_mismatch_refs':bad_refs,'missing_evidence_files':absent_evidence,
        'all_required_fields_and_links_present':not(missing_fields or bad_refs or absent_evidence),'mechanism_independently_validated':h.get('physical_mechanism_validated'),
        'semantic_quality_assessed_for_every_entry':False},
      'evidence_files':evidence_check,'linked_mismatches':ms,'linked_mismatch_types':types,'explicit_unexpected_text_valence':relation_bucket,
      'same_corner_positive_event_associations':positive,'same_corner_negative_event_associations':negative,
      'novelty':{'status':'NOT_ESTABLISHED','original_intention':original,'original_relation_cue_flags':inherited_cues,
        'verbatim_original_clauses_in_mechanism':exact_inherited,'note':'词语/原句标记仅供复核。新写出的机制不等于新知识；未匹配到原句也不证明自主发现。'}})
# Exact / high textual similarity flags only; no merging or choosing representatives for the bank.
def grams(t):
    t=norm(t);return {t[i:i+4] for i in range(max(0,len(t)-3))}
gs=[grams(h['hypothesized_mechanism']) for h in bank];duplicate_flags=[]
for i in range(len(bank)):
    for j in range(i):
        a,b=gs[i],gs[j]
        if not a or not b or min(len(a),len(b))/max(len(a),len(b))<.55:continue
        inter=len(a&b);score=inter/(len(a)+len(b)-inter)
        if score>=.55:duplicate_flags.append({'a':bank[j]['hypothesis_id'],'b':bank[i]['hypothesis_id'],'char4_jaccard':round(score,4),'kind':'EXACT_MECHANISM_TEXT' if norm(bank[j]['hypothesized_mechanism'])==norm(bank[i]['hypothesized_mechanism']) else 'POSSIBLE_TEXTUAL_DUPLICATE','not_semantically_confirmed':True})
duplicate_flags.sort(key=lambda x:-x['char4_jaccard'])
comparisons=list(all_m.values()); corners=[dict(experience_id=o['experience_id'],**c) for o in observations for c in o.get('observation',{}).get('corners',[])]
explicit=[dict(original_comparison=m,audit_valence=valence_review.get(m['mismatch_id'],('UNCLASSIFIED','未经文字审核'))[0],audit_basis=valence_review.get(m['mismatch_id'],('',''))[1]) for m in comparisons if m['type']=='UNEXPECTED_CHANGE']
representatives=[
 {'id':'X0158-H001','role':'相对较强的可检验结构','reason':'原文明确预期横向展开，分析提出转向/拉直解释，给出上部稳定与材料跟踪条件及反证；仍为 PROVISIONAL。'},
 {'id':'X0001-H001','role':'观察证据标签较强，机制仍未验证','reason':'原记录明确说明 CONFIRMED 仅指不均匀形变可见，并非已证实摩擦或材料连接。'},
 {'id':'X0340-H002','role':'支撑较弱的过程机制假设','reason':'先展开后落覆需要中间影像，而记录只有端帧。结构字段齐全不弥补关键过程缺失。'},
 {'id':'X0368-H003','role':'明确 UNKNOWN 的假设','reason':'预测依赖暂时展开和回落，但原记录承认没有直接证据证明曾经展开。'},
 {'id':'X0311-H001','role':'未提及的暂定端点收益','reason':'关联对侧袖口的暂定显露与新折叠；原意图未提及不代表明确预测不变，也不证明机制成立。'},
 {'id':'X0156-H001','role':'原意图已含关系，不能整条称新发现','reason':'原意图已提出牵动对侧上层、让下层留桌、相对位移释放袖部。分析重述此机制并增加解释；核心关系不是本轮首次提出。'},
 {'id':'X0097-H001','role':'原意图已含摩擦与对侧显露','reason':'借助桌面摩擦显露对侧已写在原 action_purpose；分析提出不同部位位移/褶皱解释，新增部分未独立验证。'}]
phase_counts={k:counter(c['phase_status'][k] for c in cases) for k in ['intention_extraction','outcome_observation','mismatch_analysis','hypothesis_generation']}
summary={
 'indexed_experiences':len(index),'observations':len(observations),'distinct_runs':len({c['run'] for c in cases}),
 'complete_status':sum(c['observation_status']=='COMPLETE' for c in cases),'phase_counts':phase_counts,
 'unprocessed_ids':[c['experience_id'] for c in cases if any(v!='COMPLETE' for v in c['phase_status'].values())],
 'empty_hypothesis_cases':counter(c['empty_result_class'] for c in cases if c['empty_result_class']),
 'effect_determinacy':counter(c['effect_determinacy'] for c in cases),'hypotheses':len(bank),
 'related_to_any_corner':sum(bool(h['related_corner_goals']) for h in bank),'categories_nonexclusive':counter(c for h in bank for c in h['categories']),
 'comparison_types':counter(m['type'] for m in comparisons),'hypothesis_links_to_mismatch_types_nonexclusive':dict(linked_types),
 'explicit_unexpected_comparison_valence_text_audit':counter(e['audit_valence'] for e in explicit),
 'hypotheses_linked_to_explicit_unexpected_valence':counter(h['explicit_unexpected_text_valence'] for h in hrows),
 'positive_event_associated_hypotheses':sum(bool(h['same_corner_positive_event_associations']) for h in hrows),
 'negative_event_associated_hypotheses':sum(bool(h['same_corner_negative_event_associations']) for h in hrows),
 'positive_event_associations_by_original_expectation_nonexclusive':counter(e for h in hrows for e in set(c['original_expectation'] for c in h['same_corner_positive_event_associations'])),
 'negative_event_associations_by_original_expectation_nonexclusive':counter(e for h in hrows for e in set(c['original_expectation'] for c in h['same_corner_negative_event_associations'])),
 'corner_change_expectation':counter(c['change']+' / '+c['original_expectation'] for c in corners),
 'corner_change_evidence':counter(c['change']+' / '+c['evidence_level'] for c in corners),
 'hypothesis_evidence_levels':counter(h['evidence_level'] for h in bank),'missing_fields':dict(missing_counts),
 'structurally_complete_hypotheses':sum(h['quality']['all_required_fields_and_links_present'] for h in hrows),
 'mechanism_validated_flags':counter(str(h.get('physical_mechanism_validated')) for h in bank),
 'novelty_not_established':len(bank),'original_relation_cue_flagged_hypotheses':sum(bool(h['novelty']['original_relation_cue_flags']) for h in hrows),
 'original_relation_cue_flagged_cases':len({h['source_experience_id'] for h in hrows if h['novelty']['original_relation_cue_flags']}),
 'exact_original_clause_in_mechanism_hypotheses':sum(bool(h['novelty']['verbatim_original_clauses_in_mechanism']) for h in hrows),
 'possible_text_duplicate_pairs':len(duplicate_flags),'integrity_issue_count':len(issues),
 'unique_experience_ids':len(obs_by),'unique_hypothesis_ids':len(bank_by),
 'explicit_expected_stop_state_cases':sum(c['intention_has_explicit_expected_stop_state'] for c in cases)}
limits=[
 '本次只检查现存结构化文字、引用及文件字节哈希；没有打开/解码 RGB，没有模型调用，没有新假设、机器人或策略更新。',
 '完成处理不等于观察确定；全部 PARTIAL 是效果证据范围标签，不是程序失败。',
 '原库没有独立的效果收益字段或知识新颖性字段。明确意外变化仅对7项原文做文字核对；未提及、未知与未确认预期单列，不编造成预测错误。',
 '同端点收益/负面事件关联只表示同一案例中关联同一 goal，不能证明每条假设都解释该事件；不据此计算成功率。',
 'X0162的EXPLICIT_UNCHANGED指避免牵动袖部，原不确定性字段明确说并不等于预测袖口可见性不变；不能只依此枚举值宣布有益预测错误。',
 '没有逐条重新做语义评审；字段齐全与反证文字存在不证明可观测条件充分、假设正确或真正可区分。低质量示例指证据支撑较弱，不是缺字段。',
 '不存在支持1222条自主新增知识的独立证据。原意图携带既有规则；本次不审计规则的历史作者身份，不能据此断言每条都源于人工，也不能宣称是自主新发现。',
 '分析同时接收原意图与Before/After；不是盲于After的独立意图提取。原句核对只验证文字溯源。',
 '532条覆盖的是冻结索引，不等于所有历史数据已被证明无遗漏。现有模型标签不作为独立物理真值。',
 '重复检测仅机制文字四字片段Jaccard≥0.55，没检出不代表没有语义重复；不合并或改写。']
# Verify source immutability including input snapshots; images are hashed, never decoded.
changed=[str(p) for p in source_paths if sha(p)!=source_hashes[str(p)]]
changed += [p for p,h in file_hash_cache.items() if sha(Path(p))!=h]
audit={'audit_version':'read_only_structured_audit_v1','created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'source_root':str(ROOT),'source_hashes':source_hashes,'input_evidence_hashes':file_hash_cache,'source_files_changed':changed,
 'audit_code_sha256':sha(Path(__file__)),'summary':summary,'limits':limits,'integrity_issues':issues,
 'explicit_unexpected_comparison_text_reviews':explicit,'representatives':representatives,'possible_duplicates':duplicate_flags,
 'experiences':cases,'hypotheses':hrows,'raw_corner_events':corners,
 'method':{'stage_completion':'Required structure present; empty hypothesis list must have original no_hypothesis_reason.',
 'empty_results':'Text-only reason review: X0319 insufficient attribution; remaining 8 no explanatory mismatch, all preserve raw reason.',
 'valence':'Text-only review of seven explicit UNEXPECTED_CHANGE comparisons; negative relative to stated goal, not physical harm/success.',
 'novelty':'No independent novelty ground truth. Preserve original intentions and mechanism side by side; flag cues and exact clause inclusion only.',
 'quality':'All1222 checked structurally; seven explicitly selected text examples illustrate differing evidence and source conditions.'}}
(ROOT/'hypothesis_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

# All source text is escaped; lazy images provide browsing only, never feed inference.
def pre(obj):return '<pre>'+esc(json.dumps(obj,ensure_ascii=False,indent=2))+'</pre>'
def table(d):return '<table>'+''.join('<tr><td>'+esc(k)+'</td><td>'+esc(v)+'</td></tr>' for k,v in d.items())+'</table>'
def link(ref,label):return '<a href="'+esc(ref)+'">'+esc(label)+'</a>'
parts=['''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Hypothesis Bank 只读审计</title><style>
body{font:16px/1.65 system-ui,sans-serif;max-width:1200px;margin:auto;padding:28px;background:#f5f7fa;color:#202734}h1,h2,h3{line-height:1.3}section,details{background:white;border:1px solid #d7deea;padding:16px;margin:14px 0;border-radius:8px}summary{cursor:pointer;font-weight:bold}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.55 monospace}table{border-collapse:collapse;width:100%}td{border:1px solid #ddd;padding:7px;overflow-wrap:anywhere}.warning{background:#fff4d7;padding:14px}.pair{display:flex;gap:8px}.pair img{width:49%;object-fit:contain}input,select{padding:10px;margin:6px;max-width:90%}a{color:#1559a1}.muted{color:#687482}.card h4{margin-bottom:3px}.card p{margin-top:2px}
</style><h1>Intention–Outcome Hypothesis Mining：只读审计</h1>''']
parts+=['<p>来源：'+esc(ROOT)+' · '+esc(audit['created_at'])+'</p><p>'+link('hypothesis_audit.json','下载完整审计 JSON')+' · '+link('hypothesis_bank.json','原始假设库')+' · '+link('experience_observations.json','原始观察记录')+'</p>',
 '<div class="warning">这是现有模型分析产物的审计，不是再次判断图片或证明技能成功。字段完整、新写出的解释、CONFIRMED 标签，均不代表物理机制已经验证。</div>',
 '<section><h2>实际产物</h2>'+table({'已处理操作':f'{summary["complete_status"]}/{len(index)}','假设':len(bank),'关联四端点':summary['related_to_any_corner'],'CORNER_REVEAL 类别':summary['categories_nonexclusive'].get('CORNER_REVEAL',0),'结构字段及证据链接完整':summary['structurally_complete_hypotheses'],'独立物理机制验证':summary['mechanism_validated_flags'],'源文件变更':changed})+'</section>',
 '<section><h2>完整性与空结果</h2>'+pre({'phase_counts':phase_counts,'empty_results':summary['empty_hypothesis_cases'],'effect_determinacy':summary['effect_determinacy'],'unprocessed':summary['unprocessed_ids'],'integrity_issues':issues})+
 '<p>9条空假设不等于9条未处理：8条没有值得建立机制假设的差异，X0319有未确认预期，但缺少归因证据。所有案例保留原始理由。</p>'+
 ''.join('<details><summary>'+c['experience_id']+' · '+esc(c['empty_result_class'])+'</summary><p>'+esc(c['original_empty_reason'])+'</p>'+link(c['observation_ref'],'原观察')+'</details>' for c in cases if c['empty_result_class'])+'</section>',
 '<section><h2>意外收益、负面变化与其他差异</h2><p>以下分开统计比较项与假设，类别允许交叉，不能相加成总数。UNSPECIFIED_CHANGE不是明确预测错误。</p>'+table(summary['comparison_types'])+
 '<h3>假设引用的差异类型（非互斥）</h3>'+table(summary['hypothesis_links_to_mismatch_types_nonexclusive'])+
 '<h3>明确意外变化：原文核对</h3>'+table(summary['explicit_unexpected_comparison_valence_text_audit'])+
 '<h3>关联这些明确意外变化的假设数（每条只记一个汇总桶）</h3>'+table(summary['hypotheses_linked_to_explicit_unexpected_valence'])+
 '<p>以上“负面”仅表示与明确目标方向相反。0项明确有益不是说没有意外收益：未提及目标的潜在收益另列，不能升级为明确预期违背。</p>'+
 ''.join('<details><summary>'+e['original_comparison']['mismatch_id']+' · '+esc(e['audit_valence'])+'</summary>'+pre(e)+'</details>' for e in explicit)+
 '<h3>四端点变化 × 原意图（事件数）</h3>'+table({k:v for k,v in summary['corner_change_expectation'].items() if k.split(' / ')[0] in ['REVEAL','VISIBILITY_IMPROVED','NEW_OCCLUSION','VISIBILITY_REDUCED']})+
 '<p>同端点正向事件关联假设 '+str(summary['positive_event_associated_hypotheses'])+' 条；负向关联 '+str(summary['negative_event_associated_hypotheses'])+' 条，可能重叠，只是关联线索。</p><h3>按原意图拆分上述关联（假设数，非互斥）</h3><p>正向关联：</p>'+table(summary['positive_event_associations_by_original_expectation_nonexclusive'])+'<p>负向关联：</p>'+table(summary['negative_event_associations_by_original_expectation_nonexclusive'])+'<p>其中42条关联原意图未提及的正向端点变化，28条关联原意图未提及的负向变化。它们是未提及效果的候选解释，不是42条明确意外收益或28条预测错误。X0162的EXPLICIT_UNCHANGED仅指避免牵动袖部，并不等于预测可见性不变。</p></section>',
 '<section><h2>知识到底新增了什么？</h2><p>库中保存的是解释性候选关系，未保存独立的新颖性判定。不能把1,222条全算作新增知识，也不能由本次审计宣布零新增。'+str(summary['original_relation_cue_flagged_cases'])+' 个原始案例中已有对侧、摩擦、间接等关系词，共关联 '+str(summary['original_relation_cue_flagged_hypotheses'])+' 条假设；这只是待核查标记。</p><p>X0156原意图已明确提出上层移动、下层留桌及相对位移显露；X0097已明确借桌面摩擦显露对侧。后续解释这些关系不能记作本轮首次自主发现。X0311提出肩部重排可能显出对侧，但仍是未经验证的事后解释，且原意图未提及不等于预测不变。</p><p>每条索引并列展示原意图与新机制，保留无法判断的部分。</p></section>',
 '<section><h2>代表性对照：质量不等于字段齐全</h2>'+''.join('<p>'+link('#'+r['id'],r['id'])+' · <b>'+esc(r['role'])+'</b>：'+esc(r['reason'])+'</p>' for r in representatives)+
 '<p>全部假设缺字段统计：'+esc(summary['missing_fields'])+'。这里较弱的案例主要弱在证据，而不是格式。</p></section>',
 '<details><summary>疑似重复：'+str(len(duplicate_flags))+' 对（不合并）</summary><p>机制文字四字片段Jaccard≥0.55；这是措辞相似检查，不是语义同一判定。</p>'+''.join('<p>'+link('#'+d['a'],d['a'])+' ↔ '+link('#'+d['b'],d['b'])+' · '+str(d['char4_jaccard'])+'</p>' for d in duplicate_flags)+'</details>',
 '<details><summary>审计边界及方法</summary><ul>'+''.join('<li>'+esc(s)+'</li>' for s in limits)+'</ul>'+pre(audit['method'])+'</details>',
 '<details><summary>全部532条处理状态</summary>'+''.join('<p>'+link(c['observation_ref'],c['experience_id'])+' · '+esc(c['phase_status'])+' · 假设 '+str(c['hypothesis_count'])+'</p>' for c in cases)+'</details>',
 '<h2>全部假设索引</h2><input id="search" placeholder="搜索编号、条件、机制或原始意图"><select id="filter"><option value="">全部</option><option value="CORNER_REVEAL">CORNER_REVEAL</option><option value="UNEXPECTED_OCCLUSION">UNEXPECTED_OCCLUSION</option><option value="UNKNOWN">UNKNOWN证据</option><option value="UNEXPECTED_CHANGE">明确意外变化关联</option></select><p id="count"></p>']
for row in hrows:
    h=row['original_hypothesis'];hid=h['hypothesis_id'];eid=h['source_experience_id']
    text=json.dumps([h,row['novelty']['original_intention']],ensure_ascii=False).lower()
    filters=h['categories']+row['linked_mismatch_types']+([h['evidence_level']] if h['evidence_level']=='UNKNOWN' else [])
    parts.append('<details class="card" id="'+hid+'" data-search="'+esc(text)+'" data-filter="'+esc(' '.join(filters))+'"><summary>'+hid+' · '+esc(', '.join(h['related_corner_goals']) or '未关联角点')+' · '+esc(h['evidence_level'])+' · '+esc(', '.join(h['categories']))+'</summary>')
    parts.append('<p>'+link(h['evidence']['original_record'],'原始意图/动作 payload')+' · '+link(h['evidence']['full_observation'],'完整观察')+' · '+link(h['evidence']['before'],'Before')+' · '+link(h['evidence']['after'],'After')+'</p>')
    parts.append('<h4>原始 Agent 意图（原文）</h4>'+pre(row['novelty']['original_intention']))
    for k,label in [('intended_effect','提取的原意图'),('observed_effect','记录的实际效果'),('mismatch','差异'),('observable_conditions','可观测条件'),('hypothesized_mechanism','候选机制'),('predicted_effect','预期效果'),('falsification_condition','可证伪条件'),('uncertainty','不确定性')]:parts.append('<h4>'+label+'</h4><p>'+esc(h.get(k,''))+'</p>')
    parts.append('<h4>原始差异引用</h4>'+pre(row['linked_mismatches'])+'<h4>审计标签</h4>'+pre({'quality':row['quality'],'novelty':row['novelty']['status'],'original_relation_cue_flags':row['novelty']['original_relation_cue_flags'],'explicit_unexpected_text_valence':row['explicit_unexpected_text_valence']})+'</details>')
parts.append('''<script>
const cards=[...document.querySelectorAll('.card')],search=document.querySelector('#search'),filter=document.querySelector('#filter');function update(){let n=0;for(const c of cards){const ok=c.dataset.search.includes(search.value.toLowerCase())&&(!filter.value||c.dataset.filter.split(' ').includes(filter.value));c.hidden=!ok;n+=ok;}document.querySelector('#count').textContent=`显示 ${n} / ${cards.length} 条；点击展开原文。`;}search.oninput=update;filter.onchange=update;update();function hash(){const e=document.getElementById(location.hash.slice(1));if(e){search.value='';filter.value='';update();e.open=true;e.scrollIntoView();}}window.onhashchange=hash;hash();</script></html>''')
(ROOT/'hypothesis_audit.html').write_text(''.join(parts))
assert not changed,changed
print('Wrote',ROOT/'hypothesis_audit.json',ROOT/'hypothesis_audit.html')
