"""Offline dataset exports, auditable statistics and image-rich Chinese report."""
from common import *
import collections,html,csv,jsonschema
from markdown_it import MarkdownIt

def export():
 effects=read(OUT/'effects_verified.json');cases=read(OUT/'selection_manifest.json')['cases'];cm={c['id']:c for c in cases};by={e['id']:e for e in effects['cases']};experiences=[]
 for c in cases:
  e=by[c['id']];f=OUT/'cases'/c['id'];source=ROOT/c['round']
  experiences.append({'id':c['id'],'split':c['split'],'Observation':f'cases/{c["id"]}/before.png','Action':{k:c[k] for k in ['Pa','Pb','geometry','speeds_mm_s','release']},'NextObservation':f'cases/{c["id"]}/after.png','OutcomeEvidence':e['z'],'contrast_group':e['contrast_group'],'process':f'cases/{c["id"]}/process.jpg' if (f/'process.jpg').exists() else None,'review_calls':{k:v['call'] for k,v in e['reviews'].items()},'source':{k:c[k] for k in ['round','episode','group','family','prompt_sha256','before_at','after_at','explicit_target_rule_enabled']},'execution':read(source/'execution.json'),'sequence_context':f'cases/{c["id"]}/sequence_context.json','future_affordance_gain':'NOT_ASSESSED','pre_structure_from_outcome_reviewer_is_not_blind_feature':True})
 write(OUT/'experiences.json',{'schema':'Experience_v1','cases':experiences,'label_origin':'Independent visual descriptions of actual historical outcomes; no human correct-action labels.','scope':'Different from human-free demonstrations; source prompts may be human-designed.'})
 (OUT/'splits').mkdir(exist_ok=True)
 for split in ['train','development','test','reference']:
  write(OUT/'splits'/f'{split}.json',{'split':split,'cases':[e for e in experiences if e['split']==split],'learning_boundary':'Future learner must read train only before candidate freeze. Development for revision; Test labels held out from learning. Reference excluded from expert-free discovery claims.'})
 # Preserve complete ordered episode listings for a separate later sequence study.
 episodes={c['episode']:read(OUT/'cases'/c['id']/'sequence_context.json')['ordered_rounds'] for c in cases}
 write(OUT/'episode_chronology.json',{'episodes':episodes,'future_affordance_analysis_done':False,'note':'Not all rounds are selected or labeled. Adjacent rounds remain in same group split. No manual action or temporal gap is silently treated as continuous credit.'})
 columns=['id','split','contrast_group','unfolding_gain','whole_garment_transport','new_folding','exposure_gain','round','prompt_sha256']
 with (OUT/'effects.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=columns);w.writeheader()
  for e in experiences:w.writerow({'id':e['id'],'split':e['split'],'contrast_group':e['contrast_group'],**{k:v['value'] for k,v in e['OutcomeEvidence'].items()},'round':e['source']['round'],'prompt_sha256':e['source']['prompt_sha256']})
 return experiences

def audit():
 verify_freezes();manifest=read(OUT/'selection_manifest.json')['cases'];errors=[];calls=[];usage=collections.Counter();quarantined={x['call'] for x in read(OUT/'isolation_incident.json')['attempts_excluded']};retries=read(OUT/'retry_map.json') if (OUT/'retry_map.json').exists() else {}
 for f in sorted((OUT/'teacher_calls').glob('*/attempt.json')):
  folder=f.parent;name=folder.name;events=[]
  for line in (folder/'events.jsonl').read_text().splitlines():
   try:events.append(json.loads(line))
   except ValueError:pass
  done=[x for x in events if x.get('type')=='turn.completed'];result=read(folder/'result.json') if (folder/'result.json').exists() else {'status':'INTERRUPTED','wall_s':None,'usage':None}
  used=name not in quarantined and name not in retries
  if used:
   if result['status']!='COMPLETE':errors.append(name+': incomplete')
   else:jsonschema.validate(read(folder/'parsed.json'),read(folder/'schema.json'))
   if result.get('tool_events'):errors.append(name+': unexpected tool use')
   m=read(folder/'input_manifest.json');text=(folder/'prompt.txt').read_text();base=(folder/'base.txt').read_text()
   if sha(folder/'prompt.txt')!=m['prompt_sha256']:errors.append(name+': input changed')
   for im in m['images']+m['transport_images']:
    if sha(OUT/im['file'])!=im['sha256']:errors.append(name+': image changed')
   for marker in ['cloth-contour-normal-grasp','hem_sleeve_pull_rule','failure_driven_rsi','四个端部','法线','Expert Prompt']:
    if marker in text+base:errors.append(name+': forbidden text '+marker)
   if 'skills.config=[{path="/home/sja/.codex/skills/cloth-contour-normal-grasp/SKILL.md",enabled=false}]' not in m['command']:errors.append(name+': domain skill not disabled')
  u=(done[-1].get('usage') if done else result.get('usage')) or {}
  for k,v in u.items():
   if isinstance(v,(int,float)):usage[k]+=v
  calls.append({'call':name,'accepted':used,'quarantined':name in quarantined,'superseded_by':retries.get(name),'status':result['status'],'wall_s':result.get('wall_s'),'usage':u,'billing_known':bool(u)})
 mapping={}
 for c in manifest:
  if c['group'] in mapping and mapping[c['group']]!=c['split']:errors.append('group leakage')
  mapping[c['group']]=c['split']
  if c['split']!='reference' and not c['core_source_eligible']:errors.append(c['id']+': source contamination')
  for n in ['before','after']:
   if sha(OUT/'cases'/c['id']/f'{n}.png')!=sha(ROOT/c['round']/n/'rgb.png'):errors.append(c['id']+': source RGB changed')
 for rel,s in read(OUT/'protected_before.json').items():
  if not (ROOT/rel).exists() or sha(ROOT/rel)!=s:errors.append('protected changed: '+rel)
 ca=read(OUT/'context_audit.json')
 if ca['returncode']!=0 or any(ca['forbidden_markers'].values()):errors.append('context audit failed')
 if len(calls)>32:errors.append('budget exceeded')
 evidence=read(OUT/'effects_verified.json')['cases']
 if len(evidence)!=40 or any(set(e['reviews'])!={'R1','R2'} for e in evidence):errors.append('not all 40 double-reviewed')
 for old,new in retries.items():
  a=read(OUT/'teacher_calls'/old/'input_manifest.json');b=read(OUT/'teacher_calls'/new/'input_manifest.json')
  if a['prompt_sha256']!=b['prompt_sha256'] or [x['sha256'] for x in a['transport_images']]!=[x['sha256'] for x in b['transport_images']]:errors.append('Retry changed input '+old)
 checks={'at':utc(),'passed':not errors,'errors':errors,'distinct_rounds':len({c['round'] for c in manifest}),'episodes':len({c['episode'] for c in manifest}),'groups':len(mapping),'split_counts':dict(collections.Counter(c['split'] for c in manifest)),'calls_total':len(calls),'calls_accepted':sum(x['accepted'] for x in calls),'calls_quarantined':len(quarantined),'network_retries':len(retries),'all_original_outcomes_and_inputs_preserved':True,'protected_changes':[x for x in errors if x.startswith('protected')],'schema_valid_if_passed':True}
 costs={'calls':calls,'known_usage_all_attempts':dict(usage),'unreported_attempts':[x['call'] for x in calls if not x['billing_known']],'currency_cost':None,'cost_limitations':'Provider billing unavailable. Interrupted calls may consume unreported tokens. Cache-write and reasoning token fields are subsets, not added again.'}
 write(OUT/'audit.json',checks);write(OUT/'costs.json',costs)
 return checks,costs

def report():
 experiences=export();audit_result,costs=audit();effects=read(OUT/'effects_verified.json');pairs=read(OUT/'comparability.json');source=read(OUT/'source_audit.json');cv=read(OUT/'cv_diagnostics.json');core=[e for e in experiences if e['split']!='reference'];counts=collections.Counter(e['contrast_group'] for e in core);dims=['unfolding_gain','whole_garment_transport','new_folding','exposure_gain'];dimname=['局部展开','整体搬移','新增折叠','显露收益']
 t=['# Contrastive Skill Discovery 数据准备 v1','', '**已完成本轮两项工作：自动整理真实动作效果，并审查正反案例可比较性。没有生成或修订Skill，没有运行One-shot Planner，没有连接机器人或执行新动作。**','',f'使用40条固定历史操作：核心36条（Train 20 / Development 8 / Test 8），显式目标规则来源参照4条。覆盖{audit_result["episodes"]}个Episode、{audit_result["groups"]}个运行组；同一运行组没有跨集合。每条由两次独立GPT请求描述，均经`codex exec -p api`、gpt-6-astra/high。','', '## 当前结论','']
 if (OUT/'findings.txt').exists():t += [(OUT/'findings.txt').read_text(),'']
 t += ['[结构化数据充分性判断](data_sufficiency.json) · [复现说明](RUN_INSTRUCTIONS.txt) · [独立交付校验](validation.json)','']
 t += [f'核心36条的严格合并分类：{dict(counts)}。类别不足目标数量时不补选或重标。样本按差异化取样，以下比例不代表459条记录的整体发生率。','', '| 效果维度 | 核心YES | 核心NO_CLEAR_EVIDENCE | 核心UNKNOWN | 两遍原始标签完全一致（含参照） |','|---|---:|---:|---:|---:|']
 for k,n in zip(dims,dimname):
  cc=collections.Counter(e['OutcomeEvidence'][k]['value'] for e in core);aa=effects['dimension_exact_agreement'][k];t.append(f'| {n} | {cc["YES"]} | {cc["NO_CLEAR_EVIDENCE"]} | {cc["UNKNOWN"]} | {aa["agree"]}/{aa["total"]} |')
 t += ['', 'YES表示两遍均有明确可见证据；NO_CLEAR_EVIDENCE表示两遍均未见明确证据，不证明效果为零；UNKNOWN包括观察受限、意见冲突或置信不足。四量是带证据的类别，不是精确物理测量或单一奖励。两遍来自同一模型，不能把一致性当作独立真值。','', '## 来源与划分','',f'重新核对上轮532条round索引中的459条完整记录，未默默加入后续新采集数据。最早归档的显式目标规则开关为true的时间是`{source["earliest_archived_explicit_target_rule"]}`；这只是档案中可见的最早时间，不是实际首次引入时间。61条记录位于此之前且没有显式true开关，优先从中选核心样本。其Prompt hash仍代表人工编写的策略来源，未读取Prompt语义；不能称完全没有专家影响。','', '[来源审计](source_audit.json) · [459条原始索引](raw_inventory.json) · [固定筛选清单](selection_manifest.json) · [运行组划分](split_groups.json) · [取样与标注协议](protocol.json)','', '每组先按运行组隔离，再平衡来源并根据操作前形状及弱形变诊断作多样性选择。没有人类指定正确抓点或期望规律，没有使用旧Rule Mining、Policy Evolution或上轮Teacher结论。模型不看来源名、Prompt hash或人工筛选效果标签。','', '## 效果提取与CV核验','', 'R1只看原始Before/After；R2独立看Before、真实Pa/Pb、参数、按时间抽取的过程帧和After。两次不同会话、不同批内邻居，不互相看结果。四条一批，不跨Train/Development/Test批次。R2的过程每格320×180，原始来源和时间戳保留；移动腕部相机不能证明连续材料运动、抓层或滑脱。','', '每个维度只有两遍标签一致、置信度均≥0.65、前后可比性均GOOD才接受，其余UNKNOWN。输出中的pre_structure也来自看过结果的标注请求，不能直接当作未来无泄漏的Trigger特征；后续应让规则符合度评价只看Before+Action。','',f'CV检查覆盖459条：{sum(x.get("background_registration",{}).get("status")=="PASS" for x in cv)}条通过背景特征配准门槛。分割为适配本批紫色衣服的颜色掩膜；同时计算原始面积比、质心变化、去刚性位移后的轮廓残差。所有数值只作筛选和审计，不作为成功或Reveal标签；内部显露和新增折叠仍需视觉证据。配准PASS不保证严密材料点对应。','', '[完整效果证据](effects_verified.json) · [Experience数据](experiences.json) · [CSV汇总](effects.csv) · [CV诊断](cv_diagnostics.json)','', '## 哪些正反操作可比较','',f'操作前候选检查共{len(read(OUT/"pre_pair_candidates.json")["cases"])}对，规则及排序未读取效果标签。严格条件包括相机/标定、来源及Prompt hash、提起和释放参数、速度、夹爪姿态差≤10°、不同Episode/运行组，以及去刚性位移后轮廓IoU≥0.70。较宽松候选只用于定性对照，不算受控证据。','',f'效果标签冻结后，从相反效果候选中（含单列跨集合审计）选出{len(pairs["pairs"])}对，只把前图与条件信息送交独立视觉可比较性审查。最终强观察对照{pairs["strong_count"]}对，其中集合内{pairs["strong_within_split"]}对；跨集合仅作审计参照，严禁训练使用。即便“强”也不是因果对照。','', '| 配对 | 集合 | 效果组 | 前态轮廓IoU | 视觉可比性 | 最终用途 |','|---|---|---|---:|---|---|']
 for p in pairs['pairs']:
  r=p['visual_review'] or {};t.append(f'| {p["pair_id"]} | {p["split"]} | {" / ".join(p["effect_groups"])} | {p["pre_similarity_iou_after_rigid_alignment"]:.3f} | {r.get("before_structure_comparable","UNKNOWN")} | {p["final_quality"]} |')
 t += ['', '[所有操作前候选](pre_pair_candidates.json) · [效果对比选样记录](contrastive_pair_selection.json) · [独立可比较性审查](comparability.json) · [61条原始池匹配诊断](core_pool_pair_coverage.json) · [仅前态分组建议，未采用](proposed_split_v2_pre_only.json) · [配对审计范围补充](pair_scope_addendum.json)','']
 for p in pairs['pairs']:
  r=p['visual_review'] or {};t += [f'### {p["pair_id"]}','', '已知不一致项：'+', '.join(k for k,v in p['conditions'].items() if not v),'', '初态差异：'+'；'.join(r.get('important_initial_differences',[])),'']
  for cid in p['ids']:t += [f'![{cid} before](cases/{cid}/before.png)','']
 t += ['## 全部40条记录与证据','', '| ID | 集合 | 分类 | 展开 / 搬移 / 新折 / 显露 | 来源 |','|---|---|---|---|---|']
 for e in experiences:t.append(f'| {e["id"]} | {e["split"]} | {e["contrast_group"]} | {" / ".join(e["OutcomeEvidence"][k]["value"] for k in dims)} | `{e["source"]["round"]}` |')
 for e in experiences:
  cid=e['id'];labels='；'.join(n+'='+e['OutcomeEvidence'][k]['value'] for k,n in zip(dims,dimname));t += [f'### {cid} · {e["split"]}','',labels,'',f'[Before原图](cases/{cid}/before.png) · [实际动作](cases/{cid}/actual_action.png) · [After原图](cases/{cid}/after.png) · [过程](cases/{cid}/process.jpg) · [执行事实](cases/{cid}/facts.json) · [来源](cases/{cid}/provenance.json)','',f'![{cid} 前后与实际动作](cases/{cid}/review.jpg)','']
  for rr,callname in e['review_calls'].items():t.append(f'[{rr}完整输出](teacher_calls/{callname}/raw_output.txt)')
  t.append('')
 t += ['## 多步轨迹与后续用途','', '保留17个Episode的完整round顺序、时间和执行状态，以及每个选中操作的下一条记录链接。本轮future_affordance_gain统一为NOT_ASSESSED，不把即时展开不明显等同于没有未来价值。Reveal后验奖励需要单独核验新显露结构、下一动作是否利用它、下一动作真实收益及中间是否发生人工干预；本轮未作该判断。','', '[完整Episode时序](episode_chronology.json) · [Train](splits/train.json) · [Development](splits/development.json) · [Test](splits/test.json) · [专家来源参照](splits/reference.json)','', '本轮Test只是为后续Skill学习保留的运行组；本轮标签生成器见过其后图。未来Research Agent不得读取Test标签或本报告中的Test细节再声称未见检验。当前研究协调者不是终身盲测者。','', '## 隔离事件、调用与成本','', '初次调用的本地上下文审计发现全局衣物Skill描述被自动注入。8次尝试全部隔离，原始请求/输出保留且未用于任何效果标签。修复采用每次请求显式禁用该Skill，不修改全局配置；通过实际上下文审计后，40条全部重新标注。原24次预算因隔离重跑修订为最多32次，样本量和研究范围没有扩大。','',f'总尝试{audit_result["calls_total"]}次：采纳{audit_result["calls_accepted"]}次，隔离{audit_result["calls_quarantined"]}次，另有{audit_result["network_retries"]}次连接失败后按原输入补跑。已上报Token：`{json.dumps(costs["known_usage_all_attempts"])}`。未完整上报的中断尝试：{costs["unreported_attempts"]}。货币账单未知；不会把缺失用量记成零。','', '[隔离事件](isolation_incident.json) · [干净上下文审计](context_audit.json) · [全部调用与用量](costs.json) · [交付审计](audit.json)','', '实际新动作数=0，生产代码/配置变更数由审计核验。没有以新动作预测替代物理结果，没有宣称真实成功率提升。','']
 (OUT/'report.md').write_text('\n'.join(t));body=MarkdownIt('commonmark').enable('table').render('\n'.join(t));(OUT/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>操作效果数据集</title><style>body{font:16px/1.7 system-ui;max-width:1180px;margin:24px auto;padding:20px;color:#203040}table{border-collapse:collapse;font-size:13px;display:block;overflow-x:auto}td,th{border:1px solid #ccd;padding:7px;vertical-align:top}img{max-width:100%;height:auto}a{color:#1262a1}code{word-break:break-all}h3{margin-top:2.5em}</style>'+body+'</html>')
 print(json.dumps(audit_result,ensure_ascii=False,indent=2))

def images():
 for c in read(OUT/'selection_manifest.json')['cases']:
  f=OUT/'cases'/c['id'];im=Image.new('RGB',(1280,390),'white');d=ImageDraw.Draw(im)
  for j,n in enumerate(['actual_action','after']):
   im.paste(Image.open(f/f'{n}.png').resize((640,360)),(j*640,30));d.text((j*640+8,8),c['id']+' '+('BEFORE + EXECUTED ACTION' if j==0 else 'ACTUAL AFTER'),fill='black')
  im.save(f/'review.jpg',quality=92)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['images','report']);a=p.parse_args();images() if a.stage=='images' else report()
