"""Local Markdown/HTML evidence report and audit. No model requests."""
from common import *
import collections,html

def audit():
 verify_freezes();records=[];total=collections.Counter();errors=[]
 import jsonschema
 for f in sorted((OUT/'teacher_calls').glob('*/result.json')):
  r=read(f);r={'call':f.parent.name,**r};records.append(r)
  if r['status']=='COMPLETE':
   jsonschema.validate(read(f.parent/'parsed.json'),read(f.parent/'schema.json'))
   for k,v in (r.get('usage') or {}).items():
    if isinstance(v,(int,float)):total[k]+=v
  if r.get('tool_events'):errors.append({'call':f.parent.name,'tool_events':r['tool_events']})
 manifest=read(OUT/'data_manifest.json');groups=[c['group'] for c in manifest['cases']]
 checks={'six_distinct_groups':len(set(groups))==6,'freeze_integrity':True,'schema_valid':True,'tool_events':errors,'leakage':[],'protected_changes':[],'input_integrity_errors':[]}
 for f in (OUT/'teacher_calls').glob('*/input_manifest.json'):
  m=read(f);name=f.parent.name
  if sha(f.parent/'prompt.txt')!=m['prompt_sha256']:checks['input_integrity_errors'].append(name+':prompt')
  for item in m['images']+m['transport_images']:
   if sha(OUT/item['file'])!=item['sha256']:checks['input_integrity_errors'].append(item['file'])
  if name in ['01_observe','02_hypothesize','03_compress']:
   if any('/H' in x['file'] for x in m['images']):checks['leakage'].append(name)
  if '_predict_' in name or name.startswith('planner_'):
   if any(Path(x['file']).name!='before.png' for x in m['images']):checks['leakage'].append(name)
  text=(f.parent/'prompt.txt').read_text()
  for marker in ['hem_sleeve_pull_rule','offline_policy_evolution_v0','failure_driven_rsi_v1','R1_D1_局部内移对照']:
   if marker in text:checks['leakage'].append({'call':name,'marker':marker})
 for rel,s in read(OUT/'protected_before.json').items():
  if not (ROOT/rel).exists() or sha(ROOT/rel)!=s:checks['protected_changes'].append(rel)
 if (OUT/'freezes/candidate_skill.json').exists():
  t=read(OUT/'freezes/candidate_skill.json')['at'];checks['rule_frozen_at']=t;checks['predictions_after_rule']=all(read(f)['at']>t for f in (OUT/'teacher_calls').glob('*_predict_*/attempt.json'))
 if (OUT/'freezes/predictions.json').exists():
  t=read(OUT/'freezes/predictions.json')['at'];checks['predictions_frozen_at']=t;checks['all_reveals_after_all_predictions']=all(read(f)['at']>t for f in (OUT/'teacher_calls').glob('evaluate_*/attempt.json'))
 checks['all_profile_hashes_same']=len({read(f)['profile_sha256'] for f in (OUT/'teacher_calls').glob('*/input_manifest.json')})==1
 checks['request_attempts']=len(list((OUT/'teacher_calls').glob('*/attempt.json')));checks['budget_pass']=checks['request_attempts']<=16
 checks['all_calls_complete']=len(records)==15 and all(r['status']=='COMPLETE' for r in records)
 checks['process_timestamp_checks']={f.parent.name:read(f) for f in (OUT/'cases').glob('*/data_check.json')}
 checks['all_process_times_valid']=len(checks['process_timestamp_checks'])==6 and all(d['real'] and d['all_frames_within_pre_post'] and d['frame_timestamps_ordered'] for d in checks['process_timestamp_checks'].values())
 checks['evaluation_clarification_before_reveal']=all(read(f)['at']>read(OUT/'evaluation_clarification.json')['at'] for f in (OUT/'teacher_calls').glob('evaluate_*/attempt.json'))
 checks['audit_pass']=all(checks[k] for k in ['six_distinct_groups','freeze_integrity','schema_valid','predictions_after_rule','all_reveals_after_all_predictions','all_profile_hashes_same','budget_pass','all_calls_complete','all_process_times_valid','evaluation_clarification_before_reveal']) and not any(checks[k] for k in ['tool_events','leakage','protected_changes','input_integrity_errors'])
 cost={'completed':sum(r['status']=='COMPLETE' for r in records),'attempts':checks['request_attempts'],'usage':dict(total),'wall_s_sum':sum(r['wall_s'] for r in records),'currency_cost':None,'currency_cost_note':'Provider billing/prices unavailable; token counts reported without inventing cost. cache_write tokens are subset metadata, not additive input.','calls':records}
 (OUT/'audit.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2));(OUT/'costs.json').write_text(json.dumps(cost,ensure_ascii=False,indent=2))
 return checks,cost

def comparisons():
 out=[]
 for f in sorted((OUT/'planner_comparison').glob('H*')):
  if not (f/'comparison.json').exists():continue
  cid=f.name;a=read(f/'A.json');b=read(f/'B.json');sa=read(f/'A_static.json');sb=read(f/'B_static.json')
  ta=read(OUT/'teacher_calls'/f'planner_{cid}_A'/'result.json')['wall_s'];tb=read(OUT/'teacher_calls'/f'planner_{cid}_B'/'result.json')['wall_s']
  out.append({'id':cid,'A':a,'B':b,'A_static':sa,'B_static':sb,'delta':read(f/'comparison.json'),'A_wall_s':ta,'B_wall_s':tb})
 result={'cases':out,'A_mean_s':float(np.mean([c['A_wall_s'] for c in out])) if out else None,'B_mean_s':float(np.mean([c['B_wall_s'] for c in out])) if out else None,'causality_limit':'Single sample per A/B; endpoint changes may include model sampling variation. Neither proposed action executed.'}
 if out:result['B_minus_A_mean_s']=result['B_mean_s']-result['A_mean_s']
 (OUT/'planner_comparison/summary.json').parent.mkdir(exist_ok=True);(OUT/'planner_comparison/summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
 return result

def make_report():
 a,cost=audit();comp=comparisons();manifest=read(OUT/'data_manifest.json');evaluations=read(OUT/'heldout_evaluation.json')['cases'] if (OUT/'heldout_evaluation.json').exists() else []
 text=['# Failure-driven RSI Pilot','',f'独立离线实验；全部模型经 `codex exec -p api`（gpt-6-astra / high）调用。完成 {cost["completed"]} 次请求，共 {cost["attempts"]} 次尝试。','']
 if (OUT/'conclusion.txt').exists():text += [(OUT/'conclusion.txt').read_text(),'']
 inventory=read(OUT/'inventory.json');eligible=read(OUT/'eligible_episodes.json')
 text += [f'数据盘点：扫描 {len(inventory)} 条 round，{sum(r["eligible"] for r in inventory)} 条满足本轮完整性条件，覆盖 {len(eligible)} 个可用 Episode、{len(set(e["group"] for e in eligible))} 个运行组；本轮使用 6 个不同 Episode、6 个不同运行组。','']
 text+=['## 设计与证据边界','','本轮按“观察 → 竞争性物理假设 → 单条规则压缩 → 冻结预测 → Planner A/B → 独立揭示评价”运行。Planner提前于揭示评价运行，以便同一独立评价请求同时审查B的触发依据；所有A/B只见当前前图。没有在线动作、生产Prompt、Host、IK或安全配置修改。','',
 '扫描原始 `reports/*/episode_*/round_*`，先固定三个独立运行组留出，再对九条发现候选及下一小批三条看原始前后图，选择所需的结果多样性。筛选标签没有进入Teacher请求。只选择当前实验未用于发现的留出；部分档案可能在以前实验使用，不声称全局从未见过。Teacher均是全新会话且不接收旧结论。历史执行控制器可能受人工规则影响，独立归纳不等于示范来源没有专家影响。','',
 '[协议](protocol.json) · [数据来源](data_manifest.json) · [完整盘点](inventory.json) · [筛选理由](screening_judgments.json) · [首批筛选](discovery_screening_pool.json) · [补充筛选](screening_extension_protocol.json) · [隔离说明](independence_note.json)','',
 '## 真实操作与数据完整性','','| ID | 原始 Episode / round | Pa → Pb | 提拉高度 | 过程帧（动作/撤回） |','|---|---|---|---|---|']
 for c in manifest['cases']:
  f=OUT/'cases'/c['id'];d=read(f/'facts.json') if (f/'facts.json').exists() else read(f/'pre.json') if (f/'pre.json').exists() else {}
  counts=d.get('available_process_counts',{});text.append(f'| {c["id"]} | `{c["round"]}` | {d.get("Pa")} → {d.get("Pb")} | {d.get("geometry",{}).get("primitive",{}).get("lift_height_mm","未揭示")} mm | {counts.get("execution_images",0)} / {counts.get("return_perception/execution_images",0)} |')
 text+=['','六例均有真实前后图、实际Pa/Pb、参数、9条命令完成状态、执行与撤回过程帧和有序时间戳，足以还原命令级操作及前后结果；执行记录均为EXECUTED，布料获取标签仍为UNKNOWN_PENDING_HUMAN_REVIEW。无法完整还原连续抓取、层间滑动及释放的物理细节。','', '原始before/after为1280×720；模型附件保留尺寸并转为JPEG，原始PNG另存。过程帧按固定等距索引，加上最接近夹爪反馈完成时间的帧，每例展示15帧；时间标签不等于接触或动作开始时刻。每张过程总览缩放至320×180/格，原尺寸抽样帧也保存。腕部相机视角变化、遮挡和抽帧间隔限制层间运动判断。没有仅凭命令完成或机械检测认定真实抓取。','']
 for c in manifest['cases']:
  cid=c['id'];f=OUT/'cases'/cid
  if not f.exists():continue
  text += [f'### {cid}','',f'[实际动作标注](cases/{cid}/actual_action.png) · [前图](cases/{cid}/before.png)'+(f' · [后图](cases/{cid}/after.png) · [过程](cases/{cid}/process.jpg) · [完整事实](cases/{cid}/facts.json) · [来源哈希](cases/{cid}/provenance.json)' if (f/'facts.json').exists() else ''),'',f'![{cid} before](cases/{cid}/before.png)','']
  if (f/'after.png').exists():text += [f'![{cid} after](cases/{cid}/after.png)','',f'![{cid}过程](cases/{cid}/process.jpg)','']
 text+=['## 三次独立 Teacher 输出','','[第一次完整原始输出：观察](teacher_calls/01_observe/raw_output.txt) · [observations.json](observations.json)','', '[第二次完整原始输出：竞争性假设](teacher_calls/02_hypothesize/raw_output.txt) · [hypotheses.json](hypotheses.json)','', '[第三次完整原始输出：压缩规则](teacher_calls/03_compress/raw_output.txt) · [候选规则冻结记录](freezes/candidate_skill.json)','']
 text += ['下图左为历史动作，右为Teacher提出但未执行的替代动作。同一抓点、只反转图像水平位移分量；不是机器人效果对比。','', '![D1历史动作与未执行替代动作](alternative_operation/D1_comparison.png)','', '[替代动作的有限静态检查](discovery_static_audit.json)','']
 if (OUT/'candidate_skill.json').exists():
  text+=['### 唯一候选 Skill（模型原文，未改写）','']
  for k,v in read(OUT/'candidate_skill.json').items():text += [f'**{k}**：{v}','']
 text+=['## 留出预测与独立揭示评价','','三个预测全部保存并冻结后，才读取后图与过程证据。NOT_APPLICABLE表示适用前提不成立；INCONCLUSIVE包括缺证据，不是反例。符合现象预测不代表物理原因被确认。若历史没有执行建议的替代动作，不能证明替代操作更好。','', '| 案例 | 独立结论 | 预测分项 | 替代操作有效性 |','|---|---|---|---|']
 for e in evaluations:
  checks='；'.join(x['prediction_id']+': '+x['verdict'] for x in e['prediction_checks']);text.append(f'| {e["id"]} | {e["overall_verdict"]} | {checks} | {e["alternative_operation_effectiveness"]} |')
 text+=['','[完整留出评价](heldout_evaluation.json) · [预测冻结哈希](freezes/predictions.json)','']
 text += ['预测原文有将D1示例绝对坐标过度绑定到留出案例的倾向。揭示结果前已冻结[评价口径澄清](evaluation_clarification.json)：区分示例坐标与规则实际条件，分别判断Trigger、动作匹配和现象预测；没有更改规则、预测或放宽历史证据要求。','']
 for e in evaluations:text += [f'**{e["id"]}**：{e["applicability"]} 反例判断：{e["hypothesis_counterexample"]}',f'[原始预测](predictions/{e["id"]}.json) · [独立评价原文](teacher_calls/evaluate_{e["id"]}/raw_output.txt)','']
 text+=['## Planner A/B：相同观测与接口','','基础Planner使用现有独立GPT调用设施和最小任务/动作接口，不载入生产FULL Prompt。B唯一额外输入为冻结Skill。两组每例各一次，调用顺序AB、BA、AB。规则需要但当前单帧未提供的时间上下文必须保持未知，不能强行触发。','', '| 案例 | A Pa → Pb | B Pa → Pb | B触发 | 静态A/B | Pa/Pb差与方向差 |','|---|---|---|---|---|---|']
 for row in comp['cases']:
  aa=row['A'];bb=row['B'];dd=row['delta'];delta=f'{dd["Pa_delta_px"]:.1f} / {dd["Pb_delta_px"]:.1f} px；{dd["direction_delta_deg"]:.1f}°' if 'Pa_delta_px' in dd else str(dd);text.append(f'| {row["id"]} | {aa["Pa"]} → {aa["Pb"]} ({aa["decision"]}) | {bb["Pa"]} → {bb["Pb"]} ({bb["decision"]}) | {bb["rule_trigger"]} | {row["A_static"]["status"]}/{row["B_static"]["status"]} | {delta} |')
 text+=['',f'平均请求墙钟：A={comp["A_mean_s"]} 秒，B={comp["B_mean_s"]} 秒，差={comp.get("B_minus_A_mean_s")} 秒。包括CLI/网络开销，不能拆成纯推理时间。单次采样无法排除模型波动；行为变化不直接算性能改善。','',
 '静态检查包括格式、图内像素、粗略布料颜色、原始深度/支撑平面、毫米距离与归档TCP端点边界；没有运行IK、关节限制、碰撞或路径检查，PASS不能当作完整安全认证。','']
 for row in comp['cases']:
  cid=row['id'];text += [f'### {cid} 动作对比','',f'![{cid} A/B，均未执行](planner_comparison/{cid}/AB.png)','',f'[A完整图](planner_comparison/{cid}/A.png) · [B完整图](planner_comparison/{cid}/B.png) · [A原文与动作](planner_comparison/{cid}/A.json) · [B原文与动作](planner_comparison/{cid}/B.json) · [A静态检查](planner_comparison/{cid}/A_static.json) · [B静态检查](planner_comparison/{cid}/B_static.json)','']
  e=next((e for e in evaluations if e['id']==cid),None)
  if e:text += ['独立触发审查：'+json.dumps(e['planner_trigger_audit'],ensure_ascii=False),'']
 text+=['## 调用、成本与验证','','| 请求 | 状态 | 墙钟秒 | 输入Token | 输出Token | 原始材料 |','|---|---|---|---|---|---|']
 for r in cost['calls']:
  n=r['call'];u=r.get('usage') or {};text.append(f'| {n} | {r["status"]} | {r["wall_s"]:.2f} | {u.get("input_tokens")} | {u.get("output_tokens")} | [输入](teacher_calls/{n}/prompt.txt) / [附件](teacher_calls/{n}/input_manifest.json) / [输出](teacher_calls/{n}/raw_output.txt) / [事件](teacher_calls/{n}/events.jsonl) |')
 text+=['',f'合计：{json.dumps(cost["usage"],ensure_ascii=False)}。货币账单未知，不虚构费用。','',
 '[完整成本](costs.json) · [冻结与泄漏审计](audit.json) · [交付验证](validation.json) · [模型可见上下文审计](context_audit.json) · [复用代码来源](code_reuse.json)','',
 'Codex CLI仍带通用工具说明和系统技能目录，但没有项目专家语义；接受的输出要求实际工具调用为零。输入、输出和重要阶段均存SHA256冻结记录。没有声称独立请求的API载荷完全不含通用harness元数据。','',
 '未执行新动作的真实物理效果全部未知。候选规则生成不等于已经实现RSI。','']
 (OUT/'report.md').write_text('\n'.join(text))
 # Local HTML companion rendered without publishing; full original report remains Markdown.
 try:
  import markdown
  body=markdown.markdown('\n'.join(text),extensions=['tables','fenced_code'])
 except ImportError:
  from markdown_it import MarkdownIt
  body=MarkdownIt('commonmark',{'html':False}).enable('table').render('\n'.join(text))
 (OUT/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Failure-driven RSI Pilot</title><style>body{font:16px/1.7 system-ui;max-width:1180px;margin:30px auto;padding:20px;color:#203040}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #ddd;padding:7px;vertical-align:top}img{max-width:100%;height:auto}pre{white-space:pre-wrap;word-break:break-word}a{color:#165da8}</style>'+body+'</html>')
 print(json.dumps({'completed':cost['completed'],'attempts':cost['attempts'],'audit':a,'comparisons':comp},ensure_ascii=False,indent=2))

if __name__=='__main__':make_report()
