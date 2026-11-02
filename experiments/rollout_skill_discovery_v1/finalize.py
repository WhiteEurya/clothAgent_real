"""Summarize recorded outputs; no model calls."""
import json,html
from pathlib import Path
import run as r
O=r.OUT
proto=r.read(O/'protocol.json');freeze=r.read(O/'freeze.json');skill=r.read(O/'skill_frozen.json')
assert r.sha(O/'skill_frozen.json')==freeze['sha256']
checks=[]
def check(name,v):checks.append({'check':name,'pass':bool(v)})
check('disjoint run groups',not {p.split('/')[1] for p in r.TRAIN}&{p.split('/')[1] for p in r.VALID})
check('skill <=150 Han characters',freeze['han_characters']<=150)
check('no fixed coordinates in skill',not any(c.isdigit() for s in skill.values() for c in s))
all_rows=[];vs=[]
for j,p in enumerate(r.VALID,1):
 source,_,_=r.collect([p],f'V{j}_',True);lookup={x['id']:x for x in source[0]['operations_in_original_order']}
 v=r.read(O/f'validation_{j}/parsed.json');rows=[]
 check(f'V{j} complete ordered record coverage',[x['id'] for x in v['operations']]==list(lookup))
 for x in v['operations']:
  src=lookup[x['id']];row={**x,'execution_status':src['execution_status'],'physical_execution':src['physical_execution'],'has_after':bool(src.get('after_attachment')),'action':src['action']}
  rows.append(row);all_rows.append(row)
 vs.append({'rollout':f'V{j}','source':p,'initial_difficulty':v['initial_difficulty'],'trajectory_uncertainty':v['trajectory_uncertainty'],'overall_assessment':v['overall_assessment'],'records':rows})
for name in ['teacher','validation_1','validation_2','validation_3']:
 f=O/name;d=r.read(f/'request_payload.json');raw=r.read(f/'raw_response.json');m=r.read(f/'model_metrics.json')
 check(name+' isolated request',len(d['input'])==1 and d['tools']==[] and 'previous_response_id' not in d and d['instructions']==r.BASE)
 check(name+' model config',raw['model']=='gpt-6-astra' and raw['reasoning']['effort']=='high')
 check(name+' no tool outputs',all(x['type'] in ['message','reasoning'] for x in raw['output']))
 check(name+' one response',m['model_response_count']==1)
 check(name+' attachments unchanged',all(r.sha(r.ROOT/a['path'])==a['sha256'] for a in r.read(f/'attachments.json')))
 if name!='teacher':check(name+' after freeze',r.read(f/'attempt.json')['started_at']>freeze['at'])
executed=[x for x in all_rows if x['execution_status']=='EXECUTED'];matched=[x for x in executed if x['trigger']=='YES' and x['action_conforms']=='YES']
def metrics(rows):
 m=[x for x in rows if x['trigger']=='YES' and x['action_conforms']=='YES']
 return {'executed_actions':len(rows),'trigger_yes':sum(x['trigger']=='YES' for x in rows),'trigger_unknown':sum(x['trigger']=='UNKNOWN' for x in rows),'matched':len(m),'matched_effects':{k:sum(x['expected_effect_observed']==k for x in m) for k in ['YES','NO_CLEAR_EVIDENCE','UNKNOWN']},'effect_unknown':sum(x['expected_effect_observed']=='UNKNOWN' for x in rows)}
summary={'skill':skill,'skill_length':freeze,'all_records':len(all_rows),'primary_denominator':'EXECUTED actual actions only; terminal and rejected records retained separately','metrics':metrics(executed),'rollouts':[{**v,'metrics':metrics([x for x in v['records'] if x['execution_status']=='EXECUTED'])} for v in vs]}
r.save(O/'validation_summary.json',summary);r.save(O/'validation_checks.json',{'checks':checks,'failed':sum(not x['pass'] for x in checks)})
assert all(x['pass'] for x in checks)
lines=['三轨迹 Skill 归纳 + 三轨迹验证（真实离线 GPT 实验）','',f"模型 gpt-6-astra/high；1 次 Teacher + 3 次独立评价调用。Skill 未修订；{freeze['han_characters']} 个汉字，含标点共 {freeze['characters']} 字符。",'','冻结 Skill：']+[f'{k}: {v}' for k,v in skill.items()]
lines+=['','归纳数据：']+[f'T{i}: {p}' for i,p in enumerate(r.TRAIN,1)]
lines+=['三条轨迹各含5次完整执行，保留终止记录；T1另有1次部分执行但缺After，明确未知。依据已有Train标签与图像选择有明显进展轨迹，不意味着所有步骤成功。Teacher没有读取标签或专家Prompt，也未获得历史Planner理由。','',f"验证：{len(all_rows)}条顺序记录，其中{len(executed)}个完整真实动作。"]
for v in summary['rollouts']:
 lines += ['',v['rollout']+' '+v['source'],json.dumps(v['metrics'],ensure_ascii=False),'初始难度：'+v['initial_difficulty'],'不确定性：'+v['trajectory_uncertainty'],'评价器总结：'+v['overall_assessment']]
 for x in v['records']:lines += [f"  {x['id']} [{x['execution_status']}] trigger={x['trigger']}; conforms={x['action_conforms']}; effect={x['expected_effect_observed']}",'    Before: '+x['before_evidence'],'    After: '+x['after_evidence'],'    不确定性: '+x['uncertainty']]
lines+=['','总体计数：'+json.dumps(summary['metrics'],ensure_ascii=False),'','解释边界：','这是独立轨迹上的事后匹配与可见效果检验，评价器可见After和后续状态；不是严格盲测预测。','没有No Experience或Action-only对照，不能证明规则来自真实反馈而非模型先验，也不能声称比先验更有预测价值。','UNKNOWN单独保留，不计为失败。样本少、轨迹内动作相关、正向进展选择偏倚，不宣称统计显著性或物理因果。','不同初始难度、固定/可变夹爪方向、距离、层数与遮挡是混杂因素。没有新增实机执行或Planner Replay，不能声称成功率提高。','Teacher真实请求包含34张原始图；无Agent CLI、无工具或历史会话ID。服务端内部上下文不可审计。','一次沙箱DNS失败发生在收到任何模型响应之前，已隔离；提权后发送相同Teacher请求，没有按结果反复采样。','所有请求体、原始响应、附件清单、Token与耗时保存在本目录。']
(O/'report.txt').write_text('\n'.join(lines)+'\n')
h=['<!doctype html><meta charset="utf-8"><title>Rollout Skill Discovery</title><style>body{font:16px sans-serif;max-width:1200px;margin:35px auto;line-height:1.6}img{max-width:100%}pre{white-space:pre-wrap}td,th{border:1px solid #ccc;padding:8px}table{border-collapse:collapse}</style><h1>三轨迹 Skill 归纳与独立验证</h1>', '<pre>'+html.escape('\n'.join(lines[:12]))+'</pre>']
for v in summary['rollouts']:
 h+=['<h2>'+v['rollout']+'</h2><p>'+html.escape(v['initial_difficulty'])+'</p><p>'+html.escape(v['trajectory_uncertainty'])+'</p><table><tr><th>记录</th><th>触发</th><th>动作符合</th><th>可见效果</th><th>证据／不确定性</th></tr>']
 for x in v['records']:h+=['<tr>'+''.join('<td>'+html.escape(str(z))+'</td>' for z in [x['id']+' '+x['execution_status'],x['trigger'],x['action_conforms'],x['expected_effect_observed'],x['before_evidence']+' / '+x['after_evidence']+' / '+x['uncertainty']])+'</tr>']
 h+=['</table><img src="review/'+v['rollout']+'_1.jpg">']
h+=['<h2>归纳轨迹</h2>']+['<h3>'+n+'</h3><img src="review/'+n+'.jpg">' for n in ['T1','T2','T3']]
(O/'review.html').write_text('\n'.join(h))
print(json.dumps(summary['metrics'],ensure_ascii=False));print('checks',len(checks),'passed')
