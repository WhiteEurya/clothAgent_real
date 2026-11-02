from pathlib import Path
import json,html,hashlib
R=Path('/mnt/newssd/sja/clothAgent_real/reports/hindsight_skill_discovery_20261029_01');ROOT=R.parents[1]
def read(p):return json.loads((R/p).read_text())
p=read('prediction_before_execution.json');o=read('outcome.json') if (R/'outcome.json').exists() else {'outcome':'UNCERTAIN','conclusion':'Outcome analysis unavailable; inspect raw evidence.'};e=read('live_execution/status.json') if (R/'live_execution/status.json').exists() else {'physical_actions':0,'status':'NOT_EXECUTED'}
protocol=read('execution_protocol.json');unchanged=all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in protocol['protected_sha256'].items())
res={'physical_actions':e['physical_actions'],'execution_status':e,'outcome':o,'prediction':p,'protected_files_unchanged':unchanged,'formal_skill_updated':False,'second_action':False};(R/'experiment_status.json').write_text(json.dumps(res,ensure_ascii=False,indent=2))
body='<h1>Hindsight Skill Discovery：一次前瞻检验</h1><p>原始经验 → 独立新会话假设 → 新RGB定位 → 冻结预测 → 一次真实动作 → 独立结果分析。没有输入此前机制解释，没有更新正式Skill。</p>'
for f,t in [('e353_before.png','E353 Before'),('e353_after.png','E353 After'),('online01_before.png','前次在线 Before'),('online01_after.png','前次在线 After'),('before_capture/observation/rgb.png','本次新采集 Before'),('planned_action_01/action_overlay.png','本次 Pa → Pb'),('live_execution/after/rgb.png','本次 After'),('process_1.jpg','过程前半段'),('process_2.jpg','过程后半段')]:
 if (R/f).exists():body+='<h2>'+t+'</h2><a href="'+f+'"><img src="'+f+'"></a>'
for t,v in [('执行前冻结的比较、假设与预测',p['proposal']),('真实结果与假设检验',o),('执行状态',e),('范围与配置',{'unchanged':unchanged,'one_action_only':True,'current_state_continues_previous_online_trial':True,'not_independent_reset':True,'configuration':protocol})]:body+='<h2>'+t+'</h2><pre>'+html.escape(json.dumps(v,ensure_ascii=False,indent=2))+'</pre>'
body+='<p><a href="calls/jpeg_isolated_hindsight_proposal/input.txt">提议会话原始输入</a> · <a href="calls/jpeg_isolated_outcome_analysis/input.txt">评价会话原始输入</a></p>'
(R/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Hindsight discovery 01</title><style>body{max-width:1280px;margin:30px auto;font:18px sans-serif;line-height:1.6}img{width:100%}pre{white-space:pre-wrap;padding:18px;background:#eee}</style>'+body)
md='# Hindsight Skill Discovery 01\n\n单次实机检验，非独立重置对照；无正式Skill更新。\n\n'
for k,v in p['proposal'].items():md+='## 执行前：'+k+'\n\n'+str(v)+'\n\n'
for k,v in o.items():md+='## 结果：'+k+'\n\n'+str(v)+'\n\n'
(R/'report.md').write_text(md);print(R/'index.html')
