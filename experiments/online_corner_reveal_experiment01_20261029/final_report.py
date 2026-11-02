from pathlib import Path
import json,html
R=Path('/mnt/newssd/sja/clothAgent_real/reports/online_corner_reveal_experiment01_20261029')
def read(p):return json.loads((R/p).read_text())
p=read('prediction_before_execution.json');e=read('live_execution/execution.json');o=read('outcome.json');g=read('planned_action_05_min_z_minus14p9/geometry.json')
result={'physical_actions':1,'outcome':o['outcome'],'execution':e,'prediction':p,'outcome_analysis':o,'geometry':g,'minimum_physical_tool_z_mm':-14.9,'second_action':False,'policy_updated':False}
(R/'experiment_status.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
body='<h1>Corner Reveal Exploration 01：一次真实动作</h1><p>已执行 1 次，已停止；没有重置、补抓或第二次探索。</p><h2>Before → After</h2><div class="pair"><div><h3>Before</h3><img src="before_capture/observation/rgb.png"></div><div><h3>After</h3><img src="live_execution/after/rgb.png"></div></div><h2>动作 Pa → Pb</h2><img src="action_overlay.png">'
for title,v in [('执行前冻结预测',p['proposal']),('真实结果与RSI分析',o),('实际动作与执行日志',{'geometry':g,'execution':e,'physical_tool_minimum_z_mm':-14.9})]:body+='<h2>'+title+'</h2><pre>'+html.escape(json.dumps(v,ensure_ascii=False,indent=2))+'</pre>'
body+='<p>当前仅一次新状态探索，不证明机制普遍成立；未合入正式策略。只有前后图，不声称观察到了抬升和牵拉中间阶段。此前拒绝与配置修正均保留。</p>'
(R/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Reveal Experiment 01 result</title><style>body{max-width:1500px;margin:25px auto;font:18px sans-serif}.pair{display:flex;gap:10px}.pair>div{width:50%}img{width:100%}pre{white-space:pre-wrap;background:#eee;padding:16px}</style>'+body)
text='# Corner Reveal Exploration 01：真实执行结果\n\n只执行1次，已结束；无重置/补抓。\n\n结果：'+o['outcome']+'\n\n'
for k,v in o.items():text+='## '+k+'\n\n'+v+'\n\n'
text+='\n边界：Arm7 已被用户物理移除。夹爪最低实体点允许Z=-14.9 mm，按用户说明消除重复基座余量，保留制动检查。该配置下完整抓取及返回观测路径通过；未修改生产控制器。先前未执行结果保留在各planned_action目录。\n'
(R/'final_summary.md').write_text(text)
print(R/'index.html')
