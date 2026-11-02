from pathlib import Path
import json,html
from PIL import Image,ImageDraw
R=Path('/mnt/newssd/sja/clothAgent_real/reports/online_corner_reveal_experiment01_20261029')
p=json.loads((R/'prediction_before_execution.json').read_text())
a=p['proposal']; im=Image.open(R/'before_capture/observation/rgb.png').convert('RGB');d=ImageDraw.Draw(im)
if a['decision']=='EXPLORATORY_ACTION':
 x,y=a['grasp_px'],a['target_px'];d.line([tuple(x),tuple(y)],fill='yellow',width=5)
 for q,c,t in [(x,'lime','Pa'),(y,'red','Pb')]:
  d.ellipse((q[0]-7,q[1]-7,q[0]+7,q[1]+7),fill=c);d.text((q[0]+10,q[1]),t+str(q),fill=c)
im.save(R/'action_overlay.png')
check=R/'planned_action_04_tool_boundary/result.json';s=json.loads(check.read_text()) if check.exists() else {'status':'PENDING'}
geo=R/'planned_action_04_tool_boundary/geometry.json';g=json.loads(geo.read_text()) if geo.exists() else {}
status='尚未执行：等待通过安全检查及人工确认。' if s['status']=='SAFETY_PLAN_PASSED_PENDING_HUMAN_APPROVAL' else '尚未执行；未获得通过的安全计划，不能检验显露效果。'
report={'physical_actions':0,'outcome':'UNCERTAIN','outcome_reason':'NOT_EXECUTED','status':s,'prediction':p,'geometry':g,'scope':'Experimental proposal despite candidate algorithm ABSTAIN. No learned grasp-side selection claim; no new physical result.','next_test':'只有通过完整安全检查并获人工确认后才可测试该冻结预测；不能绕过拒绝。'}
(R/'experiment_status.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
body='<h1>Corner Reveal Exploration 01</h1><p>'+status+'</p><p>目标：衣服语义右袖口。用户声明其自遮挡；图像判断与层序推断保留不确定性。</p><h2>Before / 提议动作</h2><img src="before_capture/observation/rgb.png"><img src="action_overlay.png"><h2>执行前冻结预测</h2><pre>'+html.escape(json.dumps(a,ensure_ascii=False,indent=2))+'</pre><h2>几何与安全检查</h2><pre>'+html.escape(json.dumps({'geometry':g,'safety':s},ensure_ascii=False,indent=2))+'</pre><h2>结论边界</h2><p>实际动作 0。没有 After，不能判断该动作能否显露袖口，也不能验证与 E353 的物理机制相同。没有更新正式策略。</p>'
(R/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Reveal Experiment 01</title><style>body{max-width:1300px;margin:30px auto;font:18px sans-serif}img{width:100%}pre{white-space:pre-wrap;background:#eee;padding:20px}</style>'+body)
print(R/'index.html')
