"""Render the independent single-case analysis; no additional inference."""
import ast,json,html,os
from pathlib import Path
from PIL import Image as PILImage
from reportlab.platypus import Image,PageBreak,SimpleDocTemplate
from reportlab.lib.pagesizes import A4
ROOT=Path('/mnt/newssd/sja/clothAgent_real');R=ROOT/'reports/x0450_discrepancy_20261102'
def load(p):return json.loads(Path(p).read_text())
def pretty(x):return json.dumps(x,ensure_ascii=False,indent=2)
def esc(x):return html.escape(str(x))
def pre(x):return '<pre>'+esc(pretty(x))+'</pre>'
a=load(R/'result.json');f=a['analysis'];src=a['source'];dispatch=load(R/'original_dispatch.json');action=load(R/'original_action.json')
cost={k:load(R/'calls'/k/'result.json') for k in ['intent','observation','discrepancy_and_bank_audit']}
cost={k:{'latency_s':round(v['latency_s'],2),'usage':v['usage']} for k,v in cost.items()};(R/'costs.json').write_text(pretty(cost))
def rel(p):return os.path.relpath(p,R)
def img(p):return '<a href="'+esc(rel(p))+'"><img src="'+esc(rel(p))+'"></a>'
parts=['<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>X0450 Intention–Outcome Discrepancy</title><style>body{font:16px/1.7 system-ui;max-width:1150px;margin:auto;padding:25px;background:#f3f6fa;color:#183247}section{padding:20px;margin:16px 0;background:white;border:1px solid #ddd}pre{white-space:pre-wrap;overflow-wrap:anywhere}img{width:100%}.pair{display:grid;grid-template-columns:1fr 1fr;gap:14px}.warn{background:#fff0cc;padding:16px}a{color:#176eb5}@media(max-width:700px){.pair{display:block}}</style><h1>X0450：独立意图—结果差异分析</h1><p><a href="report.pdf">PDF</a> · <a href="result.json">完整JSON</a> · <a href="original_dispatch.json">原始dispatch记录</a> · <a href="existing_hypotheses.json">已有两条假设</a></p>',
 '<div class="warn">只分析X0450。意图提取与图像观察为分离上下文，均未获得上一轮预测或评价。无机器人、无Skill、未修改假设库。</div>',
 '<section><h2>结论</h2><p>'+esc(f['final_answer'])+'</p>'+pre({k:f[k] for k in ['worth_followup','new_discovery_status','confirmed_corner_discovery']})+'</section>',
 '<section><h2>1. Original Intention</h2>'+pre(dispatch['plan'])+'<p>'+esc(f['original_intention_summary'])+'</p>'+pre(a['intent'])+'</section>',
 '<section><h2>2. 原始 Before / After</h2><div class="pair">'+img(src['before'])+img(src['after'])+'</div><p>'+esc(f['observed_outcome_summary'])+'</p>'+pre(a['observation'])+'</section>',
 '<section><h2>3. Expectation Gap</h2>'+pre({k:f[k] for k in ['expectation_gap','gap_evidence','intended_goal_status']})+'</section>',
 '<section><h2>4. Hypothesis Opportunity</h2><p>'+esc(f['hypothesis_opportunity'])+'</p>'+pre(f['competing_explanations'])+'<p>以上是本次报告中的待检验解释，不添加为新的Hypothesis Bank条目，不计作已学会技能。</p></section>',
 '<section><h2>5. 已有知识与重复发现审计</h2>'+pre(f['existing_hypothesis_audit'])+'<h3>已有观察与本次独立复核</h3><p>'+esc(f['existing_observation_vs_new_review'])+'</p><details><summary>原假设全文，保持原样</summary>'+pre(a['existing_hypotheses'])+'</details></section>',
 '<section><h2>来源、动作和证据边界</h2><p>'+esc(src['source'])+'</p><p><a href="'+esc(rel(Path(src['source'])/'dispatch_intent.json'))+'">原始派发意图</a> · <a href="'+esc(rel(Path(src['source'])/'execution.json'))+'">原始执行记录</a></p>'+pre(action)+'<h3>限制</h3>'+pre(f['limits'])+'<h3>隔离与成本</h3>'+pre(load(R/'protocol.json'))+pre(cost)+'</section></html>'];(R/'review.html').write_text(''.join(parts))
tree=ast.parse((ROOT/'experiments/hypothesis_audit_20261102/export_pdf.py').read_text());nodes=[]
for x in tree.body:
 if isinstance(x,(ast.Import,ast.ImportFrom,ast.FunctionDef)):nodes.append(x)
 elif isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='styles' for t in x.targets):nodes.append(x)
 elif isinstance(x,ast.Expr) and isinstance(x.value,ast.Call) and isinstance(x.value.func,ast.Attribute) and x.value.func.attr=='registerFont':nodes.append(x)
ns={};exec(compile(ast.Module(body=nodes,type_ignores=[]),__file__,'exec'),ns);p=ns['p']
def image(k):
 folder=Path(load(R/'calls/observation/result.json')['attempt_dir']);path=folder/('image_01.jpg' if k=='before' else 'image_02.jpg')
 with PILImage.open(path) as im:w,h=im.size
 return Image(str(path),width=500,height=500*h/w)
story=[p('X0450：Intention–Outcome Discrepancy','h1'),p('独立分析 / 原始记录与已有假设对照','h1'),p(f['final_answer']),p('继续研究价值：'+f['worth_followup']),p('相对已有库的新颖性：'+f['new_discovery_status']),p('已确认Corner Reveal：'+str(f['confirmed_corner_discovery'])),p('1. Original Intention','h2')]
for k in ['short_reason','action_purpose','expected_stop_state']:story.append(p('原文 '+k+'：'+dispatch['plan'][k]))
story += [p(f['original_intention_summary']),p('右袖原始预期：'+a['intent']['image_right_sleeve_expectation']),p(a['intent']['right_sleeve_basis']),p('2. Expectation Gap','h2'),p(f['expectation_gap']),p(f['gap_evidence']),p('原目标完成情况：'+f['intended_goal_status']),p('出处：'+src['source'],'small'),p('原始意图按producer保存的派发前记录提取，不根据After重写；未独立认证保存时间。')]
story += [PageBreak(),p('3. 原始图像','h1'),p('Before','h2'),image('before'),p('After','h2'),image('after')]
story += [PageBreak(),p('4. 独立Observed Outcome','h1'),p(f['observed_outcome_summary'])]
for k,v in a['observation'].items():story.append(p(k+'：'+(v if isinstance(v,str) else pretty(v))))
story += [p('5. Hypothesis Opportunity','h1'),p(f['hypothesis_opportunity'])]
for e in f['competing_explanations']:
 story += [p(e['label'],'h2')]
 for k,v in e.items():
  if k!='label':story.append(p(k+'：'+v))
story += [PageBreak(),p('6. 已有Hypothesis Bank审计','h1')]
for e in f['existing_hypothesis_audit']:
 story += [p(e['hypothesis_id'],'h2')]
 for k,v in e.items():
  if k!='hypothesis_id':story.append(p(k+'：'+v))
story += [p('旧观察与本轮独立复核','h2'),p(f['existing_observation_vs_new_review']),p('已有假设原文','h2')]
for h in a['existing_hypotheses']:
 story += [p(h['hypothesis_id'],'h2')]
 for k in ['observed_effect','hypothesized_mechanism','predicted_effect','falsification_condition','evidence_level','uncertainty']:story.append(p(k+'：'+h[k]))
story += [PageBreak(),p('7. 证据与审计边界','h1'),p('记录动作：'+pretty(action['indexed_action'])),p('执行信息：'+pretty(action['execution_information'])),p('原生指令与索引抬升保留差异；命令完成不等于实际布料夹持成功。没有用历史理由替代实际材料身份。')]
for x in f['limits']:story.append(p(x))
story += [p('信息隔离','h2'),p('意图会话只看原始dispatch；观察会话只看Before/After。两者冻结后，综合会话才看到其输出、动作及X0450已有假设和观察。没有提供先前预测结果、其他案例人工解释或既有技能。'),p('耗时与Token','h2'),p(pretty(cost)),p('所有原始记录、图像与假设库哈希在结束时复核。报告中的机制解释不写回原库。')]
def footer(c,doc):c.setFont('Latin',8);c.drawString(40,22,'X0450 / Independent discrepancy analysis');c.drawRightString(A4[0]-40,22,str(doc.page))
SimpleDocTemplate(str(R/'report.pdf'),pagesize=A4,leftMargin=40,rightMargin=40,topMargin=38,bottomMargin=40,title='X0450 Intention–Outcome Discrepancy',author='Codex').build(story,onFirstPage=footer,onLaterPages=footer)
print('REPORT READY',R,flush=True)
