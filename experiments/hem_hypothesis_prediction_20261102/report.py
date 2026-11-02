"""Render already frozen prediction and disclosed observation; no model calls."""
import ast,json,html,os
from pathlib import Path
from PIL import Image as PILImage
from reportlab.platypus import Image,Table,PageBreak,SimpleDocTemplate,Flowable
from reportlab.lib.pagesizes import A4
BASE=Path('/mnt/newssd/sja/clothAgent_real')
R=BASE/'reports/x0152_h001_prediction_20261102'
def load(p):return json.loads(Path(p).read_text())
def pretty(x):return json.dumps(x,ensure_ascii=False,indent=2)
def esc(x):return html.escape(str(x))
def pre(x):return '<pre>'+esc(pretty(x))+'</pre>'
a=load(R/'result.json');case=a['selected_case'];pred=a['prediction'];obs=a['observation'];cmp=a['comparison'];h=a['source_hypothesis']
calls={k:load(R/'calls'/k/'result.json') for k in ['selection','prediction','observation','comparison']}
usage={}
for c in calls.values():
 for k,v in (c.get('usage') or {}).items():usage[k]=usage.get(k,0)+v
cost={'seconds':{k:round(v['latency_s'],2) for k,v in calls.items()},'usage':usage,'model_calls':len(calls)}
(R/'costs.json').write_text(pretty(cost))
action=load(R/'actions'/f'{case["id"]}.json')
pa=action['indexed_action']['grasp_px'];pb=action['indexed_action']['target_px']
with PILImage.open(case['before']) as im:iw,ih=im.size
rects=[(pred['target_bbox_xyxy'],'#0a9eb5','TARGET REGION'),(pred['covering_edge_bbox_xyxy'],'#ed8100','POSSIBLE COVER EDGE')]
# Coordinate plots are report annotations only; models received clean images.
svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="{iw}" height="{ih}" viewBox="0 0 {iw} {ih}"><image href="{os.path.relpath(case["before"],R)}" width="{iw}" height="{ih}"/>'
for (x0,y0,x1,y1),color,label in rects:
 svg+=f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" fill="none" stroke="{color}" stroke-width="3"/><text x="{x0}" y="{max(18,y0-7)}" fill="{color}" font-family="sans-serif" font-size="18">{label}</text>'
svg+=f'<defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L0,6 L8,3 z" fill="#ffff00"/></marker></defs><line x1="{pa[0]}" y1="{pa[1]}" x2="{pb[0]}" y2="{pb[1]}" stroke="#ffff00" stroke-width="4" marker-end="url(#arrow)"/>'
for pt,color,label in [(pa,'#00ee60','Pa'),(pb,'#ff3030','Pb')]:svg+=f'<circle cx="{pt[0]}" cy="{pt[1]}" r="6" fill="{color}"/><text x="{pt[0]+9}" y="{pt[1]}" fill="{color}" font-size="20">{label}</text>'
svg+='</svg>'; (R/'prediction_overlay.svg').write_text(svg)
rel=lambda path:os.path.relpath(path,R)
def img(path):return '<a href="'+esc(rel(path))+'"><img src="'+esc(rel(path))+'"></a>'
parts=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>X0152-H001 冻结预测检验</title><style>body{font:16px/1.7 system-ui;max-width:1100px;margin:25px auto;background:#f6f7fa;color:#183045}section{background:white;border:1px solid #ddd;padding:20px;margin:16px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere}img{max-width:100%}.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}td,th{border:1px solid #ccc;padding:9px;vertical-align:top}table{border-collapse:collapse;width:100%}.warn{background:#fff0d2;padding:15px}</style><h1>X0152-H001：先预测，再揭示历史 After</h1><p><a href="report.pdf">PDF报告</a> · <a href="result.json">全部结构化结果</a> · <a href="prediction_frozen.json">预测冻结凭据</a></p>',
 '<div class="warn">一次历史动作的结果预测检验，不是新策略实机测试。未修改假设、未执行机器人。独立观察模型未读取假设或预测。</div>',
 '<section><h2>结论</h2><p>'+esc(cmp['conclusion'])+'</p>'+pre({k:cmp[k] for k in ['hypothesis_result','cover_edge_retreat_result','target_sleeve_result','corner_reveal_result','falsification_conditions_met','limitations']})+'</section>',
 '<section><h2>选择与隔离</h2><p>先按固定种子抽取12个不同run的Before，只由Before+实际动作选择一例。排除来源run及上一轮已查看After的25个复核案例。不按新案例结果挑选。</p>'+pre({'selected':case['experience_id'],'source':case['source'],'selection_reason':a['selection']['selection_reason'],'freeze':a['prediction_freeze'],'after_disclosure':a['after_disclosure']})+'<p>预测模型仅收到假设六个字段、单张Before和动作记录；未收到选择理由、原始意图、既有观察或After。默认codex，不使用profile。协调者已知原假设，模型请求为独立无工具上下文。</p></section>',
 '<section><h2>Before与真实记录动作</h2>'+img(case['before'])+'<object data="prediction_overlay.svg" type="image/svg+xml" style="width:100%"></object><p>仅供报告阅读的标注：黄色为历史Pa→Pb；蓝框为预测目标袖部，橙框为候选覆盖边。框不代表材料身份已确认。所有模型都接收干净原图。</p>'+pre(action)+'</section>',
 '<section><h2>冻结预测（原文）</h2>'+pre(pred)+'</section>',
 '<section><h2>预测冻结后揭示：Before / After</h2><div class="pair">'+img(case['before'])+img(case['after'])+'</div><h3>未读预测的独立观察</h3>'+pre(obs)+'</section>',
 '<section><h2>逐项核对</h2>'+pre(cmp)+'</section>',
 '<section><h2>源假设原文</h2>'+pre(h)+'</section>',
 '<section><h2>全部12个Before候选的选择记录</h2>'+pre(a['selection'])+'</section>',
 '<section><h2>耗时与调用记录</h2>'+pre(cost)+'<p>另有一次沙箱初始化失败，未产生模型输出；原错误记录保留在calls/selection_sandbox_startup_failure。其后相同输入重启，未按结果挑选输出。</p></section></html>']
(R/'review.html').write_text(''.join(parts))
# Load only font/render helper definitions, never execute old report generation.
tree=ast.parse((BASE/'experiments/hypothesis_audit_20261102/export_pdf.py').read_text());nodes=[]
for n in tree.body:
 if isinstance(n,(ast.Import,ast.ImportFrom,ast.FunctionDef)):nodes.append(n)
 elif isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='styles' for t in n.targets):nodes.append(n)
 elif isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='registerFont':nodes.append(n)
ns={};exec(compile(ast.Module(body=nodes,type_ignores=[]),__file__,'exec'),ns);p=ns['p']
def image(k,width=500):
 path=R/'calls/observation'/('image_01.jpg' if k=='before' else 'image_02.jpg')
 with PILImage.open(path) as im:w,hh=im.size
 return Image(str(path),width=width,height=width*hh/w)
class Overlay(Flowable):
 def __init__(self):super().__init__();self.width=500;self.height=500*ih/iw
 def draw(self):
  c=self.canv;s=500/iw;c.drawImage(str(R/'calls/observation/image_01.jpg'),0,0,width=self.width,height=self.height)
  for (x0,y0,x1,y1),color,label in rects:
   c.setStrokeColor(color);c.setFillColor(color);c.setLineWidth(1.4);c.rect(x0*s,(ih-y1)*s,(x1-x0)*s,(y1-y0)*s)
   c.setFont('Latin',7);c.drawString(x0*s,(ih-y0)*s+3,label)
  c.setStrokeColor('#ffff00');c.setLineWidth(2);c.line(pa[0]*s,(ih-pa[1])*s,pb[0]*s,(ih-pb[1])*s)
  for pt,color,label in [(pa,'#00ee60','Pa'),(pb,'#ff3030','Pb')]:
   c.setFillColor(color);c.circle(pt[0]*s,(ih-pt[1])*s,3,stroke=0,fill=1);c.setFont('Latin',8);c.drawString(pt[0]*s+5,(ih-pt[1])*s,label)
story=[p('X0152-H001：冻结预测后的历史检验','h1'),p('本轮：'+case['experience_id']+' / '+case['source'],'small'),p(cmp['conclusion']),p('判定：'+cmp['hypothesis_result']+'；覆盖边：'+cmp['cover_edge_retreat_result']+'；Corner Reveal：'+cmp['corner_reveal_result']),p('重要边界：这是历史单动作的预测检验。没有生成新动作、没有执行机器人，也不证明通用技能。'),p('1. 如何选择新案例','h2'),p(a['selection']['selection_reason']),p('固定种子抽取12个不同run的Before。排除来源run及上一轮已看After的案例。选择模型只有Before、实际动作和原假设；预测模型使用全新上下文，没有选择理由或新案例原始意图。'),p('2. 原假设（原文）','h2')]
for k in ['observable_conditions','hypothesized_mechanism','predicted_effect','falsification_condition','uncertainty']:story.append(p(k+'：'+h[k]))
story += [PageBreak(),p('3. 预测时唯一图像：Before','h1'),Overlay(),p('图中标注仅用于报告；模型看到干净原图。蓝框：预测目标袖区；橙框：候选覆盖边。黄色连线：历史Pa→Pb。框不是已确认材料身份。'),p('实际记录动作：'+pretty(action['indexed_action'])),p('底层指令保留在JSON及模型原始输入中；索引抬升与底层指令若有差异，不能把索引值当作唯一真值。')]
story += [p('4. 冻结预测','h2')]
for k,v in pred.items():story.append(p(k+'：'+(pretty(v) if not isinstance(v,str) else v)))
story += [PageBreak(),p('5. 冻结后揭示 After','h1'),image('after'),p('独立观察器未读取假设、预测、动作或原始意图。')]
for k,v in obs.items():story.append(p(k+'：'+(pretty(v) if not isinstance(v,str) else v)))
story += [PageBreak(),p('6. 预测与结果逐项核对','h1')]
for row in cmp['prediction_vs_observation']:
 story += [p(row['verdict'],'h2'),p('预测：'+row['prediction']),p('观察：'+row['observation'])]
for k in ['hypothesis_result','cover_edge_retreat_result','target_sleeve_result','corner_reveal_result','falsification_conditions_met','conclusion','limitations']:story.append(p(k+'：'+(pretty(cmp[k]) if not isinstance(cmp[k],str) else cmp[k])))
story += [PageBreak(),p('7. 候选筛选完整记录','h1')]
for row in a['selection']['candidates']:story.append(p(pretty(row)))
story += [PageBreak(),p('8. 审计与耗时','h1'),p(pretty(cost)),p('预测冻结：'+pretty(a['prediction_freeze'])),p('After揭示：'+pretty(a['after_disclosure'])),p('调用均为默认codex / gpt-6-astra / medium，未使用profile。各阶段独立上下文，工具调用为0。一次沙箱初始化失败后按原输入重启，记录已保留。'),p('单案例不能估计准确率；前后两帧难以确认接触、材料层身份及过程中的相对运动。没有查阅新案例原有结果标签来选择或验证。源库与预测哈希在结束时复核。')]
def footer(c,doc):
 c.setFont('Latin',8);c.drawString(40,22,'X0152-H001 / Frozen prediction test');c.drawRightString(A4[0]-40,22,str(doc.page))
SimpleDocTemplate(str(R/'report.pdf'),pagesize=A4,leftMargin=40,rightMargin=40,topMargin=38,bottomMargin=40,title='X0152-H001 冻结预测检验',author='Codex').build(story,onFirstPage=footer,onLaterPages=footer)
print('Report ready',R,flush=True)
