"""HTML/PDF renderer of frozen inputs, predictions and blind reference judgments."""
import ast,json,html,os
from pathlib import Path
from PIL import Image as PILImage
from reportlab.platypus import Image,Table,PageBreak,SimpleDocTemplate
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
ROOT=Path('/mnt/newssd/sja/clothAgent_real');R=ROOT/'reports/hypothesis_guided_prediction_20261102'
def load(p):return json.loads(Path(p).read_text())
def pretty(x):return json.dumps(x,ensure_ascii=False,indent=2)
def esc(x):return html.escape(str(x))
def pre(x):return '<pre>'+esc(pretty(x))+'</pre>'
a=load(R/'result.json');stats=load(R/'analysis.json');by={r['id']:r for r in a['sample']};h=load(R/'provided_hypothesis.json')
accepted=list(a['predictions']);same=[]
for cid,pair in a['predictions'].items():
 keys=['primary_target','semantic_corner','visibility_change','geometry_metric','geometry_change']
 if all(pair['Baseline'][k]==pair['Hindsight'][k] for k in keys):same.append(cid)
n=len(accepted)
if n:
 conclusion=f'随机抽取{len(a["sample"])}条不同run的历史动作，{n}条适用、其余拒绝且未替换。{len(same)}/{n}个配对案例两组的目标、语义侧、可见性方向和几何预测完全相同。'
 if len(same)==n:conclusion+=' 在这批有限样本中，没有观察到假设带来不同的可评分预测，也没有预测正确性增益证据。不能据此证明假设普遍无效；本轮只显示它未增加这些案例的可检验预测信息。'
 else:conclusion+=' 部分输出发生变化；变化是否有用须结合盲评价可判定率、未知比例与共同目标指标比较，不能只看解释长度。'
else:conclusion='5条随机案例中没有足够可用的测试案例。未放宽适用性、未读取After补选，未启动预测和结果评价。'
conclusion+=' 独立评价器本身也是模型，其标签属于离线视觉代理，不是真实物理真值；本轮不宣称泛化或机器人成功率改善。'
if n and all(stats['groups'][g]['graded']==0 for g in ['Baseline','Hindsight']):
 conclusion+=' 两组均无可评分预测：正确率为NA，不是0%。不能比较预测正确性，也不能把未评分计作错误。'
unpredicted=[]
for cid,pair in a['predictions'].items():
 selected_targets={v['primary_target'] for v in pair.values()}
 for reg in a['observations'][cid]['regions']:
  if reg['region'] not in selected_targets and reg['visibility']['change']=='INCREASE' and reg['visibility']['evidence_quality']=='CLEAR':
   unpredicted.append({'case_id':cid,'region':reg['region'],'visibility':reg['visibility'],'exposed_region_area':reg['exposed_region_area'],'corner_reveal':reg['corner_reveal'],'note':'盲评价中的事后观察；两组均未选择此区域，不能给任何一组追加正确预测分，也不能将其变成已确认Corner Reveal。'})
(R/'conclusion.json').write_text(pretty({'conclusion':conclusion,'identical_prediction_cases':same,'tested_cases':n}))
def rel(p):return os.path.relpath(p,R)
def img(p):return '<a href="'+esc(rel(p))+'"><img loading="lazy" src="'+esc(rel(p))+'"></a>'
def report_table():
 labels=[('cases','配对案例数'),('definite_predictions','确定方向预测数'),('unknown_prediction_total','UNKNOWN预测数'),('reference_judgeable','参考标签可判定数'),('correct','一致/正确代理数'),('incorrect','不一致/错误代理数'),('abstain_on_judgeable','参考可判定但预测UNKNOWN'),('unassessable','参考不可判定'),('conditional_accuracy','已评分预测的一致率'),('correct_concrete_geometry','正确的具体几何预测数'),('mean_latency_s','平均预测耗时(s)'),('mean_response_characters','平均输出字符数')]
 rows='<table><tr><th>指标</th><th>Baseline</th><th>Hindsight</th></tr>'
 for k,label in labels:
  vals=[stats['groups'][g][k] for g in ['Baseline','Hindsight']];rows+='<tr><td>'+label+'</td>'+''.join('<td>'+esc('NA' if v is None else round(v,3) if isinstance(v,float) else v)+'</td>' for v in vals)+'</tr>'
 return rows+'</table>'
parts=['<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Hypothesis-Guided Effect Prediction</title><style>body{font:16px/1.65 system-ui;max-width:1250px;margin:auto;padding:25px;background:#f3f6fa;color:#163046}section{background:white;padding:20px;margin:18px 0;border:1px solid #d1d9e0}pre{white-space:pre-wrap;overflow-wrap:anywhere}.pair{display:grid;grid-template-columns:1fr 1fr;gap:15px}img{width:100%}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccc;padding:9px;vertical-align:top}.warn{background:#fff2cf;padding:18px}a{color:#126bab}@media(max-width:700px){.pair{display:block}}</style><h1>Hypothesis-Guided Effect Prediction</h1><p><a href="report.pdf">完整PDF</a> · <a href="result.json">原始结构化结果</a> · <a href="analysis.json">逐项评分</a> · <a href="protocol.json">冻结协议</a></p>',
 '<div class="warn">离线小样本对照。无机器人、无新Skill、未修改Hypothesis Bank。UNKNOWN不计正确或错误。区域更清楚不等于Corner Reveal。</div>',
 '<section><h2>结论</h2><p>'+esc(conclusion)+'</p>'+report_table()+'<p>每例固定两个预测槽：区域可见性＋一个静态几何指标。标签数不是独立试验数。正确性仅比较明确预测与CLEAR盲参考，条件一致率必须结合可判定率和弃权率阅读。</p><h3>两组共同目标与指标的配对结果</h3>'+pre(stats['paired_common_slots'])+'<h3>盲评价中出现、但两组未预测的区域改善</h3>'+pre(unpredicted)+'<p>本轮允许模型自行选择一个主要预测区域，没有强制它预测假设指定的对侧；结果不能反向用于补写预测或重新评分。</p></section>',
 '<section><h2>冻结假设：X0152-H001</h2>'+pre(h)+'<p>选择依据：非直接操作袖部的外露预测。只评分静态最终形状，不评分摩擦、传力、运动先后或层间滑移是否发生。上一轮已看After的run全部排除；没有基于当前测试After选择假设。</p></section>',
 '<section><h2>隔离和时间顺序</h2><p>1. 冻结假设/指标/种子 → 2. 随机5个run、每run一个动作 → 3. Before＋动作适用性筛选 → 4. 两组独立预测全部冻结 → 5. After揭示 → 6. 不知假设/两组预测的独立评价器 → 7. 固定代码比较。</p>'+pre({k:a.get(k) for k in ['prediction_freeze','after_disclosure']})+'<p>两组提示词、Before、Action、Schema、模型相同。Hindsight仅多一个historical_hypothesis字段。各请求独立、无工具；默认codex，不使用profile。一次采样不能分离模型随机波动和稳定提示效应。</p></section>']
for cid in accepted:
 r=by[cid];pair=a['predictions'][cid]
 parts+=['<section><h2>'+cid+' / '+r['experience_id']+'</h2><p>'+esc(r['source'])+'</p><h3>Before：预测阶段唯一图片</h3>'+img(r['before'])+'<h3>历史真实记录动作</h3>'+pre(load(R/'actions'/f'{cid}.json'))+'<div class="pair"><div><h3>Baseline冻结预测</h3>'+pre(pair['Baseline'])+'</div><div><h3>Hindsight冻结预测</h3>'+pre(pair['Hindsight'])+'</div></div><h3>所有预测冻结后揭示After</h3>'+img(r['after'])+'<h3>逐项比较</h3>'+pre([x for x in stats['per_slot'] if x['case_id']==cid])+'<details><summary>独立评价器：四区域完整观察</summary>'+pre(a['observations'][cid])+'</details></section>']
parts+=['<section><h2>全部随机样本：接受与拒绝</h2><p>以下只展示Before；拒绝案例的After未揭示。</p>']
for s in a['selection']['cases']:
 r=by[s['id']];parts.append('<details><summary>'+s['id']+' / '+r['experience_id']+' / '+s['applicability']+'</summary>'+img(r['before'])+pre(s)+'<p>'+esc(r['source'])+'</p></details>')
parts+=['</section><section><h2>成本与限制</h2>'+pre({'calls':stats['calls'],'total_usage':stats['total_model_usage'],'limits':stats['limits']})+'</section></html>'];(R/'review.html').write_text(''.join(parts))
# Reuse only font helpers from the audited renderer.
tree=ast.parse((ROOT/'experiments/hypothesis_audit_20261102/export_pdf.py').read_text());nodes=[]
for x in tree.body:
 if isinstance(x,(ast.Import,ast.ImportFrom,ast.FunctionDef)):nodes.append(x)
 elif isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='styles' for t in x.targets):nodes.append(x)
 elif isinstance(x,ast.Expr) and isinstance(x.value,ast.Call) and isinstance(x.value.func,ast.Attribute) and x.value.func.attr=='registerFont':nodes.append(x)
ns={};exec(compile(ast.Module(body=nodes,type_ignores=[]),__file__,'exec'),ns);p=ns['p']
def image(path,w=500):
 # Embed uploaded JPEG if available, preserving original shape and avoiding huge PDFs.
 with PILImage.open(path) as im:iw,ih=im.size
 return Image(str(path),width=w,height=w*ih/iw)
story=[p('Hypothesis-Guided Effect Prediction','h1'),p('X0152-H001：历史假设是否改善静态结果预测？','h1'),p(conclusion),p('主要统计','h2')]
header=[p('指标'),p('Baseline'),p('Hindsight')];tab=[header]
for k,label in [('cases','独立动作数'),('definite_predictions','明确方向预测数'),('unknown_prediction_total','UNKNOWN预测数'),('reference_judgeable','参考可判定数'),('correct','正确代理数'),('incorrect','错误代理数'),('unassessable','不可评分数'),('conditional_accuracy','已评分预测一致率'),('correct_concrete_geometry','正确几何预测数'),('mean_latency_s','平均请求耗时（秒）'),('mean_response_characters','平均输出字符数')]:
 tab.append([p(label)]+[p('NA' if stats['groups'][g][k] is None else str(round(stats['groups'][g][k],2)) if isinstance(stats['groups'][g][k],float) else str(stats['groups'][g][k])) for g in ['Baseline','Hindsight']])
t=Table(tab,colWidths=[250,130,130]);t.setStyle([('GRID',(0,0),(-1,-1),.3,colors.grey),('VALIGN',(0,0),(-1,-1),'TOP')]);story.append(t)
story += [p('每例两项预测；这不是两个独立机器人试验。UNKNOWN不作为正确或错误。主要指标为静态可见性和几何变化，非Corner Reveal成功率。'),p('共同目标/指标配对：'+pretty(stats['paired_common_slots'])),p('隔离','h2'),p('所有预测冻结后才揭示After。独立评价器未获得假设、动作、选择理由或两组预测。两组唯一输入差异是历史假设。默认codex，gpt-6-astra / medium。')]
if unpredicted:story += [p('未被两组预测的区域改善（事后观察）','h2'),p(pretty(unpredicted)),p('两组均自由选择一个主要区域，没有强制预测对侧。不将After观察补入冻结预测，不追加正确分。')]
for cid in accepted:
 r=by[cid];pair=a['predictions'][cid];call=load(R/'calls'/f'observe_{cid}'/'result.json');folder=Path(call['attempt_dir'])
 story += [PageBreak(),p(cid+' / '+r['experience_id'],'h1'),p(r['source'],'small'),p('Before（预测时唯一图像）','h2'),image(folder/'image_01.jpg'),p('动作索引：'+pretty(load(R/'actions'/f'{cid}.json')['indexed_action'])),p('索引与底层指令差异按原样提供，不统一改写。')]
 for group in ['Baseline','Hindsight']:
  story += [p(group+'：冻结预测','h2')]
  for k,v in pair[group].items():story.append(p(k+'：'+(v if isinstance(v,str) else pretty(v))))
 story += [PageBreak(),p('揭示After与逐项比较','h1'),image(folder/'image_02.jpg')]
 for x in [x for x in stats['per_slot'] if x['case_id']==cid]:story += [p(x['group']+' / '+x['slot']+' / '+x['verdict'],'h2'),p(pretty(x))]
 story += [p('独立评价器：各区域证据','h2')]
 for reg in a['observations'][cid]['regions']:story += [p(reg['region'],'h2'),p(pretty(reg))]
 story.append(p(a['observations'][cid]['limitations']))
story += [PageBreak(),p('随机5条的筛选记录','h1'),p('拒绝后未补抽、未揭示拒绝案例After。')]
for s in a['selection']['cases']:
 r=by[s['id']];uploaded=Path(load(R/'calls/applicability/result.json')['attempt_dir'])/f'image_{int(s["id"][1:]):02}.jpg'
 story += [p(s['id']+' / '+r['experience_id'],'h2'),image(uploaded,400),p(pretty(s))]
story += [PageBreak(),p('冻结假设原文与实验边界','h1'),p(pretty(h)),p('成本','h2'),p(pretty(stats['calls'])),p('限制','h2')]
for x in stats['limits']:story.append(p(x))
story += [p('完整预测与观察：result.json；固定评分：analysis.json；协议：protocol.json；冻结凭据：predictions_frozen.json。')]
def footer(c,doc):c.setFont('Latin',8);c.drawString(40,22,'Hypothesis-Guided Effect Prediction / Offline pilot');c.drawRightString(A4[0]-40,22,str(doc.page))
SimpleDocTemplate(str(R/'report.pdf'),pagesize=A4,leftMargin=40,rightMargin=40,topMargin=38,bottomMargin=40,title='Hypothesis-Guided Effect Prediction',author='Codex').build(story,onFirstPage=footer,onLaterPages=footer)
print(conclusion,flush=True)
