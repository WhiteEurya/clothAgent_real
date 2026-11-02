"""Read-only rendering of the completed held-out retrieval A/B experiment."""
import ast,html,json,os
from pathlib import Path
from PIL import Image as PILImage
from reportlab.platypus import Image,PageBreak,SimpleDocTemplate,Table,Flowable
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
ROOT=Path('/mnt/newssd/sja/clothAgent_real');R=ROOT/'reports/autonomous_retrieval_ab_20261102'
def load(p):return json.loads(Path(p).read_text())
def pretty(v):return json.dumps(v,ensure_ascii=False,indent=2)
def esc(v):return html.escape(str(v))
def pre(v):return '<pre>'+esc(pretty(v))+'</pre>'
def rel(p):return os.path.relpath(p,R)
a=load(R/'result.json');s=load(R/'analysis.json');isolation=load(R/'isolation_audit.json');by={r['id']:r for r in a['sample']};o=s['overall']
if o['new_correct_non_direct_effects']:
 conclusion=f'本轮有{o["new_correct_non_direct_effects"]}项非直接抓取区域变化被Hindsight预测到、Baseline未预测到，且得到盲After评价支持，涉及{len({x["case_id"] for x in s["gains"]})}次动作、{len({x["run"] for x in s["gains"]})}个run；同时有{o["lost_correct_non_direct_effects"]}项Baseline命中而Hindsight未命中。'
else:conclusion='本轮没有观察到Hindsight额外预测到、且盲After评价支持的非直接抓取区域变化。'
conclusion+=' 这些是单次模型采样与盲视觉评价得到的离线代理结果，不能归因为稳定的经验增益，也不是Corner Reveal实机成功率。需同时看错误预测、UNKNOWN、覆盖率和检索成本。'
(R/'conclusion.json').write_text(pretty({'conclusion':conclusion,'overall':o,'no_physical_success_claim':True}))
labels=[('visibility_items','四端点区域预测项'),('judgeable_visibility','可评价项'),('correct','正确代理数'),('incorrect','错误代理数'),('na_abstain','参考明确、预测UNKNOWN'),('na_reference','参考不可判定NA'),('unknown_predictions','预测UNKNOWN数'),('reference_coverage','参考可判定覆盖率'),('scored_over_judgeable','参考可判定项目的明确预测覆盖率'),('conditional_accuracy','已评分预测一致率'),('correct_over_judgeable','可评价项中的正确覆盖率'),('correct_non_direct_changes','正确预测非直接区域变化'),('non_direct_changed_items','可判定非直接变化总数'),('contradicted_change_predictions','After明确否定的变化预测'),('change_predictions_with_unknown_reference','变化预测但After无法核验')]
def val(v):return 'NA' if v is None else f'{v:.3f}' if isinstance(v,float) else str(v)
def table(headers,rows):return '<table><thead><tr>'+''.join('<th>'+esc(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'
summaryrows=[[label]+[val(s['summary'][g][key]) for g in ['Baseline','Hindsight']] for key,label in labels]
def svg_before(cid):
 row=by[cid];act=load(R/'actions'/f'{cid}.json')['indexed_action'];pa,pb=act['grasp_px'],act['target_px']
 with PILImage.open(row['before']) as im:w,h=im.size
 z=f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" style="width:100%;height:auto"><image href="{esc(rel(row["before"]))}" width="{w}" height="{h}"/>'
 z+=f'<defs><marker id="arrow{cid}" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L0,6 L8,3 z" fill="yellow"/></marker></defs><line x1="{pa[0]}" y1="{pa[1]}" x2="{pb[0]}" y2="{pb[1]}" stroke="yellow" stroke-width="4" marker-end="url(#arrow{cid})"/>'
 for pt,color,name in [(pa,'#00ef70','Pa'),(pb,'#ff3333','Pb')]:z+=f'<circle cx="{pt[0]}" cy="{pt[1]}" r="6" fill="{color}"/><text x="{pt[0]+10}" y="{pt[1]}" font-size="22" fill="{color}">{name}</text>'
 return z+'</svg>'
def case_rows(cid):
 pred=a['predictions'][cid];obs={r['goal']:r for r in a['observations'][cid]['regions']};bp={r['goal']:r for r in pred['Baseline']['regions']};hp={r['goal']:r for r in pred['Hindsight']['regions']};rows=[]
 for goal in ['LS','RS','LH','RH']:
  scores={r['group']:r for r in s['per_region'] if r['case_id']==cid and r['goal']==goal}
  rows.append([goal,bp[goal]['visibility_change'],hp[goal]['visibility_change'],obs[goal]['visibility_change']['value'],obs[goal]['visibility_change']['quality'],obs[goal]['direct_grasp_relation'],scores['Baseline']['verdict']+' / '+scores['Hindsight']['verdict']])
 return rows
parts=['<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Autonomous Hypothesis Retrieval A/B</title><style>body{font:16px/1.65 system-ui;max-width:1250px;margin:auto;padding:25px;background:#f3f6fa;color:#193244}section,details{background:white;border:1px solid #d3dce3;padding:18px;margin:15px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere}img{max-width:100%}.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #cbd5df;padding:8px;text-align:left;vertical-align:top}.warn{background:#fff0cf;padding:16px}a{color:#146ab2}summary{cursor:pointer;font-weight:bold}@media(max-width:750px){.pair{display:block}table{font-size:11px}}</style><h1>Autonomous Hypothesis Retrieval A/B Test</h1><p><a href="report.pdf">PDF报告</a> · <a href="result.json">完整结果JSON</a> · <a href="analysis.json">逐项计分JSON</a> · <a href="protocol.json">冻结协议</a> · <a href="isolation_audit.json">隔离审计</a></p>',
 '<div class="warn">4个完整run隔离、8次历史动作全部保留。没有机器人执行、没有新假设、没有Skill。所有预测冻结后才读取测试After。区域可辨性改善不是严格Corner Reveal。</div>',
 '<section><h2>核心结论</h2><p>'+esc(conclusion)+'</p>'+pre(o)+table(['指标','Baseline','Hindsight'],summaryrows)+'<p>NA不计正确或错误。“After明确否定”不包括无法判断的项目。32个区域判断来自8个动作、4个run，不能当成32个独立实验。非目标指标实际使用非直接抓取区域，不推断原Agent意图。</p></section>',
 '<section><h2>额外预测到的非直接区域变化</h2>'+pre(s['gains'])+'<h3>Baseline预测到、Hindsight遗漏的变化</h3>'+pre(s['losses'])+'</section>',
 '<section><h2>隔离、检索与冻结</h2>'+pre({k:isolation[k] for k in ['test_runs','whole_test_run_operations','duplicate_expansion_ids','allowed_hypotheses','derived_reports_used','limits']})+'<p>排除54条测试run相关假设；训练索引从原库重新构建，只读。模型生成Before/Action检索描述；BM25覆盖全部允许假设，取前24条；独立选择器取最多3条或NONE。并非逐条视觉阅读全部假设，存在检索漏召回风险。未使用Top-15或已知案例清单指导选样或检索。</p>'+pre({'prediction_freeze':a['prediction_freeze'],'after_disclosure':a['after_disclosure']})+'<p>协调者此前见过部分历史案例；本轮不声称研究者从未见过测试集。独立模型会话未接收先前报告、预测或人工结论，不能调用工具。</p></section>']
for cid,row in by.items():
 retrieval=a['retrieval'][cid];pred=a['predictions'][cid];act=load(R/'actions'/f'{cid}.json')
 parts+=['<section id="'+cid+'"><h2>'+cid+' / '+row['experience_id']+'</h2><p>'+esc(row['source'])+'</p><h3>Before与记录动作</h3>'+svg_before(cid)+'<p>箭头仅为报告标注；预测模型均看到干净原图。<a href="'+esc(rel(row['before']))+'">打开完整原图</a></p>'+pre(act['indexed_action']),
 '<details><summary>原生命令参数</summary>'+pre(act['native_command_sequence'])+'</details><h3>自主检索</h3>'+pre(retrieval['selection'])]
 for h in retrieval['selected_hypotheses']:parts+=['<details><summary>'+esc(h['hypothesis_id'])+' / '+esc(h['source_run'])+'</summary>'+pre(h)+'</details>']
 parts+=['<details><summary>查询与前24条候选ID</summary>'+pre({'query':retrieval['query'],'shortlist':retrieval['shortlist_ids']})+'<p><a href="retrieval/'+cid+'_all_scores.json">全部训练条目的检索分数</a></p></details>',
 '<h3>两组冻结预测</h3><div class="pair"><div><h4>Baseline</h4>'+pre(pred['Baseline'])+'</div><div><h4>Hindsight</h4>'+pre(pred['Hindsight'])+'</div></div>',
 '<h3>冻结后揭示After</h3><a href="'+esc(rel(row['after']))+'"><img src="'+esc(rel(row['after']))+'"></a>',
 '<h3>盲评价与逐项比较</h3>'+table(['目标','Baseline','Hindsight','盲观察','证据质量','抓点关系','B/H计分'],case_rows(cid)),
 '<h4>具体静态效果</h4>'+pre([r for r in s['static_geometry'] if r['case_id']==cid])+'<h4>严格Corner Reveal（单列）</h4>'+pre([r for r in s['strict_reveal'] if r['case_id']==cid]),
 '<details><summary>不知道两组预测的完整盲观察</summary>'+pre(a['observations'][cid])+'</details></section>']
parts+=['<section><h2>按run汇总</h2>'+pre(s['per_run'])+'<h2>预测是否有实质差异</h2>'+pre(s['substantive_differences'])+'<h2>全部耗时与Token</h2>'+pre(s['costs'])+'<p>Hindsight总开销需包含查询、候选选择和预测，不能只报告最后一次预测请求。请求秒数之和不等于并发墙钟时间。</p><h2>限制</h2>'+pre(s['limits'])+'</section></html>'];(R/'review.html').write_text(''.join(parts))
# Load only font helpers, not other report data or inference.
tree=ast.parse((ROOT/'experiments/hypothesis_audit_20261102/export_pdf.py').read_text());nodes=[]
for node in tree.body:
 if isinstance(node,(ast.Import,ast.ImportFrom,ast.FunctionDef)):nodes.append(node)
 elif isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='styles' for t in node.targets):nodes.append(node)
 elif isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='registerFont':nodes.append(node)
ns={};exec(compile(ast.Module(body=nodes,type_ignores=[]),__file__,'exec'),ns);p=ns['p']
def pdf_table(headers,rows,widths):
 t=Table([[p(str(x),'small') for x in headers]]+[[p(str(x),'small') for x in row] for row in rows],colWidths=widths,repeatRows=1);t.setStyle([('GRID',(0,0),(-1,-1),.3,colors.grey),('VALIGN',(0,0),(-1,-1),'TOP')]);return t
def photo(cid,k):
 d=Path(load(R/'calls'/('observe_'+cid)/'result.json')['attempt_dir']);path=d/('image_01.jpg' if k=='before' else 'image_02.jpg')
 with PILImage.open(path) as im:w,h=im.size
 return Image(str(path),width=500,height=500*h/w)
class ActionFigure(Flowable):
 def __init__(self,cid):
  super().__init__();self.cid=cid;self.path=Path(load(R/'calls'/('observe_'+cid)/'result.json')['attempt_dir'])/'image_01.jpg'
  with PILImage.open(self.path) as im:self.iw,self.ih=im.size
  self.width=500;self.height=500*self.ih/self.iw
 def draw(self):
  c=self.canv;scl=500/self.iw;c.drawImage(str(self.path),0,0,width=self.width,height=self.height);act=load(R/'actions'/f'{self.cid}.json')['indexed_action'];pa,pb=act['grasp_px'],act['target_px'];c.setStrokeColor('#ffff00');c.setLineWidth(2);c.line(pa[0]*scl,(self.ih-pa[1])*scl,pb[0]*scl,(self.ih-pb[1])*scl)
  for pt,color,label in [(pa,'#00ee60','Pa'),(pb,'#ff3030','Pb')]:c.setFillColor(color);c.circle(pt[0]*scl,(self.ih-pt[1])*scl,3,stroke=0,fill=1);c.setFont('Latin',8);c.drawString(pt[0]*scl+5,(self.ih-pt[1])*scl,label)
story=[p('Autonomous Hypothesis Retrieval A/B','h1'),p('整run隔离的离线效果预测对照','h1'),p(conclusion),p(pretty(o)),pdf_table(['指标','Baseline','Hindsight'],summaryrows,[300,105,105]),p('正确率为与盲视觉代理标签的一致率。UNKNOWN与NA不计错误或正确。未进行物理验证；8个动作聚类于4个run。'),p('额外正确的非直接区域变化','h2'),p(pretty(s['gains'])),p('Baseline命中而Hindsight遗漏','h2'),p(pretty(s['losses']))]
story += [PageBreak(),p('数据隔离与固定流程','h1'),p('原始532对/1222条假设。随机完整留出4个run，共23次操作；排除54条相关假设，剩1168条只读索引。没有按After效果选择样本，没有按适用性删去测试案例。'),p('检索：Before+动作生成查询 → 全索引BM25 → 前24条 → 自主选择0–3条。每个模型请求独立上下文，无工具调用。'),p('两组模型、提示词、图像、动作与Schema相同；Hindsight只多返回的既有假设。查询和选择理由单独记录，不额外喂给预测器。'),p('全部预测冻结之后才读取测试After；盲评价器只看Before/After/动作，不知组别、预测或假设。'),p('测试run：'+pretty(a['protocol']['test_runs'])),p('隔离局限：协调者可能见过部分历史案例。模型不接收那些旧结论；不能宣称研究者完全盲。'),p('成本','h2'),p(pretty({k:v for k,v in s['costs'].items() if k not in ['per_case']}))]
for cid,row in by.items():
 pred=a['predictions'][cid];ret=a['retrieval'][cid]
 story += [PageBreak(),p(cid+' / '+row['experience_id'],'h1'),p(row['source'],'small'),p('Before与历史Pa→Pb（标注仅供阅读）','h2'),ActionFigure(cid),p('After（全部预测冻结后揭示）','h2'),photo(cid,'after')]
 story += [PageBreak(),p(cid+'：检索与两组冻结预测','h1'),p('动作：'+pretty(load(R/'actions'/f'{cid}.json')['indexed_action'])),p('检索选择：'+pretty(ret['selection']))]
 for h in ret['selected_hypotheses']:
  story += [p(h['hypothesis_id'],'h2'),p('适用条件：'+h['observable_conditions']),p('既有机制（未验证）：'+h['hypothesized_mechanism']),p('既有预测：'+h['predicted_effect'])]
 story += [p('四区域可见性预测与盲评价','h2'),pdf_table(['目标','Baseline','Hindsight','观察','质量','抓点关系'],[r[:6] for r in case_rows(cid)],[35,90,90,90,90,115])]
 for group in ['Baseline','Hindsight']:
  story += [p(group+'冻结理由','h2')]
  for reg in pred[group]['regions']:story += [p(reg['goal']+' / '+reg['visibility_change']+' / Reveal='+reg['strict_corner_reveal']),p('依据：'+reg['basis']),p('不确定性：'+reg['uncertainty'])]
  story += [p('操作区形变：'+pred[group]['operated_region_local_deformation']),p('非直接区域：'+pred[group]['non_direct_region_changes']),p('整体移动/出画：'+pred[group]['global_motion_or_out_of_view']),p('具体静态预测：'+pretty(pred[group]['static_effect']))]
 story += [p('该例逐项计分','h2'),p(pretty([x for x in s['paired_visibility'] if x['case_id']==cid])),p('具体几何计分：'+pretty([x for x in s['static_geometry'] if x['case_id']==cid])),p('盲评价证据','h2')]
 for ref in a['observations'][cid]['regions']:story += [p(ref['goal'],'h2'),p('图像位置：'+ref['image_region_description']),p('可见性：'+pretty(ref['visibility_change'])),p('严格Reveal：'+ref['strict_corner_reveal']+'；'+ref['reveal_evidence']),p('抓点关系：'+ref['direct_grasp_relation']+'；'+ref['grasp_relation_basis'])]
 story += [p('局部变化：'+a['observations'][cid]['operated_region_observed_deformation']),p('整体变化/视野：'+a['observations'][cid]['global_motion_and_out_of_view']),p('限制：'+a['observations'][cid]['limits'])]
story += [PageBreak(),p('汇总与限制','h1'),p('逐run：'+pretty(s['per_run'])),p('具体预测差异：'+pretty(s['substantive_differences'])),p('逐案例成本：'+pretty(s['costs']['per_case']))]
for limit in s['limits']:story.append(p(limit))
story += [p('完整JSON保留所有模型输入/输出、索引、排除原因、查询、检索分数、原图、预测哈希、After揭示时间和逐项评分。')]
def footer(c,doc):c.setFont('Latin',8);c.drawString(40,22,'Autonomous Retrieval A/B / Offline evidence only');c.drawRightString(A4[0]-40,22,str(doc.page))
SimpleDocTemplate(str(R/'report.pdf'),pagesize=A4,leftMargin=40,rightMargin=40,topMargin=38,bottomMargin=40,title='Autonomous Hypothesis Retrieval A/B',author='Codex').build(story,onFirstPage=footer,onLaterPages=footer)
print(conclusion,flush=True)
if __name__=='__main__':print('REPORT READY',R,flush=True)
