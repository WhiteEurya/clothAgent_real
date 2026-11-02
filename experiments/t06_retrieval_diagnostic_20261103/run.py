"""Read existing T06 artifacts; render an audit. No inference or source edits."""
import ast, hashlib, html, json, os
from pathlib import Path
from reportlab.platypus import SimpleDocTemplate, PageBreak, Table, Image
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
ROOT=Path('/mnt/newssd/sja/clothAgent_real')
R=ROOT/'reports/autonomous_retrieval_ab_20261102'
OUT=ROOT/'reports/t06_retrieval_diagnostic_20261103'
def load(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def pretty(x): return json.dumps(x,ensure_ascii=False,indent=2)
sources=[R/'result.json',R/'retrieval/T06.json',R/'retrieval/T06_all_scores.json',R/'training_index.json',R/'predictions.json',R/'predictions_frozen.json',R/'after_disclosure.json',R/'analysis.json',R/'isolation_audit.json',R/'prediction_prompt.txt',R/'retrieval_prompt.txt',ROOT/'experiments/autonomous_retrieval_ab_v1/run.py',ROOT/'reports/intention_outcome_hypothesis_bank_v1/hypothesis_bank.json']
for n in ['query_T06','retrieve_T06','predict_T06_Baseline','predict_T06_Hindsight','observe_T06']:
 sources.extend([R/'calls'/n/'result.json',R/'calls'/n/'attempt_01/input.txt',R/'calls'/n/'attempt_01/output.json'])
hashes={str(p):sha(p) for p in sources}
a=load(R/'result.json'); ret=load(R/'retrieval/T06.json'); idx=load(R/'training_index.json'); hb={h['hypothesis_id']:h for h in idx}
scores=load(R/'retrieval/T06_all_scores.json'); ranks={h['hypothesis_id']:{'bm25_rank':i+1,'score':h['score']} for i,h in enumerate(scores)}
sample=next(x for x in a['sample'] if x['id']=='T06'); pred=a['predictions']['T06']; obs=a['observations']['T06']
selected=[]
for i,h in enumerate(ret['selected_hypotheses']):
 selected.append({'selection_list_position':i+1,**ranks[h['hypothesis_id']],'original_hypothesis':h,'original_relevance_judgment':next(x for x in ret['selection']['chosen'] if x['hypothesis_id']==h['hypothesis_id'])})
diagnostic_ids=['X0169-H003','X0156-H001','X0305-H001','X0222-H001','X0506-H001']
unselected=[{'posthoc_diagnostic_only':True,**ranks[k],'in_original_shortlist':k in ret['shortlist_ids'],'original_hypothesis':hb[k]} for k in diagnostic_ids]
bank=load(sources[12]); held=[h for h in bank if h['hypothesis_id'] in ['X0294-H003','X0294-H004']]
selector_input=(R/'calls/retrieve_T06/attempt_01/input.txt').read_text();predictor_input=(R/'calls/predict_T06_Hindsight/attempt_01/input.txt').read_text()
visibility={k:{'selector_input':k in selector_input,'predictor_input':k in predictor_input,'allowed_index':k in hb} for k in ['X0169-H003','X0229-H001','X0412-H002','X0281-H001','X0294-H003','X0294-H004']}
sections=[
('结论：主要断点在24→3选择，随后是预测综合',[
'本例不能解释为假设库没有相关知识，也不能解释为BM25完全未召回：X0169-H003原排名第3，已进入选择器的24条候选，明确讨论操作一侧后对侧袖口可辨性改善；但它没有进入最终3条，预测器没有读取它。',
'最终3条主要提供整体随动、夹持材料不明、胸片与下摆差异形变。Hindsight确实利用了它们，改变了整体运动及操作侧相对伸展预测；但RS仍被预测为UNCHANGED。没有证据证明只要加入第3名就一定能改对，本轮未做这种反事实重跑。',
'盲评价对RS给出CLEAR/IMPROVED，但严格Corner Reveal为NO_REVEAL。此次漏掉的是对侧袖区可辨性及相对展开改善，不是一个已经确认的隐藏袖口恢复。']),
('1. 当前输入与语义位置',[
'实际记录Pa=(832,423)，Pb=(925,420)：画面右侧袖片附近抓取，向右略向上。索引水平距离约70.16 mm；索引抬升50 mm，原生命令Z增加65 mm，均不等于实测布料位移。',
'两组预测都将画面右上操作侧暂定为LS，将画面左上较宽袖片暂定为RS。盲评价对RS的语义对应为CLEAR，因此这里不是把画面右侧误称RS导致的评分错误。',
'查询将左袖描述为“展开较宽”，将注意力集中在右侧窄袖片、夹持身份、整体搬移和出画下摆；没有明确提出对侧袖部进一步展开这一检索问题。该查询倾向是原文可见事实，但无法据此量化它对最终错误的因果贡献。']),
('2. 检索与知识传递路径',[
'原库1222条；整run留出后本轮只读索引1168条。T06三条查询经BM25排序，中文双字片段与英文/数字词匹配；可观测条件加倍，机制、预测、观察、意图和不确定性参与评分。前24条交给独立选择器，最多3条交给预测器。',
'最终顺序为X0229-H001（原BM25第4，59.252902）、X0412-H002（第6，56.697558）、X0281-H001（第8，54.046235）。此顺序仅为保存的选择列表顺序，选择器没有输出数值相关性评分，也没有声明它是优劣排序。原文与逐条适用性判断见附录。',
'这3条没有明确提出“操作某侧使另一侧袖口可见性改善”的预测。它们包含非局部联动，但指定的改善/变化对象主要是衣身标志、胸片、下摆或被操作袖口；不能把一般联动扩写成原文已经预测对侧袖口改善。']),
('3. 相关知识有没有被遗漏？',[
'有。X0169-H003，第3，60.911452，明确预测“对侧可见性改善”伴随相对胸片错动、折角变化或搭接减少，并明确区分辨识改善与真正揭露。选择器输入确实含该完整记录，预测器输入不含。它是最直接的24→3信息丢失证据。',
'该历史条目的操作侧与T06相反，且其目标对侧呈紧凑回折、袖段不明确；T06查询将待改善的对侧描述为较宽展开。因此不能假定镜像迁移或条件完全匹配。记录没有针对该条的拒绝理由，无法确定是条件不足、竞争排序还是其他模型选择偏好。',
'另外X0156-H001第11提供覆盖边相对袖片退移；X0305-H001第21和X0222-H001第23提供另一袖区随动/差动响应。它们也在24条候选中，均未被选中；随动本身不保证可辨性增加。',
'仅作事后文本检索诊断：允许库中的X0506-H001也明确提到另一侧袖口更易辨认，但原排名1005、18.606721，未进前24。说明首轮词面排序还可能低估关系相关性；然而第3名已召回，所以BM25漏召回不是本例唯一或最直接瓶颈。此项不是修改排名，也不证明该条在T06一定适用。']),
('4. 最终为何仍预测RS不变？',[
'Hindsight保存的RS依据是：“该袖区已有较清楚外轮廓，距直接操作区较远，最佳预测为随衣身移动而可辨识程度基本保持。”不确定性是：“肩部重排可能改变袖片褶皱，但没有足够依据预测新的真实端点显露。”',
'可审计的决策是：将“原本已有可辨轮廓＋远离直接抓点＋整体随动”综合成“不变”。它承认肩部重排与袖片褶皱变化的可能，但没有将其转成RS可辨性改善预测。',
'存在指标混用迹象：用“没有足够依据预测新的真实端点显露”支持区域UNCHANGED，而冻结提示已明确区域改善不需要严格Reveal。这不能证明模型内部把二者完全混同，但原文不能支持“无严格显露证据，因此区域必定不改善”的推断。',
'不是完全没有利用历史假设：Baseline预测操作侧LS相对连接处CONTOUR_EXTENSION增加、局部展开为主；Hindsight改为相对伸展基本不变、整体随动占主导，且文字直接呼应X0229-H001。两组四端点可见性标签却完全相同。原记录没有逐句假设ID归因，故此处是文本对应证据，不是对内部推理的确定还原。']),
('5. 已揭示After与错误边界',[
'冻结盲评价：RS为NON_DIRECT；Before袖部向下收拢、袖端轮廓弯曲，After袖口斜边、宽袖面及袖身接缝更易区分。EXPOSED_AREA=INCREASE/CLEAR，CONTOUR_EXTENSION=INCREASE/CLEAR；FOLD_COVERAGE=UNKNOWN。',
'RS的严格Reveal为NO_REVEAL：Before已能看到袖口区域，没有确认同一真实端点由自遮挡到显露。因此两组RS可见性UNCHANGED均错，严格Reveal=NO均与评价一致。',
'Outcome Observability不是该RS可辨性漏判的主要原因：原盲评价有明确终态形状证据。但它限制机制判断：没有中间帧，无法区分接触滞留、抬升或落放时的重排；评价本身也只是一次独立模型视觉参考，不是新增物理真值。LS身份及边界不够明确，其几何计分为NA；不得据此声称Hindsight的整体搬移解释已验证。']),
('6. 分阶段瓶颈判定',[
'Retrieval／选择：强证据。有关对侧可辨性改善的第3名进入选择器，却未传递给预测器；现有库并非空缺。首轮排序的补充问题由第1005名提供线索，但不能取代24→3这一直接证据。',
'Hypothesis Applicability：真实限制、原因未决。入选3条有局部几何/命令幅度相似性，选择器也保存了材料、接触、朝向等未知条件。未选第3名需要跨侧及折形迁移；没有逐条淘汰日志，不能断言其被正确或错误地判定不适用。',
'Prediction Synthesis：有证据的第二瓶颈。已利用“整体运动”信息，却将RS随动与区域不变绑定；同时用严格Reveal证据不足解释区域判断。现有文本支持这两点，但不能证明若换入另一假设，预测一定改善。',
'Outcome Observability：对区域改善不是主要瓶颈，对严格Reveal与物理机制仍是重要边界。RS终态可判定，但机制不可验证；不得把NO_REVEAL改成成功。']),
('7. 留出假设与只读边界',[
'X0294-H003/H004只在本次预测冻结后的审计附录出现。它们既不在1168条可检索索引，也未出现在T06选择器和预测器输入中。H003提出桌面接触滞留，H004提出抬升/回落重排，均为原有PROVISIONAL竞争解释，physical_mechanism_validated=false。',
'这两条只能说明本例事后分析曾提出哪些解释，不能补作Hindsight当时可用知识，不能用来责备它没有读到应当被隔离的信息。',
'本次没有读取新图像作重新视觉判定，没有调用模型，没有修改预测、评价、排名或Hypothesis Bank；没有生成新物理假设、Skill或动作。引用其他经验仅为核查T06允许索引和原排名，不进行其他案例的效果分析。诊断依据是保存的输出，不包含隐藏推理；无法证明候选选择错误的唯一原因。'])]
checks={'source_files_unchanged':True,'new_model_calls':0,'new_visual_evaluations':0,'robot_calls':0,'input_presence':visibility,'rank3_in_selector_not_predictor':visibility['X0169-H003']['selector_input'] and not visibility['X0169-H003']['predictor_input'],'heldout_hypotheses_excluded':all(not any(visibility[k].values()) for k in ['X0294-H003','X0294-H004'])}
audit={'case':'T06 / X0294','scope':'post-freeze read-only diagnostic','sections':[{'title':t,'paragraphs':v} for t,v in sections],'sample':sample,'retrieval_original':ret,'selected_with_original_ranks':selected,'original_top24':scores[:24],'posthoc_unselected_diagnostics':unselected,'frozen_predictions':pred,'blind_observation':obs,'heldout_posthoc_only':held,'source_hashes':hashes,'checks':checks}
OUT.mkdir(exist_ok=True)
(OUT/'diagnostic.json').write_text(pretty(audit))
def esc(x):return html.escape(str(x))
def block(x):return '<pre>'+esc(pretty(x))+'</pre>'
parts=['<!doctype html><meta charset="utf-8"><title>T06 / X0294 知识利用诊断</title><style>body{max-width:1050px;margin:30px auto;font:17px/1.7 sans-serif;color:#203045}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f5f7;padding:14px}img{max-width:100%}section{border-bottom:1px solid #ccc;padding:12px}summary{cursor:pointer}</style><h1>T06 / X0294：完整知识利用路径</h1><p><a href="diagnostic.pdf">PDF</a> · <a href="diagnostic.json">JSON及原文</a></p>']
for title,paras in sections:parts+=['<section><h2>'+esc(title)+'</h2>']+['<p>'+esc(x)+'</p>' for x in paras]+['</section>']
parts+=['<h2>原图引用（不重新评价）</h2>']
for k in ['before','after']:parts+=['<h3>'+k+'</h3><img src="'+esc(os.path.relpath(sample[k],OUT))+'">']
appendices=[('原始查询',ret['query']),('原始Top24分数',scores[:24]),('入选3条：全文、原排名和相关性判断',selected),('未入选条目的事后路径核查',unselected),('两组冻结预测全文',pred),('原盲评价全文',obs),('留出H003/H004：仅供冻结后的审计',held),('源文件哈希与隔离检查',{'hashes':hashes,'checks':checks})]
for title,data in appendices:parts+=['<details><summary>'+esc(title)+'</summary>'+block(data)+'</details>']
(OUT/'review.html').write_text(''.join(parts))
# Use only existing font helpers, never execute the previous report's data reads.
tree=ast.parse((ROOT/'experiments/hypothesis_audit_20261102/export_pdf.py').read_text());nodes=[]
for n in tree.body:
 if isinstance(n,(ast.Import,ast.ImportFrom,ast.FunctionDef)):nodes.append(n)
 elif isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='styles' for t in n.targets):nodes.append(n)
 elif isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='registerFont':nodes.append(n)
ns={};exec(compile(ast.Module(body=nodes,type_ignores=[]),__file__,'exec'),ns);p=ns['p']
story=[p('T06 / X0294','h1'),p('只读知识利用路径与瓶颈诊断','h1')]
for title,paras in sections:story += [p(title,'h2')]+[p(x) for x in paras]
story += [PageBreak(),p('原始Before / After（未重新评价）','h1')]
for k in ['before','after']:story += [p(k,'h2'),Image(sample[k],width=500,height=281.25)]
for title,data in appendices[:-1]:story += [PageBreak(),p(title,'h1'),p(pretty(data),'small')]
story += [p('审计检查','h2'),p(pretty(checks)),p('完整源文件SHA256及路径保存在diagnostic.json。原文复制不改写；审计叙述与原始模型输出分开保存。')]
def footer(c,d):c.setFont('Latin',8);c.drawString(40,23,'T06 / X0294 — Read-only diagnostic');c.drawRightString(A4[0]-40,23,str(d.page))
SimpleDocTemplate(str(OUT/'diagnostic.pdf'),pagesize=A4,leftMargin=40,rightMargin=40,topMargin=38,bottomMargin=40).build(story,onFirstPage=footer,onLaterPages=footer)
assert all(sha(Path(k))==v for k,v in hashes.items())
assert checks['rank3_in_selector_not_predictor'] and checks['heldout_hypotheses_excluded']
print(pretty(checks));print(OUT)
