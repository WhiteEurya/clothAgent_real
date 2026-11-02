"""Render frozen retrieval judgments, no model calls or new hypotheses."""
import ast,json,html,os
from pathlib import Path
from collections import Counter
BASE=Path('/mnt/newssd/sja/clothAgent_real')
R=BASE/'reports/hypothesis_retrieval_20261102'
def load(p):return json.loads(p.read_text())
def esc(v):return html.escape(str(v))
def pretty(x):return json.dumps(x,ensure_ascii=False,indent=2)
def rel(p):return os.path.relpath(p,R)
a=load(R/'retrieval_analysis.json');rows=a['all_hypotheses'];by={r['blind_id']:r for r in rows};reviews={r['blind_id']:r for r in a['reviews']};rank=a['ranking'];known=load(R/'postfreeze_known_case_diagnostic.json') if (R/'postfreeze_known_case_diagnostic.json').exists() else None
raw_types={r['blind_id']:{m['type'] for m in load(R/'inputs'/f'{r["blind_id"]}.json')['linked_comparisons']} for r in rows}
gap_disagreements=[{'hypothesis_id':r['hypothesis_id'],'blind_id':r['blind_id'],'screening_effect_expectation':r['effect_expectation'],'raw_types':sorted(raw_types[r['blind_id']]),'basis':r['expectation_basis']} for r in rows if r['effect_expectation']=='CONTRADICTED' and 'UNEXPECTED_CHANGE' not in raw_types[r['blind_id']]]
(R/'expectation_label_discrepancies.json').write_text(pretty({'note':'分类分歧，未经全量图像复核，不能认定为新的明确预测错误；不修改原排名。','rows':gap_disagreements})+'\n')
stats={'覆盖':a['coverage'],'候选假设数':len(rank['candidate_ids']),'来源操作数':a['candidate_source_operations'],
 '候选中有已记录改善线索':sum(r['observed_benefit_clue'] for r in rows if r['preliminary_candidate']),
 '候选中原CORNER_REVEAL类别':sum('CORNER_REVEAL' in r['original_hypothesis']['categories'] for r in rows if r['preliminary_candidate']),
 '初筛排除原因（可重叠）':dict(Counter(reason for r in rows for reason in r['exclusion_reasons'])),
 '效果预期关系':dict(Counter(r['effect_expectation'] for r in rows)),
 '操作关系预期':dict(Counter(r['relation_expectation'] for r in rows)),
 '新颖性评分':dict(Counter(r['operational_novelty'] for r in rows)),
 '检索标为CONTRADICTED但引用中无UNEXPECTED_CHANGE的条目数':len(gap_disagreements),
 '重点复核结论':dict(Counter(r['review']['inversion_potential'] for r in a['reviews'] if r['group']=='top')),
 '未入选抽查结论':dict(Counter(r['review']['inversion_potential'] for r in a['reviews'] if r['group']=='random_excluded'))}
call_results=[]
for result_path in sorted((R/'calls').glob('*/attempt_*/result.json')):
 result=load(result_path)
 if not result.get('usage'):
  events=[]
  for line in (result_path.parent/'events.jsonl').read_text().splitlines():
   try:events.append(json.loads(line))
   except ValueError:pass
  result=dict(result,usage=next((e['usage'] for e in reversed(events) if 'usage' in e),None))
 call_results.append(result)
usage=Counter()
for rr in call_results:
 for k,v in (rr.get('usage') or {}).items():
  if isinstance(v,(float,int)):usage[k]+=v
stats['模型成本']={'attempts':len(call_results),'completed':sum(r['status']=='COMPLETE' for r in call_results),'sum_request_seconds':round(sum(r['latency_s'] for r in call_results),1),'wall_clock_seconds':round(max(r['finished_at'] for r in call_results)-min(r['finished_at']-r['latency_s'] for r in call_results),1),'reported_usage':dict(usage),'failed_attempts':[r.get('error','') for r in call_results if r['status']!='COMPLETE'],'note':'请求秒数含并发，不等于墙钟时间；token子项不额外相加。'}
worth=[r for r in a['reviews'] if r['group']=='top' and r['review']['inversion_potential']=='WORTH_CONSIDERING_PROVISIONALLY']
misses=[r for r in a['reviews'] if r['group']=='random_excluded' and r['review']['inversion_potential']=='WORTH_CONSIDERING_PROVISIONALLY']
conclusion={'provisional_priority_ids':[r['hypothesis_id'] for r in worth],
 'sampled_potential_misses':[r['hypothesis_id'] for r in misses],
 'reading':'这是模型复核认为值得后续Goal Inversion的候选，不是已验证规律；未在本轮生成新动作。抽查仅10条，不能据此断言筛选无漏检。',
 'scope':'原意图未提及不等于明确预测不会发生。Novelty只相对本条记录。',
 'screening_limit':'初筛候选池很宽；新颖性评分几乎都为新增/部分新增，不能把候选数量等同于有价值的新知识。检索与原库预期差异标签有分歧，应单列而非升级为明确预测错误。',
 'isolation_limit':'协调者曾读过已知案例；全量评分及图像复核使用匿名独立调用，不包含此前聊天、人工结论或已知案例名单。不同条目在同一文字批次内共享上下文。'}
(R/'report_statistics.json').write_text(pretty({'statistics':stats,'conclusion':conclusion})+'\n')
styles='''body{font:16px/1.65 system-ui;max-width:1250px;margin:auto;padding:25px;background:#f5f7fa;color:#182735}section,details{padding:16px;background:white;margin:14px 0;border:1px solid #cdd8e5;border-radius:8px}summary{cursor:pointer;font-weight:bold}pre{white-space:pre-wrap;overflow-wrap:anywhere}a{color:#145fa8}img{width:49%;object-fit:contain}.warn{padding:15px;background:#fff0cc}input,select{padding:10px;margin:5px}table{border-collapse:collapse;width:100%}td,th{padding:6px;border:1px solid #ccd3db;text-align:left;vertical-align:top}h3{margin-bottom:4px}p{margin-top:3px}'''
def link(p,t):return '<a href="'+esc(p)+'">'+esc(t)+'</a>'
def pre(x):return '<pre>'+esc(pretty(x))+'</pre>'
def facts(row,show_images=False):
 h=row['original_hypothesis'];input_data=load(R/'inputs'/f'{row["blind_id"]}.json')
 s='<p>'+link(rel(row['payload']),'原始决策记录')+' · '+link(rel(row['before']),'完整Before')+' · '+link(rel(row['after']),'完整After')+' · '+link('../intention_outcome_hypothesis_bank_v1/cases/'+row['source_experience_id']+'.html','原始观察')+'</p>'
 if show_images:s+='<div><img loading="lazy" src="'+esc(rel(row['before']))+'"><img loading="lazy" src="'+esc(rel(row['after']))+'"></div>'
 s+='<h3>原始意图</h3>'+pre(input_data['original_intention'])+'<h3>已记录动作与执行状态</h3>'+pre({'action':input_data['executed_action'],'execution':input_data['execution_information']})
 s+='<h3>原假设记录的效果</h3><p>'+esc(h['observed_effect'])+'</p><h3>已有机制（原文）</h3><p>'+esc(h['hypothesized_mechanism'])+'</p><h3>已有预测</h3><p>'+esc(h['predicted_effect'])+'</p><h3>已有反证条件</h3><p>'+esc(h['falsification_condition'])+'</p><p>原证据标签：'+esc(h['evidence_level'])+'；机制验证标志：'+esc(h['physical_mechanism_validated'])+'</p>'
 s+='<h3>原库引用的差异类别</h3>'+pre(sorted(set(m['type'] for m in input_data['linked_comparisons'])))+'<h3>逐条初筛 A–E</h3>'+pre({k:row[k] for k in ['goal_relevance','goal_basis','effect_expectation','relation_expectation','expectation_basis','operational_novelty','new_vs_original','testability','testability_basis','evidence_strength','evidence_basis','observed_benefit_clue','limitations','preliminary_candidate','exclusion_reasons']})
 if row['blind_id'] in reviews:s+='<h3>独立图像证据复核</h3>'+pre(reviews[row['blind_id']]['review'])
 return s
parts=['<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Hypothesis Bank Retrieval</title><style>'+styles+'</style><h1>Hypothesis Bank Retrieval &amp; Analysis</h1>',
 '<p>'+link('retrieval_report.pdf','PDF 主报告')+' · '+link('retrieval_full_index.pdf','全部条目 PDF')+' · '+link('retrieval_analysis.json','完整 JSON')+' · '+link('protocol.json','冻结协议')+'</p>',
 '<div class="warn">全量阅读的是既有假设及观察；新图像复核仅覆盖前15条和10条未入选抽样。模型打分不是物理真值。未新增、合并假设，未生成Skill或执行机器人。</div>',
 '<section><h2>覆盖、候选及结果</h2>'+pre(stats)+'<p>重要限制：本轮检索新颖性评分没有0分，说明相对简短原意图新增机制解释非常普遍，单靠该维度区分度有限。'+link('expectation_label_discrepancies.json','查看预期差异标签分歧')+'；不能把检索模型的CONTRADICTED直接当成已确认预测错误。</p>'+'<h3>核心结论</h3>'+pre(conclusion)+'</section>',
 '<section><h2>冻结规则</h2><p>A：目标相关性0–3；C：相对原意图新增关系0–2；D：可检验性0–4；E：相关观察证据0–3。B分别判断效果与关系：明确预期、未提及、违背明示预测或不确定。分数是检索判断，不是成功概率。</p>'+pre({k:a['protocol'][k] for k in ['profile','model','reasoning_effort','provider','seed','candidate_rule','ranking','random_sampling','isolation']})+'<p>所有候选保留，包括未进入前15条者。初筛原排名不随图像复核更改。</p></section>',
 '<h2>前15条：按冻结排名展示</h2><p>阅读提示：前4条主要讨论整体运动、局部展开未确认及角点出画，并非显露成功。原排名第6、11、14条的既有复核直接讨论另一侧袖部外露/可辨性变化；可重点对照其原始证据。这是阅读导航，未更改排名，也不是新一轮筛选。</p>']
for n,bid in enumerate(rank['top_ids'],1):
 row=by[bid];parts.append('<section id="top_'+bid+'"><h2>'+str(n)+'. '+row['hypothesis_id']+' / '+row['source_experience_id']+'</h2>'+facts(row,True)+'</section>')
parts.append('<h2>未入选随机抽查：单独报告</h2><p>固定种子抽样10条。即使复核认为有价值，也不修改原候选排名，作为潜在漏检保留。</p>')
for bid in rank['random_excluded_ids']:parts.append('<details><summary>'+by[bid]['hypothesis_id']+' / '+esc(reviews[bid]['review']['inversion_potential'])+'</summary>'+facts(by[bid],True)+'</details>')
# Data-driven contrast: no manually nominated historical IDs.
generic=sorted([r for r in rows if r['testability']<=2],key=lambda r:(r['testability'],r['blind_id']))[:3]
inherited=sorted([r for r in rows if r['relation_expectation']=='EXPLICIT_EXPECTED'],key=lambda r:r['blind_id'])[:3]
parts.append('<h2>对照：空泛/弱可检验条目与原意图已表达的关系</h2><p>按固定评分及匿名ID顺序各取最多3条，未追加图像复核；仅为文字检索判断示例。</p>')
for row in generic+inherited:parts.append('<details><summary>'+row['hypothesis_id']+'</summary>'+facts(row)+'</details>')
if known:parts.append('<details><summary>分析冻结后的已知案例召回诊断</summary>'+pre({'experience_id':known['experience_id'],'checked_after_analysis_freeze':True,'rows':[{'hypothesis_id':r['hypothesis_id'],'rank':r['frozen_candidate_rank'],'preliminary_candidate':r['preliminary_candidate'],'exclusion_reasons':r['exclusion_reasons']} for r in known['rows']]})+'</details>')
parts+=['<h2>全部1,222条：候选、未入选与原因</h2><input id="q" placeholder="搜索编号、机制、评价"><select id="f"><option value="all">全部</option><option value="yes">候选</option><option value="no">未入选</option></select><p id="n"></p>']
for row in sorted(rows,key=lambda r:r['hypothesis_id']):
 parts.append('<details class="item" data-search="'+esc(pretty(row).lower())+'" data-candidate="'+('yes' if row['preliminary_candidate'] else 'no')+'"><summary>'+row['hypothesis_id']+' · '+('候选' if row['preliminary_candidate'] else '未入选')+' · A/E/C/D='+str([row[k] for k in ['goal_relevance','evidence_strength','operational_novelty','testability']])+'</summary>'+facts(row)+'</details>')
parts.append('''<script>const q=document.querySelector('#q'),f=document.querySelector('#f'),items=[...document.querySelectorAll('.item')];function update(){let n=0;for(const e of items){e.hidden=!(e.dataset.search.includes(q.value.toLowerCase())&&(f.value==='all'||e.dataset.candidate===f.value));n+=!e.hidden;}document.querySelector('#n').textContent=`${n} / ${items.length}`;}q.oninput=f.onchange=update;update();</script></html>''')
(R/'review.html').write_text(''.join(parts))
# Fixed mixed Chinese/Latin font helpers, imported as definitions without running old export.
tree=ast.parse((BASE/'experiments/hypothesis_audit_20261102/export_pdf.py').read_text());nodes=[]
for node in tree.body:
 if isinstance(node,(ast.Import,ast.ImportFrom,ast.FunctionDef)):nodes.append(node)
 elif isinstance(node,ast.Assign):
  names=[n.id for t in node.targets for n in ast.walk(t) if isinstance(n,ast.Name)]
  if 'styles'in names:nodes.append(node)
 elif isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='registerFont':nodes.append(node)
ns={};exec(compile(ast.Module(body=nodes,type_ignores=[]),__file__,'exec'),ns);ns['R']=R
p,table,build,PageBreak=ns['p'],ns['table'],ns['build'],ns['PageBreak']
from reportlab.platypus import Image,Table,Spacer
from PIL import Image as PILImage

def pdfrow(row,images=False):
 h=row['original_hypothesis'];inp=load(R/'inputs'/f'{row["blind_id"]}.json');story=[p(row['hypothesis_id']+' / '+row['source_experience_id'],'h2')]
 if images:
  imgs=[]
  for k in ['before','after']:
   image_path=R/'calls'/('review_'+row['blind_id'])/'attempt_01'/(k+'.jpg')
   if not image_path.exists():image_path=Path(row[k])
   with PILImage.open(image_path) as im:w,ht=im.size
   imgs.append(Image(str(image_path),width=250,height=250*ht/w))
  story += [p('Before（左） / After（右）','small'),Table([imgs],colWidths=[257,257])]
 story += [p('原始意图：'+pretty(inp['original_intention'])),p('已记录动作：'+pretty(inp['executed_action'])),p('执行状态：'+pretty(inp['execution_information'])),p('原记录效果：'+h['observed_effect']),p('已有机制：'+h['hypothesized_mechanism']),p('已有预测：'+h['predicted_effect']),p('原证据等级：'+h['evidence_level']+'；所有机制未经新实机验证。')]
 story += [p('原库差异类别：'+str(sorted(set(m['type'] for m in inp['linked_comparisons'])))),p('初筛 A/E/C/D：'+str([row[k] for k in ['goal_relevance','evidence_strength','operational_novelty','testability']])),p('预期差异：'+row['expectation_basis']),p('相对新增关系：'+row['new_vs_original']),p('可检验性：'+row['testability_basis']),p('证据：'+row['evidence_basis']),p('限制：'+row['limitations']),p('未入选原因：'+str(row['exclusion_reasons']))]
 if row['blind_id']in reviews:
  rev=reviews[row['blind_id']]['review']
  for k,label in [('visual_support','视觉支持'),('visible_before_after_evidence','图像证据'),('semantic_identity_limits','语义身份限制'),('already_in_original','原意图已有'),('added_relation','新增关系'),('evidence_vs_mechanism','观察/解释区别'),('inversion_potential','后续研究价值'),('reason','复核理由'),('limitations','限制')]:story.append(p(label+'：'+str(rev[k])))
 story += [p('原始记录：'+row['payload'],'small'),p('Before：'+row['before'],'small'),p('After：'+row['after'],'small')]
 return story
story=[p('Hypothesis Bank Retrieval & Analysis','h1'),p('只读检索：原始意图未明确表达的 Corner Reveal 关系','h1'),p('默认codex；gpt-6-astra / medium / openai。此前api profile因额度不足失败，未产生本轮检索结果。'),p('本轮不新增假设，不合并，不生成Skill，不执行机器人。全部结果保留原始证据；初筛候选不等于知识已验证。'),p(pretty(stats)),p('核心结论','h2'),p(pretty(conclusion)),p('冻结规则','h2'),p('A=目标相关性0–3；C=相对原意图新增关系0–2；D=可检验性0–4；E=相关观察证据0–3。B分别判断效果与关系是否已预期、未提及、与明示预测不一致或不确定。评分不是成功概率。'),p(a['protocol']['candidate_rule']),p(a['protocol']['ranking']),p('全部候选保留；仅前15条及10条随机未入选条目复核原图。抽查结果不改变初始排名。'),p('阅读导航：原排名第6、11、14条直接涉及另一侧袖部外露或可辨性改善；前4条主要讨论运动、裁切和可见性损失。仍按原排名展示，不重排，也不宣称显露已确认。'),p('初筛新颖性评分没有0分，候选池非常宽，不能据此宣称获得大量高价值知识。检索CONTRADICTED与原库分类分歧共'+str(len(gap_disagreements))+'条，保存在expectation_label_discrepancies.json；未经复核不升级为明确预测错误。')]
for n,bid in enumerate(rank['top_ids'],1):story += [PageBreak(),p('重点候选 #'+str(n),'h1')]+pdfrow(by[bid],True)
story += [PageBreak(),p('未入选随机抽查（单独统计）','h1'),p('潜在漏检：'+str(conclusion['sampled_potential_misses'])+'。抽样规模不足以证明无漏检。')]
for bid in rank['random_excluded_ids']:story += [PageBreak()]+pdfrow(by[bid],True)
story += [PageBreak(),p('高价值候选的文字对照','h1'),p('以下低可检验性/已表达关系示例仅经全量文字检索，不是额外图像复核。')]
for row in generic+inherited:story+=pdfrow(row)
if known:story += [PageBreak(),p('冻结后召回诊断','h1'),p(pretty({'experience_id':known['experience_id'],'rows':[{'hypothesis_id':r['hypothesis_id'],'rank':r['frozen_candidate_rank'],'exclusion_reasons':r['exclusion_reasons']} for r in known['rows']]}))]
build('retrieval_report.pdf',story)
story=[p('全部1,222条检索判断与来源','h1'),p('逐条保留候选和排除原因。仅有指定25条经过图像复核；不把文字评分当作物理真值。')]
for row in sorted(rows,key=lambda r:r['hypothesis_id']):story += [PageBreak()]+pdfrow(row)
build('retrieval_full_index.pdf',story)
print(pretty({'statistics':stats,'conclusion':conclusion}),flush=True)
