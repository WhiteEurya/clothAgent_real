"""Read-only structured extraction; PDF formatting only."""
from pathlib import Path
import json, hashlib, ast
R=Path('/mnt/newssd/sja/clothAgent_real/reports/intention_outcome_hypothesis_bank_v1')
paths=[R/'experience_observations.json',R/'hypothesis_bank.json',R/'inputs/X0518/payload.json']
hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
o=next(x for x in json.loads(paths[0].read_text()) if x['experience_id']=='X0518')
hs=[h for h in json.loads(paths[1].read_text()) if h['source_experience_id']=='X0518']
payload=json.loads(paths[2].read_text())
notes={
 'scope':'仅提取现有结构化文字；未查看或重分析RGB、未调用模型、未生成新假设。',
 'originally_explicit':'原始记录只明确表达：直接抓右侧内收袖口，向右下牵拉，以释放同侧叠置袖部。没有独立 action_purpose 或 expected_stop_state。',
 'knowledge_limit':'本轮只能判断这条决策记录明确表达了什么，不能从一条短理由断言原Agent完全不知道其他关系；也未检查生产Prompt或模型全部知识。',
 'H001_difference':'原意图期待局部释放；事后假设引入张力传至衣身、偏心牵引与不均匀支撑导致整衣平移/转向。这些解释没有写入该条原始理由，但未证明是自主获得的新知识。',
 'H002_difference':'原意图把对象认作袖口；事后提出可能夹到覆盖层或多层，反思原始材料身份判断，并与H001竞争。没有直接夹持影像，不能把误抓当作事实。',
 'H003_difference':'原意图只针对被抓侧；事后提出抬升经衣身传递，使未直接操作的另一侧袖片在重力/落放中被动翻出。这是原记录未表达的远端作用解释，仍为暂定假设。',
 'classification':'整条经验并无唯一差异类别。M001=EXPECTED_OBSERVED，M002=EXPECTED_NOT_CONFIRMED，M003/M004=UNSPECIFIED_CHANGE；没有UNEXPECTED_CHANGE。',
 'reveal_status':'画面左侧/暂定穿着者右袖RS：SUSPECTED_OCCLUDED→VISIBLE、REVEAL、PROVISIONAL、原预期UNSPECIFIED。侧别依赖正面朝上且图像未镜像的解释。',
 'key_limit':'未提及另一侧不等于明确预测它不变；只能说记录未指定的潜在收益，不能证明Agent主观上惊讶或明确预测错误。'}
result={'experience':o,'hypotheses':hs,'payload':payload,'extraction_notes':notes,'source_sha256':hashes}
(R/'X0518_readonly_extract.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
# Reuse fixed-font PDF helpers without executing their report-building statements.
source=Path(__file__).with_name('export_pdf.py').read_text();tree=ast.parse(source)
keep=[]
for node in tree.body:
 if isinstance(node,(ast.Import,ast.ImportFrom,ast.FunctionDef)):
  keep.append(node)
 elif isinstance(node,ast.Assign):
  names=[n.id for t in node.targets for n in ast.walk(t) if isinstance(n,ast.Name)]
  if set(names)&{'R','styles'}:keep.append(node)
 elif isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='registerFont':keep.append(node)
ns={};exec(compile(ast.Module(body=keep,type_ignores=[]),str(__file__),'exec'),ns)
p,table,build,PageBreak=ns['p'],ns['table'],ns['build'],ns['PageBreak']
story=[p('X0518：原始意图—实际效果—全部假设','h1'),p(notes['scope']),p(o['source'],'small'),p('1. 原始意图与动作','h2'),p('原文：'+payload['original_intention']['short_reason']),p(notes['originally_explicit']),p('Pa=(878,400) → Pb=(978,447)；lift_pull_place；夹爪角0°；索引水平位移约85.63 mm。命令已执行9条，但布料获取状态仍为UNKNOWN_PENDING_HUMAN_REVIEW，不能把命令完成当作抓取身份确认。'),p('2. 已记录的实际效果','h2')]
for t in o['observation']['observed_effect']['whole_garment_changes']:story.append(p('• '+t))
story += [p(notes['reveal_status']),p('上述是原分析记录的观察，不是本轮重新确认。实际效果总标签为PARTIAL。'),p('3. 差异类别','h2'),table({m['mismatch_id']:m['type'] for m in o['observation']['mismatches']}),p(notes['classification']),p(notes['key_limit']),p('4. 与原Agent明确表达的操作知识相比','h2'),p(notes['knowledge_limit'])]
for k in ['H001_difference','H002_difference','H003_difference']:story.append(p(notes[k]))
story += [p('三条假设均为PROVISIONAL，physical_mechanism_validated=false。H003最直接涉及未直接操作袖部的暂定显露；H001/H002主要解释整衣移动及目标局部释放未确认。')]
labels={'intended_effect':'原意图','observed_effect':'记录的效果','mismatch':'差异','observable_conditions':'可观测条件','hypothesized_mechanism':'可能机制','predicted_effect':'预期效果','falsification_condition':'可证伪条件','uncertainty':'不确定性'}
for h in hs:
 story += [PageBreak(),p(h['hypothesis_id']+' — 全部原文','h1')]
 for k,label in labels.items():story.append(p(label+'：'+h[k]))
 for k in ['evidence_level','mismatch_ids','related_corner_goals','categories','physical_mechanism_validated']:story.append(p(k+'：'+json.dumps(h[k],ensure_ascii=False)))
 for k,v in h['evidence'].items():story.append(p(k+'：'+str(R/v),'small'))
story += [PageBreak(),p('差异比较原文与证据边界','h1')]
for m in o['observation']['mismatches']:
 story.append(p(m['mismatch_id']+' / '+m['type'],'h2'))
 for k in ['intended_effect','observed_effect','comparison_evidence','uncertainty']:story.append(p(k+'：'+m[k]))
story += [p('执行证据限制','h2'),p(o['observation']['observed_effect']['execution_evidence_limits']),p('原始输入：'+str(paths[2]),'small')]
build('X0518_readonly_extract.pdf',story)
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
print('Saved X0518_readonly_extract.pdf and .json; originals unchanged.')
