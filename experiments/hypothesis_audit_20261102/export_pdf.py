"""Export existing audit only. No inference or changes to source data."""
import json
from pathlib import Path
from html import escape
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
R=Path('/mnt/newssd/sja/clothAgent_real/reports/intention_outcome_hypothesis_bank_v1')
a=json.loads((R/'hypothesis_audit.json').read_text());s=a['summary']
pdfmetrics.registerFont(TTFont('Chinese','/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'))
pdfmetrics.registerFont(TTFont('Latin','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
def markup(value):
    runs=[];current=None;buf=[]
    for c in str(value):
        font='Latin' if ord(c) in pdfmetrics.getFont('Latin').face.charWidths else 'Chinese'
        if c not in '\n\r\t' and ord(c) not in pdfmetrics.getFont(font).face.charWidths:
            raise ValueError('Missing glyph: '+repr(c))
        if font!=current and buf:
            runs.append('<font name="'+current+'">'+escape(''.join(buf)).replace('\n','<br/>')+'</font>');buf=[]
        current=font;buf.append(c)
    if buf:runs.append('<font name="'+current+'">'+escape(''.join(buf)).replace('\n','<br/>')+'</font>')
    return ''.join(runs)
styles={k:ParagraphStyle(k,fontName='Chinese',fontSize=z,leading=z*1.5,spaceAfter=7,wordWrap='CJK',textColor=colors.HexColor('#202734')) for k,z in [('body',9),('h1',19),('h2',13),('small',7.5)]}
def p(t,style='body'):return Paragraph(markup(t),styles[style])
def table(d):
 t=Table([[p(k),p(v)] for k,v in d.items()],colWidths=[235,280],hAlign='LEFT')
 t.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('GRID',(0,0),(-1,-1),.3,colors.HexColor('#ccd3dd')),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]));return t

def footer(c,doc):
 c.setFont('Latin',8);c.setFillColor(colors.grey);c.drawString(40,23,'Hypothesis Bank / Read-only audit / 2026-11-02');c.drawRightString(A4[0]-40,23,str(doc.page))
def build(name,story):
 SimpleDocTemplate(str(R/name),pagesize=A4,leftMargin=40,rightMargin=40,topMargin=38,bottomMargin=40,title='Intention–Outcome Hypothesis Mining 只读审计',author='Codex').build(story,onFirstPage=footer,onLaterPages=footer)

def summary():
 story=[p('Intention–Outcome Hypothesis Mining','h1'),p('532 对操作：只读审计','h1'),p('审计对象：现有结构化记录。未重新分析 RGB，未调用模型，未生成假设或执行机器人。'),p(str(R),'small'),
 p('1. 实际产物','h2'),table({'完成处理':'532 / 532；70 个 run','假设总数':1222,'关联四端点':663,'CORNER_REVEAL 类别':38,'UNEXPECTED_OCCLUSION 类别':41,'字段及证据链接完整':'1222 / 1222','独立验证的物理机制':0,'原始文件变化':0}),
 p('完成处理不等于观察确定：全部 532 条实际效果标为 PARTIAL。原库中有 3 项 PROVISIONAL 的 REVEAL，0 项 CONFIRMED 的 REVEAL。'),
 p('2. 完整性与空结果','h2'),p('意图提取、实际结果观察、差异比较、假设生成四阶段各完成 532 条。9 条空假设：8 条没有值得建立机制假设的差异；X0319 有未确认预期，但缺少机制归因证据。未处理 0 条。'),
 table(s['comparison_types']),p('UNSPECIFIED_CHANGE 表示原意图未提及，不是明确预测错误。EXPECTED_NOT_CONFIRMED 也不等于已证伪。'),
 p('3. 意外收益与负面效果','h2'),p('7 项明确 UNEXPECTED_CHANGE 的文字核对：5 项偏离原目标，1 项混合效果，1 项收益不明确。关联假设分别为 12、2、2 条；其余 1,206 条没有这种明确意外变化引用。该分类只审计原文，不重新核实物理效果。'),
 p('同案例、同端点正向变化关联 73 条假设，负向关联 32 条。正向中 42 条关联原意图未提及的变化，30 条关联原本预期的变化；这些分组非互斥。负向中 28 条属于原意图未提及，4 条属于原本预期改变的区域。不能把关联计数直接称为意外收益或预测错误。'),
 p('X0162 的 EXPLICIT_UNCHANGED 指避免牵动袖部，原记录明确说并不等于预测袖口可见性不变。'),
 p('4. 质量与新增知识','h2'),p('1,222 条假设的条件、机制、预测、反证及原始证据链接均存在，没有发现字段缺失或引用断裂。但这是结构完整性，不证明每条假设具有充分证据或独立新颖性。'),
 table(s['hypothesis_evidence_levels']),p('CONFIRMED 是原模型的证据标签；全部 physical_mechanism_validated=false。'),
 p('41 个原始案例已出现对侧、摩擦、间接等关系词，共关联 102 条假设。这只是来源审计线索，不能自动判断整条假设是重复或新增。X0156 原意图已提出移动上层、下层留桌、相对位移释放袖部；X0097 已提出借桌面摩擦显露对侧。不能将这些既有关系记作本轮首次自主发现。'),
 p('当前实际产物是 1,222 条可溯源的解释性候选，而不是 1,222 条已获得的新知识。原库没有独立新颖性标签，不能据此给出可信的自主新发现数量。'),
 p('5. 疑似重复与审计边界','h2'),p('按机制文字四字片段 Jaccard≥0.55 标记相似对，本次检出 '+str(s['possible_text_duplicate_pairs'])+' 对。未检出不代表没有语义重复；没有合并或改写。')]
 for t in a['limits']:story.append(p('• '+t))
 story += [p('6. 空假设案例原始理由','h2')]
 for c in a['experiences']:
  if c['empty_result_class']:story += [p(c['experience_id'],'h2'),p(c['original_empty_reason']),p(c['source'],'small')]
 story += [p('7. 代表性对照','h1')]
 by={r['hypothesis_id']:r for r in a['hypotheses']}
 for r in a['representatives']:
  h=by[r['id']];story += [PageBreak(),p(r['id']+' · '+r['role'],'h2'),p('审计说明：'+r['reason'])]
  story += entry(h,compact=True)
 story += [PageBreak(),p('8. 交付文件','h2'),p('主报告：hypothesis_audit.pdf\n完整逐条附录：hypothesis_audit_full_appendix.pdf\n可搜索网页：hypothesis_audit.html\n完整结构化审计：hypothesis_audit.json'),p('完整附录逐条保留全部 1,222 条假设、原始意图、差异引用和证据路径；网页用于快速定位。')]
 return story

def entry(row,compact=False):
 h=row['original_hypothesis'];out=[]
 if not compact:out.append(p(h['hypothesis_id']+' · '+h['source_experience_id'],'h2'))
 out.append(p('原始 Agent 意图：'))
 for k,v in row['novelty']['original_intention'].items():out.append(p(k+': '+str(v)))
 for k,label in [('intended_effect','提取的意图'),('observed_effect','记录的效果'),('mismatch','差异'),('observable_conditions','可观测条件'),('hypothesized_mechanism','候选机制'),('predicted_effect','预期效果'),('falsification_condition','反证条件'),('uncertainty','不确定性')]:out.append(p(label+'：'+h[k]))
 out.append(p('原始证据等级：'+h['evidence_level']+'；角点：'+str(h['related_corner_goals'])+'；类别：'+str(h['categories'])))
 if not compact:
  for m in row['linked_mismatches']:out.append(p(m['mismatch_id']+' / '+m['type']+'：'+m['comparison_evidence']))
 out.append(p('审计：字段与链接完整='+str(row['quality']['all_required_fields_and_links_present'])+'；机制独立验证='+str(row['quality']['mechanism_independently_validated'])+'；新颖性='+row['novelty']['status']))
 for k,v in h['evidence'].items():out.append(p(k+': '+str(R/v),'small'))
 return out
build('hypothesis_audit.pdf',summary())
print('Main PDF complete',flush=True)
story=[p('完整假设附录：全部 1,222 条','h1'),p('来源为现有审计 JSON，仅排版导出。所有假设保持原文，不重新分析或补写。正文的质量与新颖性边界同样适用于本附录。')]
for row in a['hypotheses']:story += [PageBreak()]+entry(row)
build('hypothesis_audit_full_appendix.pdf',story)
print('Full appendix complete',flush=True)
