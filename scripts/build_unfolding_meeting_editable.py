#!/usr/bin/env python3
"""Native editable PowerPoint from the same audited meeting sources."""
from pathlib import Path
import json,zipfile,math
from PIL import Image
from pptx import Presentation
from pptx.util import Inches,Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE,MSO_CONNECTOR
from pptx.enum.text import MSO_ANCHOR
from pptx.oxml.xmlchemy import OxmlElement
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'reports';O=R/'unfolding_meeting_brief_20261021'
def read(p):return json.loads(Path(p).read_text())
meta=read(O/'source_manifest.json');totals=read(O/'timing_totals.json');sets=meta['selection']
prs=Presentation();prs.slide_width=Inches(16);prs.slide_height=Inches(9)
FONT='Noto Sans CJK SC'
def rgb(c):return RGBColor.from_string(c.lstrip('#'))
def pos(px):return Inches(px/120)
def style(run,size,color,bold=False):
 run.font.name=FONT;run.font.size=Pt(size*.6);run.font.color.rgb=rgb(color);run.font.bold=bold
 rp=run._r.get_or_add_rPr()
 ea=OxmlElement('a:ea');ea.set('typeface',FONT);rp.append(ea)
def text(s,txt,x,y,w=1780,h=100,size=28,color='#334155',bold=False):
 shape=s.shapes.add_textbox(pos(x),pos(y),pos(w),pos(h));tf=shape.text_frame
 tf.word_wrap=True;tf.margin_left=0;tf.margin_right=0;tf.margin_top=0;tf.margin_bottom=0
 for i,line in enumerate(str(txt).split('\n')):
  p=tf.paragraphs[0] if i==0 else tf.add_paragraph();p.space_after=Pt(7);p.line_spacing=1.12
  r=p.add_run();r.text=line;style(r,size,color,bold)
 return shape
def rect(s,x,y,w,h,color,line=None,round=False):
 shape=s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if round else MSO_SHAPE.RECTANGLE,pos(x),pos(y),pos(w),pos(h));shape.fill.solid();shape.fill.fore_color.rgb=rgb(color)
 if line:shape.line.color.rgb=rgb(line)
 else:shape.line.fill.background()
 return shape
def arrow(s,a,b,color='#187a91',width=3):
 sh=s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,pos(a[0]),pos(a[1]),pos(b[0]),pos(b[1]));sh.line.color.rgb=rgb(color);sh.line.width=Pt(width)
 end=OxmlElement('a:tailEnd');end.set('type','triangle');sh.line._get_or_add_ln().append(end);return sh
def picture(s,p,box):
 with Image.open(p) as im:iw,ih=im.size
 x,y,w,h=box;scale=min(w/iw,h/ih);pw,ph=iw*scale,ih*scale;x+=(w-pw)/2;y+=(h-ph)/2
 s.shapes.add_picture(str(p),pos(x),pos(y),width=pos(pw),height=pos(ph));return x,y,scale
def action_image(s,p,box,a,b):
 x,y,k=picture(s,p,box);pa=(x+a[0]*k,y+a[1]*k);pb=(x+b[0]*k,y+b[1]*k);arrow(s,pa,pb,'#e8a014',2.3)
 for pt,label,color in [(pa,'Pa','#e1496c'),(pb,'Pb','#0099b5')]:
  dot=s.shapes.add_shape(MSO_SHAPE.OVAL,pos(pt[0]-4),pos(pt[1]-4),pos(8),pos(8));dot.fill.solid();dot.fill.fore_color.rgb=rgb(color);dot.line.fill.background()
  text(s,label,pt[0]+7,pt[1]-12,45,30,20,color,True)
def base(title,sub,notes=None):
 s=prs.slides.add_slide(prs.slide_layouts[6]);s.background.fill.solid();s.background.fill.fore_color.rgb=rgb('#f7f9fc')
 rect(s,0,0,1920,13,'#187a91');text(s,title,65,35,1790,72,46,'#102b3b',True);text(s,sub,65,115,1780,68,27,'#536575')
 text(s,f'UNFOLDING  /  2026.10.21    |    {len(prs.slides):02d}    |    可编辑版',65,1035,1700,30,20,'#64748b')
 s.notes_slide.notes_text_frame.text=(meta['slides'][len(prs.slides)-1]['notes'] if notes is None else notes)+'\nAll text, tables, arrows are native editable objects. Sources: source_manifest.json; timing_report.md.'
 return s
def table(s,headers,rows,y=220,widths=None,x=65,rh=65,size=26):
 widths=widths or [1780/len(headers)]*len(headers)
 tab=s.shapes.add_table(len(rows)+1,len(headers),pos(x),pos(y),pos(sum(widths)),pos(rh*(len(rows)+1))).table
 for i,w in enumerate(widths):tab.columns[i].width=pos(w)
 for j,row in enumerate([headers]+rows):
  tab.rows[j].height=pos(rh)
  for i,val in enumerate(row):
   c=tab.cell(j,i);c.fill.solid();c.fill.fore_color.rgb=rgb('#dcebf0' if j==0 else ('#ffffff' if j%2 else '#edf2f6'));c.margin_left=pos(12);c.margin_right=pos(10);c.margin_top=pos(4);c.margin_bottom=pos(4);c.vertical_anchor=MSO_ANCHOR.MIDDLE
   tf=c.text_frame;tf.word_wrap=True;p=tf.paragraphs[0];p.space_before=Pt(0);p.space_after=Pt(0);r=p.add_run();r.text=str(val);style(r,size,'#102b3b',j==0)
 return tab
G=R/'arm6_incremental_replay_20261019_224938';g=read(G/'graph_used.json');tr=read(R/'arm6_graph_incremental_20261019_v3/compiled/trajectory.json')
s=base('01  Graph：把状态与人工动作连成图','真实 replay 快照：93 nodes / 111 edges / 202 reference images；下图为实际子图。')
for j,n in enumerate(['n000','n001','n002']):
 node=next(a for a in g['nodes'] if a['node_id']==n);p=tr['observations'][node['representative_observations'][0]]['rgb']['path'];x=65+j*610
 rect(s,x,205,550,490,'#ffffff','#ccd9e2',True);picture(s,p,(x+12,225,526,350));text(s,n+'  '+node['task_context'],x+15,600,520,100,27)
 if j<2:arrow(s,(x+555,420),(x+603,420),width=3)
text(s,'e000：袖片内折',420,730,500,55,30,'#187a91');text(s,'e001：另一侧袖片内折',1010,730,600,55,30,'#187a91')
text(s,'在线：当前 RGB → 检索参考 → GPT 确认节点 → 选边 / grounding → Host → 执行',65,825,1780,60,33)
text(s,'节点记录可见布层关系；边记录已保存动作。此旧 Graph 含 folding 数据，不等同于后面的 139 条 unfolding memory。',65,920,1780,70,26)
a=read(G/'step_0000/matching/matching_attempts.json')['attempts'][0]
s=base('02  看起来相近，仍无法确认同一节点','保存的 GPT 判定：NO_MATCH；检索候选相近 ≠ 可确认动作可迁移。不是人工标注的识别错误率。')
text(s,'CURRENT',65,190,650,50,30,'#187a91',True);picture(s,G/'step_0000/before.png',(65,245,650,430))
text(s,'模型拒绝依据：\n图案露出比例、相邻素色布片覆盖、\n下方折边形状无法同时对齐。\n\n展示原 Top-10 中前 6 张；\n最终 NO_MATCH 针对整组候选。',65,710,650,280,27)
for i,oid in enumerate(a['image_ids'][1:7]):
 x=760+i%3*365;y=215+i//3*370;node=next(n['node_id'] for n in g['nodes'] if oid in n['representative_observations'])
 text(s,f'{i+1}. {node} / {oid}',x,y,350,45,24);picture(s,tr['observations'][oid]['rgb']['path'],(x,y+45,350,260))
text(s,'检索 0.158 s  +  GPT 复核 120.916 s；无动作执行。',760,960,1090,50,26,'#a4552c')
s=base('03  历史经验：before → executed action → after','冻结 memory：139 accepted unfolding steps / 16 sessions；不要求 exact state matching。')
mem=R/'trajectory_memory_full_episode_v1_collection_aligned/round_000/selected_experiences'
for i,(tid,step) in enumerate([('20261019T024000697635Z','traj_0002'),('20261020T072800968698Z','traj_0003')]):
 p=mem/tid/step;y=215+i*370;act=read(p/'action.json');start,target=[v['pixel_xy'] for v in act['selected_points']]
 text(s,tid+' / '+step,65,y,1200,40,24,'#187a91')
 for j,label in enumerate(['BEFORE','EXECUTED ACTION','AFTER']):text(s,label,65+j*415,y+45,400,35,21)
 picture(s,p/'before.png',(65,y+90,400,245));action_image(s,p/'before.png',(480,y+90,400,245),start,target);picture(s,p/'after.png',(895,y+90,400,245))
text(s,'真实 episode 使用情况\n\n7 次动作 / 12 次经验引用\n7 个 distinct cited steps\n139 个历史 step 被访问\n208 张去重源图被读取\n\n引用由原 plan.json 解析；\n读取不等于经验造成改善。',1350,240,500,650,29)
subs=[('历史 memory：7 步展开后人工停止','139 条经验 + 当前 RGB-D + 本 episode 历史','USER_STOP\n最终人工展开判定待评审'),('Goal-only Agent：直接规划','当前 RGB-D + 固定目标 + episode history；无历史 memory','2 + 6 动作\n深度拒绝后经授权换视角续跑\n最终 DONE'),('One-shot：展开效果较好的成功候选','current RGB + goal RGB → 单次 direct GPT；12 轮，8 动作','4 次拒绝 / 4 段人工续跑\n终止 ROUND_LIMIT\n人工展开判定待评审'),('One-shot：覆盖退化的失败案例','同一 one-shot policy；round_003 为重点复盘动作','6 动作后安全停止\n箭头动作把已展开区域重新覆盖')]
for i,((key,label,runs,vid),(title,sub,end)) in enumerate(zip(sets,subs)):
 s=base(f'{i+4:02d}  '+title,sub);poster=R/runs[0]/'round_000/before/rgb.png'
 s.shapes.add_movie(str(O/f'videos/{key}.mp4'),pos(65),pos(195),pos(1290),pos(726),poster_frame_image=str(poster),mime_type='video/mp4')
 t=totals[i];text(s,'点击视频播放\n\n'+end+f"\n\n规划均值 {t['planning_stats']['mean']:.2f} s\n运行累计 {t['wall_s']/60:.2f} min",1390,245,460,650,28)
 text(s,'保存的 OBS 组成的幻灯片视频；非连续运动录像，播放速度不代表真实耗时。',65,950,1780,55,26)
s=base('08  失败动作复盘：已展开区域被重新覆盖','One-shot restart02 / round_003：与最后的 Host 安全停止分开讨论。')
p=R/'one_shot_goal_full_episode_v1_restart02/round_003';plan=read(p/'plan.json')
text(s,'动作前 + 原始 Pa→Pb',65,185,870,45,30);text(s,'动作后',980,185,870,45,30)
action_image(s,p/'before/rgb.png',(65,230,870,580),plan['grasp_px'],plan['target_px']);picture(s,p/'after/rgb.png',(980,230,870,580))
text(s,'人工先前指出左侧展开区域被覆盖。展示原始前后图，不把 Host 拒绝或执行完成当作策略质量标签。',65,865,1780,100,30)
s=base('09  耗时总览：主要差异在规划侧','同一模型系列 / high；不同初态、输入和干预，属于代表案例描述，不是严格消融。')
table(s,['方法','调用数','规划 median','规划 mean','规划 p95','运行累计'],[[t['label'],t['planning_stats']['n'],f"{t['planning_stats']['median']:.2f}s",f"{t['planning_stats']['mean']:.2f}s",f"{t['planning_stats']['p95']:.2f}s",f"{t['wall_s']/60:.2f}min"] for t in totals],widths=[440,180,285,285,285,305],size=28)
text(s,'仅已执行 ACTION 的规划均值：',65,640,1780,55,32,'#187a91');text(s,'  /  '.join(f"{t['label']} {t['executed_action_planning_stats']['mean']:.2f}s" for t in totals),65,710,1780,80,28)
text(s,'Agent wall 包含启动、工具、读图与模型等待；内部模型 API 次数及纯推理时长无法从现有日志拆开。\nOne-shot wall 为单次 direct 请求；不含机器人时间。四个代表 run 均早于 shadow gate，gate=0。',65,840,1780,140,27)
s=base('10  总时长拆分：互斥项可加回运行总时长','单位：秒；续跑间人工等待不计入“运行累计”，单列于详细报告。')
labels=[('initialization_s','初始化（含其模型/机器人）'),('initialization_review_s','初始化确认等待'),('planner_s','展开规划 Agent / direct'),('host_s','Host 准备/几何检查'),('observation_s','每轮记录的采图'),('robot_s','操作 + 返回观测位'),('round_other_s','轮内其他 / 未单列'),('outside_round_other_s','轮外启动收尾 / 未单列'),('wall_s','运行总时长')]
table(s,['互斥时间项','Memory','Goal-only','One-shot 好','One-shot 失败'],[[lab]+[f'{t[k]:.2f}' for t in totals] for k,lab in labels],y=190,widths=[620,290,290,290,290],size=25)
text(s,'初始化模型是初始化总时间的子集；机器人返回是机器人总时间的子集，不能重复相加。',65,895,1780,55,27);text(s,'逐轮、初始化子项、动作/返回阶段、跨段等待：见 timing_report.md 与 per_round_timing.csv。',65,955,1780,55,26)
s=base('11  Graph 与机器人阶段：再细一层','Graph 采用同一 NO_MATCH run；机器人子阶段来自 recorder stage 时间戳，为近似分段。')
rt=a['retrieval_timing'];gt=read(G/'timing.json')
table(s,['Graph 在线项目','秒'],[[k,f'{rt[k]:.6f}'] for k in ['query_image_decode_s','dino_preprocess_s','dino_forward_s','ann_search_s']]+[['检索总计（包含以上项）',f"{rt['retrieval_total_s']:.6f}"],['GPT rerank',f"{a['gpt_rerank_s']:.3f}"],['全 run（含启动/采图/选择）',f"{gt['total_s']:.3f}"]],y=220,widths=[730,240],size=25)
text(s,'机器人执行子阶段（s）',1100,220,740,55,30,'#187a91')
for i,t in enumerate(totals):text(s,t['label']+f"\n操作阶段 ≈{t['manipulation_stage_approx_s']:.1f}  / 返回阶段 ≈{t['return_stage_approx_s']:.1f}",1100,310+i*130,740,115,27)
text(s,'返回阶段含回 Home/观测位、松爪及少量收尾；不是纯运动控制器时长。Graph 离线构建不含在在线耗时中。',65,930,1780,70,26)
# Three method sections with audited Graph comparison statistics.
import csv,hashlib
import numpy as np
B=R/'unfold_visual_retrieval_benchmark_20261019'
mem_index=read(B/'memory/memory.json');groups={r['image_id']:r['proxy_group'] for r in mem_index['images']}
bench_rows=list(csv.DictReader((B/'run/benchmark.csv').open()))
audit=[];graph_metrics={}
for mode in ['retrieval_only','retrieval_gpt_rerank','current_global_match']:
 trials=[r for r in bench_rows if r['condition']==mode];v=[float(r['total_s']) for r in trials]
 result=dict(n=len(v),median=float(np.median(v)),mean=float(np.mean(v)),p95=float(np.percentile(v,95)),completed=sum(r['status']=='OK' for r in trials))
 if mode!='retrieval_only':
  for r in trials:
   source=B/'run/trials'/mode/f"{int(r['iteration']):03d}/trial.json";trial=read(source);m=trial.get('match',{})
   candidate=m.get('best_candidate',m.get('node_id'));pred_group=groups.get(candidate,candidate);truth=groups[r['query_id']]
   hit=trial['status']=='OK' and m.get('status')=='MATCH' and pred_group==truth
   audit.append(dict(mode=mode,query_id=r['query_id'],expected_proxy_group=truth,selected_candidate=candidate,
    predicted_proxy_group=pred_group,status=trial['status'],match_status=m.get('status'),proxy_hit=hit,total_s=float(r['total_s']),
    source=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
  result['proxy_hits']=sum(r['proxy_hit'] for r in audit if r['mode']==mode)
 result['stage_mean_s']={k:float(np.mean([float(r[k]) for r in trials])) for k in ['query_image_decode_s','dino_preprocess_s','dino_forward_s','ann_search_s','candidate_loading_s','gpt_rerank_s','other_host_s']}
 graph_metrics[mode]=result
assert graph_metrics['retrieval_gpt_rerank']['proxy_hits']==9
assert graph_metrics['current_global_match']['proxy_hits']==9
(O/'graph_search_comparison.json').write_text(json.dumps(dict(metrics=graph_metrics,per_query=audit,
 label_type='Existing model-compiled graph groups, not human ground truth',planned_pairs=20,actual_pairs=10,
 early_stop='Operator requested outcome-dependent early stop at 10 pairs',reference_images=120),ensure_ascii=False,indent=2))
with (O/'graph_search_per_query.csv').open('w') as stream:
 writer=csv.DictWriter(stream,fieldnames=list(audit[0]));writer.writeheader();writer.writerows(audit)
comparison_notes='Graph comparison: '+str(B/'run/summary.md')+'; matching proxy accuracy re-derived from trial.json and memory/memory.json, see graph_search_comparison.json. Same 10 attempted queries; one wrong-group DINO+GPT match and one global request failure. Proxy groups are not human ground truth. End-to-end latencies include failures.'
s=base('区块一  Graph 搜索','当前图 → 搜索可复用状态 → 确认节点 → 选动作边',notes=comparison_notes)
for x,mode,title,process,color in [
 (65,'retrieval_gpt_rerank','DINO 检索 + GPT','DINO 筛 Top-10 → GPT 复核','#187a91'),
 (975,'current_global_match','全图搜索 + GPT','全部参考图 → GPT 全局匹配','#6b5c9a')]:
 t=graph_metrics[mode];rect(s,x,225,880,500,'#ffffff','#ccd9e2',True)
 text(s,title,x+30,255,820,60,38,color,True);text(s,process,x+30,330,820,55,29,color)
 text(s,f"耗时中位数  {t['median']:.2f} s",x+30,420,820,65,40,'#102b3b',True)
 text(s,f"mean {t['mean']:.2f} s   /   p95 {t['p95']:.2f} s",x+30,495,820,50,27)
 text(s,f"匹配命中（代理标签）  {t['proxy_hits']}/{t['n']} = 90%",x+30,565,820,55,31,color,True)
 reason='1 次匹配到不同组；请求完成 10/10' if mode=='retrieval_gpt_rerank' else '1 次超时；其余 9 次命中，请求完成 9/10'
 text(s,reason,x+30,640,820,55,26)
text(s,'仅 DINO 候选检索：median 48.55 ms；Top-10 代理组召回 108/108（不作最终匹配判定）。',65,770,1780,65,28,'#187a91')
text(s,'同一 120 张参考库、10 个配对 query；时间含失败。DINO + GPT 的中位数约快 4.47×。',65,855,1780,55,27)
text(s,'代理标签未人工核验；近邻帧可能相似。原计划 20 对，观察结果后提前停在 10 对；非展开成功率。',65,925,1780,65,25,'#64748b')
s=base('区块二  Memory / Agent 规划','不要求匹配 Graph 节点；在当前图重新确定单步动作。',notes='Sources: unfolding_method_comparison_20261020/comparison.md and original saved protocols. Goal-only is the no-memory Agent comparison, not memory retrieval. No human confirmed unfolding success rate.')
for x,t,title,inputs,process,result,color in [
 (65,totals[0],'历史 memory Agent','当前观测 + 139 条人工经验','浏览 before / action / after → 迁移操作模式','7 动作后人工停止；展开成功率待人工评审','#6b5c9a'),
 (975,totals[1],'Goal-only Agent（无 memory 对照）','当前观测 + 目标图 + 本场历史','看图、用工具 → 自由规划','8 动作，授权续跑后 DONE；成功率待人工评审','#b47a35')]:
 rect(s,x,230,880,600,'#ffffff','#ccd9e2',True)
 text(s,title,x+30,260,820,65,33,color,True);text(s,inputs,x+30,360,820,60,29)
 text(s,process,x+30,435,820,80,28,color)
 tstats=t['planning_stats'];text(s,f"规划 median  {tstats['median']:.2f} s",x+30,545,820,65,38,'#102b3b',True)
 text(s,f"mean {tstats['mean']:.2f} s / p95 {tstats['p95']:.2f} s（n={tstats['n']}）",x+30,625,820,55,27)
 text(s,result,x+30,720,820,85,27)
text(s,'这里报告单次 Agent 规划 wall time（含工具循环），不是纯检索时间；两者均会使用本场已执行反馈。',65,910,1780,90,28)
s=base('区块三  One-shot 规划','当前 RGB + 目标 RGB → 一次 GPT 请求 → 一个动作',notes='Sources: selected one-shot runs in source_manifest.json. Same original prompt, no tools, no episode history. Success candidate is user-selected; no human fold-ready label. Both cases remain pending human review.')
rect(s,65,215,1780,120,'#e5f0eb')
text(s,'gpt-6-astra / high     ·     无工具     ·     无历史     ·     无 retry / repair',100,250,1700,65,32,'#267f67',True)
for x,t,title,result in [(65,totals[2],'展开效果较好的案例','12 轮 / 8 动作，含人工续跑；ROUND_LIMIT'),(975,totals[3],'覆盖退化的失败案例','7 轮 / 6 动作；已展开区域被重新覆盖')]:
 rect(s,x,390,880,420,'#ffffff','#ccd9e2',True);text(s,title,x+30,425,820,65,35,'#267f67',True)
 ts=t['planning_stats'];text(s,f"规划 median  {ts['median']:.2f} s",x+30,530,820,65,38,'#102b3b',True)
 text(s,f"mean {ts['mean']:.2f} s / p95 {ts['p95']:.2f} s（n={ts['n']}）",x+30,610,820,65,27)
 text(s,result,x+30,710,820,65,27)
text(s,'观察重点：规划是否更快、动作是否合理。最终展开成功率尚无人工确认，不能从执行次数推算。',65,900,1780,95,29)
# Move methodology slides to the front without altering media relationships.
ids=prs.slides._sldIdLst;order=list(ids);new_order=[order[i] for i in [11,0,1,12,2,3,4,13,5,6,7,8,9,10]]
for node in list(ids):ids.remove(node)
for node in new_order:ids.append(node)
for i,slide in enumerate(prs.slides):
 for shape in slide.shapes:
  if not shape.has_text_frame:continue
  value=shape.text
  if value.startswith('UNFOLDING  /'):
   shape.text_frame.paragraphs[0].runs[0].text=f'UNFOLDING  /  2026.10.21    |    {i+1:02d}    |    可编辑版'
  elif shape.top==pos(35):
   import re
   title=re.sub(r'^(?:方法论 )?\d{2}  ', '',value)
   shape.text_frame.paragraphs[0].runs[0].text=f'{i+1:02d}  '+title
out=O/'unfolding_brief_editable_sections.pptx';prs.save(out)
# Structural validation: no slide-sized raster; real tables, text and movie objects.
check=Presentation(out);counts=[]
for i,s in enumerate(check.slides):
 c=dict(slide=i+1,text_boxes=sum(x.has_text_frame for x in s.shapes),tables=sum(x.has_table for x in s.shapes),pictures=sum(x.shape_type==13 for x in s.shapes),shapes=len(s.shapes))
 assert c['text_boxes']>=4
 for x in s.shapes:
  if x.shape_type==13:assert x.width<prs.slide_width*.95 or x.height<prs.slide_height*.95
 counts.append(c)
assert sum(c['tables'] for c in counts)==3
assert len(check.slides)==14
with zipfile.ZipFile(out) as z:assert len([n for n in z.namelist() if n.endswith('.mp4')])==4;assert z.testzip() is None
(O/'editable_sections_verification.json').write_text(json.dumps(dict(file=str(out),slides=counts,embedded_videos=4,native_tables=3,no_full_slide_raster=True),ensure_ascii=False,indent=2))
print(out)
