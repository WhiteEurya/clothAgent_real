"""Render the historical index and independently authored hypotheses; no inference/actions."""
from index import *
import html,os,urllib.parse
from PIL import Image
E=lambda x:html.escape(str(x))
def href(p):return urllib.parse.quote(os.path.relpath(ROOT/p,OUT),safe='/')
def link(p,label):return f'<a href="{href(p)}">{E(label)}</a>' if (ROOT/p).exists() else ''
def blob(x):return '<pre>'+E(json.dumps(x,ensure_ascii=False,indent=2))+'</pre>'
def images(s):
 parts=[]
 for side in ['before','after']:
  path=s.get(side)
  if not path:parts.append('<div>缺失 '+side+'</div>');continue
  w,h=Image.open(ROOT/path).size
  if side=='before' and s['action'].get('grasp_px') and s['action'].get('target_px'):
   x,y=s['action']['grasp_px'];a,b=s['action']['target_px'];mid=s['id']
   visual=f'<div style="position:relative"><img loading="lazy" src="{href(path)}" alt="before"><svg style="position:absolute;inset:0" viewBox="0 0 {w} {h}" role="img" aria-label="before及已执行命令"><defs><marker id="arrow{mid}" markerWidth="7" markerHeight="7" refX="5" refY="3" orient="auto"><path d="M0,0 L0,6 L6,3 z" fill="#ffde38"/></marker></defs><circle cx="{x}" cy="{y}" r="9" fill="none" stroke="#ffde38" stroke-width="4"/><path d="M{x},{y} L{a},{b}" stroke="#ffde38" stroke-width="4" marker-end="url(#arrow{mid})"/></svg></div>'
  else:visual=f'<img loading="lazy" src="{href(path)}" alt="{side}">'
  parts.append('<figure>'+visual+'<figcaption>'+link(path,side+' 原始RGB')+'</figcaption></figure>')
 return '<div class="pair">'+''.join(parts)+'</div>'
def step(s,full=False):
 ev=s['evaluation'];delta=ev['coverage']['delta_ratio'] if ev else None;motion=ev['rigid_motion'] if ev and 'rigid_motion' in ev else ev.get('motion',{}) if ev else {}
 # Preserve evaluator schema verbatim below; no replacement of unknowns with zeros.
 text=f"{s['step']} · {s['execution'].get('status','无执行记录')} · 面积变化 {delta*100:+.2f}%" if delta is not None else f"{s['step']} · {s['execution'].get('status','无执行记录')} · 面积 UNKNOWN"
 paths=[s['source']+'/'+n for n in ['execution.json','dispatch_intent.json','geometry.json','points.json','outcome_review.json','status.json']]+s['recorded_evaluation_files']
 body=images(s)+'<p>命令 Pa '+E(s['action'].get('grasp_px'))+' → Pb '+E(s['action'].get('target_px'))+'；距离 '+E(s['action'].get('distance_mm'))+' mm；抬升 '+E(s['action'].get('lift_height_mm'))+' mm。命令完成不证明夹住布料。</p>'
 body+='<p>'+ ' · '.join(link(p,Path(p).name) for p in dict.fromkeys(paths) if (ROOT/p).exists())+'</p>'
 body+='<details><summary>固定评价、动作参数与来源</summary>'+blob({'source':s['source'],'captured_at':s['captured_at'],'before_sha256':s['before_sha256'],'after_sha256':s['after_sha256'],'action':s['action'],'execution':s['execution'],'exclusion':s.get('exclusion'),'evaluation_recomputed':ev})+'</details>'
 return ('<article><h3>'+E(text)+'</h3>'+body+'</article>') if full else '<details class="step"><summary>'+E(text)+'</summary>'+body+'</details>'
def timeline(r):
 ss=r['steps'];w=760;h=110;n=max(len(ss),1);out=[f'<svg class="timeline" viewBox="0 0 {w} {h}"><line x1="0" y1="48" x2="760" y2="48" stroke="#8493a6"/>']
 for i,s in enumerate(ss):
  e=s['evaluation'];v=e['coverage']['delta_ratio'] if e else None;x=12+i*(w-24)/n
  if v is None:out.append(f'<text x="{x}" y="52" fill="#7e8790">?</text>')
  else:
   y=48-max(-.4,min(.4,v))*100;out.append(f'<rect x="{x}" y="{min(y,48)}" width="{max(3,(w-24)/n-5)}" height="{max(1,abs(y-48))}" fill="{"#168766" if v>0 else "#ba4b4b"}"><title>{E(s["step"])} {v:+.2%}</title></rect>')
  out.append(f'<text x="{x}" y="91" font-size="11">{E(s["step"].split("_")[-1])}</text>')
 out.append('</svg>');return ''.join(out)
def main():
 data=read(OUT/'index.json');p=read(OUT/'protocol.json');a=read(OUT/'independent_analysis.json');mapping=read(OUT/'case_mapping.json');allsteps={s['id']:s for r in data['runs'] for s in r['steps']}
 assert a and len(a['competing_explanations'])>=2,'Independent analysis is required'
 physical=[r for r in data['runs'] if r['physical_command_steps']];sensitivity={str(t):{'nonprogress_runs':sum(bool(r['screen'][str(t)]['windows']) for r in physical),'plateau_runs':sum(r['screen'][str(t)]['first_plateau'] is not None for r in physical)} for t in [.01,.02,.03]}
 timeouts=[]
 for r in data['runs']:
  evidence=json.dumps([r['summary'],r['status'],[s['execution'] for s in r['steps']]],ensure_ascii=False)
  if re.search(r'timeout|timed out|超时',evidence,re.I):timeouts.append(r['path'])
 save(OUT/'screen_summary.json',{'summary':data['summary'],'sensitivity':sensitivity,'timeout_evidence_runs':timeouts,'counts_overlap':True})
 audit={'independent_process':'Codex subagent /root/independent_failure_analysis, fork_turns=none; original CLI did not start','supplied_files':['analysis_packet.json','analysis_images.json']+['analysis_images/'+Path(x).name for x in read(OUT/'analysis_images.json')],'parent_history_supplied':False,'old_skills_and_planner_rationales_supplied':False,'input_hashes':{n:sha(OUT/n) for n in ['analysis_packet.json','analysis_images.json']},'output_sha256':sha(OUT/'independent_analysis.json'),'physical_actions':0,'existing_policy_modified':False,'context_audit_from_independent_analyst':a['context_audit'],'limitations':'Input isolation is not absence of pretrained knowledge. Commands completed do not establish grasp. Retrieval is not matched physical controls.'};save(OUT/'audit.json',audit)
 page=['''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>离线失败挖掘 · 20261029</title><style>body{font:16px/1.65 system-ui;margin:0;color:#213047;background:#eef2f6}main{max-width:1260px;margin:auto;padding:26px}h1,h2,h3{line-height:1.35}section,article,.run{background:white;padding:22px;border-radius:12px;margin:18px 0;border:1px solid #dae2eb}a{color:#0865ac}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f6f9;padding:16px;font-size:13px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border-bottom:1px solid #dce2e9;padding:8px;text-align:left;overflow-wrap:anywhere}summary{cursor:pointer;font-weight:650;padding:8px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:14px}figure{margin:0;min-width:0}img,figure svg{width:100%;height:auto}figcaption{font-size:13px}.timeline{width:100%;max-height:110px}input,select{padding:10px;margin:7px;font-size:15px}.note{border-left:5px solid #c99216;padding:16px;background:#fff6dc}.metrics{font-size:21px}.small{font-size:13px;color:#58677a}nav{position:sticky;top:0;background:#e4edf5;padding:10px;z-index:2}nav a{margin-right:18px}@media(max-width:700px){.pair{grid-template-columns:1fr}main{padding:10px}td{font-size:12px}}</style><main><h1>从真实衣物操作记录中发现值得验证的问题</h1><p>只做离线挖掘与独立假设分析。新增实机动作 0 次，现有策略修改 0 次。</p><nav><a href="#finding">研究问题</a><a href="#cases">证据序列</a><a href="#experiment">验证实验</a><a href="#index">run 索引</a><a href="#limits">范围与审计</a></nav>''']
 page.append('<section><h2>筛选概况</h2><p>冻结时间窗：'+E(p['start'])+' 至 '+E(p['end'])+'，Asia/Singapore，滚动 7 天。</p><p class="metrics">93 个有执行证据的 run · 27 个含人工中断 · 81 个未记录 DONE · 6 个连续无增长候选 · 2 个小变化平台</p><p>这些集合重叠。另索引 17 个没有确认执行动作的运行尝试。532 对动作前后记录通过来源检查，其中 421 对可计算固定面积代理。28 份复制记录去重；初始化和仅采图资料另存。</p><div class="note">未 DONE、人工中断、轮数上限均不自动等于失败。“有效动作”在本报告仅指执行完成且图像／动作来源可核验，不代表有效夹住布料。</div><p>初筛：连续三步各自面积增长 ≤2%；其中每步绝对变化 ≤2% 才标为平台。负向大变化归入无增长候选，不叫“几乎无形变”。阈值未经噪声标定，未知评价会中断窗口。</p>'+blob(sensitivity)+'<p>在已有终止、状态和执行日志中检出超时字样的 run 数：'+str(len(timeouts))+'。缺失终止日志保留 UNKNOWN。</p></section>')
 page.append('<section id="finding"><h2>独立 Codex 选择的问题</h2><h3>'+E(a['chosen_problem'])+'</h3><p>'+E(a['why_this_problem'])+'</p>')
 for o in a['repeated_observations']:page.append('<h3>'+E(o['case_steps'])+'</h3><p><b>图像证据：</b>'+E(o['visible_evidence'])+'</p><p><b>重复动作：</b>'+E(o['action_pattern'])+'</p><p><b>未知结构：</b>'+E(o['uncertain_structure'])+'</p>')
 page.append('<h3>其他候选与反例</h3><p>'+E(a['other_patterns'])+'</p></section><section><h2>竞争解释：均为待检验机制</h2>')
 for h in a['competing_explanations']:page.append('<h3>'+E(h['id'])+' · '+E(h['mechanism'])+'</h3><p>支持证据：'+E(h['support'])+'</p><p>反证／未知：'+E(h['counterevidence_or_unknown'])+'</p><p><b>区分预测：</b>'+E(h['distinct_prediction'])+'</p>')
 page.append('</section><section id="experiment"><h2>优先验证实验——只建议，未执行</h2>')
 labels={'reference_state':'参考状态','action':'建议动作','changed_variable':'改变变量','reset_and_comparison':'重置与对照','predictions_by_mechanism':'各机制的预期结果','measurements':'测量','ambiguity_and_safety':'不确定性及安全约束'}
 for k,v in a['priority_experiment'].items():page.append('<h3>'+labels[k]+'</h3><p>'+E(v)+'</p>')
 page.append('</section><section id="cases"><h2>代表性序列及检索对照</h2><p>'+E(a['control_assessment'])+'</p><p>对照检索按相同相机／标定、跨 run、面积增长 &gt;5% 的历史动作筛选，再按平移和尺度归一化轮廓 IoU 排序；没有匹配层序、局部皱褶或实际夹取条件。<a href="control_search.json">检索结果</a></p></section>')
 for name,case in mapping.items():
  page.append('<section><h2>'+E(name)+' · '+E(case['run'])+'</h2>')
  rr=next((r for r in data['runs'] if r['path']==case['run']),None)
  if rr:page.append('<p>终止：'+E(rr['termination'])+'；首次持续无增长：'+E((rr['screen']['0.02']['first_nonprogress'] or {}).get('start','未命中'))+'；首次平台：'+E((rr['screen']['0.02']['first_plateau'] or {}).get('start','未命中'))+'</p>'+timeline(rr))
  for ident in case['ids']:page.append(step(allsteps[ident],True))
  page.append('</section>')
 page.append('''<section id="index"><h2>全部 run 索引与 RGB 序列</h2><p>展开 run，再展开步骤，可查看原始图片、动作参数、固定重算评价与历史日志链接。黄色箭头表示命令，不表示材料实际运动。</p><input id="query" placeholder="搜索 run 名称"><select id="filter"><option value="all">全部运行尝试</option><option value="physical">有真实执行</option><option value="interrupt">人工中断</option><option value="nonprogress">持续无增长</option><option value="plateau">小变化平台</option><option value="nodone">有执行且未 DONE</option><option value="timeout">日志含超时</option></select><span id="shown"></span></section>''')
 for r in data['runs']:
  phys=r['physical_command_steps']>0;w=r['screen']['0.02']['first_nonprogress'];pl=r['screen']['0.02']['first_plateau'];tags=['all']
  for flag,on in [('physical',phys),('interrupt',r['human_interrupted']),('nonprogress',bool(w)),('plateau',bool(pl)),('nodone',phys and not r['done_recorded']),('timeout',r['path'] in timeouts)]:
   if on:tags.append(flag)
  page.append('<details class="run" data-tags="'+' '.join(tags)+'" data-name="'+E(r['path'])+'"><summary>'+E(r['path'])+'<br><span class="small">终止 '+E(r['termination'])+' · 执行证据 '+str(r['physical_command_steps'])+' 步 · 数值评价 '+str(r['numeric_pairs'])+' 步 · 首次无增长 '+E(w and w['start'])+' · 首次平台 '+E(pl and pl['start'])+'</span></summary>'+timeline(r)+blob({'termination':r['summary'],'status':r['status'],'human_review':r['human_review'],'first_nonprogress':w,'first_plateau':pl}))
  for s in r['steps']:page.append(step(s))
  page.append('</details>')
 page.append('<section id="limits"><h2>范围、限制与可复核文件</h2><p>'+E(a['limitations'])+'</p><p>扫描覆盖项目内 rg --files --hidden --no-ignore 返回的 execution、observation 与 episode_summary 记录，以真实观测捕获时间纳入。没有以文件修改时间代替实验日期。无 before 元数据的旧格式执行日志单独列出，无法证明时间或图像绑定的记录未进入量化序列；因此这不是对所有硬件动作完整无遗漏的承诺。</p><p>局部材料点跟踪沿用既有评价器 UNKNOWN；刚体配准成功只表示一种可用运动拟合，不能证明衣物完全刚体移动。不同记录不是严格控制变量实验。独立分析只接收匿名化图像、命令、评价及机器人约束；父会话和旧人工技能未传入。</p>'+blob(audit)+'<p>'+ ' · '.join(f'<a href="{x}">{E(x)}</a>' for x in ['index.json','screen_summary.json','protocol.json','analysis_packet.json','independent_analysis.json','case_mapping.json','duplicates.json','excluded.json','auxiliary_observations.json','unhandled_execution_paths.json','audit.json'])+'</p></section>')
 page.append('''<script>function filter(){let q=document.querySelector('#query').value.toLowerCase(),f=document.querySelector('#filter').value,n=0;document.querySelectorAll('.run').forEach(e=>{let yes=e.dataset.name.toLowerCase().includes(q)&&e.dataset.tags.split(' ').includes(f);e.hidden=!yes;if(yes)n++});document.querySelector('#shown').textContent=n+' 个 run'}document.querySelector('#query').addEventListener('input',filter);document.querySelector('#filter').addEventListener('change',filter);filter();</script></main></html>''')
 (OUT/'review.html').write_text('\n'.join(page));print('Rendered',OUT/'review.html')
if __name__=='__main__':main()
