"""Offline report only: read frozen records; never import robot/policy modules."""
import json, math, hashlib, base64, html
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/corner_reveal_mining_20261029'
rows=json.loads((OUT/'inventory.json').read_text())
notes={
98:('NOT_SELECTED_ALREADY_EXPOSED','RS','操作前图像左侧已有带缝边的袖口末段；动作后展开更明显，不足以证明从遮挡中显露。'),
108:('UNCERTAIN','RS','前图左侧折叠边中已有疑似袖口缝边；后图袖口明确，但不能排除原本已暴露。'),
133:('NOT_SELECTED_ALREADY_EXPOSED','LS','图像右侧袖口缝边在操作前已经露出；整体姿态变化不能作为 reveal。'),
153:('SUCCESS_CANDIDATE','RS','前图左侧肩部下方的袖部折在衣身侧后，印花前片遮住向内收回的袖口区域；未见可连续定位的袖口开口中点。后图同侧袖子释放，袖口缝边完整可辨。'),
178:('UNCERTAIN','RS','前图左侧露出一条袖部边缘，是否包含目标袖口中点无法确认；后图袖口明显，但有整体平移。'),
254:('NOT_SELECTED_ALREADY_EXPOSED','RS','前图左侧袖口及缝边已经外露；主要是袖部进一步展开。'),
285:('UNCERTAIN','RS','前后左袖仍贴在衣身侧，后图未形成足够明确的袖口目标点，不能确认显露。'),
311:('UNCERTAIN','RS','前图左侧已有小段袖部凸出；无法排除袖口已暴露但卷曲。'),
327:('UNCERTAIN','RS','前图左袖被斜跨的前片遮住的解释合理；但近左边缘可能已有部分袖口，暂不列入最强候选。'),
353:('SUCCESS_CANDIDATE','RS','前图左肩下方是向内收拢的连续衣身边，左侧袖口开口及其中点不可见；后图同一肩部连接出袖子和袖口边，腋下出现背景间隙。'),
376:('UNCERTAIN','RS','前图袖部卷成小团，后图仍窄且卷；不能证明袖口中点完成遮挡到显露的转换。'),
431:('NOT_SELECTED_ALREADY_EXPOSED','RS','前图横跨胸部的袖口已可见；后图主要改变位置和展开程度。另一袖口是否被覆盖仍不确定。'),
495:('UNCERTAIN','RS','前图层叠复杂且局部过曝，存在疑似已露出的袖口；后图展开不足以反推此前遮挡。')}
selected=[]; audits=[]
for n,(status,target,note) in notes.items():
 r=rows[n-1]; audits.append({'event_id':r['candidate_id'],'status':status,'target_semantic_endpoint':target,'reason':note,'review_level':'native_resolution_region_review','source':r['source'],'before':r['before'],'after':r['after']})
 if status!='SUCCESS_CANDIDATE': continue
 a=dict(r['action']); dx=a['target_px'][0]-a['grasp_px'][0];dy=a['target_px'][1]-a['grasp_px'][1]
 a.update(direction_image_delta_px=[dx,dy],direction_image_degrees=math.degrees(math.atan2(dy,dx)),direction_convention='x right, y down; atan2(dy,dx)',parameter_scope='executed command parameters; not measured fabric displacement',measured_fabric_distance_mm=None)
 for k in ['before','after']:
  assert hashlib.sha256((ROOT/r[k]).read_bytes()).hexdigest()==r[k+'_sha256']
 exe=json.loads((ROOT/r['source']/'execution.json').read_text());assert exe['physical_execution'] and exe['status']=='EXECUTED'
 roi=[465,240,560,420] if n==153 else [520,235,650,435]
 p=[396,403] if n==153 else [568,390]
 selected.append({'event_id':r['candidate_id'],'run':r['run'],'round':r['step'],'source':r['source'],'timestamp':r['captured_at'],'status':status,'human_review_status':'PENDING','human_confirmed':False,'eligible_as_confirmed_training_label':False,'target_semantic_endpoint':target,'identity_basis':'正面印花、领口与肩袖连续关系：本例图像左侧对应穿着者右袖 RS；不按轮廓尖角命名。','before':{'path':r['before'],'sha256':r['before_sha256'],'state':'SELF_OCCLUDED_CANDIDATE','point_px':None,'inspection_roi_xyxy':roi,'evidence':note.split('后图')[0]},'after':{'path':r['after'],'sha256':r['after_sha256'],'state':'EXPOSED','point_px_approx':p,'point_annotation':'人工视觉近似中点，非追踪真值','evidence':'同一右肩延续到外露袖部，可辨认袖口开口的缝边；目标点约在该边中部。'},'actual_executed_action':a,'execution':exe,'same_view_from_index':r['evaluation']['quality']['same_view'],'execution_record':r['source']+'/execution.json','geometry_record':r['source']+'/geometry.json','dispatch_record':r['source']+'/dispatch_intent.json','other_endpoints':'未作为筛选条件；未对另外三个端点逐一给出真值。','out_of_view_excluded':'原始全幅图中衣物和目标肩袖区域均在相机视野内，工作台边框位置一致。','uncertainties':['单视角不能直接看到覆盖层下的材料；自遮挡判断是前后图及连接关系支持的视觉解释。','未跟踪隐藏袖口的同一材料点，无法确定被抓住的布层和显露发生的具体动作阶段。','袖口中点坐标为近似定位，候选仍需人工确认。'],'evidence_summary':note})
result={'schema_version':1,'task':'offline_corner_reveal_mining','source_index':'reports/offline_failure_mining_20261029/index.json','source_index_sha256':hashlib.sha256((ROOT/'reports/offline_failure_mining_20261029/index.json').read_bytes()).hexdigest(),'semantics':{'LS':'穿着者左袖口开口边缘中点','RS':'穿着者右袖口开口边缘中点','LH':'穿着者左侧下摆与侧缝交点','RH':'穿着者右侧下摆与侧缝交点'},'semantic_source':['cloth_agent/experiments/semantic_corner_oracle/core.py','cloth_agent/experiments/semantic_corner_oracle/review.py'],'coverage':{'source_runs_with_execution':93,'verified_pairs':532,'overview_pairs_screened':532,'overview_pages':27,'native_resolution_region_reviews':len(notes),'full_frame_candidate_reviews':2,'success_candidates':len(selected),'human_confirmed_successes':0,'not_shortlisted_meaning':'未入选不是已确认失败；缩略图初筛可能漏检，不能据此估计成功率。'},'method':'按时间顺序查看所有原始图像缩略对；视觉端点显露线索初筛；13 对原生分辨率局部复查；2 对全幅原图确认视野与语义连接。未使用面积阈值或旧策略理由筛选。','physical_actions_executed_this_analysis':0,'policy_changes':0,'success_candidates':selected,'reviewed_other_events':[r for r in audits if r['status']!='SUCCESS_CANDIDATE'],'limitations':['未建立全部 532 对的四端点真值标注。','两个候选均为 RS；不能据此推断 LS 或下摆的恢复规律。','这是事件证据整理，不是动作算法或因果机制验证。']}
(OUT/'candidates.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
(OUT/'screening_audit.json').write_text(json.dumps({'overview_pages_reviewed':list(range(1,28)),'detail_reviews':audits},ensure_ascii=False,indent=2))
def esc(x):return html.escape(str(x))
def im(path):
 p=ROOT/path;mime='image/png' if p.suffix=='.png' else 'image/jpeg';return 'data:'+mime+';base64,'+base64.b64encode(p.read_bytes()).decode()
parts=['''<!doctype html><html lang="zh"><meta charset="utf-8"><title>Corner Reveal 原始证据审核</title><style>body{font:16px/1.65 system-ui;margin:28px auto;max-width:1500px;padding:0 22px;background:#f4f6f8;color:#162533}section{background:white;padding:24px;margin:24px 0;border-radius:12px}img{max-width:100%;height:auto}.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}figure{margin:0}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccd4dc;padding:8px;text-align:left}.badge{background:#fff0b5;padding:5px}textarea{width:95%;height:70px}button,select{padding:8px}summary{cursor:pointer}@media(max-width:800px){.pair{grid-template-columns:1fr}}</style><h1>Corner Reveal：真实动作成功候选</h1><p>离线审核 · 2026-10-29 · 仅整理证据，无机器人执行、无策略修改。</p><section><h2>结果与证据范围</h2><p><b>保留 2 个较强视觉候选：E153、E353，均为 RS（穿着者右袖口）。人工确认数为 0。</b>两例动作前袖口区域收在衣身附近并被覆盖，动作后可辨认袖口开口边。判断仍需研究者审核，不能直接当作训练真值。</p><p>复用 93-run 索引，核验 532 对原图哈希；逐页初筛全部 532 对，13 对放大复查，2 个候选另查看全幅原图。未用面积增长筛选。未入选不代表已确认失败，不能计算成功率。</p><p>LS/RS 指穿着者左/右袖口开口边缘中点；LH/RH 指左右下摆与侧缝交点。本报告两例均为正面朝上，图像左侧对应 RS。普通褶尖不算端点；其他端点卷曲或消失不构成排除理由。</p><p>图片为未经绘制覆盖的原始 RGB，已嵌入本 HTML；点击图片可在新窗口放大。标注坐标单独列出，不覆盖原图。动作参数来自有 EXECUTED 证据的记录，距离是执行命令的位移参数，不是实测布料位移。</p><a href="candidates.json">结构化 JSON</a> · <a href="inventory.json">532 对索引</a> · <a href="screening_audit.json">复查记录</a></section>''']
for c in selected:
 a=c['actual_executed_action'];parts.append(f'<section id="{c["event_id"]}"><h2>{c["event_id"]} · RS 显露候选</h2><span class="badge">SUCCESS_CANDIDATE / PENDING HUMAN REVIEW</span><p>{esc(c["source"])}</p><div class="pair">')
 for k in ['before','after']:
  uri=im(c[k]['path']);parts.append(f'<figure><figcaption>{k.upper()} · 原始 1280×720</figcaption><a href="{uri}" target="_blank"><img src="{uri}"></a></figure>')
 parts.append(f'</div><p><b>操作前：</b>{esc(c["before"]["evidence"])} 检查区域 xyxy={c["before"]["inspection_roi_xyxy"]}，隐藏点不填猜测坐标。</p><p><b>操作后：</b>{esc(c["after"]["evidence"])} 近似中点 {c["after"]["point_px_approx"]}。</p><p>{esc(c["identity_basis"])} {esc(c["out_of_view_excluded"])}</p><table><tr><th>抓点 → 目标点（原图像素）</th><th>方向 Δx, Δy</th><th>命令距离</th><th>名义抬升</th><th>夹爪角度</th></tr><tr><td>{a["grasp_px"]} → {a["target_px"]}</td><td>{a["direction_image_delta_px"]}，{a["direction_image_degrees"]:.2f}°</td><td>{a["distance_mm"]:.2f} mm</td><td>{a["lift_height_mm"]} mm</td><td>{a["grasp_angle_deg"]}°</td></tr></table><p>方向角以图像向右为 0°、向下为正；不等于夹爪角度。动作类型 {a["motion_type"]}；执行完成 9 条命令。抓取层数、真实材料路径未知。</p><p><b>不确定性：</b>{esc("；".join(c["uncertainties"]))}</p><details><summary>来源、哈希与完整记录</summary><pre>{esc(json.dumps(c,ensure_ascii=False,indent=2))}</pre></details><p>人工审核：<select data-id="{c["event_id"]}"><option>PENDING</option><option>CONFIRMED_REVEAL</option><option>UNCERTAIN</option><option>REJECTED_ALREADY_VISIBLE</option><option>REJECTED_IDENTITY</option></select></p><textarea data-note="{c["event_id"]}" placeholder="请记录遮挡与袖口身份的判断依据"></textarea></section>')
parts.append('<button onclick="saveReview()">导出人工审核 JSON（不会写回历史数据）</button><section><h2>未作为成功纳入的复查线索</h2><p>这些记录用于展示不确定性和误报风险，不计为成功候选。放大图是固定裁剪，仅辅助检查，完整路径见 JSON。</p>')
for a in audits:
 if a['status']=='SUCCESS_CANDIDATE':continue
 parts.append(f'<details><summary>{a["event_id"]} · {a["status"]} — {esc(a["reason"])}</summary><img loading="lazy" src="detail/{a["event_id"]}.jpg"><p>{esc(a["source"])}</p></details>')
parts.append('</section><section><h2>全量初筛索引</h2><p>27 页按时间排列；每对左 Before、右 After。缩略图仅供导航，不能单凭它确认隐藏点。</p>')
for i in range(1,28):parts.append(f'<details><summary>第 {i} 页：E{(i-1)*20+1:03d}–E{min(i*20,532):03d}</summary><img loading="lazy" src="overview/page_{i:02d}.jpg"></details>')
parts.append('''</section><section><h2>结论边界</h2><p>已找到两个可供后续人工确认的真实端点显露候选。尚未证明任何可复用抓取规律；两例均为右袖口，不能外推到下摆或其他状态。单视角下遮挡层次仍有歧义，保留不确定标签比补齐成功数量更重要。本轮未自动开始 RSI。</p></section><script>function saveReview(){const reviews=[...document.querySelectorAll('select[data-id]')].map(e=>({event_id:e.dataset.id,status:e.value,note:document.querySelector('[data-note="'+e.dataset.id+'"]').value}));const b=new Blob([JSON.stringify({schema_version:1,reviewed_at:new Date().toISOString(),reviews},null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='corner_reveal_human_review.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}</script></html>''')
(OUT/'review.html').write_text(''.join(parts))
print(json.dumps(result['coverage'],ensure_ascii=False));print('HTML bytes', (OUT/'review.html').stat().st_size)
