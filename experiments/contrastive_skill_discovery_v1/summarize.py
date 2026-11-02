from run import *
import statistics

def ratio_text(x):
 return f"{x['numerator']}/{x['denominator']}"+(f" ({100*x['rate']:.1f}%)" if x['rate'] is not None else ' (不可估计)')
def summarize():
 costs=[]
 for p in sorted(OUT.rglob('result.json')):
  r=read(p);costs.append({'request':str(p.parent.relative_to(OUT)),**r})
 total={k:sum((r.get('usage') or {}).get(k,0) or 0 for r in costs) for k in ['input_tokens','cached_input_tokens','output_tokens','reasoning_output_tokens']}
 save(OUT/'costs.json',{'requests':costs,'totals':total,'request_count':len(costs),'unmetered_early_audit_sessions':3,'native_transport_retry_usage_may_be_incomplete':True,'successful':sum(r['status']=='COMPLETE' for r in costs),'summed_wall_s':sum(r['wall_s'] for r in costs),'currency_cost':None,'currency_note':'No verified billing rates; tokens reported, not fabricated currency cost.','audit_overhead':'debug prompt-input local rendering not included in inference wall_s; one READY request separately identifiable','latency_note':'Observed completed model turn, CLI shutdown excluded. Includes native client/network; exact server compute time unavailable.'})
 replay=[]
 for state in ['E29','E30']:
  for g in ['Base','A','B','C']:
   for rep in [1,2]:
    folder=OUT/f'planner_replay/{state}_{g}_r{rep}'
    if not (folder/'parsed.json').exists():continue
    action=read(folder/'parsed.json');base=read(OUT/f'planner_replay/{state}_Base_r{rep}/parsed.json');other=read(OUT/f'planner_replay/{state}_{g}_r{3-rep}/parsed.json')
    def angle(a,b):
     va=[a['Pb'][i]-a['Pa'][i] for i in range(2)];vb=[b['Pb'][i]-b['Pa'][i] for i in range(2)];den=math.hypot(*va)*math.hypot(*vb)
     return math.degrees(math.acos(max(-1,min(1,sum(x*y for x,y in zip(va,vb))/den)))) if den else None
    r={'state':state,'condition':g,'repeat':rep,'action':action,'geometry':read(folder/'geometry.json'),'usage':read(folder/'result.json')['usage'],'wall_s':read(folder/'result.json')['wall_s'],'vs_base':{'Pa_distance_px':math.dist(action['Pa'],base['Pa']),'Pb_distance_px':math.dist(action['Pb'],base['Pb']),'direction_angle_deg':angle(action,base)},'within_condition_other_repeat':{'Pa_distance_px':math.dist(action['Pa'],other['Pa']),'Pb_distance_px':math.dist(action['Pb'],other['Pb']),'direction_angle_deg':angle(action,other)},'overlay':str((folder/'action_overlay.jpg').relative_to(OUT))};replay.append(r)
 stats={}
 for g in ['Base','A','B','C']:
  rows=[r for r in replay if r['condition']==g]
  if not rows:continue
  stats[g]={'n':len(rows),'mean_wall_s':statistics.mean(r['wall_s'] for r in rows),'median_wall_s':statistics.median(r['wall_s'] for r in rows),'mean_input_tokens':statistics.mean((r['usage'] or {}).get('input_tokens',0) for r in rows),'mean_output_tokens':statistics.mean((r['usage'] or {}).get('output_tokens',0) for r in rows),'mean_pa_change_vs_base_px':statistics.mean(r['vs_base']['Pa_distance_px'] for r in rows),'mean_within_condition_pa_variation_px':statistics.mean(r['within_condition_other_repeat']['Pa_distance_px'] for r in rows),'mean_direction_change_vs_base_deg':statistics.mean(r['vs_base']['direction_angle_deg'] for r in rows if r['vs_base']['direction_angle_deg'] is not None),'mean_within_direction_variation_deg':statistics.mean(r['within_condition_other_repeat']['direction_angle_deg'] for r in rows if r['within_condition_other_repeat']['direction_angle_deg'] is not None),'trigger_yes':sum(r['action']['skill_trigger']=='YES' for r in rows)}
 save(OUT/'planner_replay/summary.json',{'cases':replay,'statistics':stats,'pairing_note':'replicate-index comparisons are descriptive, not shared random seeds or causal paired trials','physical_execution':False,'geometry_limits':'2D finite coordinates, bounds, nonzero displacement and reused heuristic cloth mask; not IK/collision/depth/graspability certification.'})
 lines=['# Contrastive Skill Discovery v1','', '本实验离线比较模型先验、历史动作与真实操作反馈产生的 Skill。没有执行机器人动作，不能推断真实成功率提升。', '', '## 实际执行与隔离', '',f"共保存 {len(costs)} 条 CLI 模型请求记录（含连接检查及失败重试），成功 {sum(r['status']=='COMPLETE' for r in costs)} 次。统一 gpt-6-astra/high，原生 OpenAI 通道，无 api profile；每次独立 ephemeral 会话，工具调用即拒收。",'', '重要偏差：前期编排会话误显示了原 split 文件内嵌的 Test 标签。未向独立 Teacher/评价器传递这些内容，但不能声称整个编排过程完全盲测。随后冻结了通用评分、样本顺序和预算，不按结果挑样本。旧上下文审计请求已隔离。Cloth Skill 已移出技能库并保留哈希备份；正式请求逐次保存上下文审计、提示与附件。', '', '原始划分 Train 20 / Development 8 / Test 8 / Reference 4 保持不变；Reference 不提供给模型。Teacher A 无经验；B 仅 Train Before+动作；C 增加 Train After 与四维既有标签及不确定性。不使用旧结论或专家 Prompt。', '', '## 冻结 Skill', '']
 for g in 'ABC':
  initial=read(OUT/'skills'/f'{g}_initial.json');final=read(OUT/'skills'/f'{g}_final.json');core=''.join(final[k] for k in ['skill','trigger','grasp_selection','direction_selection']);cjk=sum('\u4e00'<=c<='\u9fff' for c in core)
  lines += [f'### {g}', '',f'最终四字段：{len(core)} 字符，其中汉字 {cjk}。', '', '```json', json.dumps(final,ensure_ascii=False,indent=2),'```','',f"一次修订决策：{read(OUT/f'teacher_{g}/revision/parsed.json')['decision']}；详见对应 version.diff 及原始修订输出。",'']
 for split in ['development','test']:
  ev=read(OUT/f'{split}_evaluation.json');lines += [f'## {split} 历史操作预测','', '主要标签为既有 unfolding_gain；NO_CLEAR_EVIDENCE 表示无明确收益证据，并非确定物理失败。UNKNOWN 不进入二分类真值分母；已知真值上的预测 UNKNOWN 作为弃权，另行报告。','', '|组|正向精确率|正向召回|无明确收益召回|适用率|预测UNKNOWN|真值UNKNOWN|','|---|---|---|---|---|---|---|']
  for g in 'ABC':
   m=ev['arms'][g]['metrics'];lines.append('|'+g+'|'+'|'.join(ratio_text(m[k]) for k in ['positive_precision','positive_recall','low_benefit_recall','applicability','prediction_unknown','truth_unknown'])+'|')
  lines += ['','|组|正向触发覆盖|低收益触发覆盖|正向动作符合覆盖|低收益动作符合覆盖|','|---|---|---|---|---|']
  for g in 'ABC':
   m=ev['arms'][g]['metrics'];lines.append('|'+g+'|'+'|'.join(ratio_text(m[k]) for k in ['positive_trigger_coverage','low_benefit_trigger_coverage','positive_conforms_coverage','low_benefit_conforms_coverage'])+'|')
  lines += ['', '|组|Trigger UNKNOWN|动作符合判断 UNKNOWN|', '|---|---|---|']
  for g in 'ABC':
   m=ev['arms'][g]['metrics'];lines.append('|'+g+'|'+ratio_text(m['trigger_unknown'])+'|'+ratio_text(m['conforms_unknown'])+'|')
  for g in 'ABC':
   lines += ['',f'### {g} 逐例原始结果','','|ID|Trigger|符合原则|预测|既有真值|依据与不确定性|','|---|---|---|---|---|---|']
   for r in ev['arms'][g]['cases']:lines.append('|'+ '|'.join(str(r[k]).replace('|','/').replace('\n',' ') for k in ['id','trigger','conforms','prediction','truth'])+'|'+(r['reason']+'；'+r['uncertainty']).replace('|','/').replace('\n',' ')+'|')
  lines+=['','每组只有8例，包含同运行组相关样本，不宣称统计显著性；关联不等于物理因果。四维结果及完整混淆表见 JSON。','']
 lines += ['## One-shot Planner Replay','','每个状态/条件/重复使用一个独立模型请求，直接附当前 RGB，无在线检索或 Research Agent。Planner只注入最终Skill的四个操作字段（原文白名单提取，无改写）；完整七字段用于历史评价并保留审计。历史评价与部署的文本范围不同，不能将前者的全部预测价值归因于后者的短规则。两个 Test Before 状态不参与 Skill 学习；它们用于历史盲评，因此并非所有实验环节均未见过的新状态。重复编号仅方便描述对照，不代表共享随机种子。','','|条件|平均耗时秒|平均输入Token|平均输出Token|相对Base抓点差px|组内重复抓点差px|相对Base方向差度|组内方向差度|','|---|---|---|---|---|---|---|---|']
 for g,s in stats.items():lines.append('|'+g+'|'+'|'.join(f'{s[k]:.1f}' for k in ['mean_wall_s','mean_input_tokens','mean_output_tokens','mean_pa_change_vs_base_px','mean_within_condition_pa_variation_px','mean_direction_change_vs_base_deg','mean_within_direction_variation_deg'])+'|')
 lines += ['','|状态|条件|重复|Pa|Pb|触发|二维检查|Pa在掩码内|动作图|','|---|---|---|---|---|---|---|---|---|']
 for r in replay:lines.append(f"|{r['state']}|{r['condition']}|{r['repeat']}|{r['action']['Pa']}|{r['action']['Pb']}|{r['action']['skill_trigger']}|{r['geometry']['finite_2d_in_bounds'] and r['geometry']['nonzero_pixel_displacement']}|{r['geometry'].get('Pa_in_heuristic_mask','UNKNOWN')}|[查看]({r['overlay']})|")
 lines += ['','静态检查覆盖有限二维坐标、图像范围、非零位移和复用的颜色衣物掩码；掩码只能检查二维布料区域，不能确认可抓布层、夹持成功、IK、碰撞或深度，不代表动作可安全执行或物理成功。', '', '![E29动作对照](planner_replay/E29_comparison.jpg)', '', '![E30动作对照](planner_replay/E30_comparison.jpg)', '', '## 成本与可复现性','',f"累计 Token：{json.dumps(total,ensure_ascii=False)}。详细逐次输入/输出、延迟和缓存用量见 costs.json；没有可核验账单单价，因此不估造货币成本。早期另有3个隔离审计会话未提供Token用量；网络重连可能存在未返回用量，累计Token仅为可见记录之和。",'', 'Replay 仅每组4次，时延受模型随机性、输出长度、缓存、网络影响。Token 增量可以测量，耗时差不能由此归因为 Skill 的必然代价。', '', '## 七个研究问题', '', '待基于已冻结的实际输出补充逐项解释。']
 (OUT/'report.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':summarize()
