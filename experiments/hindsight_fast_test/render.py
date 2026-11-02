"""Format saved results only; no model calls, selection or experiment mutations."""
import json, html
from pathlib import Path
R=Path(__file__).resolve().parents[2]/'reports/hindsight_fast_test_20261029'
def read(name,default=None):
    p=R/name
    return json.loads(p.read_text()) if p.exists() else default
def e(x):return html.escape(str(x))
def dump(x):return '<pre>'+e(json.dumps(x,ensure_ascii=False,indent=2))+'</pre>'
def pair(id):return '<div class="pair">'+''.join('<figure><a href="evidence/'+id+'/'+s+'.png"><img src="evidence/'+id+'/'+s+'.png"></a><figcaption>'+s.upper()+'</figcaption></figure>' for s in ['before','after'])+'</div>'
def render():
    summary=read('summary.json',{});hyp=read('skill_hypotheses.json',{});selected=read('selected_case.json',{});skill=read('skill_synthesis.json',{});vals=read('validation_results.json',[])
    if not summary:raise RuntimeError('No completed result to render')
    calls=[(p.parent.name,json.loads(p.read_text())) for p in sorted((R/'calls').glob('*/result.json'))]
    b=['<h1>Hindsight Skill Discovery · 端到端快速测试</h1><p class="badge">'+e(summary['status'])+' · 仅离线推理 · 物理验证 0 次</p>']
    b+=['<section><h2>本轮实际完成了什么</h2><table><tr><th>历史图像对</th><th>发现集</th><th>放大候选</th><th>深入案例</th><th>新图验证</th><th>模型请求</th></tr><tr>'+''.join('<td>'+e(x)+'</td>' for x in [summary.get('historical_pairs'),summary.get('discovery_pairs'),summary.get('enlarged_pairs'),1 if selected.get('case') else 0,len(vals),summary.get('model_calls')])+'</tr></table><p>原始索引 → 固定种子预留完整 run → Contact Sheet 自主浏览 → 最多10对放大 → 自动选1例 → 一轮机制与几何归纳 → 生成并冻结Python Skill → 新RGB推理。</p><p>研究结论：本页展示程序链路和实验性动作生成，不能证明真实显露成功或技能泛化。模型观察仍需独立物理证据支持。</p></section>']
    assessment=read('result_analysis.json',{})
    if assessment:
        b.append('<section><h2>本轮结论：链路完成，研究假设仍未证实</h2><p><b>'+e(assessment['conclusion'])+'</b></p><ul>'+''.join('<li>'+e(x)+'</li>' for x in assessment['limitations'])+'</ul><p>新图决策平均 '+str(round(assessment['validation_latency_s']['mean'],1))+' s，P95 '+str(round(assessment['validation_latency_s']['p95'],1))+' s（仅3次，描述性统计）。</p><p><a href="result_analysis.json">阶段结论与成本统计</a> · <a href="audit.json">隔离与预算审计</a> · <a href="code_contract_tests.json">Python接口测试</a></p></section>')
    if summary.get('unscreened_due_to_transport'):
        b.append('<section><h2>传输失败与覆盖范围</h2><p>计划发现集 '+e(summary['discovery_pairs'])+' 对；实际完成视觉初筛 '+e(summary['visually_screened_pairs'])+' 对；'+e(summary['unscreened_due_to_transport'])+' 对因API超时未完成审阅，不能记为没有显露。没有重试这些请求；沿用其余已返回候选继续同一轮流程。</p><p><a href="transport_recovery.json">恢复审计</a> · <a href="unscreened_pairs.json">未审阅清单</a></p></section>')
    if selected.get('case'):
        c=selected['case'];b+=['<section><h2>1. Agent 自动选择的现象</h2><p>'+e(c['id'])+' · '+e(selected.get('source'))+'</p>'+pair(c['id'])]
        source=R.parents[1]/selected['source']
        dispatch=json.loads((source/'dispatch_intent.json').read_text())
        raw=dispatch.get('plan',{})
        b.append('<h3>原始决策原文（未按结果改写）</h3>'+dump({k:raw[k] for k in ['short_reason','purpose','expected_effect','expected_stop_condition','grasp_px','target_px','grasp_angle_deg','is_smoothing'] if k in raw}))
        b+=['<p><b>为什么选它：</b>按冻结规则排序：意外收益证据 → 显露证据 → 研究价值 → 固定哈希；并非研究者指定。</p>']
        for k,t in [('intended_object','原来以为操作什么'),('intended_effect','原先预期'),('actual_command','实际命令'),('actual_material_identity','实际夹持身份'),('observed_local_change','可见变化'),('unplanned_benefit','意外收益状态'),('limitations','证据局限')]:b.append('<p><b>'+t+'：</b>'+e(c.get(k))+'</p>')
        b.append('</section>')
    if hyp:
        b.append('<section><h2>2. 目标重标记与竞争解释</h2><p><b>新的主动目标：</b>'+e(hyp.get('hindsight_goal'))+'</p><p><b>意图—结果差异：</b>'+e(hyp.get('intention_outcome_mismatch'))+'</p>')
        for h in hyp.get('hypotheses',[]):
            b.append('<article><h3>'+e(h['name'])+'</h3>')
            for k,t in [('physical_explanation','物理解释'),('observed_support','观察支持'),('unobserved_assumptions','尚未观察的假设'),('different_prediction','可区分的预测'),('falsifier','反证条件')]:b.append('<p><b>'+t+'：</b>'+e(h[k])+'</p>')
            b.append('</article>')
        b.append('</section><section><h2>3. 自主归纳的几何与动作规则</h2><h3>可观察几何</h3><ul>'+''.join('<li>'+e(x)+'</li>' for x in hyp.get('observable_geometry',[]))+'</ul><h3>动作生成规则</h3><ol>'+''.join('<li>'+e(x)+'</li>' for x in hyp.get('action_generation_rules',[]))+'</ol><p><b>适用条件：</b>'+e(hyp.get('applicability'))+'</p><p><b>不确定性：</b>'+e(hyp.get('uncertainty'))+'</p></section>')
    if skill:
        b.append('<section><h2>4. 生成的 Python Skill</h2><p>'+e(skill.get('algorithm_explanation'))+'</p><p>函数通过注入的 VISION_MODEL 对当前 RGB 做一次推理；若使用此接口，它是 VLM 驱动的技能，不代表已实现独立的确定性CV检测器。</p><p><a href="candidate_skill.py">Python 源码</a> · <a href="inference_prompt.txt">冻结推理Prompt</a> · <a href="skill_freeze.json">冻结哈希</a></p><pre>'+e(skill.get('python_code'))+'</pre><details><summary>完整冻结Prompt</summary><pre>'+e(skill.get('inference_prompt'))+'</pre></details><p><b>工程假设：</b>'+e(skill.get('engineering_assumptions'))+'</p></section>')
    if vals:
        b.append('<section><h2>5. 从未参与归纳的新 RGB</h2><p>每个测试来自预留完整 run；没有提供对应 After 或历史动作。AUTO 目标由当前图判断，没有人工指定目标。</p>')
        for v in vals:
            a=v['action'];b.append('<article><h3>'+e(v['id'])+' · '+e(a['decision'])+'</h3><p>'+e(v['before'])+'</p><a href="validation/'+v['id']+'/overlay.png"><img class="full" src="validation/'+v['id']+'/overlay.png"></a><p><b>Pa → Pb：</b>'+e(a['grasp_px'])+' → '+e(a['target_px'])+'；'+str(round(v['distance_px'],1))+' px；夹爪角 '+e(a['grasp_angle_deg'])+'°</p>')
            for k,t in [('target_corner','目标端点'),('target_state','目标状态'),('observable_geometry','当前结构依据'),('grasp_region','抓取区域'),('direction_reason','方向理由'),('expected_change','预期改变'),('falsifier','反证结果'),('uncertainty','不确定性')]:b.append('<p><b>'+t+'：</b>'+e(a[k])+'</p>')
            b.append('<p>耗时 '+str(round(v['latency_s'],1))+' s；像素范围检查 '+e(v['pixels_in_image'])+'；Host可执行性未测；物理成功未测。</p></article>')
        b.append('</section>')
    b.append('<section><h2>6. 资源、隔离与人工介入审计</h2><p>'+e(summary.get('isolation'))+'</p><p>未由人工指定候选、修改机制或提供测试图答案。编排者可见既往会话，这不是对编排者的完全盲测；独立模型请求不包含既往人工发现报告。历史动作本身可能受专家策略影响。</p><table><tr><th>请求</th><th>状态</th><th>秒</th><th>输入Token</th><th>输出Token</th></tr>')
    for name,c in calls:
        u=c.get('usage') or {};b.append('<tr>'+''.join('<td>'+e(x)+'</td>' for x in [name,c['status'],round(c.get('latency_s',0),1),u.get('input_tokens'),u.get('output_tokens')])+'</tr>')
    b.append('</table><p>从首次请求到结果生成：'+str(round(summary.get('wall_time_s',0)/60,1))+' 分钟；包含并发请求，单项延迟不能直接相加当作总耗时。失败请求未返回usage时，其Token用量和费用未知，不能按零计。</p></section>')
    b.append('<section><h2>7. 全部放大候选与可追溯文件</h2>')
    for c in read('detailed_effects.json',[]):b.append('<details><summary>'+e(c['id'])+' · '+e(c['reveal'])+' · 意外收益 '+e(c['unplanned_benefit'])+'</summary>'+pair(c['id'])+dump(c)+'</details>')
    for f in ['protocol.json','split.json','contact_sheets.json','screening_results.json','shortlist.json','detailed_effects.json','selected_case.json','skill_hypotheses.json','skill_synthesis.json','validation_results.json','summary.json']:b.append('<p><a href="'+f+'">'+f+'</a></p>')
    b.append('</section>')
    (R/'review.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>Hindsight Fast Test</title><style>body{font:18px/1.7 system-ui;max-width:1450px;margin:30px auto;padding:20px;background:#f4f6f8;color:#172434}section{background:white;padding:24px;margin:24px 0;border-radius:12px}article{border-top:1px solid #ccc;padding:20px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eef1f4;padding:18px;font-size:14px}.pair{display:flex;gap:12px}.pair figure{width:50%;margin:0}.pair img,.full{width:100%}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:10px}.badge{background:#fff0bc;padding:12px}summary{cursor:pointer}details{padding:12px}</style>'+''.join(b)+'</html>')
if __name__=='__main__':render()
