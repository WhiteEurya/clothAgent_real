"""Render existing artifacts only; never calls a model or modifies rankings."""
from pathlib import Path
import json, html, shutil, statistics

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/hindsight_corner_continuation_20261030'
def read(name, default=None):
    p=OUT/name
    return json.loads(p.read_text()) if p.exists() else default
def e(x): return html.escape(str(x))
def dump(x): return '<pre>'+e(json.dumps(x,ensure_ascii=False,indent=2))+'</pre>'
def image(path,caption):
    return '<figure><a href="'+e(path)+'"><img loading="lazy" src="'+e(path)+'"></a><figcaption>'+e(caption)+'</figcaption></figure>'
def fields(x, keys):
    return ''.join('<p><b>'+e(label)+'：</b>'+e(x.get(key,''))+'</p>' for key,label in keys)
def main():
    s=read('summary.json'); assert s
    manifest=read('input_manifest.json'); reviews=read('reviews.json',[])
    ranking=read('ranking.json',{}); selected=read('selected_case.json',{})
    intent=read('intention_analysis.json',{}); hyp=read('skill_hypotheses.json',{})
    spec=read('skill_synthesis.json',{}); tests=read('validation_results.json',[])
    calls=[(p.parent.name,json.loads(p.read_text())) for p in sorted((OUT/'calls').glob('*/result.json'))]
    usage={}
    for _,c in calls:
        for k,v in (c.get('usage') or {}).items():
            if isinstance(v,(int,float)): usage[k]=usage.get(k,0)+v
    durations=[c['latency_s'] for _,c in calls if 'latency_s' in c]
    costs={'requests':len(calls),'token_usage':usage,'request_seconds_sum':sum(durations),'wall_seconds':s['wall_time_s'],'mean_request_s':statistics.mean(durations) if durations else None,'median_request_s':statistics.median(durations) if durations else None,'billing_cost':'NOT_AVAILABLE; no billing rate inferred'}
    (OUT/'cost_summary.json').write_text(json.dumps(costs,ensure_ascii=False,indent=2))
    counts='；'.join(k+' '+str(v) for k,v in s['review_status_counts'].items())
    body=['<h1>从17个候选到语义端点恢复假设</h1>',
          '<p class="notice">已知页面诊断的续实验。全程离线，机器人动作数为0。CONFIRMED仅表示模型认为图像证据较强，不代表独立材料真值。新图动作没有执行，不能据此计算技能成功率。</p>',
          '<p><b>完整图审核：</b>'+e(counts)+'。符合预先冻结门槛的候选 '+str(s['eligible'])+' 个。</p>',
          '<p><b>流程结果：</b>'+e(s['status'])+'；模型调用 '+str(s['model_calls'])+' 次；墙钟 '+str(round(s['wall_time_s']/60,1))+' 分钟。</p>']
    if selected:
        body+=['<p><b>自动选中：</b>'+e(s['selected'])+'，锁定目标 '+e(s['goal'])+'。先按端点显露证据排序，之后才查看原始意图。</p>']
        candidates=ranking.get('candidates',[])
        score_keys=['before_score','identity_score','transition_score','semantic_side_score']
        ties=[x['id'] for x in candidates if [x[k] for k in score_keys]==[selected['review'][k] for k in score_keys] and x['status']==selected['review']['status']]
        if len(ties)>1:body+=['<p>同等级同评分候选为 '+e('、'.join(ties))+'；由固定种子哈希破同分，不能声称选中案例的证据明显优于其余并列案例。</p>']
    if tests:
        actions=sum(t['action']['decision']=='ACTION' for t in tests)
        body+=['<p><b>新图结果：</b>'+str(len(tests))+' 张，'+str(actions)+' 个候选动作，'+str(len(tests)-actions)+' 次弃权。'+('本批没有得到新图牵引动作，尚未展示动作生成的迁移表现。' if not actions else '产生像素动作不等于可安全执行，也不等于能显露目标端点。')+'</p>']
        if not actions:
            body+=['<p class="notice"><b>本轮结论：</b>Agent完成了证据审核、假设提出、候选函数生成和新图调用，但未在这批新状态上生成Corner Reveal动作。弃权不是物理失败，也不能作为技能迁移成功。目标身份判定、适用条件保守程度和测试状态覆盖不足的影响，尚不能由本轮区分。</p>']
    body+=['<nav><a href="#audit">17例审核</a> · <a href="#selection">选择与原始效果</a> · <a href="#hypothesis">机制假设</a> · <a href="#skill">技能</a> · <a href="#tests">新图推理</a> · <a href="#integrity">隔离与成本</a></nav>', '<h2 id="audit">1. 全部17对审核</h2><p>点击图片查看完整分辨率。目标左右是衣服语义左右，不能直接当作画面左右。</p>']
    byid={x['id']:x for x in reviews}
    body+=['<table><tr><th>案例</th><th>审核状态</th><th>目标</th><th>遮挡/身份/转换/侧别评分</th><th>审核依据</th></tr>']
    for r in manifest['mapping']:
        x=byid.get(r['id'],{})
        score='/'.join(str(x.get(k,'?')) for k in ['before_score','identity_score','transition_score','semantic_side_score'])
        body+=['<tr><td><a href="#'+r['id']+'">'+r['id']+'</a></td>'+''.join('<td>'+e(v)+'</td>' for v in [x.get('status','FAILED'),x.get('goal'),score,x.get('review_reason','')])+'</tr>']
    body+=['</table>']
    for r in manifest['mapping']:
        x=byid.get(r['id'],{})
        body+=['<details id="'+r['id']+'"><summary>'+r['id']+' · '+e(x.get('status'))+' · '+e(x.get('goal'))+'</summary><div class="pair">'+image('evidence/'+r['id']+'/before.png','Before')+image('evidence/'+r['id']+'/after.png','After')+'</div>',fields(x,[('before_evidence','Before遮挡线索'),('after_evidence','After端点线索'),('correspondence','前后对应'),('counterevidence','反证'),('uncertainty','不确定性')]),'<details><summary>来源与完整结构化记录</summary>'+e(r['source'])+dump(x)+'</details></details>']
    body+=['<h2 id="selection">2. 自动选择与原始意图—真实结果</h2><p>冻结排序：满足四角恢复门槛 → 强视觉确认优先 → 四项证据的最低分 → 总分 → 固定种子哈希破同分。意外收益不参与排序。</p>']
    body+=['<p>符合门槛的顺序：'+e(' → '.join(x['id'] for x in ranking.get('candidates',[])))+'</p>']
    if selected:
        c=s['selected']; body+=['<h3>'+e(c)+' · '+e(s['goal'])+'</h3><div class="pair">'+image('evidence/'+c+'/before.png','选中案例 Before')+image('evidence/'+c+'/after.png','选中案例 After')+'</div>']
        body+=[fields(intent,[('intended_manipulation','Intended Manipulation'),('actual_executed_command','实际指令'),('actual_contact_and_material','Actual Manipulation'),('observed_effect','Observed Effect'),('expected_changes_supported','符合原先预期'),('expected_changes_not_established','未证实预期'),('unplanned_benefit','意外收益证据等级'),('unplanned_benefit_evidence','意外收益依据'),('intention_outcome_mismatch','意图与结果差异'),('uncertainty','不确定性')])]
        frames=read('process_frame_manifest.json',[])
        if frames:
            body+=['<p>执行过程按序列25%、50%、75%位置采样；没有精确阶段标注，腕载视角变化不等于布料运动。</p><div class="triple">']
            for i,f in enumerate(frames):
                dest=OUT/'process_frames'/f'{i+1}.jpg';dest.parent.mkdir(exist_ok=True);shutil.copyfile(f['source'],dest)
                body+=[image(str(dest.relative_to(OUT)),f'过程序列样本 {i+1}')]
            body+=['</div>']
        body+=['<details><summary>原始意图和实际执行参数</summary>'+dump(read('selected_raw_evidence.json'))+'</details>']
    body+=['<h2 id="hypothesis">3. 目标重标记与竞争性解释</h2>']
    if hyp:
        body+=[fields(hyp,[('goal','固定目标'),('goal_description','目标定义'),('success_criterion','未来物理成功判据'),('goal_relevance','与角点恢复的关系')])]
        for h in hyp.get('hypotheses',[]):
            body+=['<article><h3>'+e(h['name'])+'</h3>'+fields(h,[('physical_explanation','物理解释'),('observed_support','观测支持'),('unobserved_assumptions','未观测假设'),('different_prediction','区别预测'),('falsifier','反证条件')])+'</article>']
        body+=['<h3>自主归纳的动作生成原则</h3><ol>'+''.join('<li>'+e(x)+'</li>' for x in hyp.get('action_generation_rules',[]))+'</ol>'+fields(hyp,[('applicability','适用条件'),('uncertainty','局限')])]
    else: body+=['<p>未触发：没有足够的合格候选，或前序调用未完成。没有替换恢复目标。</p>']
    body+=['<h2 id="skill">4. 候选Skill与冻结记录</h2>']
    if spec:
        body+=[fields(spec,[('status','状态'),('algorithm_explanation','实现方式'),('limitations','限制')]),'<details open><summary>Python代码</summary><pre>'+e(spec['python_code'])+'</pre></details><details><summary>冻结的推理Prompt</summary><pre>'+e(spec['inference_prompt'])+'</pre></details>',dump(read('skill_freeze.json'))]
    else: body+=['<p>未生成候选Skill。</p>']
    body+=['<h2 id="tests">5. 新RGB离线推理</h2><p>测试run在审核前预留，图像不参与归纳。输入只包含新Before RGB和冻结技能；没有提供After、原始动作。只检查流程、固定目标和像素合同，没有验证深度、可达性或真实效果。</p>']
    for t in tests:
        a=t['action'];body+=['<article><h3>'+e(t['id'])+' · '+e(a['decision'])+' · '+e(a['target_state'])+'</h3>'+image('validation/'+t['id']+'/overlay.png','黄色箭头为候选Pa→Pb；ABSTAIN图不画动作'),fields(a,[('goal','固定目标'),('target_identity_basis','目标身份依据'),('target_occlusion_evidence','遮挡依据'),('grasp_region','选择的可见布料'),('grasp_px','Pa'),('target_px','Pb'),('direction_reason','方向理由'),('expected_target_change','预期目标变化'),('falsifier','反证'),('abstain_reason','弃权理由'),('uncertainty','不确定性')]),'<p>耗时 '+str(round(t['latency_s'],1))+' 秒；目标合同违规：'+e(t['goal_contract_violations'])+'</p></article>']
    body+=['<h2 id="integrity">6. 信息隔离、成本与结论边界</h2><p>每次视觉分析通过codex -p api创建独立临时会话；发现模型未接收旧筛选结论、人工规则、既有技能报告或历史成功案例编号。审核使用C01–C17匿名ID。主协调会话保留历史聊天上下文，因此不能宣称整个研究过程完全无先验；排序规则和样本已事先冻结，未按结果改排名。本轮候选来自指定页面和已有高召回队列，存在选择偏差。</p>']
    notes=read('interpretation_notes.json',{})
    if notes:
        body+=['<h3>冻结后审计发现的问题</h3><p>以下问题未反馈给发现或测试模型，没有用于改变排序或技能。</p>']
        for n in notes.get('findings',[]):body+=['<p><b>'+e(n['topic'])+'：</b>'+e(n['finding'])+'</p>']
    body+=[dump(costs),'<p>本轮能验证程序链路、模型给出的证据、假设与行为生成；不能证明隐藏材料身份、物理机制、跨状态成功率或真实任务改善。弃权可能意味着指定目标已可见或证据不足，不能自动视为技能失败；生成动作也不代表成功。</p><table><tr><th>调用</th><th>状态</th><th>秒</th><th>Token</th></tr>']
    for n,c in calls:body+=['<tr>'+''.join('<td>'+e(v)+'</td>' for v in [n,c['status'],round(c.get('latency_s',0),1),c.get('usage')])+'</tr>']
    body+=['</table><p>'+ ' · '.join('<a href="'+x+'">'+e(x)+'</a>' for x in ['protocol.json','input_manifest.json','reviews.json','ranking.json','summary.json','environment.json','cost_summary.json','artifact_audit.json','candidate_skill.py','validation_results.json'] if (OUT/x).exists())+'</p>']
    (OUT/'review.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>Corner Reveal · 17例续实验</title><style>body{font:17px/1.65 system-ui;max-width:1500px;margin:32px auto;padding:20px;color:#202b36}h1,h2{color:#143e67}h2{margin-top:50px}nav,.notice{padding:18px;background:#eaf1f7}table{border-collapse:collapse;width:100%}td,th{border:1px solid #cbd3dc;padding:9px;vertical-align:top}pre{white-space:pre-wrap;overflow-wrap:anywhere;padding:16px;background:#f1f4f7}details,article{border:1px solid #d5dce3;border-radius:6px;margin:16px 0;padding:16px}summary{cursor:pointer;font-weight:bold}.pair,.triple{display:flex;gap:12px}.pair figure{width:50%}.triple figure{width:33%}figure{margin:12px 0}img{width:100%}figcaption{color:#586678}a{color:#116b9c}</style>'+''.join(body)+'</html>')
    print(OUT/'review.html')
if __name__=='__main__':main()
