"""Render existing frozen results; no learning or model calls."""
from pathlib import Path
import difflib, html, json, hashlib
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/rsi_code_iteration_01_20261029'
def read(p):return json.loads(Path(p).read_text())
def esc(x):return html.escape(str(x))
def block(title,text):return '<h2>'+esc(title)+'</h2><pre>'+esc(text)+'</pre>'

def main():
    a=read(OUT/'stage1/answer.json');b=read(OUT/'stage2/answer.json');evaluation=read(OUT/'evaluation.json');selected=a['selected_replay']
    v0=(OUT/'policies/policy_v0.py').read_text();v1=(OUT/'policies/policy_v1.py').read_text()
    diff=''.join(difflib.unified_diff(v0.splitlines(True),v1.splitlines(True),fromfile='policy_v0.py',tofile='policy_v1.py'))
    (OUT/'policy_diff.patch').write_text(diff)
    metrics={}
    for version in ['v0','v1']:
        metrics[version]={}
        for split in ['seed','selected_replay','seen_input_unrevealed_outcome','unseen_state']:
            rows=[r for r in evaluation['predictions'] if r['version']==version and r['exposure']==split]
            errors=[r['absolute_error'] for r in rows if r['absolute_error'] is not None]
            metrics[version][split]={'n':len(rows),'numeric_predictions':len(errors),'mae_area_proxy':sum(errors)/len(errors) if errors else None}
        rows=[r for r in evaluation['predictions'] if r['version']==version and r['split']=='heldout']
        errors=[r['absolute_error'] for r in rows if r['absolute_error'] is not None]
        metrics[version]['heldout_runs']={'n':len(rows),'numeric_predictions':len(errors),'mae_area_proxy':sum(errors)/len(errors) if errors else None}
        rows=[r for r in evaluation['predictions'] if r['version']==version]
        metrics[version]['action_coverage']={'n':len(rows),'actions':sum(r['generated_action']['decision']=='ACTION' for r in rows),'abstentions':sum(r['generated_action']['decision']=='ABSTAIN' for r in rows)}
    summary={'one_iteration_complete':True,'physical_actions':0,'selected_replay':selected,'hypotheses':a['hypotheses'],'selection_reason':a['selection_reason'],'information_gain_reason':a['information_gain_reason'],'fixed_selected_feedback':read(OUT/'data'/selected/'feedback.json'),'interpretation':{k:v for k,v in b.items() if k!='policy_code'},'prediction_metrics':metrics,'equivariance':evaluation['equivariance'],'code_diff_nonempty':bool(diff),'policy_success_rate':None,'limits':evaluation['limitations']}
    findings='闭环完成，但没有发现已验证的可泛化算法。v0 在 8 个状态中输出 5 个动作，v1 全部弃权，动作覆盖率退化。两个真正未见状态均无数值预测，无法评估预测泛化。v1 变换后仍全部弃权，不能据此宣称动作几何一致性。P2 面积下降 14.73%，同时落在两种预先预测区间之外；只能反驳乐观预测，不能确认另一种物理机制。v0 的文字 H1 预测为增加，可执行预测器却为 -1.47%，必须分开记录。v1 对已揭示 P2 的预测为 -3.21%，误差仍达 11.53 个百分点。缩短动作、折扣权重均为未经验证的修订。'
    summary['primary_findings']=findings
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    checks={p:digest(p)==h for p,h in read(OUT/'evaluator_freeze.json').items()}
    assert all(checks.values())
    calls=[]
    for stage in [1,2]:
        folder=OUT/f'stage{stage}'
        events=[json.loads(line) for line in (folder/'events.jsonl').read_text().splitlines() if line.strip()]
        item_types=[e['item']['type'] for e in events if 'item' in e]
        calls.append({'stage':stage,'thread_ids':[e['thread_id'] for e in events if e['type']=='thread.started'],'completed':any(e['type']=='turn.completed' for e in events),'item_types':item_types,'status':read(folder/'status.json'),'invocation':read(folder/'invocation.json'),'prompt_sha256':digest(folder/'prompt.txt')})
    policy_checks={v:digest(OUT/f'policies/policy_{v}.py')==read(OUT/f'policies/{v}_manifest.json')['code_sha256'] for v in ['v0','v1']}
    selection=read(OUT/'selection_freeze.json')
    assert all(policy_checks.values()) and selection['stage1_response_sha256']==digest(OUT/'stage1/answer.json')
    images_ok=all(digest(OUT/'data'/r['id']/(s+'.png'))==r[s+'_sha256'] for r in read(OUT/'inventory.json') for s in ['before','after'])
    assert images_ok
    audit={'context':read(OUT/'context_audit.json'),'evaluator_hash_matches':checks,'policy_hash_matches':policy_checks,'images_match_inventory':images_ok,'selection_response_hash_matches':True,'native_calls':calls,'limitations':['Native invocation/events audit is not a server-side full prompt audit.','Stage 2 reports an extra dash image-read error; intended two images acknowledged. Preserve this anomaly; no learning retry was performed.','Static code screening is not a general security sandbox.']}
    (OUT/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    page=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>一次离线 RSI 代码迭代</title><style>body{font:17px system-ui;max-width:1250px;margin:32px auto;line-height:1.6;color:#203047}pre{white-space:pre-wrap;background:#f1f4f8;padding:18px;border-radius:8px;overflow-wrap:anywhere}.pair{display:flex;gap:15px}.pair img{width:49%;object-fit:contain}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccd5df;padding:10px}h1,h2{color:#152b47}</style><h1>一次离线 RSI：观察 → 假设 → 选择回放 → 固定反馈 → 修改代码</h1><p>仅一个迭代，0 次新物理操作。3 条种子记录、3 条候选中揭示1条结果、2个未见过的独立 run 状态。发现者未获得人工技能、旧 prompt 或旧规划理由。</p>']
    page.append(block('主要结果：策略退化，泛化未得到验证',findings))
    page.append(block('自主假设（反馈揭示前冻结）',json.dumps(a['hypotheses'],ensure_ascii=False,indent=2)))
    page.append(block('选择的实验：'+selected,a['visible_scope_check']+'\n'+a['selection_reason']+'\n'+a['information_gain_reason']+'\n预先预测：\n'+a['predictions']))
    page.append('<div class="pair"><img src="data/'+selected+'/before.png"><img src="data/'+selected+'/after.png"></div>')
    page.append(block('实际已执行动作',json.dumps(read(OUT/'data'/selected/'action.json'),ensure_ascii=False,indent=2)))
    page.append(block('冻结评价结果（面积是代理指标，不是真实平整度）',json.dumps(summary['fixed_selected_feedback'],ensure_ascii=False,indent=2)))
    for field,title in [('selected_feedback_interpretation','Agent 对反馈的解释'),('supported','得到支持'),('refuted','受到反驳'),('unknown','仍未知'),('code_changes','实际代码修改'),('next_experiment','下一次最有信息价值的实验'),('generalization_limits','泛化限制')]:page.append(block(title,b[field]))
    page.append(block('代码差异',diff))
    page.append(block('历史动作的面积变化预测误差：不是新策略成功率',json.dumps(metrics,ensure_ascii=False,indent=2)))
    page.append(block('几何变换一致性：不是物理泛化试验',json.dumps(evaluation['equivariance'],ensure_ascii=False,indent=2)))
    page.append('<h2>策略输出与真实历史反馈（各自分开）</h2><table><tr><th>状态</th><th>版本 / 数据划分</th><th>新生成动作（未执行）</th><th>对历史动作的面积变化预测 / 实测代理值</th></tr>')
    for r in evaluation['predictions']:
        page.append('<tr><td>'+esc(r['id'])+'</td><td>'+esc(r['version']+' / '+r['exposure'])+'</td><td>'+esc(json.dumps(r['generated_action'],ensure_ascii=False))+'</td><td>'+esc(str(r['predicted_recorded_action_area_delta'])+' / '+str(r['actual_recorded_action_area_delta']))+'</td></tr>')
    page.append('</table><p><a href="policies/policy_v0.py">policy_v0.py</a> · <a href="policies/policy_v1.py">policy_v1.py</a> · <a href="audit.json">隔离与冻结审计</a> · <a href="summary.json">完整结果 JSON</a></p></html>')
    (OUT/'review.html').write_text('\n'.join(page));print(json.dumps(metrics,ensure_ascii=False))
if __name__=='__main__':main()
