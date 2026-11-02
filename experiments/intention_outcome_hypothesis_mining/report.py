"""Mechanical aggregation and browseable evidence index; no synthesis model."""
import collections,html,json,time
from pathlib import Path
try:
    from .storage import read,save
except ImportError:
    from storage import read,save

def load_state(root,rid):
    folder=root/'calls'/rid
    for name in ['result.json','state.json']:
        if (folder/name).exists():return read(folder/name)
    attempts=sorted(folder.glob('attempt_*/result.json'))
    if attempts:
        value=read(attempts[-1]);return value if value['status']=='COMPLETE' else {**value,'status':'PENDING_RETRY'}
    return {'status':'PENDING'}

def aggregate(root,records,run_status):
    observations=[];bank=[];pending=[];states=collections.Counter();types=collections.Counter();evidence=collections.Counter()
    no_hyp=[];indeterminate=[];latencies=[]
    for r in records:
        rid=r['experience_id'];state=load_state(root,rid);states[state['status']]+=1
        entry={'experience_id':rid,'status':state['status'],'source':r['source'],'run':r['run'],
               'evidence_directory':f'inputs/{rid}','observation':None}
        if state['status']=='COMPLETE':
            o=state['output'];entry['observation']=o;entry['latency_s']=state.get('latency_s');latencies.append(state.get('latency_s',0))
            types.update(m['type'] for m in o['mismatches'])
            if not o['hypotheses']:no_hyp.append(rid)
            if o['observed_effect']['effect_determinacy']!='CLEAR':indeterminate.append(rid)
            for h in o['hypotheses']:
                bank.append({**h,'evidence':{'before':f'inputs/{rid}/before.png','after':f'inputs/{rid}/after.png',
                              'original_record':f'inputs/{rid}/payload.json','full_observation':f'cases/{rid}.html'},
                             'physical_mechanism_validated':False})
                evidence[h['evidence_level']]+=1
        else:
            entry['error']=state.get('error');pending.append({'experience_id':rid,'status':state['status'],'error':state.get('error'),'source':r['source']})
        observations.append(entry)
    attempts=[]
    for p in (root/'calls').glob('*/attempt_*/result.json'):
        x=read(p);attempts.append(x)
    usage=collections.Counter()
    for x in attempts:
        for k,v in (x.get('usage') or {}).items():
            if isinstance(v,(int,float)):usage[k]+=v
    done=states['COMPLETE'];total=len(records)
    protocol=read(root/'protocol.json')
    summary={'run_status':run_status,'updated_at':time.time(),'valid_experiences':total,'processed_experiences':done,
             'coverage':done/total if total else 0,'states':dict(states),'hypothesis_count':len(bank),
             'mismatch_items_excluding_expected_observed':sum(v for k,v in types.items() if k!='EXPECTED_OBSERVED'),
             'comparison_types':dict(types),'hypothesis_evidence_levels':dict(evidence),
             'corner_reveal_hypotheses':sum('CORNER_REVEAL' in h['categories'] for h in bank),
             'unexpected_occlusion_hypotheses':sum('UNEXPECTED_OCCLUSION' in h['categories'] for h in bank),
             'completed_without_hypotheses':len(no_hyp),'effect_not_fully_determined':len(indeterminate),
             'unprocessed_experiences':len(pending),'attempts_finished':len(attempts),
             'attempts_failed_or_interrupted':sum(x['status']!='COMPLETE' for x in attempts),
             'usage_reported':dict(usage),'usage_scope':'Successful or otherwise reported CLI usage only; failed calls may have unreported billed tokens. Reasoning tokens are a subset, not additional.',
             'mean_successful_request_s':sum(latencies)/len(latencies) if latencies else None,
             'started_at':protocol.get('started_at'),'physical_actions':0,
             'limits':'Model observations/hypotheses are not independent truth or validated mechanisms. No cross-case merging, strategy generation, Skill or RSI.'}
    save(root/'hypothesis_bank.json',bank);save(root/'experience_observations.json',observations)
    save(root/'unprocessed.json',pending);save(root/'summary.json',summary)
    return summary,observations,bank

STYLE='''body{font:16px/1.6 system-ui;max-width:1400px;margin:24px auto;padding:16px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eff3f7;padding:14px}img{max-width:100%}.pair{display:flex;gap:10px}.pair a{width:50%}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccd3db;padding:8px;vertical-align:top}input,select{padding:8px;margin:8px;max-width:90%}.notice{background:#fff0ca;padding:14px}progress{width:60%;height:24px}.hyp{border:1px solid #ccd3db;padding:15px;margin:15px 0}summary{cursor:pointer}'''
def esc(x):return html.escape(str(x))
def dump(x):return '<pre>'+esc(json.dumps(x,ensure_ascii=False,indent=2))+'</pre>'
def write_page(path,title,body):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.html.tmp');temporary.write_text('<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>'+esc(title)+'</title><style>'+STYLE+'</style></head><body>'+body+'</body></html>');temporary.replace(path)

def render(root,records,run_status,detail_ids=None):
    summary,observations,bank=aggregate(root,records,run_status);byid={r['experience_id']:r for r in records}
    for entry in observations:
        rid=entry['experience_id'];dest=root/'cases'/f'{rid}.html'
        if detail_ids is not None and rid not in detail_ids and dest.exists():continue
        record=byid[rid];payload=read(record['payload']);o=entry['observation']
        body='<p><a href="../review.html">返回索引</a></p><h1>'+rid+'</h1><p>'+esc(record['source'])+'</p>'
        body+='<div class="pair">'+''.join(f'<a href="../inputs/{rid}/{k}.png"><img loading="lazy" src="../inputs/{rid}/{k}.png" alt="{k}"></a>' for k in ['before','after'])+'</div>'
        body+='<h2>原始输入：意图、动作与执行证据</h2>'+dump(payload)
        if not o:body+='<p class="notice">未完成分析，不能视为没有发现。</p>'+dump(entry)
        else:
            body+='<h2>完整意图—结果比较</h2>'+dump({k:v for k,v in o.items() if k!='hypotheses'})
            for h in o['hypotheses']:body+='<section class="hyp" id="'+esc(h['hypothesis_id'])+'"><h2>'+esc(h['hypothesis_id'])+'</h2>'+dump(h)+'</section>'
            body+='<p>假设机制未通过物理验证。<a href="../calls/'+rid+'/result.json">模型原始结果与调用耗时</a></p>'
        write_page(dest,rid,body)
    body='<h1>Intention–Outcome Hypothesis Bank</h1><p class="notice">只逐条建立假设库；不合并、不提炼Skill、不生成策略、不执行机器人。模型证据不是独立物理真值。未处理项单独保留。</p>'
    body+=f'<p><b>{esc(run_status)}</b> · 已分析 {summary["processed_experiences"]}/{summary["valid_experiences"]} · 假设 {summary["hypothesis_count"]} · 未处理 {summary["unprocessed_experiences"]}</p><progress value="{summary["processed_experiences"]}" max="{max(1,len(records))}"></progress>'
    body+='<p>此页运行期间定期更新；浏览器刷新可查看最新结果。</p><details><summary>数量、覆盖率与成本明细</summary>'+dump(summary)+'</details>'
    body+='<p>'+ ' · '.join(f'<a href="{f}">{f}</a>' for f in ['hypothesis_bank.json','experience_observations.json','unprocessed.json','data_audit.json','protocol.json','summary.json','run.log'])+'</p>'
    body+='<h2>全部经验索引</h2><input id="searchE" placeholder="搜索编号、run或状态" oninput="filterRows(\'experiences\',this.value)"><select onchange="filterRows(\'experiences\',this.value)"><option value="">全部</option><option>COMPLETE</option><option>PENDING</option><option>FAILED</option><option>NO_HYPOTHESIS</option><option>EFFECT_UNCERTAIN</option></select><table id="experiences"><tr><th>编号</th><th>状态</th><th>run / 原始记录</th><th>假设数</th></tr>'
    for e in observations:
        o=e['observation'];count=len(o['hypotheses']) if o else None
        tags=e['status']+(' NO_HYPOTHESIS' if count==0 else '')+(' EFFECT_UNCERTAIN' if o and o['observed_effect']['effect_determinacy']!='CLEAR' else '')
        search=esc(e['experience_id']+' '+e['source']+' '+tags)
        body+=f'<tr data-search="{search}"><td><a href="cases/{e["experience_id"]}.html">{e["experience_id"]}</a></td><td>{esc(tags)}</td><td>{esc(e["source"])}</td><td>{count if count is not None else "未处理"}</td></tr>'
    body+='</table><h2>全部假设索引（重复假设完整保留）</h2><input placeholder="搜索假设或证据" oninput="filterRows(\'hypotheses\',this.value)"><select onchange="filterRows(\'hypotheses\',this.value)"><option value="">全部</option><option>CORNER_REVEAL</option><option>UNEXPECTED_OCCLUSION</option><option>PROVISIONAL</option><option>UNKNOWN</option><option>CONFIRMED</option></select><table id="hypotheses"><tr><th>假设</th><th>分类 / 证据</th><th>机制与预测</th></tr>'
    for h in bank:
        searchable=' '.join(str(v) for k,v in h.items() if k!='evidence')
        body+=f'<tr data-search="{esc(searchable)}"><td><a href="cases/{h["source_experience_id"]}.html#{h["hypothesis_id"]}">{esc(h["hypothesis_id"])}</a></td><td>{esc(h["categories"])}<br>{esc(h["related_corner_goals"])}<br>{esc(h["evidence_level"])}</td><td>{esc(h["hypothesized_mechanism"])}<br><b>预测：</b>{esc(h["predicted_effect"])}</td></tr>'
    body+='</table><script>function filterRows(id,q){q=q.toLowerCase();document.querySelectorAll("#"+id+" tr[data-search]").forEach(r=>r.hidden=!r.dataset.search.toLowerCase().includes(q));}</script>'
    write_page(root/'review.html','Hypothesis Bank',body)
    return summary
