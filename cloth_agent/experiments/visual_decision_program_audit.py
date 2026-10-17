"""Expanded frozen-program audit; independent recompiles never receive queries."""
import csv
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor,as_completed
from .visual_decision_program import (COMPILE,PROGRAM,call,execute_query,validate_program,sha,save)
from .discovery_model import CodexMultimodal

ROOT=Path('reports/visual_program_S1_S2')
OUT=ROOT/'audit'
DATA=Path('reports/local_node_classifier_e000')

def setup():
    OUT.mkdir(parents=True,exist_ok=True)
    freeze=json.loads((ROOT/'program_freeze.json').read_text())
    assert sha(ROOT/'visual_program.json')==freeze['sha256']
    for name in ['visual_program.json','program_freeze.json']:
        dest=OUT/name
        if not dest.exists():shutil.copy2(ROOT/name,dest)
        assert sha(dest)==sha(ROOT/name)
    manifest=json.loads((ROOT/'manifest.json').read_text())
    queries=[r for r in csv.DictReader((DATA/'samples.csv').open()) if r['run_id'] not in manifest['compile_runs']]
    for r in queries:
        r['original_node_label']=r['node_label']
        if r['node_label'] not in ('S1','S2'):r['node_label']='OTHER'
        assert sha(r['image'])==r['file_sha256']
    audit=dict(queries=queries,program_sha256=freeze['sha256'],source_manifest_sha256=sha(ROOT/'manifest.json'),
        other_definition='All held-out nodes except S1/S2, including S0. Entire seed run excluded.',
        recompiles='Five independent isolated requests with identical original compiler inputs; not used for predictions.')
    if (OUT/'manifest.json').exists():assert json.loads((OUT/'manifest.json').read_text())==audit
    save(OUT/'manifest.json',audit)
    return manifest,queries

def recompile(i,manifest):
    folder=OUT/'recompiles'/f'compile_{i:02d}'
    try:
        p,meta=call(CodexMultimodal(manifest['model'],900),COMPILE+json.dumps(manifest['compiler_inputs']),
            [r['path'] for r in manifest['representatives']],PROGRAM,folder/'request')
        validate_program(p);save(folder/'visual_program.json',p)
        result=dict(index=i,status='VALID',sha256=sha(folder/'visual_program.json'),elapsed_s=meta['elapsed_s'])
    except Exception as e:
        result=dict(index=i,status='INVALID',error=str(e))
    save(folder/'status.json',result);return result

def main():
    manifest,queries=setup();program=json.loads((OUT/'visual_program.json').read_text())
    args=SimpleNamespace(output=OUT,model=manifest['model'],timeout=900)
    rows=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs={pool.submit(recompile,i,manifest):('compile',i) for i in range(1,6)}
        jobs.update({pool.submit(execute_query,args,program,r):('query',r) for r in queries})
        for f in as_completed(jobs):
            kind,source=jobs[f]
            try:
                result=f.result()
                if kind=='query':
                    result['original_node_label']=source['original_node_label']
                    result['expected']='UNKNOWN' if result['label']=='OTHER' else result['label']
                    result['correct']=result['prediction']==result['expected'];rows.append(result)
            except Exception as e:
                print('FAIL',kind,str(source),repr(e),flush=True);raise
            print(kind,json.dumps(result),flush=True)
    assert sha(OUT/'visual_program.json')==sha(ROOT/'visual_program.json')
    save(OUT/'results.json',rows)
    with (OUT/'predictions.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(sorted(rows,key=lambda r:r['sample_id']))
    print('DONE',len(rows),flush=True)



def summarize():
    """Host-only audit; never revises or selects a program using query results."""
    import statistics
    from .visual_decision_program import decide
    program=json.loads((OUT/'visual_program.json').read_text())
    rows=json.loads((OUT/'results.json').read_text())
    references=json.loads((OUT/'predicate_reference.json').read_text())
    refs={r['sample_id']:r['values'] for r in references['rows']}
    manifest=json.loads((OUT/'manifest.json').read_text())
    details=[];values={}
    for r in rows:
        r['expected']='UNKNOWN' if r['label']=='OTHER' else r['label']
        r['correct']=r['prediction']==r['expected']
        folder=OUT/r['audit']
        pred=json.loads((folder/'predicate_results.json').read_text())['results']
        assert decide(program,pred)['decision']==r['prediction']
        assert json.loads((folder/'host_decision.json').read_text())['program_sha256']==sha(OUT/'visual_program.json')
        values[r['sample_id']]={v['predicate_id']:v for v in pred}
        for v in pred:
            reference=refs[r['sample_id']][v['predicate_id']]
            details.append(dict(sample_id=r['sample_id'],label=r['label'],original_node=r['original_node_label'],
                predicate_id=v['predicate_id'],prediction=v['value'],reference=reference,confidence=v['confidence'],
                error_vs_visual_review=reference!='UNKNOWN' and v['value']!='UNKNOWN' and reference!=v['value'],
                unknown=v['value']=='UNKNOWN',reference_unscorable=reference=='UNKNOWN',visual_evidence=v['visual_evidence']))
    def csvsave(name,items):
        with (OUT/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(items[0]));w.writeheader();w.writerows(items)
    save(OUT/'results.json',rows)
    csvsave('predictions.csv',sorted(rows,key=lambda r:r['sample_id']));csvsave('predicate_audit.csv',details)
    known=[r for r in rows if r['label'] in ('S1','S2')];other=[r for r in rows if r['label']=='OTHER']
    counts=dict(total=len(rows),known=len(known),known_correct=sum(r['correct'] for r in known),
        known_accuracy=sum(r['correct'] for r in known)/len(known),unknown_total=sum(r['prediction']=='UNKNOWN' for r in rows),
        known_unknown=sum(r['prediction']=='UNKNOWN' for r in known),other=len(other),
        other_false_accept=sum(r['prediction']!='UNKNOWN' for r in other),
        mean_elapsed_s=statistics.mean(r['elapsed_s'] for r in rows),median_elapsed_s=statistics.median(r['elapsed_s'] for r in rows),
        min_elapsed_s=min(r['elapsed_s'] for r in rows),max_elapsed_s=max(r['elapsed_s'] for r in rows))
    stats=[]
    for p in program['predicates']:
        vs=[r for r in details if r['predicate_id']==p['id']]
        stats.append(dict(predicate=p['id'],errors=sum(v['error_vs_visual_review'] for v in vs),
            unknown=sum(v['unknown'] for v in vs),reference_unscorable=sum(v['reference_unscorable'] for v in vs),
            known_reference=len([v for v in vs if not v['reference_unscorable']]),
            scored_yes_no=len([v for v in vs if not v['reference_unscorable'] and not v['unknown']])))
    hard=next(r for r in rows if r['hard_negative'])
    save(OUT/'summary.json',dict(metrics=counts,predicate_stats=stats,hard_negative=hard,
        false_accepts=[r for r in other if not r['correct']],reference_provenance=references['provenance']))
    # Explicit semantic aliases for comparing the five generated predicate sets.
    # This mapping is for reporting only and never used to classify any query.
    aliases={'left_inward_flap':'left_inward','right_extended':'right_extended',
             'right_rounded_cuff':'right_curl','right_open_curl':'right_curl',
             'torso_open':'torso_open','body_open':'torso_open','torso_broad':'torso_open'}
    comparison=[]
    original_request=json.loads((ROOT/'compile/request/request.json').read_text())
    for i in range(1,6):
        folder=OUT/'recompiles'/f'compile_{i:02d}'
        p=json.loads((folder/'visual_program.json').read_text())
        request=json.loads((folder/'request/request.json').read_text())
        assert request==original_request, 'recompile input must exactly match original compile'
        comparison.append(dict(index=i,region_count=len(p['regions']),regions=p['regions'],predicates=p['predicates'],
            semantic_predicates=[aliases[v['id']] for v in p['predicates']],rules=p['rules'],
            explicit_no_longitudinal_panel=any('longitudinal' in v['question'] for v in p['predicates']),
            input_sha256=sha(folder/'request/request.json'),program_sha256=sha(folder/'visual_program.json')))
    save(OUT/'compile_stability.json',dict(method='Manual semantic alias alignment plus exact input/hash checks; reporting only',programs=comparison))
    assert sha(ROOT/'visual_program.json')==manifest['program_sha256']==sha(OUT/'visual_program.json')
    lines=['# Visual Decision Program：冻结程序扩展审计','',
        f'原程序 SHA256：`{manifest["program_sha256"]}`，未改 predicate/rule。',
        '数据范围：原有 42 张 benchmark 图片，排除整个 seed run 的 13 张；其余 29 张全部重新执行。OTHER 包括 S0，非 S1/S2 均为拒识目标。',
        '节点标签沿用已有 graph/matcher，不是人工真值。五次独立编译使用与原始有效 compile 完全一致的请求和代表图；未接触测试图，也未用于替换评测程序。','',
        '## Held-out 结果',f'- S1/S2 accuracy：{counts["known_correct"]}/{len(known)} = {counts["known_accuracy"]:.2%}（UNKNOWN 算未正确识别）。',
        f'- 最终 UNKNOWN：{counts["unknown_total"]}/{len(rows)}；其中 S1/S2 UNKNOWN：{counts["known_unknown"]}。',
        f'- OTHER false accept：{counts["other_false_accept"]}/{len(other)}。',
        f'- 每图模型耗时：平均 {counts["mean_elapsed_s"]:.2f}s，中位 {counts["median_elapsed_s"]:.2f}s，范围 {counts["min_elapsed_s"]:.2f}–{counts["max_elapsed_s"]:.2f}s；含 ROI 定位与 predicate 执行，不含 compile。',
        '', '| 原节点 | 样本数 | S1 | S2 | UNKNOWN |','|---|---:|---:|---:|---:|']
    for label in sorted({r['original_node_label'] for r in rows}):
        subset=[r for r in rows if r['original_node_label']==label]
        lines.append(f'| {label} | {len(subset)} | '+' | '.join(str(sum(r['prediction']==v for r in subset)) for v in ('S1','S2','UNKNOWN'))+' |')
    lines+=['','## Hard negative',f'`{hard["sample_id"]}`：Host = **{hard["prediction"]}**；旧 DINO/RoMa 均误判 S1。','',
        '| Predicate | 回答 | Confidence | Visual evidence |','|---|---|---:|---|']
    for v in values[hard['sample_id']].values():lines.append(f'| {v["predicate_id"]} | {v["value"]} | {v["confidence"]} | {v["visual_evidence"]} |')
    lines+=['','核心区分是 right_extended 与 right_rounded_cuff 的联合取值；left_inward_flap 与 torso_open 是共同前提。按冻结 conjunction，单独翻转任一必要项可能变 UNKNOWN，不能把联合规则归因成某一个 predicate 单独决定。',
        '','## Predicate 可靠性',
        '错误数是与先行原图视觉复核的 YES/NO 不一致数，非独立人工真值错误率。复核 UNKNOWN 不评分；模型 UNKNOWN 另列。复核由本助手完成，且此前已知第一轮 S1/S2 结果；不能声称完全独立或盲测人工标注。',
        '| Predicate | 不一致 | 可评分 YES/NO | 模型 UNKNOWN | 复核不可评分 |','|---|---:|---:|---:|---:|']
    for s in stats:lines.append(f'| {s["predicate"]} | {s["errors"]} | {s["scored_yes_no"]} | {s["unknown"]} | {s["reference_unscorable"]} |')
    lines += ['', '本批可评分视觉复核中，right_extended 为 29/29 一致且无 UNKNOWN，观察最稳定。right_rounded_cuff 也无确定复核错误，但它与不外伸的组合同时覆盖了 S3，存在分类排他性不足；这不同于把 predicate 的视觉回答判错。torso_open 是唯一出现 UNKNOWN 的项（蓝色无中央印花的 OTHER）。左袖复核有 4 张不可评分，不能据此宣称其全部回答正确。',
        'S3 false accept 的四项回答与 S2 hard negative 相同；现有程序不能在相同 predicate 向量下给出不同标签。未添加区分更靠内的袖口或其他新 predicate，也未更改 rule。']
    lines+=['','## 五次 compile 稳定性','| Compile | Regions | Predicates | 明确要求无纵向衣身折叠 |','|---|---:|---|---|']
    for c in comparison:lines.append(f'| {c["index"]} | {c["region_count"]} | {", ".join(c["semantic_predicates"])} | {c["explicit_no_longitudinal_panel"]} |')
    lines+=['','五次均包含左右袖区域和三项核心判据：左袖向内、右袖外伸、右袖开放卷曲袖口；S1 要求外伸 YES/卷曲 NO，S2 相反。',
        '四次包含 torso 区域及展开前提；一次完全省略 torso。仅两次明确把无纵向衣身折叠写入 torso predicate，另外两次只要求宽阔轮廓/印花可见。核心二类判据稳定，但 OTHER 拒识边界不完全稳定。',
        '','## 逐图结果','| Query | 原节点 | Host | 秒 |','|---|---|---|---:|']
    for r in sorted(rows,key=lambda r:r['sample_id']):lines.append(f'| [{r["sample_id"]}]({r["audit"]}/roi_overlay.png) | {r["original_node_label"]} | {r["prediction"]} | {r["elapsed_s"]:.2f} |')
    lines+=['','完整 confidence/evidence/ROI 见 audit_images；predicate_audit.csv 保存逐项复核差异；compile_stability.json 及 recompiles 保存五次原始区域、问题和规则。复核及稳定性分析不反馈到测试执行器。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(counts,indent=2));print(json.dumps(stats,indent=2))


if __name__=='__main__':
    main()
    summarize()
