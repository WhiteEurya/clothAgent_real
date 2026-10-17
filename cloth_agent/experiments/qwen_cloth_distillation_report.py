"""Report frozen distillation evaluation and independently supplied human audit."""
import csv,json,math,statistics
from pathlib import Path
from collections import Counter
from .qwen_cloth_distillation import OUT,readl,PIDS,load,save

def audit_metrics():
    f=OUT/'human_audit/human_labels.csv';rows=list(csv.DictReader(f.open()));labels={r['sample_id']:r for r in readl(OUT/'teacher_labels.jsonl')}
    expected={r['sample_id'] for r in load(OUT/'human_audit/manifest.json')}
    if len(rows)!=40 or {r['sample_id'] for r in rows}!=expected:raise ValueError('Human audit IDs do not match frozen random sample')
    submitted=[r for r in rows if all(r.get(p) in ('YES','NO','UNKNOWN') for p in PIDS) and r.get('reviewer','').strip()]
    for r in submitted:
        cached=OUT/'teacher_requests'/r['sample_id']/'parsed.json'
        if r['sample_id'] not in labels and cached.exists():labels[r['sample_id']]=dict(predicates=load(cached))
    rows=[r for r in submitted if r['sample_id'] in labels];nrows=len(rows)
    save(OUT/'human_audit/status.json',dict(selected=40,human_completed=len(submitted),paired_with_teacher=nrows,complete=len(submitted)==nrows==40))
    if not rows:return None
    result={}
    for p in PIDS:
        pairs=[(labels[r['sample_id']]['predicates'][p]['value'],r[p]) for r in rows];acc=sum(a==b for a,b in pairs)
        s=dict(exact_agreement=acc,n=nrows,human_unknown=sum(b=='UNKNOWN' for a,b in pairs),teacher_unknown=sum(a=='UNKNOWN' for a,b in pairs))
        for v in ['YES','NO']:
            n=sum(a==v for a,b in pairs);k=sum(a==b==v for a,b in pairs);rate=k/n if n else None
            z=1.96
            if n:
                center=(rate+z*z/(2*n))/(1+z*z/n);delta=z*math.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/(1+z*z/n);ci=[center-delta,center+delta]
            else:ci=None
            s[v+'_precision']=dict(correct=k,total=n,rate=rate,wilson95=ci)
        result[p]=s
    save(OUT/'human_audit/metrics.json',result)
    lines=['# GPT-6 teacher human audit','',f'Human completed: {len(submitted)}/40; paired teacher labels: {nrows}/40. Partial metrics are provisional.','40 images selected uniformly without replacement before seeing labels; seed 20261015. Human page hides teacher answers. The human reviewer supplies the CSV; no AI review is counted. Human UNKNOWN does not verify a teacher YES/NO.','','| Predicate | Exact agreement | YES precision | NO precision | Human UNKNOWN |','|---|---:|---:|---:|---:|']
    for p,s in result.items():
        a=s['YES_precision'];b=s['NO_precision'];lines.append(f"| {p} | {s['exact_agreement']}/{nrows} | {a['correct']}/{a['total']} | {b['correct']}/{b['total']} | {s['human_unknown']} |")
    filtered={}
    lines+=['','## Prespecified confidence >=0.85 and YES/NO filter','| Predicate | Retained agreement | Retained YES precision |','|---|---:|---:|']
    for p in PIDS:
        pairs=[(labels[r['sample_id']]['predicates'][p],r[p]) for r in rows]
        pairs=[(d['value'],v) for d,v in pairs if d['value'] in ('YES','NO') and d['confidence']>=.85]
        n=sum(a=='YES' for a,b in pairs);k=sum(a==b=='YES' for a,b in pairs)
        filtered[p]=dict(eligible_n=len(pairs),eligible_correct=sum(a==b for a,b in pairs),YES_correct=k,YES_n=n)
        lines.append(f"| {p} | {filtered[p]['eligible_correct']}/{len(pairs)} | {k}/{n} |")
    save(OUT/'human_audit/confidence_filtered_precision.json',filtered)
    lines+=['','Left-flap and cuff positive labels still disagree with this reviewer after the confidence filter. These are not adjudicated teacher errors: the user noted that human interpretation of the frozen visual definitions may differ. Training keeps the original GPT labels and prespecified threshold; human answers are used for audit, not silently substituted into training. Any improvement against GPT test references must be interpreted as teacher agreement, not a claim of absolute visual correctness. Chinese human questions preserve the frozen definitions; the explanatory cuff examples use original compiler representatives, not held-out or audited images.']
    lines+=['','YES precision means human YES among teacher YES, not confidence. Zero denominator is N/A. Full counts and Wilson intervals are in human_audit/metrics.json. The 40-image audit has wide uncertainty for rare YES cases. Audit labels are not used to tune frozen test prompts or thresholds.']
    (OUT/'audit.md').write_text('\n'.join(lines)+'\n');return result

def report():
    audit=audit_metrics();rows=list(csv.DictReader((OUT/'evaluation.csv').open()));stats={}
    for method in ['base','fine_tuned']:
        stats[method]={}
        for p in PIDS:
            rs=[r for r in rows if r['method']==method and r['predicate']==p];known=[r for r in rs if r['node_label'] in ('S1','S2')]
            assert len(rs)==29 and len(known)==11
            lat=[float(r['elapsed_s']) for r in rs if r['elapsed_s']];mem=[float(r['peak_allocated_mib']) for r in rs if r['peak_allocated_mib']]
            stats[method][p]=dict(agreement=sum(r['value']==r['reference'] for r in rs),n=29,known_agreement=sum(r['value']==r['reference'] for r in known),known_n=11,distribution=dict(Counter(r['value'] for r in rs)),errors=sum(r['status']!='OK' for r in rs),mean_latency_s=statistics.mean(lat),median_latency_s=statistics.median(lat),peak_allocated_mib=max(mem))
    special={sid:{method:{r['predicate']:dict(value=r['value'],reference=r['reference']) for r in rows if r['sample_id']==sid and r['method']==method} for method in stats} for sid in ['obs_7f66d64e53de5d54','obs_cb681fd8ddcdf275']}
    from .visual_decision_program import decide
    special_host={}
    for sid,methods in special.items():
        special_host[sid]={}
        for method,values in methods.items():
            result=decide(load(OUT/'visual_program.json'),[dict(predicate_id=p,value=d['value']) for p,d in values.items()]) if all(d['value'] in ('YES','NO','UNKNOWN') for d in values.values()) else dict(decision='ERROR')
            special_host[sid][method]=result['decision']
    save(OUT/'special_host_decisions.json',special_host)
    paired={}
    for p in PIDS:
        base={r['sample_id']:r for r in rows if r['method']=='base' and r['predicate']==p};ft={r['sample_id']:r for r in rows if r['method']=='fine_tuned' and r['predicate']==p};gain=sum(ft[s]['value']==ft[s]['reference'] and base[s]['value']!=base[s]['reference'] for s in base);loss=sum(ft[s]['value']!=ft[s]['reference'] and base[s]['value']==base[s]['reference'] for s in base);n=gain+loss;k=min(gain,loss);pv=min(1,2*sum(math.comb(n,i) for i in range(k+1))/2**n) if n else 1
        paired[p]=dict(gains=gain,losses=loss,exploratory_mcnemar_p=pv)
    train=[r for r in readl(OUT/'atomic_samples.jsonl') if r['split']=='train'];majority={}
    for p in PIDS:
        counts=Counter(r['value'] for r in train if r['predicate']==p);v=max(['YES','NO'],key=lambda x:counts[x]);rs=[r for r in rows if r['method']=='base' and r['predicate']==p]
        majority[p]=dict(train_majority_value=v,agreement=sum(r['reference']==v for r in rs),known_agreement=sum(r['reference']==v for r in rs if r['node_label'] in ('S1','S2')))
    save(OUT/'evaluation_summary.json',dict(methods=stats,special=special,paired=paired,train_majority_baseline=majority))
    ds=load(OUT/'dataset_stats.json');cfg=load(OUT/'training_config.json');best=load(OUT/'best_checkpoint.json')
    draws=[r for f in sorted(OUT.glob('epoch_*_draws.json')) for r in load(f)]
    usage=dict(actual_train_unique_images=len({r['sample_id'] for r in draws}),actual_atomic_draws=len(draws),draw_distributions={p:dict(Counter(r['value'] for r in draws if r['predicate']==p)) for p in PIDS})
    save(OUT/'actual_training_usage.json',usage)
    lines=['# GPT → Qwen cloth predicate distillation proof-of-concept','','## Data and protocol','Actual training draws / unique images: '+json.dumps(usage),f"Frozen source images: {ds['images']}; split {ds['splits']}; retained atomic QA: {ds['atomic_samples']}. Per-class counts in dataset_summary.md. Entire test day/run quarantined, validation separated by acquisition day. Exact/near-copy checks are documented in manifest.json.",'','Teacher: gpt-6-astra, single RGB, only frozen four predicates; no node/action output or stored chain-of-thought. Confidence >=0.85, YES/NO retained. Low confidence/UNKNOWN excluded. Inverse-frequency training sampling balances predicate/answer cells; repeated draws are not new independent images.','',f"QLoRA NF4 + BF16 compute; rank {cfg['r']}, alpha {cfg['lora_alpha']}, dropout {cfg['lora_dropout']}, language attention/MLP projections; frozen vision. {cfg['epochs']} epochs, LR {cfg['learning_rate']}, microbatch 1, accumulation 8. Best epoch {best['epoch']} by macro validation value-token loss. No post-test hyperparameter changes or additional epochs.",'','Training uses exact atomic-original text and single-predicate JSON envelope, with loss only on YES/NO answer-value tokens; no bundled four-predicate supervision or confidence training. Test inference uses the same BF16 checkpoint for base and base+adapter, frozen default processor, same FlashAttention2, same constrained JSON grammar, greedy decoding. QLoRA quantization is only for training.','','## Human teacher audit', ('Complete: see audit.md.' if load(OUT/'human_audit/status.json')['complete'] else 'Partial human audit: see audit.md; full 40-image precision remains pending.') if audit else '**PENDING human audit: teacher precision cannot be reported yet; high confidence is not human validation.**','','## Held-out predicate agreement with frozen GPT reference','| Predicate | Base /29 | Fine-tuned /29 | Base S1/S2 /11 | Fine-tuned S1/S2 /11 |','|---|---:|---:|---:|---:|']
    for p in PIDS:
        a=stats['base'][p];b=stats['fine_tuned'][p];lines.append(f"| {p} | {a['agreement']} | {b['agreement']} | {a['known_agreement']} | {b['known_agreement']} |")
    lines+=['','These are agreements with historical GPT outputs, not independently human-verified test accuracy. The 29 images are small and correlated within runs; exploratory paired p-values are not corrected for multiplicity. Confidence output is not calibrated or used to threshold test.','','## Distributions, latency and VRAM','| Method | Predicate | YES / NO / UNKNOWN | Mean / median seconds per call | Peak allocated GiB |','|---|---|---|---|---:|']
    for method,ss in stats.items():
        for p,s in ss.items():lines.append(f"| {method} | {p} | {' / '.join(str(s['distribution'].get(v,0)) for v in ['YES','NO','UNKNOWN'])} | {s['mean_latency_s']:.3f} / {s['median_latency_s']:.3f} | {s['peak_allocated_mib']/1024:.2f} |")
    lines+=['','Latency includes image load/processing, transfer, constrained generation and synchronization; excludes model load and one training-image warmup per method. Four atomic calls per image. Process/device sampled VRAM in evaluation_memory.json; training VRAM in training_memory.json. Errors remain explicit, never converted to UNKNOWN.','','## Hard S2 / S3','```json',json.dumps(special,indent=2),'```','Frozen Host-rule decisions: '+json.dumps(special_host),'The S3 case is an OTHER image. Matching its historical GPT predicates can reproduce the known S3→S2 false accept; that is teacher agreement, not correct state classification. Host decisions are post-hoc and never training targets.','','## Interpretation','Train-majority constant-answer comparator (chosen only on train counts): '+json.dumps(majority), 'Compare both difficult predicates on all 29 and the 11 S1/S2 subset; class distributions show whether the model collapsed to a constant. If only train/validation improves, evidence is consistent with overfit or insufficient diversity, not successful held-out transfer. Even an observed test gain supports only this small teacher-reference proof-of-concept, not broad garment understanding. Human audit remains a separate requirement.','','## Reproducibility','Source: cloth_agent/experiments/qwen_cloth_distillation{,_train,_report}.py. manifests freeze test exclusion, prompts, program and chosen best adapter; per-request raw answers in teacher_requests and eval_raw. No robot code changed.', '', 'Implementation references: [PEFT QLoRA](https://huggingface.co/docs/peft/v0.18.0/developer_guides/quantization), [Qwen model card](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct).']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n');print(json.dumps(stats,indent=2))
if __name__=='__main__':
    import sys
    if len(sys.argv)>1 and sys.argv[1]=='audit':print(audit_metrics())
    else:report()
