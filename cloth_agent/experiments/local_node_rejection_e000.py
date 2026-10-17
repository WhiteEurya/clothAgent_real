"""Nested run-held-out rejection audit using existing frozen features only."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from .local_node_classifier_e000 import evaluate, unit, csvwrite, sha


def thresholds(values):
    # Only calibration scores produce candidate cutoffs. Endpoints allow accept/reject all.
    a=np.unique(np.asarray(values,dtype=float))
    return np.r_[-1.,(a[:-1]+a[1:])/2,1.000001]


def calibrate(rows, rule):
    known=np.array([r['label']!='OTHER' for r in rows])
    correct=np.array([r['prediction']==r['label'] for r in rows])
    sim=np.array([max(r['sim_S0'],r['sim_S1']) for r in rows])
    margin=np.array([r['margin'] for r in rows])
    if not known.any() or not (~known).any():
        return dict(sim_threshold=1.000001,margin_threshold=1.000001,
                    status='INSUFFICIENT_CALIBRATION_REJECT_ALL')
    best=None
    for st in thresholds(sim):
        for mt in (thresholds(margin) if rule=='similarity_and_margin' else [-1.]):
            accepted=(sim>=st)&(margin>=mt)
            if np.any(accepted & (~known | ~correct)):continue
            # Maximize valid coverage subject to zero false accepts on calibration only.
            key=(int(np.sum(accepted & known & correct)),-float(mt),-float(st))
            if best is None or key>best[0]:best=(key,float(st),float(mt))
    return dict(sim_threshold=best[1],margin_threshold=best[2],status='CALIBRATED',
        calibration_known=int(known.sum()),calibration_other=int((~known).sum()),
        calibration_correct_accepted=best[0][0],calibration_false_accepts=0)


def run_fold(features,samples,holdout,rule):
    train=[i for i,s in enumerate(samples) if s['run_id']!=holdout]
    test=[i for i,s in enumerate(samples) if s['run_id']==holdout]
    training=[samples[i] for i in train];f=features[train]
    calibration=evaluate(f,training,grouped=True)
    choice=calibrate(calibration,rule)
    pools=[f[[s['label']==label for s in training]] for label in ['S0','S1']]
    if any(len(p)==0 for p in pools):raise ValueError('outer prototype missing class')
    proto=np.stack([unit(p.mean(0)) for p in pools]);rows=[]
    for i in test:
        sim=proto@features[i];margin=float(abs(sim[0]-sim[1]));pred='S0' if sim[0]>sim[1] else 'S1' if sim[1]>sim[0] else 'TIE'
        accept=bool(max(sim)>=choice['sim_threshold'] and margin>=choice['margin_threshold'] and pred!='TIE')
        rows.append(dict(sample_id=samples[i]['sample_id'],run_id=holdout,label=samples[i]['label'],
            node_label=samples[i]['node_label'],sim_S0=float(sim[0]),sim_S1=float(sim[1]),margin=margin,
            nearest=pred,prediction=pred if accept else 'UNKNOWN',accepted=accept,
            sim_threshold=choice['sim_threshold'],margin_threshold=choice['margin_threshold']))
    audit=dict(holdout_run=holdout,rule=rule,threshold_selection=choice,
        train_sample_ids=[s['sample_id'] for s in training],test_sample_ids=[samples[i]['sample_id'] for i in test],
        calibration=calibration)
    return rows,audit


def metrics(rows):
    known=[r for r in rows if r['label']!='OTHER'];other=[r for r in rows if r['label']=='OTHER']
    accepted=[r for r in known if r['accepted']];correct=sum(r['prediction']==r['label'] for r in accepted)
    return dict(known_total=len(known),known_accepted=len(accepted),known_correct_accepted=correct,
        known_coverage=len(accepted)/len(known),accepted_known_accuracy=correct/len(accepted) if accepted else None,
        known_wrong_accepted=len(accepted)-correct,known_rejected=len(known)-len(accepted),
        other_total=len(other),other_false_accepts=sum(r['accepted'] for r in other),
        other_rejection_rate=sum(not r['accepted'] for r in other)/len(other) if other else None,
        total_gpt_escalations=sum(not r['accepted'] for r in rows),
        accepted_error_rate=(len(accepted)-correct+sum(r['accepted'] for r in other))/max(1,sum(r['accepted'] for r in rows)))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,default=Path('reports/local_node_classifier_e000'))
    ap.add_argument('--output',type=Path,default=Path('reports/local_node_classifier_e000/rejection_audit'))
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    samples=list(csv.DictReader((args.input/'samples.csv').open()))
    meta=json.loads((args.input/'features/metadata.json').read_text())
    assert [s['sample_id'] for s in samples]==meta['sample_order']
    allrows=[];audits=[];summary={}
    for method in ['cls','patch_mean']:
        features=np.load(args.input/'features'/f'{method}.npy')
        for rule in ['similarity_only','similarity_and_margin']:
            result=[]
            for run in sorted({s['run_id'] for s in samples}):
                rows,audit=run_fold(features,samples,run,rule)
                for row in rows:row.update(feature=method,rule=rule)
                audit['feature']=method;audits.append(audit);result+=rows
            allrows+=result;summary[method+'/'+rule]=metrics(result)
        baseline=evaluate(features,samples,grouped=True)
        for r in baseline:r['accepted']=r['prediction']!='TIE'
        summary[method+'/no_rejection']=metrics(baseline)
    csvwrite(args.output/'predictions.csv',allrows)
    (args.output/'fold_audit.json').write_text(json.dumps(audits,indent=2))
    result=dict(protocol='outer leave-one-run-out; inner leave-one-run-out calibration on outer training runs only',
        calibration_objective='zero OTHER or wrong-known accepts on inner calibration; maximize correct-known coverage',
        tie_break='prefer lower margin threshold, then lower similarity threshold',
        samples_sha256=sha(args.input/'samples.csv'),features_sha256={m:sha(args.input/'features'/f'{m}.npy') for m in ['cls','patch_mean']},
        limitations=['same fixed corpus, not newly acquired independent runs','historical GPT/graph labels, not human ground truth',
            'OTHER only first two examples per node, concentrated in few runs','threshold calibration is not network training',
            'no threshold selected from outer test; calibration zero errors does not guarantee test zero errors'],summary=summary)
    (args.output/'report.json').write_text(json.dumps(result,indent=2))
    text=['# e000：UNKNOWN拒识与按run留出评估','',
        '继续使用第一轮固定42张图片和已提取的冻结特征，不读取新图来调参；不调用GPT、不训练网络、不接入机器人。',
        '', '## 方法',
        '外层每次留出一个完整run。其余run内再做leave-one-run-out获取校准分数；阈值只使用这些内层分数。最后用外层训练集构造S0/S1 prototype，测试留出的run。',
        '比较两种简单门限：max(sim_S0, sim_S1) >= τ；或上述条件且 margin >= δ。选择内层零OTHER误接收、零已知类错接收下覆盖最多已知样本的门限。无可用校准数据时全部拒识。',
        'OTHER只用于门限校准，不构造第三类prototype，也不训练分类器。未做端到端网络训练。',
        '', '## 外层测试结果',
        '|特征/规则|已知类正确放行|已知类拒识|OTHER误放行|OTHER拒识率|已放行已知类准确率|',
        '|---|---:|---:|---:|---:|---:|']
    for k,v in summary.items():
        acc='N/A' if v['accepted_known_accuracy'] is None else f"{v['accepted_known_accuracy']:.1%}"
        text.append(f"|{k}|{v['known_correct_accepted']}/{v['known_total']}|{v['known_rejected']}|{v['other_false_accepts']}/{v['other_total']}|{v['other_rejection_rate']:.1%}|{acc}|")
    text+=['','已知类拒识应升级给GPT；OTHER误放行意味着可能错误沿图执行。二者不能合并成单个accuracy掩盖。',
        '', '## OTHER误放行案例']
    for row in allrows:
        if row['label']=='OTHER' and row['accepted']:
            text.append(f"- {row['feature']}/{row['rule']}: {row['sample_id']} ({row['node_label']}, {row['run_id']}) → {row['prediction']}; maxsim={max(row['sim_S0'],row['sim_S1']):.6f}, margin={row['margin']:.6f}, τ={row['sim_threshold']:.6f}, δ={row['margin_threshold']:.6f}")
    manifest=json.loads((args.input/'manifest.json').read_text());latest=Path(manifest['latest_e000']['run']).name
    text+=['','## 最新e000：整组未参与阈值选择',f'Run: `{latest}`','|特征/规则|标签|输出|maxsim|margin|','|---|---|---|---:|---:|']
    for row in allrows:
        if row['run_id']==latest and row['label'] in ['S0','S1']:
            text.append(f"|{row['feature']}/{row['rule']}|{row['label']}|{row['prediction']}|{max(row['sim_S0'],row['sim_S1']):.6f}|{row['margin']:.6f}|")
    text+=['','## 局限与下一步','这不是新增独立采集的泛化实验，而是现有数据的整组留出检验。前述数据集中OTHER集中在少数run且只有18张，不能据此设定线上安全阈值。',
        '先比较误放行和已知覆盖率，再决定是否需要更明确的局部裁剪/结构特征及新场景验证；本轮不自动修改线上matcher。',
        '', '## 复现','```bash','/home/sja/miniconda3/envs/cali/bin/python -m cloth_agent.experiments.local_node_rejection_e000','```']
    (args.output/'report.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
