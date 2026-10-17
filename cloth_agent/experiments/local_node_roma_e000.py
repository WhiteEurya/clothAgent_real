"""Frozen RoMa e000 audit. No robot, planner, training, or label generation."""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .local_node_classifier_e000 import csvwrite, sha
from .local_node_rejection_e000 import calibrate, metrics

HARD = 'obs_7f66d64e53de5d54'
RULE = 'similarity_and_margin'


def dense_scores(warp, confidence):
    """RoMa symmetric (H,2W,4) coordinates in [-1,1]. No RANSAC or fit."""
    w = np.asarray(warp).reshape(-1, 4)
    c = np.asarray(confidence).reshape(-1)
    valid = np.isfinite(w).all(1) & np.isfinite(c) & (np.abs(w) < 1).all(1) & (c > 0)
    high = valid & (c >= .5)
    clean = np.where(valid, np.clip(c, 0, 1), 0)
    coverage = []
    for offset in (0, 2):
        xy = w[high, offset:offset+2]
        cells = np.clip(((xy+1)*8).astype(int), 0, 15)
        coverage.append(len(np.unique(cells[:, 0]+16*cells[:, 1]))/256)
    return dict(score=float(clean.mean()), valid_match_ratio=float(valid.mean()),
                high_confidence_match_ratio=float(high.mean()),
                mean_match_confidence=float(clean[valid].mean()) if valid.any() else 0.,
                median_match_confidence=float(np.median(clean[valid])) if valid.any() else 0.,
                workspace_coverage_A=coverage[0], workspace_coverage_B=coverage[1])


def bank_score(matrix, samples, query, candidates):
    pools = [[i for i in candidates if samples[i]['label'] == label] for label in ('S0', 'S1')]
    if any(not pool for pool in pools):
        raise ValueError('Missing representative bank class')
    values = [float(np.mean(matrix[query, pool])) for pool in pools]
    if not np.isfinite(values).all():
        raise ValueError('Missing pair score; no partial evaluation allowed')
    pred = 'S0' if values[0] > values[1] else 'S1' if values[1] > values[0] else 'TIE'
    return dict(sample_id=samples[query]['sample_id'], label=samples[query]['label'],
                prediction=pred, sim_S0=values[0], sim_S1=values[1], margin=abs(values[0]-values[1]),
                bank_S0=[samples[i]['sample_id'] for i in pools[0]],
                bank_S1=[samples[i]['sample_id'] for i in pools[1]])


def run_fold(matrix, samples, holdout):
    train = [i for i, s in enumerate(samples) if s['run_id'] != holdout]
    test = [i for i, s in enumerate(samples) if s['run_id'] == holdout]
    calibration = []
    for i in train:
        bank = [j for j in train if samples[j]['run_id'] != samples[i]['run_id']]
        calibration.append(bank_score(matrix, samples, i, bank))
    choice = calibrate(calibration, RULE)
    rows = []
    for i in test:
        row = bank_score(matrix, samples, i, train)
        accept = bool(max(row['sim_S0'], row['sim_S1']) >= choice['sim_threshold'] and
                      row['margin'] >= choice['margin_threshold'] and row['prediction'] != 'TIE')
        row.update(nearest=row['prediction'], prediction=row['prediction'] if accept else 'UNKNOWN',
                   accepted=accept, run_id=holdout, node_label=samples[i]['node_label'],
                   sim_threshold=choice['sim_threshold'], margin_threshold=choice['margin_threshold'])
        row['output'] = {'S0':'SOURCE', 'S1':'TARGET'}.get(row['prediction'], 'UNKNOWN')
        rows.append(row)
    return rows, dict(holdout_run=holdout, threshold_selection=choice, calibration=calibration,
                     train_sample_ids=[samples[i]['sample_id'] for i in train],
                     test_sample_ids=[samples[i]['sample_id'] for i in test])


def visualization(out, samples, crops, pair_data, rows):
    q = next(i for i,s in enumerate(samples) if s['sample_id'] == HARD)
    row = next(r for r in rows if r['sample_id'] == HARD)
    canvas = Image.new('RGB', (1100, 840), 'white'); draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), f"S2 hard negative: {HARD}; RoMa = {row['output']}", fill='black')
    for line, label in enumerate(('S0', 'S1')):
        # Only outer training representatives. Best match is display-only, not bank aggregation.
        candidates = [i for i,s in enumerate(samples) if s['sample_id'] in row['bank_'+label]]
        j = max(candidates, key=lambda i: pair_data[tuple(sorted((q,i)))]['score'])
        key = tuple(sorted((q,j))); data = pair_data[key]
        a,b=key; top=40+line*395
        for x,idx in ((10,a),(560,b)):
            canvas.paste(crops[idx].resize((530,340)),(x,top+30))
            draw.text((x,top),f"{samples[idx]['node_label']} {samples[idx]['sample_id']}",fill='black')
        for index, (match, c) in enumerate(zip(data['display_warp'],data['display_confidence'])):
            if c < .5 or index % 5: continue
            ax=10+(match[0]+1)*265;ay=top+30+(match[1]+1)*170
            bx=560+(match[2]+1)*265;by=top+30+(match[3]+1)*170
            color=((index*47)%180+60,(index*83)%180+60,(index*131)%180+60)
            draw.line((ax,ay,bx,by),fill=color,width=1)
            for x,y in ((ax,ay),(bx,by)):
                draw.ellipse((x-3,y-3,x+3,y+3),fill=color,outline='black')
        draw.text((10,top+374),f"Training {label} best pair: score={data['score']:.4f}; bank mean={row['sim_'+label]:.4f}",fill='black')
    canvas.save(out/'hard_negative_visualization.png')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,default=Path('reports/local_node_classifier_e000'))
    ap.add_argument('--output',type=Path,default=Path('reports/local_node_classifier_e000/roma'))
    ap.add_argument('--weights',type=Path,required=True)
    ap.add_argument('--backbone-weights',type=Path,required=True)
    ap.add_argument('--device',default='cuda')
    args=ap.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    samples=list(csv.DictReader((args.input/'samples.csv').open()))
    manifest=json.loads((args.input/'manifest.json').read_text())
    baseline=json.loads((args.input/'rejection_audit/report.json').read_text())
    assert sha(args.input/'samples.csv') == baseline['samples_sha256'], 'Dataset changed'
    crop=manifest['crop_xyxy']
    config=dict(backend='romatch 0.1.2 roma_outdoor',coarse_res=560,upsample_res=864,
                symmetric=True,custom_corr=False,crop=crop,high_confidence=.5,coverage_grid=16,
                score='mean dense confidence over all symmetric workspace pixels; invalid = 0',
                bank_aggregation='arithmetic mean over ALL eligible training representatives',
                calibration=RULE,seed=0,device=args.device,samples_sha256=sha(args.input/'samples.csv'),
                weights_sha256=sha(args.weights),backbone_sha256=sha(args.backbone_weights))
    config_hash=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    (out/'config.json').write_text(json.dumps(config,indent=2))
    cache=out/'pair_cache'/config_hash;cache.mkdir(parents=True,exist_ok=True)
    crops=[];hashes=[]
    for s in samples:
        if sha(Path(s['image'])) != s['file_sha256']:raise ValueError('Source image changed')
        crops.append(Image.open(s['image']).convert('RGB').crop(crop));hashes.append(s['file_sha256'])
    import torch
    from romatch import roma_outdoor
    torch.manual_seed(0);torch.set_num_threads(4)
    if args.device == 'cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    model=None;matrix=np.full((len(samples),len(samples)),np.nan);pairs={};cache_hits=0
    needed=[(i,j) for i in range(len(samples)) for j in range(i+1,len(samples))
            if samples[i]['run_id']!=samples[j]['run_id'] and
            (samples[i]['label']!='OTHER' or samples[j]['label']!='OTHER')]
    for number,(i,j) in enumerate(needed):
        key=hashlib.sha256((hashes[i]+hashes[j]).encode()).hexdigest();path=cache/(key+'.json')
        if path.exists():data=json.loads(path.read_text());cache_hits+=1
        else:
            if model is None:
                model=roma_outdoor(device=args.device,weights=torch.load(args.weights,map_location='cpu',weights_only=True),
                    dinov2_weights=torch.load(args.backbone_weights,map_location='cpu',weights_only=True),
                    use_custom_corr=False).eval()
                model.requires_grad_(False)
                # Warmup excluded from steady-state timings, uses first eligible pair without labels.
                with torch.inference_mode():model.match(crops[i],crops[j],device=args.device)
            if args.device=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            start=time.perf_counter()
            with torch.inference_mode():warp,conf=model.match(crops[i],crops[j],device=args.device)
            if args.device=='cuda':torch.cuda.synchronize()
            latency=time.perf_counter()-start
            peak=torch.cuda.max_memory_allocated() if args.device=='cuda' else None
            w=warp[0].float().cpu().numpy();c=conf[0].float().cpu().numpy()
            data=dense_scores(w,c)
            data.update(sample_A=samples[i]['sample_id'],sample_B=samples[j]['sample_id'],
                latency_s=latency,peak_vram_bytes=peak)
            if HARD in (samples[i]['sample_id'],samples[j]['sample_id']):
                # Sparse deterministic display only; full dense array was used for scoring.
                data.update(display_warp=w[::108,::108].reshape(-1,4).tolist(),
                            display_confidence=c[::108,::108].reshape(-1).tolist())
            path.write_text(json.dumps(data));del warp,conf
        pairs[(i,j)]=data;matrix[i,j]=matrix[j,i]=data['score']
        if number%25==0:print(f'pairs {number+1}/{len(needed)} cache_hits={cache_hits}',flush=True)
    csvwrite(out/'pair_scores.csv',[{k:v for k,v in d.items() if not k.startswith('display_')} for d in pairs.values()])
    rows=[];audits=[]
    for run in sorted({s['run_id'] for s in samples}):
        r,a=run_fold(matrix,samples,run);rows+=r;audits.append(a)
    by_id={s['sample_id']:i for i,s in enumerate(samples)}
    for row in rows:
        q=by_id[row['sample_id']]
        bank=row['bank_S0']+row['bank_S1']
        row['bank_pair_count']=len(bank)
        row['query_pair_inference_sum_s']=sum(pairs[tuple(sorted((q,by_id[b])))]['latency_s'] for b in bank)
    csvwrite(out/'predictions.csv',rows)
    (out/'fold_audit.json').write_text(json.dumps(audits,indent=2))
    m=metrics(rows);base=baseline['summary']['cls/similarity_and_margin']
    lat=[d['latency_s'] for d in pairs.values()];vram=[d['peak_vram_bytes'] for d in pairs.values() if d['peak_vram_bytes'] is not None]
    hard=next(r for r in rows if r['sample_id']==HARD)
    result=dict(roma=m,dino_baseline=base,hard_negative=hard,pair_count=len(pairs),cache_hits=cache_hits,
                query_pair_inference_sum_median_s=float(np.median([r['query_pair_inference_sum_s'] for r in rows])),
                latency_median_s=float(np.median(lat)),latency_p95_s=float(np.percentile(lat,95)),
                peak_vram_bytes=max(vram) if vram else None,config=config,
                gpu=torch.cuda.get_device_name() if args.device=='cuda' else None,
                torch=torch.__version__,gpt_calls=0,robot_calls=0)
    (out/'report.json').write_text(json.dumps(result,indent=2))
    visualization(out,samples,crops,pairs,rows)
    if m['other_false_accepts']<=base['other_false_accepts'] and m['known_correct_accepted']>=base['known_correct_accepted'] and m['known_wrong_accepted']<=base['known_wrong_accepted']:
        verdict='RoMa在本数据的拒识/覆盖指标上不劣于DINO；仍须独立新数据验证。'
    elif m['other_false_accepts']>=base['other_false_accepts'] and m['known_correct_accepted']<=base['known_correct_accepted'] and m['known_wrong_accepted']>=base['known_wrong_accepted']:
        verdict='DINO在本数据的拒识/覆盖指标上不劣于RoMa；本次没有替换依据。'
    else:verdict='两者存在覆盖率与误放行的权衡，没有全面胜者；不能仅以hard negative一例决定替换。'
    lines=['# RoMa e000 held-out audit','',verdict,'',
           '|指标|RoMa|DINO CLS+margin|','|---|---:|---:|',
           f"|S0/S1正确接受|{m['known_correct_accepted']}/24|{base['known_correct_accepted']}/24|",
           f"|已知类错误接受|{m['known_wrong_accepted']}|{base['known_wrong_accepted']}|",
           f"|OTHER拒绝|{18-m['other_false_accepts']}/18|{18-base['other_false_accepts']}/18|",
           f"|OTHER误放行|{m['other_false_accepts']}/18|{base['other_false_accepts']}/18|",
           f"|S2 hard negative|{hard['output']}|TARGET（误放行）|",'',
           f"单对模型推理 median {np.median(lat):.4f}s，P95 {np.percentile(lat,95):.4f}s；峰值 allocated VRAM {max(vram)/2**30:.3f} GiB。" if vram else f'CPU latency median {np.median(lat):.4f}s; VRAM N/A.',
           '计时含模型内部预处理/H2D与推理，CUDA同步，排除加载、warmup、输出D2H与评分；包含常驻模型显存，不是系统总显存。',
           f'共有{len(pairs)}对跨run匹配，缓存命中{cache_hits}。每次新query要匹配整组代表库，不能把单对延迟当作节点分类总延迟。',
           f"按实测单对耗时累加，一次整bank查询的推理耗时中位数估算为{result['query_pair_inference_sum_median_s']:.3f}s（不含评分/IO，非在线端到端实测）。",
           'DINO历史记录是CPU批量特征提取，未记录同配置单对latency/VRAM，因此不做不公平速度比值。','',
           '## 固定协议',
           '保持42张图、原标签、固定workspace crop、外层leave-one-run-out不变；内层同样按run留出。原calibrate()/metrics()直接复用。',
           'S0/S1 bank使用该折所有已知训练样本。内层query所属run和外层test run均不得进入bank。OTHER仅用于原有门限校准。',
           '主score在查看结果前固定：所有双向workspace像素的有效confidence均值（无效置零），然后对bank算术平均；margin取两类score差绝对值。不是cosine similarity。',
           '额外逐对保存valid ratio、高置信度(>=0.5)比例、有效match confidence均值/中位数、两图16x16网格覆盖率；这些诊断量未事后调权或择优。',
           'RoMa Outdoor官方预训练，560 coarse / 864 upsample，双向，无自定义CUDA相关核；冻结参数，不训练。',
           '没有稳定garment mask，coverage是既有固定workspace ROI覆盖，包含桌面背景，不能称为衣物像素覆盖。','',
           '## Hard negative','![Dense correspondences](hard_negative_visualization.png)',
           '图中仅选该外层训练bank最高分代表用于展示；最终预测使用bank均值，未用test挑选分类代表。',
           f"S2: sim_S0={hard['sim_S0']:.6f}, sim_S1={hard['sim_S1']:.6f}, margin={hard['margin']:.6f}; thresholds={hard['sim_threshold']:.6f}/{hard['margin_threshold']:.6f}",'',
           '## 最新完整e000']
    latest=Path(manifest['latest_e000']['run']).name
    for r in rows:
        if r['run_id']==latest:lines.append(f"- {r['sample_id']} ({r['label']}) → {r['output']}; scores={r['sim_S0']:.6f}/{r['sim_S1']:.6f}")
    lines+=['','## 相同指标完整比较','|指标|RoMa|DINO CLS+margin|','|---|---:|---:|']
    for key in base:
        lines.append(f'|{key}|{m[key]}|{base[key]}|')
    lines+=['','其中total_gpt_escalations沿用baseline字段名，仅表示拒识数量；本实验实际GPT调用为0。','', '## 限制','同一衣物/相机的少量历史样本，标签来自原graph/matcher，不是独立人工真值。Dense correspondence confidence不等价于decision-state confidence；纹理/桌面匹配可能掩盖袖口层次差异。',
            '所有负结果保留。未调用GPT、未改机器人、未运行MATCHA。','',
            '官方实现与权重来源：https://github.com/Parskatt/RoMa','',
            '复现命令与环境：[运行说明](../../../docs/experiments/local_node_roma_e000.md)','',
            '完整指标见report.json；逐折bank/阈值/校准记录见fold_audit.json；原始匹配指标见pair_scores.csv。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='config'},indent=2))

if __name__=='__main__':main()
