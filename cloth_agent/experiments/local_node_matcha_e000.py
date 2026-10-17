"""Frozen MATCHA dense features: fixed global/spatial bank scores, original run folds."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image

from .local_node_classifier_e000 import csvwrite, sha
from .local_node_rejection_e000 import metrics
from .local_node_roma_e000 import run_fold, HARD

ASSETS = Path('/mnt/newssd/sja/clothAgent_real/model_assets/local_node_benchmark')
METHODS = ('global_mean', 'spatial_8x8')


def aggregate_dense(dense):
    """CHW -> unit global mean / 8x8 cell-normalized flattened descriptor.

    Cosine of spatial descriptors equals average corresponding-cell cosine.
    Neither aggregation fits parameters or reads labels.
    """
    x = np.asarray(dense, dtype=np.float32)
    if x.ndim != 3 or min(x.shape[1:]) < 8 or any(n % 8 for n in x.shape[1:]) or not np.isfinite(x).all():
        raise ValueError('Invalid dense feature map')
    def unit(v):
        return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)
    global_mean = unit(x.mean((1, 2)))
    # Fixed equal-area spatial cells; no per-image crop or matching-count score.
    cells = np.stack([x[:, a:b, c:d].mean((1, 2))
                      for a,b in zip(np.linspace(0,x.shape[1],9,dtype=int)[:-1],np.linspace(0,x.shape[1],9,dtype=int)[1:])
                      for c,d in zip(np.linspace(0,x.shape[2],9,dtype=int)[:-1],np.linspace(0,x.shape[2],9,dtype=int)[1:])])
    return {'global_mean': global_mean, 'spatial_8x8': unit(cells).reshape(-1)/8}


def evaluate_bank(features, samples, baseline_folds):
    matrix = features @ features.T
    rows, audits = [], []
    for run in sorted({s['run_id'] for s in samples}):
        result, audit = run_fold(matrix, samples, run)
        for key in ('train_sample_ids', 'test_sample_ids'):
            if audit[key] != baseline_folds[run][key]:
                raise ValueError('Split differs from baseline')
        rows += result; audits.append(audit)
    return rows, audits


def load_model(source, weights, sd_path, assets):
    """Use official model directly; avoid loading unrelated sparse detectors.

    Local backbone routing and PyTorch SDPA replace download and optional xformers
    calls only; checkpoint/model math is not replaced with another network.
    """
    import torch
    from unittest.mock import patch
    sys.path.insert(0, str(source.resolve()))
    from third_party.dift import dift_sd
    from diffusers.models.attention_processor import AttnProcessor2_0
    from diffusers import StableDiffusionPipeline
    original = dift_sd.SDFeaturizer.__init__
    def local_init(self, sd_id=None, *a, **kw):
        return original(self, str(sd_path.resolve()), *a, **kw)
    def sdpa(self, *a, **kw):
        self.unet.set_attn_processor(AttnProcessor2_0())
        self.vae.set_attn_processor(AttnProcessor2_0())
    torch.hub.set_dir(str(assets/'matcha_torch/hub'))
    from matcha.model.matcha import Matcha
    with patch.object(dift_sd.SDFeaturizer, '__init__', local_init), patch.object(
            StableDiffusionPipeline, 'enable_xformers_memory_efficient_attention', sdpa):
        model = Matcha().cuda().eval()
    state = torch.load(weights, map_location='cpu', weights_only=True)
    # Official README loads the raw Matcha state dict, not a trainer checkpoint.
    incompatible = model.load_state_dict(state, strict=False)
    missing = [k for k in incompatible.missing_keys if not k.startswith(('dift.', 'dinov2.'))]
    if missing or incompatible.unexpected_keys:
        raise ValueError(f'Checkpoint incompatible: missing={missing}, unexpected={incompatible.unexpected_keys}')
    model.requires_grad_(False)
    return model, dict(missing_backbone_keys=incompatible.missing_keys, unexpected=incompatible.unexpected_keys,
                      attention='PyTorch SDPA (xformers replacement)', ensemble_size=model.ensemble_size)


def prepare_tensor(image, crop):
    import torch
    import torch.nn.functional as F
    # Official ImageProcessor: RGB [0,1], bilinear resize to512, no ImageNet normalize.
    x=np.asarray(image.convert('RGB').crop(crop),dtype=np.float32).copy()/255
    return F.interpolate(torch.from_numpy(x).permute(2,0,1)[None],(512,512),mode='bilinear').cuda()


def gpu_aggregate(dense, method):
    import torch.nn.functional as F
    if method=='global_mean':return F.normalize(dense.mean((-2,-1)),dim=1)[0]
    cells=F.adaptive_avg_pool2d(dense,(8,8)).flatten(2).transpose(1,2)
    return F.normalize(cells,dim=-1).flatten()/8


def missing_report(out, reason, baseline, roma):
    result=dict(status='BLOCKED_MISSING_WEIGHTS',reason=reason,model_calls=0,
                gpt_calls=0,robot_calls=0,metrics=None,latency=None,
                dino=baseline['summary']['cls/similarity_and_margin'],roma=roma['roma'])
    (out/'status.json').write_text(json.dumps(result,indent=2))
    (out/'report.md').write_text('# MATCHA e000 — 未运行\n\n'+reason+'\n\n'
        '未生成真实预测或延迟，不能比较优劣。固定13折、标签与校准协议的代码已实现；不使用随机权重替代。\n\n'
        '|方法|S0/S1正确接受|OTHER拒绝|OTHER误放行|S2|latency|\n|---|---:|---:|---:|---|---|\n'
        '|DINO CLS+margin|15/24|17/18|1/18|误接受为S1|未测同配置|\n'
        '|RoMa|6/24|17/18|1/18|误接受为S1|单对307ms；整bank约6.76s推理估算|\n'
        '|MATCHA global mean|未测|未测|未测|未测|未测|\n'
        '|MATCHA spatial8x8|未测|未测|未测|未测|未测|\n\n'
        '计划评分已固定，两种方法全部报告；测试集不能用于选阈值。详见[运行说明](../../../docs/experiments/local_node_matcha_e000.md)。\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 2


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,default=Path('reports/local_node_classifier_e000'))
    ap.add_argument('--output',type=Path,default=Path('reports/local_node_classifier_e000/matcha'))
    ap.add_argument('--assets',type=Path,default=ASSETS)
    ap.add_argument('--weights',type=Path)
    args=ap.parse_args();assets=args.assets;out=args.output;out.mkdir(parents=True,exist_ok=True)
    for name,sub in [('HF_HOME','hf_cache'),('TORCH_HOME','matcha_torch'),('TMPDIR','tmp'),('XDG_CACHE_HOME','cache')]:
        (assets/sub).mkdir(parents=True,exist_ok=True);os.environ[name]=str(assets/sub)
    os.environ['HF_HUB_OFFLINE']='1'
    for path in ('matcha_deps','roma_deps'):sys.path.insert(0,str(assets/path))
    weights=args.weights or assets/'matcha_pretrained.pth'
    samples=list(csv.DictReader((args.input/'samples.csv').open()))
    baseline=json.loads((args.input/'rejection_audit/report.json').read_text())
    roma=json.loads((args.input/'roma/report.json').read_text())
    if sha(args.input/'samples.csv')!=baseline['samples_sha256']:raise ValueError('Dataset changed')
    folds={a['holdout_run']:a for a in json.loads((args.input/'rejection_audit/fold_audit.json').read_text())
           if a['feature']=='cls' and a['rule']=='similarity_and_margin'}
    if len(folds)!=13:raise ValueError('Expected original13 folds')
    if not weights.is_file():return missing_report(out,'未找到官方预训练 matcha_pretrained.pth；当前官方Drive链接404。',baseline,roma)
    source=assets/'matcha_source';sd=assets/'matcha_sd21'
    required=[sd/'unet/diffusion_pytorch_model.safetensors',sd/'vae/diffusion_pytorch_model.safetensors',sd/'text_encoder/model.safetensors']
    if any(not p.is_file() for p in required):return missing_report(out,'MATCHA已存在，但SD2.1 backbone下载未完成。',baseline,roma)
    crop=json.loads((args.input/'manifest.json').read_text())['crop_xyxy']
    config=dict(model='nv-dvl/matcha full',source_commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
                checkpoint_sha256=sha(weights),sd_sha256={str(p.relative_to(sd)):sha(p) for p in required},
                dinov2_sha256=sha(assets/'roma_weights/dinov2_vitl14_pretrain.pth'),samples_sha256=sha(args.input/'samples.csv'),
                input_size=512,crop=crop,semantic_mode=True,ensemble_size=8,seed=0,attention='SDPA',
                aggregations=list(METHODS),bank='all training known representatives; arithmetic mean pair cosine',
                calibration='unchanged nested leave-one-run-out similarity_and_margin')
    (out/'config.json').write_text(json.dumps(config,indent=2))
    cachekey=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    cache=assets/'matcha_features'/cachekey;cache.mkdir(parents=True,exist_ok=True)
    import torch
    torch.set_num_threads(4);torch.manual_seed(0)
    if not torch.cuda.is_available():raise RuntimeError('RTX4090 required for latency audit')
    model,loading=load_model(source,weights,sd,assets)
    (out/'loading.json').write_text(json.dumps(loading,indent=2))
    descriptors={k:[] for k in METHODS};measurements=[]
    images=[]
    for s in samples:
        if sha(Path(s['image']))!=s['file_sha256']:raise ValueError('Image changed')
        images.append(Image.open(s['image']).convert('RGB'))
    # Warmup does not read labels.
    with torch.inference_mode():model(prepare_tensor(images[0],crop),semantic_mode=True)
    for i,s in enumerate(samples):
        file=cache/(s['file_sha256']+'.npz');timing=cache/(s['file_sha256']+'.json')
        if file.exists() and timing.exists():
            f=np.load(file);pooled={m:f[m] for m in METHODS};measurement=json.loads(timing.read_text())
        else:
            torch.manual_seed(0);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            t=time.perf_counter();x=prepare_tensor(images[i],crop);torch.cuda.synchronize();start=time.perf_counter()
            with torch.inference_mode():dense=model(x,semantic_mode=True)
            torch.cuda.synchronize();elapsed=time.perf_counter()-start
            # Store actual dense descriptors, then aggregate independently and label-free.
            cpu=dense[0].float().cpu().numpy();pooled=aggregate_dense(cpu)
            measurement=dict(sample_id=s['sample_id'],feature_extraction_s=elapsed,
                             preprocess_s=start-t,peak_vram_bytes=torch.cuda.max_memory_allocated(),dense_shape=list(cpu.shape))
            np.savez_compressed(file,dense=cpu,**pooled);timing.write_text(json.dumps(measurement));del x,dense,cpu
        for m in METHODS:descriptors[m].append(pooled[m])
        measurements.append(measurement);print(f'features {i+1}/{len(samples)}',flush=True)
    rows=[];audits=[];summary={};features={}
    for m in METHODS:
        features[m]=np.stack(descriptors[m]);r,a=evaluate_bank(features[m],samples,folds)
        for row in r:row['method']=m
        for audit in a:audit['method']=m
        rows+=r;audits+=a;summary[m]=metrics(r)
    # Actual fresh-query timers: cached training bank; reread/resize/forward/pool/score/gate included.
    # Disk decode excluded: RGB images are already in host RAM. No test sample in its bank.
    bank_gpu={m:torch.from_numpy(features[m]).cuda() for m in METHODS}
    query_times=[]
    for m in METHODS:
        for i,s in enumerate(samples):
            row=next(r for r in rows if r['method']==m and r['sample_id']==s['sample_id'])
            bank_ids=[[j for j,b in enumerate(samples) if b['run_id']!=s['run_id'] and b['label']==lab] for lab in ('S0','S1')]
            banks=[bank_gpu[m][ids] for ids in bank_ids]
            torch.manual_seed(0);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
            with torch.inference_mode():
                x=prepare_tensor(images[i],crop);dense=model(x,semantic_mode=True);q=gpu_aggregate(dense,m)
                sim=torch.stack([(b@q).mean() for b in banks]);values=sim.cpu().numpy()
                accept=bool(values.max()>=row['sim_threshold'] and abs(values[0]-values[1])>=row['margin_threshold'] and values[0]!=values[1])
            torch.cuda.synchronize();elapsed=time.perf_counter()-start
            if not np.allclose(values,[row['sim_S0'],row['sim_S1']],atol=2e-4):raise ValueError('Fresh-query/cache scores differ')
            query_times.append(dict(sample_id=s['sample_id'],method=m,query_total_s=elapsed,
                                    peak_vram_bytes=torch.cuda.max_memory_allocated(),bank_size=sum(map(len,bank_ids))))
            del x,dense,q,sim
            print(f'timing {m} {i+1}/{len(samples)}',flush=True)
    csvwrite(out/'predictions.csv',rows);csvwrite(out/'feature_latency.csv',measurements);csvwrite(out/'query_latency.csv',query_times)
    (out/'fold_audit.json').write_text(json.dumps(audits,indent=2))
    hard={m:next(r for r in rows if r['method']==m and r['sample_id']==HARD) for m in METHODS}
    result=dict(status='COMPLETE',summary=summary,hard_negative=hard,dino=baseline['summary']['cls/similarity_and_margin'],roma=roma['roma'],
                feature_latency_median_s=float(np.median([r['feature_extraction_s'] for r in measurements])),
                query_latency_median_s={m:float(np.median([r['query_total_s'] for r in query_times if r['method']==m])) for m in METHODS},
                peak_vram_bytes=max(r['peak_vram_bytes'] for r in measurements+query_times),gpu=torch.cuda.get_device_name(),
                source_config=config,feature_cache=str(cache),gpt_calls=0,robot_calls=0)
    (out/'report.json').write_text(json.dumps(result,indent=2));(out/'status.json').write_text(json.dumps({'status':'COMPLETE'}))
    text=['# MATCHA e000：固定两种聚合','', '|方法|S0/S1正确接受|OTHER拒绝|OTHER误放行|S2|单query耗时|','|---|---:|---:|---:|---|---|']
    for name,metric,status,latency in [('DINO CLS+margin',result['dino'],'TARGET（误放行）','未测同配置'),('RoMa',result['roma'],'TARGET（误放行）',f"约{roma['query_pair_inference_sum_median_s']:.3f}s（推理累加估算）")]+[(m,summary[m],hard[m]['output'],f"{result['query_latency_median_s'][m]:.3f}s（实测）") for m in METHODS]:
        text.append(f"|{name}|{metric['known_correct_accepted']}/24|{18-metric['other_false_accepts']}/18|{metric['other_false_accepts']}/18|{status}|{latency}|")
    text += ['',f"MATCHA单图特征提取中位数{result['feature_latency_median_s']:.3f}s；peak allocated VRAM {result['peak_vram_bytes']/2**30:.3f}GiB。",'Query计时含CPU内存RGB预处理、H2D、dense特征提取、聚合、与完整训练bank比较及门限；不含图片磁盘解码、模型加载、bank离线构建。CUDA同步，排除warmup。',
             '', '全局：dense map均值后L2归一化；空间：固定workspace8x8网格，逐cell均值/L2归一化，对应cell cosine平均。各类使用该折全部训练representatives的平均相似度，无match-count。',
             '两种规则均在测试前固定，全部报告，不事后选择最佳版本作为无偏结果。阈值直接复用既有calibrate；13外层fold与baseline逐项核对；inner calibration同样排除query所属整个run。',
             '固定ROI包含背景，没有稳定衣物mask。MATCHA使用官方semantic_mode、512输入和ensemble8；扩散随机种子固定0。',
             '少样本历史标签不是独立人工真值。RoMa计时为估算且DINO未测同硬件单query，不作严格速度排名。负结果保留；不训练、不调用GPT、不改机器人、不继续LightGlue。',
             '', '官方来源：https://github.com/nv-dvl/matcha','', '完整指标：report.json；逐图预测：predictions.csv；逐折校准与bank：fold_audit.json。']
    (out/'report.md').write_text('\n'.join(text)+'\n');print(json.dumps(result,indent=2));return 0

if __name__=='__main__':raise SystemExit(main())
