"""Qwen3-VL Instruct local evaluation with constrained decoding, no GPT calls."""
import os,json,csv,time,statistics,re
from pathlib import Path
from PIL import Image
from .visual_program_inheritance import PIDS,parse,load,MemoryMonitor
from .visual_decision_program import sha,save
SOURCE=Path('reports/visual_program_S1_S2')
PREVIOUS=Path('reports/visual_program_inheritance')
OUT=PREVIOUS/'qwen3_vl_8b_instruct'
REVISION='0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'
CHECKPOINT=Path.home()/'.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots'/REVISION
# Grammar fixes only syntax and field order; it never selects semantic values.
WS=r'[ \t\r\n]{0,8}'
NUMBER=r'(0(\.[0-9]{1,3})?|1(\.0{1,3})?)'
VALUE=r'"(YES|NO|UNKNOWN)"'
ENTRY=r'\{'+WS+r'"value"'+WS+':'+WS+VALUE+WS+','+WS+r'"confidence"'+WS+':'+WS+NUMBER+WS+r'\}'
GRAMMARS={'zero_shot':r'(S1|S2|UNKNOWN)','inheritance':r'\{'+WS+(WS+','+WS).join('"'+p+'"'+WS+':'+WS+ENTRY for p in PIDS)+WS+r'\}'}

def classify(text,method,program):
    try:
        p=parse(text,method,program)
        if re.fullmatch(GRAMMARS[method],text) is None:raise ValueError('Decoded text does not satisfy enforced grammar')
        return p|dict(status='OK',error=None)
    except Exception as e:return dict(prediction='PARSE_ERROR',predicates=None,status='PARSE_ERROR',error=str(e))

def metrics(rows):
    known=[r for r in rows if r['label'] in ('S1','S2')];other=[r for r in rows if r['label']=='OTHER']
    return dict(known_correct=sum(r['prediction']==r['label'] for r in known),known_total=len(known),unknown=sum(r['prediction']=='UNKNOWN' for r in rows),other_false_accept=sum(r['prediction'] in ('S1','S2') for r in other),other_total=len(other),parse_errors=sum(r['status']=='PARSE_ERROR' for r in rows),inference_errors=sum(r['status']=='INFERENCE_ERROR' for r in rows),hard_s2=next(r['prediction'] for r in rows if r['sample_id']=='obs_7f66d64e53de5d54'),s3_case=next(r['prediction'] for r in rows if r['sample_id']=='obs_cb681fd8ddcdf275'),mean_latency_s=statistics.mean(r['elapsed_s'] for r in rows),median_latency_s=statistics.median(r['elapsed_s'] for r in rows),peak_allocated_mib=max(r['peak_allocated_mib'] for r in rows),peak_reserved_mib=max(r['peak_reserved_mib'] for r in rows))

def report(results,memory):
    stat={k:metrics(v) for k,v in results.items()};agreement={}
    for p in PIDS:
        equal=valid=0
        for r in results['inheritance']:
            ref=load(SOURCE/'audit/audit_images'/r['sample_id']/'predicate_results.json')['results'];rv=next(v['value'] for v in ref if v['predicate_id']==p)
            equal+=r[p]==rv;valid+=r['status']=='OK'
        agreement[p]=dict(equal=equal,total=29,valid=valid,agreement=equal/29)
    a={r['sample_id']:r for r in results['zero_shot']};paired=dict(known_improved=[],known_worsened=[],other_rejections_gained=[],other_rejections_lost=[])
    for r in results['inheritance']:
        z=a[r['sample_id']];known=r['label']!='OTHER';target=r['label'] if known else 'UNKNOWN';before=z['prediction']==target;after=r['prediction']==target
        if after and not before:paired['known_improved' if known else 'other_rejections_gained'].append(r['sample_id'])
        if before and not after:paired['known_worsened' if known else 'other_rejections_lost'].append(r['sample_id'])
    summary=dict(methods=stat,predicate_agreement=agreement,paired=paired,memory=memory);save(OUT/'summary.json',summary)
    old=load(PREVIOUS/'summary.json');manifest=load(OUT/'manifest.json')
    lines=['# Qwen3-VL Visual Program Experience Inheritance','','模型：**Qwen/Qwen3-VL-8B-Instruct**，非Thinking；revision `'+REVISION+'`。BF16，无训练、无GPT调用、无机器人执行。Attention实现：`'+manifest['attention_implementation']+'`。',
      '与Molmo实验相同29张held-out、完全相同A/B prompt、原visual_program、Host rules和GPT baseline。仅更换模型及其原生图像处理器，并按本次要求增加token级结构化输出约束。两者的对照不因此成为纯模型单因素实验。',
      'A解码仅允许S1/S2/UNKNOWN；B解码固定四个键和值域，confidence在0到1（最多3位小数）。使用lm-format-enforcer逐token屏蔽不合法候选。B最终分类仅由Host执行原rules，不使用confidence阈值。',
      'Greedy decoding（do_sample=False），max_new_tokens=384；全精度原图交给官方processor，无手工ROI。每方法一次seed图warmup，test每图每方法独立一次调用，顺序交替。无test调参或重试。',
      'PARSE_ERROR与INFERENCE_ERROR作为错误单列，不计UNKNOWN、不计OTHER false accept，也不计正确接受。固定11/18分母保留，不删除失败样本。',
      '', '| Method | S1/S2 | UNKNOWN | OTHER FA | Hard S2 | S3 case | Predicate agreement | Latency mean / median | Peak allocated VRAM |','|---|---:|---:|---:|---|---|---|---|---|']
    for name,s in [('MolmoPoint zero-shot',old['methods']['zero_shot']),('MolmoPoint + GPT visual program',old['methods']['inheritance']),('Qwen3-VL zero-shot',stat['zero_shot']),('Qwen3-VL + GPT visual program',stat['inheritance'])]:
        pa='—' if 'zero-shot' in name else str(sum(v['equal'] for v in (old['predicate_agreement'] if name.startswith('Molmo') else agreement).values()))+'/116'
        lines.append(f"| {name} | {s['known_correct']}/11 | {s['unknown']} | {s['other_false_accept']}/18 | {s['hard_s2']} | S3→{s['s3_case']} | {pa} | {s['mean_latency_s']:.2f}s / {s['median_latency_s']:.2f}s | {s['peak_allocated_mib']/1024:.2f} GiB |")
    lines+=['| GPT-6 visual program baseline | 11/11 | 17 | 1/18 | correct | S3→S2 | reference | 37.31s / 36.76s | N/A |',
      '', '| Predicate | Molmo agreement | Qwen agreement |','|---|---:|---:|']
    for p in PIDS:lines.append(f"| {p} | {old['predicate_agreement'][p]['equal']}/29 | {agreement[p]['equal']}/29 ({agreement[p]['agreement']:.1%}) |")
    lines+=['','## Metrics and errors','```json',json.dumps(summary,indent=2),'```',
      '', 'Latency包含读图、processor、token约束构建、GPU传输、同步生成及解码，不含模型加载/warmup、结果写盘。Qwen相比旧Molmo额外含结构化约束开销。GPU allocated包含权重，reserved是PyTorch池；nvidia-smi进程/全设备峰值每0.5s采样，全设备含其他进程。',
      '仅一套冻结prompt和单次确定性执行；S2仅一张，标签来自已有graph/matcher而非人工真值。predicate agreement是与历史GPT输出的一致性，不是人工准确率。',
      '', '## Reproduction','```bash','HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /home/sja/miniconda3/envs/molmo/bin/python -m cloth_agent.experiments.visual_program_inheritance_qwen','```',
      '', '官方模型及kernel来源：[Qwen model card](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)、[FlashAttention2 kernel](https://huggingface.co/kernels-community/flash-attn2)、[LM Format Enforcer](https://github.com/noamgat/lm-format-enforcer)。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n');print(json.dumps(stat),flush=True)

def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--attention',default='kernels-community/flash-attn2@c3f3c93a4a39e3d2cf49c13eac49229bc726ea13');args=ap.parse_args()
    import torch,transformers,importlib.metadata
    from transformers import Qwen3VLForConditionalGeneration,AutoProcessor
    from lmformatenforcer import RegexParser
    from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn,build_token_enforcer_tokenizer_data
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'manifest.json').exists():raise RuntimeError('Existing run; refuse overwriting or selective test reruns')
    prior=load(PREVIOUS/'manifest.json');queries=prior['queries'];program=load(PREVIOUS/'visual_program.json');prompts=prior['prompts']
    for r in queries:assert sha(r['image'])==r['file_sha256']
    assert sha(PREVIOUS/'visual_program.json')==sha(SOURCE/'visual_program.json')==prior['program_sha256']
    (OUT/'visual_program.json').write_bytes((PREVIOUS/'visual_program.json').read_bytes())
    manifest=dict(model='Qwen/Qwen3-VL-8B-Instruct',revision=REVISION,checkpoint=str(CHECKPOINT),dtype='bfloat16',quantization=None,attention_implementation=args.attention,
        queries=queries,prompts=prompts,grammars=GRAMMARS,program_sha256=sha(OUT/'visual_program.json'),do_sample=False,max_new_tokens=384,seed=0,
        processor='Official AutoProcessor defaults; original RGB image; no manual resize or ROI',device=torch.cuda.get_device_name(),torch=torch.__version__,transformers=transformers.__version__,
        packages={k:importlib.metadata.version(k) for k in ['lm-format-enforcer','kernels','huggingface-hub']},
        implementation_sha256=sha(__file__),previous_manifest_sha256=sha(PREVIOUS/'manifest.json'),gpt_calls=0,training=False,physical_execution=False,parse_errors='Explicit PARSE_ERROR, never UNKNOWN',frozen_unix_s=time.time())
    save(OUT/'manifest.json',manifest);(OUT/'frozen_implementation.py').write_bytes(Path(__file__).read_bytes())
    monitor=MemoryMonitor();monitor.start();torch.manual_seed(0);torch.cuda.manual_seed_all(0);started=time.perf_counter()
    model=Qwen3VLForConditionalGeneration.from_pretrained(str(CHECKPOINT),dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation=args.attention,local_files_only=True).eval()
    processor=AutoProcessor.from_pretrained(str(CHECKPOINT),local_files_only=True)
    tokenizer_data=build_token_enforcer_tokenizer_data(processor.tokenizer)
    save(OUT/'model_load.json',dict(elapsed_s=time.perf_counter()-started,parameters=sum(p.numel() for p in model.parameters()),device_map=str(model.hf_device_map),config=model.config.to_dict(),processor=processor.to_dict()))
    def infer(path,method):
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter();im=Image.open(path).convert('RGB')
        msgs=[dict(role='user',content=[dict(type='text',text=prompts[method]),dict(type='image',image=im)])]
        inputs=processor.apply_chat_template(msgs,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt').to('cuda')
        allowed=build_transformers_prefix_allowed_tokens_fn(tokenizer_data,RegexParser(GRAMMARS[method]))
        with torch.inference_mode():output=model.generate(**inputs,prefix_allowed_tokens_fn=allowed,max_new_tokens=384,do_sample=False)
        gen=output[:,inputs['input_ids'].shape[-1]:];text=processor.batch_decode(gen,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        return dict(text=text,elapsed_s=elapsed,output_tokens=int(gen.size(1)),input_tokens=int(inputs['input_ids'].shape[-1]),image_grid_thw=inputs['image_grid_thw'].tolist(),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
    warmup=load(SOURCE/'manifest.json')['representatives'][0]['path']
    for method in prompts:
        result=infer(warmup,method);result.update(classify(result['text'],method,program));save(OUT/'warmup'/f'{method}.json',result)
        if result['status']!='OK':raise RuntimeError('Warmup structured decoding failed; no test started')
    results={k:[] for k in prompts}
    for i,r in enumerate(queries):
        for method in (['zero_shot','inheritance'] if i%2==0 else ['inheritance','zero_shot']):
            start=time.perf_counter()
            try:raw=infer(r['image'],method);raw.update(classify(raw['text'],method,program))
            except Exception as exc:
                raw=dict(text=None,elapsed_s=time.perf_counter()-start,prediction='INFERENCE_ERROR',predicates=None,status='INFERENCE_ERROR',error=type(exc).__name__+': '+str(exc),output_tokens=0,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
                torch.cuda.empty_cache()
            raw.update(sample_id=r['sample_id'],image_sha256=r['file_sha256']);save(OUT/'audit'/r['sample_id']/f'{method}.json',raw)
            row={k:raw[k] for k in ['sample_id','prediction','status','error','elapsed_s','peak_allocated_mib','peak_reserved_mib','output_tokens']};row.update(label=r['node_label'],original_node=r['original_node_label'])
            if method=='inheritance':
                for p in PIDS:
                    row[p]=raw['predicates'][p]['value'] if raw['predicates'] else raw['status'];row[p+'_confidence']=raw['predicates'][p]['confidence'] if raw['predicates'] else None
            results[method].append(row)
            with (OUT/f'predictions_{method}.csv').open('w') as f:
                writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader();writer.writerows(results[method])
            print(method,r['sample_id'],row['prediction'],round(row['elapsed_s'],2),flush=True)
    memory=monitor.finish();save(OUT/'memory.json',memory)
    assert sha(OUT/'visual_program.json')==sha(SOURCE/'visual_program.json')==manifest['program_sha256']
    report(results,memory)
if __name__=='__main__':main()
