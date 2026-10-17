"""Frozen local small-VLM comparison. No GPT invocation, training or robot runtime."""
import csv,json,hashlib,statistics,time,re,os,threading,subprocess
from pathlib import Path
from PIL import Image
from .visual_decision_program import save,sha,decide
OUT=Path('reports/visual_program_inheritance')
SOURCE=Path('reports/visual_program_S1_S2')
CHECKPOINT=Path('/home/sja/.cache/huggingface/hub/models--allenai--MolmoPoint-8B/snapshots/188130f961c8e0888a34e11121a1423c461a01ba')
PIDS=['left_inward_flap','right_extended','right_rounded_cuff','torso_open']
SIDE='Mentally rotate the garment so the collar is above the hem, without mirroring. Left/right mean viewer sides in that orientation, not in the unrotated image.'
ZERO=SIDE+'''\nClassify the current image using these state definitions:
S1: Left sleeve forms an inward flap on the upper torso; right sleeve remains extended. Torso and hem retain a broad open footprint.
S2: Left sleeve remains inward; right sleeve forms a rounded visibly open cuff near the outer shoulder. Torso remains broad with the central print exposed.
UNKNOWN: Neither definition is adequately supported, or evidence is uncertain. Other garment states are possible.
Return exactly one label: S1, S2, or UNKNOWN. No explanation.'''
INHERIT=SIDE+'''\nExecute the following frozen visual program on the current image. Do not classify a node, choose an action, or change any predicate or rule.
Answer each of the four existing predicate questions independently. Values must be YES, NO or UNKNOWN; use UNKNOWN when visible evidence is insufficient. Confidence is a number from 0 to 1.
Only the Host applies the rules. Return ONLY a JSON object with exactly the four predicate IDs as keys; each value is {"value":"YES|NO|UNKNOWN","confidence":0.0}. Do not output a node label or additional text.
Frozen program:\n'''

def load(p):return json.loads(Path(p).read_text())
def parse(text,method,program):
    if method=='zero_shot':
        if text.strip() not in ('S1','S2','UNKNOWN'):raise ValueError('Expected exact node label')
        return dict(prediction=text.strip(),predicates=None)
    raw=text.strip()
    if raw.startswith('```'):raw=re.sub(r'^```(?:json)?\s*|\s*```$','',raw)
    data=json.loads(raw)
    if not isinstance(data,dict) or set(data)!=set(PIDS):raise ValueError('Missing or additional output keys')
    for p in PIDS:
        d=data[p]
        if not isinstance(d,dict) or set(d)!={'value','confidence'}:raise ValueError('Invalid predicate fields')
        if d['value'] not in ('YES','NO','UNKNOWN'):raise ValueError('Invalid predicate value')
        if type(d['confidence']) not in (int,float) or not 0<=d['confidence']<=1:raise ValueError('Invalid confidence')
    decision=decide(program,[dict(predicate_id=p,value=data[p]['value']) for p in PIDS])
    return dict(prediction=decision['decision'],predicates=data,host_decision=decision)

class MemoryMonitor:
    def __init__(self):self.peak_process_mib=0;self.peak_device_mib=0;self.stop=threading.Event()
    def start(self):
        def poll():
            while not self.stop.is_set():
                try:
                    s=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=3)
                    for row in s.splitlines():
                        pid,mem=row.split(',')
                        if int(pid.strip())==os.getpid():self.peak_process_mib=max(self.peak_process_mib,int(mem.strip()))
                    s=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True,timeout=3)
                    self.peak_device_mib=max(self.peak_device_mib,int(s.strip().splitlines()[0]))
                except Exception:pass
                self.stop.wait(.5)
        self.thread=threading.Thread(target=poll,daemon=True);self.thread.start()
    def finish(self):self.stop.set();self.thread.join();return dict(sampled_process_peak_mib=self.peak_process_mib,sampled_device_peak_mib=self.peak_device_mib,sampling_interval_s=.5)

def main():
    import torch,transformers
    from transformers import AutoProcessor,AutoModelForImageTextToText
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'manifest.json').exists():raise RuntimeError('Experiment already initialized; do not overwrite or selectively rerun test outputs.')
    source=load(SOURCE/'audit/manifest.json');queries=source['queries'];program=load(SOURCE/'visual_program.json')
    assert sha(SOURCE/'visual_program.json')==load(SOURCE/'program_freeze.json')['sha256']
    for r in queries:assert sha(r['image'])==r['file_sha256']
    (OUT/'visual_program.json').write_bytes((SOURCE/'visual_program.json').read_bytes())
    prompts=dict(zero_shot=ZERO,inheritance=INHERIT+json.dumps(program,ensure_ascii=False))
    manifest=dict(model='allenai/MolmoPoint-8B',checkpoint=str(CHECKPOINT),revision=CHECKPOINT.name,queries=queries,prompts=prompts,
        program_sha256=sha(OUT/'visual_program.json'),dtype='bfloat16',quantization=None,max_crops=4,max_new_tokens=384,do_sample=False,seed=0,
        torch=torch.__version__,transformers=transformers.__version__,device=torch.cuda.get_device_name(),
        inference_only=True,gpt_calls=0,training=False,physical_execution=False,confidence_policy='Recorded only, no threshold or test calibration',
        ordering='Alternating A/B order by image index; independent fresh chat each request',warmup='One seed representative per method; excluded from latency and accuracy; no prompt tuning',
        latency='Image open + preprocessing + transfer + synchronized generation + decode; excludes model load and metrics/serialization',
        parse_error_policy='No repair/retry; record format error and operational UNKNOWN separately',
        implementation_sha256=sha(__file__),checkpoint_files={p.name:dict(size=p.stat().st_size,blob=p.resolve().name) for p in CHECKPOINT.iterdir()})
    save(OUT/'manifest.json',manifest);(OUT/'frozen_implementation.py').write_bytes(Path(__file__).read_bytes())
    torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    monitor=MemoryMonitor();monitor.start();started=time.perf_counter()
    model=AutoModelForImageTextToText.from_pretrained(str(CHECKPOINT),trust_remote_code=True,dtype=torch.bfloat16,device_map={'':'cuda:0'},local_files_only=True).eval()
    processor=AutoProcessor.from_pretrained(str(CHECKPOINT),trust_remote_code=True,padding_side='left',local_files_only=True)
    save(OUT/'model_load.json',dict(elapsed_s=time.perf_counter()-started,parameters=sum(p.numel() for p in model.parameters()),device_map=str(model.hf_device_map)))
    def infer(image,prompt):
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        im=Image.open(image).convert('RGB')
        msgs=[dict(role='user',content=[dict(type='text',text=prompt),dict(type='image',image=im)])]
        inputs=processor.apply_chat_template(msgs,tokenize=True,add_generation_prompt=True,return_tensors='pt',return_dict=True,padding=True,return_pointing_metadata=True,images_kwargs={'max_crops':4})
        inputs.pop('metadata');inputs={k:v.to('cuda') if hasattr(v,'to') else v for k,v in inputs.items()}
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            output=model.generate(**inputs,logits_processor=model.build_logit_processor_from_inputs(inputs),max_new_tokens=384,do_sample=False)
        gen=output[:,inputs['input_ids'].size(1):]
        text=processor.post_process_image_text_to_text(gen,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        return dict(text=text,elapsed_s=elapsed,input_tokens=int(inputs['input_ids'].size(1)),output_tokens=int(gen.size(1)),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
    warmup=load(SOURCE/'manifest.json')['representatives'][0]['path']
    for method in prompts:save(OUT/'warmup'/f'{method}.json',infer(warmup,prompts[method]))
    allrows={m:[] for m in prompts}
    for i,r in enumerate(queries):
        methods=['zero_shot','inheritance'] if i%2==0 else ['inheritance','zero_shot']
        for method in methods:
            result=infer(r['image'],prompts[method]);folder=OUT/'audit'/r['sample_id'];folder.mkdir(parents=True,exist_ok=True)
            try:parsed=parse(result['text'],method,program);error=None
            except Exception as exc:parsed=dict(prediction='UNKNOWN',predicates=None);error=str(exc)
            result.update(parsed);result['format_error']=error;result['sample_id']=r['sample_id'];result['image_sha256']=r['file_sha256'];save(folder/f'{method}.json',result)
            row=dict(sample_id=r['sample_id'],label=r['node_label'],original_node=r['original_node_label'],prediction=result['prediction'],format_error=error or '',elapsed_s=result['elapsed_s'],peak_allocated_mib=result['peak_allocated_mib'],peak_reserved_mib=result['peak_reserved_mib'],output_tokens=result['output_tokens'])
            if method=='inheritance':
                for p in PIDS:
                    row[p]=parsed['predicates'][p]['value'] if parsed['predicates'] else 'FORMAT_ERROR'
                    row[p+'_confidence']=parsed['predicates'][p]['confidence'] if parsed['predicates'] else None
            allrows[method].append(row)
            with (OUT/f'predictions_{method}.csv').open('w') as f:
                wr=csv.DictWriter(f,fieldnames=list(row));wr.writeheader();wr.writerows(allrows[method])
            print(method,r['sample_id'],row['prediction'],round(row['elapsed_s'],2),error or '',flush=True)
    memory=monitor.finish();save(OUT/'memory.json',memory)
    assert sha(OUT/'visual_program.json')==sha(SOURCE/'visual_program.json')==manifest['program_sha256']
    summarize(allrows,memory)

def summarize(allrows,memory):
    results={};hard='obs_7f66d64e53de5d54';s3='obs_cb681fd8ddcdf275'
    for method,rows in allrows.items():
        known=[r for r in rows if r['label']!='OTHER'];others=[r for r in rows if r['label']=='OTHER']
        results[method]=dict(known_correct=sum(r['prediction']==r['label'] for r in known),known_total=len(known),unknown=sum(r['prediction']=='UNKNOWN' for r in rows),genuine_unknown=sum(r['prediction']=='UNKNOWN' and not r['format_error'] for r in rows),format_errors=sum(bool(r['format_error']) for r in rows),other_false_accept=sum(r['prediction']!='UNKNOWN' for r in others),other_total=len(others),hard_s2=next(r['prediction'] for r in rows if r['sample_id']==hard),s3_case=next(r['prediction'] for r in rows if r['sample_id']==s3),mean_latency_s=statistics.mean(r['elapsed_s'] for r in rows),median_latency_s=statistics.median(r['elapsed_s'] for r in rows),peak_allocated_mib=max(r['peak_allocated_mib'] for r in rows),peak_reserved_mib=max(r['peak_reserved_mib'] for r in rows))
    agreement={}
    for p in PIDS:
        valid=equal=0
        for r in allrows['inheritance']:
            ref=load(SOURCE/'audit/audit_images'/r['sample_id']/'predicate_results.json')['results'];rv=next(v['value'] for v in ref if v['predicate_id']==p)
            valid+=r[p]!='FORMAT_ERROR';equal+=r[p]==rv
        agreement[p]=dict(equal=equal,total=29,valid=valid,agreement=equal/29)
    zero={r['sample_id']:r for r in allrows['zero_shot']};paired=dict(improved=[],worsened=[])
    for r in allrows['inheritance']:
        z=zero[r['sample_id']];target=r['label'] if r['label']!='OTHER' else 'UNKNOWN'
        if z['prediction']!=target and r['prediction']==target:paired['improved'].append(r['sample_id'])
        if z['prediction']==target and r['prediction']!=target:paired['worsened'].append(r['sample_id'])
    save(OUT/'summary.json',dict(methods=results,predicate_agreement=agreement,paired=paired,memory=memory))
    lines=['# Visual Program Experience Inheritance','','模型：allenai/MolmoPoint-8B；checkpoint revision `'+CHECKPOINT.name+'`，本地缓存，BF16，无量化，RTX 4090。未调用GPT-6、未训练、未修改或执行机器人代码。',
      '同一29张held-out（10 S1、1 S2、18 OTHER）。原program和Host rules逐字节保留；不提供representative images。A只给简短状态定义，B给完整冻结program并仅回答四个predicates。每图每方法一次独立确定性推理，max_crops=4，max_new_tokens=384，顺序交替。',
      '模型加载和每方法一次seed图warmup不计入耗时；耗时含读图、预处理、GPU传输、同步生成和解码。confidence不参与Host阈值筛选。格式错误单独统计，并作为operational UNKNOWN，不重试。',
      '', '| Method | S1/S2 | UNKNOWN | OTHER FA | Hard S2 | S3 case | Latency mean / median |','|---|---:|---:|---:|---|---|---|']
    for method,name in [('zero_shot','Zero-shot small VLM'),('inheritance','+ GPT visual program')]:
        s=results[method];lines.append(f"| {name} | {s['known_correct']}/11 | {s['unknown']} | {s['other_false_accept']}/18 | {s['hard_s2']} | S3→{s['s3_case']} | {s['mean_latency_s']:.2f}s / {s['median_latency_s']:.2f}s |")
    lines+=['| GPT-6 visual program baseline | 11/11 | 17 | 1/18 | correct | S3→S2 | 37.31s / 36.76s |','','## Predicates 与存档 GPT-6 program executor 的一致率','| Predicate | Equal / all | Valid outputs |','|---|---:|---:|']
    for p,s in agreement.items():lines.append(f"| {p} | {s['equal']}/29 ({s['agreement']:.1%}) | {s['valid']}/29 |")
    lines+=['','一致率不是人工标注准确率；GPT基线自身存在S3误接收。格式错误计为不一致。','','## 完整指标','```json',json.dumps(dict(methods=results,paired=paired,memory=memory),indent=2),'```',
      '', '显存：per-method torch peak allocated / reserved含驻留模型权重；nvidia-smi进程峰值是0.5秒采样下界，全设备峰值还含桌面等进程。GPT基线是既有结果，未重新调用。',
      '该checkpoint是项目已有8B视觉理解/定位模型。本实验只测一个checkpoint和一套预先固定提示，不能外推所有小VLM；样本中S2仅1张。训练标签来源为既有graph/matcher，非独立人工真值。',
      '', '复现：`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /home/sja/miniconda3/envs/molmo/bin/python -m cloth_agent.experiments.visual_program_inheritance`。已有输出目录会拒绝覆盖，避免选择性重跑。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n');print(json.dumps(results),flush=True)
if __name__=='__main__':main()
