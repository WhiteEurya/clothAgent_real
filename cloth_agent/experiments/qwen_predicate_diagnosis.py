"""Atomic predicate diagnosis with frozen checkpoint and archived orientation."""
import csv,json,math,re,time,statistics,hashlib
from pathlib import Path
from collections import Counter
from PIL import Image
from .visual_decision_program import save,sha
from .visual_program_inheritance import load,SIDE,MemoryMonitor
from .visual_program_inheritance_qwen import CHECKPOINT,REVISION,ENTRY
OUT=Path('reports/qwen_predicate_diagnosis')
PREV=Path('reports/visual_program_inheritance/qwen3_vl_8b_instruct')
SOURCE=Path('reports/visual_program_S1_S2')
UNIFIED=Path('reports/unified_visual_representation_e001')
PIDS=['left_inward_flap','right_extended','torso_open']
CONDITIONS=['A_atomic_original','B_atomic_simple','C_canonical_frame']
SIMPLE=dict(left_inward_flap='Is most of the left sleeve lying over or inside the upper torso area?',right_extended='Is the right sleeve still visibly sticking out sideways beyond the torso?',torso_open='Is the torso still broadly spread out rather than folded inward?')
PREFIX=SIDE+'\nAnswer only the following single visual question about the current image. Use UNKNOWN if visible evidence is insufficient. Do not classify a node or provide reasoning. Return exactly {"value":"YES|NO|UNKNOWN","confidence":0.0}, with confidence from 0 to 1.\nQuestion: '

def angle(collar,hem):
    dx=hem[0]-collar[0];dy=hem[1]-collar[1]
    if math.hypot(dx,dy)<1:raise ValueError('Degenerate garment axis')
    return math.degrees(math.atan2(-dx,dy))

def parse(text):
    try:
        if re.fullmatch(ENTRY,text) is None:raise ValueError('Output violates constrained JSON grammar')
        value=json.loads(text)
        if set(value)!={'value','confidence'} or value['value'] not in ('YES','NO','UNKNOWN') or type(value['confidence']) not in (int,float) or not 0<=value['confidence']<=1:raise ValueError('Invalid fields/value/confidence')
        return dict(**value,status='OK',error=None)
    except Exception as e:return dict(value='PARSE_ERROR',confidence=None,status='PARSE_ERROR',error=str(e))

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'manifest.json').exists():raise RuntimeError('Refuse to overwrite an initialized diagnosis experiment')
    previous=load(PREV/'manifest.json');queries=previous['queries'];program=load(PREV/'visual_program.json')
    assert sha(PREV/'visual_program.json')==sha(SOURCE/'visual_program.json')==previous['program_sha256']
    (OUT/'visual_program.json').write_bytes((SOURCE/'visual_program.json').read_bytes())
    original={p['id']:p['question'] for p in program['predicates'] if p['id'] in PIDS}
    prompts={c:{p:PREFIX+(original[p] if c==CONDITIONS[0] else SIMPLE[p]) for p in PIDS} for c in CONDITIONS}
    orientation=[]
    for row in queries:
        assert sha(row['image'])==row['file_sha256']
        f=UNIFIED/'test'/row['sample_id']/'variables.json';g=load(f)['garment'];im=Image.open(row['image']).convert('RGB')
        if g['collar_anchor'] is not None and g['hem_anchor'] is not None:
            rotation=angle(g['collar_anchor'],g['hem_anchor']);mode='archived_collar_hem_axis';source=str(f);source_sha=sha(f);evidence=None
        else:
            assert row['sample_id']=='obs_26602fe6c9d46643'
            f=SOURCE/'audit/audit_images'/row['sample_id']/'localize/parsed.json'
            d=load(f);evidence=next(r['visual_evidence'] for r in d['regions'] if r['region_id']=='torso')
            assert 'hem' in evidence and 'top' in evidence and 'below' in evidence
            rotation=180.;mode='archived_hem_top_shoulders_below_collar_cropped';source=str(f);source_sha=sha(f)
        dest=OUT/'canonical_images'/f"{row['sample_id']}.png";dest.parent.mkdir(exist_ok=True)
        im.rotate(rotation,resample=Image.Resampling.BICUBIC,expand=True,fillcolor=(0,0,0)).save(dest)
        orientation.append(dict(sample_id=row['sample_id'],angle_deg=rotation,source=source,source_sha256=source_sha,source_mode=mode,source_evidence=evidence,archived_garment=g,original_size=list(im.size),canonical_size=list(Image.open(dest).size),canonical_path=str(dest),canonical_sha256=sha(dest),mirrored=False))
    save(OUT/'orientation.json',orientation)
    manifest=dict(model=previous['model'],revision=REVISION,checkpoint=str(CHECKPOINT),dtype='bfloat16',attention_implementation=previous['attention_implementation'],queries=queries,predicates=PIDS,conditions=CONDITIONS,prompts=prompts,grammar=ENTRY,program_sha256=sha(OUT/'visual_program.json'),orientation_sha256=sha(OUT/'orientation.json'),implementation_sha256=sha(__file__),frozen_unix_s=time.time(),do_sample=False,max_new_tokens=128,seed=0,ordering='Rotate the 9 condition/predicate jobs by image index; fresh chat per call',calls=261,reference='Existing GPT-6 audit predicate outputs only, no new GPT calls',parse_failure='PARSE_ERROR or INFERENCE_ERROR separately; never UNKNOWN',canonical='PIL positive CCW rotation, expand=True, bicubic, black fill, no mirroring; original C image archived',limitations=['A versus bundled also removes full-program context/rules, not solely predicate count','B wording broadens/simplifies original predicates, notably removing torso print requirement','C entails resampling/padding and native processor scaling differences','One OTHER collar cropped; C uses archived coarse 180-degree orientation','Historical orientation is estimated, not human ground truth; no new orientation model calls'])
    save(OUT/'manifest.json',manifest);(OUT/'frozen_implementation.py').write_bytes(Path(__file__).read_bytes())
    return manifest,orientation

def summarize(rows):
    result={};paired={}
    for c in CONDITIONS:
        subset=[r for r in rows if r['condition']==c];s={}
        for p in PIDS:
            group=[r for r in subset if r['predicate']==p];known=[r for r in group if r['label'] in ('S1','S2')]
            s[p]=dict(equal=sum(r['value']==r['gpt_reference'] for r in group),total=len(group),known_equal=sum(r['value']==r['gpt_reference'] for r in known),known_total=len(known),unknown=sum(r['value']=='UNKNOWN' for r in group),errors=sum(r['status']!='OK' for r in group),mean_latency_s=statistics.mean(r['elapsed_s'] for r in group),median_latency_s=statistics.median(r['elapsed_s'] for r in group))
        per_image=[sum(r['elapsed_s'] for r in subset if r['sample_id']==sid) for sid in {r['sample_id'] for r in subset}]
        result[c]=dict(predicates=s,mean_call_s=statistics.mean(r['elapsed_s'] for r in subset),median_call_s=statistics.median(r['elapsed_s'] for r in subset),mean_three_calls_per_image_s=statistics.mean(per_image),median_three_calls_per_image_s=statistics.median(per_image),peak_allocated_mib=max(r['peak_allocated_mib'] for r in subset))
    lookup={(r['condition'],r['predicate'],r['sample_id']):r for r in rows}
    for prev,cur in [('bundled',CONDITIONS[0]),(CONDITIONS[0],CONDITIONS[1]),(CONDITIONS[1],CONDITIONS[2])]:
        key=prev+' -> '+cur;paired[key]={}
        for p in PIDS:
            improved=[];worsened=[];before_equal=after_equal=0
            for r in [x for x in rows if x['condition']==cur and x['predicate']==p]:
                before=r['bundled_value'] if prev=='bundled' else lookup[(prev,p,r['sample_id'])]['value']
                b=before==r['gpt_reference'];a=r['value']==r['gpt_reference'];before_equal+=b;after_equal+=a
                if a and not b:improved.append(r['sample_id'])
                if b and not a:worsened.append(r['sample_id'])
            n=len(improved)+len(worsened);k=min(len(improved),len(worsened));pv=min(1.,2*sum(math.comb(n,i) for i in range(k+1))/2**n) if n else 1.
            paired[key][p]=dict(before_equal=before_equal,after_equal=after_equal,improved=improved,worsened=worsened,mcnemar_exact_two_sided_p=pv,p_note='Exploratory paired disagreement test; unadjusted for multiple comparisons')
    special={sid:{c:{p:lookup[c,p,sid]['value'] for p in PIDS} for c in CONDITIONS} for sid in ['obs_7f66d64e53de5d54','obs_cb681fd8ddcdf275']}
    bundled={p:dict(equal=sum(r['bundled_value']==r['gpt_reference'] for r in rows if r['condition']==CONDITIONS[0] and r['predicate']==p),known_equal=sum(r['bundled_value']==r['gpt_reference'] for r in rows if r['condition']==CONDITIONS[0] and r['predicate']==p and r['label'] in ('S1','S2'))) for p in PIDS}
    save(OUT/'summary.json',dict(conditions=result,bundled=bundled,paired=paired,special=special))
    lines=['# Qwen3-VL predicate failure diagnosis','','同一Instruct checkpoint、同一29张held-out；无训练、无GPT调用、原visual program只读。每次仅输出一个predicate的YES/NO/UNKNOWN+confidence，token级JSON约束，解析失败不改成UNKNOWN。A问原问题，B问预先固定的简单问题，C仅将B图片按已有衣物轴旋正。',
      '', '| Condition | left_inward_flap /29 | right_extended /29 | torso_open /29 | mean / median call(s) | mean three calls/image(s) |','|---|---:|---:|---:|---|---:|',
      f"| Archived bundled | {bundled[PIDS[0]]['equal']} | {bundled[PIDS[1]]['equal']} | {bundled[PIDS[2]]['equal']} | 2.60 / 2.61 (four predicates together) | — |"]
    for c,s in result.items():lines.append(f"| {c} | {s['predicates'][PIDS[0]]['equal']} | {s['predicates'][PIDS[1]]['equal']} | {s['predicates'][PIDS[2]]['equal']} | {s['mean_call_s']:.3f} / {s['median_call_s']:.3f} | {s['mean_three_calls_per_image_s']:.3f} |")
    lines+=['','## 11张S1/S2：与GPT历史predicate的一致数','| Condition | left /11 | right /11 | torso /11 |','|---|---:|---:|---:|',f"| Archived bundled | {bundled[PIDS[0]]['known_equal']} | {bundled[PIDS[1]]['known_equal']} | {bundled[PIDS[2]]['known_equal']} |"]
    for c,s in result.items():lines.append(f"| {c} | {s['predicates'][PIDS[0]]['known_equal']} | {s['predicates'][PIDS[1]]['known_equal']} | {s['predicates'][PIDS[2]]['known_equal']} |")
    lines+=['','这是predicate一致率，不是节点accuracy：本实验未测right_rounded_cuff，不能从三个predicate独立计算完整原rules。','','## Hard S2 / S3','```json',json.dumps(special,indent=2),'```','','## 完整指标与配对变化','```json',json.dumps(dict(conditions=result,paired=paired),indent=2),'```',
      '', '### 实验边界', 'A相对旧bundled同时减少输出项、移除了其他问题和完整规则上下文；因此改善支持bundled-context干扰，但不能严格归因于predicate数量。B简化了语义，尤其torso问题去掉print要求，与旧GPT判据并非完全同义。C增加旋转重采样、padding和processor尺度变化。历史GPT答案不是人工真值。',
      '28张用历史collar/hem轴，1张OTHER collar被裁切，以既有定位记录的hem-top/shoulder-below关系使用180°；完整出处、角度、坐标、hash见orientation.json，旋转图保存在canonical_images。',
      '单调用耗时含读图、processor、结构化约束、GPU传输、同步生成和解码；不含一次性的离线旋转、模型加载及3次seed warmup。原始三组输出逐条保存在audit目录。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n');print(json.dumps(result),flush=True)

def main():
    import torch,transformers
    from transformers import Qwen3VLForConditionalGeneration,AutoProcessor
    from lmformatenforcer import RegexParser
    from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn,build_token_enforcer_tokenizer_data
    manifest,orientation=prepare();byid={r['sample_id']:r for r in orientation};torch.manual_seed(0);torch.cuda.manual_seed_all(0);monitor=MemoryMonitor();monitor.start();t=time.perf_counter()
    model=Qwen3VLForConditionalGeneration.from_pretrained(str(CHECKPOINT),dtype=torch.bfloat16,attn_implementation=manifest['attention_implementation'],device_map={'':'cuda:0'},local_files_only=True).eval()
    processor=AutoProcessor.from_pretrained(str(CHECKPOINT),local_files_only=True);tokenizer_data=build_token_enforcer_tokenizer_data(processor.tokenizer)
    save(OUT/'model_load.json',dict(elapsed_s=time.perf_counter()-t,torch=torch.__version__,transformers=transformers.__version__,device=torch.cuda.get_device_name(),device_map=str(model.hf_device_map)))
    def infer(path,prompt):
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter();im=Image.open(path).convert('RGB')
        messages=[dict(role='user',content=[dict(type='text',text=prompt),dict(type='image',image=im)])]
        inputs=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt').to('cuda')
        allowed=build_transformers_prefix_allowed_tokens_fn(tokenizer_data,RegexParser(ENTRY))
        with torch.inference_mode():output=model.generate(**inputs,prefix_allowed_tokens_fn=allowed,max_new_tokens=128,do_sample=False)
        generated=output[:,inputs['input_ids'].shape[-1]:];text=processor.batch_decode(generated,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0];torch.cuda.synchronize()
        return dict(text=text,elapsed_s=time.perf_counter()-t,output_tokens=int(generated.shape[-1]),image_grid=inputs['image_grid_thw'].tolist(),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,**parse(text))
    seed=Path(load(SOURCE/'manifest.json')['representatives'][0]['path']);g=load(UNIFIED/'train/obs_1a16fe194725defa/variables.json')['garment'];rot=OUT/'warmup/canonical.png';rot.parent.mkdir(exist_ok=True);Image.open(seed).convert('RGB').rotate(angle(g['collar_anchor'],g['hem_anchor']),resample=Image.Resampling.BICUBIC,expand=True,fillcolor=(0,0,0)).save(rot)
    for c in CONDITIONS:
        v=infer(rot if c==CONDITIONS[2] else seed,manifest['prompts'][c][PIDS[0]]);save(OUT/'warmup'/f'{c}.json',v)
        assert v['status']=='OK'
    rows=[];jobs=[(c,p) for c in CONDITIONS for p in PIDS]
    for i,r in enumerate(manifest['queries']):
        references={v['predicate_id']:v['value'] for v in load(SOURCE/'audit/audit_images'/r['sample_id']/'predicate_results.json')['results']};bundle=load(PREV/'audit'/r['sample_id']/'inheritance.json')['predicates']
        for c,p in jobs[i%len(jobs):]+jobs[:i%len(jobs)]:
            path=byid[r['sample_id']]['canonical_path'] if c==CONDITIONS[2] else r['image'];t=time.perf_counter()
            try:v=infer(path,manifest['prompts'][c][p])
            except Exception as e:
                v=dict(text=None,value='INFERENCE_ERROR',confidence=None,status='INFERENCE_ERROR',error=type(e).__name__+': '+str(e),elapsed_s=time.perf_counter()-t,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20);torch.cuda.empty_cache()
            save(OUT/'audit'/r['sample_id']/f'{c}_{p}.json',dict(**v,input_image=str(path),input_sha256=sha(path),prompt=manifest['prompts'][c][p]))
            row=dict(sample_id=r['sample_id'],label=r['node_label'],original_node=r['original_node_label'],condition=c,predicate=p,value=v['value'],confidence=v['confidence'],status=v['status'],error=v['error'],gpt_reference=references[p],bundled_value=bundle[p]['value'],elapsed_s=v['elapsed_s'],peak_allocated_mib=v['peak_allocated_mib']);rows.append(row)
            with (OUT/'predictions.csv').open('w') as f:
                wr=csv.DictWriter(f,fieldnames=list(row));wr.writeheader();wr.writerows(rows)
        print(f"{i+1}/29 {r['sample_id']} completed",flush=True)
    save(OUT/'memory.json',monitor.finish());assert sha(OUT/'visual_program.json')==sha(SOURCE/'visual_program.json')==manifest['program_sha256'];summarize(rows)
if __name__=='__main__':main()
