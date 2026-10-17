"""QLoRA atomic-answer training. No held-out reads until eval stage after best freeze."""
import argparse,csv,json,math,random,time,statistics,hashlib
from pathlib import Path
from collections import Counter,defaultdict
from PIL import Image
from .qwen_cloth_distillation import OUT,readl,jsonl
from .visual_program_inheritance import load,MemoryMonitor,PIDS
from .visual_program_inheritance_qwen import CHECKPOINT,REVISION,ENTRY
from .qwen_predicate_diagnosis import parse
from .visual_decision_program import save,sha
ATTN='kernels-community/flash-attn2@c3f3c93a4a39e3d2cf49c13eac49229bc726ea13'

def config():
    p=OUT/'training_config.json'
    c=dict(model='Qwen/Qwen3-VL-8B-Instruct',revision=REVISION,checkpoint=str(CHECKPOINT),method='QLoRA',quantization='NF4 double quantization, BF16 compute',r=16,lora_alpha=32,lora_dropout=.05,target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],vision_encoder='frozen BF16; language modules only adapted',epochs=2,batch_size=1,gradient_accumulation=8,learning_rate=1e-4,weight_decay=.01,max_grad_norm=1.0,warmup_ratio=.05,optimizer='torch AdamW adapter parameters only',seed=20261014,processor='unchanged checkpoint defaults; no image resize/rotation/crop override',attention_implementation=ATTN,gradient_checkpointing=True,confidence_cutoff=.85,sampling='inverse train frequency of predicate x YES/NO; replacement; draws/epoch equal retained train QA count',loss='Only assistant answer-value tokens supervised (YES/NO); prompt/image/JSON/confidence tokens masked. Single-predicate JSON envelope matches original prompt.',validation='20260923 day; natural distribution; best epoch by macro predicate/value CE; no test reads during training',evaluation='Base BF16 vs BF16 base+best adapter, unchanged atomic-original prompts and deterministic grammar, all 29 only after frozen best',max_epochs_after_test=0,created_unix_s=time.time())
    if p.exists():return load(p)
    save(p,c);return c

def messages(image,prompt):return [dict(role='user',content=[dict(type='text',text=prompt),dict(type='image',image=image)])]

def encode(processor,row):
    import torch
    im=Image.open(row['image']).convert('RGB');msg=messages(im,row['prompt'])
    inputs=processor.apply_chat_template(msg,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt')
    # Prompt identical to evaluation. Append an atomic JSON answer; only value token learns.
    answer=json.dumps(dict(value=row['value'],confidence=0.0),separators=(',',':'))+'<|im_end|>'
    tok=processor.tokenizer(answer,add_special_tokens=False,return_offsets_mapping=True)
    aids=torch.tensor([tok['input_ids']],dtype=torch.long);labels=torch.full_like(aids,-100)
    start=answer.index(row['value']);end=start+len(row['value'])
    for i,(a,b) in enumerate(tok['offset_mapping']):
        if a<end and b>start:labels[0,i]=aids[0,i]
    assert (labels!=-100).any()
    plen=inputs['input_ids'].shape[-1];inputs['input_ids']=torch.cat([inputs['input_ids'],aids],-1)
    inputs['attention_mask']=torch.ones_like(inputs['input_ids'])
    inputs['labels']=torch.cat([torch.full((1,plen),-100,dtype=torch.long),labels],-1)
    return inputs

def train(smoke=False):
    import torch
    from transformers import Qwen3VLForConditionalGeneration,AutoProcessor,BitsAndBytesConfig
    from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training
    c=config();random.seed(c['seed']);torch.manual_seed(c['seed']);monitor=MemoryMonitor();monitor.start()
    if (OUT/'best_checkpoint.json').exists():raise RuntimeError('Training already frozen; no post-test retraining')
    if smoke:
        manifest=load(OUT/'manifest.json');rows=[]
        for r in readl(OUT/'teacher_labels.jsonl'):
            for p,d in r['predicates'].items():
                if d['value'] in ('YES','NO') and d['confidence']>=.85:
                    rows.append(dict(**{k:r[k] for k in ['sample_id','image','file_sha256','run','split']},predicate=p,value=d['value'],prompt=manifest['atomic_prompts'][p]))
    else:rows=readl(OUT/'atomic_samples.jsonl')
    tr=[r for r in rows if r['split']=='train'];va=[r for r in rows if r['split']=='val']
    assert tr and (smoke or va) and not {r['run'] for r in tr}&{r['run'] for r in va}
    for r in rows:assert sha(r['image'])==r['file_sha256']
    processor=AutoProcessor.from_pretrained(str(CHECKPOINT),local_files_only=True)
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16,llm_int8_skip_modules=['visual','lm_head'])
    model=Qwen3VLForConditionalGeneration.from_pretrained(str(CHECKPOINT),dtype=torch.bfloat16,quantization_config=quant,attn_implementation=ATTN,device_map={'':'cuda:0'},local_files_only=True)
    model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False})
    # prepare_model_for_kbit_training upcasts nonquantized weights; keep frozen vision BF16.
    model.model.visual.to(dtype=torch.bfloat16)
    model=get_peft_model(model,LoraConfig(r=c['r'],lora_alpha=c['lora_alpha'],lora_dropout=c['lora_dropout'],target_modules=c['target_modules'],bias='none',task_type='CAUSAL_LM'))
    assert all('lora_' in n and 'visual' not in n for n,p in model.named_parameters() if p.requires_grad)
    model.config.use_cache=False
    save(OUT/('smoke_model.json' if smoke else 'train_model.json'),dict(trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),total=sum(p.numel() for p in model.parameters()),processor=processor.image_processor.to_dict(),train_samples=len(tr),val_samples=len(va)))
    model.print_trainable_parameters();optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=c['learning_rate'],weight_decay=c['weight_decay'])
    def loss(row):
        inputs=encode(processor,row).to('cuda')
        with torch.autocast('cuda',dtype=torch.bfloat16):out=model(**inputs)
        v=out.loss
        if not torch.isfinite(v):raise RuntimeError('Nonfinite loss')
        return v
    if smoke:
        model.train();v=loss(tr[0]);v.backward();grad=sum(float(p.grad.abs().sum()) for p in model.parameters() if p.requires_grad and p.grad is not None);assert grad>0
        save(OUT/'smoke_result.json',dict(loss=v.item(),nonzero_adapter_gradient=grad,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,memory=monitor.finish()));print('SMOKE OK',v.item(),flush=True);return
    log=[];step=0;best=float('inf');total_updates=math.ceil(len(tr)/c['gradient_accumulation'])*c['epochs'];warmup=max(1,int(total_updates*c['warmup_ratio']));start=time.perf_counter()
    def record(**kw):
        row=dict(epoch=kw.get('epoch'),step=step,phase=kw['phase'],loss=kw.get('loss'),macro_value_loss=kw.get('macro_value_loss'),lr=optimizer.param_groups[0]['lr'],elapsed_s=time.perf_counter()-start,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20);log.append(row)
        with (OUT/'training_log.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerows(log)
        print(json.dumps(row),flush=True)
    def validate(epoch):
        model.eval();ls=[];groups=defaultdict(list)
        with torch.no_grad():
            for r in va:
                v=loss(r).item();ls.append(v);groups[r['predicate']+'_'+r['value']].append(v)
        macro=statistics.mean(statistics.mean(x) for x in groups.values());record(epoch=epoch,phase='val',loss=statistics.mean(ls),macro_value_loss=macro);save(OUT/f'validation_epoch_{epoch}.json',{p:dict(n=len(v),loss=statistics.mean(v)) for p,v in groups.items()});return macro
    validate(0)
    for epoch in range(1,c['epochs']+1):
        model.train();weights=torch.tensor([r['sampling_weight'] for r in tr],dtype=torch.double);gen=torch.Generator().manual_seed(c['seed']+epoch);indices=torch.multinomial(weights,len(tr),replacement=True,generator=gen).tolist()
        save(OUT/f'epoch_{epoch}_draws.json',[dict(sample_id=tr[i]['sample_id'],predicate=tr[i]['predicate'],value=tr[i]['value']) for i in indices])
        optimizer.zero_grad(set_to_none=True);losses=[]
        for k,i in enumerate(indices):
            group_start=k//c['gradient_accumulation']*c['gradient_accumulation'];den=min(c['gradient_accumulation'],len(indices)-group_start)
            v=loss(tr[i]);losses.append(v.item());(v/den).backward()
            if (k+1)%c['gradient_accumulation']==0 or k+1==len(indices):
                step+=1;scale=min(step/warmup,max(0.,(total_updates-step)/max(1,total_updates-warmup)))
                for g in optimizer.param_groups:g['lr']=c['learning_rate']*scale
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],c['max_grad_norm']);optimizer.step();optimizer.zero_grad(set_to_none=True)
                record(epoch=epoch,phase='train',loss=statistics.mean(losses));losses=[]
        metric=validate(epoch)
        cp=OUT/f'adapter_epoch_{epoch}';model.save_pretrained(cp,safe_serialization=True)
        if metric<best:best=metric;best_epoch=epoch;best_path=cp
    save(OUT/'best_checkpoint.json',dict(path=str(best_path),epoch=best_epoch,macro_validation_value_loss=best,adapter_sha256=sha(best_path/'adapter_model.safetensors'),config_sha256=sha(OUT/'training_config.json'),samples_sha256=sha(OUT/'atomic_samples.jsonl'),frozen_unix_s=time.time()))
    save(OUT/'training_memory.json',monitor.finish());print('BEST FROZEN',best_path,flush=True)

def evaluate():
    import torch
    from transformers import Qwen3VLForConditionalGeneration,AutoProcessor
    from peft import PeftModel
    from lmformatenforcer import RegexParser
    from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn,build_token_enforcer_tokenizer_data
    # The only stage reading held-out reference labels; requires frozen trained checkpoint.
    best=load(OUT/'best_checkpoint.json');assert sha(Path(best['path'])/'adapter_model.safetensors')==best['adapter_sha256']
    if (OUT/'evaluation.csv').exists():raise RuntimeError('Evaluation already started; do not selectively rerun')
    m=load(OUT/'manifest.json');tm=Path('reports/qwen_predicate_diagnosis/manifest.json');assert sha(tm)==m['test_manifest_sha256'];queries=load(tm)['queries']
    torch.manual_seed(0);monitor=MemoryMonitor();monitor.start();model=Qwen3VLForConditionalGeneration.from_pretrained(str(CHECKPOINT),dtype=torch.bfloat16,attn_implementation=ATTN,device_map={'':'cuda:0'},local_files_only=True).eval();processor=AutoProcessor.from_pretrained(str(CHECKPOINT),local_files_only=True);td=build_token_enforcer_tokenizer_data(processor.tokenizer)
    def infer(r,p):
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();beg=time.perf_counter()
        inputs=processor.apply_chat_template(messages(Image.open(r['image']).convert('RGB'),m['atomic_prompts'][p]),tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt').to('cuda')
        allowed=build_transformers_prefix_allowed_tokens_fn(td,RegexParser(ENTRY))
        with torch.inference_mode():o=model.generate(**inputs,prefix_allowed_tokens_fn=allowed,do_sample=False,max_new_tokens=128)
        text=processor.batch_decode(o[:,inputs['input_ids'].shape[-1]:],skip_special_tokens=True,clean_up_tokenization_spaces=False)[0];torch.cuda.synchronize()
        return dict(text=text,elapsed_s=time.perf_counter()-beg,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,image_grid=inputs['image_grid_thw'].tolist(),**parse(text))
    rows=[]
    for method in ['base','fine_tuned']:
        if method=='fine_tuned':model=PeftModel.from_pretrained(model,best['path'],is_trainable=False).eval()
        warm=m['images'][0];save(OUT/'eval_warmup'/f'{method}.json',infer(warm,PIDS[0]))
        for r in queries:
            assert sha(r['image'])==r['file_sha256']
            refs={x['predicate_id']:x['value'] for x in load(Path('reports/visual_program_S1_S2/audit/audit_images')/r['sample_id']/'predicate_results.json')['results']}
            for p in PIDS:
                try:v=infer(r,p)
                except Exception as e:v=dict(text=None,elapsed_s=None,peak_allocated_mib=None,value='INFERENCE_ERROR',confidence=None,status='INFERENCE_ERROR',error=str(e))
                save(OUT/'eval_raw'/method/r['sample_id']/f'{p}.json',v)
                row=dict(method=method,sample_id=r['sample_id'],node_label=r['node_label'],original_node=r['original_node_label'],predicate=p,value=v['value'],confidence=v['confidence'],reference=refs[p],correct=v['value']==refs[p],status=v['status'],elapsed_s=v['elapsed_s'],peak_allocated_mib=v['peak_allocated_mib']);rows.append(row)
                with (OUT/'evaluation.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerows(rows)
            print('EVAL',method,r['sample_id'],flush=True)
    save(OUT/'evaluation_memory.json',monitor.finish())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['config','smoke','train','eval']);a=p.parse_args()
    if a.stage=='config':config()
    elif a.stage=='eval':evaluate()
    else:train(smoke=a.stage=='smoke')
