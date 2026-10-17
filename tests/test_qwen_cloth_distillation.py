import json
from pathlib import Path
from cloth_agent.experiments.qwen_cloth_distillation import OUT,PIDS,load,sha,PROGRAM,TEST
from cloth_agent.experiments.qwen_predicate_diagnosis import PREFIX

def test_split_and_quarantine():
 m=load(OUT/'manifest.json');rows=m['images'];tests=load(TEST)['queries']
 assert 200<=len(rows)<=400
 assert not {r['run'] for r in rows if r['split']=='train'} & {r['run'] for r in rows if r['split']=='val'}
 assert not {r['pixel_sha256'] for r in rows}&{r['pixel_sha256'] for r in tests}
 assert all(r['date']<'20261012' for r in rows)
 assert all(r['date']=='20260923' for r in rows if r['split']=='val')
 assert len({r['pixel_sha256'] for r in rows})==len(rows)

def test_frozen_program_prompts():
 m=load(OUT/'manifest.json');program=load(PROGRAM)
 assert sha(PROGRAM)==m['program_sha256']==sha(OUT/'visual_program.json')
 assert sha(TEST)==m['test_manifest_sha256']
 assert set(m['atomic_prompts'])==set(PIDS)
 for p in program['predicates']:assert m['atomic_prompts'][p['id']]==PREFIX+p['question']
 for p,s in load(TEST)['prompts']['A_atomic_original'].items():assert m['atomic_prompts'][p]==s

def test_budget_and_no_full_finetune():
 c=load(OUT/'training_config.json')
 assert c['method']=='QLoRA' and 1<=c['epochs']<=3
 assert c['max_epochs_after_test']==0

def test_atomic_answer_mask_only():
 from cloth_agent.experiments.qwen_cloth_distillation_train import encode,CHECKPOINT
 from transformers import AutoProcessor
 p=AutoProcessor.from_pretrained(str(CHECKPOINT),local_files_only=True)
 m=load(OUT/'manifest.json');r=m['images'][0]
 for value in ['YES','NO','UNKNOWN']:
  encoded=encode(p,dict(r,value=value,prompt=m['atomic_prompts']['right_extended']))
  supervised=encoded['labels'][encoded['labels']!=-100]
  assert p.tokenizer.decode(supervised)==value
  assert encoded['input_ids'].shape==encoded['labels'].shape
  assert supervised.numel()<encoded['labels'].numel()/10

def test_teacher_transient_completion(tmp_path):
 from cloth_agent.experiments.qwen_cloth_teacher import teacher_completion_ok
 f=tmp_path/'answer.json';f.write_text('{"value":"YES"}')
 schema={'type':'object','properties':{'value':{'enum':['YES','NO','UNKNOWN']}},'required':['value'],'additionalProperties':False}
 transient={'final_turn_completed':True,'errors':[{'message':'Reconnecting... 2/5 (429 Too Many Requests)'}]}
 assert teacher_completion_ok(transient,f,schema)
 assert not teacher_completion_ok(dict(transient,final_turn_completed=False),f,schema)
 assert not teacher_completion_ok(dict(transient,errors=[{'message':'invalid credentials'}]),f,schema)
 f.write_text('{"reasoning":"not a label"}')
 assert not teacher_completion_ok(transient,f,schema)
