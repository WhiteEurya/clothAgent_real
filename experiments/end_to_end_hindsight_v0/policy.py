"""Frozen runtime template: learned procedure only; current RGB, no history."""
from pathlib import Path
from transport import call,read,save
from contracts import ACTION_SCHEMA,ACTION_PROMPT

def generate(rgb,target_corner,artifact_dir,request_dir):
 artifact_dir=Path(artifact_dir);request_dir=Path(request_dir);skill=read(artifact_dir/'skill_spec.json')
 payload={'target_corner':target_corner,'candidate_strategy':skill,'image_order':['CURRENT RGB only'],'image_size':__import__('PIL.Image',fromlist=['Image']).open(rgb).size}
 result=call(request_dir,'generate_action',ACTION_PROMPT,payload,[Path(rgb)],ACTION_SCHEMA)
 if result['status']!='COMPLETE':return {'decision':'ABSTAIN','abstain_reason':'INSUFFICIENT_OBSERVATION','error':result.get('error')}
 action=result['output'];save(request_dir/'generated_action.json',action);return action
