"""Single-call policy + local geometry + observed-outcome feedback adapter."""
import json,time
from pathlib import Path
from dataclasses import asdict
from PIL import Image
from cloth_agent import geometry_report, hem_sleeve_pull_rule
from cloth_agent.experiments import one_shot_goal_planner_probe_v1 as io
from cloth_agent.restricted_context_one_shot import SCHEMA,validate
from .gpt_policy import PROMPT
from .dataset import load_observation
from .interfaces import Action
from ..cv_harness.geometry_check import check
from ..cv_harness.outcome_observer import observe
from ..cv_harness.feedback_generator import compress

REPORTS_PREVIOUS_EFFECT=False

def plan_round(root,folder,before,headers,timeout_s,direct,previous_context=None,**kwargs):
    body=io.payload(before.image,root/'goal/target.png',list(Image.open(before.image).size));body['instructions']=hem_sleeve_pull_rule.apply(root,PROMPT)
    body['text']['format'].update(name='cv_feedback_single_action',schema=SCHEMA)
    feedback=None
    if previous_context:
        previous=root/previous_context['source_round']/'cv_feedback.txt'
        if previous.exists():
            feedback=previous.read_text();body['input'][0]['content'].append({'type':'input_text','text':'上一实际动作的自动观测反馈：'+feedback})
    io.save(folder/'request_manifest.json',{'mode':'CV_GEOMETRY_AND_OBSERVED_OUTCOME','model_calls':1,'previous_feedback':feedback,'input_image_count':2,
        'semantic_grounding_model':False,'human_region_labels_used':False,'automatic_replans':False})
    output,metrics=direct(folder,body,headers,timeout_s)
    # Shared episode runner reads this evidence when applying the action limit.
    # Persist the raw decision, as the other one-shot adapters do.
    io.save(folder/'decision_evidence.json',output)
    plan=validate(output,Image.open(before.image).size,False)
    io.save(folder/'model_metrics.json',metrics);io.save(folder/'plan.json',plan);return plan,metrics

def verify(observation,plan,folder):
    from cloth_agent.dual_arm.geometry import DualArmError
    result=check(observation,plan);result['observation_id']=observation.meta.get('observation_id');io.save(folder/'cv_geometry_check.json',result)
    if result['status']!='ACCEPTED':raise DualArmError('CV geometry '+result['status']+': '+result['reason'])
    return result

def record_outcome(before,after_dir,plan,execution,folder):
    started=time.perf_counter()
    try:
        action=Action(tuple(plan['grasp_px']),tuple(plan['target_px']),plan.get('grasp_angle_deg'),plan.get('action_purpose'))
        outcome,_,_=observe(load_observation(before.directory),action,load_observation(after_dir),execution)
        io.save(folder/'cv_outcome.json',asdict(outcome));(folder/'cv_feedback.txt').write_text(compress(outcome,action))
        io.save(folder/'cv_outcome_sources.json',{'before':str(before.directory),'after':str(after_dir),'execution_status':execution.get('status')})
    except Exception as exc:
        io.save(folder/'cv_outcome.json',{'status':'UNKNOWN','error':f'{type(exc).__name__}: {exc}'})
        (folder/'cv_feedback.txt').write_text('Previous action outcome analysis unavailable. Grasp and unfolding progress UNKNOWN; do not infer success.')
    return time.perf_counter()-started


def render_round(folder):
    import html
    parts=['<!doctype html><meta charset="utf-8"><title>CV Geometry + Observed Outcome</title><style>body{font:17px system-ui;margin:25px}img{max-width:95%}pre{white-space:pre-wrap}</style><h1>'+html.escape(folder.name)+' · 几何检查与真实结果</h1>']
    for name in ('plan.json','cv_geometry_check.json','host_validation.json','execution.json','cv_outcome.json','timing.json'):
        if (folder/name).exists():parts.append('<h2>'+name+'</h2><pre>'+html.escape((folder/name).read_text())+'</pre>')
    if (folder/'cv_feedback.txt').exists():parts.append('<h2>下一轮反馈</h2><pre>'+html.escape((folder/'cv_feedback.txt').read_text())+'</pre>')
    for name in ('before/rgb.png','proposed_action_overlay.png','after/rgb.png'):
        if (folder/name).exists():parts.append('<h2>'+name+'</h2><img src="'+name+'">')
    (folder/'decision.html').write_text('\n'.join(parts))
