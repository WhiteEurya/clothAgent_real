"""Pilot: compile one observation module, observe/execute/reobserve, then reflect."""
from __future__ import annotations
import argparse
import time
from pathlib import Path

from .common import canonical, digest, read_json, write_json
from .candidate_patch import TEST_SCHEMA
from .executors.observation import RegisteredObservationHost
from .information_flow import STATE, REQUEST, request_schema, validate_state
from .model import RuntimeClaude
from .policy import PolicyError, obj, validate_schema
from .reasoning_contract import freeze_evidence, load_evidence, verify_evidence
from .reasoning_learning import DebugLog, call_metrics
from .skills.registry import ObservationSkill, SkillRegistry, SPEC_SCHEMA, RECIPE_SCHEMA

TEXT={'type':'string'}
MODULE_SCHEMA=obj({'specification':SPEC_SCHEMA,'source':{'type':'string','maxLength':16000},
                   'tests':{'type':'array','minItems':2,'maxItems':4,'items':TEST_SCHEMA}})
COMPILE='''Compile ONE reusable pure observation module from the draft experience and its review notes.
Do not optimize latency or generate grasp/motion code. The code must handle deterministic image
preparation; Claude must decide applicability, current ROI/orientation and information sufficiency.
Choose a coherent useful information-acquisition method and implement prepare(request,source,available).
Only one function, assignments, if/else, arithmetic, dict/list literals, indexing and return.
Allowed functions: min,max,abs,round,int,float,len,floor,ceil. No imports, attributes, loops, IO or robot APIs.
request uses supplied REQUEST schema; source has image_id,role,size; available has cached_views,remaining_ops.
Return RECIPE_SCHEMA. Roles [clean] mean operate on the selected view; [clean,overlay] requires an aligned
clean/overlay source. No semantic findings in code. Coordinates must come from request, not remembered
scene values. Use normalized ROI in the selected SOURCE frame, including intermediate sources. Ensure
null ROI and nonzero rotation have explicit handling, with at most four operations. Declare limitations.
Tests must contain complete requests matching the specification.id, source_size, expected_recipe;
use different dynamic ROIs/sizes and calculate exact expected recipes. Host executes image IO and retains
lineage. The draft is fallible: follow review notes, do not make integer zoom mandatory or infer a hinge
from an unidentified line. Keep success_check about obtaining information, not code execution.
Return specification, source and tests only. No patch activation or claims of validation.
'''
API_DETAILS={
    'source_example':{'image_id':'image_0','role':'clean','size':[720,1280]},
    'source_role_values':['clean','overlay','reference','hint','clean_crop','overlay_crop'],
    'available_example':{'cached_views':0,'remaining_ops':12},
    'types':'cached_views is an INTEGER count, not a list. size is [width,height] in pixels. ROI coordinates are normalized source-frame values.',
    'pair_rule':'For source role clean or overlay, roles [clean,overlay] requests the aligned pair. No role named pair, aligned_pair or clean_overlay exists. Other roles use [clean] to process that selected view.',
    'unknown_rule':'Do not treat an unidentified line as a hinge or a seam. Report UNKNOWN if the requested identity is unresolved.',
}
OBSERVE='''Inspect the attached current views before deciding any next operation. This is an offline
information-acquisition trial, not a grasp or motion planner. Resolve a useful visible information need
for the supplied goal using the module when applicable; do not force unnecessary edits just to exercise it.
First assess the last code result, if any, against its success_check using the actual delivered views.
SUFFICIENT means the requested information is readable (a clear negative answer is valid); execution
alone is not success. If seam identity remains unclear, mark UNKNOWN, not an assumed hinge.
Preserve prior information IDs and needs; update findings and cite image IDs. Then choose EXECUTE with
one dynamic request, DONE if the scoped information task is resolved, or UNKNOWN with the exact gap.
Binding uses the selected source image's normalized ROI, never a remembered root ROI on a crop. All
images in catalog are available in attachment order. Use lineage for root-coordinate reporting.
Code stays fixed during this trial. Do not repeat the same edit unless it addresses a new concrete gap.
At remaining_executions=0, assess the last result and stop: return DONE or UNKNOWN with request=null.
A requested view is not yet a finding. No robot decisions. Return only the schema.
'''


def observe_schema(registry):
    assessment=obj({'outcome':{'enum':['SUFFICIENT','INSUFFICIENT','UNKNOWN']},'reason':TEXT,
                    'source_image_ids':{'type':'array','items':TEXT,'uniqueItems':True}})
    return obj({'decision':{'enum':['EXECUTE','DONE','UNKNOWN']},'reason':TEXT,'information':STATE,
                'last_result':{'anyOf':[{'type':'null'},assessment]},
                'request':{'anyOf':[{'type':'null'},request_schema(registry)]}})


def compile_module(value):
    validate_schema(value,MODULE_SCHEMA)
    skill=ObservationSkill(value['specification'],value['source'])
    registry=SkillRegistry();registry.register(skill)
    for case in value['tests']:
        validate_schema(case['request'],request_schema(registry))
        recipe=skill.prepare(case['request'],{'image_id':case['request']['source_image_id'],
                            'role':'clean','size':case['source_size']},
                            {'cached_views':0,'remaining_ops':12})
        if recipe!=case['expected_recipe']:
            raise PolicyError('Generated module test does not match its expected recipe')
    if len({canonical([t['request'],t['source_size']]) for t in value['tests']})<2:
        raise PolicyError('Module tests must exercise distinct bindings')
    return registry,skill


def run(evidence,experience,notes,output,vision,text,*,base=None,max_executions=2,max_ops=12,module_from=None,initial_observation_from=None,observations_from=None):
    if not 1<=max_executions<=5 or not 1<=max_ops<=24:
        raise ValueError('Trial allows up to five code executions and twenty-four edits')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();debug=DebugLog(output)
    report={'status':'RUNNING','robot_actions':0,'automatic_activation':False,'trace':[],
            'scope':'Observation information only; no grasp planning, physical test, or speed comparison'}
    host=None
    def invoke(model,name,prompt,spec,images):
        write_json(output/(name+'_input.json'),{'prompt':prompt,'schema':spec,'images_sent':len(images)})
        debug.event('call_start',stage=name)
        value=model.invoke(prompt=prompt,schema=spec,images=images,output=output/name,stage=name)
        write_json(output/(name+'_returned.json'),value)
        validate_schema(value,spec)
        debug.event('call_validated',stage=name)
        return value
    try:
        frozen=freeze_evidence(evidence,output/'evidence',base=base)
        images=verify_evidence(frozen,output/'evidence');obs=frozen['evidence']
        if module_from:
            generated=read_json(module_from)
            write_json(output/'compile_reused.json',{'source':str(Path(module_from).resolve()),'module':generated})
        else:
            generated=invoke(text,'compile',COMPILE+canonical({'draft_lessons':experience['lessons'],
                'review_notes':notes,'request_schema':REQUEST,'recipe_schema':RECIPE_SCHEMA,
                'api_details':API_DETAILS}),MODULE_SCHEMA,[])
        try:
            registry,skill=compile_module(generated)
        except PolicyError as exc:
            generated=invoke(text,'compile_repair',COMPILE+'\nRepair this existing module once, preserving its information task. '
                'Fix interface type/role mistakes and recompute test expectations, not the experiment goal. '+canonical({
                    'module':generated,'error':str(exc),'review_notes':notes,'api_details':API_DETAILS,
                    'request_schema':REQUEST,'recipe_schema':RECIPE_SCHEMA}),MODULE_SCHEMA,[])
            registry,skill=compile_module(generated)
        write_json(output/'module.json',generated)
        (output/'module.py').write_text(skill.source)
        write_json(output/'module_tests.json',{'status':'PASSED','count':len(generated['tests']),
                                             'scope':'Pure recipe tests; real image outputs validated by Host'})
        report['module_hash']=skill.content_hash
        host=RegisteredObservationHost(obs,images,{'skills':registry.catalog()},output/'observations',max_ops,
                                       registry=registry,reuse_observations=False)
        information=[];used=0;last=None
        for index in range(max_executions+1):
            shared={'goal':obs['fold_goal'],'module':skill.specification,'catalog':[
                {k:v for k,v in row.items() if k not in ('path','source_view_id')} for row in host.catalog],
                'information':information,'last_execution':last,'remaining_executions':max_executions-used,
                'remaining_host_ops':max_ops-host.ops}
            prompt=OBSERVE+canonical(shared)
            cached_from=observations_from or (initial_observation_from if index==0 else None)
            name=f'observe_{index:02d}'
            if cached_from and (Path(cached_from)/(name+'_returned.json')).exists():
                previous=Path(cached_from)
                saved=read_json(previous/(name+'_input.json'))
                if saved['prompt']!=prompt or saved['images_sent']!=len(host.paths):
                    raise PolicyError('Cached initial observation context differs from this trial')
                # Catalog in the exact prompt binds image pixel hashes, order, geometry,
                # module specification, goal and execution limits. No robot state is reused.
                if read_json(previous/'report.json').get('module_hash')!=skill.content_hash:
                    raise PolicyError('Cached initial observation module differs from this trial')
                judgment=read_json(previous/(name+'_returned.json'))
                validate_schema(judgment,observe_schema(registry))
                write_json(output/(name+'_returned.json'),judgment)
                write_json(output/(name+'_input.json'),saved)
                report.setdefault('observations_reused',[]).append({'stage':name,'source':str(previous.resolve())})
                debug.event('observation_reused',stage=name,source=str(previous))
            else:
                judgment=invoke(vision,f'observe_{index:02d}',prompt,observe_schema(registry),list(host.paths))
            warnings=validate_state(judgment['information'],host.catalog,information)
            if last:
                assessment=judgment['last_result']
                if assessment is None:
                    raise PolicyError('Post-operation observation must assess its result')
                refs=set(assessment['source_image_ids'])
                if not refs <= {i['image_id'] for i in host.catalog}:
                    raise PolicyError('Result assessment cites unavailable images')
                if assessment['outcome']=='SUFFICIENT' and not refs.intersection(last['delivered_image_ids']):
                    raise PolicyError('Information success must cite a delivered result view')
            elif judgment['last_result'] is not None:
                if judgment['last_result']['outcome']=='UNKNOWN' and not judgment['last_result']['source_image_ids']:
                    warnings.append('INITIAL_EMPTY_UNKNOWN_NORMALIZED_TO_NULL')
                    judgment['last_result']=None
                else:
                    raise PolicyError('Cannot assess an operation before one has executed')
            information=judgment['information']
            ignored=None
            if judgment['decision']!='EXECUTE' and judgment['request'] is not None:
                ignored=judgment['request']
                judgment['request']=None
                warnings.append('STOP_REQUEST_RETAINED_AS_UNEXECUTED_SUGGESTION')
            report['trace'].append({'stage':'observe','judgment':judgment,'warnings':warnings,
                                    'unexecuted_suggestion':ignored})
            write_json(output/'report.json',report)
            if judgment['decision']!='EXECUTE':
                if judgment['decision']=='DONE' and last and judgment['last_result']['outcome']!='SUFFICIENT':
                    raise PolicyError('DONE contradicts unresolved operation result')
                report['outcome']=judgment['decision'] if used else 'NOT_EXERCISED'
                break
            if used>=max_executions or judgment['request'] is None:
                raise PolicyError('Execution decision exceeds budget or lacks a binding')
            if skill.content_hash!=report['module_hash']:
                raise PolicyError('Module changed during trial')
            failure=host.execute([judgment['request']],information)
            if failure:
                report['outcome']=failure
                break
            used+=1;last=host.history[-1]
            report['trace'].append({'stage':'code','execution':last,'module_hash':skill.content_hash})
            write_json(output/'report.json',report)
        feedback_schema=obj({'summary':TEXT,'information_obtained':TEXT,'remaining_unknowns':TEXT,
            'code_assessment':TEXT,'binding_assessment':TEXT,'next_revision':TEXT,
            'evidence':{'type':'array','items':obj({'trace_index':{'type':'integer','minimum':0,
                              'maximum':len(report['trace'])-1},'lesson':TEXT})}})
        feedback=invoke(text,'feedback',
            'Review this completed observation/code trial in Chinese. Use actual before/after judgments and operations. '
            'Separate code execution, information acquisition and binding quality. UNKNOWN is not failure to see an absent '
            'feature. No latency ranking, no physical correctness claim. Suggest a future revision only; do not change the '
            'module. Cite trace_index (zero-based). '+canonical({'module':generated,'trial':report}),feedback_schema,[])
        write_json(output/'feedback.json',feedback)
        report.update(status='COMPLETED',feedback=feedback,code_executions=used)
    except Exception as exc:
        report.update(status='ERROR',error=f'{type(exc).__name__}: {exc}')
    finally:
        report['elapsed_s']=time.monotonic()-started
        report['host_image_ops']=host.ops if host else 0
        report['metrics']=call_metrics(text.calls+vision.calls,output)
        report['metrics']['host_image_ops']=report['host_image_ops']
        report['metrics']['observation_responses_reused']=len(report.get('observations_reused',[]))
        report['metrics']['reuse_note']='Cached model-call time/tokens are excluded; see observations_reused source runs.'
        write_json(output/'report.json',report)
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence',type=Path,required=True);p.add_argument('--experience',type=Path,required=True)
    p.add_argument('--review-notes',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--ssh-host',default='company-planner');p.add_argument('--backend',choices=['local','remote'],default='remote')
    p.add_argument('--call-timeout',type=float,default=600);p.add_argument('--model')
    p.add_argument('--max-executions',type=int,default=2);p.add_argument('--max-host-ops',type=int,default=12)
    p.add_argument('--module-from',type=Path,help='Reuse an existing generated module, validate and repair once if needed')
    p.add_argument('--initial-observation-from',type=Path,help='Reuse first observation only when its exact context and module hash match')
    p.add_argument('--observations-from',type=Path,help='Reuse available observation responses with exact context/hash checks; rerun deterministic Host operations')
    a=p.parse_args(argv)
    opts={'backend':a.backend,'ssh_host':a.ssh_host,'timeout_s':a.call_timeout,'model':a.model}
    result=run(load_evidence(a.evidence),read_json(a.experience),a.review_notes.read_text(),a.output,
        RuntimeClaude(**opts),RuntimeClaude(**opts,text_only=True),base=a.evidence.parent,
        max_executions=a.max_executions,max_ops=a.max_host_ops,module_from=a.module_from,
        initial_observation_from=a.initial_observation_from,observations_from=a.observations_from)
    print(canonical({k:v for k,v in result.items() if k not in ('trace','feedback')}),flush=True)
    return 0 if result['status']=='COMPLETED' else 2


if __name__=='__main__':
    raise SystemExit(main())
